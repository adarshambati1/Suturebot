"""Webcam hand-tracking teleop for the REAL Rizon4s (Flexiv "Titania").

Same hand-tracking front-end as webcam_hand_publisher.py, but adapted for
the real robot driven by OpenSai with suturebot_grav_real.xml:

  * Robot name is "Titania" (the Flexiv driver), so all Redis keys differ
    from the sim "Rizon4s" keys.
  * The gripper is the hemostat, commanded as a binary MODE string
    ("o" = open, "g" = grip) via opensai::commands::Titania::gripper::mode
    -- not a continuous width like the sim Grav gripper.
  * On every fist (engage) the anchor is the robot's ACTUAL current pose and
    orientation, read back from Redis, so the arm never jumps or rotates. No
    fixed home is assumed (the real robot reports in a different frame than
    the sim). MAX_REACH caps how far one clench can drive the arm.

Topology: this script runs on your MacBook (it owns the camera). The robot,
OpenSai, and the controller run on the Linux PC. Redis is the bridge, so the
Mac must reach the PC's Redis. Two options:

  A) SSH tunnel (recommended, nothing exposed to the LAN). On the Mac:
         ssh -N -L 6379:localhost:6379 <user>@<linux-pc>
     then run this script with the default --host 127.0.0.1.

  B) LAN: on the PC, start redis bound to its LAN IP (redis-server
     --bind 0.0.0.0 --protected-mode no), then run this with
     --host <linux-pc-ip>. Only do this on a trusted network.

Mapping (relative to the pose where each fist forms):
    hand x in image    -> robot Y  (lateral; mirrored so it feels natural)
    hand y in image    -> robot Z  (up in image = up in robot)
    hand bbox diagonal -> robot X  (bigger hand = closer to camera = +X)
    pinch (thumb-index) -> gripper mode: wide = open "o", pinched = grip "g"

Control (two layers):
    'e'          -> engaged / disengaged. Disengaged = zero Redis writes.
    fist         -> active control while engaged. Robot follows your hand
                    only while you hold a fist; open hand freezes it.
    'r'          -> recenter while fisted (re-anchor here).
    'q'          -> quit. Auto-disengages if no hand for > 1.0 s.

Safety for first bring-up:
    * Start with --no-gripper so the pre-loaded needle stays gripped and the
      hand only drives position. Add gripper control once position is trusted.
    * Keep --scale small (default 2 cm half-range).
    * Motion is RELATIVE to wherever the arm is when you clench, and its
      current orientation is held -- engaging never jumps or rotates the arm.
      MAX_REACH caps how far one clench can drive it.
    * Image axes (hand x/y, depth) may not line up with the real robot's
      x/y/z directions -- expect to flip a sign or two once it moves, like the
      left/right flip we did in the sim version.

Install (one-time, on the Mac):
    pip install mediapipe opencv-python redis numpy

Usage:
    # via SSH tunnel, position only:
    python vr_teleop/test_harness/webcam_hand_publisher_real.py --no-gripper
    # direct to PC, with gripper, slightly larger motion:
    python vr_teleop/test_harness/webcam_hand_publisher_real.py \
        --host 192.168.1.50 --scale 0.03
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import redis
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# HandLandmarker model — downloaded on first run.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.expanduser("~/.cache/suturebot/hand_landmarker.task")

# --- Real-robot Redis wiring -------------------------------------------------
# Mirrors python_examples/suturebot_grav_pierce_oussama_push_grip.py with
# ROBOT_NAME = "Titania". Keep in sync if those keys move.
ROBOT_NAME = "Titania"
EXPECTED_CONFIG = "suturebot_grav_real.xml"

K_GOAL_POS = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position"
K_GOAL_ORI = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_orientation"
K_CUR_POS  = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_position"
K_CUR_ORI  = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_orientation"
K_GRIP_MODE = f"opensai::commands::{ROBOT_NAME}::gripper::mode"
K_ACTIVE   = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
K_CONFIG   = "::sai-interfaces-webui::config_file_name"

CONTROLLER_TO_USE = "cartesian_controller"
GRIP_OPEN_CMD = "o"
GRIP_CLOSE_CMD = "g"

# --- Safety: motion is relative to the arm's ACTUAL pose at each engage ------
# We do NOT assume a fixed home (the real robot reports in a different frame
# than the sim). On every fist the script reads the robot's current pose and
# moves relative to it, holding its current orientation. MAX_REACH caps how far
# a single clench can drive the arm from that engage pose (per axis, meters).
# Keep small for first bring-up; widen deliberately.
MAX_REACH = np.array([0.40, 0.40, 0.40])   # original: np.array([0.15, 0.15, 0.15])

# --- Gesture / mapping tuning ------------------------------------------------
# Pinch (thumb-index distance normalized by hand size) thresholds, with a
# dead band between them for hysteresis so the binary gripper doesn't chatter.
PINCH_GRIP_BELOW = 0.45   # below this -> command grip "g"
PINCH_OPEN_ABOVE = 0.75   # above this -> command open "o"

# Hand bbox diagonal (normalized image coords) range mapped to ±X scale.
BBOX_NEAR = 0.55   # hand close to camera
BBOX_FAR  = 0.20   # arm extended

# 1-euro filter defaults (adaptive low-pass; far better than EMA for hand jitter).
EURO_MIN_CUTOFF = 1.0   # Hz; LOWER = smoother when the hand is slow. Drop to ~0.5 if jittery.
EURO_BETA = 0.1         # speed coefficient; higher = less lag when moving fast.

# Per-axis cutoff scale applied to --mincutoff. Order matches the offset vector
# from map_hand_to_offset: [X depth, Y lateral (left/right), Z vertical (up/down)].
# The lateral axis is the most jitter-amplified on this arm, so smooth it hardest
# (smaller scale -> lower cutoff -> heavier smoothing on that axis alone).
AXIS_CUTOFF_SCALE = np.array([0.6, 0.3, 1.0])

DEADBAND_M = 0.0015     # don't rewrite the goal unless it moved this far (m)
NO_HAND_TIMEOUT = 1.0   # auto-disengage if no hand this long (s)

# Edges between landmark indices for drawing the hand skeleton.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Tip / PIP indices for the four non-thumb fingers (fist detection).
FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]
FIST_MIN_CURLED = 3
FIST_DEBOUNCE = 3


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def draw_hand(frame: np.ndarray, lm_list, w: int, h: int) -> None:
    pts = [(int(p.x * w), int(p.y * h)) for p in lm_list]
    for a, b in HAND_EDGES:
        cv2.line(frame, pts[a], pts[b], (180, 180, 180), 1)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 255, 255), -1)


def is_fist(lm) -> bool:
    wrist = lm[0]
    curled = 0
    for tip_i, pip_i in FINGER_TIP_PIP:
        d_tip = (lm[tip_i].x - wrist.x) ** 2 + (lm[tip_i].y - wrist.y) ** 2
        d_pip = (lm[pip_i].x - wrist.x) ** 2 + (lm[pip_i].y - wrist.y) ** 2
        if d_tip < d_pip:
            curled += 1
    return curled >= FIST_MIN_CURLED


class OneEuroFilter:
    """Vectorized 1-euro filter for an N-dim signal (Casiez et al. 2012).

    Adaptive low-pass: smooths hard at low speed to kill jitter while the hand
    holds still, and lightens up at high speed so fast moves don't lag. Each
    axis adapts independently, so the noisy depth axis gets smoothed harder
    than the cleaner lateral axes on its own.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def map_hand_to_offset(
    wrist_x: float, wrist_y: float, bbox_diag: float,
    ref_wrist_x: float, ref_wrist_y: float, ref_bbox: float,
    scale: float,
) -> np.ndarray:
    """Hand state -> (dx, dy, dz) robot offset relative to a reference.

    Half the frame (0.5 normalized) maps to `scale` meters; same for a bbox
    change of half the BBOX_NEAR-BBOX_FAR range. Per-axis clamp to ±scale.
    """
    img_x_norm = np.clip((wrist_x - ref_wrist_x) * 2.0, -1.0, 1.0)
    img_y_norm = np.clip((wrist_y - ref_wrist_y) * 2.0, -1.0, 1.0)
    half_depth_range = max((BBOX_NEAR - BBOX_FAR) / 2.0, 1e-6)
    depth_norm = np.clip((bbox_diag - ref_bbox) / half_depth_range, -1.0, 1.0)

    dy_robot = -img_x_norm * scale   # flip so hand-left = robot-left in view
    dz_robot = -img_y_norm * scale   # image y grows downward
    dx_robot = depth_norm * scale
    return np.array([dx_robot, dy_robot, dz_robot])


def read_current_pos(r: redis.Redis) -> np.ndarray | None:
    raw = r.get(K_CUR_POS)
    if raw is None:
        return None
    try:
        return np.array(json.loads(raw.decode("utf-8")), dtype=float)
    except (ValueError, TypeError):
        return None


def read_current_ori(r: redis.Redis) -> np.ndarray | None:
    raw = r.get(K_CUR_ORI)
    if raw is None:
        return None
    try:
        return np.array(json.loads(raw.decode("utf-8")), dtype=float)
    except (ValueError, TypeError):
        return None


def draw_hud(frame, engaged, fisted, pos, grip_mode, fps, hand_seen, clamped):
    h, w = frame.shape[:2]
    if engaged and fisted:
        color, label = (0, 220, 0), "MOVING (fist)"
    elif engaged:
        color, label = (0, 200, 220), "ENGAGED (open hand = freeze)"
    else:
        color, label = (0, 120, 255), "DISENGAGED (press e)"
    cv2.rectangle(frame, (0, 0), (w, 56), (30, 30, 30), -1)
    cv2.putText(frame, "REAL ROBOT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.putText(frame, label, (170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    info = f"pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]"
    if clamped:
        info += " CLAMPED"
    cv2.putText(frame, info, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230) if not clamped else (0, 200, 255), 1)
    if grip_mode is not None:
        gtxt = "grip" if grip_mode == GRIP_CLOSE_CMD else "open"
        cv2.putText(frame, f"gripper={gtxt}", (560, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    bot = (f"e: engage   fist: move   r: recenter   q: quit   "
           f"fps={fps:4.1f}   hand={'yes' if hand_seen else 'no '}")
    cv2.putText(frame, bot, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Redis host (the Linux PC, or 127.0.0.1 via SSH tunnel)")
    p.add_argument("--port", type=int, default=6379, help="Redis port")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--scale", type=float, default=0.10,   # original default: 0.02
                   help="Half-range of robot motion in meters (default: 0.10)")
    p.add_argument("--mincutoff", type=float, default=EURO_MIN_CUTOFF,
                   help=f"1-euro min cutoff (Hz); LOWER = smoother when slow. "
                        f"Drop toward 0.5 if still jittery (default {EURO_MIN_CUTOFF}).")
    p.add_argument("--beta", type=float, default=EURO_BETA,
                   help=f"1-euro speed coefficient; higher = less lag on fast moves "
                        f"(default {EURO_BETA})")
    p.add_argument("--deadband", type=float, default=DEADBAND_M,
                   help=f"Skip goal rewrites that move less than this many meters, "
                        f"kills idle jitter (default {DEADBAND_M})")
    p.add_argument("--no-gripper", action="store_true",
                   help="Don't command the gripper (recommended for first runs)")
    p.add_argument("--no-activate", action="store_true",
                   help="Skip setting active_controller (assume already set)")
    p.add_argument("--no-check", action="store_true",
                   help="Skip the config_file_name == suturebot_grav_real.xml check")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()

    # Confirm OpenSai is running the real-robot scene, not the sim.
    if not args.no_check:
        cfg = r.get(K_CONFIG)
        cfg_name = cfg.decode("utf-8") if cfg is not None else None
        if cfg_name != EXPECTED_CONFIG:
            print(f"Refusing to start: OpenSai config is {cfg_name!r}, "
                  f"expected {EXPECTED_CONFIG!r}.")
            print("Launch OpenSai with suturebot_grav_real.xml, or pass --no-check.")
            return

    if not args.no_activate:
        while r.get(K_ACTIVE) is None or r.get(K_ACTIVE).decode() != CONTROLLER_TO_USE:
            r.set(K_ACTIVE, CONTROLLER_TO_USE)
            time.sleep(0.05)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    WINDOW = "Suturebot webcam teleop (REAL)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    landmarker_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(landmarker_options)

    print(f"REAL ROBOT teleop. Redis {args.host}:{args.port}, robot={ROBOT_NAME}")
    print(f"  scale=±{args.scale*100:.1f}cm   max reach from engage=±{MAX_REACH.tolist()} m")
    print("  motion is RELATIVE to the arm's pose when you clench (orientation held)")
    print(f"  gripper {'DISABLED' if args.no_gripper else 'enabled (pinch -> o/g)'}")
    print("Keys: 'e' engage   fist = move   open hand = freeze   'r' recenter   'q' quit")

    engaged = False
    fisted = False
    fist_streak = 0
    offset_filt = np.zeros(3)
    last_hand_time = 0.0
    seed = read_current_pos(r)
    last_pos = seed.copy() if seed is not None else np.zeros(3)
    last_written: np.ndarray | None = None   # last goal actually sent (deadband)
    grip_mode: str | None = None   # last commanded gripper mode (edge-triggered)

    anchor_pos = last_pos.copy()
    ref_wrist_x = 0.5
    ref_wrist_y = 0.5
    ref_bbox = (BBOX_NEAR + BBOX_FAR) / 2.0
    euro = OneEuroFilter(args.mincutoff * AXIS_CUTOFF_SCALE, args.beta)

    frame_count = 0
    t_fps = time.monotonic()
    fps = 0.0

    def reanchor(wrist, bbox_diag) -> bool:
        """Anchor at the robot's ACTUAL current pose and hold its current
        orientation, so engaging never jumps or rotates the arm."""
        nonlocal anchor_pos, ref_wrist_x, ref_wrist_y, ref_bbox, last_pos, last_written
        actual = read_current_pos(r)
        if actual is None:
            print("No current_position from OpenSai; is the controller running? "
                  "Not engaging.")
            return False
        anchor_pos = actual.copy()
        last_pos = actual.copy()
        last_written = None   # force the first goal write after (re)anchoring
        ref_wrist_x, ref_wrist_y, ref_bbox = wrist.x, wrist.y, bbox_diag
        offset_filt[:] = 0.0
        euro.reset()
        cur_ori = read_current_ori(r)
        if cur_ori is not None:
            r.set(K_GOAL_ORI, json.dumps(cur_ori.tolist()))
        return True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; retrying.")
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.monotonic() * 1000)
        res = landmarker.detect_for_video(mp_image, ts_ms)
        hand_seen = bool(res.hand_landmarks)

        desired_grip = grip_mode
        if hand_seen:
            last_hand_time = time.monotonic()
            lm = res.hand_landmarks[0]
            wrist, thumb, index, middle_mcp = lm[0], lm[4], lm[8], lm[9]

            xs = np.array([p.x for p in lm])
            ys = np.array([p.y for p in lm])
            bbox_diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))

            if is_fist(lm):
                fist_streak += 1
            else:
                fist_streak = 0
            fist_rising = (not fisted) and fist_streak >= FIST_DEBOUNCE
            fist_falling = fisted and fist_streak == 0

            if engaged and fist_rising:
                if reanchor(wrist, bbox_diag):
                    fisted = True
            elif fist_falling:
                fisted = False

            offset = map_hand_to_offset(
                wrist.x, wrist.y, bbox_diag,
                ref_wrist_x, ref_wrist_y, ref_bbox, args.scale,
            )
            offset_filt = euro(offset, time.monotonic())

            # Pinch -> binary gripper mode with hysteresis.
            hand_size = float(np.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y))
            pinch_norm = float(np.hypot(thumb.x - index.x, thumb.y - index.y)) / max(hand_size, 1e-6)
            if pinch_norm < PINCH_GRIP_BELOW:
                desired_grip = GRIP_CLOSE_CMD
            elif pinch_norm > PINCH_OPEN_ABOVE:
                desired_grip = GRIP_OPEN_CMD

            fh, fw = frame.shape[:2]
            draw_hand(frame, lm, fw, fh)
        else:
            fist_streak = 0
            fisted = False

        if engaged and (time.monotonic() - last_hand_time) > NO_HAND_TIMEOUT:
            engaged = False
            fisted = False
            print("Auto-disengaged: no hand detected.")

        target_raw = anchor_pos + offset_filt
        target_pos = np.clip(target_raw, anchor_pos - MAX_REACH, anchor_pos + MAX_REACH)
        clamped = bool(np.any(target_pos != target_raw))

        if engaged and fisted and hand_seen:
            if last_written is None or np.linalg.norm(target_pos - last_written) > args.deadband:
                r.set(K_GOAL_POS, json.dumps(target_pos.tolist()))
                last_written = target_pos.copy()
            last_pos = target_pos
            if not args.no_gripper and desired_grip is not None and desired_grip != grip_mode:
                r.set(K_GRIP_MODE, desired_grip)
                grip_mode = desired_grip

        frame_count += 1
        if frame_count >= 15:
            now = time.monotonic()
            fps = frame_count / (now - t_fps)
            frame_count = 0
            t_fps = now

        draw_hud(frame, engaged, fisted, last_pos, grip_mode, fps, hand_seen, clamped)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            engaged = not engaged
            if engaged:
                print("Engaged. Make a fist to move; open hand to freeze.")
            else:
                fisted = False
                print("Disengaged. No writes until you press 'e' again.")
        if key == ord('r') and engaged and fisted and hand_seen:
            if reanchor(wrist, bbox_diag):
                print(f"Recentered. anchor={anchor_pos.tolist()}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
