"""Webcam hand teleop for the REAL Rizon4s ("Titania") WITH wrist orientation.

Superset of webcam_hand_publisher_real.py: same clutched, relative POSITION
control, plus the robot end-effector now follows your wrist ORIENTATION while
you hold a fist.

How orientation works:
  * MediaPipe gives hand_world_landmarks (metric 3D). We build a hand frame
    from them (across-palm, along-hand, palm-normal) -> a 3x3 rotation in the
    camera frame.
  * On each fist we capture the hand frame AND the robot's current orientation
    as references. Each frame we compute the hand's rotation *relative* to its
    reference, map that into the robot frame, scale by --ori-gain, smooth it,
    and apply it on top of the captured robot orientation. So engaging never
    snaps the wrist, and re-clenching re-zeros orientation too.

Single-camera caveat: in-plane twist (rolling the hand like a steering wheel)
is tracked well; pitching/yawing toward the camera is noisier. Orientation is
smoothed harder than position and capped by MAX_ROT. Lower --ori-gain (e.g.
0.5) for a gentler, steadier feel; pass --no-ori to fall back to position-only.

Everything else (topology, SSH tunnel, gripper, safety) matches
webcam_hand_publisher_real.py -- see that file's header.

Usage:
    python vr_teleop/test_harness/webcam_hand_publisher_real_ori.py --no-gripper --port 6380
    # gentler orientation:
    python vr_teleop/test_harness/webcam_hand_publisher_real_ori.py --port 6380 --ori-gain 0.5
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

# --- Position: relative to the arm's ACTUAL pose at each engage --------------
MAX_REACH = np.array([0.40, 0.40, 0.40])   # original: np.array([0.15, 0.15, 0.15])

# --- Orientation tuning ------------------------------------------------------
ORI_GAIN_DEFAULT = 1.0    # wrist-rotation -> robot-rotation gain. Lower = gentler.
MAX_ROT = math.pi / 2     # cap commanded rotation from the engage orientation (rad)

# Map the relative hand rotation (axis-angle, camera frame) onto robot axes.
# Signed permutation, same idea as the position axis flips: edit signs/rows if
# twisting your wrist turns the robot the wrong way or about the wrong axis.
# Calibrated from observation: wrist roll/pitch/yaw -> robot roll/pitch/yaw.
# cam rotvec axes: [0]=pitch, [1]=yaw, [2]=roll. Robot axes (from testing):
# out[0]=roll, out[1]=pitch, out[2]=yaw. So route roll->out0, pitch->out1,
# yaw->out2 (a 3-way cycle). Original (identity): [[1,0,0],[0,1,0],[0,0,1]].
# Flip a row's sign if that rotation goes the wrong direction.
ORI_AXIS_MAP = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# Orientation 1-euro (smoothed harder than position; it's noisier).
ORI_MIN_CUTOFF = 0.4    # original: 0.8  (lower = steadier orientation at rest)
ORI_BETA = 0.1

# --- Gesture / mapping tuning ------------------------------------------------
PINCH_GRIP_BELOW = 0.45
PINCH_OPEN_ABOVE = 0.75
BBOX_NEAR = 0.55
BBOX_FAR  = 0.20

EURO_MIN_CUTOFF = 0.5   # original: 1.0  (lower = smoother when hand is still)
EURO_BETA = 0.1
AXIS_CUTOFF_SCALE = np.array([0.6, 0.3, 1.0])

DEADBAND_M = 0.003      # original: 0.0015  (freezes sub-3mm idle quiver)
NO_HAND_TIMEOUT = 1.0

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

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


def hand_frame(world_lm) -> np.ndarray | None:
    """Orthonormal hand orientation (3x3, camera frame) from world landmarks.

    Columns: [across-palm, along-hand, palm-normal]. Returns None if the
    landmarks are degenerate.
    """
    def p(i):
        return np.array([world_lm[i].x, world_lm[i].y, world_lm[i].z])

    wrist, idx_mcp, pinky_mcp, mid_mcp = p(0), p(5), p(17), p(9)
    up = mid_mcp - wrist
    side = idx_mcp - pinky_mcp
    nu, ns = np.linalg.norm(up), np.linalg.norm(side)
    if nu < 1e-6 or ns < 1e-6:
        return None
    up /= nu
    side /= ns
    normal = np.cross(side, up)
    nn = np.linalg.norm(normal)
    if nn < 1e-6:
        return None
    normal /= nn
    side = np.cross(up, normal)   # re-orthogonalize
    return np.column_stack([side, up, normal])


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rv))
    if theta < 1e-8:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if abs(theta - math.pi) < 1e-6:
        # Near 180 deg: recover axis from (R + I) / 2 = k k^T.
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        if i == 0:
            k[1] = math.copysign(k[1], A[0, 1]); k[2] = math.copysign(k[2], A[0, 2])
        elif i == 1:
            k[0] = math.copysign(k[0], A[0, 1]); k[2] = math.copysign(k[2], A[1, 2])
        else:
            k[0] = math.copysign(k[0], A[0, 2]); k[1] = math.copysign(k[1], A[1, 2])
        n = np.linalg.norm(k)
        return theta * (k / n) if n > 1e-9 else np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * axis / (2.0 * math.sin(theta))


def map_hand_to_offset(
    wrist_x: float, wrist_y: float, bbox_diag: float,
    ref_wrist_x: float, ref_wrist_y: float, ref_bbox: float,
    scale: float,
) -> np.ndarray:
    img_x_norm = np.clip((wrist_x - ref_wrist_x) * 2.0, -1.0, 1.0)
    img_y_norm = np.clip((wrist_y - ref_wrist_y) * 2.0, -1.0, 1.0)
    half_depth_range = max((BBOX_NEAR - BBOX_FAR) / 2.0, 1e-6)
    depth_norm = np.clip((bbox_diag - ref_bbox) / half_depth_range, -1.0, 1.0)

    dy_robot = -img_x_norm * scale   # flip so hand-left = robot-left in view
    dz_robot = -img_y_norm * scale   # image y grows downward
    dx_robot = depth_norm * scale
    return np.array([dx_robot, dy_robot, dz_robot])


class OneEuroFilter:
    """Vectorized 1-euro filter (Casiez et al. 2012) for an N-dim signal."""

    def __init__(self, min_cutoff, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

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


def read_mat(r: redis.Redis, key: str) -> np.ndarray | None:
    raw = r.get(key)
    if raw is None:
        return None
    try:
        return np.array(json.loads(raw.decode("utf-8")), dtype=float)
    except (ValueError, TypeError):
        return None


def draw_hud(frame, engaged, fisted, pos, grip_mode, fps, hand_seen, clamped, ori_on, rot_deg):
    h, w = frame.shape[:2]
    if engaged and fisted:
        color, label = (0, 220, 0), "MOVING (fist)"
    elif engaged:
        color, label = (0, 200, 220), "ENGAGED (open hand = freeze)"
    else:
        color, label = (0, 120, 255), "DISENGAGED (press e)"
    cv2.rectangle(frame, (0, 0), (w, 56), (30, 30, 30), -1)
    cv2.putText(frame, "REAL ROBOT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, label, (170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    info = f"pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]"
    if clamped:
        info += " CLAMPED"
    cv2.putText(frame, info, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230) if not clamped else (0, 200, 255), 1)
    if ori_on:
        cv2.putText(frame, f"rot={rot_deg:4.0f}deg", (430, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    if grip_mode is not None:
        gtxt = "grip" if grip_mode == GRIP_CLOSE_CMD else "open"
        cv2.putText(frame, f"gripper={gtxt}", (600, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    bot = (f"e: engage   fist: move+rotate   r: recenter   q: quit   "
           f"fps={fps:4.1f}   hand={'yes' if hand_seen else 'no '}")
    cv2.putText(frame, bot, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


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
    p.add_argument("--ori-gain", type=float, default=ORI_GAIN_DEFAULT,
                   help=f"Wrist-rotation -> robot-rotation gain; lower = gentler "
                        f"(default {ORI_GAIN_DEFAULT})")
    p.add_argument("--no-ori", action="store_true",
                   help="Disable orientation streaming (hold orientation, position only)")
    p.add_argument("--mincutoff", type=float, default=EURO_MIN_CUTOFF,
                   help=f"position 1-euro min cutoff Hz (default {EURO_MIN_CUTOFF})")
    p.add_argument("--beta", type=float, default=EURO_BETA,
                   help=f"position 1-euro speed coefficient (default {EURO_BETA})")
    p.add_argument("--deadband", type=float, default=DEADBAND_M,
                   help=f"skip goal rewrites smaller than this (m) (default {DEADBAND_M})")
    p.add_argument("--no-gripper", action="store_true",
                   help="Don't command the gripper (recommended for first runs)")
    p.add_argument("--no-activate", action="store_true",
                   help="Skip setting active_controller (assume already set)")
    p.add_argument("--no-check", action="store_true",
                   help="Skip the config_file_name == suturebot_grav_real.xml check")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ori_on = not args.no_ori

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()

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

    WINDOW = "Suturebot webcam teleop (REAL + ori)"
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

    print(f"REAL ROBOT teleop + ORIENTATION. Redis {args.host}:{args.port}, robot={ROBOT_NAME}")
    print(f"  scale=±{args.scale*100:.1f}cm   max reach=±{MAX_REACH.tolist()} m")
    print(f"  orientation {'ON (gain %.2f)' % args.ori_gain if ori_on else 'OFF (held)'}")
    print(f"  gripper {'DISABLED' if args.no_gripper else 'enabled (pinch -> o/g)'}")
    print("Keys: 'e' engage   fist = move+rotate   open hand = freeze   'r' recenter   'q' quit")

    engaged = False
    fisted = False
    fist_streak = 0
    offset_filt = np.zeros(3)
    last_hand_time = 0.0
    seed = read_mat(r, K_CUR_POS)
    last_pos = seed.copy() if seed is not None else np.zeros(3)
    last_written: np.ndarray | None = None
    grip_mode: str | None = None
    rot_deg = 0.0

    anchor_pos = last_pos.copy()
    ref_wrist_x = 0.5
    ref_wrist_y = 0.5
    ref_bbox = (BBOX_NEAR + BBOX_FAR) / 2.0
    R_hand_ref = np.eye(3)     # hand frame at engage (camera)
    R_robot_ref = np.eye(3)    # robot orientation at engage (base)
    euro_pos = OneEuroFilter(args.mincutoff * AXIS_CUTOFF_SCALE, args.beta)
    euro_ori = OneEuroFilter(ORI_MIN_CUTOFF, ORI_BETA)

    frame_count = 0
    t_fps = time.monotonic()
    fps = 0.0

    def reanchor(wrist, bbox_diag, R_hand_now) -> bool:
        """Anchor position AND orientation at the robot's ACTUAL current pose."""
        nonlocal anchor_pos, ref_wrist_x, ref_wrist_y, ref_bbox, last_pos
        nonlocal last_written, R_hand_ref, R_robot_ref
        actual = read_mat(r, K_CUR_POS)
        if actual is None:
            print("No current_position from OpenSai; is the controller running? Not engaging.")
            return False
        anchor_pos = actual.copy()
        last_pos = actual.copy()
        last_written = None
        ref_wrist_x, ref_wrist_y, ref_bbox = wrist.x, wrist.y, bbox_diag
        offset_filt[:] = 0.0
        euro_pos.reset()
        euro_ori.reset()
        cur_ori = read_mat(r, K_CUR_ORI)
        R_robot_ref = cur_ori if cur_ori is not None else np.eye(3)
        R_hand_ref = R_hand_now if R_hand_now is not None else np.eye(3)
        if cur_ori is not None:
            r.set(K_GOAL_ORI, json.dumps(R_robot_ref.tolist()))
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
        R_goal_ori = None
        if hand_seen:
            last_hand_time = time.monotonic()
            lm = res.hand_landmarks[0]
            wrist, thumb, index, middle_mcp = lm[0], lm[4], lm[8], lm[9]

            # Hand orientation from metric world landmarks (camera frame).
            R_hand_now = None
            if res.hand_world_landmarks:
                R_hand_now = hand_frame(res.hand_world_landmarks[0])

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
                if reanchor(wrist, bbox_diag, R_hand_now):
                    fisted = True
            elif fist_falling:
                fisted = False

            offset = map_hand_to_offset(
                wrist.x, wrist.y, bbox_diag,
                ref_wrist_x, ref_wrist_y, ref_bbox, args.scale,
            )
            offset_filt = euro_pos(offset, time.monotonic())

            # Orientation: relative hand rotation -> robot frame -> smoothed.
            if ori_on and R_hand_now is not None:
                R_rel_cam = R_hand_now @ R_hand_ref.T
                rotvec_cam = matrix_to_rotvec(R_rel_cam)
                rotvec = ORI_AXIS_MAP @ rotvec_cam * args.ori_gain
                ang = float(np.linalg.norm(rotvec))
                if ang > MAX_ROT:
                    rotvec = rotvec / ang * MAX_ROT
                rotvec_filt = euro_ori(rotvec, time.monotonic())
                rot_deg = math.degrees(float(np.linalg.norm(rotvec_filt)))
                R_goal_ori = rotvec_to_matrix(rotvec_filt) @ R_robot_ref

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
            if R_goal_ori is not None:
                r.set(K_GOAL_ORI, json.dumps(R_goal_ori.tolist()))
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

        draw_hud(frame, engaged, fisted, last_pos, grip_mode, fps, hand_seen, clamped, ori_on, rot_deg)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            engaged = not engaged
            if engaged:
                print("Engaged. Make a fist to move + rotate; open hand to freeze.")
            else:
                fisted = False
                print("Disengaged. No writes until you press 'e' again.")
        if key == ord('r') and engaged and fisted and hand_seen:
            if reanchor(wrist, bbox_diag, R_hand_now):
                print(f"Recentered. anchor={anchor_pos.tolist()}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
