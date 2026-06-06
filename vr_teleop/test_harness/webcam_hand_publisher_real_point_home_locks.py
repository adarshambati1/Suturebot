"""Webcam hand teleop for REAL Rizon4s -- point/pinch + home + axis locks.

Same as webcam_hand_publisher_real_point_home.py, with one addition: per-axis
LOCKS you can toggle from the keyboard. A locked axis ignores hand motion so
the robot only moves along the axes you leave free -- handy for constrained
work like sliding along a single direction or holding orientation while
translating.

Locks (each press toggles; both translation and rotation locks are independent):
    'x' / 'y' / 'z'                -> lock robot X / Y / Z translation
    'X' / 'Y' / 'Z' (shift+letter) -> lock rotation about robot X / Y / Z
                                      (= roll / pitch / yaw with X=forward,
                                      Y=lateral, Z=up).

Locked translation: the offset for that axis is forced to 0, so target stays
at anchor's value on that axis. Locked rotation: that rotvec component (in the
robot frame, after ORI_AXIS_MAP) is forced to 0, so the EE's rotation about
that robot axis holds at the engage value. Locks persist across clutches and
home returns; toggle again to release.

Other controls (unchanged):
    'e'    -> engaged / disengaged (high-level).
    point  -> active: move + rotate while held (middle/ring/pinky curled).
    pinch  -> toggle gripper open<->grip (one flip per pinch).
    'r'    -> recenter while active (re-anchor here).
    'h'    -> return to the home pose captured at startup (disengages).
    'q'    -> quit. Auto-disengages if no hand for > 1.0 s.

Usage:
    python vr_teleop/test_harness/webcam_hand_publisher_real_point_home_locks.py --port 6380
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

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.expanduser("~/.cache/suturebot/hand_landmarker.task")

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

MAX_REACH = np.array([0.40, 0.40, 0.40])

ORI_GAIN_DEFAULT = 0.5
MAX_ROT = math.pi / 2

ORI_AXIS_MAP = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

ORI_MIN_CUTOFF = 0.4
ORI_BETA = 0.1

ACTIVE_TIP_PIP = [(12, 10), (16, 14), (20, 18)]
ACTIVE_MIN_CURLED = 2
POSE_DEBOUNCE = 3

PINCH_TOGGLE_BELOW = 0.40
PINCH_RELEASE_ABOVE = 0.60

BBOX_NEAR = 0.55
BBOX_FAR  = 0.20
DEPTH_LM_IDX = [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

EURO_MIN_CUTOFF = 0.01
EURO_BETA = 3.0
AXIS_CUTOFF_SCALE = np.array([0.6, 0.3, 1.0])

DEADBAND_M = 0.0
NO_HAND_TIMEOUT = 1.0

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


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


def is_active_pose(lm) -> bool:
    wrist = lm[0]
    curled = 0
    for tip_i, pip_i in ACTIVE_TIP_PIP:
        d_tip = (lm[tip_i].x - wrist.x) ** 2 + (lm[tip_i].y - wrist.y) ** 2
        d_pip = (lm[pip_i].x - wrist.x) ** 2 + (lm[pip_i].y - wrist.y) ** 2
        if d_tip < d_pip:
            curled += 1
    return curled >= ACTIVE_MIN_CURLED


def hand_frame(world_lm) -> np.ndarray | None:
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
    side = np.cross(up, normal)
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

    dy_robot = -img_x_norm * scale
    dz_robot = -img_y_norm * scale
    dx_robot = depth_norm * scale
    return np.array([dx_robot, dy_robot, dz_robot])


class OneEuroFilter:
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


def draw_hud(frame, engaged, active, pos, grip_mode, fps, hand_seen, clamped,
             ori_on, rot_deg, homing, lock_pos, lock_rot):
    h, w = frame.shape[:2]
    if homing:
        color, label = (255, 180, 0), "RETURNING TO HOME"
    elif engaged and active:
        color, label = (0, 220, 0), "MOVING (point)"
    elif engaged:
        color, label = (0, 200, 220), "ENGAGED (open fingers = freeze)"
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
    gtxt = "?" if grip_mode is None else ("grip" if grip_mode == GRIP_CLOSE_CMD else "open")
    cv2.putText(frame, f"gripper={gtxt}", (600, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    # Locks: letter = locked, '-' = free. Position uses X/Y/Z, rotation uses r/p/y.
    p_chars = ''.join(c if l else '-' for c, l in zip('XYZ', lock_pos))
    r_chars = ''.join(c if l else '-' for c, l in zip('rpy', lock_rot))
    any_lock = bool(lock_pos.any() or lock_rot.any())
    cv2.putText(frame, f"lock pos[{p_chars}] rot[{r_chars}]", (10, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 200, 255) if any_lock else (140, 140, 140), 1)

    bot = (f"e engage  point=move  pinch=grip  h home  r recenter  "
           f"x/y/z lock trans  X/Y/Z lock rot  q quit  "
           f"fps={fps:4.1f}  hand={'y' if hand_seen else 'n'}")
    cv2.putText(frame, bot, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Redis host (the Linux PC, or 127.0.0.1 via SSH tunnel)")
    p.add_argument("--port", type=int, default=6379, help="Redis port")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--scale", type=float, default=0.20,
                   help="Half-range of robot motion in meters (default: 0.20)")
    p.add_argument("--ori-gain", type=float, default=ORI_GAIN_DEFAULT,
                   help=f"Wrist-rotation -> robot-rotation gain (default {ORI_GAIN_DEFAULT})")
    p.add_argument("--no-ori", action="store_true",
                   help="Disable orientation streaming (hold orientation, position only)")
    p.add_argument("--mincutoff", type=float, default=EURO_MIN_CUTOFF,
                   help=f"position 1-euro min cutoff Hz (default {EURO_MIN_CUTOFF})")
    p.add_argument("--beta", type=float, default=EURO_BETA,
                   help=f"position 1-euro speed coefficient (default {EURO_BETA})")
    p.add_argument("--deadband", type=float, default=DEADBAND_M,
                   help=f"skip goal rewrites smaller than this (m) (default {DEADBAND_M})")
    p.add_argument("--no-gripper", action="store_true",
                   help="Ignore pinches; never command the gripper")
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

    home_pos = read_mat(r, K_CUR_POS)
    home_ori = read_mat(r, K_CUR_ORI)
    if home_pos is None:
        print("Warning: couldn't read current_position at startup; 'h' (home) disabled.")
    else:
        print(f"HOME captured: pos={home_pos.tolist()}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    WINDOW = "Suturebot webcam teleop (REAL: point/pinch + home + locks)"
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

    print(f"REAL ROBOT teleop (point/pinch + home + locks). Redis {args.host}:{args.port}, robot={ROBOT_NAME}")
    print(f"  scale=±{args.scale*100:.1f}cm   max reach=±{MAX_REACH.tolist()} m")
    print(f"  orientation {'ON (gain %.2f)' % args.ori_gain if ori_on else 'OFF (held)'}")
    print(f"  gripper {'DISABLED' if args.no_gripper else 'pinch to toggle open<->grip'}")
    print("Keys: 'e' engage | point=move+rotate | pinch=toggle grip | 'h' home | 'r' recenter")
    print("      x/y/z = lock translation axis    X/Y/Z = lock rotation axis (roll/pitch/yaw)")
    print("      'q' quit")

    engaged = False
    active = False
    pose_streak = 0
    pinched = False
    grip_closed = False
    lock_pos = np.array([False, False, False])   # x, y, z translation locks
    lock_rot = np.array([False, False, False])   # rotation locks (roll, pitch, yaw = rot about robot X, Y, Z)
    offset_filt = np.zeros(3)
    last_hand_time = 0.0
    seed = read_mat(r, K_CUR_POS)
    last_pos = seed.copy() if seed is not None else np.zeros(3)
    last_written: np.ndarray | None = None
    grip_mode: str | None = None
    rot_deg = 0.0
    homing_until = 0.0

    anchor_pos = last_pos.copy()
    ref_wrist_x = 0.5
    ref_wrist_y = 0.5
    ref_bbox = (BBOX_NEAR + BBOX_FAR) / 2.0
    R_hand_ref = np.eye(3)
    R_robot_ref = np.eye(3)
    euro_pos = OneEuroFilter(args.mincutoff * AXIS_CUTOFF_SCALE, args.beta)
    euro_ori = OneEuroFilter(ORI_MIN_CUTOFF, ORI_BETA)

    frame_count = 0
    t_fps = time.monotonic()
    fps = 0.0

    def reanchor(wrist, bbox_diag, R_hand_now) -> bool:
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

    def go_home() -> None:
        nonlocal engaged, active, last_written, last_pos
        if home_pos is None:
            print("No home pose to return to.")
            return
        r.set(K_GOAL_POS, json.dumps(home_pos.tolist()))
        if home_ori is not None:
            r.set(K_GOAL_ORI, json.dumps(home_ori.tolist()))
        engaged = False
        active = False
        last_written = home_pos.copy()
        last_pos = home_pos.copy()
        print(f"-> HOME pos={home_pos.tolist()}. Disengaged; press 'e' to resume.")

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

        R_goal_ori = None
        if hand_seen:
            last_hand_time = time.monotonic()
            lm = res.hand_landmarks[0]
            wrist, thumb, index, middle_mcp = lm[0], lm[4], lm[8], lm[9]

            R_hand_now = None
            if res.hand_world_landmarks:
                R_hand_now = hand_frame(res.hand_world_landmarks[0])

            xs = np.array([lm[i].x for i in DEPTH_LM_IDX])
            ys = np.array([lm[i].y for i in DEPTH_LM_IDX])
            bbox_diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))

            if is_active_pose(lm):
                pose_streak += 1
            else:
                pose_streak = 0
            pose_rising = (not active) and pose_streak >= POSE_DEBOUNCE
            pose_falling = active and pose_streak == 0

            if engaged and pose_rising:
                if reanchor(wrist, bbox_diag, R_hand_now):
                    active = True
            elif pose_falling:
                active = False

            offset = map_hand_to_offset(
                wrist.x, wrist.y, bbox_diag,
                ref_wrist_x, ref_wrist_y, ref_bbox, args.scale,
            )
            offset_filt = euro_pos(offset, time.monotonic())

            if ori_on and R_hand_now is not None:
                R_rel_cam = R_hand_now @ R_hand_ref.T
                rotvec_cam = matrix_to_rotvec(R_rel_cam)
                rotvec = ORI_AXIS_MAP @ rotvec_cam * args.ori_gain
                ang = float(np.linalg.norm(rotvec))
                if ang > MAX_ROT:
                    rotvec = rotvec / ang * MAX_ROT
                rotvec_filt = euro_ori(rotvec, time.monotonic())
                # Zero out locked rotation axes (in robot rotvec frame).
                rotvec_masked = np.where(lock_rot, 0.0, rotvec_filt)
                rot_deg = math.degrees(float(np.linalg.norm(rotvec_masked)))
                R_goal_ori = rotvec_to_matrix(rotvec_masked) @ R_robot_ref

            hand_size = float(np.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y))
            pinch_norm = float(np.hypot(thumb.x - index.x, thumb.y - index.y)) / max(hand_size, 1e-6)
            if pinch_norm < PINCH_TOGGLE_BELOW and not pinched:
                pinched = True
                if engaged and not args.no_gripper:
                    grip_closed = not grip_closed
                    grip_mode = GRIP_CLOSE_CMD if grip_closed else GRIP_OPEN_CMD
                    r.set(K_GRIP_MODE, grip_mode)
                    print(f"Gripper -> {'GRIP' if grip_closed else 'OPEN'}")
            elif pinch_norm > PINCH_RELEASE_ABOVE:
                pinched = False

            fh, fw = frame.shape[:2]
            draw_hand(frame, lm, fw, fh)
        else:
            pose_streak = 0
            active = False
            pinched = False

        if engaged and (time.monotonic() - last_hand_time) > NO_HAND_TIMEOUT:
            engaged = False
            active = False
            print("Auto-disengaged: no hand detected.")

        # Apply translation locks: zero the offset on locked axes so those
        # components stay at anchor_pos.
        offset_masked = np.where(lock_pos, 0.0, offset_filt)
        target_raw = anchor_pos + offset_masked
        target_pos = np.clip(target_raw, anchor_pos - MAX_REACH, anchor_pos + MAX_REACH)
        clamped = bool(np.any(target_pos != target_raw))

        if engaged and active and hand_seen:
            if last_written is None or np.linalg.norm(target_pos - last_written) > args.deadband:
                r.set(K_GOAL_POS, json.dumps(target_pos.tolist()))
                last_written = target_pos.copy()
            if R_goal_ori is not None:
                r.set(K_GOAL_ORI, json.dumps(R_goal_ori.tolist()))
            last_pos = target_pos

        frame_count += 1
        if frame_count >= 15:
            now = time.monotonic()
            fps = frame_count / (now - t_fps)
            frame_count = 0
            t_fps = now

        homing_banner = time.monotonic() < homing_until
        draw_hud(frame, engaged, active, last_pos, grip_mode, fps, hand_seen,
                 clamped, ori_on, rot_deg, homing_banner, lock_pos, lock_rot)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            engaged = not engaged
            if engaged:
                print("Engaged. Point (curl middle/ring/pinky) to move+rotate; pinch to toggle grip.")
            else:
                active = False
                print("Disengaged. No writes until you press 'e' again.")
        if key == ord('r') and engaged and active and hand_seen:
            if reanchor(wrist, bbox_diag, R_hand_now):
                print(f"Recentered. anchor={anchor_pos.tolist()}")
        if key == ord('h'):
            go_home()
            homing_until = time.monotonic() + 2.0

        # Translation-axis locks (lowercase).
        if key == ord('x'):
            lock_pos[0] = not lock_pos[0]
            print(f"Lock X (translation): {'ON' if lock_pos[0] else 'OFF'}")
        if key == ord('y'):
            lock_pos[1] = not lock_pos[1]
            print(f"Lock Y (translation): {'ON' if lock_pos[1] else 'OFF'}")
        if key == ord('z'):
            lock_pos[2] = not lock_pos[2]
            print(f"Lock Z (translation): {'ON' if lock_pos[2] else 'OFF'}")
        # Rotation-axis locks (uppercase = shift + letter).
        if key == ord('X'):
            lock_rot[0] = not lock_rot[0]
            print(f"Lock roll (rot about X): {'ON' if lock_rot[0] else 'OFF'}")
        if key == ord('Y'):
            lock_rot[1] = not lock_rot[1]
            print(f"Lock pitch (rot about Y): {'ON' if lock_rot[1] else 'OFF'}")
        if key == ord('Z'):
            lock_rot[2] = not lock_rot[2]
            print(f"Lock yaw (rot about Z): {'ON' if lock_rot[2] else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
