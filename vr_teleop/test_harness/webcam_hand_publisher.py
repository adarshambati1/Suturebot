"""Webcam hand-tracking publisher for Rizon4s teleop.

Reads the laptop webcam, runs MediaPipe Hands, and maps hand pose to the
robot's Cartesian goal_position via Redis. Writes the *same* keys as
fake_quest_publisher.py and the Unity RedisTeleopBridge so OpenSai
doesn't know the difference.

Mapping (right-handed, all motion relative to scene home):
    hand x in image    -> robot Y  (lateral; image mirrored so it feels natural)
    hand y in image    -> robot Z  (up in image = up in robot)
    hand bbox diagonal -> robot X  (bigger hand = closer to camera = +X)
    pinch (thumb-index distance, normalized by hand size) -> gripper width

Deadman / clutch:
    Two-layer control:
      'e'          -> engaged/disengaged (high-level). Disengaged = no writes.
      fist gesture -> active control while engaged. Robot follows your hand
                      only while you're holding a fist (index/middle/ring/pinky
                      curled; thumb position is free so pinch->gripper still
                      works). Open hand = robot freezes.

    On each fist (rising edge) the *current* hand position becomes the new
    zero, so the robot doesn't jump. To reposition: open your hand, move it
    wherever feels good, then re-clench. Press 'r' to recenter while still
    fisted. Press 'q' to quit. Auto-disengages if no hand for > 1.0 s.

Install (one-time):
    pip install mediapipe opencv-python redis numpy

Usage:
    # Match the XML OpenSai is running
    python vr_teleop/test_harness/webcam_hand_publisher.py --scene oussama_push
    python vr_teleop/test_harness/webcam_hand_publisher.py --scene grav_motion

    # Tune motion scale (defaults to small for safety)
    python vr_teleop/test_harness/webcam_hand_publisher.py --scene grav_motion --scale 0.04
"""

from __future__ import annotations

import argparse
import json
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

# Edges between landmark indices for drawing the hand skeleton.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
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


# Tip / PIP landmark indices for the four non-thumb fingers.
# A finger is "curled" if its tip is closer to the wrist than its PIP joint
# (rotation-invariant, no assumption about palm orientation).
FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]
FIST_MIN_CURLED = 3   # at least 3 of 4 fingers curled -> fist


def is_fist(lm) -> bool:
    wrist = lm[0]
    curled = 0
    for tip_i, pip_i in FINGER_TIP_PIP:
        d_tip = (lm[tip_i].x - wrist.x) ** 2 + (lm[tip_i].y - wrist.y) ** 2
        d_pip = (lm[pip_i].x - wrist.x) ** 2 + (lm[pip_i].y - wrist.y) ** 2
        if d_tip < d_pip:
            curled += 1
    return curled >= FIST_MIN_CURLED

# Redis keys — keep these in sync with python_examples/*.py,
# vr_teleop/unity_scripts/RedisTeleopBridge.cs, and fake_quest_publisher.py.
ROBOT_NAME = "Rizon4s"
K_GOAL_POS = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position"
K_GOAL_ORI = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_orientation"
K_GRIP     = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::gripper_fingers::goal_position"
K_ACTIVE   = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"

CONTROLLER_TO_USE = "cartesian_controller"

# Scene presets — match fake_quest_publisher.py.
SCENES = {
    "grav_motion": {
        "home": np.array([0.51, -0.20, 0.03]),
        "ori": np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ]),
        "scale_default": 0.04,   # 4 cm half-range
    },
    "oussama_push": {
        "home": np.array([0.6686, 0.1485, 0.3146]),
        "ori": np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ]),
        "scale_default": 0.03,   # 3 cm half-range
    },
}

# Gripper width range (meters) — matches python_examples conventions.
GRIP_CLOSED = 0.005
GRIP_OPEN   = 0.05

# Pinch distance (normalized by hand size) -> gripper interpolation.
# Below PINCH_MIN -> fully closed; above PINCH_MAX -> fully open.
PINCH_MIN = 0.25
PINCH_MAX = 1.20

# Hand bbox diagonal (normalized image coords) range that maps to ±X scale.
# Tune if your camera/hand distance is unusual; defaults assume ~50 cm.
BBOX_NEAR = 0.55   # very close to camera
BBOX_FAR  = 0.20   # arm extended

# EMA smoothing on the mapped offsets (0 = no smoothing, 1 = frozen).
SMOOTH_ALPHA = 0.65

# Auto-disengage if no hand for this long (s).
NO_HAND_TIMEOUT = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene", choices=sorted(SCENES.keys()), default="oussama_push",
                   help="Which scene this is being run against. (default: oussama_push)")
    p.add_argument("--host", default="127.0.0.1", help="Redis host")
    p.add_argument("--port", type=int, default=6379, help="Redis port")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--scale", type=float, default=None,
                   help="Half-range of robot motion in meters (defaults per scene)")
    p.add_argument("--no-gripper", action="store_true",
                   help="Skip writing gripper width")
    p.add_argument("--no-activate", action="store_true",
                   help="Skip setting active_controller (assume already set)")
    p.add_argument("--mirror", action="store_true", default=True,
                   help="Mirror the camera image horizontally (default: on)")
    return p.parse_args()


def map_hand_to_offset(
    wrist_x: float,
    wrist_y: float,
    bbox_diag: float,
    ref_wrist_x: float,
    ref_wrist_y: float,
    ref_bbox: float,
    scale: float,
) -> np.ndarray:
    """Convert hand state to a (dx, dy, dz) offset *relative to a reference*.

    Image coords: x in [0,1] left->right, y in [0,1] top->bottom.
    Robot frame: see scene ORI. We treat image-x as robot-Y (lateral),
    image-y as robot-Z (up = -image y), bbox size as robot-X (depth).

    A hand displacement of 0.5 in normalized image coords (half the frame)
    maps to `scale` meters at the robot. Same for bbox: a depth change of
    half the BBOX_NEAR-BBOX_FAR range maps to `scale` meters.
    """
    img_x_norm = np.clip((wrist_x - ref_wrist_x) * 2.0, -1.0, 1.0)
    img_y_norm = np.clip((wrist_y - ref_wrist_y) * 2.0, -1.0, 1.0)

    half_depth_range = max((BBOX_NEAR - BBOX_FAR) / 2.0, 1e-6)
    depth_norm = np.clip((bbox_diag - ref_bbox) / half_depth_range, -1.0, 1.0)

    dy_robot = -img_x_norm * scale   # flip: hand-left = robot-left in mirrored view
    dz_robot = -img_y_norm * scale   # image y grows downward
    dx_robot = depth_norm * scale

    return np.array([dx_robot, dy_robot, dz_robot])


def pinch_to_width(pinch_norm: float) -> float:
    t = (pinch_norm - PINCH_MIN) / max(PINCH_MAX - PINCH_MIN, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    return GRIP_CLOSED + t * (GRIP_OPEN - GRIP_CLOSED)


def draw_hud(
    frame: np.ndarray,
    engaged: bool,
    fisted: bool,
    pos: np.ndarray,
    width: float | None,
    fps: float,
    hand_seen: bool,
) -> None:
    h, w = frame.shape[:2]
    if engaged and fisted:
        color = (0, 220, 0)
        label = "MOVING (fist)"
    elif engaged:
        color = (0, 200, 220)
        label = "ENGAGED (open hand = freeze)"
    else:
        color = (0, 120, 255)
        label = "DISENGAGED (press e)"
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    info = f"pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]"
    cv2.putText(frame, info, (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1)

    if width is not None:
        cv2.putText(frame, f"grip={width*1000:4.1f}mm", (560, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    bot = (f"e: engage/clutch   r: recenter   q: quit   "
           f"fps={fps:4.1f}   hand={'yes' if hand_seen else 'no '}")
    cv2.putText(frame, bot, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)


def main() -> None:
    args = parse_args()
    scene = SCENES[args.scene]
    home = scene["home"]
    ori = scene["ori"]
    scale = args.scale if args.scale is not None else scene["scale_default"]

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()

    if not args.no_activate:
        while r.get(K_ACTIVE) is None or r.get(K_ACTIVE).decode() != CONTROLLER_TO_USE:
            r.set(K_ACTIVE, CONTROLLER_TO_USE)
            time.sleep(0.05)

    # Orientation is held fixed in this first pass (same as fake publisher).
    r.set(K_GOAL_ORI, json.dumps(ori.tolist()))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # WINDOW_NORMAL lets you drag-resize the window (AUTOSIZE locks it).
    WINDOW = "Suturebot webcam teleop"
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

    print(f"Scene: {args.scene}   home={home.tolist()}   scale=±{scale*100:.1f}cm")
    print(f"Publishing to {args.host}:{args.port} on {K_GOAL_POS}")
    print("Keys: 'e' engage/disengage   fist = move   open hand = freeze")
    print("      'r' recenter (while fisted)   'q' quit")
    print("Window must be focused for key input.")

    engaged = False
    fisted = False           # debounced fist state
    fist_streak = 0          # consecutive frames the fist has been detected
    FIST_DEBOUNCE = 3        # frames needed before "fist" is committed
    offset_filt = np.zeros(3)
    last_hand_time = 0.0
    last_pos = home.copy()
    last_width: float | None = None

    # Clutch state: while engaged AND fisted, anchor_pos is where the robot
    # was when the fist formed and ref_hand is where the hand was then. Robot
    # follows hand_now - ref_hand from anchor_pos, so unfist+move+refist
    # never jumps.
    anchor_pos = home.copy()
    ref_wrist_x = 0.5
    ref_wrist_y = 0.5
    ref_bbox = (BBOX_NEAR + BBOX_FAR) / 2.0

    frame_count = 0
    t_fps = time.monotonic()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; retrying.")
            continue
        if args.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.monotonic() * 1000)
        res = landmarker.detect_for_video(mp_image, ts_ms)
        hand_seen = bool(res.hand_landmarks)

        if hand_seen:
            last_hand_time = time.monotonic()
            lm = res.hand_landmarks[0]
            wrist = lm[0]
            thumb = lm[4]
            index = lm[8]
            middle_mcp = lm[9]

            xs = np.array([p.x for p in lm])
            ys = np.array([p.y for p in lm])
            bbox_diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))

            # Debounced fist detection. Engage (rising edge) is slightly
            # delayed to filter detector flicker; release is immediate so
            # the robot freezes the instant the hand opens.
            if is_fist(lm):
                fist_streak += 1
            else:
                fist_streak = 0

            fist_rising = (not fisted) and fist_streak >= FIST_DEBOUNCE
            fist_falling = fisted and fist_streak == 0
            if fist_rising:
                fisted = True
            elif fist_falling:
                fisted = False

            if engaged and fist_rising:
                # Re-anchor on every new fist: current robot pose becomes
                # the base, current hand state becomes the new zero.
                anchor_pos = last_pos.copy()
                ref_wrist_x = wrist.x
                ref_wrist_y = wrist.y
                ref_bbox = bbox_diag
                offset_filt[:] = 0.0

            raw_offset = map_hand_to_offset(
                wrist.x, wrist.y, bbox_diag,
                ref_wrist_x, ref_wrist_y, ref_bbox,
                scale,
            )
            offset_filt = SMOOTH_ALPHA * offset_filt + (1.0 - SMOOTH_ALPHA) * raw_offset

            # Pinch: thumb-index distance, normalized by hand size (wrist->middle MCP)
            # so pinch is invariant to how close the hand is to the camera.
            hand_size = float(np.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y))
            pinch_raw = float(np.hypot(thumb.x - index.x, thumb.y - index.y))
            pinch_norm = pinch_raw / max(hand_size, 1e-6)
            width = pinch_to_width(pinch_norm)

            fh, fw = frame.shape[:2]
            draw_hand(frame, lm, fw, fh)
        else:
            width = last_width
            fist_streak = 0
            fisted = False

        # Auto-disengage if hand lost.
        if engaged and (time.monotonic() - last_hand_time) > NO_HAND_TIMEOUT:
            engaged = False
            fisted = False
            print("Auto-disengaged: no hand detected.")

        target_pos = anchor_pos + offset_filt
        if engaged and fisted and hand_seen:
            r.set(K_GOAL_POS, json.dumps(target_pos.tolist()))
            if not args.no_gripper and width is not None:
                r.set(K_GRIP, json.dumps([width]))
                last_width = width
            last_pos = target_pos

        # FPS counter (rolling).
        frame_count += 1
        if frame_count >= 15:
            now = time.monotonic()
            fps = frame_count / (now - t_fps)
            frame_count = 0
            t_fps = now

        draw_hud(frame, engaged, fisted, last_pos, last_width, fps, hand_seen)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            engaged = not engaged
            if engaged:
                print("Engaged. Make a fist to move the robot; open hand to freeze.")
            else:
                fisted = False
                print("Disengaged. No writes until you press 'e' again.")
        if key == ord('r') and engaged and fisted and hand_seen:
            # Recenter without opening the hand: re-anchor here.
            anchor_pos = last_pos.copy()
            ref_wrist_x = wrist.x
            ref_wrist_y = wrist.y
            ref_bbox = bbox_diag
            offset_filt[:] = 0.0
            print(f"Recentered. anchor={anchor_pos.tolist()}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
