"""fruit_scan.py — scan a table, find a fruit, find the blue-strip "cut", hover above it.

Everything talks over local Redis, so this is meant to run **on the robot
machine** alongside `realsense_publisher.py` (wrist-mounted Intel RealSense
D405) and OpenSai. The pipeline:

  1. SCAN   — sweep the wrist camera over the table by driving the robot through
              a small grid of poses CENTRED ON THE ARM'S CURRENT POSE: same xy
              neighbourhood, same height, same orientation it started in (a
              gentle local search, no jump, configuration preserved). At each
              pose, grab the live color frame (``suturebot::realsense::color``)
              and run YOLO. Stop at the first pose where a fruit is detected.
  2. CUT    — inside the fruit's bounding box, HSV-threshold for the blue strip
              that marks the cut and report whether / where it was found.
  3. LOCATE — back-project the fruit-center pixel through the live depth frame
              (``suturebot::realsense::depth``) + intrinsics to a camera-frame 3D
              point, lift it to the flange frame via the hand-eye extrinsic
              ``T_FLANGE_CAM``, then to world using the robot's live flange pose.
  4. HOVER  — command the robot to hover ``APPROACH_HEIGHT`` above the fruit.

After the scan it writes an annotated overlay (bounding box + label, blue strip,
fruit centre, target) to ``log_files/vision_logs/`` and, with ``--show``, pops a
live window so you can see the bounding box.

Sim vs. real — the ``ROBOT_NAME`` toggle, like every other client. IMPORTANT: the
OpenSai sim has no RealSense, so the full scan→hover loop only runs for real on
the robot machine. For local development:
  * ``--no-robot``     skip ALL motion; just detect on whatever frame is available.
  * ``--image PATH``   read a still image instead of Redis (no depth; uses
                       ``--assume-depth``). Good for validating detection + the
                       annotated overlay offline.

CALIBRATION: the scan grid is anchored to the live start pose, so it needs no
table coordinates. ``APPROACH_HEIGHT`` and especially the hand-eye extrinsic
``T_FLANGE_CAM`` are placeholders in the same hand-tuned style as the other
clients. The 3D math is correct; ``T_FLANGE_CAM`` must be calibrated on the real
cell before the hover target is trustworthy. Position the arm above the table
(camera looking down at it) before starting, since the sweep searches from there.

Install:  pip install redis numpy opencv-python ultralytics
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
import redis


# ----------------------------------------------------------------------------
# Sim vs. real — same toggle as the other clients.
# ----------------------------------------------------------------------------
# "Rizon4s" -> OpenSai simulation; "Titania" -> real Flexiv driver.
ROBOT_NAME = "Rizon4s"

# This client only reads camera keys and commands the cartesian controller, so
# it is compatible with any of the grav scene XMLs. We verify against the
# outward-push scene by default (matches the most complete suturing clients);
# override with --config-file if you launch a different scene.
CONFIG_FILE_FOR_THIS_SCRIPT = (
    "suturebot_grav_real.xml" if ROBOT_NAME == "Titania" else "suturebot_grav_oussama_push.xml"
)
CONTROLLER_TO_USE = "cartesian_controller"


# ----------------------------------------------------------------------------
# Redis keys — robot control (namespaced by ROBOT_NAME) + camera feed.
# ----------------------------------------------------------------------------
@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_orientation"
    )
    cartesian_task_current_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_orientation"
    )
    active_controller: str = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"
    # Camera feed published by realsense_publisher.py.
    rs_color: str = "suturebot::realsense::color"
    rs_depth: str = "suturebot::realsense::depth"
    rs_intrinsics: str = "suturebot::realsense::intrinsics"


REDIS_KEYS = RedisKeys()


# ----------------------------------------------------------------------------
# Geometry — HAND-CALIBRATED PLACEHOLDERS (world frame, metres). Tune on the cell.
# ----------------------------------------------------------------------------
# How far above the detected fruit the flange should hover at the end.
APPROACH_HEIGHT = 0.15

# The scan sweep is anchored to the arm's live start pose (xy/height/orientation),
# so there is no fixed scan region or camera-down orientation to calibrate — the
# motion is a local search from wherever the arm already is.

# Hand-eye extrinsic: pose of the CAMERA frame expressed in the FLANGE frame.
# p_flange = R @ p_cam + t.  PLACEHOLDER — must be calibrated for the real D405
# mount (config_folder/.../visual/d405-camera-mount.obj). Identity rotation with
# a small -Z offset is a reasonable starting guess for a down-looking wrist cam;
# the structure is right, the numbers are not.
T_FLANGE_CAM = np.array([
    [1.0, 0.0, 0.0,  0.000],
    [0.0, 1.0, 0.0,  0.000],
    [0.0, 0.0, 1.0,  0.050],
    [0.0, 0.0, 0.0,  1.000],
])

# Fruit classes to accept (COCO names as used by ultralytics models).
DEFAULT_FRUIT_CLASSES = ("apple", "orange", "banana")

# Blue-strip HSV thresholds (OpenCV HSV: H 0-179, S/V 0-255).
BLUE_HSV_LO = np.array([100, 80, 40], dtype=np.uint8)
BLUE_HSV_HI = np.array([130, 255, 255], dtype=np.uint8)
MIN_BLUE_AREA_PX = 60   # ignore specks smaller than this inside the fruit box

MOVE_TIME = 2.5         # seconds to settle at each scan pose / the hover pose


class State(Enum):
    SCAN = auto()
    FOUND = auto()
    HOVER = auto()
    DONE = auto()


# ----------------------------------------------------------------------------
# Robot I/O
# ----------------------------------------------------------------------------
def set_goal(r: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    r.set(REDIS_KEYS.cartesian_task_goal_position, json.dumps(np.asarray(pos).tolist()))
    r.set(REDIS_KEYS.cartesian_task_goal_orientation, json.dumps(np.asarray(ori).tolist()))


def read_actual_pose(r: redis.Redis):
    p = r.get(REDIS_KEYS.cartesian_task_current_position)
    o = r.get(REDIS_KEYS.cartesian_task_current_orientation)
    if p is None or o is None:
        return None, None
    pos = np.array(json.loads(p))
    ori = np.array(json.loads(o))
    return pos, ori


def verify_and_activate(r: redis.Redis, expected_config: str) -> bool:
    cfg = r.get(REDIS_KEYS.config_file_name)
    if cfg is None or cfg.decode("utf-8") != expected_config:
        got = None if cfg is None else cfg.decode("utf-8")
        print(f"[robot] OpenSai is running '{got}', this client expects "
              f"'{expected_config}'. Launch the matching scene, or pass "
              f"--config-file / use --no-robot.")
        return False
    # Force the cartesian controller active.
    cur = r.get(REDIS_KEYS.active_controller)
    tries = 0
    while (cur is None or cur.decode("utf-8") != CONTROLLER_TO_USE) and tries < 50:
        r.set(REDIS_KEYS.active_controller, CONTROLLER_TO_USE)
        time.sleep(0.05)
        cur = r.get(REDIS_KEYS.active_controller)
        tries += 1
    return True


def move_to(r: redis.Redis, pos: np.ndarray, ori: np.ndarray, dwell: float, msg: str) -> None:
    print(f"[robot] {msg}: flange -> ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    set_goal(r, pos, ori)
    time.sleep(dwell)
    actual, _ = read_actual_pose(r)
    if actual is not None:
        err = actual - pos
        print(f"        actual ({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})"
              f"  err ({err[0]:+.3f}, {err[1]:+.3f}, {err[2]:+.3f})")


# ----------------------------------------------------------------------------
# Camera I/O
# ----------------------------------------------------------------------------
def _strip_ts_header(raw: bytes):
    """realsense_publisher prepends an 8-byte little-endian double timestamp."""
    if raw is None or len(raw) < 8:
        return None, None
    ts = struct.unpack("<d", raw[:8])[0]
    return ts, raw[8:]


def read_color(r: redis.Redis):
    ts, payload = _strip_ts_header(r.get(REDIS_KEYS.rs_color))
    if payload is None:
        return None, None
    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    return img, ts


def read_depth(r: redis.Redis):
    ts, payload = _strip_ts_header(r.get(REDIS_KEYS.rs_depth))
    if payload is None:
        return None, None
    depth = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)  # uint16
    return depth, ts


def read_intrinsics(r: redis.Redis):
    raw = r.get(REDIS_KEYS.rs_intrinsics)
    if raw is None:
        return None
    return json.loads(raw)


def fresh_color(r: redis.Redis, retries: int = 30, delay: float = 0.05):
    """Pull a color frame, retrying briefly so a just-moved camera settles."""
    for _ in range(retries):
        img, _ = read_color(r)
        if img is not None:
            return img
        time.sleep(delay)
    return None


# ----------------------------------------------------------------------------
# Perception
# ----------------------------------------------------------------------------
def load_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is required for fruit detection.\n"
              "  pip install ultralytics\n"
              "(or run with --image to test color/blue-strip handling only "
              "after stubbing detection).", file=sys.stderr)
        sys.exit(1)
    print(f"[vision] loading YOLO model '{model_name}' (downloads weights on first run)...")
    return YOLO(model_name)


def detect_fruit(model, bgr, roi, classes, conf_thresh: float):
    """Return the highest-confidence fruit detection whose centre is inside the
    ROI, as a dict {label, conf, box=(x1,y1,x2,y2), cx, cy}, or None."""
    rx, ry, rw, rh = roi
    results = model(bgr, verbose=False)
    best = None
    for res in results:
        names = res.names
        for b in res.boxes:
            conf = float(b.conf[0])
            if conf < conf_thresh:
                continue
            label = names[int(b.cls[0])]
            if label not in classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                continue
            if best is None or conf > best["conf"]:
                best = {"label": label, "conf": conf,
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "cx": int(round(cx)), "cy": int(round(cy))}
    return best


def detect_blue_strip(bgr, box, hsv_lo, hsv_hi):
    """Search the fruit's bounding box for the blue strip. Returns
    {found, cx, cy, area, contour} in full-image pixel coords (contour offset
    back into the full frame), or {found: False}."""
    x1, y1, x2, y2 = box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, bgr.shape[1]), min(y2, bgr.shape[0])
    if x2 <= x1 or y2 <= y1:
        return {"found": False}
    crop = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False}
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < MIN_BLUE_AREA_PX:
        return {"found": False}
    m = cv2.moments(biggest)
    if m["m00"] == 0:
        return {"found": False}
    cx = int(m["m10"] / m["m00"]) + x1
    cy = int(m["m01"] / m["m00"]) + y1
    contour = biggest + np.array([[x1, y1]])   # offset into full-frame coords
    return {"found": True, "cx": cx, "cy": cy, "area": float(area), "contour": contour}


def sample_depth_m(depth_u16, u, v, depth_scale, win=5):
    """Median of the nonzero depth (in metres) in a (2*win+1) window around (u,v).
    Returns None if no valid depth there."""
    if depth_u16 is None:
        return None
    h, w = depth_u16.shape[:2]
    u0, u1 = max(u - win, 0), min(u + win + 1, w)
    v0, v1 = max(v - win, 0), min(v + win + 1, h)
    patch = depth_u16[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


def deproject(u, v, z, intr):
    """Back-project pixel (u,v) at depth z (m) to a camera-frame 3D point (m)."""
    x = (u - intr["ppx"]) * z / intr["fx"]
    y = (v - intr["ppy"]) * z / intr["fy"]
    return np.array([x, y, z])


def cam_to_world(p_cam, flange_pos, flange_ori):
    """camera frame -> flange frame (T_FLANGE_CAM) -> world (live flange pose)."""
    R, t = T_FLANGE_CAM[:3, :3], T_FLANGE_CAM[:3, 3]
    p_flange = R @ p_cam + t
    return flange_pos + flange_ori @ p_flange


# ----------------------------------------------------------------------------
# Annotation
# ----------------------------------------------------------------------------
def annotate(bgr, roi, fruit, strip, target_world, cam_point):
    out = bgr.copy()
    rx, ry, rw, rh = roi
    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (160, 160, 160), 1)
    cv2.putText(out, "table ROI", (rx + 4, ry + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    if fruit is not None:
        x1, y1, x2, y2 = fruit["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"{fruit['label']} {fruit['conf']:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.circle(out, (fruit["cx"], fruit["cy"]), 4, (0, 220, 0), -1)

    if strip and strip.get("found"):
        cv2.drawContours(out, [strip["contour"]], -1, (255, 80, 0), 2)
        cv2.circle(out, (strip["cx"], strip["cy"]), 4, (255, 80, 0), -1)
        cv2.putText(out, "cut (blue strip)", (strip["cx"] + 6, strip["cy"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 2, cv2.LINE_AA)

    lines = []
    if cam_point is not None:
        lines.append(f"cam xyz (m): {cam_point[0]:+.3f} {cam_point[1]:+.3f} {cam_point[2]:+.3f}")
    if target_world is not None:
        lines.append(f"hover world (m): {target_world[0]:+.3f} "
                     f"{target_world[1]:+.3f} {target_world[2]:+.3f}")
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, out.shape[0] - 12 - 20 * (len(lines) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_overlay(out, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"fruit_scan_{stamp}.png")
    cv2.imwrite(path, out)
    return path


# ----------------------------------------------------------------------------
# Scan poses
# ----------------------------------------------------------------------------
def scan_grid(center, half_x, half_y, nx, ny, height):
    """Lawnmower grid of flange (x,y) at a fixed height, centred on `center`."""
    xs = [center[0]] if nx <= 1 else np.linspace(center[0] - half_x, center[0] + half_x, nx)
    ys = [center[1]] if ny <= 1 else np.linspace(center[1] - half_y, center[1] + half_y, ny)
    poses = []
    for i, x in enumerate(xs):
        row = ys if i % 2 == 0 else ys[::-1]   # serpentine
        for y in row:
            poses.append(np.array([x, y, height]))
    return poses


# ----------------------------------------------------------------------------
# Live preview
# ----------------------------------------------------------------------------
def run_live(r, model, classes, args):
    """Continuously read the latest Redis frame, run detection, and refresh a
    window with the YOLO box + blue strip drawn. Preview only — no robot motion.
    Press 'q' (or Esc) to quit."""
    intr = read_intrinsics(r)
    print("[live] preview — press 'q' in the window to quit.")
    win = "fruit_scan (live)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        bgr, ts = read_color(r)
        if bgr is None:
            time.sleep(0.05)
            continue
        roi = tuple(args.roi) if args.roi else (0, 0, bgr.shape[1], bgr.shape[0])
        fruit = detect_fruit(model, bgr, roi, classes, args.conf)
        strip = (detect_blue_strip(bgr, fruit["box"], BLUE_HSV_LO, BLUE_HSV_HI)
                 if fruit is not None else None)
        cam_point = None
        if fruit is not None and intr is not None:
            depth_u16, _ = read_depth(r)
            z = sample_depth_m(depth_u16, fruit["cx"], fruit["cy"],
                               intr.get("depth_scale", 0.001))
            if z is not None:
                cam_point = deproject(fruit["cx"], fruit["cy"], z, intr)
        out = annotate(bgr, roi, fruit, strip, None, cam_point)
        # frame-age HUD so you can see if the publisher stalls
        age = (time.time() - ts) if ts else None
        hud = "no fruit" if fruit is None else f"{fruit['label']} {fruit['conf']:.2f}"
        if age is not None:
            hud += f"   frame age {age:4.2f}s"
        cv2.putText(out, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # q or Esc
            break
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Redis host (default localhost)")
    p.add_argument("--port", type=int, default=6379, help="Redis port (default 6379)")
    p.add_argument("--model", default="yolov8n.pt", help="ultralytics YOLO weights")
    p.add_argument("--conf", type=float, default=0.35, help="min detection confidence")
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_FRUIT_CLASSES),
                   help="fruit class names to accept")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   default=None, help="table region in pixels; default = full frame")
    p.add_argument("--approach-height", type=float, default=APPROACH_HEIGHT,
                   help="metres to hover above the detected fruit")
    p.add_argument("--grid", nargs=2, type=int, metavar=("NX", "NY"), default=(3, 3),
                   help="scan grid columns x rows (default 3 3)")
    p.add_argument("--half-extent", nargs=2, type=float, metavar=("HX", "HY"),
                   default=(0.12, 0.12), help="half-width of scan sweep in x,y (m)")
    p.add_argument("--config-file", default=CONFIG_FILE_FOR_THIS_SCRIPT,
                   help="OpenSai XML this client should verify is running")
    p.add_argument("--no-robot", action="store_true",
                   help="skip ALL robot motion; detect on the current frame only "
                        "(local dev without the arm)")
    p.add_argument("--image", default=None,
                   help="read a still image instead of Redis (implies --no-robot, "
                        "no depth; uses --assume-depth)")
    p.add_argument("--assume-depth", type=float, default=0.30,
                   help="fallback fruit depth in metres when no depth frame is "
                        "available (--image mode, or missing depth pixel)")
    p.add_argument("--show", action="store_true",
                   help="pop a window with the single analyzed frame (blocks until a key)")
    p.add_argument("--live", action="store_true",
                   help="continuous preview: loop reading the latest Redis frame, "
                        "draw YOLO box + blue strip live, refresh until 'q'. No "
                        "robot motion (preview only).")
    p.add_argument("--save-dir", default="log_files/vision_logs",
                   help="where to write the annotated overlay")
    return p.parse_args()


def main():
    args = parse_args()
    classes = set(args.classes)
    use_robot = not (args.no_robot or args.image)

    r = redis.Redis(host=args.host, port=args.port)
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        print(f"Cannot reach Redis at {args.host}:{args.port}.", file=sys.stderr)
        sys.exit(1)

    model = load_model(args.model)

    # Live preview is a standalone mode: no robot, no XML check, no scan.
    if args.live:
        run_live(r, model, classes, args)
        return

    if use_robot and not verify_and_activate(r, args.config_file):
        sys.exit(1)

    intr = None if args.image else read_intrinsics(r)

    # --- SCAN -----------------------------------------------------------------
    fruit, scan_frame, hold_ori = None, None, None
    if args.image:
        scan_frame = cv2.imread(args.image)
        if scan_frame is None:
            print(f"Could not read image '{args.image}'.", file=sys.stderr)
            sys.exit(1)
        roi = tuple(args.roi) if args.roi else (0, 0, scan_frame.shape[1], scan_frame.shape[0])
        fruit = detect_fruit(model, scan_frame, roi, classes, args.conf)
    elif not use_robot:
        scan_frame = fresh_color(r)
        if scan_frame is None:
            print("No color frame on Redis — is realsense_publisher.py running?",
                  file=sys.stderr)
            sys.exit(1)
        roi = tuple(args.roi) if args.roi else (0, 0, scan_frame.shape[1], scan_frame.shape[0])
        fruit = detect_fruit(model, scan_frame, roi, classes, args.conf)
    else:
        # Anchor the sweep to the arm's live pose: same xy neighbourhood, same
        # height, same orientation it started in. Local search, no jump.
        start_pos, start_ori = read_actual_pose(r)
        if start_pos is None:
            print("Could not read the robot's current pose from Redis — is the "
                  "controller running? (Use --no-robot for a static scan.)",
                  file=sys.stderr)
            sys.exit(1)
        hold_ori = start_ori
        poses = scan_grid(start_pos, args.half_extent[0], args.half_extent[1],
                          args.grid[0], args.grid[1], start_pos[2])
        print(f"[scan] sweeping {len(poses)} poses around the start pose "
              f"({start_pos[0]:+.3f}, {start_pos[1]:+.3f}, {start_pos[2]:+.3f}), "
              f"holding current orientation...")
        for i, pose in enumerate(poses):
            move_to(r, pose, hold_ori, MOVE_TIME, f"scan pose {i + 1}/{len(poses)}")
            frame = fresh_color(r)
            if frame is None:
                print("       (no color frame yet)")
                continue
            scan_frame = frame
            roi = tuple(args.roi) if args.roi else (0, 0, frame.shape[1], frame.shape[0])
            cand = detect_fruit(model, frame, roi, classes, args.conf)
            if cand is not None:
                fruit = cand
                print(f"       found {fruit['label']} (conf {fruit['conf']:.2f}) "
                      f"at pixel ({fruit['cx']}, {fruit['cy']})")
                break
            print("       no fruit in view")

    roi = tuple(args.roi) if args.roi else (0, 0, scan_frame.shape[1], scan_frame.shape[0])

    if fruit is None:
        print("[result] no fruit found.")
        out = annotate(scan_frame, roi, None, None, None, None)
        path = save_overlay(out, args.save_dir)
        print(f"[result] overlay -> {path}")
        if args.show:
            cv2.imshow("fruit_scan", out); cv2.waitKey(0); cv2.destroyAllWindows()
        return

    # --- CUT ------------------------------------------------------------------
    strip = detect_blue_strip(scan_frame, fruit["box"], BLUE_HSV_LO, BLUE_HSV_HI)
    if strip.get("found"):
        print(f"[cut] blue strip found at pixel ({strip['cx']}, {strip['cy']}), "
              f"area {strip['area']:.0f}px")
    else:
        print("[cut] no blue strip detected on the fruit.")

    # --- LOCATE ---------------------------------------------------------------
    target_world, cam_point = None, None
    if args.image:
        print("[locate] --image mode: no depth/intrinsics, skipping 3D localisation.")
    elif intr is None:
        print("[locate] no intrinsics on Redis — cannot back-project. "
              "Is realsense_publisher.py publishing depth?")
    else:
        depth_u16, _ = read_depth(r)
        depth_scale = intr.get("depth_scale", 0.001)
        z = sample_depth_m(depth_u16, fruit["cx"], fruit["cy"], depth_scale)
        if z is None:
            z = args.assume_depth
            print(f"[locate] no valid depth at fruit centre; assuming {z:.3f} m.")
        cam_point = deproject(fruit["cx"], fruit["cy"], z, intr)
        print(f"[locate] fruit camera-frame xyz = "
              f"({cam_point[0]:+.3f}, {cam_point[1]:+.3f}, {cam_point[2]:+.3f}) m")
        if use_robot:
            flange_pos, flange_ori = read_actual_pose(r)
            if flange_pos is not None:
                fruit_world = cam_to_world(cam_point, flange_pos, flange_ori)
                target_world = fruit_world + np.array([0.0, 0.0, args.approach_height])
                print(f"[locate] fruit world xyz   = "
                      f"({fruit_world[0]:+.3f}, {fruit_world[1]:+.3f}, {fruit_world[2]:+.3f}) m")
                print(f"[locate] hover target xyz  = "
                      f"({target_world[0]:+.3f}, {target_world[1]:+.3f}, {target_world[2]:+.3f}) m"
                      f"  (note: depends on T_FLANGE_CAM calibration)")

    # --- overlay --------------------------------------------------------------
    out = annotate(scan_frame, roi, fruit, strip, target_world, cam_point)
    path = save_overlay(out, args.save_dir)
    print(f"[result] overlay -> {path}")
    if args.show:
        cv2.imshow("fruit_scan", out); cv2.waitKey(0); cv2.destroyAllWindows()

    # --- HOVER ----------------------------------------------------------------
    if use_robot and target_world is not None:
        move_to(r, target_world, hold_ori, MOVE_TIME, "hover above fruit")
        print("[done] robot positioned above the fruit.")
    elif use_robot:
        print("[done] scan complete, but no 3D target computed — robot left in place.")
    else:
        print("[done] scan complete (no-robot mode).")


if __name__ == "__main__":
    main()
