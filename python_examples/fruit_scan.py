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
ROBOT_NAME = "Titania"

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

# End-effector (tool) working point expressed in the FLANGE frame, metres. This
# is the point you want placed above the fruit when --align tool (default). It is
# the analogue of NEEDLE_TIP_OFFSET / JAWS_OFFSET in the stitch clients (~0.28 m
# out along the tool axis). PLACEHOLDER — calibrate to your passive end-effector;
# cross-check against suturebot_grav_stitch_oussama_push_grip.py's offsets. Note
# those are written in a world-aligned frame and converted with ORI.T; here the
# vector is already in the flange frame (p_world = flange_pos + flange_ori @ EE).
EE_OFFSET_FLANGE = np.array([0.0613, 0.0015, -0.2826])

# Fruit classes to accept (COCO names as used by ultralytics models).
DEFAULT_FRUIT_CLASSES = ("apple", "orange", "banana")

# Preset hues (OpenCV H, 0..179) to seed the colour tracker when YOLO acquisition
# is unreliable. "redorange" suits a reddish-orange apple.
COLOR_HUES = {"red": 0, "redorange": 8, "red-orange": 8, "orange": 15,
              "yellow": 27, "green": 60}

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
# Hand-eye calibration (auto-solve T_FLANGE_CAM from the arm watching one fruit)
# ----------------------------------------------------------------------------
# Model: the fruit is a single fixed world point P. At calibration pose i the arm
# reports flange pose (R_i, p_i) and the camera measures the fruit at camera-frame
# point q_i. They are tied by
#       p_i + R_i @ (R_fc @ q_i + t_fc) = P                         (for all i)
# where (R_fc, t_fc) = T_FLANGE_CAM (camera-in-flange) is what we solve for, along
# with the unknown fruit position P. Unknowns: R_fc (3) + t_fc (3) + P (3) = 9.
# Each pose gives 3 equations, so >=3 poses WITH ORIENTATION VARIETY are needed to
# separate R_fc from P (pure translation leaves the tilt unobservable). Solved by
# Levenberg-Marquardt with a numeric Jacobian (rotation parametrised by a Rodrigues
# vector via cv2.Rodrigues) — needs only numpy + cv2.
def _handeye_residuals(params, Rs, ps, qs):
    R_fc, _ = cv2.Rodrigues(params[0:3])
    t_fc = params[3:6]
    P = params[6:9]
    res = np.empty(3 * len(Rs))
    for i, (R_i, p_i, q_i) in enumerate(zip(Rs, ps, qs)):
        res[3 * i:3 * i + 3] = (p_i + R_i @ (R_fc @ q_i + t_fc)) - P
    return res


def _handeye_jacobian(params, Rs, ps, qs, eps=1e-6):
    r0 = _handeye_residuals(params, Rs, ps, qs)
    J = np.empty((r0.size, params.size))
    for k in range(params.size):
        dp = params.copy()
        dp[k] += eps
        J[:, k] = (_handeye_residuals(dp, Rs, ps, qs) - r0) / eps
    return J, r0


def solve_hand_eye(records, iters=300):
    """records: list of (R_flange 3x3, p_flange 3, p_cam 3).
    Returns (R_fc 3x3, t_fc 3, P 3, rms_metres)."""
    Rs = [np.asarray(r[0], float) for r in records]
    ps = [np.asarray(r[1], float) for r in records]
    qs = [np.asarray(r[2], float) for r in records]
    # init: identity rotation, a small +z offset, P from the mean prediction.
    rvec = np.zeros(3)
    t_fc = np.array([0.0, 0.0, 0.05])
    R0, _ = cv2.Rodrigues(rvec)
    P = np.mean([p + R @ (R0 @ q + t_fc) for R, p, q in zip(Rs, ps, qs)], axis=0)
    x = np.concatenate([rvec, t_fc, P])
    lam = 1e-3
    cost = float(_handeye_residuals(x, Rs, ps, qs) @ _handeye_residuals(x, Rs, ps, qs))
    for _ in range(iters):
        J, r = _handeye_jacobian(x, Rs, ps, qs)
        A = J.T @ J + lam * np.eye(x.size)
        g = J.T @ r
        try:
            dx = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            break
        x_new = x + dx
        new_cost = float(_handeye_residuals(x_new, Rs, ps, qs) @
                         _handeye_residuals(x_new, Rs, ps, qs))
        if new_cost < cost:                  # accept the step
            improved = cost - new_cost
            x, cost = x_new, new_cost
            lam = max(lam * 0.5, 1e-12)
            if improved < 1e-15:             # converged
                break
        else:                                # reject: damp harder and retry
            lam = min(lam * 3.0, 1e8)
            if lam >= 1e8:
                break
    R_fc, _ = cv2.Rodrigues(x[0:3])
    res = _handeye_residuals(x, Rs, ps, qs).reshape(-1, 3)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    return R_fc, x[3:6], x[6:9], rms


def _orientation_spread(records):
    """Rough measure (radians) of how much the flange orientation varied across
    calibration poses — near 0 means pure translation (tilt unobservable)."""
    Rs = [np.asarray(r[0], float) for r in records]
    R_mean = Rs[0]
    angs = []
    for R in Rs:
        rvec, _ = cv2.Rodrigues(R @ R_mean.T)
        angs.append(float(np.linalg.norm(rvec)))
    return max(angs) if angs else 0.0


def selftest_calibration():
    """Validate the solver locally with synthetic noisy data (no robot/camera)."""
    rng = np.random.default_rng(0)
    # Ground-truth camera-in-flange: ~22 deg down-tilt about flange X, jut +4cm
    # forward / +5cm down.
    true_rvec = np.array([np.deg2rad(22.0), 0.0, 0.0])
    R_true, _ = cv2.Rodrigues(true_rvec)
    t_true = np.array([0.00, 0.04, 0.05])
    P_true = np.array([0.55, 0.00, 0.05])         # fruit world position

    records = []
    for _ in range(12):
        # random-ish flange orientation near camera-down with tilts
        rv = np.array([np.deg2rad(180.0), 0.0, 0.0]) + rng.normal(0, 0.20, 3)
        R_i, _ = cv2.Rodrigues(rv)
        p_i = np.array([0.5, 0.0, 0.45]) + rng.normal(0, 0.03, 3)
        # exact camera-frame measurement of the fruit, then add pixel/depth noise
        q_i = R_true.T @ (R_i.T @ (P_true - p_i) - t_true)
        q_i = q_i + rng.normal(0, 0.002, 3)        # ~2mm measurement noise
        records.append((R_i, p_i, q_i))

    R_fc, t_fc, P, rms = solve_hand_eye(records)
    rvec_est, _ = cv2.Rodrigues(R_fc)
    ang_err = np.rad2deg(np.linalg.norm(
        cv2.Rodrigues(R_fc @ R_true.T)[0]))
    print("=== hand-eye solver self-test (synthetic) ===")
    print(f"orientation spread used: {np.rad2deg(_orientation_spread(records)):.1f} deg")
    print(f"true  t_fc = {np.round(t_true, 4)}   est t_fc = {np.round(t_fc, 4)}")
    print(f"true  tilt = 22.0 deg about X   est rvec(deg) = {np.round(np.rad2deg(rvec_est), 2)}")
    print(f"rotation error  = {ang_err:.2f} deg")
    print(f"t_fc error      = {np.linalg.norm(t_fc - t_true)*1000:.1f} mm")
    print(f"fruit P error   = {np.linalg.norm(P - P_true)*1000:.1f} mm")
    print(f"residual rms    = {rms*1000:.2f} mm")
    ok = ang_err < 2.0 and np.linalg.norm(t_fc - t_true) < 0.01
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------------
# Calibration persistence (solved T_FLANGE_CAM auto-loads on every run)
# ----------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(_REPO_ROOT, "calibration", "t_flange_cam.json")


def save_calibration(T, rms, n):
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump({"T_FLANGE_CAM": np.asarray(T).tolist(),
                   "rms_mm": rms * 1000.0, "n_poses": n}, f, indent=2)
    return CALIB_PATH


def load_calibration():
    if not os.path.exists(CALIB_PATH):
        return None
    try:
        with open(CALIB_PATH) as f:
            return np.array(json.load(f)["T_FLANGE_CAM"], float)
    except Exception as e:
        print(f"[calib] could not read {CALIB_PATH}: {e}")
        return None


# ----------------------------------------------------------------------------
# Detection + visual-servo centering (used by the scan run and calibration)
# ----------------------------------------------------------------------------
class FruitTracker:
    """Acquire the fruit with YOLO, then lock onto it by colour. YOLO is reliable
    from a distance but drops out when the camera is close/overhead (the fruit
    fills the frame at an odd scale) — exactly the poses calibration/centering
    need. Once YOLO sees it once, we learn its hue and track the coloured blob,
    falling back to YOLO whenever it succeeds again."""

    def __init__(self):
        self.h_center = None     # learned hue (OpenCV 0..179)
        self.last = None         # last (cx, cy) to disambiguate blobs

    def _learn(self, bgr, box):
        x1, y1, x2, y2 = (max(int(v), 0) for v in box)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0].reshape(-1).astype(int)
        s = hsv[..., 1].reshape(-1)
        v = hsv[..., 2].reshape(-1)
        m = (s > 60) & (v > 50) & ~((h >= 100) & (h <= 130))   # drop dull + blue strip
        if int(m.sum()) < 30:
            return
        ang = np.deg2rad(h[m] * 2.0)                            # circular mean of hue
        mean = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
        self.h_center = int((np.rad2deg(mean) % 360) / 2.0)

    def _mask(self, bgr, d=15, smin=60, vmin=50):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        c = self.h_center
        if c - d < 0 or c + d > 179:                            # red wraps around 0/179
            m1 = cv2.inRange(hsv, np.array([0, smin, vmin]),
                             np.array([(c + d) % 180, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([(c - d) % 180, smin, vmin]),
                             np.array([179, 255, 255]))
            return cv2.bitwise_or(m1, m2)
        return cv2.inRange(hsv, np.array([c - d, smin, vmin]),
                           np.array([c + d, 255, 255]))

    def _color_detect(self, bgr, roi):
        if self.h_center is None:
            return None
        mask = self._mask(bgr)
        rx, ry, rw, rh = roi
        roimask = np.zeros(mask.shape, np.uint8)
        roimask[ry:ry + rh, rx:rx + rw] = 255
        mask = cv2.bitwise_and(mask, roimask)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= 150]
        if not cnts:
            return None
        if self.last is not None:
            def cdist(c):
                M = cv2.moments(c)
                if M["m00"] == 0:
                    return 1e9
                return np.hypot(M["m10"] / M["m00"] - self.last[0],
                                M["m01"] / M["m00"] - self.last[1])
            best = min(cnts, key=cdist)
        else:
            best = max(cnts, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(best)
        return {"label": "fruit(color)", "conf": 0.0,
                "box": (x, y, x + w, y + h), "cx": cx, "cy": cy}

    def detect(self, model, bgr, classes, conf, roi):
        f = detect_fruit(model, bgr, roi, classes, conf)
        if f is not None:
            self._learn(bgr, f["box"])
            self.last = (f["cx"], f["cy"])
            return f
        f = self._color_detect(bgr, roi)
        if f is not None:
            self.last = (f["cx"], f["cy"])
        return f


def resolve_seed_hue(args):
    """Hue (OpenCV 0..179) to pre-seed the colour tracker, or None."""
    if args.fruit_hue is not None:
        return int(args.fruit_hue) % 180
    if args.fruit_color:
        return COLOR_HUES[args.fruit_color]
    return None


# Set True (by run_calibrate) to pop a diagnostic window at every detection.
_VIS = False
_VIS_WIN = "fruit_scan (calibrate)"


def _vis_calib(bgr, fruit, p_cam, note=""):
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
    if fruit is not None:
        x1, y1, x2, y2 = fruit["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.circle(out, (fruit["cx"], fruit["cy"]), 4, (0, 220, 0), -1)
    status = (f"{'FRUIT' if fruit is not None else 'no fruit'} | "
              f"{'depth OK' if p_cam is not None else 'NO DEPTH'}")
    cv2.putText(out, f"{note}  {status}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def grab_detection(r, model, classes, args, note="", tracker=None):
    """Read a fresh frame; return (fruit, p_cam, bgr). fruit/p_cam may be None.
    With a FruitTracker, acquire via YOLO then track by colour (robust close-up)."""
    bgr = fresh_color(r)
    if bgr is None:
        return None, None, None
    roi = tuple(args.roi) if args.roi else (0, 0, bgr.shape[1], bgr.shape[0])
    if tracker is not None:
        fruit = tracker.detect(model, bgr, classes, args.conf, roi)
    else:
        fruit = detect_fruit(model, bgr, roi, classes, args.conf)
    p_cam = None
    if fruit is not None:
        intr = read_intrinsics(r)
        if intr is not None:
            depth_u16, _ = read_depth(r)
            z = sample_depth_m(depth_u16, fruit["cx"], fruit["cy"],
                               intr.get("depth_scale", 0.001), win=9)
            if z is not None:
                p_cam = deproject(fruit["cx"], fruit["cy"], z, intr)
    if _VIS:
        try:
            cv2.imshow(_VIS_WIN, _vis_calib(bgr, fruit, p_cam, note))
            cv2.waitKey(1)
        except cv2.error:
            pass
    return fruit, p_cam, bgr


def _rot_about(axis, deg):
    v = np.asarray(axis, float)
    v = v / np.linalg.norm(v) * np.deg2rad(deg)
    R, _ = cv2.Rodrigues(v)
    return R


def estimate_image_jacobian(r, model, classes, args, ori, base_pos,
                            probe=0.02, settle=None, tracker=None):
    """Probe +probe m in world x then y, measure how the fruit pixel moves.
    Returns 2x2 J (d_pixel / d_world_xy), or None. Auto-captures the camera
    mounting sign so the servo never needs hand-set directions."""
    settle = settle if settle is not None else MOVE_TIME
    move_to(r, base_pos, ori, settle, "probe base")
    f0, _, _ = grab_detection(r, model, classes, args, tracker=tracker)
    if f0 is None:
        return None
    px0 = np.array([f0["cx"], f0["cy"]], float)
    cols = []
    for axis, name in ((0, "x"), (1, "y")):
        pp = base_pos.copy(); pp[axis] += probe
        move_to(r, pp, ori, settle, f"probe +{name}")
        fi, _, _ = grab_detection(r, model, classes, args, tracker=tracker)
        if fi is None:
            move_to(r, base_pos, ori, settle, "probe back")
            return None
        cols.append((np.array([fi["cx"], fi["cy"]], float) - px0) / probe)
    move_to(r, base_pos, ori, settle, "probe back")
    return np.column_stack(cols)


def center_on_fruit(r, model, classes, args, ori, pos, J,
                    tol_px=18, gain=0.5, max_step=0.03, max_iter=15, settle=None,
                    tracker=None):
    """Translate in world XY until the fruit sits at the image centre."""
    settle = settle if settle is not None else MOVE_TIME
    try:
        Jinv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        print("[center] image Jacobian singular; skipping centering.")
        return pos
    pos = pos.copy()
    for it in range(max_iter):
        f, _, bgr = grab_detection(r, model, classes, args, tracker=tracker)
        if f is None:
            print("[center] lost the fruit; stopping.")
            return pos
        h, w = bgr.shape[:2]
        err = np.array([f["cx"] - w / 2.0, f["cy"] - h / 2.0])
        if np.linalg.norm(err) < tol_px:
            print(f"[center] centred (err {np.linalg.norm(err):.0f}px) in {it} steps.")
            return pos
        d_xy = -gain * (Jinv @ err)
        n = np.linalg.norm(d_xy)
        if n > max_step:
            d_xy *= max_step / n
        pos[0] += d_xy[0]; pos[1] += d_xy[1]
        move_to(r, pos, ori, settle,
                f"center step {it + 1} (err {np.linalg.norm(err):.0f}px)")
    print("[center] hit max iterations (still off-centre).")
    return pos


# ----------------------------------------------------------------------------
# Auto hand-eye calibration routine
# ----------------------------------------------------------------------------
def run_calibrate(r, model, classes, args):
    global _VIS
    _VIS = True   # show a diagnostic window at every detection during calibration
    if not verify_and_activate(r, args.config_file):
        return
    if read_intrinsics(r) is None:
        print("[calib] no intrinsics on Redis — start realsense_publisher.py.")
        return
    start_pos, start_ori = read_actual_pose(r)
    if start_pos is None:
        print("[calib] cannot read robot pose; is the controller running?")
        return
    # Depth sanity check up front so 'no depth' failures are obvious immediately.
    if read_depth(r) is None:
        print("[calib] WARNING: no depth frame on Redis (key "
              f"'{REDIS_KEYS.rs_depth}'). Run the publisher WITHOUT --no-depth — "
              "calibration needs depth at the fruit pixel.")

    # Acquire with YOLO from a distance, then track by colour (robust when the
    # camera gets close/overhead and YOLO can no longer recognise the fruit).
    tracker = FruitTracker()
    seed_hue = resolve_seed_hue(args)
    if seed_hue is not None:
        tracker.h_center = seed_hue
        print(f"[calib] seeded colour tracker at hue {seed_hue} "
              "(colour tracking active from the first frame).")

    # 1) coarse: sweep to bring the fruit into view (YOLO learns its colour here).
    poses = scan_grid(start_pos, args.half_extent[0], args.half_extent[1],
                      args.grid[0], args.grid[1], start_pos[2])
    print(f"[calib] sweeping {len(poses)} poses to find the fruit...")
    found = False
    for i, pose in enumerate(poses):
        move_to(r, pose, start_ori, MOVE_TIME, f"find {i + 1}/{len(poses)}")
        f, _, _ = grab_detection(r, model, classes, args, note="find", tracker=tracker)
        if f is not None:
            found = True
            break
    if not found:
        print("[calib] never saw the fruit during the sweep.")
        return
    if tracker.h_center is None:
        print("[calib] WARNING: colour not learned (only a colour-fallback hit?). "
              "Start with the fruit clearly in view so YOLO acquires it once.")

    # 2) centre it so tilts keep it in frame.
    base_pos, _ = read_actual_pose(r)
    J = estimate_image_jacobian(r, model, classes, args, start_ori, base_pos,
                                tracker=tracker)
    if J is None:
        print("[calib] could not estimate image Jacobian (lost fruit).")
        return
    base_pos = center_on_fruit(r, model, classes, args, start_ori, base_pos, J,
                               tracker=tracker)

    # 3) collect (R_flange, p_flange, p_cam) over a spread of tilts; re-centre
    #    after each tilt so the fruit stays visible (and adds position variety).
    tilts = [(None, 0.0),
             ([1, 0, 0], +args.calib_tilt), ([1, 0, 0], -args.calib_tilt),
             ([0, 1, 0], +args.calib_tilt), ([0, 1, 0], -args.calib_tilt),
             ([1, 1, 0], +args.calib_tilt * 0.7), ([1, -1, 0], +args.calib_tilt * 0.7),
             ([1, 1, 0], -args.calib_tilt * 0.7), ([1, -1, 0], -args.calib_tilt * 0.7)]
    records = []
    for k, (axis, deg) in enumerate(tilts):
        ori = start_ori if axis is None else _rot_about(axis, deg) @ start_ori
        move_to(r, base_pos, ori, MOVE_TIME, f"tilt {k + 1}/{len(tilts)} ({deg:+.0f}deg)")
        center_on_fruit(r, model, classes, args, ori, base_pos, J, tracker=tracker)
        f, p_cam, _ = grab_detection(r, model, classes, args, note=f"tilt {k + 1}",
                                     tracker=tracker)
        if f is None:
            print(f"       tilt {k + 1}: NO FRUIT in view after tilt — skipping "
                  "(tilt may be too large / fruit left frame).")
            continue
        if p_cam is None:
            print(f"       tilt {k + 1}: fruit seen at pixel ({f['cx']},{f['cy']}) "
                  "but NO DEPTH there — skipping (depth hole / out of range).")
            continue
        cur_pos, cur_ori = read_actual_pose(r)
        records.append((cur_ori, cur_pos, p_cam))
        print(f"       tilt {k + 1}: recorded (cam xyz "
              f"{p_cam[0]:+.3f},{p_cam[1]:+.3f},{p_cam[2]:+.3f})")

    spread = np.rad2deg(_orientation_spread(records)) if records else 0.0
    print(f"[calib] collected {len(records)} poses, orientation spread {spread:.1f} deg")
    if len(records) < 4:
        print("[calib] too few good poses to solve (need >=4). Aborting.")
        return
    if spread < 5.0:
        print("[calib] WARNING: little orientation variety — tilt result may be poor.")

    # 4) solve + save.
    R_fc, t_fc, P, rms = solve_hand_eye(records)
    T = np.eye(4); T[:3, :3] = R_fc; T[:3, 3] = t_fc
    rvec, _ = cv2.Rodrigues(R_fc)
    print(f"[calib] solved T_FLANGE_CAM: t_fc={np.round(t_fc, 4)} m, "
          f"tilt={np.round(np.rad2deg(rvec), 1)} deg, residual rms={rms * 1000:.1f} mm")
    if rms > 0.02:
        print("[calib] WARNING: residual > 20 mm — detection/depth may be noisy; "
              "consider re-running.")
    path = save_calibration(T, rms, len(records))
    print(f"[calib] saved -> {path} (auto-loads on future runs).")


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
    p.add_argument("--fruit-color", choices=list(COLOR_HUES.keys()), default=None,
                   help="seed the colour tracker with this fruit colour so it "
                        "tracks by colour from the start (no YOLO acquisition "
                        "needed). Use 'redorange' for a reddish-orange apple.")
    p.add_argument("--fruit-hue", type=int, default=None,
                   help="seed the colour tracker with an explicit OpenCV hue "
                        "(0..179); overrides --fruit-color")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   default=None, help="table region in pixels; default = full frame")
    p.add_argument("--approach-height", type=float, default=APPROACH_HEIGHT,
                   help="metres to hover above the detected fruit")
    p.add_argument("--align", choices=["tool", "flange", "camera"], default="tool",
                   help="which point to place above the fruit: 'tool' = the "
                        "end-effector tip (EE_OFFSET_FLANGE, default), 'camera' = "
                        "the camera, 'flange' = the flange origin")
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
    # Calibration / centering -----------------------------------------------------
    p.add_argument("--calibrate", action="store_true",
                   help="auto-solve the camera->wrist transform (T_FLANGE_CAM): "
                        "find the fruit, view it from several tilts, solve, and "
                        "save to calibration/t_flange_cam.json (auto-loaded after).")
    p.add_argument("--calib-tilt", type=float, default=12.0,
                   help="max wrist tilt in degrees used during --calibrate (default 12)")
    p.add_argument("--center", action="store_true",
                   help="before placing the tool, visual-servo the fruit to the "
                        "image centre (more accurate, needs a calibrated camera)")
    p.add_argument("--selftest-calib", action="store_true",
                   help="run the hand-eye solver on synthetic data and exit "
                        "(no robot/camera needed)")
    return p.parse_args()


def main():
    args = parse_args()
    # Self-test needs neither robot nor camera — run and exit.
    if args.selftest_calib:
        selftest_calibration()
        return

    classes = set(args.classes)
    use_robot = not (args.no_robot or args.image)

    r = redis.Redis(host=args.host, port=args.port)
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        print(f"Cannot reach Redis at {args.host}:{args.port}.", file=sys.stderr)
        sys.exit(1)

    # Auto-load a saved hand-eye calibration over the placeholder, if present.
    global T_FLANGE_CAM
    loaded = load_calibration()
    if loaded is not None:
        T_FLANGE_CAM = loaded
        print(f"[calib] loaded T_FLANGE_CAM from {CALIB_PATH}")
    elif not args.image:
        print("[calib] no saved calibration — using placeholder T_FLANGE_CAM "
              "(run --calibrate for accuracy).")

    model = load_model(args.model)

    # Calibration is a standalone mode (moves the robot, solves, saves, exits).
    if args.calibrate:
        run_calibrate(r, model, classes, args)
        return

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
        # One tracker for the whole run: YOLO acquires, colour holds. Seed it if
        # a --fruit-color/--fruit-hue was given (then it works without any YOLO hit).
        tracker = FruitTracker()
        seed_hue = resolve_seed_hue(args)
        if seed_hue is not None:
            tracker.h_center = seed_hue
            print(f"[scan] colour tracker seeded at hue {seed_hue}.")
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
            cand = tracker.detect(model, frame, classes, args.conf, roi)
            if cand is not None:
                fruit = cand
                print(f"       found {fruit['label']} (conf {fruit['conf']:.2f}) "
                      f"at pixel ({fruit['cx']}, {fruit['cy']})")
                break
            print("       no fruit in view")

        # Optional: visual-servo the fruit to the image centre before locating,
        # so the depth reading is on-axis and the placement is most accurate.
        if fruit is not None and args.center:
            cur_pos, _ = read_actual_pose(r)
            J = estimate_image_jacobian(r, model, classes, args, hold_ori, cur_pos,
                                        tracker=tracker)
            if J is not None:
                center_on_fruit(r, model, classes, args, hold_ori, cur_pos, J,
                                tracker=tracker)
                f2, _, frame2 = grab_detection(r, model, classes, args, tracker=tracker)
                if f2 is not None:
                    fruit, scan_frame = f2, frame2
                    print(f"       centred on {fruit['label']} at pixel "
                          f"({fruit['cx']}, {fruit['cy']})")
            else:
                print("       [center] Jacobian probe failed; skipping centering.")

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
                # Where we want the CHOSEN point (tool / camera / flange) to be.
                desired = fruit_world + np.array([0.0, 0.0, args.approach_height])
                # That point's offset in the flange frame:
                #   tool   -> EE_OFFSET_FLANGE
                #   camera -> camera origin in flange = T_FLANGE_CAM translation
                #   flange -> zero
                if args.align == "tool":
                    offset_in_flange = EE_OFFSET_FLANGE
                elif args.align == "camera":
                    offset_in_flange = T_FLANGE_CAM[:3, 3]
                else:  # flange
                    offset_in_flange = np.zeros(3)
                # Command the flange so the chosen point lands on `desired`:
                #   desired = flange_goal + flange_ori @ offset_in_flange
                target_world = desired - flange_ori @ offset_in_flange
                print(f"[locate] fruit world xyz   = "
                      f"({fruit_world[0]:+.3f}, {fruit_world[1]:+.3f}, {fruit_world[2]:+.3f}) m")
                print(f"[locate] aligning '{args.align}' {args.approach_height:.3f} m "
                      f"above the fruit -> flange goal "
                      f"({target_world[0]:+.3f}, {target_world[1]:+.3f}, {target_world[2]:+.3f}) m"
                      f"  (depends on T_FLANGE_CAM"
                      f"{' + EE_OFFSET_FLANGE' if args.align == 'tool' else ''} calibration)")

    # --- overlay --------------------------------------------------------------
    out = annotate(scan_frame, roi, fruit, strip, target_world, cam_point)
    path = save_overlay(out, args.save_dir)
    print(f"[result] overlay -> {path}")
    if args.show:
        cv2.imshow("fruit_scan", out); cv2.waitKey(0); cv2.destroyAllWindows()

    # --- HOVER ----------------------------------------------------------------
    if use_robot and target_world is not None:
        move_to(r, target_world, hold_ori, MOVE_TIME, "hover above fruit")
        # Verify where the chosen point actually ended up vs the fruit.
        act_pos, act_ori = read_actual_pose(r)
        if act_pos is not None:
            offsets = {"tool": EE_OFFSET_FLANGE,
                       "camera": T_FLANGE_CAM[:3, 3],
                       "flange": np.zeros(3)}
            pt = act_pos + act_ori @ offsets[args.align]
            print(f"[verify] {args.align} now at "
                  f"({pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f}) m; fruit at "
                  f"({fruit_world[0]:+.3f}, {fruit_world[1]:+.3f}, {fruit_world[2]:+.3f}) m "
                  f"-> xy offset ({pt[0] - fruit_world[0]:+.3f}, {pt[1] - fruit_world[1]:+.3f}) m, "
                  f"height +{pt[2] - fruit_world[2]:.3f} m")
        print(f"[done] robot positioned with the {args.align} above the fruit.")
    elif use_robot:
        print("[done] scan complete, but no 3D target computed — robot left in place.")
    else:
        print("[done] scan complete (no-robot mode).")


if __name__ == "__main__":
    main()
