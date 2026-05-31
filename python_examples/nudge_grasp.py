"""Interactive grasp nudger: dial in how the needle sits in the gripper, live.

Pins the needle in the gripper and holds the arm at a fixed ready pose; you
rotate/shift the needle's seating with single keypresses and watch it move in
the sim window. When it looks right, print the values and paste them into
init_needle.py.

MUST be run in a REAL terminal (it reads raw keypresses). With suture_pad.xml
already running:
    python3 python_examples/nudge_grasp.py

Keys:
    q / e   roll   -/+      (about the needle's local X)
    w / s   pitch  +/-      (about local Y)
    a / d   yaw    -/+      (about local Z)
    r / f   deeper / shallower in the jaws  (grasp +Z)
    t / g   lateral X  -/+
    y / h   lateral Y  -/+
    [ / ]   step size  smaller / larger
    p       print current GRASP_POS / GRASP_R
    x       print and quit
"""
import json
import select
import sys
import termios
import time
import tty

import numpy as np
import redis

ROBOT = "Rizon4s"
ORI = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
READY = np.array([0.45, -0.30, 0.42])

# dialed in: gripped 1/3 from the swage, needle ~perpendicular to the driver
# with a slight slant. (Nudger 0 resets here.)
GRASP_POS = np.array([-0.00072, 0.00438, 0.28769])
GRASP_R = np.array([[0.0, -1.0, 0.0],
                    [-0.75471, 0.0, 0.65606],
                    [-0.65606, 0.0, -0.75471]])

K = {
    "gp": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_position",
    "go": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_orientation",
    "cp": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::current_position",
    "co": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::current_orientation",
    "ac": f"opensai::controllers::{ROBOT}::active_controller_name",
    "npose": "opensai::commands::Needle::pose",
    "nkin":  "opensai::commands::Needle::kinematic",
    "config": "::sai-interfaces-webui::config_file_name",
}


NEEDLE_R = 0.011                       # arc radius (match gen_needle_urdf.py)
ARC = np.radians(135.0)                # 3/8 circle
GRIP_FRAC = 1.0 / 3.0                  # gripped 1/3 down the arc from the swage (blue end)
GRIP_ANGLE = ARC * GRIP_FRAC
GRIP_LOCAL = np.array([NEEDLE_R * np.cos(GRIP_ANGLE), NEEDLE_R * np.sin(GRIP_ANGLE), 0])

def Rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
def Ry(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rotate_about_grip(R, pos, Rd):
    """Rotate the needle by Rd (flange-frame axis) about its GRIP point (1/3
    down from the swage), keeping that point fixed in the jaws. Returns (R, pos)."""
    grip = R @ GRIP_LOCAL + pos      # grip point in flange frame (held fixed)
    R2 = Rd @ R
    return R2, grip - R2 @ GRIP_LOCAL


def flange_T(r):
    rp, ro = r.get(K["cp"]), r.get(K["co"])
    if not rp or not ro:
        return None
    sp, so = rp.decode(), ro.decode()
    if "nan" in sp.lower() or "nan" in so.lower():   # diverged robot state
        return None
    try:
        p = np.array(json.loads(sp)); Rm = np.array(json.loads(so))
    except (ValueError, TypeError):
        return None
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = p
    return T


def dump(pos, R):
    rows = ",\n                    ".join("[" + ", ".join(f"{v: .5f}" for v in row) + "]" for row in R)
    print("\r\n--- paste into init_needle.py ---")
    print(f"GRASP_POS = np.array([{pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}])")
    print(f"GRASP_R = np.array([{rows}])\r")


def main():
    r = redis.Redis()
    cfg = r.get(K["config"])
    if cfg is None or cfg.decode() != "suture_pad.xml":
        print(f"Expected OpenSai running suture_pad.xml, got {cfg!r}."); return
    if not sys.stdin.isatty():
        print("Run this in a real terminal (needs raw keypresses)."); return

    pos = GRASP_POS.copy().astype(float)
    R = GRASP_R.copy().astype(float)
    step = np.radians(12.0)
    dpos = 0.005

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print("nudging needle grasp -- watch the sim. keys: qe/ws/ad rotate, rf depth, tg/yh lateral, [] step, p print, x quit")
    try:
        tty.setcbreak(fd)
        while True:
            r.set(K["ac"], "cartesian_controller")
            r.set(K["go"], json.dumps(ORI.tolist()))
            r.set(K["gp"], json.dumps(READY.tolist()))
            Tf = flange_T(r)
            if Tf is None:
                sys.stdout.write("\rwaiting for valid robot state (NaN/empty) -- relaunch the sim if this persists   ")
                sys.stdout.flush()
            else:
                G = np.eye(4); G[:3, :3] = R; G[:3, 3] = pos
                r.set(K["nkin"], "1")
                r.set(K["npose"], json.dumps((Tf @ G).tolist()))
                sys.stdout.write(f"\rpos={pos.round(3).tolist()}  step={np.degrees(step):.1f}deg/{dpos*1000:.0f}mm   ")
                sys.stdout.flush()

            if select.select([sys.stdin], [], [], 0.02)[0]:
                c = sys.stdin.read(1)
                if   c == 'q': R, pos = rotate_about_grip(R, pos, Rx(-step))
                elif c == 'e': R, pos = rotate_about_grip(R, pos, Rx(step))
                elif c == 'w': R, pos = rotate_about_grip(R, pos, Ry(step))
                elif c == 's': R, pos = rotate_about_grip(R, pos, Ry(-step))
                elif c == 'a': R, pos = rotate_about_grip(R, pos, Rz(-step))
                elif c == 'd': R, pos = rotate_about_grip(R, pos, Rz(step))
                elif c == 'r': pos[2] += dpos
                elif c == 'f': pos[2] -= dpos
                elif c == 't': pos[0] -= dpos
                elif c == 'g': pos[0] += dpos
                elif c == 'y': pos[1] -= dpos
                elif c == 'h': pos[1] += dpos
                elif c == '0': R = GRASP_R.copy(); pos = GRASP_POS.copy()   # reset to default
                elif c == '[': step /= 1.5; dpos /= 1.5
                elif c == ']': step *= 1.5; dpos *= 1.5
                elif c == 'p': dump(pos, R)
                elif c in ('x', '\x03'): dump(pos, R); break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


if __name__ == "__main__":
    main()
