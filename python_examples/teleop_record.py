"""Teleoperate the arm + gripper and RECORD the trajectory for later smoothing
and playback. Teach the stitch by hand instead of solving it analytically -- a
human-guided path naturally avoids the wrist singularities the solver hit.

Two modes:
  (default, sim)   jog the end-effector with the keyboard; the cartesian
                   controller follows. Records the resulting joint trajectory.
  --record-only    don't jog -- just record while you move the arm yourself
                   (Flexiv free-drive / hand-guiding on the real robot). g/b
                   still command + log the gripper.

Run in a REAL terminal (raw keypresses). Keys:
  w/s  x +/-      a/d  y -/+      r/f  z +/-      (translate, world frame)
  u/o  roll -/+   i/k  pitch +/-  j/l  yaw -/+    (rotate, world frame)
  g    close gripper (grip)       b   open gripper (release)
  [ ]  jog step smaller / larger
  x    stop and save

Pauses (while you open/close or reposition) are captured with timestamps;
playback_smooth.py removes them. Saves to log_files/demos/demo_<n>.npz.

    cd ../OpenSai && sh scripts/launch.sh config_folder/xml_config_files/suture_teleop.xml
    python3 python_examples/teleop_record.py
"""
import argparse
import glob
import json
import os
import select
import sys
import termios
import time
import tty

import numpy as np
import redis

ROBOT = "Rizon4s"           # "Titania" on the real robot
N_JOINTS = 7

# needle grasp (sim): when the gripper closes, pin the needle to the flange here
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
    "q":  f"opensai::sensors::{ROBOT}::joint_positions",
    "grip_mode": f"opensai::commands::{ROBOT}::gripper::mode",      # consumed by the real driver
    "npose": "opensai::commands::Needle::pose",                    # sim needle pin
    "nkin":  "opensai::commands::Needle::kinematic",
}
DEMO_DIR = "log_files/demos"


def Rx(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
def Ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])


def jget(r, key):
    v = r.get(key)
    if not v:
        return None
    s = v.decode()
    if "nan" in s.lower():
        return None
    try:
        return np.array(json.loads(s))
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-only", action="store_true",
                    help="don't jog; just record while you hand-guide (real robot free-drive)")
    args = ap.parse_args()
    jog = not args.record_only

    r = redis.Redis()
    if not sys.stdin.isatty():
        print("Run in a real terminal (raw keypresses)."); return
    q0 = jget(r, K["q"])
    if q0 is None or len(q0) < N_JOINTS:
        print("No valid joint_positions yet -- is the sim/robot up?"); return

    # seed jog goal from the current pose
    pos = jget(r, K["cp"]); R = jget(r, K["co"])
    if jog and (pos is None or R is None):
        print("No valid flange pose yet."); return
    dp, dth = 0.005, np.radians(4)
    GRASP = np.eye(4); GRASP[:3,:3] = GRASP_R; GRASP[:3,3] = GRASP_POS

    times, qs, gripper = [], [], []
    grip_closed = False
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    print(f"[{'JOG' if jog else 'RECORD-ONLY'}] recording... g/b gripper, x to save. "
          + ("w/s a/d r/f translate, u/o i/k j/l rotate, [] step." if jog else "move the arm by hand."))
    t0 = time.time()
    try:
        tty.setcbreak(fd)
        while True:
            if jog:
                r.set(K["ac"], "cartesian_controller")
                r.set(K["gp"], json.dumps(np.asarray(pos, float).tolist()))
                r.set(K["go"], json.dumps(np.asarray(R, float).tolist()))
            if grip_closed:                      # hold the needle pinned to the hand (sim)
                Tf = np.eye(4)
                cp = jget(r, K["cp"]); co = jget(r, K["co"])
                if cp is not None and co is not None:
                    Tf[:3,:3] = co; Tf[:3,3] = cp
                    r.set(K["npose"], json.dumps((Tf @ GRASP).tolist())); r.set(K["nkin"], "1")

            q = jget(r, K["q"])
            if q is not None:
                times.append(time.time() - t0); qs.append(q[:N_JOINTS].tolist()); gripper.append(1 if grip_closed else 0)
            sys.stdout.write(f"\r t={time.time()-t0:6.1f}s  samples={len(qs)}  grip={'CLOSED' if grip_closed else 'open'}   ")
            sys.stdout.flush()

            if select.select([sys.stdin], [], [], 0.02)[0]:
                c = sys.stdin.read(1)
                if jog and c in "wsadrfuoikjl[]":
                    if   c == 'w': pos[0] += dp
                    elif c == 's': pos[0] -= dp
                    elif c == 'a': pos[1] -= dp
                    elif c == 'd': pos[1] += dp
                    elif c == 'r': pos[2] += dp
                    elif c == 'f': pos[2] -= dp
                    elif c == 'u': R = Rx(-dth) @ R
                    elif c == 'o': R = Rx(dth) @ R
                    elif c == 'i': R = Ry(dth) @ R
                    elif c == 'k': R = Ry(-dth) @ R
                    elif c == 'j': R = Rz(-dth) @ R
                    elif c == 'l': R = Rz(dth) @ R
                    elif c == '[': dp /= 1.5; dth /= 1.5
                    elif c == ']': dp *= 1.5; dth *= 1.5
                elif c == 'g':
                    grip_closed = True; r.set(K["grip_mode"], "g")
                elif c == 'b':
                    grip_closed = False; r.set(K["grip_mode"], "o"); r.set(K["nkin"], "0")
                elif c == 'x':
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old); print()

    if len(qs) < 2:
        print("nothing recorded."); return
    os.makedirs(DEMO_DIR, exist_ok=True)
    n = len(glob.glob(os.path.join(DEMO_DIR, "demo_*.npz"))) + 1
    path = os.path.join(DEMO_DIR, f"demo_{n}.npz")
    np.savez(path, times=np.array(times), q=np.array(qs), gripper=np.array(gripper), robot=ROBOT)
    print(f"saved {len(qs)} samples over {times[-1]:.1f}s -> {path}")


if __name__ == "__main__":
    main()
