"""Smooth a recorded demo and replay it in JOINT space.

Replaying the recorded joint trajectory directly (no IK) avoids the wrist
singularities that flailed the cartesian solver -- the joints just follow the
path your hand already proved feasible.

Smoothing:
  1. trim pauses  -- drop stretches where the arm is ~stationary AND the
     gripper state is unchanged (the "while I fiddle" gaps).
  2. low-pass     -- moving-average filter each joint to remove jitter.
  3. resample     -- uniform time steps.
Gripper open/close transitions are preserved at the right trajectory points.

    python3 python_examples/playback_smooth.py log_files/demos/demo_1.npz
    python3 python_examples/playback_smooth.py log_files/demos/demo_1.npz --speed 0.5 --raw
"""
import argparse
import json
import time

import numpy as np
import redis

GRASP_POS = np.array([-0.00072, 0.00438, 0.28769])
GRASP_R = np.array([[0.0, -1.0, 0.0],
                    [-0.75471, 0.0, 0.65606],
                    [-0.65606, 0.0, -0.75471]])


def keys(robot):
    return {
        "ac":   f"opensai::controllers::{robot}::active_controller_name",
        "jgoal":f"opensai::controllers::{robot}::joint_controller::joint_task::goal_position",
        "cp":   f"opensai::controllers::{robot}::cartesian_controller::cartesian_task::current_position",
        "co":   f"opensai::controllers::{robot}::cartesian_controller::cartesian_task::current_orientation",
        "qsens":f"opensai::sensors::{robot}::joint_positions",
        "grip_mode": f"opensai::commands::{robot}::gripper::mode",
        "npose": "opensai::commands::Needle::pose",
        "nkin":  "opensai::commands::Needle::kinematic",
    }


def trim_pauses(t, q, g, move_eps=np.radians(0.5)):
    keep = [0]
    for i in range(1, len(q)):
        moved = np.max(np.abs(q[i] - q[keep[-1]])) > move_eps
        grip_changed = g[i] != g[keep[-1]]
        if moved or grip_changed or i == len(q) - 1:
            keep.append(i)
    return t[keep], q[keep], g[keep]


def moving_average(q, w=5):
    if w < 2 or len(q) < w:
        return q
    pad = np.pad(q, ((w//2, w//2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], np.ones(w)/w, mode="valid")[:len(q)] for j in range(q.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true", help="replay raw (no smoothing)")
    ap.add_argument("--robot", default=None, help='override; defaults to the robot the demo was recorded on')
    args = ap.parse_args()

    d = np.load(args.demo, allow_pickle=True)
    t, q, g = d["times"], d["q"], d["gripper"]
    robot = args.robot or (str(d["robot"]) if "robot" in d.files else "Rizon4s")
    K = keys(robot)
    print(f"loaded {len(q)} samples, {t[-1]:.1f}s, {q.shape[1]} joints, robot={robot}")

    if not args.raw:
        t, q, g = trim_pauses(t, q, g)
        q = moving_average(q, w=5)
        print(f"smoothed -> {len(q)} waypoints (pauses trimmed, filtered)")

    r = redis.Redis()
    GRASP = np.eye(4); GRASP[:3,:3] = GRASP_R; GRASP[:3,3] = GRASP_POS
    while r.get(K["ac"]) and r.get(K["ac"]).decode() != "joint_controller":
        r.set(K["ac"], "joint_controller")

    def grip(closed):
        r.set(K["grip_mode"], "g" if closed else "o")
        if not closed:
            r.set(K["nkin"], "0")

    print(f"replaying at {args.speed}x ...")
    grip(bool(g[0]))
    for i in range(len(q)):
        r.set(K["jgoal"], json.dumps(q[i].tolist()))
        if i > 0 and g[i] != g[i-1]:
            grip(bool(g[i]))
            print(f"  gripper -> {'CLOSED' if g[i] else 'open'} at waypoint {i}")
        # hold the needle pinned while gripped (sim)
        if g[i]:
            cp = r.get(K["cp"]); co = r.get(K["co"])
            if cp and co and "nan" not in cp.decode().lower():
                Tf = np.eye(4); Tf[:3,:3] = np.array(json.loads(co.decode())); Tf[:3,3] = np.array(json.loads(cp.decode()))
                r.set(K["npose"], json.dumps((Tf @ GRASP).tolist())); r.set(K["nkin"], "1")
        dt = (t[i] - t[i-1]) if i > 0 else 0.02
        time.sleep(max(dt, 0.005) / args.speed)
    print("playback done")


if __name__ == "__main__":
    main()
