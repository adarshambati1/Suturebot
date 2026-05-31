"""B3: grip / carry / release the curved needle via the kinematic-pin patch.

While "gripped", the needle's body pose is pinned to a fixed transform off the
flange (the grasp seating), updated every tick so it rigidly follows the hand.
"Release" flips kinematic off and the needle falls under gravity.

This is the hook the running-stitch FSM is built on:
    grip()   -> commands::Needle::kinematic = 1, pose driven from the flange
    release()-> commands::Needle::kinematic = 0  (drops)

Run against suture_pad.xml:
    cd ../OpenSai && sh scripts/launch.sh config_folder/xml_config_files/suture_pad.xml
    python3 python_examples/needle_grip_test.py

GRASP_POS / GRASP_RPY below are the needle clocking (held perpendicular, at a
slant). They are first-pass defaults -- the real seating is trial-and-error.
"""
import json
import time

import numpy as np
import redis

ROBOT = "Rizon4s"
CONFIG_FILE = "suture_pad.xml"

# --- grasp seating: needle BODY frame (arc center) relative to the flange ---
# The needle hangs below the flange (flange +Z points down with the tool-down
# ORI), arc-center ~10 cm below. GRASP_RPY clocks the arc plane -- tune it.
GRASP_POS = np.array([0.0, 0.0, 0.10])
GRASP_RPY = np.array([0.0, 0.0, 0.0])

# Tool-down flange orientation (matches the other suturebot clients).
ORI = np.array([[1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]])

K = {
    "goal_pos":  f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_position",
    "goal_ori":  f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_orientation",
    "cur_pos":   f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::current_position",
    "cur_ori":   f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::current_orientation",
    "active":    f"opensai::controllers::{ROBOT}::active_controller_name",
    "needle_cmd_pose": "opensai::commands::Needle::pose",
    "needle_kin":      "opensai::commands::Needle::kinematic",
    "needle_pose":     "opensai::sensors::Needle::object_pose",
    "config":          "::sai-interfaces-webui::config_file_name",
}


def rpy_to_R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_from(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


GRASP = T_from(rpy_to_R(GRASP_RPY), GRASP_POS)


class Needle:
    def __init__(self, r):
        self.r = r
        self.gripped = False

    def flange_T(self):
        p = np.array(json.loads(self.r.get(K["cur_pos"]).decode()))
        Rm = np.array(json.loads(self.r.get(K["cur_ori"]).decode()))
        return T_from(Rm, p)

    def pin(self):
        """Pin the needle to flange * GRASP (call repeatedly while gripped)."""
        T = self.flange_T() @ GRASP
        self.r.set(K["needle_cmd_pose"], json.dumps(T.tolist()))

    def grip(self):
        self.pin()
        self.r.set(K["needle_kin"], "1")
        self.gripped = True
        print("  [grip]    needle pinned to flange")

    def release(self):
        self.r.set(K["needle_kin"], "0")
        self.gripped = False
        print("  [release] needle dropped to dynamics")

    def pose(self):
        return np.array(json.loads(self.r.get(K["needle_pose"]).decode()))[:3, 3]


def move(r, needle, goal, dur):
    """Command the flange to `goal` and spin for `dur`s, keeping the pin live."""
    r.set(K["goal_pos"], json.dumps(np.asarray(goal, float).tolist()))
    r.set(K["goal_ori"], json.dumps(ORI.tolist()))
    t_end = time.monotonic() + dur
    while time.monotonic() < t_end:
        if needle.gripped:
            needle.pin()
        time.sleep(0.02)


def main():
    r = redis.Redis()
    cfg = r.get(K["config"])
    if cfg is None or cfg.decode() != CONFIG_FILE:
        print(f"Expected OpenSai running {CONFIG_FILE}, got {cfg!r}.")
        return
    while r.get(K["active"]).decode() != "cartesian_controller":
        r.set(K["active"], "cartesian_controller")

    n = Needle(r)

    print("approach pickup pose")
    move(r, n, [0.45, -0.20, 0.32], 3.0)

    n.grip()
    move(r, n, [0.45, -0.20, 0.32], 0.5)
    fl = n.flange_T()[:3, 3]; ne = n.pose()
    print(f"  on grip: flange={fl.round(3).tolist()}  needle={ne.round(3).tolist()}")

    print("carry: lift")
    move(r, n, [0.45, -0.20, 0.42], 2.5)
    print(f"  carried: needle={n.pose().round(3).tolist()} (should track the hand up)")
    print("carry: translate")
    move(r, n, [0.47, -0.17, 0.42], 2.5)
    print("carry: lower over pad")
    move(r, n, [0.46, -0.20, 0.30], 2.5)
    carried = n.pose().copy()
    print(f"  carried: needle={carried.round(3).tolist()}")

    n.release()
    move(r, n, [0.46, -0.20, 0.30], 2.5)   # hold flange; needle falls
    dropped = n.pose().copy()
    print(f"  after release: needle={dropped.round(3).tolist()}")

    fell = dropped[2] < carried[2] - 0.05
    tracked = abs(carried[2] - 0.20) < 0.05   # needle ~0.10 below flange@0.30
    print("\nRESULT:", "grip/carry/release OK" if (tracked and fell) else
          f"check (tracked={tracked}, fell={fell})")


if __name__ == "__main__":
    main()
