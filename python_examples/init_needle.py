"""Set the needle's starting state for a run.

  --mode gripper : seat the needle in the (closed) gripper -- the real start,
                   matching FSM state 1 (SEAT). Moves to the ready pose, pins
                   the needle into the jaws, and holds it there.
  --mode wall    : leave the needle leaning against the support wall, as a free
                   dynamic body, for practicing the kill-window recovery /
                   pickup (FSM state 9).

Both re-assert their redis commands every tick to beat the sim's startup
re-seeding race (otherwise kinematic gets reset to 0 right after launch).

    python3 python_examples/init_needle.py            # gripper (default)
    python3 python_examples/init_needle.py --mode wall
"""
import argparse
import json
import time

import numpy as np
import redis

ROBOT = "Rizon4s"

# grasp seating: needle body (arc center) relative to the flange.
# Home flange orientation: tool-down rolled -35 deg. This SLANT pulls the wrist
# off the J6=0 singularity (J6 ~39 deg instead of ~3 deg), giving roll room for
# the bite drive. GRASP recomputed for this slant so the needle keeps the
# dialed-in look (gripped 1/3 from the swage).
ORI = np.array([[1.0, 0.0, 0.0],
                [0.0, -0.81915, -0.57358],
                [0.0, 0.57358, -0.81915]])
GRASP_POS = np.array([0.00778, 0.00587, 0.28510])
GRASP_R = np.array([[0.0, -1.0, 0.0],
                    [-0.75471, 0.0, 0.65606],
                    [-0.65606, 0.0, -0.75471]])
# flange pose that puts the needle grip over the slit (0.45,-0.30,0.25) given
# the slant; J6 stays ~23 deg, clear of the singularity.
READY = np.array([0.45, -0.139, 0.479])

# wall-lean start pose for the needle body (against the support wall, ~X=0.50).
WALL_POS = np.array([0.492, -0.30, 0.038])
WALL_R = np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0]])   # arc plane vertical, tilted toward the wall

K = {
    "gp": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_position",
    "go": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_orientation",
    "cp": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::current_position",
    "co": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::current_orientation",
    "ac": f"opensai::controllers::{ROBOT}::active_controller_name",
    "npose": "opensai::commands::Needle::pose",
    "nkin":  "opensai::commands::Needle::kinematic",
    "nsense": "opensai::sensors::Needle::object_pose",
    "config": "::sai-interfaces-webui::config_file_name",
}


def T(R, p):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p
    return M


def flange_T(r):
    """Flange pose, or None if the controller state isn't valid yet (NaN at
    startup for the first control cycle, or empty keys)."""
    rp, ro = r.get(K["cp"]), r.get(K["co"])
    if not rp or not ro:
        return None
    sp, so = rp.decode(), ro.decode()
    if "nan" in sp.lower() or "nan" in so.lower():
        return None
    try:
        return T(np.array(json.loads(so)), np.array(json.loads(sp)))
    except (ValueError, TypeError):
        return None


def needle_pos(r):
    return np.array(json.loads(r.get(K["nsense"]).decode()))[:3, 3]


def seat_in_gripper(r, hold=14.0):
    grasp = T(GRASP_R, GRASP_POS)
    t = time.monotonic()
    while time.monotonic() - t < hold:
        r.set(K["ac"], "cartesian_controller")
        r.set(K["go"], json.dumps(ORI.tolist()))
        r.set(K["gp"], json.dumps(READY.tolist()))
        r.set(K["nkin"], "1")
        Tf = flange_T(r)
        if Tf is not None:                      # skip the startup-NaN ticks
            r.set(K["npose"], json.dumps((Tf @ grasp).tolist()))
        time.sleep(0.02)
    p = flange_T(r)[:3, 3]; n = needle_pos(r)
    print(f"seated: flange={p.round(3).tolist()} needle={n.round(3).tolist()} "
          f"({round((p[2]-n[2])*100,1)} cm below flange)")


def lean_on_wall(r, settle=3.0):
    # pin the needle to the lean pose, then release so it rests against the wall.
    pose = T(WALL_R, WALL_POS)
    t = time.monotonic()
    while time.monotonic() - t < settle:
        r.set(K["nkin"], "1")
        r.set(K["npose"], json.dumps(pose.tolist()))
        time.sleep(0.02)
    r.set(K["nkin"], "0")            # release -> rests against the wall
    time.sleep(2.0)
    print(f"leaning on wall: needle={needle_pos(r).round(3).tolist()} (released to dynamics)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gripper", "wall"], default="gripper")
    args = ap.parse_args()

    r = redis.Redis()
    cfg = r.get(K["config"])
    if cfg is None or cfg.decode() != "suture_pad.xml":
        print(f"Expected OpenSai running suture_pad.xml, got {cfg!r}."); return

    if args.mode == "gripper":
        seat_in_gripper(r)
    else:
        lean_on_wall(r)


if __name__ == "__main__":
    main()
