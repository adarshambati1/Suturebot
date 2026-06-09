"""Execute a stitch PLAN (from analyze_demo.py) as clean point-to-point joint
moves + clamp toggles -- the non-jittery autonomous version of the demo.

DRY-RUN by default: prints every step and, via FK, where the flange/needle go,
WITHOUT moving the robot. Add --execute to actually drive it (joint controller,
slow, with a confirmation prompt).

Joint-space on purpose: replaying the demo's joint keyframes avoids the wrist
singularities that an IK/cartesian path can hit. This is the v1 player -- it
reproduces the taught stitch at the taught location (apple where you taught it).
Vision anchoring to a detected cut is the next layer.

    python3 python_examples/play_plan.py log_files/demos/demo_4_plan.json
    python3 python_examples/play_plan.py demo_4_plan.json --execute --speed 0.3
"""

from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np

import kinematics as K


def keys(robot):
    base = f"opensai::controllers::{robot}"
    return {
        "active": f"{base}::active_controller_name",
        "jgoal": f"{base}::joint_controller::joint_task::goal_position",
        "qsens": f"opensai::sensors::{robot}::joint_positions",
        "grip": f"opensai::commands::{robot}::gripper::mode",
        "config": "::sai-interfaces-webui::config_file_name",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="plan json from analyze_demo.py")
    ap.add_argument("--execute", action="store_true", help="actually move the robot")
    ap.add_argument("--speed", type=float, default=0.3,
                    help="0..1 fraction of nominal speed (default 0.3 = slow)")
    ap.add_argument("--move-time", type=float, default=3.0,
                    help="seconds per MOVE at speed=1 (scaled by 1/speed)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    robot = plan.get("robot", "Titania")
    steps = plan["steps"]

    # ---- DRY RUN: print the plan + FK of each target (no motion) ----
    print(f"PLAN: {len(steps)} steps, robot={robot}, source={plan.get('source','?')}\n")
    for i, s in enumerate(steps):
        if s["type"] == "clamp":
            print(f"  [{i:2d}] CLAMP {s['action'].upper()}")
        else:
            q = np.array(s["q"])
            T = K.fk_flange(q)
            tip, _ = K.fk_needle_tip(q)
            print(f"  [{i:2d}] MOVE  flange=({T[0,3]:+.3f},{T[1,3]:+.3f},{T[2,3]:+.3f})  "
                  f"needle_tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})  "
                  f"q(deg)=[{', '.join(f'{x:+.0f}' for x in np.degrees(q))}]")

    if not args.execute:
        print("\nDRY RUN only (no motion). Re-run with --execute to drive the robot.")
        return

    # ---- EXECUTE ----
    import redis
    r = redis.Redis(host=args.host, port=args.port)
    r.ping()
    K_ = keys(robot)

    cfg = r.get(K_["config"])
    print(f"\nOpenSai config: {cfg.decode() if cfg else None}")
    cur = r.get(K_["qsens"])
    if cur is None:
        sys.exit("No joint stream — is OpenSai + the driver running?")
    q_now = np.array(json.loads(cur))

    # First MOVE target — warn if the arm is far from it (big first move).
    first = next((s for s in steps if s["type"] == "move"), None)
    if first is not None:
        d = np.degrees(np.linalg.norm(np.array(first["q"]) - q_now))
        print(f"arm is {d:.0f} deg from the plan's first pose "
              f"(it will move there first).")
    if input("Execute on the REAL robot? e-stop ready. [y/N]: ").strip().lower() not in ("y", "yes"):
        print("aborted.")
        return

    # joint controller active
    r.set(K_["active"], "joint_controller")
    time.sleep(0.5)
    move_time = args.move_time / max(args.speed, 0.05)

    for i, s in enumerate(steps):
        if s["type"] == "clamp":
            r.set(K_["grip"], "g" if s["action"] == "close" else "o")
            print(f"  [{i:2d}] clamp {s['action']}")
            time.sleep(2.0)
            continue
        target = np.array(s["q"])
        cur = np.array(json.loads(r.get(K_["qsens"])))
        nsub = max(int(move_time / 0.05), 1)
        print(f"  [{i:2d}] move ({np.degrees(np.linalg.norm(target-cur)):.0f} deg) over {move_time:.1f}s")
        for k in range(1, nsub + 1):
            q_i = cur + (target - cur) * (k / nsub)
            r.set(K_["jgoal"], json.dumps(q_i.tolist()))
            time.sleep(0.05)
        time.sleep(0.3)
    print("\n[done] plan executed.")


if __name__ == "__main__":
    main()
