"""playback_cartesian.py — replay a demo in CARTESIAN space (stiff controller).

The demo recorded JOINT angles; this FK's each one to a flange WORLD pose and
streams goal_position + goal_orientation to the cartesian_controller -- the SAME
stiff (kp=400) controller the foam clients use, which holds position cleanly
(no gravity sag, unlike the soft default joint controller). The demo's joint
config is also fed to the cartesian controller's NULLSPACE joint_task to pin the
elbow near the demonstrated configuration (avoids 7-DOF redundancy drift).

Gripper: left AS-IS at start (a pre-loaded needle isn't dropped); toggled at the
recorded transitions. Smoothing/pause-trim reused from playback_smooth.

    python3 python_examples/playback_cartesian.py log_files/demos/demo_N.npz --speed 0.5
    python3 python_examples/playback_cartesian.py demo_N.npz --raw --no-nullspace

WATCH the first run, e-stop ready: cartesian control uses IK, so if the demo
path nears a wrist singularity it may move oddly. If it flails, fall back to the
joint-space playback_smooth.py.
"""
import argparse
import json
import sys
import time

import numpy as np
import redis

import kinematics as K
from playback_smooth import trim_pauses, moving_average


def keys(robot):
    base = f"opensai::controllers::{robot}"
    return {
        "ac": f"{base}::active_controller_name",
        "cpos": f"{base}::cartesian_controller::cartesian_task::goal_position",
        "cori": f"{base}::cartesian_controller::cartesian_task::goal_orientation",
        "curp": f"{base}::cartesian_controller::cartesian_task::current_position",
        "njoint": f"{base}::cartesian_controller::joint_task::goal_position",
        "grip": f"opensai::commands::{robot}::gripper::mode",
        "qsens": f"opensai::sensors::{robot}::joint_positions",
        "config": "::sai-interfaces-webui::config_file_name",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo")
    ap.add_argument("--speed", type=float, default=0.5, help="0..1 fraction of recorded speed")
    ap.add_argument("--raw", action="store_true", help="replay raw (no smoothing/trim)")
    ap.add_argument("--no-nullspace", action="store_true",
                    help="don't pin the demo joint config in the cartesian nullspace")
    ap.add_argument("--extra-push", type=float, default=0.0,
                    help="cm of EXTRA penetration on the initial pierce (the segment "
                         "before the first clamp-close), along the needle axis. The "
                         "cartesian controller under-penetrates vs the hand-pushed demo "
                         "by ~force/kp; this overshoots the target to push harder. "
                         "Ramps 0->full over the pierce. Start small (~1) and tune up.")
    ap.add_argument("--init-grip", action="store_true",
                    help="force the gripper to the demo's initial state (default: leave as-is)")
    ap.add_argument("--robot", default=None, help="override; default = robot the demo was recorded on")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    args = ap.parse_args()

    d = np.load(args.demo, allow_pickle=True)
    t, q, g = d["times"], d["q"], d["gripper"]
    robot = args.robot or (str(d["robot"]) if "robot" in d.files else "Titania")
    print(f"loaded {len(q)} samples, {t[-1]:.1f}s, {q.shape[1]} joints, robot={robot}")
    if not args.raw:
        t, q, g = trim_pauses(t, q, g)
        q = moving_average(q, w=5)
        print(f"smoothed -> {len(q)} waypoints (pauses trimmed, filtered)")

    # precompute flange world poses from the (smoothed) joint trajectory
    poses = [K.fk_flange(qi) for qi in q]

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()
    K_ = keys(robot)

    cfg = r.get(K_["config"])
    print(f"OpenSai config: {cfg.decode() if cfg else None}")
    if r.get(K_["qsens"]) is None:
        sys.exit("No joint stream — is OpenSai + the driver running?")

    # activate the cartesian controller
    while r.get(K_["ac"]) and r.get(K_["ac"]).decode() != "cartesian_controller":
        r.set(K_["ac"], "cartesian_controller")
        time.sleep(0.05)

    # PRE-FLIGHT: first commanded flange pose vs where the arm is now.
    cur = r.get(K_["curp"])
    T0 = poses[0]
    if cur is not None and "nan" not in cur.decode().lower():
        now = np.array(json.loads(cur.decode()))
        dist = np.linalg.norm(now - T0[:3, 3])
        print(f"\n[pre-flight] flange now ({now[0]:+.3f},{now[1]:+.3f},{now[2]:+.3f})")
        print(f"             first pose ({T0[0,3]:+.3f},{T0[1,3]:+.3f},{T0[2,3]:+.3f})  "
              f"-> first move {dist*100:.1f} cm")
        if dist > 0.10:
            print("   WARNING: large first move — the cartesian controller will drive "
                  "straight there. Consider jogging closer first; e-stop ready.")
    if input("Execute on the REAL robot? e-stop ready. [y/N]: ").strip().lower() not in ("y", "yes"):
        print("aborted."); return

    def grip(closed):
        r.set(K_["grip"], "g" if closed else "o")

    if args.init_grip:
        grip(bool(g[0]))
        print(f"  gripper -> {'CLOSED' if g[0] else 'open'} (initial state from demo)")
    else:
        print("  leaving gripper AS-IS at start (use --init-grip to force demo's initial state)")

    # initial-pierce window = up to the FIRST clamp-close (the through/regrip).
    closes = [i for i in range(1, len(g)) if g[i] and not g[i - 1]]
    push_end = closes[0] if closes else len(q)
    extra_m = args.extra_push / 100.0
    needle_dir = K.NEEDLE_TIP_OFFSET / np.linalg.norm(K.NEEDLE_TIP_OFFSET)
    if extra_m > 0:
        print(f"  EXTRA PUSH: +{args.extra_push:.1f} cm along needle axis, ramped over "
              f"the initial pierce (waypoints 0..{push_end}).")

    print(f"replaying at {args.speed}x in cartesian space ...")
    for i in range(len(q)):
        T = poses[i]
        pos = T[:3, 3].copy()
        if extra_m > 0 and i < push_end:
            ramp = (i + 1) / push_end                       # 0 -> 1 across the pierce
            pos = pos + extra_m * ramp * (T[:3, :3] @ needle_dir)
        r.set(K_["cpos"], json.dumps(pos.tolist()))
        r.set(K_["cori"], json.dumps(T[:3, :3].tolist()))
        if not args.no_nullspace:                       # pin elbow near demo config
            r.set(K_["njoint"], json.dumps(q[i].tolist()))
        if i > 0 and g[i] != g[i - 1]:
            grip(bool(g[i]))
            print(f"  gripper -> {'CLOSED' if g[i] else 'open'} at waypoint {i}")
        dt = (t[i] - t[i - 1]) if i > 0 else 0.02
        time.sleep(max(dt, 0.005) / args.speed)
    print("playback done")


if __name__ == "__main__":
    main()
