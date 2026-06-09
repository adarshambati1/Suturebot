"""cartesian_replay.py — standalone cartesian replay of one demo, COORDINATES EXPOSED.

By default this produces the EXACT same motion as
    playback_cartesian.py <demo> --speed SPEED
(same pause-trim + smoothing, FK each joint config to a flange world pose, stream
goal_position/goal_orientation to the cartesian_controller, pin the demo joint
config in the nullspace, leave the gripper as-is + toggle at recorded events).

The difference: every coordinate knob is collected in the CONTROLS block below so
you can adjust the trajectory in WORLD space without digging through the loop.
All knobs default to no-op -> identical to playback_cartesian. As failure modes
come up, set them:

  GLOBAL_OFFSET     shift the whole trajectory in world (m)
  EXTRA_PUSH_CM     deeper initial pierce (cm along needle axis, ramped)
  PHASE_OFFSETS     per-time-window world shifts: (t_start, t_end, [dx,dy,dz] m)

    python3 python_examples/cartesian_replay.py log_files/demos/demo_21.npz --speed 0.5

Watch the first run, e-stop ready (cartesian/IK can move oddly near a wrist
singularity; fall back to playback_smooth if it flails).
"""
from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np
import redis

import kinematics as K
from playback_smooth import trim_pauses, moving_average


# ============================ CONTROLS (edit me) ============================
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.0])   # world xyz shift on EVERY waypoint (m)
EXTRA_PUSH_CM = 0.0                          # extra initial-pierce depth (cm, needle axis)
PHASE_OFFSETS: list = [                      # per-time-window world shifts (m)
    # (t_start_s, t_end_s, np.array([dx, dy, dz])),
    # e.g. (30.0, 46.0, np.array([0.0, 0.0, -0.01])),  # 2nd pierce 1cm lower
]
# ===========================================================================


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
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--raw", action="store_true", help="no smoothing/trim")
    ap.add_argument("--no-nullspace", action="store_true")
    ap.add_argument("--init-grip", action="store_true",
                    help="force gripper to demo's initial state (default: leave as-is)")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    args = ap.parse_args()

    d = np.load(args.demo, allow_pickle=True)
    t, q, g = d["times"], d["q"], d["gripper"]
    robot = args.robot or (str(d["robot"]) if "robot" in d.files else "Titania")
    t = t - t[0]
    print(f"loaded {len(q)} samples, {t[-1]:.1f}s, robot={robot}")
    if not args.raw:
        t, q, g = trim_pauses(t, q, g)
        q = moving_average(q, w=5)
        print(f"smoothed -> {len(q)} waypoints")

    poses = [K.fk_flange(qi) for qi in q]

    # ---- apply CONTROLS to the world poses -------------------------------------
    closes = [i for i in range(1, len(g)) if g[i] and not g[i - 1]]
    push_end = closes[0] if closes else len(q)
    extra_m = EXTRA_PUSH_CM / 100.0
    needle_dir = K.NEEDLE_TIP_OFFSET / np.linalg.norm(K.NEEDLE_TIP_OFFSET)
    adj = []
    for i, T in enumerate(poses):
        pos = T[:3, 3].copy() + GLOBAL_OFFSET
        for (ts, te, off) in PHASE_OFFSETS:
            if ts <= t[i] <= te:
                pos = pos + np.asarray(off, float)
        if extra_m > 0 and i < push_end:
            pos = pos + extra_m * ((i + 1) / push_end) * (T[:3, :3] @ needle_dir)
        adj.append((pos, T[:3, :3]))
    if not np.allclose(GLOBAL_OFFSET, 0) or extra_m > 0 or PHASE_OFFSETS:
        print(f"  CONTROLS active: offset={GLOBAL_OFFSET}, extra_push={EXTRA_PUSH_CM}cm, "
              f"{len(PHASE_OFFSETS)} phase offsets")

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()
    K_ = keys(robot)
    cfg = r.get(K_["config"])
    print(f"OpenSai config: {cfg.decode() if cfg else None}")
    if r.get(K_["qsens"]) is None:
        sys.exit("No joint stream — is OpenSai + the driver running?")
    while r.get(K_["ac"]) and r.get(K_["ac"]).decode() != "cartesian_controller":
        r.set(K_["ac"], "cartesian_controller")
        time.sleep(0.05)

    cur = r.get(K_["curp"])
    p0 = adj[0][0]
    if cur is not None and "nan" not in cur.decode().lower():
        now = np.array(json.loads(cur.decode()))
        dist = np.linalg.norm(now - p0)
        print(f"\n[pre-flight] flange now ({now[0]:+.3f},{now[1]:+.3f},{now[2]:+.3f}) "
              f"-> first pose ({p0[0]:+.3f},{p0[1]:+.3f},{p0[2]:+.3f})  = {dist*100:.1f} cm")
        if dist > 0.10:
            print("   WARNING: large first move — jog closer first; e-stop ready.")
    if input("Execute on the REAL robot? e-stop ready. [y/N]: ").strip().lower() not in ("y", "yes"):
        print("aborted."); return

    def grip(closed):
        r.set(K_["grip"], "g" if closed else "o")

    if args.init_grip:
        grip(bool(g[0]))
    else:
        print("  leaving gripper AS-IS at start")

    print(f"replaying at {args.speed}x ...")
    for i in range(len(q)):
        pos, R = adj[i]
        r.set(K_["cpos"], json.dumps(pos.tolist()))
        r.set(K_["cori"], json.dumps(R.tolist()))
        if not args.no_nullspace:
            r.set(K_["njoint"], json.dumps(q[i].tolist()))
        if i > 0 and g[i] != g[i - 1]:
            grip(bool(g[i]))
            print(f"  gripper -> {'CLOSED' if g[i] else 'open'} at waypoint {i} (t={t[i]:.0f}s)")
        dt = (t[i] - t[i - 1]) if i > 0 else 0.02
        time.sleep(max(dt, 0.005) / args.speed)
    print("playback done")


if __name__ == "__main__":
    main()
