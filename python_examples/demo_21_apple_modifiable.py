"""demo_21_apple_modifiable.py — replay demo_21 in cartesian space, apple-relative coords modifiable.

Specific to demo_21 (the "perfect except initial push" two-stitch). Base motion =
playback_cartesian (trim+smooth, FK each joint config -> flange world pose, stream
goal_position/goal_orientation to the cartesian_controller, pin the demo joint
config in the nullspace, gripper as-is + toggle at recorded events). All tunables
are in the CONTROLS block below.

    python3 python_examples/demo_21_apple_modifiable.py            # uses demo_21
    python3 python_examples/demo_21_apple_modifiable.py --speed 0.3

Watch the first run, e-stop ready (cartesian/IK can move oddly near a wrist
singularity; fall back to playback_smooth if it flails).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import redis

import kinematics as K
from playback_smooth import trim_pauses, moving_average


# ============================ CONTROLS (edit me) ============================
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.0])   # world xyz shift on EVERY waypoint (m)

PIERCE_DEEPEN_CM = 0.5      # deepen the INITIAL pierce by this much, along the
                            # direction it already pierced ("same as it did").
                            # Localized to the pierce (ramped in/out), 0 = off.

SKIP_CLAMP_EVENTS = [0, 1]  # raw clamp-transition indices to NOT actuate. 0,1 =
                            # the static first close@8.5s + open@9.2s (no motion).
                            # (events 2,3 @17-22.5s are also static if you want them too)

PHASE_OFFSETS: list = [     # per-time-window world shifts (m): (t0, t1, [dx,dy,dz])
    # (3.0, 36.0, np.array([0.0, 0.0, -0.01])),
]
# demo_21 phases: pierce1 ~start-36s | through/regrips ~36-72s | pull ~72-97s | final ~97-114s
# ===========================================================================

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEMO = os.path.join(REPO, "log_files", "demos", "demo_21.npz")


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
    ap.add_argument("demo", nargs="?", default=DEFAULT_DEMO)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--raw", action="store_true", help="no smoothing/trim")
    ap.add_argument("--no-nullspace", action="store_true")
    ap.add_argument("--init-grip", action="store_true")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    args = ap.parse_args()

    if not os.path.exists(args.demo):
        sys.exit(f"demo not found: {args.demo}")
    d = np.load(args.demo, allow_pickle=True)
    t, q, g = d["times"], d["q"], d["gripper"]
    robot = args.robot or (str(d["robot"]) if "robot" in d.files else "Titania")
    t = t - t[0]
    print(f"loaded {len(q)} samples, {t[-1]:.1f}s, robot={robot} ({os.path.basename(args.demo)})")
    if not args.raw:
        t, q, g = trim_pauses(t, q, g)
        q = moving_average(q, w=5)
        print(f"smoothed -> {len(q)} waypoints")

    poses = [K.fk_flange(qi) for qi in q]
    P0 = poses[0][:3, 3]

    # ---- initial-pierce deepening (localized bump at the actual PIERCE APEX) ----
    # The deepest pierce = where the NEEDLE TIP is driven furthest into the apple,
    # NOT where the clamp grabs (the grab is after the needle already retreated a
    # bit). Find the tip apex (max tip-z) before the first moved grab, and deepen
    # there along the empirical PLUNGE direction (how the tip actually went in).
    tips = np.array([T[:3, 3] + T[:3, :3] @ K.NEEDLE_TIP_OFFSET for T in poses])
    closes = [i for i in range(1, len(g)) if g[i] and not g[i - 1]]
    first_grab = next((i for i in closes if np.linalg.norm(poses[i][:3, 3] - P0) > 0.02),
                      (closes[0] if closes else len(q)))
    apex = int(np.argmax(tips[:first_grab, 2]))      # deepest tip before the grab
    RAMP_IN, TAPER = 30, 25
    deepen_m = PIERCE_DEEPEN_CM / 100.0
    kk = min(25, apex)
    pdir = tips[apex] - tips[apex - kk]              # plunge direction ("same as it did")
    n = np.linalg.norm(pdir)
    pdir = pdir / n if n > 1e-6 else np.zeros(3)

    def pierce_bump(i):
        if deepen_m <= 0:
            return 0.0
        if apex - RAMP_IN <= i <= apex:
            return deepen_m * (i - (apex - RAMP_IN)) / RAMP_IN
        if apex < i <= apex + TAPER:
            return deepen_m * (1 - (i - apex) / TAPER)
        return 0.0

    adj = []
    for i, T in enumerate(poses):
        pos = T[:3, 3].copy() + GLOBAL_OFFSET
        for (ts, te, off) in PHASE_OFFSETS:
            if ts <= t[i] <= te:
                pos = pos + np.asarray(off, float)
        pos = pos + pierce_bump(i) * pdir
        adj.append((pos, T[:3, :3]))
    if deepen_m > 0:
        print(f"  PIERCE deepen +{PIERCE_DEEPEN_CM}cm along {np.round(pdir,2)} "
              f"peaking at wp{apex} (t={t[apex]:.0f}s)")
    if SKIP_CLAMP_EVENTS:
        print(f"  SKIP clamp events {SKIP_CLAMP_EVENTS} (no actuation)")

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()
    K_ = keys(robot)
    print(f"OpenSai config: {(r.get(K_['config']) or b'?').decode()}")
    if r.get(K_["qsens"]) is None:
        sys.exit("No joint stream — is OpenSai + the driver running?")
    while r.get(K_["ac"]) and r.get(K_["ac"]).decode() != "cartesian_controller":
        r.set(K_["ac"], "cartesian_controller")
        time.sleep(0.05)

    cur = r.get(K_["curp"])
    p0 = adj[0][0]
    if cur is not None and "nan" not in cur.decode().lower():
        now = np.array(json.loads(cur.decode()))
        print(f"\n[pre-flight] first move = {np.linalg.norm(now - p0)*100:.1f} cm")
        if np.linalg.norm(now - p0) > 0.10:
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
    trans_idx = -1
    for i in range(len(q)):
        pos, R = adj[i]
        if deepen_m > 0 and i == max(0, apex - RAMP_IN):
            print(f"  >> pierce deepen: ramping in @t={t[i]:.0f}s")
        if deepen_m > 0 and i == apex:
            print(f"  >> pierce deepen: PEAK +{PIERCE_DEEPEN_CM}cm @t={t[i]:.0f}s")
        r.set(K_["cpos"], json.dumps(pos.tolist()))
        r.set(K_["cori"], json.dumps(R.tolist()))
        if not args.no_nullspace:
            r.set(K_["njoint"], json.dumps(q[i].tolist()))
        if i > 0 and g[i] != g[i - 1]:
            trans_idx += 1
            act = "close" if g[i] else "open"
            if trans_idx in SKIP_CLAMP_EVENTS:
                print(f"  (skipped clamp event {trans_idx} {act} @t={t[i]:.0f}s)")
            else:
                grip(bool(g[i]))
                print(f"  gripper -> {'CLOSED' if g[i] else 'open'} "
                      f"[event {trans_idx}] @t={t[i]:.0f}s")
        dt = (t[i] - t[i - 1]) if i > 0 else 0.02
        time.sleep(max(dt, 0.005) / args.speed)
    print("playback done")


if __name__ == "__main__":
    main()
