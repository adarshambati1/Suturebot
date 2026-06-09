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

PIERCE_DEEPEN_CM = 2.0      # extra penetration at EACH push-in below, along its own
                            # pierce direction. More = harder push (cartesian position
                            # control under-penetrates against contact). 0 = off.
PIERCE_TIMES = [28.0, 78.0] # approx times (s) of the push-ins-from-below to deepen
                            # (nearest tip apex is used; add/remove to taste)

CUT_TOP_LOOP = True         # at the top of the apple: keep the same-spot grabs but
                            # cut the little loop before the pull-through
PULL_REBASE = False         # also shift the pull grab to the first-grab spot. OFF:
                            # the needle stays at the insert spot, so shifting the
                            # jaws up closed ABOVE the needle (missed) -> grab where
                            # the needle actually is (the demo's original 3rd spot)

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

    # ---- push-in deepening (one localized overshoot per PIERCE_TIME) ----
    # Each push-in's deepest point = where the NEEDLE TIP is driven furthest into
    # the apple (max tip-z near that time). Overshoot there along the tip's plunge
    # direction so the position controller pushes HARDER (more force) and the
    # needle seats deeper. Applied to every push-in listed in PIERCE_TIMES.
    tips = np.array([T[:3, 3] + T[:3, :3] @ K.NEEDLE_TIP_OFFSET for T in poses])
    deepen_m = PIERCE_DEEPEN_CM / 100.0

    # locate each push-in: snap the requested time to the nearest tip-z apex
    def nearest_apex(tt):
        c = int(np.argmin(np.abs(t - tt)))
        lo, hi = max(0, c - 30), min(len(q), c + 30)
        return lo + int(np.argmax(tips[lo:hi, 2]))

    apexes = [nearest_apex(tt) for tt in PIERCE_TIMES]
    pdirs = []
    for a in apexes:
        v = tips[a] - tips[max(0, a - 25)]           # this push-in's penetration dir
        nv = np.linalg.norm(v)
        pdirs.append(v / nv if nv > 1e-6 else np.zeros(3))
    RAMP, TAPER = 25, 20                              # ramp in over the rise, taper after

    def pierce_off(i):
        off = np.zeros(3)
        if deepen_m == 0:
            return off
        for a, pd in zip(apexes, pdirs):
            if a - RAMP <= i <= a:
                off = off + deepen_m * (i - (a - RAMP)) / RAMP * pd
            elif a < i <= a + TAPER:
                off = off + deepen_m * (1 - (i - a) / TAPER) * pd
        return off

    # ---- top cleanup: cut the little loop + pull-through from the first-grab spot ----
    flange = np.array([T[:3, 3] for T in poses])
    keep = list(range(len(q)))

    def pull_offset(i):
        return np.zeros(3)

    if CUT_TOP_LOOP:
        closes_all = [i for i in range(1, len(g)) if g[i] and not g[i - 1]]
        opens_all = [i for i in range(1, len(g)) if not g[i] and g[i - 1]]
        top = [i for i in closes_all if 33 < t[i] < 76]      # grabs at the top of the apple
        if len(top) >= 2:
            third = top[int(np.argmin([flange[i, 2] for i in top]))]   # lower/drifted grab = pull
            same = [i for i in top if i != third]
            first_grab_pos = np.median(flange[same], axis=0)
            offset = (first_grab_pos - flange[third]) if PULL_REBASE else np.zeros(3)
            prev_open = max([o for o in opens_all if o < third], default=third)
            pull_end = min([o for o in opens_all if o > third], default=len(q) - 1)
            keep = list(range(0, prev_open + 1)) + list(range(third, len(q)))  # drop the loop
            TAPER2 = 30

            def pull_offset(i, _o=offset, _t=third, _p=pull_end, _T=TAPER2):
                if _t <= i <= _p:
                    return _o
                if _p < i <= _p + _T:
                    return _o * (1 - (i - _p) / _T)
                return np.zeros(3)

            shift = f"shifted {np.round(offset*100,1)}cm to first-grab spot" if PULL_REBASE \
                else "grab at the needle's actual spot (no shift)"
            print(f"  CUT TOP LOOP wp{prev_open+1}-{third} (t{t[prev_open+1]:.0f}-{t[third]:.0f}s); "
                  f"pull-through: {shift}")

    adj = []
    for i in keep:
        pos = poses[i][:3, 3] + GLOBAL_OFFSET + pierce_off(i) + pull_offset(i)
        for (ts, te, off) in PHASE_OFFSETS:
            if ts <= t[i] <= te:
                pos = pos + np.asarray(off, float)
        adj.append((pos, poses[i][:3, :3], i))      # carry original index for q/g/t
    if deepen_m != 0:
        for a, pd in zip(apexes, pdirs):
            print(f"  PIERCE deepen {PIERCE_DEEPEN_CM:+.1f}cm @ push-in t={t[a]:.0f}s "
                  f"(wp{a}) along {np.round(pd,2)}")
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
    prev_i = None
    for pos, R, i in adj:
        if deepen_m != 0 and i in apexes:
            print(f"  >> pierce deepen: PEAK {PIERCE_DEEPEN_CM:+.1f}cm @t={t[i]:.0f}s (push-in)")
        r.set(K_["cpos"], json.dumps(pos.tolist()))
        r.set(K_["cori"], json.dumps(R.tolist()))
        if not args.no_nullspace:
            r.set(K_["njoint"], json.dumps(q[i].tolist()))
        if prev_i is not None and g[i] != g[prev_i]:    # transition vs previous KEPT frame
            trans_idx += 1
            act = "close" if g[i] else "open"
            if trans_idx in SKIP_CLAMP_EVENTS:
                print(f"  (skipped clamp event {trans_idx} {act} @t={t[i]:.0f}s)")
            else:
                grip(bool(g[i]))
                print(f"  gripper -> {'CLOSED' if g[i] else 'open'} "
                      f"[event {trans_idx}] @t={t[i]:.0f}s")
        if prev_i is None:
            dt = 0.02
        elif i - prev_i > 1:           # a CUT junction (loop removed) -> don't wait the gap
            dt = 0.1
        else:
            dt = t[i] - t[prev_i]
        time.sleep(max(dt, 0.005) / args.speed)
        prev_i = i
    print("playback done")


if __name__ == "__main__":
    main()
