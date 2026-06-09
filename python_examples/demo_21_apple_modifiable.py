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
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.002])   # +2mm z

PIERCE_DEEPEN_CM = [2.0, 5.3]  # PER-PIERCE extra +z (pierce1, pierce2). p2=5.3 puts the
                            # 2nd pierce at the SAME commanded z as the 1st: the demo has
                            # pierce2 3.3cm lower (tip 0.532 vs 0.565), + 2 to match p1's
                            # overshoot -> both reach tip-z ~0.585. (use 3.3 to match p1's
                            # demo level instead of its overshot level)
PIERCE_TIMES = [28.0, 99.0] # push-in times: pierce1 ~28s, pierce2 ~99s. (t78 was MIS-
                            # tagged: that's stitch-1's pull-THROUGH, not a pierce -- left out)
PIERCE_PUSH_VERTICAL = True # push the deepen straight UP into the apple (+z) instead of
                            # along the recorded plunge -- pierce1's plunge was 40% lateral,
                            # wasting the overshoot; +z puts the full 2cm deeper
RELAX_NULLSPACE_IN_PIERCE = True  # drop the nullspace joint pin during the pierce HOLD so
                            # it stops biasing the arm back to the un-offset (shallower) pose

CUT_TOP_LOOP = True         # at the top of the apple: keep the same-spot grabs but
                            # cut the little loop before the pull-through
PULL_REBASE = False         # also shift the pull grab to the first-grab spot. OFF:
                            # the needle stays at the insert spot, so shifting the
                            # jaws up closed ABOVE the needle (missed) -> grab where
                            # the needle actually is (the demo's original 3rd spot)

RAISE_TOP_REGRIP_CM = 0.0   # OFF -- the real problem isn't a 1/4cm bite, it's a 2-5cm
                            # SCOOP (flange dips down then up at the grab). See DESCOOP.
RAISE_REGRIP_TIMES = [56.0, 104.0]

DESCOOP_WINDOWS = [(62.0, 72.0), (99.0, 108.0)]  # flatten the flange-z DIP (scoop) here:
                            # stitch-1 last/3rd regrip + the last stitch's grip.
DESCOOP_STRENGTH = 0.5      # soften the scoop (the ~5cm plunge down to the grab). It's a
                            # DIAL: 0 = full demo scoop (grabs deepest at the needle),
                            # 1 = flat (grabs high, in the air). 0.5 ~= halfway. Now that the
                            # global -2.5cm height is set, tune this so it grabs the needle
                            # without a dramatic dig: lower if it grabs high, raise if it digs.

EVENT10_RAISE_CM = 0.0      # was 3.5 to lift event 10 out of the over-deep caused by the
                            # -2.5cm global; reset to 0 now the global's gone + events 9,10
                            # are skipped (grip just holds). Re-eval if the pull dips.

GRIP12_X_BACK_CM = 1.0      # pull stitch-1 grips 1&2 (catch ~36s + regrip ~51s) BACK in -x
                            # by this much -- less far forward, but NOT onto the pierce plane
                            # (which is ~1.7cm back); keeps fruit clearance. 0 = off.

SKIP_CLAMP_EVENTS = [0, 1, 9, 10]  # raw clamp-transition indices to NOT actuate.
                            # 0,1 = static first close/open (no motion). 9,10 = the release
                            # after the perfect event-8 grip + the over-deep re-grab -> skip
                            # both so the good grip just HOLDS through the pull (no re-grab).

PHASE_OFFSETS: list = [     # per-time-window world shifts (m): (t0, t1, [dx,dy,dz])
    # (96.0, 102.0, np.array([0.0,0.0,-0.02])),  # re-add to lower the 2nd pierce if it's
    #                                            # still too high now the driver's normal
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
    # per-pierce deepen (list aligned with PIERCE_TIMES; scalar broadcasts to all)
    _dc = PIERCE_DEEPEN_CM if isinstance(PIERCE_DEEPEN_CM, (list, tuple)) else [PIERCE_DEEPEN_CM] * len(PIERCE_TIMES)
    deepen_list = [c / 100.0 for c in _dc]
    any_deepen = any(abs(x) > 1e-9 for x in deepen_list)

    # locate each push-in: snap the requested time to the nearest tip-z apex
    def nearest_apex(tt):
        c = int(np.argmin(np.abs(t - tt)))
        lo, hi = max(0, c - 30), min(len(q), c + 30)
        return lo + int(np.argmax(tips[lo:hi, 2]))

    apexes = [nearest_apex(tt) for tt in PIERCE_TIMES]
    pdirs = []
    for a in apexes:
        if PIERCE_PUSH_VERTICAL:
            pdirs.append(np.array([0.0, 0.0, 1.0]))    # full push straight up into the apple
        else:
            v = tips[a] - tips[max(0, a - 25)]         # recorded plunge (40% lateral on p1)
            nv = np.linalg.norm(v)
            pdirs.append(v / nv if nv > 1e-6 else np.zeros(3))
    # ramp in over the rise, HOLD at max so the controller actually pushes in; TAPER so
    # pierce_off decays to 0 BEFORE the catch grab (HOLD+TAPER=24: apex84 -> ends wp108 <
    # catch wp109), otherwise the leftover push drives the jaws into the apple at the grab.
    RAMP, HOLD, TAPER = 20, 16, 8

    def in_pierce_hold(i):
        return any(a <= i < a + HOLD for a in apexes)

    def pierce_off(i):
        off = np.zeros(3)
        for a, pd, dm in zip(apexes, pdirs, deepen_list):
            if dm == 0:
                continue
            if a - RAMP <= i < a:
                off = off + dm * (i - (a - RAMP)) / RAMP * pd
            elif a <= i < a + HOLD:
                off = off + dm * pd                              # HOLD at full depth
            elif a + HOLD <= i <= a + HOLD + TAPER:
                off = off + dm * (1 - (i - a - HOLD) / TAPER) * pd
        return off

    # ---- top cleanup: cut the little loop + pull-through from the first-grab spot ----
    flange = np.array([T[:3, 3] for T in poses])
    keep = list(range(len(q)))
    closes_all = [i for i in range(1, len(g)) if g[i] and not g[i - 1]]
    opens_all = [i for i in range(1, len(g)) if not g[i] and g[i - 1]]

    # ---- lift the last on-top regrip of each pierce (+z) so jaws don't bite the apple ----
    raise_m = RAISE_TOP_REGRIP_CM / 100.0
    raise_grabs = []
    for tt in RAISE_REGRIP_TIMES:
        cand = [c for c in closes_all if abs(t[c] - tt) < 9]
        if cand:
            raise_grabs.append(min(cand, key=lambda c: abs(t[c] - tt)))
    RW = 12

    def raise_off(i):
        if raise_m == 0:
            return np.zeros(3)
        for c in raise_grabs:
            if abs(i - c) <= RW:
                return np.array([0.0, 0.0, raise_m])
        return np.zeros(3)

    def pull_offset(i):
        return np.zeros(3)

    # ---- pull stitch-1 grips 1&2 back in -x (less forward; keep fruit clearance) ----
    gx_m = GRIP12_X_BACK_CM / 100.0

    def grip_off(i):
        # time-windowed so it covers the catch (~36s) + first regrip (~51s) but NOT the
        # 3rd grab (~56s); 1.5s taper in/out to avoid a step.
        if gx_m == 0:
            return np.zeros(3)
        lo, hi, tp = 34.5, 54.0, 1.5
        if lo <= t[i] <= hi:
            f = max(0.0, min((t[i] - lo) / tp, (hi - t[i]) / tp, 1.0))
            return np.array([-gx_m * f, 0.0, 0.0])
        return np.zeros(3)

    # ---- raise the event-10 pull grab (+z) out of the over-deepened spot ----
    e10_m = EVENT10_RAISE_CM / 100.0

    def event10_off(i):
        if e10_m == 0:
            return np.zeros(3)
        lo, hi, tp = 71.0, 85.0, 1.5
        if lo <= t[i] <= hi:
            f = max(0.0, min((t[i] - lo) / tp, (hi - t[i]) / tp, 1.0))
            return np.array([0.0, 0.0, e10_m * f])
        return np.zeros(3)

    # ---- flatten SCOOPS: lift the flange-z dip up to a straight line across the window ----
    descoop = []
    for (ts, te) in DESCOOP_WINDOWS:
        lo = int(np.argmin(np.abs(t - ts)))
        hi = int(np.argmin(np.abs(t - te)))
        if hi > lo:
            descoop.append((lo, hi, flange[lo, 2], flange[hi, 2]))

    def descoop_z(i):
        for (lo, hi, zlo, zhi) in descoop:
            if lo <= i <= hi:
                base = zlo + (zhi - zlo) * (i - lo) / (hi - lo)   # straight line across
                if flange[i, 2] < base:
                    return np.array([0.0, 0.0, DESCOOP_STRENGTH * (base - flange[i, 2])])
        return np.zeros(3)

    if CUT_TOP_LOOP:
        top = [i for i in closes_all if 33 < t[i] < 76]      # grabs at the top of the apple
        if len(top) >= 2:
            third = top[int(np.argmin([flange[i, 2] for i in top]))]   # lower/drifted grab = pull
            same = [i for i in top if i != third]
            first_grab_pos = np.median(flange[same], axis=0)
            offset = (first_grab_pos - flange[third]) if PULL_REBASE else np.zeros(3)
            prev_open = max([o for o in opens_all if o < third], default=third)
            pull_end = min([o for o in opens_all if o > third], default=len(q) - 1)
            # Extend the cut THROUGH the post-grip downward dip: the needle is already in
            # hand (event-8 grip held), so there is no reason to scoop down to the old
            # re-grab spot. Cut from the grip until the flange-z recovers to the grip
            # level (where the real pull-through begins) -> arm goes straight across.
            grip_z = flange[prev_open, 2]
            cut_end = third
            while cut_end < len(q) - 1 and flange[cut_end, 2] < grip_z - 0.002:
                cut_end += 1
            keep = list(range(0, prev_open + 1)) + list(range(cut_end, len(q)))
            TAPER2 = 30

            def pull_offset(i, _o=offset, _t=cut_end, _p=pull_end, _T=TAPER2):
                if _t <= i <= _p:
                    return _o
                if _p < i <= _p + _T:
                    return _o * (1 - (i - _p) / _T)
                return np.zeros(3)

            shift = f"shifted {np.round(offset*100,1)}cm" if PULL_REBASE else "no down-scoop"
            print(f"  CUT wp{prev_open+1}-{cut_end} (t{t[prev_open+1]:.0f}-{t[cut_end]:.0f}s): "
                  f"loop + post-grip dip -> straight to pull ({shift})")

    adj = []
    for i in keep:
        pos = (poses[i][:3, 3] + GLOBAL_OFFSET + pierce_off(i) + pull_offset(i)
               + raise_off(i) + grip_off(i) + descoop_z(i) + event10_off(i))
        for (ts, te, off) in PHASE_OFFSETS:
            if ts <= t[i] <= te:
                pos = pos + np.asarray(off, float)
        adj.append((pos, poses[i][:3, :3], i))      # carry original index for q/g/t
    if any_deepen:
        for a, pd, dm in zip(apexes, pdirs, deepen_list):
            print(f"  PIERCE deepen {dm*100:+.1f}cm @ push-in t={t[a]:.0f}s "
                  f"(wp{a}) along {np.round(pd,2)}")
    if SKIP_CLAMP_EVENTS:
        print(f"  SKIP clamp events {SKIP_CLAMP_EVENTS} (no actuation)")
    if raise_m != 0 and raise_grabs:
        print(f"  RAISE regrips +{RAISE_TOP_REGRIP_CM}cm (+z) at " +
              ", ".join(f"t={t[c]:.0f}s(wp{c})" for c in raise_grabs))
    if gx_m != 0:
        print(f"  GRIP12 back -{GRIP12_X_BACK_CM}cm (-x) over t=34.5-54s (grips 1&2, "
              "keeps clearance, not on the pierce plane)")
    if PIERCE_PUSH_VERTICAL:
        print("  PIERCE push = vertical (+z), full overshoot goes deeper")
    if RELAX_NULLSPACE_IN_PIERCE:
        print("  NULLSPACE relaxed during pierce HOLD (lets the overshoot through)")
    if descoop:
        print(f"  DE-SCOOP flatten z dip in windows {DESCOOP_WINDOWS}")
    if not np.allclose(GLOBAL_OFFSET, 0):
        print(f"  GLOBAL_OFFSET {np.round(GLOBAL_OFFSET*100,1)}cm (driver re-zero comp)")
    if e10_m != 0:
        print(f"  EVENT10 raise +{EVENT10_RAISE_CM}cm (+z) over t71-85 (pull region)")

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
        if any_deepen and i in apexes:
            dm = deepen_list[apexes.index(i)]
            print(f"  >> pierce deepen: PEAK {dm*100:+.1f}cm @t={t[i]:.0f}s (push-in)")
        r.set(K_["cpos"], json.dumps(pos.tolist()))
        r.set(K_["cori"], json.dumps(R.tolist()))
        # skip the nullspace pin during the pierce HOLD: pinning to the un-offset demo
        # config biases the arm to under-penetrate the very overshoot we're commanding.
        if not args.no_nullspace and not (RELAX_NULLSPACE_IN_PIERCE and in_pierce_hold(i)):
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
