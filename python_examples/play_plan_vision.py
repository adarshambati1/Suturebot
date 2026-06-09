"""play_plan_vision.py — replay a demo PLAN as an FSM, vision-correcting the regrips.

The demo (analyze_demo.py -> plan.json) is the BACKBONE: it encodes the whole
stitch as an ordered list of MOVE (joint pose) + CLAMP (open/close) states, and
it already works. This player makes it ROBUST by correcting the regrips with
vision -- WITHOUT doing fragile vision-during-motion:

  At every CLAMP-CLOSE (a regrip), the arm STOPS at the taught pose (camera now
  static, where detection is reliable), tries to detect the needle tip in world,
  and:
    * needle VISIBLE  -> shift the regrip so the forceps lands on the ACTUAL
                         needle (delta from a recorded reference, FK->IK).
    * needle NOT seen  -> fall back to the taught pose (blind bottom-side regrip).
  Visibility itself decides top vs bottom -- no hardcoded classification.

Correction is delta anchoring. Run once to record references, then replay:

  # 1) record references (drives the robot; logs needle world at each regrip)
  python3 python_examples/play_plan_vision.py demo_4_plan.json --record-refs --execute

  # 2) autonomous, vision-corrected replay
  python3 python_examples/play_plan_vision.py demo_4_plan.json --execute --speed 0.3

DRY-RUN by default (prints the FSM + which steps are vision-corrected, no motion,
no camera). Needs the CAD calibration (calibration/t_flange_cam.json) loaded for
the world conversion; verify it like the live world readout first.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as K
try:                       # opencv/ultralytics only needed for the live run, not dry-run
    import fruit_scan as fs
except Exception:
    fs = None


def keys(robot):
    base = f"opensai::controllers::{robot}"
    return {
        "active": f"{base}::active_controller_name",
        "jgoal": f"{base}::joint_controller::joint_task::goal_position",
        "qsens": f"opensai::sensors::{robot}::joint_positions",
        "grip": f"opensai::commands::{robot}::gripper::mode",
        "config": "::sai-interfaces-webui::config_file_name",
    }


def needle_tip(nd, fruit):
    """Tip = detected endpoint nearer the apple centre (same rule as live_view)."""
    if not nd.get("found"):
        return None
    p1, p2 = np.array(nd["p1"], float), np.array(nd["p2"], float)
    ctr = np.array([fruit["cx"], fruit["cy"]], float) if fruit else (p1 + p2) / 2
    return p1 if np.linalg.norm(p1 - ctr) < np.linalg.norm(p2 - ctr) else p2


def detect_needle_world(r, model, tracker, intr, dark_v, conf, n=15, delay=0.06,
                        show=False, win_label=""):
    """At the current (stopped) pose, detect the needle tip and deproject it to
    world. Returns the median world XYZ over n frames, or None if not seen
    reliably. Pose comes from FK of the live joints (consistent with the
    joint-space player, no dependence on cartesian keys).

    show=True pops a window for each frame (apple box + green needle dot + dark
    mask) so you can WATCH it find / miss the needle at this regrip."""
    if intr is None:
        return None
    cur = r.get(keys_cache["qsens"])
    if cur is None:
        return None
    Tf = K.fk_flange(np.array(json.loads(cur)))
    fpos, fori = Tf[:3, 3], Tf[:3, :3]
    ds = intr.get("depth_scale", 0.001)
    if show:
        import cv2
    pts = []
    for _ in range(n):
        bgr, _ = fs.read_color(r)
        if bgr is None:
            time.sleep(delay); continue
        h, w = bgr.shape[:2]
        fruit = tracker.detect(model, bgr, classes_cache, conf, (0, 0, w, h))
        strip = tip = dmask = None
        if fruit is not None:
            strip = fs.detect_blue_line(bgr, fruit["box"], fs.BLUE_HSV_LO, fs.BLUE_HSV_HI)
            nd, dmask = fs.detect_needle_darkline(bgr, line=strip, box=fruit["box"],
                                                  dark_v=dark_v, return_mask=True)
            tip = needle_tip(nd, fruit)
            if tip is not None:
                depth, _ = fs.read_depth(r)
                if depth is not None:
                    z = fs.sample_depth_m(depth, int(tip[0]), int(tip[1]), ds, win=5)
                    if z is not None:
                        pts.append(fs.cam_to_world(fs.deproject(tip[0], tip[1], z, intr),
                                                   fpos, fori))
        if show:
            out = fs.annotate_stitch_plan(bgr, fruit, strip, [])
            if tip is not None:
                cv2.circle(out, tuple(tip.astype(int)), 7, (0, 255, 0), -1)
                cv2.circle(out, tuple(tip.astype(int)), 9, (0, 255, 0), 2)
            if dmask is not None:
                dm = dmask if dmask.ndim == 3 else cv2.cvtColor(dmask, cv2.COLOR_GRAY2BGR)
                dh, dw = dm.shape[:2]
                sc = min(220 / max(dw, 1), 220 / max(dh, 1), 1.0)
                dm = cv2.resize(dm, (int(dw * sc), int(dh * sc)))
                out[h - dm.shape[0]:h, w - dm.shape[1]:w] = dm
            found = tip is not None
            cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(out, f"{win_label} needle {'FOUND' if found else '--- searching'}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if found else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("regrip detect", out)
            cv2.waitKey(1)
        time.sleep(delay)
    if len(pts) < max(3, n // 3):          # too few hits -> "not visible"
        return None
    return np.median(np.array(pts), axis=0)


# globals set in main (kept simple; this is a single-run CLI)
keys_cache = {}
classes_cache = set()


def goto(r, K_, q_target, move_time):
    cur = np.array(json.loads(r.get(K_["qsens"])))
    nsub = max(int(move_time / 0.05), 1)
    for k in range(1, nsub + 1):
        q_i = cur + (q_target - cur) * (k / nsub)
        r.set(K_["jgoal"], json.dumps(q_i.tolist()))
        time.sleep(0.05)
    time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="plan json from analyze_demo.py")
    ap.add_argument("--execute", action="store_true", help="actually move the robot")
    ap.add_argument("--record-refs", action="store_true",
                    help="record the needle world ref at each regrip (implies a real run)")
    ap.add_argument("--refs", default=None, help="refs json (default <plan>_refs.json)")
    ap.add_argument("--speed", type=float, default=0.3, help="0..1 fraction of nominal speed")
    ap.add_argument("--move-time", type=float, default=3.0, help="s per MOVE at speed=1")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--fruit-color", default=None)
    ap.add_argument("--dark-v", type=int, default=160)
    ap.add_argument("--max-correction", type=float, default=0.02,
                    help="reject a vision regrip correction larger than this (m) "
                         "and fall back to the taught pose (default 0.02 = 2cm)")
    ap.add_argument("--show", action="store_true",
                    help="pop a camera window at each regrip so you can watch the "
                         "detector find/miss the needle (apple box + green dot + mask)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    robot = plan.get("robot", "Titania")
    steps = plan["steps"]
    refs_path = args.refs or (os.path.splitext(args.plan)[0] + "_refs.json")
    refs = {}
    if not args.record_refs and os.path.exists(refs_path):
        refs = {int(k): np.array(v) for k, v in json.load(open(refs_path)).get("refs", {}).items()}

    # ---- annotate the FSM (which moves are vision-corrected regrips) ----
    def is_regrip(i):
        return (steps[i]["type"] == "move" and i + 1 < len(steps)
                and steps[i + 1]["type"] == "clamp" and steps[i + 1]["action"] == "close")

    print(f"PLAN: {len(steps)} steps, robot={robot}, source={plan.get('source','?')}")
    print(f"refs: {refs_path} ({'recording' if args.record_refs else f'{len(refs)} loaded'})\n")
    ri = 0
    for i, s in enumerate(steps):
        if s["type"] == "clamp":
            print(f"  [{i:2d}] CLAMP {s['action'].upper()}")
        else:
            q = np.array(s["q"])
            T = K.fk_flange(q); tip, _ = K.fk_needle_tip(q)
            tag = ""
            if is_regrip(i):
                have = "ref" if ri in refs else "no-ref"
                tag = f"   <- REGRIP #{ri} (vision-correct, {have})"
                ri += 1
            print(f"  [{i:2d}] MOVE  flange=({T[0,3]:+.3f},{T[1,3]:+.3f},{T[2,3]:+.3f})"
                  f"  q(deg)=[{', '.join(f'{x:+.0f}' for x in np.degrees(q))}]{tag}")

    if not (args.execute or args.record_refs):
        print("\nDRY RUN only. Re-run with --execute (or --record-refs --execute) to drive.")
        return

    # ---- live run ----
    if fs is None:
        sys.exit("fruit_scan/opencv unavailable — install the rig venv (redis numpy "
                 "opencv-python ultralytics) to run with --execute.")
    import redis
    r = redis.Redis(host=args.host, port=args.port); r.ping()
    K_ = keys(robot)
    global keys_cache, classes_cache
    keys_cache = K_
    classes_cache = set(fs.DEFAULT_FRUIT_CLASSES)

    loaded = fs.load_calibration()
    if loaded is not None:
        fs.T_FLANGE_CAM = loaded
        print("[calib] loaded T_FLANGE_CAM from calibration file.")
    else:
        print("[calib] WARNING: no calibration file; world coords will be wrong.")
    intr = fs.read_intrinsics(r)
    model = fs.load_model(args.model)
    tracker = fs.FruitTracker()
    tracker.lock_conf = 2.0          # don't latch: the apple may shift between regrip sites
    if args.fruit_color:
        tracker.h_center = fs.COLOR_HUES[args.fruit_color]

    cur = r.get(K_["qsens"])
    if cur is None:
        sys.exit("No joint stream — is OpenSai + the driver running?")
    q_now = np.array(json.loads(cur))

    # PRE-FLIGHT: there is NO safe-home move — the first step drives straight to
    # the demo's pose 0 via slow joint interpolation (no collision check). Show
    # how big that first move is so you can abort if the arm is parked far away.
    first = next((s for s in steps if s["type"] == "move"), None)
    if first is not None:
        q0 = np.array(first["q"])
        dq = np.degrees(np.linalg.norm(q0 - q_now))
        Tn, T0 = K.fk_flange(q_now), K.fk_flange(q0)
        print(f"\n[pre-flight] arm is {dq:.0f} deg from the plan's first pose.")
        print(f"   flange now    ({Tn[0,3]:+.3f},{Tn[1,3]:+.3f},{Tn[2,3]:+.3f})")
        print(f"   flange pose0  ({T0[0,3]:+.3f},{T0[1,3]:+.3f},{T0[2,3]:+.3f})  <- first move goes here")
        if dq > 45:
            print("   WARNING: large first move — consider freedriving the arm near "
                  "pose 0 first, and keep a hand on the e-stop.")
    if input("Execute on the REAL robot? e-stop ready. [y/N]: ").strip().lower() not in ("y", "yes"):
        print("aborted."); return

    r.set(K_["active"], "joint_controller"); time.sleep(0.5)
    move_time = args.move_time / max(args.speed, 0.05)
    new_refs = {}
    ri = 0
    for i, s in enumerate(steps):
        if s["type"] == "clamp":
            r.set(K_["grip"], "g" if s["action"] == "close" else "o")
            print(f"  [{i:2d}] clamp {s['action']}"); time.sleep(2.0)
            continue
        q_demo = np.array(s["q"])
        if not is_regrip(i):
            print(f"  [{i:2d}] move (taught)")
            goto(r, K_, q_demo, move_time)
            continue
        # regrip: go to taught pose, STOP, detect needle world
        print(f"  [{i:2d}] REGRIP #{ri}: move to taught pose, then look")
        goto(r, K_, q_demo, move_time)
        N_now = detect_needle_world(r, model, tracker, intr, args.dark_v, args.conf,
                                    show=args.show, win_label=f"regrip #{ri}")
        if args.record_refs:
            if N_now is not None:
                new_refs[ri] = N_now.tolist()
                print(f"        recorded needle world {np.round(N_now,4)}")
            else:
                print("        needle NOT seen here (blind/occluded) — no ref")
        else:
            N_ref = refs.get(ri)
            if N_now is not None and N_ref is not None:
                delta = N_now - N_ref
                dmag = float(np.linalg.norm(delta))
                if dmag > args.max_correction:                 # gate: implausible jump
                    print(f"        REJECT correction {dmag*1000:.0f}mm > "
                          f"{args.max_correction*1000:.0f}mm cap — taught pose "
                          "(likely a vision glitch)")
                else:
                    T = K.fk_flange(q_demo); T[:3, 3] = T[:3, 3] + delta
                    q_corr, ok, pe, oe = K.ik_flange(T, q_demo)
                    if ok:
                        print(f"        VISION-CORRECT by {np.round(delta*1000,1)}mm "
                              f"(IK pe={pe*1000:.1f}mm)")
                        goto(r, K_, q_corr, move_time)
                    else:
                        print(f"        IK failed (pe={pe*1000:.1f}mm) — taught pose")
            else:
                why = "needle not seen" if N_now is None else "no ref recorded"
                print(f"        blind fallback ({why}) — taught pose")
        ri += 1

    if args.record_refs:
        with open(refs_path, "w") as f:
            json.dump({"plan": os.path.basename(args.plan), "refs": new_refs}, f, indent=2)
        print(f"\n[refs] wrote {len(new_refs)} needle refs -> {refs_path}")
    print("\n[done] plan executed.")


if __name__ == "__main__":
    main()
