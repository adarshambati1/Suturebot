"""Turn a hand-guided demo (.npz) into a clean autonomous PLAN.

A teach-by-demonstration recording (freedrive_record.py / record_demo_redis.py)
is hundreds of seconds of jittery 100 Hz samples. But the *intent* is a short
sequence of: move to a pose, hold, toggle the clamp, move again. This extracts
that sequence so autonomous mode replays clean point-to-point moves instead of
the raw tremor.

How it works:
  1. joint speed over the demo -> stretches below --vel-thresh held for
     >--min-hold seconds are "held key-poses" (where your hand paused).
  2. clamp open/close transitions are events too.
  3. consecutive key-poses within --dedup-deg of each other collapse into one
     (your hand drifting while holding).
  4. result = an ordered plan of MOVE (target joint config) + CLAMP actions.

Saves plan to log_files/demos/<demo>_plan.json:
  {"robot": "...", "steps": [{"type":"move","q":[...7 rad...],"t":...},
                             {"type":"clamp","action":"close"|"open","t":...}, ...]}

    python3 python_examples/analyze_demo.py log_files/demos/demo_2.npz
    python3 python_examples/analyze_demo.py demo_2.npz --dedup-deg 10 --min-hold 0.8
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np


def extract_plan(t, q, g, vel_thresh=0.06, min_hold=0.6, dedup_deg=6.0, smooth=25):
    g = np.asarray(g).astype(int)
    dt = np.gradient(t)
    vel = np.linalg.norm(np.gradient(q, axis=0) / dt[:, None], axis=1)
    vels = np.convolve(vel, np.ones(smooth) / smooth, mode="same")
    held = vels < vel_thresh

    # held key-poses
    kf, i, n = [], 0, len(t)
    while i < n:
        if held[i]:
            j = i
            while j < n and held[j]:
                j += 1
            if t[j - 1] - t[i] > min_hold:
                kf.append((i + j) // 2)
            i = j
        else:
            i += 1
    trans = [k + 1 for k in np.where(np.diff(g) != 0)[0]]

    events = sorted([(t[k], "move", k) for k in kf] +
                    [(t[k], "clamp", k) for k in trans])

    steps, last_pose = [], None
    for ti, kind, idx in events:
        if kind == "move":
            cfg = q[idx]
            if last_pose is not None and \
               np.degrees(np.linalg.norm(cfg - last_pose)) < dedup_deg:
                continue
            last_pose = cfg
            steps.append({"type": "move", "q": cfg.tolist(), "t": float(ti)})
        else:
            steps.append({"type": "clamp",
                          "action": "close" if g[idx] > 0 else "open",
                          "t": float(ti)})
            last_pose = None
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo", help="path to the demo .npz")
    ap.add_argument("--vel-thresh", type=float, default=0.06,
                    help="joint speed (rad/s) below which the arm counts as held")
    ap.add_argument("--min-hold", type=float, default=0.6,
                    help="min seconds held to count as a key-pose")
    ap.add_argument("--dedup-deg", type=float, default=6.0,
                    help="collapse consecutive key-poses within this joint distance (deg)")
    ap.add_argument("--out", default=None, help="output plan json (default <demo>_plan.json)")
    args = ap.parse_args()

    d = np.load(args.demo, allow_pickle=True)
    t, q, g = d["times"], d["q"], d["gripper"]
    robot = str(d["robot"]) if "robot" in d.files else "Titania"
    steps = extract_plan(t, q, g, args.vel_thresh, args.min_hold, args.dedup_deg)

    print(f"demo: {t[-1]:.0f}s, {len(t)} samples -> {len(steps)} plan steps\n")
    for i, s in enumerate(steps):
        if s["type"] == "clamp":
            tag = "CLOSE (grab needle)" if s["action"] == "close" else "OPEN (release)"
            print(f"  [{i:2d}] CLAMP {tag}   @t={s['t']:.0f}s")
        else:
            print(f"  [{i:2d}] MOVE  q(deg)=[{', '.join(f'{x:+.0f}' for x in np.degrees(s['q']))}]   @t={s['t']:.0f}s")

    out = args.out or (os.path.splitext(args.demo)[0] + "_plan.json")
    with open(out, "w") as f:
        json.dump({"robot": robot, "source": os.path.basename(args.demo),
                   "steps": steps}, f, indent=2)
    print(f"\nsaved plan -> {out}")
    print("This is the clean autonomous sequence (smooth moves + clamp toggles),")
    print("not the raw jittery recording.")


if __name__ == "__main__":
    main()
