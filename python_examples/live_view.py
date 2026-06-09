"""Always-on live camera view for testing — run in its own terminal.

Continuously shows the wrist camera (suturebot::realsense::color) with live
overlays so you can watch what the system sees while ANY other client runs
(fruit_scan --plan-stitch, --calibrate, the stitch player, etc). It's a separate
process reading the same Redis feed, so it never interferes and survives the
robot moving.

Overlays (all optional):
  * fruit box + label (YOLO + colour tracker)
  * blue cut line
  * projected NEEDLE TIP (from the calibration) — where the needle is in view
  * frame-age HUD (climbs if the publisher stalls)

    python3 python_examples/live_view.py                      # full overlay
    python3 python_examples/live_view.py --fruit-color redorange
    python3 python_examples/live_view.py --raw                # just the stream, no detection

Press q or Esc to quit.
"""

from __future__ import annotations
import argparse
import time
from collections import deque

import cv2
import numpy as np
import redis

import fruit_scan as fs


def needle_consensus(hist, min_frac=0.5):
    """Most-consistent needle over a rolling window: median endpoints of the
    frames that found it (normalised order). Returns (p1, p2) or None."""
    found = [x for x in hist if x is not None]
    if len(found) < max(2, int(len(hist) * min_frac)):
        return None
    P1, P2 = [], []
    for a, b in found:
        if (a[0], a[1]) > (b[0], b[1]):
            a, b = b, a
        P1.append(a); P2.append(b)
    return np.median(P1, axis=0), np.median(P2, axis=0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--robot", default="Titania")
    p.add_argument("--raw", action="store_true", help="just the stream, no detection")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--classes", nargs="+", default=list(fs.DEFAULT_FRUIT_CLASSES))
    p.add_argument("--fruit-color", choices=list(fs.COLOR_HUES.keys()), default=None)
    p.add_argument("--show-proj", action="store_true",
                   help="draw the calibration-projected needle tip (red x); off by "
                        "default since it's unreliable across re-grips")
    p.add_argument("--needle-debug", action="store_true",
                   help="show the dark mask the needle detector sees (to tune --dark-v)")
    p.add_argument("--dark-v", type=int, default=160,
                   help="needle darkness threshold (lower = only very dark pixels; "
                        "raise until the needle shows but the apple doesn't)")
    p.add_argument("--needle-window", type=int, default=7,
                   help="frames to aggregate; the needle shown is the consensus "
                        "(median) over this window (default 7)")
    return p.parse_args()


def needle_tip_pixel(intr):
    """Project the (fixed-in-flange) needle tip into the camera image using the
    loaded calibration. Returns (u, v) or None."""
    T = fs.T_FLANGE_CAM
    R_fc, t_fc = T[:3, :3], T[:3, 3]
    tip_cam = R_fc.T @ (fs.EE_OFFSET_FLANGE - t_fc)   # both fixed in flange
    if tip_cam[2] <= 1e-4:
        return None
    u = intr["fx"] * tip_cam[0] / tip_cam[2] + intr["ppx"]
    v = intr["fy"] * tip_cam[1] / tip_cam[2] + intr["ppy"]
    return int(round(u)), int(round(v))


def main():
    args = parse_args()
    r = redis.Redis(host=args.host, port=args.port)
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        raise SystemExit(f"Cannot reach Redis at {args.host}:{args.port}.")

    loaded = fs.load_calibration()
    if loaded is not None:
        fs.T_FLANGE_CAM = loaded
    intr = fs.read_intrinsics(r)

    model = tracker = None
    if not args.raw:
        model = fs.load_model(args.model)
        tracker = fs.FruitTracker()
        if args.fruit_color:
            tracker.h_center = fs.COLOR_HUES[args.fruit_color]

    win = "live view"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[live] press q/Esc to quit.")
    classes = set(args.classes)
    needle_hist = deque(maxlen=args.needle_window)
    while True:
        bgr, ts = fs.read_color(r)
        if bgr is None:
            time.sleep(0.05)
            continue
        h, w = bgr.shape[:2]
        out = bgr

        if not args.raw:
            roi = (0, 0, w, h)
            fruit = tracker.detect(model, bgr, classes, args.conf, roi)
            strip = fs.detect_blue_line(bgr, fruit["box"], fs.BLUE_HSV_LO, fs.BLUE_HSV_HI) \
                if fruit is not None else None
            nodes = fs.plan_stitch_nodes(strip["p1"], strip["p2"], 3) \
                if strip and strip.get("found") else []
            out = fs.annotate_stitch_plan(bgr, fruit, strip, nodes)   # box+label+cut+nodes
            nd, dmask = ({"found": False}, None)
            if fruit is not None:
                nd, dmask = fs.detect_needle_darkline(
                    bgr, line=strip, box=fruit["box"], dark_v=args.dark_v, return_mask=True)
            needle_hist.append((np.array(nd["p1"]), np.array(nd["p2"]))
                               if nd.get("found") else None)
            cons = needle_consensus(needle_hist)
            if cons is not None:                 # stable, most-common needle
                cv2.line(out, tuple(cons[0].astype(int)), tuple(cons[1].astype(int)),
                         (0, 255, 0), 2)
            if args.needle_debug and dmask is not None:
                # show the dark mask (what the detector sees) in the corner;
                # cyan = candidate blobs, green = chosen needle.
                dm = dmask if dmask.ndim == 3 else cv2.cvtColor(dmask, cv2.COLOR_GRAY2BGR)
                dh, dw = dm.shape[:2]
                sc = min(220 / max(dw, 1), 220 / max(dh, 1), 1.0)
                dm = cv2.resize(dm, (int(dw * sc), int(dh * sc)))
                out[h - dm.shape[0]:h, w - dm.shape[1]:w] = dm
            hud = (f"{fruit['label']} {fruit['conf']:.2f}" if fruit is not None else "no fruit")
            hud += f"  cut:{'Y' if strip and strip.get('found') else 'n'}"
            hud += f"  needle:{'Y' if cons is not None else 'n'}"
        else:
            out = bgr.copy()
            hud = "raw"

        # optional calibration projection (red ✕) — off by default (unreliable
        # across re-grips); enable with --show-proj.
        if args.show_proj and intr is not None:
            px = needle_tip_pixel(intr)
            if px is not None and 0 <= px[0] < w and 0 <= px[1] < h:
                cv2.drawMarker(out, px, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 1)

        age = (time.time() - ts) if ts else None
        if age is not None:
            hud += f"  age {age:4.2f}s"
        # HUD bar at the very top, away from the fruit
        cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, out)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
