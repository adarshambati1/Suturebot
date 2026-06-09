"""needle_check.py — one-shot test of the needle detector (Phase-0 step 1).

Grabs a single frame (live wrist camera over Redis, or a still via --image),
runs the SAME detector live_view/fruit_scan use (detect_needle_darkline + the
"tip = endpoint nearer the apple centre" rule), draws the green tip dot + the
debug mask, saves the annotated result, and prints a pass/fail readout. No
robot, no motion — pure perception check you can re-run while tuning --dark-v.

    python3 python_examples/needle_check.py                       # live feed
    python3 python_examples/needle_check.py --image shot.png      # still image
    python3 python_examples/needle_check.py --dark-v 140          # retune threshold

Stage the frame like a real top-side regrip: apple in view, straight needle
PIERCED through the apple and protruding on the visible side, tip toward centre.
"""

from __future__ import annotations
import argparse
import os
import sys

import cv2
import numpy as np
import redis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fruit_scan as fs


def needle_tip(nd, fruit):
    """Tip = the detected endpoint CLOSER to the apple centre (pierce is upward).
    Same rule as live_view.py."""
    if not nd.get("found"):
        return None
    p1, p2 = np.array(nd["p1"], float), np.array(nd["p2"], float)
    ctr = np.array([fruit["cx"], fruit["cy"]], float) if fruit else (p1 + p2) / 2
    return p1 if np.linalg.norm(p1 - ctr) < np.linalg.norm(p2 - ctr) else p2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=None,
                   help="run on a still image instead of the live Redis feed")
    p.add_argument("--dark-v", type=int, default=160,
                   help="needle darkness threshold (lower = stricter dark; default 160)")
    p.add_argument("--conf", type=float, default=0.35, help="min YOLO confidence")
    p.add_argument("--model", default="yolov8n.pt", help="ultralytics weights")
    p.add_argument("--classes", nargs="+", default=list(fs.DEFAULT_FRUIT_CLASSES),
                   help="fruit classes to accept")
    p.add_argument("--fruit-color", default=None, help="seed the colour tracker")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--out", default="needle_check_out.png",
                   help="annotated output image path")
    p.add_argument("--save-frame", default="needle_frame.png",
                   help="where to save the raw captured frame (live mode only)")
    p.add_argument("--no-show", action="store_true", help="don't pop a window")
    return p.parse_args()


def main():
    args = parse_args()

    # --- get one frame ---------------------------------------------------------
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"could not read image: {args.image}")
        print(f"[in] image {args.image}  {bgr.shape[1]}x{bgr.shape[0]}")
    else:
        r = redis.Redis(host=args.host, port=args.port)
        try:
            r.ping()
        except redis.exceptions.ConnectionError:
            raise SystemExit(f"cannot reach Redis at {args.host}:{args.port} "
                             "(is realsense_publisher.py running?)")
        bgr, ts = fs.read_color(r)
        if bgr is None:
            raise SystemExit("no color frame on suturebot::realsense::color "
                             "(is realsense_publisher.py running?)")
        cv2.imwrite(args.save_frame, bgr)
        print(f"[in] live frame  {bgr.shape[1]}x{bgr.shape[0]}  -> saved {args.save_frame}")

    h, w = bgr.shape[:2]

    # --- detect: apple -> cut -> needle ---------------------------------------
    model = fs.load_model(args.model)
    tracker = fs.FruitTracker()
    if args.fruit_color:
        tracker.h_center = fs.COLOR_HUES[args.fruit_color]
    fruit = tracker.detect(model, bgr, set(args.classes), args.conf, (0, 0, w, h))
    strip = (fs.detect_blue_line(bgr, fruit["box"], fs.BLUE_HSV_LO, fs.BLUE_HSV_HI)
             if fruit is not None else None)
    nodes = (fs.plan_stitch_nodes(strip["p1"], strip["p2"], 3)
             if strip and strip.get("found") else [])
    out = fs.annotate_stitch_plan(bgr, fruit, strip, nodes)

    nd, dmask = ({"found": False}, None)
    if fruit is not None:
        nd, dmask = fs.detect_needle_darkline(
            bgr, line=strip, box=fruit["box"], dark_v=args.dark_v, return_mask=True)
    tip = needle_tip(nd, fruit)

    # --- draw green tip dot + debug mask corner -------------------------------
    if tip is not None:
        cv2.circle(out, tuple(tip.astype(int)), 7, (0, 255, 0), -1)
        cv2.circle(out, tuple(tip.astype(int)), 9, (0, 255, 0), 2)
    if dmask is not None:
        dm = dmask if dmask.ndim == 3 else cv2.cvtColor(dmask, cv2.COLOR_GRAY2BGR)
        dh, dw = dm.shape[:2]
        sc = min(220 / max(dw, 1), 220 / max(dh, 1), 1.0)
        dm = cv2.resize(dm, (int(dw * sc), int(dh * sc)))
        out[h - dm.shape[0]:h, w - dm.shape[1]:w] = dm

    cv2.imwrite(args.out, out)

    # --- readout ---------------------------------------------------------------
    print("\n=== needle_check ===")
    if fruit is not None:
        print(f"apple   : YES  {fruit['label']} ({fruit['conf']:.2f})")
    else:
        print("apple   : NO   (detector needs the apple to bound its dark search)")
    print(f"cut line: {'YES' if strip and strip.get('found') else 'no'}")
    if nd.get("found"):
        p1, p2 = np.array(nd['p1']), np.array(nd['p2'])
        print(f"needle  : YES  endpoints {tuple(p1.astype(int))} - {tuple(p2.astype(int))}")
        print(f"tip     : {tuple(tip.astype(int))}  (green dot = endpoint nearer apple centre)")
    else:
        print("needle  : NO  -> try a lower --dark-v, or move the needle within ~20px of the apple")
    print(f"dark_v  : {args.dark_v}")
    print(f"[out] annotated -> {args.out}")
    print("\nPASS if: green blob in the corner mask is ON the needle, and the")
    print("green dot sits on the tip end nearest the apple centre.")

    if not args.no_show:
        cv2.namedWindow("needle_check", cv2.WINDOW_NORMAL)
        cv2.imshow("needle_check", out)
        print("\n(press any key in the window to close)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
