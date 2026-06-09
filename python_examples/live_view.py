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

import cv2
import numpy as np
import redis

import fruit_scan as fs


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
    p.add_argument("--no-needle", action="store_true", help="don't draw the projected needle tip")
    return p.parse_args()


def needle_tip_pixel(intr):
    """Project the (fixed-in-flange) needle tip into the camera image using the
    loaded calibration. Returns (u, v) or None."""
    T = fs.T_FLANGE_CAM
    R_fc, t_fc = T[:3, :3], T[:3, 3]
    tip_cam = R_fc.T @ (fs.NEEDLE_TIP_OFFSET - t_fc)   # both fixed in flange
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
            nodes = []
            if strip and strip.get("found"):
                nodes = fs.plan_stitch_nodes(strip["p1"], strip["p2"], 3)
            out = fs.annotate_stitch_plan(bgr, fruit, strip, nodes)
            hud = (f"{fruit['label']} {fruit['conf']:.2f}" if fruit is not None else "no fruit")
        else:
            out = bgr.copy()
            hud = "raw"

        # projected needle tip (red ✕, from calibration) + DETECTED magenta tip
        # (magenta ○, from vision). If they coincide, calibration + needle offset
        # are good. The detected one is the closed-loop re-grip signal.
        if not args.no_needle and intr is not None:
            px = needle_tip_pixel(intr)
            if px is not None and 0 <= px[0] < w and 0 <= px[1] < h:
                cv2.drawMarker(out, px, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.putText(out, "proj", (px[0] + 8, px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if not args.raw:
            det, area = fs.detect_needle_pixel(bgr)
            if det is not None:
                cv2.circle(out, det, 8, (255, 0, 255), 2)
                cv2.putText(out, "needle", (det[0] + 10, det[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        age = (time.time() - ts) if ts else None
        if age is not None:
            hud += f"   age {age:4.2f}s"
        cv2.putText(out, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, out)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
