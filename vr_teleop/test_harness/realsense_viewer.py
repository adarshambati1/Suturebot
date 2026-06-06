"""RealSense viewer (Mac side) -- pull JPEG frames from Redis through the SSH tunnel.

Companion to python_examples/realsense_publisher.py, which runs on the PC.
Reads the latest JPEG frame from a Redis key and displays it with OpenCV.
Uses the existing SSH tunnel (default --port 6380), so no new networking
setup is needed.

The publisher writes an 8-byte timestamp header before the JPEG; this viewer
parses it to skip duplicate frames and show end-to-end latency in the HUD.

Install (one-time, on the Mac, in the same venv as the teleop):
    pip install opencv-python redis numpy

Usage (Mac side, with the SSH tunnel already running):
    python vr_teleop/test_harness/realsense_viewer.py --port 6380
    # Quit with 'q' or close the window.
"""

from __future__ import annotations

import argparse
import struct
import time

import cv2
import numpy as np
import redis


DEFAULT_KEY = "suturebot::realsense::color"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="Redis host (default: 127.0.0.1, the tunnel endpoint)")
    p.add_argument("--port", type=int, default=6380,
                   help="Redis port (default: 6380 = tunnel)")
    p.add_argument("--key", default=DEFAULT_KEY,
                   help=f"Redis key the publisher writes to (default: {DEFAULT_KEY})")
    p.add_argument("--poll-hz", type=float, default=60.0,
                   help="how often to check Redis for a new frame (default 60)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r = redis.Redis(host=args.host, port=args.port)
    r.ping()

    window = "RealSense (live)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    print(f"Reading {args.key} from {args.host}:{args.port}. Press 'q' to quit.")
    poll_period = 1.0 / max(args.poll_hz, 1.0)
    last_pub_ts = -1.0
    frames_seen = 0
    fps = 0.0
    t_fps = time.monotonic()
    waiting_logged = False

    try:
        while True:
            data = r.get(args.key)
            if data is None or len(data) < 8:
                if not waiting_logged:
                    print("  no frames yet -- is realsense_publisher.py running on the PC?")
                    waiting_logged = True
                cv2.waitKey(int(poll_period * 1000))
                continue
            waiting_logged = False

            pub_ts = struct.unpack("<d", data[:8])[0]
            if pub_ts == last_pub_ts:
                # Same frame as last poll -- skip the decode/display work.
                key = cv2.waitKey(int(poll_period * 1000)) & 0xFF
                if key == ord('q'):
                    break
                continue
            last_pub_ts = pub_ts

            arr = np.frombuffer(data[8:], dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            frames_seen += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                fps = frames_seen / (now - t_fps)
                frames_seen = 0
                t_fps = now

            lag_ms = (time.time() - pub_ts) * 1000.0
            hud = f"{fps:4.1f} fps   lag {lag_ms:4.0f} ms"
            cv2.putText(img, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
