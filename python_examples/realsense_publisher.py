"""RealSense color-stream publisher (PC side).

Grabs frames from the D405 mounted on the arm, JPEG-encodes them, and writes
them to a Redis key that the Mac-side viewer (vr_teleop/test_harness/
realsense_viewer.py or python_examples/realsense_viewer.py) can pull through
the existing SSH tunnel. Reuses the Redis you already have running on the PC
for OpenSai -- no new ports, no extra tunnel needed.

Defaults: 640x480 color @ 30 fps, JPEG quality 80, Redis key
"suturebot::realsense::color". A small 8-byte timestamp header is prepended
so the viewer can detect new frames and show lag.

Install (one-time, on the PC):
    pip install pyrealsense2 opencv-python redis numpy

Usage (on the PC):
    python python_examples/realsense_publisher.py
    # higher framerate / smaller frame:
    python python_examples/realsense_publisher.py --width 424 --height 240 --fps 60
"""

from __future__ import annotations

import argparse
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import redis


DEFAULT_KEY = "suturebot::realsense::color"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Redis host (default: localhost)")
    p.add_argument("--port", type=int, default=6379, help="Redis port (default: 6379)")
    p.add_argument("--key", default=DEFAULT_KEY,
                   help=f"Redis key for the JPEG stream (default: {DEFAULT_KEY})")
    p.add_argument("--width", type=int, default=640, help="frame width (default 640)")
    p.add_argument("--height", type=int, default=480, help="frame height (default 480)")
    p.add_argument("--fps", type=int, default=30, help="frame rate (default 30)")
    p.add_argument("--jpeg-quality", type=int, default=80,
                   help="JPEG quality 1-100 (default 80). Lower = smaller frames, faster.")
    p.add_argument("--device-serial", default=None,
                   help="RealSense serial number to bind to (rs-enumerate-devices). "
                        "Optional; the first found device is used by default.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.device_serial:
        config.enable_device(args.device_serial)
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    dev = profile.get_device()
    print(f"Streaming {dev.get_info(rs.camera_info.name)} "
          f"(SN {dev.get_info(rs.camera_info.serial_number)}) at "
          f"{args.width}x{args.height}@{args.fps}fps")

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()
    print(f"Publishing JPEG to Redis {args.host}:{args.port} key '{args.key}'")
    print("Ctrl-C to stop.")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    n = 0
    t0 = time.monotonic()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            ok, buf = cv2.imencode(".jpg", img, encode_params)
            if not ok:
                continue
            # 8-byte LE float64 timestamp header so viewers can detect new frames
            # and show end-to-end latency.
            header = struct.pack("<d", time.time())
            r.set(args.key, header + buf.tobytes())
            n += 1
            if n % args.fps == 0:
                dt = time.monotonic() - t0
                print(f"  {n} frames, {n / dt:5.1f} fps, "
                      f"last frame {len(buf)/1024:5.1f} KB")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
