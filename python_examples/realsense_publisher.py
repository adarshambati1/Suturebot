"""RealSense publisher (PC side) -- color + textured 3D point cloud.

Reads from the D405 mounted on the arm and writes to two Redis keys that the
Mac-side viewers (vr_teleop/test_harness/realsense_viewer.py and the
_rs3d_ teleop variants) can pull through the existing SSH tunnel:

  * Color stream (existing): JPEG-encoded BGR -> key  suturebot::realsense::color
  * 3D point-cloud render:    JPEG of a rendered    -> key  suturebot::realsense::pointcloud
    point cloud, textured with the color image, viewed from a tunable angle.

Both messages are 8-byte LE float64 timestamp + JPEG bytes. The 3D render is
a painter's-algorithm rasterization (no OpenGL needed) at a configurable
resolution and tilt angle. Pass --no-3d to skip it if you only want color.

Install (one-time, on the PC):
    pip install pyrealsense2 opencv-python redis numpy

Usage (on the PC):
    python python_examples/realsense_publisher.py
    # smaller / faster render:
    python python_examples/realsense_publisher.py --render-width 480 --render-height 360 \\
        --max-points 15000 --tilt-deg 25
"""

from __future__ import annotations

import argparse
import math
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import redis


DEFAULT_COLOR_KEY = "suturebot::realsense::color"
DEFAULT_3D_KEY    = "suturebot::realsense::pointcloud"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Redis host (default: localhost)")
    p.add_argument("--port", type=int, default=6379, help="Redis port (default: 6379)")
    p.add_argument("--key", default=DEFAULT_COLOR_KEY,
                   help=f"Redis key for color JPEG (default: {DEFAULT_COLOR_KEY})")
    p.add_argument("--width", type=int, default=640, help="color frame width (default 640)")
    p.add_argument("--height", type=int, default=480, help="color frame height (default 480)")
    p.add_argument("--fps", type=int, default=30, help="color frame rate (default 30)")
    p.add_argument("--jpeg-quality", type=int, default=80,
                   help="JPEG quality 1-100 (default 80)")
    p.add_argument("--device-serial", default=None,
                   help="RealSense serial number (rs-enumerate-devices). Default = first device.")
    # 3D point cloud render options ----------------------------------------------
    p.add_argument("--no-3d", action="store_true",
                   help="Skip the 3D point cloud render (publish color only).")
    p.add_argument("--threed-key", default=DEFAULT_3D_KEY,
                   help=f"Redis key for the rendered 3D point cloud JPEG "
                        f"(default: {DEFAULT_3D_KEY})")
    p.add_argument("--depth-width", type=int, default=640, help="depth width (default 640)")
    p.add_argument("--depth-height", type=int, default=480, help="depth height (default 480)")
    p.add_argument("--render-width", type=int, default=640,
                   help="3D render output width (default 640)")
    p.add_argument("--render-height", type=int, default=480,
                   help="3D render output height (default 480)")
    p.add_argument("--max-points", type=int, default=25000,
                   help="downsample point cloud to at most this many points (default 25000)")
    p.add_argument("--tilt-deg", type=float, default=18.0,
                   help="rotate the view around Y by this many degrees for a 3D-looking "
                        "parallax (default 18). 0 = head-on, larger = more side angle.")
    p.add_argument("--z-near", type=float, default=0.05,
                   help="discard points closer than this in meters (default 0.05)")
    p.add_argument("--z-far", type=float, default=2.0,
                   help="discard points farther than this in meters (default 2.0)")
    p.add_argument("--point-radius", type=int, default=1,
                   help="point splat radius in px (1 = single pixels, 2-3 = thicker)")
    p.add_argument("--threed-jpeg-quality", type=int, default=70,
                   help="JPEG quality for the 3D render (default 70)")
    return p.parse_args()


def render_pointcloud(
    verts: np.ndarray,
    colors: np.ndarray,
    tilt_deg: float,
    view_w: int,
    view_h: int,
    point_radius: int = 1,
) -> np.ndarray:
    """Rasterize a textured point cloud to a (view_h, view_w, 3) BGR image.

    Camera looks down -Z (RealSense convention: +Z into scene). We rotate the
    cloud around Y by `tilt_deg` to give a parallax view, then do a pinhole
    projection with painter's-algorithm depth sorting (back-to-front), and
    write each point's color into the canvas.
    """
    if len(verts) == 0:
        return np.zeros((view_h, view_w, 3), dtype=np.uint8)

    ang = math.radians(tilt_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    Ry = np.array([[ca, 0.0, sa],
                   [0.0, 1.0, 0.0],
                   [-sa, 0.0, ca]], dtype=np.float32)
    v_view = verts @ Ry.T

    # Drop points behind the virtual camera after the rotation.
    z = v_view[:, 2]
    in_front = z > 0.05
    v_view = v_view[in_front]
    cc = colors[in_front]
    if len(v_view) == 0:
        return np.zeros((view_h, view_w, 3), dtype=np.uint8)

    # Pinhole projection. Focal heuristic: a 1 m wide patch at z=1 m fills ~70% of view.
    focal = view_h * 1.2
    u = (focal * v_view[:, 0] / v_view[:, 2] + view_w * 0.5).astype(np.int32)
    v = (focal * v_view[:, 1] / v_view[:, 2] + view_h * 0.5).astype(np.int32)

    in_bounds = (u >= 0) & (u < view_w) & (v >= 0) & (v < view_h)
    u, v = u[in_bounds], v[in_bounds]
    cc = cc[in_bounds]
    z = v_view[in_bounds, 2]

    # Painter's: sort back-to-front so nearer points overwrite farther ones.
    order = np.argsort(-z)
    u, v, cc = u[order], v[order], cc[order]

    canvas = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    if point_radius <= 1:
        canvas[v, u] = cc
    else:
        # Thicker splats via dilation per offset (still fully vectorized).
        for dy in range(-point_radius + 1, point_radius):
            for dx in range(-point_radius + 1, point_radius):
                uu = u + dx
                vv = v + dy
                m = (uu >= 0) & (uu < view_w) & (vv >= 0) & (vv < view_h)
                canvas[vv[m], uu[m]] = cc[m]

    # Light HUD: shows the tilt in the corner so the viewer knows the angle.
    cv2.putText(canvas, f"tilt {tilt_deg:+.0f}deg  {len(u)} pts", (10, view_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.device_serial:
        config.enable_device(args.device_serial)
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    if not args.no_3d:
        config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height,
                             rs.format.z16, args.fps)
    profile = pipeline.start(config)
    dev = profile.get_device()
    print(f"Streaming {dev.get_info(rs.camera_info.name)} "
          f"(SN {dev.get_info(rs.camera_info.serial_number)}) at "
          f"{args.width}x{args.height}@{args.fps}fps "
          f"({'color only' if args.no_3d else 'color + depth'})")

    align = rs.align(rs.stream.color) if not args.no_3d else None
    pc = rs.pointcloud() if not args.no_3d else None

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()
    print(f"Publishing color JPEG to Redis {args.host}:{args.port} key '{args.key}'")
    if not args.no_3d:
        print(f"Publishing 3D pointcloud JPEG to key '{args.threed_key}' "
              f"({args.render_width}x{args.render_height}, "
              f"max {args.max_points} pts, tilt {args.tilt_deg:.0f}deg)")
    print("Ctrl-C to stop.")

    color_encode = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    threed_encode = [int(cv2.IMWRITE_JPEG_QUALITY), args.threed_jpeg_quality]
    n = 0
    t0 = time.monotonic()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            ok, buf = cv2.imencode(".jpg", img, color_encode)
            if not ok:
                continue
            ts = time.time()
            header = struct.pack("<d", ts)
            r.set(args.key, header + buf.tobytes())

            if pc is not None:
                depth = frames.get_depth_frame()
                if depth:
                    pc.map_to(color)
                    points = pc.calculate(depth)
                    verts = (np.asanyarray(points.get_vertices())
                             .view(np.float32).reshape(-1, 3))
                    texc = (np.asanyarray(points.get_texture_coordinates())
                            .view(np.float32).reshape(-1, 2))

                    h, w = img.shape[:2]
                    u_tex = np.clip((texc[:, 0] * w).astype(np.int32), 0, w - 1)
                    v_tex = np.clip((texc[:, 1] * h).astype(np.int32), 0, h - 1)
                    colors_bgr = img[v_tex, u_tex]

                    mask = ((verts[:, 2] > args.z_near) &
                            (verts[:, 2] < args.z_far))
                    verts = verts[mask]
                    colors_bgr = colors_bgr[mask]

                    if len(verts) > args.max_points:
                        idx = np.random.choice(len(verts), args.max_points, replace=False)
                        verts = verts[idx]
                        colors_bgr = colors_bgr[idx]

                    rendered = render_pointcloud(
                        verts, colors_bgr, args.tilt_deg,
                        args.render_width, args.render_height,
                        args.point_radius,
                    )
                    ok2, buf2 = cv2.imencode(".jpg", rendered, threed_encode)
                    if ok2:
                        r.set(args.threed_key, struct.pack("<d", ts) + buf2.tobytes())

            n += 1
            if n % args.fps == 0:
                dt = time.monotonic() - t0
                msg = (f"  {n} frames, {n / dt:5.1f} fps, "
                       f"color {len(buf)/1024:5.1f} KB")
                if not args.no_3d and depth:
                    msg += f", 3D {len(buf2)/1024:5.1f} KB"
                print(msg)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
