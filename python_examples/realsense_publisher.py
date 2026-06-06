"""RealSense publisher (PC side) -- color + realsense-viewer-style textured 3D.

Reads from the D405 and writes two Redis keys that the Mac-side viewers
(vr_teleop/test_harness/realsense_viewer.py and the _rs / _rs3d teleop
variants) can pull through the existing SSH tunnel:

  * Color stream:    JPEG-encoded BGR -> key  suturebot::realsense::color
  * 3D mesh render:  Open3D-rendered  -> key  suturebot::realsense::pointcloud
    textured triangulated depth mesh, lit and z-buffered. This is the
    realsense-viewer "3D" tab look: continuous colored surfaces, NOT a point
    splat. Tilt angle is configurable.

How the 3D render works:
  - Every depth pixel becomes a vertex (back-projected via the color
    camera intrinsics). Adjacent pixels are triangulated into two triangles
    per grid cell.
  - Triangles whose vertices straddle a depth discontinuity (edge longer
    than --max-edge in meters) are dropped, so distant background doesn't
    "skirt" through the foreground.
  - Vertex colors come from the aligned color image.
  - Open3D's OffscreenRenderer rasterizes the mesh with a virtual camera
    orbited by --tilt-deg around the scene center, gives proper z-buffering
    and (optional) lighting.

Install (one-time, on the PC):
    pip install pyrealsense2 opencv-python redis numpy open3d

Usage (on the PC):
    python python_examples/realsense_publisher.py
    # smaller / faster 3D panel:
    python python_examples/realsense_publisher.py --render-width 480 --render-height 360 --tilt-deg 25
"""

from __future__ import annotations

import argparse
import math
import struct
import time

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as o3d_rendering
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
    p.add_argument("--fps", type=int, default=30, help="capture frame rate (default 30)")
    p.add_argument("--jpeg-quality", type=int, default=80,
                   help="color JPEG quality 1-100 (default 80)")
    p.add_argument("--device-serial", default=None,
                   help="RealSense serial number; default = first device")
    # 3D mesh render options -----------------------------------------------------
    p.add_argument("--no-3d", action="store_true",
                   help="Skip the 3D mesh render (publish color only).")
    p.add_argument("--threed-key", default=DEFAULT_3D_KEY,
                   help=f"Redis key for 3D mesh render JPEG (default: {DEFAULT_3D_KEY})")
    p.add_argument("--depth-width", type=int, default=640, help="depth width (default 640)")
    p.add_argument("--depth-height", type=int, default=480, help="depth height (default 480)")
    p.add_argument("--render-width", type=int, default=640,
                   help="3D render output width (default 640)")
    p.add_argument("--render-height", type=int, default=480,
                   help="3D render output height (default 480)")
    p.add_argument("--tilt-deg", type=float, default=18.0,
                   help="orbit the view around Y by this many degrees for a 3D-looking "
                        "parallax (default 18). 0 = head-on, larger = more side angle.")
    p.add_argument("--z-near", type=float, default=0.05,
                   help="discard mesh vertices closer than this in meters (default 0.05)")
    p.add_argument("--z-far", type=float, default=2.0,
                   help="discard mesh vertices farther than this in meters (default 2.0)")
    p.add_argument("--max-edge", type=float, default=0.03,
                   help="drop triangles whose depth jump across an edge exceeds this in "
                        "meters (default 0.03). Lower = sharper silhouettes, more holes.")
    p.add_argument("--fov", type=float, default=55.0,
                   help="virtual camera field-of-view in degrees (default 55)")
    p.add_argument("--lit", action="store_true",
                   help="Use 'defaultLit' shader with a sun light (adds shading). "
                        "Off by default for realsense-viewer-style flat-textured look.")
    p.add_argument("--center-x", type=float, default=0.0,
                   help="x coord of the orbit center, in the camera frame (m)")
    p.add_argument("--center-y", type=float, default=0.0,
                   help="y coord of the orbit center, in the camera frame (m)")
    p.add_argument("--center-z", type=float, default=None,
                   help="z coord (depth) of the orbit center, in meters. "
                        "Default = auto from the median depth of the visible scene, "
                        "smoothed across frames so it stays stable.")
    p.add_argument("--orbit-dist", type=float, default=None,
                   help="distance from the orbit center back to the virtual camera (m). "
                        "Default = ~1.5x the scene's depth extent so the mesh fits.")
    p.add_argument("--auto-center-alpha", type=float, default=0.92,
                   help="EMA smoothing on the auto-computed scene center; higher = "
                        "more stable, slower to follow scene changes (default 0.92)")
    p.add_argument("--3d-rotate", dest="threed_rotate",
                   type=int, choices=[0, 90, 180, 270], default=180,
                   help="rotate the rendered 3D image by this many degrees clockwise "
                        "before publishing. Default 180 (camera mount is inverted). "
                        "Does NOT affect the 2D color stream.")
    p.add_argument("--threed-jpeg-quality", type=int, default=70,
                   help="JPEG quality for the 3D render (default 70)")
    return p.parse_args()


def depth_to_mesh(
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    z_near: float, z_far: float, max_edge: float,
) -> o3d.geometry.TriangleMesh:
    """Triangulate a depth image into a mesh in the camera frame. Texture
    coordinates are assigned so the aligned color image can later be applied
    as an albedo texture in the renderer (vertex colors are notoriously
    unreliable with Open3D's offscreen lit/unlit shaders; texture path works).

    Triangles whose vertices span large depth jumps (`max_edge`) are dropped
    so distant background doesn't drag a skirt through the foreground.
    """
    h, w = depth_m.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    z = depth_m
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)

    # Per-vertex texture coords in image-normalized space (0,0 = top-left).
    u_norm = (uu / max(w - 1, 1)).reshape(-1)
    v_norm = (vv / max(h - 1, 1)).reshape(-1)
    uv_per_vertex = np.stack([u_norm, v_norm], axis=-1)

    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    tl = idx[:-1, :-1].reshape(-1)
    tr = idx[:-1, 1:].reshape(-1)
    bl = idx[1:, :-1].reshape(-1)
    br = idx[1:, 1:].reshape(-1)
    tri1 = np.stack([tl, bl, br], axis=-1)
    tri2 = np.stack([tl, br, tr], axis=-1)
    triangles = np.vstack([tri1, tri2])

    z_flat = z.reshape(-1)
    z0 = z_flat[triangles[:, 0]]
    z1 = z_flat[triangles[:, 1]]
    z2 = z_flat[triangles[:, 2]]
    valid = ((z0 > z_near) & (z0 < z_far) &
             (z1 > z_near) & (z1 < z_far) &
             (z2 > z_near) & (z2 < z_far) &
             (np.abs(z0 - z1) < max_edge) &
             (np.abs(z1 - z2) < max_edge) &
             (np.abs(z0 - z2) < max_edge))
    triangles = triangles[valid]

    # triangle_uvs in Open3D is (3*num_triangles, 2): one UV per triangle corner.
    tri_uvs = uv_per_vertex[triangles].reshape(-1, 2)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
    # Lit shader needs normals; safe to compute for unlit too.
    mesh.compute_vertex_normals()
    return mesh


def make_renderer(width: int, height: int, lit: bool):
    """Create an Open3D offscreen renderer with a sensible background.
    Default is the unlit shader (flat textured = realsense-viewer look).
    --lit switches to defaultLit with a sun light for shading."""
    renderer = o3d_rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.05, 0.05, 0.05, 1.0])
    mat = o3d_rendering.MaterialRecord()
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    if lit:
        mat.shader = "defaultLit"
        try:
            renderer.scene.scene.set_sun_light([0.0, -0.3, -1.0], [1.0, 1.0, 1.0], 80000.0)
            renderer.scene.scene.enable_sun_light(True)
        except Exception:
            pass
    else:
        mat.shader = "defaultUnlit"
    return renderer, mat


def render_mesh_view(
    renderer,
    mat,
    mesh: o3d.geometry.TriangleMesh,
    color_rgb: np.ndarray,
    tilt_deg: float,
    fov_deg: float,
    center_xyz: np.ndarray,
    orbit_dist: float,
    render_w: int,
    render_h: int,
    rotate_deg: int = 0,
) -> np.ndarray:
    """Add `mesh` (with triangle_uvs) to the renderer's scene, set the color
    image as its albedo texture, and snap a BGR image from a camera orbited
    around a FIXED scene center. Fixed center = stable view across frames."""
    if np.asarray(mesh.triangles).shape[0] == 0:
        return np.zeros((render_h, render_w, 3), dtype=np.uint8)

    # Apply the latest color image as the mesh's albedo texture.
    mat.albedo_img = o3d.geometry.Image(np.ascontiguousarray(color_rgb))

    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("frame", mesh, mat)

    center = np.asarray(center_xyz, dtype=np.float64)
    ang = math.radians(tilt_deg)
    offset = np.array([orbit_dist * math.sin(ang), 0.0, -orbit_dist * math.cos(ang)])
    eye = center + offset
    up = np.array([0.0, -1.0, 0.0])

    renderer.setup_camera(fov_deg, center.tolist(), eye.tolist(), up.tolist())

    img = renderer.render_to_image()
    rgb = np.asarray(img)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if rotate_deg == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_deg == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rotate_deg == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


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

    align = None
    color_intr = None
    depth_scale = 1.0 / 1000.0
    renderer = None
    mat = None

    if not args.no_3d:
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ci = color_profile.get_intrinsics()
        color_intr = (ci.fx, ci.fy, ci.ppx, ci.ppy)
        renderer, mat = make_renderer(args.render_width, args.render_height, args.lit)
        print(f"Open3D OffscreenRenderer ready: {args.render_width}x{args.render_height}, "
              f"shader={'defaultLit + sun' if args.lit else 'defaultUnlit (textured)'}")
        if args.center_z is None:
            print(f"  orbit center z = auto (EMA alpha {args.auto_center_alpha:.2f})")
        else:
            print(f"  orbit center z = {args.center_z:.3f} m (fixed)")
        if args.orbit_dist is None:
            print("  orbit dist = auto (from scene depth extent)")
        else:
            print(f"  orbit dist = {args.orbit_dist:.2f} m (fixed)")
        print(f"  tilt = {args.tilt_deg:.0f} deg, rotate-3d = {args.threed_rotate} deg")

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()
    print(f"Publishing color JPEG to Redis {args.host}:{args.port} key '{args.key}'")
    if not args.no_3d:
        print(f"Publishing 3D mesh JPEG to key '{args.threed_key}' "
              f"(tilt {args.tilt_deg:.0f}deg, FOV {args.fov:.0f}deg, "
              f"max_edge {args.max_edge*1000:.0f}mm)")
    print("Ctrl-C to stop.")

    color_encode = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    threed_encode = [int(cv2.IMWRITE_JPEG_QUALITY), args.threed_jpeg_quality]
    n = 0
    t0 = time.monotonic()
    last_3d_kb = 0.0
    # EMA-smoothed scene depth + extent, populated from the depth image each
    # frame when --center-z or --orbit-dist is left as auto.
    smoothed_center_z: float | None = None
    smoothed_extent_z: float | None = None
    ema_alpha = args.auto_center_alpha

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

            if renderer is not None:
                depth = frames.get_depth_frame()
                if depth:
                    depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
                    fx, fy, cx, cy = color_intr   # type: ignore
                    mesh = depth_to_mesh(depth_m, fx, fy, cx, cy,
                                         args.z_near, args.z_far, args.max_edge)

                    # Auto-derive a stable scene center + orbit distance from
                    # the visible depths, smoothed across frames. User-supplied
                    # values (--center-z / --orbit-dist) override.
                    valid_mask = (depth_m > args.z_near) & (depth_m < args.z_far)
                    if valid_mask.any():
                        valid_z = depth_m[valid_mask]
                        new_center = float(np.median(valid_z))
                        # 5th-95th percentile spread is a robust "extent".
                        z_lo, z_hi = np.percentile(valid_z, [5, 95])
                        new_extent = max(float(z_hi - z_lo), 0.05)
                        if smoothed_center_z is None:
                            smoothed_center_z = new_center
                            smoothed_extent_z = new_extent
                        else:
                            smoothed_center_z = (ema_alpha * smoothed_center_z
                                                 + (1.0 - ema_alpha) * new_center)
                            smoothed_extent_z = (ema_alpha * smoothed_extent_z
                                                 + (1.0 - ema_alpha) * new_extent)

                    cz = args.center_z if args.center_z is not None else (smoothed_center_z or 0.4)
                    od = args.orbit_dist if args.orbit_dist is not None else max(
                        (smoothed_extent_z or 0.3) * 1.5, 0.25)
                    orbit_center = np.array([args.center_x, args.center_y, cz],
                                             dtype=np.float64)

                    color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rendered = render_mesh_view(renderer, mat, mesh, color_rgb,
                                                args.tilt_deg, args.fov,
                                                orbit_center, od,
                                                args.render_width, args.render_height,
                                                args.threed_rotate)
                    ok2, buf2 = cv2.imencode(".jpg", rendered, threed_encode)
                    if ok2:
                        r.set(args.threed_key,
                              struct.pack("<d", ts) + buf2.tobytes())
                        last_3d_kb = len(buf2) / 1024.0

            n += 1
            if n % args.fps == 0:
                dt = time.monotonic() - t0
                msg = (f"  {n} frames, {n / dt:5.1f} fps, "
                       f"color {len(buf)/1024:5.1f} KB")
                if renderer is not None:
                    msg += f", 3D {last_3d_kb:5.1f} KB"
                print(msg)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
