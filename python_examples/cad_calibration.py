"""Derive the camera->wrist transform (T_FLANGE_CAM) from CAD and write the
calibration file fruit_scan.py auto-loads. No robot or camera needed.

It composes the URDF kinematic chain (Rizon4s_Grav_hemostat.urdf):
  * link7 -> flange : xyz (0,0,0.124),  rpy (0,0,-pi)
  * link7 -> mount  : xyz (0.045,0,0.046), rpy (1.5717,3.1415,1.5717)
with the D405 lens position read from the "Fake Camera Part For Measurement"
group in rizon4s/visual/d405-camera-mount.obj.

Result (flange frame): the lens sits ~2 cm above the flange and ~7.8 cm to the
-x side (the opposite side of the forceps, which are at +x), looking along
flange -z (the same direction the needle points). This matches the hand
description: "other side of the forceps", "~11 cm above the gripper base".

Because fruit_scan centres the fruit on the optical axis before placing the
tool, the position (t_fc) and forward axis dominate accuracy; the image-roll
columns of the rotation matter little. Verify with:
    python3 python_examples/fruit_scan.py --no-robot --show
and check the printed 'cam xyz' is sensible (small x/y, depth ~ distance).

Usage:
    python3 python_examples/cad_calibration.py
    python3 python_examples/cad_calibration.py --cam-z-above-flange 0.02   # override z
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ = os.path.join(REPO_ROOT, "config_folder", "robot_files", "rizon4s",
                   "visual", "d405-camera-mount.obj")
CALIB_PATH = os.path.join(REPO_ROOT, "calibration", "t_flange_cam.json")


def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def rpy(r, p, y): return Rz(y) @ Ry(p) @ Rx(r)        # URDF fixed-axis xyz


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def lens_front_center_in_mount():
    """Front-face centre of the 'Fake Camera Part For Measurement' group (the
    lens), in the mount's local frame. The part pokes out in -y, so the lens
    faces +y toward the workspace; its front face is the most -y face."""
    verts, idx, cur = [], set(), None
    with open(OBJ) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append((float(t[1]), float(t[2]), float(t[3])))
            elif t[0] == "g":
                cur = line[2:].strip()
            elif t[0] == "f" and cur == "Fake Camera Part For Measurement":
                for p in t[1:]:
                    idx.add(int(p.split("/")[0]) - 1)
    if not idx:
        raise SystemExit("Could not find the 'Fake Camera Part For Measurement' "
                         f"group in {OBJ}")
    pts = np.array(verts)[sorted(idx)]
    cx, cz = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 2].min() + pts[:, 2].max()) / 2
    front_y = pts[:, 1].min()                 # most -y face = lens face
    return np.array([cx, front_y, cz])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-z-above-flange", type=float, default=None,
                    help="override the lens height above the flange (m) instead "
                         "of using the CAD value")
    args = ap.parse_args()

    T_l7_mount = T(rpy(1.5717, 3.1415, 1.5717), [0.045, 0, 0.046])
    T_l7_flange = T(rpy(0, 0, -np.pi), [0, 0, 0.124])
    T_flange_mount = np.linalg.inv(T_l7_flange) @ T_l7_mount

    lens_mount = lens_front_center_in_mount()
    t_fc = (T_flange_mount @ np.append(lens_mount, 1.0))[:3]
    if args.cam_z_above_flange is not None:
        t_fc[2] = args.cam_z_above_flange

    # Optical frame in the mount: lens looks +y_mount (toward the workspace);
    # image-right = +x_mount, image-down = -z_mount (right-handed, z forward).
    R_mount_optical = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]], float)
    R_fc = T_flange_mount[:3, :3] @ R_mount_optical
    Tcam = T(R_fc, t_fc)

    fwd = R_fc[:, 2]
    np.set_printoptions(suppress=True, precision=4)
    print(f"lens (mount frame)      : {np.round(lens_mount, 4)} m")
    print(f"camera pos (flange)t_fc : {np.round(t_fc, 4)} m  "
          f"(-x side of forceps, ~{t_fc[2]*100:.1f} cm above flange)")
    print(f"optical forward (flange): {np.round(fwd, 4)}  (flange -z = toward tool/workspace)")
    print("T_FLANGE_CAM =\n", np.round(Tcam, 5))

    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump({"T_FLANGE_CAM": Tcam.tolist(),
                   "source": "cad_calibration.py (URDF chain + OBJ lens marker)",
                   "rms_mm": None}, f, indent=2)
    print(f"\nWrote {CALIB_PATH} — fruit_scan.py will auto-load it.")
    print("Verify:  python3 python_examples/fruit_scan.py --no-robot --show")


if __name__ == "__main__":
    main()
