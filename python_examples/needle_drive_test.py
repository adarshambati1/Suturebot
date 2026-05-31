"""Phase 2: the arc-drive primitive.

The needle's body frame is at its arc CENTER, so driving the tip through tissue
is a pure rotation of the needle about its local Z (the arc-plane normal),
while the arc center stays pinned at a fixed world PIVOT. The tip then sweeps a
circle of radius R through the slit: enter one side -> dip below the surface ->
emerge the other side.

R_SEAT orients the arc plane vertical and perpendicular to the slit (arc normal
along the slit's Y axis), so the sweep happens in the world XZ plane (across the
slit and down). The flange follows so the arm holds the needle through the
drive. PIVOT / R_SEAT / the sweep range are the geometry to trial-and-error.

Run against suture_pad.xml (sim already launched).
"""
import json
import time

import numpy as np
import redis

ROBOT = "Rizon4s"
CONFIG_FILE = "suture_pad.xml"

R_CURV  = 0.011                       # needle radius of curvature (must match gen_needle_urdf.py)
ARC     = np.radians(135.0)           # 3/8 circle
TIP_LOCAL = np.array([R_CURV * np.cos(ARC), R_CURV * np.sin(ARC), 0.0])

# --- drive geometry (tune) ---
# PIVOT at the pad surface so the needle's circle (radius R) crosses the surface
# on both sides of the slit; sweeping through the bottom gives enter -> dip ->
# emerge. With this seating the tip world angle is (135 deg + phi), so phi in
# (45, 225) takes the tip from one surface crossing, through the bottom, to the
# other.
PIVOT   = np.array([0.45, -0.20, 0.0254])  # arc center at the pad surface
SWEEP   = (np.radians(45.0), np.radians(225.0))    # clean enter->emerge bite
SAFE    = np.array([0.45, -0.20, 0.40])    # parked flange pose (arm stays put)
N_STEPS = 90
STEP_DT = 0.05

# arc plane = world XZ, arc normal (local z) -> world -Y (the slit direction).
R_SEAT = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0]])

# grasp seating (needle body relative to flange), same as the grip test.
GRASP_POS = np.array([0.0, 0.0, 0.20])   # ~20 cm down the tool axis = the Grav jaws

K = {
    "goal_pos": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_position",
    "goal_ori": f"opensai::controllers::{ROBOT}::cartesian_controller::cartesian_task::goal_orientation",
    "active":   f"opensai::controllers::{ROBOT}::active_controller_name",
    "needle_cmd_pose": "opensai::commands::Needle::pose",
    "needle_kin":      "opensai::commands::Needle::kinematic",
    "needle_pose":     "opensai::sensors::Needle::object_pose",
    "config":          "::sai-interfaces-webui::config_file_name",
}


def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0, 0, 1]])


def T_from(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


GRASP = T_from(np.eye(3), GRASP_POS)
GRASP_INV = np.linalg.inv(GRASP)


def needle_body(phi):
    """Needle body pose at sweep angle phi: rotate about local Z, center at PIVOT."""
    return T_from(R_SEAT @ Rz(phi), PIVOT)


def tip_world(phi):
    R = R_SEAT @ Rz(phi)
    return PIVOT + R @ TIP_LOCAL


def main():
    r = redis.Redis()
    cfg = r.get(K["config"])
    if cfg is None or cfg.decode() != CONFIG_FILE:
        print(f"Expected OpenSai running {CONFIG_FILE}, got {cfg!r}."); return
    while r.get(K["active"]).decode() != "cartesian_controller":
        r.set(K["active"], "cartesian_controller")

    # seat the needle at the start of the sweep and grip it
    phi0, phi1 = SWEEP
    Tn = needle_body(phi0)
    r.set(K["needle_cmd_pose"], json.dumps(Tn.tolist()))
    r.set(K["needle_kin"], "1")
    Tf = Tn @ GRASP_INV
    r.set(K["goal_pos"], json.dumps(Tf[:3, 3].tolist()))
    r.set(K["goal_ori"], json.dumps(Tf[:3, :3].tolist()))
    time.sleep(2.5)

    print(f"driving: sweep {np.degrees(phi0):.0f} -> {np.degrees(phi1):.0f} deg about the arc center")
    print(f"  PIVOT={PIVOT.tolist()}  (slit at X=0.45, pad top z=0.0254)")
    print(f"  {'phi':>6} {'tipX':>8} {'tipZ':>8}")
    for i in range(N_STEPS + 1):
        phi = phi0 + (phi1 - phi0) * i / N_STEPS
        Tn = needle_body(phi)
        r.set(K["needle_cmd_pose"], json.dumps(Tn.tolist()))
        Tf = Tn @ GRASP_INV
        r.set(K["goal_pos"], json.dumps(Tf[:3, 3].tolist()))
        r.set(K["goal_ori"], json.dumps(Tf[:3, :3].tolist()))
        if i % 10 == 0:
            t = tip_world(phi)
            print(f"  {np.degrees(phi):6.0f} {t[0]:8.4f} {t[2]:8.4f}")
        time.sleep(STEP_DT)

    # report the tip path: it should cross the slit (X passes 0.45) and dip
    # below the pad surface (z < 0.0254) mid-sweep.
    xs = [tip_world(phi0 + (phi1 - phi0) * i / N_STEPS)[0] for i in range(N_STEPS + 1)]
    zs = [tip_world(phi0 + (phi1 - phi0) * i / N_STEPS)[2] for i in range(N_STEPS + 1)]
    crossed = min(xs) < 0.45 < max(xs)
    dipped = min(zs) < 0.0254
    print(f"\ntip X range [{min(xs):.4f}, {max(xs):.4f}] crosses slit(0.45): {crossed}")
    print(f"tip Z min {min(zs):.4f} dips below surface(0.0254): {dipped}")
    print("RESULT:", "tip arcs through the slit" if (crossed and dipped)
          else "geometry needs tuning (adjust PIVOT / SWEEP)")


if __name__ == "__main__":
    main()
