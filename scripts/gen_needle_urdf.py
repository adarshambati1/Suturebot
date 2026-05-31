#!/usr/bin/env python3
"""Generate the curved suture-needle dynamic_object and inject it into
world_suture_pad.urdf between the NEEDLE_START / NEEDLE_END markers.

Models a Ferguson 3/8-circle (135 deg) reverse-cutting needle.

  VISUAL    : a chain of thin cylinder segments forming the arc, a red tip
              sphere, and a blue "flat" marker at the swage (grip) end for
              clocking. (URDF has no triangular primitive, so the reverse-
              cutting cross-section is approximated by a round tube -- cosmetic.)
  COLLISION : a SINGLE box aligned with the swage->tip chord (a straight stick
              approximating the arc).
              *** This SAI build SEGFAULTS on a dynamic_object with more than
              one <collision> element (the multi-mesh combine path is broken),
              so the needle gets exactly one collision primitive. *** A single
              chord box resists rolling and lies flat, which is what the
              kill-window tip-over needs; swap in a single curved mesh in
              Phase 4 if higher fidelity is required.

Key choice: the dynamic_object body frame sits at the arc's CENTER OF
CURVATURE, in the local XY plane. So driving the needle through tissue is a
pure rotation about the body's local Z -- the arc-drive primitive. The gripper
holds the swage; the flange->needle grasp transform (calibrated later) encodes
that the body frame is R away from the grip.

Re-run after changing any parameter; it rewrites the marker block.
"""
import math
import os
import re

# --- needle parameters (tune here) ---------------------------------------
NAME    = "Needle"
R       = 0.011        # radius of curvature (m): 3/8 circle, ~26 mm arc
ARC_DEG = 135.0        # 3/8 of a full circle
N_VIS   = 9            # visual arc segments
TUBE_R  = 0.0008       # wire radius (m), visual
COL_W   = 0.0024       # collision box cross-section (m); a bit fatter than the wire for stable contact
MASS    = 0.050        # kg (heavier than a real needle; light bodies jitter on contact in sim)
ORIGIN  = (0.45, -0.30, 0.10)   # initial body-frame (arc-center) pose; pinned by the FSM later
RPY     = (0.0, 0.0, 0.0)

STEEL = ("needle_steel", "0.75 0.78 0.82 1.0")
TIP   = ("needle_tip",   "0.85 0.15 0.15 1.0")
FLAT  = ("needle_flat",  "0.20 0.55 0.90 1.0")

WORLD = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..",
    "config_folder", "world_files", "world_suture_pad.urdf"))


def vis_cyl(cx, cy, cz, roll, pitch, yaw, length):
    o = f'<origin xyz="{cx:.5f} {cy:.5f} {cz:.5f}" rpy="{roll:.5f} {pitch:.5f} {yaw:.5f}" />'
    g = f'<cylinder radius="{TUBE_R:.5f}" length="{length:.5f}" />'
    return (f'\t\t<visual>{o}<geometry>{g}</geometry>'
            f'<material name="{STEEL[0]}"><color rgba="{STEEL[1]}" /></material></visual>\n')


def vis_sphere(cx, cy, cz, radius, mat):
    o = f'<origin xyz="{cx:.5f} {cy:.5f} {cz:.5f}" rpy="0 0 0" />'
    return (f'\t\t<visual>{o}<geometry><sphere radius="{radius:.5f}" /></geometry>'
            f'<material name="{mat[0]}"><color rgba="{mat[1]}" /></material></visual>\n')


def vis_box(cx, cy, cz, sx, sy, sz, mat):
    o = f'<origin xyz="{cx:.5f} {cy:.5f} {cz:.5f}" rpy="0 0 0" />'
    return (f'\t\t<visual>{o}<geometry><box size="{sx:.5f} {sy:.5f} {sz:.5f}" /></geometry>'
            f'<material name="{mat[0]}"><color rgba="{mat[1]}" /></material></visual>\n')


def col_box(cx, cy, cz, yaw, length, w):
    o = f'<origin xyz="{cx:.5f} {cy:.5f} {cz:.5f}" rpy="0 0 {yaw:.5f}" />'
    return f'\t\t<collision>{o}<geometry><box size="{length:.5f} {w:.5f} {w:.5f}" /></geometry></collision>\n'


def build():
    arc = math.radians(ARC_DEG)

    # --- visual: cylinder chord segments along the arc ---
    dth = arc / N_VIS
    r_mid = R * math.cos(dth / 2.0)
    seg_len = 2.0 * R * math.sin(dth / 2.0)
    parts = []
    for i in range(N_VIS):
        thm = (i + 0.5) * dth
        # cylinder default axis = +Z; rpy=(0, pi/2, thm+pi/2) lays it in the XY
        # plane along the chord (tangent) direction.
        parts.append(vis_cyl(r_mid * math.cos(thm), r_mid * math.sin(thm), 0.0,
                             0.0, math.pi / 2.0, thm + math.pi / 2.0, seg_len))

    # tip sphere (red) at theta = arc; flat marker (blue) at the swage (R,0,0).
    sx, sy = R, 0.0                          # swage
    tx, ty = R * math.cos(arc), R * math.sin(arc)   # tip
    parts.append(vis_sphere(tx, ty, 0.0, 1.4 * TUBE_R, TIP))
    parts.append(vis_box(sx, sy, 0.0, 0.0009, 0.004, 0.0022, FLAT))

    # --- collision: single box along the swage->tip chord (one primitive only) ---
    mx, my = (sx + tx) / 2.0, (sy + ty) / 2.0
    chord_len = math.hypot(tx - sx, ty - sy)
    chord_yaw = math.atan2(ty - sy, tx - sx)
    parts.append(col_box(mx, my, 0.0, chord_yaw, chord_len, COL_W))

    # COM at the chord midpoint = the collision box center (NOT the arc centroid,
    # and NOT the body origin / arc center). With a single straight collision
    # box, the COM must sit over that box or gravity tips the needle off it and
    # it creeps. (When the collision becomes a true curved mesh in Phase 4, move
    # the COM to the arc centroid.)
    ix, iy, iz = ORIGIN
    rr, pp, yy = RPY
    inertia = max(MASS * R * R, 1e-7)
    return (
        f'\t<dynamic_object name="{NAME}">\n'
        f'\t\t<origin xyz="{ix:.5f} {iy:.5f} {iz:.5f}" rpy="{rr} {pp} {yy}" />\n'
        f'\t\t<inertial>\n'
        f'\t\t\t<origin xyz="{mx:.5f} {my:.5f} 0" rpy="0 0 0" />\n'
        f'\t\t\t<mass value="{MASS}" />\n'
        f'\t\t\t<inertia ixx="{inertia:.3e}" iyy="{inertia:.3e}" izz="{inertia:.3e}" '
        f'ixy="0" ixz="0" iyz="0" />\n'
        f'\t\t</inertial>\n'
        + "".join(parts)
        + '\t</dynamic_object>\n')


def main():
    block = build()
    with open(WORLD, "r") as f:
        text = f.read()
    pat = re.compile(r"(<!-- NEEDLE_START[^\n]*-->\n).*?(\t*<!-- NEEDLE_END -->)", re.DOTALL)
    if not pat.search(text):
        raise SystemExit("NEEDLE_START/NEEDLE_END markers not found in world file")
    new = pat.sub(lambda m: m.group(1) + block + "\t" + m.group(2).lstrip("\t"), text)
    with open(WORLD, "w") as f:
        f.write(new)
    print(f"Injected '{NAME}': {N_VIS} visual segments + 1 chord collision box, "
          f"R={R*1000:.1f} mm, {ARC_DEG:.0f} deg arc")


if __name__ == "__main__":
    main()
