"""Forward kinematics for the Rizon4s+Grav arm: joint angles -> flange /
needle-tip pose in the robot base frame.

Why this exists: a teach-by-demo recording (record_demo_redis.py) is joint-only
(OpenSai is off during hand-guiding, so it never published the cartesian pose).
To anchor the taught stitch to the camera-detected cut we need to know WHERE the
needle tip went in 3D -- this composes the URDF chain base_link->flange to give
that from joints alone. (At runtime OpenSai publishes the flange pose to Redis;
this is the offline equivalent for the recorded demo.)

The chain (from Rizon4s_Grav_hemostat.urdf):
  joint1..joint7 (revolute, take q[0..6]) then fixed link7_to_flange.
The controlled link is "flange" (same as OpenSai's cartesian task), so
fk_flange(q) should match opensai::...::cartesian_task::current_position.

  python3 python_examples/kinematics.py --q "0,0,0,0,0,0,0"   # print FK for a config
  python3 python_examples/kinematics.py --check-redis          # FK vs OpenSai's live pose (on robot)
  python3 python_examples/kinematics.py --selftest             # math sanity, no robot
"""

from __future__ import annotations
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config_folder", "robot_files", "Rizon4s_Grav_hemostat.urdf")

# Needle tip in the flange frame (straight needle, perpendicular to the forceps,
# 5.8 cm; tip at +x). Same value the clients use.
NEEDLE_TIP_OFFSET = np.array([0.0613, 0.0015, -0.2826])


def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                      # URDF fixed-axis xyz


def _axis_R(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def _T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def load_chain(urdf=URDF):
    raw = re.sub(r"<!--.*?-->", "", open(urdf).read(), flags=re.S)  # strip comments
    root = ET.fromstring(raw)
    by_child = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        ax = j.find("axis")
        by_child[j.find("child").get("link")] = dict(
            name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"),
            xyz=[float(x) for x in ((o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split())],
            rpy=[float(x) for x in ((o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split())],
            axis=[float(x) for x in ax.get("xyz").split()] if ax is not None else None)
    chain, link = [], "flange"
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link]["parent"]
    return chain[::-1]


_CHAIN = load_chain()
_N_REV = sum(1 for j in _CHAIN if j["type"] == "revolute")


def fk_flange(q):
    """4x4 pose of the flange in the base frame for joint vector q (len 7)."""
    q = np.asarray(q, float)
    T = np.eye(4)
    qi = 0
    for j in _CHAIN:
        T = T @ _T(_rpy(*j["rpy"]), j["xyz"])
        if j["type"] == "revolute":
            T = T @ _T(_axis_R(j["axis"], q[qi]), [0, 0, 0])
            qi += 1
    return T


def fk_needle_tip(q):
    """(tip_position_in_base, flange_rotation) for joint vector q."""
    T = fk_flange(q)
    tip = T[:3, 3] + T[:3, :3] @ NEEDLE_TIP_OFFSET
    return tip, T[:3, :3]


def selftest():
    print(f"chain: {len(_CHAIN)} joints ({_N_REV} revolute) -> flange")
    z = fk_flange(np.zeros(7))
    R = z[:3, :3]
    ortho = np.linalg.norm(R.T @ R - np.eye(3))
    print(f"FK(0): flange pos {np.round(z[:3,3],4)} m  |pos|={np.linalg.norm(z[:3,3]):.3f} m")
    print(f"       rotation orthonormal error = {ortho:.2e} (should be ~0)")
    tip, _ = fk_needle_tip(np.zeros(7))
    print(f"FK(0): needle tip {np.round(tip,4)} m")
    reach = np.linalg.norm(z[:3, 3])
    ok = ortho < 1e-9 and 0.1 < reach < 1.5
    # determinant should be +1 (proper rotation)
    det = np.linalg.det(R)
    print(f"       det(R)={det:.6f} (should be +1), reach in [0.1,1.5]m: {0.1<reach<1.5}")
    print("RESULT:", "PASS" if (ok and abs(det - 1) < 1e-9) else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q", default=None, help="comma-separated 7 joint angles (rad)")
    ap.add_argument("--check-redis", action="store_true",
                    help="compare FK(joints) to OpenSai's published flange pose (run on robot)")
    ap.add_argument("--selftest", action="store_true", help="math sanity, no robot")
    ap.add_argument("--robot", default="Titania")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.check_redis:
        import redis
        r = redis.Redis()
        q = json.loads(r.get(f"opensai::sensors::{args.robot}::joint_positions"))
        pos = np.array(json.loads(r.get(
            f"opensai::controllers::{args.robot}::cartesian_controller::cartesian_task::current_position")))
        T = fk_flange(q)
        err = T[:3, 3] - pos
        print(f"FK flange pos      : {np.round(T[:3,3],4)}")
        print(f"OpenSai current_pos: {np.round(pos,4)}")
        print(f"position error     : {np.round(err,4)} m  (|err|={np.linalg.norm(err)*1000:.1f} mm)")
        ori_ok = True
        ori_raw = r.get(f"opensai::controllers::{args.robot}::cartesian_controller::cartesian_task::current_orientation")
        if ori_raw is not None:
            Ro = np.array(json.loads(ori_raw))
            ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3] @ Ro.T) - 1) / 2, -1, 1)))
            ori_ok = ang < 1.0
            print(f"orientation error  : {ang:.2f} deg")
        if np.linalg.norm(err) < 0.005 and ori_ok:
            print("FK MATCHES OpenSai -> kinematics validated.")
        else:
            print("MISMATCH -- send me these numbers and I'll fix the frame/axis convention.")
        return

    q = np.zeros(7) if args.q is None else np.array([float(x) for x in args.q.split(",")])
    T = fk_flange(q)
    tip, _ = fk_needle_tip(q)
    np.set_printoptions(suppress=True, precision=4)
    print("flange pose (base frame):\n", np.round(T, 4))
    print("needle tip (base frame): ", np.round(tip, 4))


if __name__ == "__main__":
    main()
