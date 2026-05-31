"""Verify the OpenSai grasp patch: kinematic pin + drive + release.

The patch adds, per dynamic object, two redis receive keys in the simviz
interface:
    opensai::commands::<obj>::pose       JSON 4x4 row-major homogeneous transform
    opensai::commands::<obj>::kinematic  "1" = pin to commanded pose, "0" = dynamics
and publishes the resulting pose at:
    opensai::sensors::<obj>::object_pose JSON 4x4

This script pins TestBox up high (must NOT fall), drives it upward (pose must
track), then releases it (must fall under gravity) -- all observed via the
published object_pose. Run against pin_test.xml:

    cd ../OpenSai && sh scripts/launch.sh config_folder/xml_config_files/pin_test.xml
    python3 python_examples/pin_test.py
"""

import json
import time

import numpy as np
import redis

CONFIG_FILE = "pin_test.xml"
OBJ = "TestBox"

K_POSE    = f"opensai::commands::{OBJ}::pose"
K_KIN     = f"opensai::commands::{OBJ}::kinematic"
K_OBJPOSE = f"opensai::sensors::{OBJ}::object_pose"
K_CONFIG  = "::sai-interfaces-webui::config_file_name"


def T(x: float, y: float, z: float) -> np.ndarray:
    M = np.eye(4)
    M[0, 3], M[1, 3], M[2, 3] = x, y, z
    return M


def read_z(r: redis.Redis):
    raw = r.get(K_OBJPOSE)
    if raw is None:
        return None
    return float(np.array(json.loads(raw.decode("utf-8")))[2, 3])


def main() -> None:
    r = redis.Redis()

    # Wait (up to 25s) for the sim to come up and start publishing the object.
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        cfg = r.get(K_CONFIG)
        if cfg is not None and cfg.decode("utf-8") == CONFIG_FILE and r.get(K_OBJPOSE) is not None:
            break
        time.sleep(0.5)
    else:
        cfg = r.get(K_CONFIG)
        print(f"Sim not ready: config={cfg!r}, object_pose={'set' if r.get(K_OBJPOSE) else 'missing'}. "
              f"Launch pin_test.xml first.")
        return

    results = []

    # Phase 1: pin and hold at z=0.5. A free body would fall; a pinned one holds.
    r.set(K_POSE, json.dumps(T(0.4, 0.3, 0.5).tolist()))
    r.set(K_KIN, "1")
    time.sleep(2.0)
    z_hold = read_z(r)
    held = z_hold is not None and abs(z_hold - 0.5) < 0.02
    results.append(held)
    print(f"[HOLD]    commanded z=0.500  observed z={z_hold}  -> {'PASS' if held else 'FAIL'}")

    # Phase 2: drive the commanded pose up to z=0.7; published pose must track.
    for z in np.linspace(0.5, 0.7, 40):
        r.set(K_POSE, json.dumps(T(0.4, 0.3, float(z)).tolist()))
        time.sleep(0.05)
    time.sleep(0.5)
    z_drive = read_z(r)
    tracked = z_drive is not None and abs(z_drive - 0.7) < 0.02
    results.append(tracked)
    print(f"[DRIVE]   commanded z=0.700  observed z={z_drive}  -> {'PASS' if tracked else 'FAIL'}")

    # Phase 3: release -> the box must fall under gravity.
    r.set(K_KIN, "0")
    time.sleep(2.0)
    z_fall = read_z(r)
    fell = z_fall is not None and z_fall < 0.3
    results.append(fell)
    print(f"[RELEASE] released from z=0.700  observed z={z_fall}  (expect drop)  -> {'PASS' if fell else 'FAIL'}")

    print("\nRESULT:", "ALL PASS -- patch works" if all(results) else "FAIL -- walk back the patch")


if __name__ == "__main__":
    main()
