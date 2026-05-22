"""Fake Quest controller publisher.

Streams pose updates to the same Redis keys the Unity RedisTeleopBridge
will write to, at the same rate (~60 Hz). Lets you validate the SAI
controller end of the teleop pipeline without a headset.

Motion: a slow lissajous figure around the scene's home pose, holding
orientation fixed at that scene's expected end-effector orientation.
Gripper width oscillates between open and closed every few seconds.

Usage:
    # Pick the scene matching the XML OpenSai is running
    python vr_teleop/test_harness/fake_quest_publisher.py --scene oussama_push
    python vr_teleop/test_harness/fake_quest_publisher.py --scene grav_motion
    python vr_teleop/test_harness/fake_quest_publisher.py --scene grav_motion --scale 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time

import numpy as np
import redis

# Redis keys — keep these in sync with python_examples/*.py
# and vr_teleop/unity_scripts/RedisTeleopBridge.cs.
ROBOT_NAME = "Rizon4s"
K_GOAL_POS = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position"
K_GOAL_ORI = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_orientation"
K_GRIP     = f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::gripper_fingers::goal_position"
K_ACTIVE   = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"

CONTROLLER_TO_USE = "cartesian_controller"


# Scene presets. Each entry pins down the home pose and end-effector
# orientation that match a specific XML in config_folder/xml_config_files/.
# The values mirror the constants at the top of the matching python_examples/
# scripts; keep them in sync if the scenes shift.
SCENES = {
    # suturebot_grav_motion.py — foam block, +Y pierce direction.
    "grav_motion": {
        "home": np.array([0.51, -0.20, 0.03]),
        "ori": np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ]),
        "amplitude_default": 0.03,   # 3 cm; foam block is small but home is far
    },
    # suturebot_grav_pierce_oussama_push.py — hemostat gripper, +X pierce.
    # Home is X_HOME = 0.6686, Y is first stitch (0.15 - needle offset),
    # WORKING_Z = 0.3146, ORI = [[1,0,0],[0,-1,0],[0,0,-1]].
    "oussama_push": {
        "home": np.array([0.6686, 0.1485, 0.3146]),
        "ori": np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ]),
        "amplitude_default": 0.02,   # 2 cm; jaws start 75 mm from foam, safe
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", choices=sorted(SCENES.keys()), default="oussama_push",
                   help="Which XML scene this is being run against. "
                        "Sets home pose and orientation. (default: oussama_push)")
    p.add_argument("--host", default="127.0.0.1", help="Redis host")
    p.add_argument("--port", type=int, default=6379, help="Redis port")
    p.add_argument("--rate", type=float, default=60.0,
                   help="Update rate in Hz (default: 60)")
    p.add_argument("--scale", type=float, default=None,
                   help="Lissajous amplitude in meters (defaults per scene)")
    p.add_argument("--period", type=float, default=8.0,
                   help="Time for one lissajous cycle in seconds (default: 8)")
    p.add_argument("--gripper-period", type=float, default=4.0,
                   help="Gripper open/close oscillation period in seconds")
    p.add_argument("--no-gripper", action="store_true",
                   help="Skip writing gripper width (matches motion.py no-op)")
    p.add_argument("--no-activate", action="store_true",
                   help="Skip setting active_controller (assume already set)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene = SCENES[args.scene]
    home = scene["home"]
    ori = scene["ori"]
    amplitude = args.scale if args.scale is not None else scene["amplitude_default"]

    r = redis.Redis(host=args.host, port=args.port)
    r.ping()  # fail fast if Redis isn't up

    if not args.no_activate:
        # Match the activation loop in python_examples/*.py so OpenSai's
        # cartesian controller is the one consuming our writes.
        while r.get(K_ACTIVE) is None or r.get(K_ACTIVE).decode() != CONTROLLER_TO_USE:
            r.set(K_ACTIVE, CONTROLLER_TO_USE)
            time.sleep(0.05)

    period = 1.0 / args.rate
    t0 = time.monotonic()
    stop = False

    def handle_sigint(_sig, _frame):  # noqa: ANN001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    print(f"Scene: {args.scene}")
    print(f"  home={home.tolist()}")
    print(f"  amplitude={amplitude*100:.1f} cm, period={args.period}s, rate={args.rate} Hz")
    print(f"Publishing to {args.host}:{args.port}")
    print("Ctrl-C to stop.")

    # Orientation doesn't change during the lissajous; write it once.
    r.set(K_GOAL_ORI, json.dumps(ori.tolist()))

    writes = 0
    next_tick = time.monotonic()

    while not stop:
        t = time.monotonic() - t0

        # 3:2 Lissajous in XY, small sinusoid in Z so the controller sees
        # all three axes change.
        omega = 2.0 * math.pi / args.period
        dx = amplitude * math.sin(3.0 * omega * t)
        dy = amplitude * math.sin(2.0 * omega * t)
        dz = 0.25 * amplitude * math.sin(omega * t)
        pos = home + np.array([dx, dy, dz])

        r.set(K_GOAL_POS, json.dumps(pos.tolist()))

        if not args.no_gripper:
            # Square-wave-ish gripper toggle so you can see grip key updates too.
            phase = (t % args.gripper_period) / args.gripper_period
            width = 0.005 if phase < 0.5 else 0.05
            r.set(K_GRIP, json.dumps([width]))

        writes += 1
        if writes % int(args.rate) == 0:
            print(f"  t={t:6.2f}s  pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
                  f"  ({writes} writes)")

        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # We fell behind; reset the schedule so we don't spin forever.
            next_tick = time.monotonic()

    print(f"\nStopped after {writes} writes.")


if __name__ == "__main__":
    main()
