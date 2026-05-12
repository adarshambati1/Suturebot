"""
Suturebot V1 motion choreography.

Drives the gripper through the suturing motion the team wants to demo:

    home -> GRIP -> approach block -> RELEASE -> lift over -> cross to far side
         -> descend -> GRIP -> retreat

The cartesian task is on the Rizon4s `flange` link (the gripper attachment
point). GRIP / RELEASE are choreographic only — nothing is actually grasped.
The script just walks cartesian-task goals through the waypoints via redis.

Usage (in two terminals, from the OpenSai/ directory):
    # terminal 1
    sh scripts/launch.sh suturebot.xml
    # terminal 2
    python python_examples/suturebot_motion.py
"""

import json
import math
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import redis


class State(Enum):
    GOTO_HOME = auto()
    HOLD_AT_HEIGHT = auto()
    GRIP_1 = auto()
    APPROACH = auto()
    RELEASE = auto()
    LIFT = auto()
    CROSS = auto()
    DESCEND = auto()
    GRIP_2 = auto()
    RETREAT = auto()


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation"
    )
    active_controller: str = "opensai::controllers::Rizon4s::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"


REDIS_KEYS = RedisKeys()
CONFIG_FILE_FOR_THIS_SCRIPT = "suturebot.xml"
CONTROLLER_TO_USE = "cartesian_controller"

# --- Geometry ---------------------------------------------------------------
# TissueBlock (from world_suturebot.urdf), rotated 90° about Z:
#   center (0.55, 0, 0.20), world extent: 0.18 (X) x 0.04 (Y) x 0.40 (Z)
#   approach face y=-0.02, far face y=+0.02, top at z=0.40
# The puncture axis is now world Y (lateral wrist motion at fixed radius),
# not world X, which keeps the arm in a comfortable pose throughout.
BLOCK_X   = 0.55  # radial distance to block center (stays constant)
WORKING_Z = 0.30  # puncture height — well inside the block in Z
LIFT_Z    = 0.55  # clear of the block top (z=0.40)

# Flange orientation: camera (and parallel-jaw) facing straight down.
# Flange-Z aligned with world -Z; jaws perpendicular to the (world +Y) needle
# axis. Implemented as a 180° rotation about world Y so the d405 camera
# mount on link7 protrudes in +X (away from the side camera) instead of
# −X (toward it), which is what was making the wrist look slanted.
ORI_DOWN = np.array([[-1.0, 0.0,  0.0],
                     [ 0.0, 1.0,  0.0],
                     [ 0.0, 0.0, -1.0]])

# --- Waypoints --------------------------------------------------------------
# HOME is already at the puncture height, so the first move is purely
# horizontal — the needle has uninterrupted room to drive into the block.
# Block occupies y in [-0.02, +0.02]. Link7 (with d405 camera mount) sits
# ~12 cm above the flange when the camera is pointing down and is several
# cm thick, so the flange waypoints stop 8 cm shy of each block face
# (y=±0.10) — that gives link7's body real clearance from the block.
# A conceptual needle dangling from the flange in +/-Y does the puncturing.
P_HOME    = np.array([BLOCK_X, -0.25, WORKING_Z])  # -Y side, at puncture height
P_BLOCK   = np.array([BLOCK_X, -0.10, WORKING_Z])  # flange 8 cm before approach face
P_LIFT    = np.array([BLOCK_X, -0.10, LIFT_Z])     # straight up — well clear of block in Y
P_CROSS   = np.array([BLOCK_X,  0.10, LIFT_Z])     # translate over to +Y side (above block top)
P_FAR     = np.array([BLOCK_X,  0.10, WORKING_Z])  # flange 8 cm past far face
P_RETREAT = np.array([BLOCK_X,  0.25, WORKING_Z])  # pull horizontally away from block

SEGMENT_TIME = 2.5  # seconds per motion segment
GRIP_PAUSE   = 1.0  # seconds for grip / release dwell
HEIGHT_HOLD  = 2.5  # seconds spent stationary at puncture height before going at the block


def set_goal(redis_client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    redis_client.set(REDIS_KEYS.cartesian_task_goal_position, json.dumps(pos.tolist()))
    redis_client.set(REDIS_KEYS.cartesian_task_goal_orientation, json.dumps(ori.tolist()))


def main() -> None:
    redis_client = redis.Redis()

    config_file_name = redis_client.get(REDIS_KEYS.config_file_name)
    if config_file_name is None or config_file_name.decode("utf-8") != CONFIG_FILE_FOR_THIS_SCRIPT:
        print(f"This script expects OpenSai_main to be running with {CONFIG_FILE_FOR_THIS_SCRIPT}.")
        return

    # Make sure the cartesian controller is the active one.
    while redis_client.get(REDIS_KEYS.active_controller).decode("utf-8") != CONTROLLER_TO_USE:
        redis_client.set(REDIS_KEYS.active_controller, CONTROLLER_TO_USE)

    plan = [
        (State.GOTO_HOME,      P_HOME, ORI_DOWN, SEGMENT_TIME, "going to home pose (at puncture height)"),
        (State.HOLD_AT_HEIGHT, P_HOME, ORI_DOWN, HEIGHT_HOLD,  "holding at puncture height — settling before horizontal motion"),
        (State.GRIP_1,         P_HOME, ORI_DOWN, GRIP_PAUSE,   "GRIP (choreographic)"),
        (State.APPROACH,  P_BLOCK,   ORI_DOWN, SEGMENT_TIME, "horizontal puncture — driving needle through the block"),
        (State.RELEASE,   P_BLOCK,   ORI_DOWN, GRIP_PAUSE,   "RELEASE (choreographic)"),
        (State.LIFT,      P_LIFT,    ORI_DOWN, SEGMENT_TIME, "lifting clear of the block top"),
        (State.CROSS,     P_CROSS,   ORI_DOWN, SEGMENT_TIME, "crossing over to far side"),
        (State.DESCEND,   P_FAR,     ORI_DOWN, SEGMENT_TIME, "descending to the protruding needle on far side"),
        (State.GRIP_2,    P_FAR,     ORI_DOWN, GRIP_PAUSE,   "GRIP (choreographic)"),
        (State.RETREAT,   P_RETREAT, ORI_DOWN, SEGMENT_TIME, "retreating horizontally, pulling needle out"),
    ]

    for state, pos, ori, dwell, msg in plan:
        print(f"[{state.name}] {msg}")
        set_goal(redis_client, pos, ori)
        time.sleep(dwell)

    print("[DONE] Sequence complete. Holding final pose.")


if __name__ == "__main__":
    main()
