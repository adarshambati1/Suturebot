"""
Suturebot needle-puncture choreography.

The arm is bare Rizon4s, but the choreography imagines a tool chain
hanging from the flange:
    flange  --  Grav gripper (0.20 m)
            --  mosquito forceps (~0.10 m extending past gripper)
            --  straight needle (0.0698 m = 69.81 mm)
The total flange-to-needle-back offset along world -Z is ~0.30 m; the
needle itself extends 69.81 mm in world +Y from the back end.

For each stitch x position the motion drives the needle laterally
through the 7.5 mm strip of foam that sits above the wall tops:
    1. APPROACH        — flange far in -Y, needle clear of foam
    2. PIERCE_THROUGH  — drive flange +Y until needle tip exits foam
    3. PULL_THROUGH    — continue +Y, fully extracting needle past foam

Between stitches the flange translates in X to the next stitch's home.

Usage (in two terminals, from OpenSai/):
    sh scripts/launch.sh suturebot.xml
    python python_examples/suturebot_stitch_sequence.py
"""

import json
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import redis


class State(Enum):
    GOTO_HOME       = auto()
    HOLD_AT_HEIGHT  = auto()
    PIERCE_THROUGH  = auto()
    PULL_THROUGH    = auto()
    REPOSITION      = auto()


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation"
    )
    active_controller: str = "opensai::controllers::Rizon4s::active_controller_name"
    config_file_name: str  = "::sai-interfaces-webui::config_file_name"


REDIS_KEYS = RedisKeys()
CONFIG_FILE_FOR_THIS_SCRIPT = "suturebot.xml"
CONTROLLER_TO_USE = "cartesian_controller"

# --- Tool chain (imagined) -------------------------------------------------
CHAIN_OFFSET_Z = 0.30   # flange to needle back end (gripper + forceps), -Z
NEEDLE_LENGTH  = 0.0698 # 69.81 mm — needle extends in +Y from its back end

# --- Foam geometry (from world_suturebot.urdf) -----------------------------
# FoamBlock 0.130 (X) x 0.0057 (Y) x 0.0225 (Z), center (0.55, 0, 0.01925).
# -Y face y=-0.00285, +Y face y=+0.00285. Exposed strip z=0.023 to 0.0305.
NEEDLE_Z       = 0.027  # needle height — middle of the exposed foam strip
WORKING_Z      = NEEDLE_Z + CHAIN_OFFSET_Z   # flange height to place needle at NEEDLE_Z

FOAM_LEFT_Y  = -0.00285
FOAM_RIGHT_Y = +0.00285

# Y waypoints. Flange y → needle tip y is offset by +NEEDLE_LENGTH.
Y_HOME   = -0.15                                    # tip at -0.0802, well clear
Y_PIERCE = FOAM_RIGHT_Y - NEEDLE_LENGTH             # tip at +0.00285, just past foam
Y_PULL   = +0.05                                    # tip at +0.1198, fully extracted

# --- Stitch positions ------------------------------------------------------
# Foam X range: 0.485 to 0.615. Three stitches at 5 cm spacing.
STITCH_X_POSITIONS = [0.50, 0.55, 0.60]

# --- Orientation -----------------------------------------------------------
# Flange camera-down (flange-Z = world -Z); 180° rotation about world X so
# flange-X = +X (the gripper's grip axis lies along world X, perpendicular
# to the world-Y needle direction).
ORI_DOWN = np.array([[ 1.0,  0.0,  0.0],
                     [ 0.0, -1.0,  0.0],
                     [ 0.0,  0.0, -1.0]])

# --- Timing ----------------------------------------------------------------
MOVE_TIME   = 2.0
HEIGHT_HOLD = 2.0
DWELL       = 0.8


def set_goal(redis_client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    redis_client.set(REDIS_KEYS.cartesian_task_goal_position, json.dumps(pos.tolist()))
    redis_client.set(REDIS_KEYS.cartesian_task_goal_orientation, json.dumps(ori.tolist()))


def step(redis_client: redis.Redis, state: State, pos: np.ndarray, dwell: float, msg: str) -> None:
    print(f"[{state.name}] {msg}")
    set_goal(redis_client, pos, ORI_DOWN)
    time.sleep(dwell)


def run_stitch(redis_client: redis.Redis, x: float, idx: int) -> None:
    print(f"\n=== Stitch {idx + 1} at x = {x:.2f} ===")

    step(redis_client, State.PIERCE_THROUGH,
         np.array([x, Y_PIERCE, WORKING_Z]), MOVE_TIME,
         "driving needle through foam")
    step(redis_client, State.PIERCE_THROUGH,
         np.array([x, Y_PIERCE, WORKING_Z]), DWELL,
         "needle through foam — brief pause")

    step(redis_client, State.PULL_THROUGH,
         np.array([x, Y_PULL, WORKING_Z]), MOVE_TIME,
         "pulling needle fully through")
    step(redis_client, State.PULL_THROUGH,
         np.array([x, Y_PULL, WORKING_Z]), DWELL,
         "stitch complete — brief pause")


def main() -> None:
    redis_client = redis.Redis()

    config_file_name = redis_client.get(REDIS_KEYS.config_file_name)
    if config_file_name is None or config_file_name.decode("utf-8") != CONFIG_FILE_FOR_THIS_SCRIPT:
        print(f"This script expects OpenSai_main running with {CONFIG_FILE_FOR_THIS_SCRIPT}.")
        return

    while redis_client.get(REDIS_KEYS.active_controller).decode("utf-8") != CONTROLLER_TO_USE:
        redis_client.set(REDIS_KEYS.active_controller, CONTROLLER_TO_USE)

    first_x = STITCH_X_POSITIONS[0]
    home = np.array([first_x, Y_HOME, WORKING_Z])
    step(redis_client, State.GOTO_HOME, home, MOVE_TIME,
         "going to home — flange high, needle clear of foam")
    step(redis_client, State.HOLD_AT_HEIGHT, home, HEIGHT_HOLD,
         "holding at puncture height before first stitch")

    for i, x in enumerate(STITCH_X_POSITIONS):
        run_stitch(redis_client, x, i)
        if i + 1 < len(STITCH_X_POSITIONS):
            next_x = STITCH_X_POSITIONS[i + 1]
            step(redis_client, State.REPOSITION,
                 np.array([next_x, Y_HOME, WORKING_Z]), MOVE_TIME,
                 f"repositioning to next stitch at x={next_x:.2f}")

    print("\n[DONE] All stitches complete. Holding final pose.")


if __name__ == "__main__":
    main()
