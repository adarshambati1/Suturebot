"""
Suturebot pierce-only motion - Grav-with-hemostat variant, Oussama Push.

270°-about-Z-rotated cousin of suturebot_grav_pierce.py.

  R_z(270°) = R_z(-90°) = R_z(90°) o R_z(180°)

Compared to the un-rotated pierce:
   - Whole scene rotated about world Z at the foam center (0.55, 0).
   - Gripper commanded ORI is R_z(-90°) @ ORI_old; the needle now points
     world +X (was +Y) and the jaws-needle axis lies along world X.
   - Foam is thin in X, long in Y (same world URDF as the inward variant).
     The slot runs in Y.
   - Pierce drives the flange in +X (outward, away from robot base);
     stitches space along Y.

State sequence:
   1. GOTO_HOME       - jaws 75 mm from foam -X face (on -X side), needle clear
   2. HOLD_AT_HEIGHT  - settle
   3. PIERCE_THROUGH  - drive +X until jaws flush with foam -X face
                        (needle fully traverses foam and exits +X side)
   4. RETURN_HOME     - back to home x (needle clears foam)
   5. (loop to next stitch y via TRANSIT_Y)
"""

import json
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import redis


# Toggle: "Rizon4s" for sim, "Titania" for the real Flexiv driver.
ROBOT_NAME = "Rizon4s"

# Matching xml config that OpenSai_main must be running.
CONFIG_FILE_FOR_THIS_SCRIPT = (
    "suturebot_grav_real.xml" if ROBOT_NAME == "Titania" else "suturebot_grav_oussama_push.xml"
)


class State(Enum):
    GOTO_HOME       = auto()
    HOLD_AT_HEIGHT  = auto()
    PIERCE_THROUGH  = auto()
    RETURN_HOME     = auto()
    TRANSIT_Y       = auto()


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_orientation"
    )
    cartesian_task_current_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_orientation"
    )
    active_controller: str = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
    config_file_name: str  = "::sai-interfaces-webui::config_file_name"


REDIS_KEYS = RedisKeys()
CONTROLLER_TO_USE = "cartesian_controller"

# Old offsets (from suturebot_grav_pierce.py) rotated by R_z(-90°):
#   R_z(-90°) @ (x, y, z) = (y, -x, z)
#   JAWS_OFFSET_old   = (-0.0015, -0.0085, -0.2826) -> (-0.0085, +0.0015, -0.2826)
#   NEEDLE_OFFSET_old = (-0.0015, +0.0613, -0.2826) -> (+0.0613, +0.0015, -0.2826)
JAWS_OFFSET       = np.array([-0.0085, +0.0015, -0.2826])
NEEDLE_TIP_OFFSET = np.array([+0.0613, +0.0015, -0.2826])

FOAM_CENTER_X  =  0.738
FOAM_NEG_X     = FOAM_CENTER_X - 0.00285
FOAM_POS_X     = FOAM_CENTER_X + 0.00285
NEEDLE_Z       =  0.027

HOME_GAP        = 0.075
APPROACH_BUFFER = 0.005    # jaws stop 5 mm shy of foam -X face

NEEDLE_Y_STITCHES = [-0.05, 0.00, +0.05]
FLANGE_Y = [ny - NEEDLE_TIP_OFFSET[1] for ny in NEEDLE_Y_STITCHES]

WORKING_Z = NEEDLE_Z - NEEDLE_TIP_OFFSET[2]
HOME_Z    = WORKING_Z


def flange_x_for_jaws(jaws_x: float) -> float:
    return jaws_x - JAWS_OFFSET[0]


# Approach from -X side; drive +X to pierce outward.
X_HOME        = flange_x_for_jaws(FOAM_NEG_X - HOME_GAP)
X_FLUSH_LEFT  = flange_x_for_jaws(FOAM_NEG_X - APPROACH_BUFFER)

ORI = np.array([[1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]])

MOVE_TIME   = 2.0
HEIGHT_HOLD = 2.0
DWELL       = 0.5


def set_goal(redis_client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    redis_client.set(REDIS_KEYS.cartesian_task_goal_position, json.dumps(pos.tolist()))
    redis_client.set(REDIS_KEYS.cartesian_task_goal_orientation, json.dumps(ori.tolist()))


def read_actual_pose(redis_client: redis.Redis):
    pos_str = redis_client.get(REDIS_KEYS.cartesian_task_current_position)
    ori_str = redis_client.get(REDIS_KEYS.cartesian_task_current_orientation)
    if pos_str is None or ori_str is None:
        return None, None
    return (np.array(json.loads(pos_str.decode("utf-8"))),
            np.array(json.loads(ori_str.decode("utf-8"))))


JAWS_IN_FLANGE   = ORI.T @ JAWS_OFFSET
NEEDLE_IN_FLANGE = ORI.T @ NEEDLE_TIP_OFFSET


def compute_world_from_flange(flange_pos: np.ndarray, flange_ori: np.ndarray,
                              point_in_flange: np.ndarray) -> np.ndarray:
    return flange_pos + flange_ori @ point_in_flange


def step(redis_client: redis.Redis, state: State, pos: np.ndarray, dwell: float, msg: str) -> None:
    print(f"[{state.name:<16}] {msg}")
    print(f"  cmd  flange = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
    set_goal(redis_client, pos, ORI)
    time.sleep(dwell)
    actual_pos, actual_ori = read_actual_pose(redis_client)
    if actual_pos is not None:
        needle_tip = compute_world_from_flange(actual_pos, actual_ori, NEEDLE_IN_FLANGE)
        ball_pos   = compute_world_from_flange(actual_pos, actual_ori, JAWS_IN_FLANGE)
        err = actual_pos - pos
        print(f"  actl flange = ({actual_pos[0]:+.4f}, {actual_pos[1]:+.4f}, {actual_pos[2]:+.4f})"
              f"  err=({err[0]:+.4f}, {err[1]:+.4f}, {err[2]:+.4f})")
        print(f"  red ball    = ({ball_pos[0]:+.4f}, {ball_pos[1]:+.4f}, {ball_pos[2]:+.4f})")
        print(f"  needle tip  = ({needle_tip[0]:+.4f}, {needle_tip[1]:+.4f}, {needle_tip[2]:+.4f})")


def run_pierce(redis_client: redis.Redis, fy: float, ny: float, idx: int) -> None:
    print(f"\n=== Pierce {idx + 1} at needle y = {ny:+.2f} ===")

    step(redis_client, State.PIERCE_THROUGH,
         np.array([X_FLUSH_LEFT, fy, WORKING_Z]), MOVE_TIME,
         "driving needle through foam")
    time.sleep(DWELL)

    step(redis_client, State.RETURN_HOME,
         np.array([X_HOME, fy, WORKING_Z]), MOVE_TIME,
         "returning to home x")
    time.sleep(DWELL)


def main() -> None:
    redis_client = redis.Redis()

    config_file_name = redis_client.get(REDIS_KEYS.config_file_name)
    if config_file_name is None or config_file_name.decode("utf-8") != CONFIG_FILE_FOR_THIS_SCRIPT:
        print(f"This script expects OpenSai_main running with {CONFIG_FILE_FOR_THIS_SCRIPT}.")
        return

    while redis_client.get(REDIS_KEYS.active_controller).decode("utf-8") != CONTROLLER_TO_USE:
        redis_client.set(REDIS_KEYS.active_controller, CONTROLLER_TO_USE)

    print(f"WORKING_Z={WORKING_Z:.4f}, X_HOME={X_HOME:.4f}, X_FLUSH_LEFT={X_FLUSH_LEFT:.4f}")

    first_fy = FLANGE_Y[0]
    home = np.array([X_HOME, first_fy, HOME_Z])
    step(redis_client, State.GOTO_HOME, home, MOVE_TIME, "going to home")
    step(redis_client, State.HOLD_AT_HEIGHT, home, HEIGHT_HOLD, "holding at home")

    for i, (fy, ny) in enumerate(zip(FLANGE_Y, NEEDLE_Y_STITCHES)):
        run_pierce(redis_client, fy, ny, i)
        if i + 1 < len(FLANGE_Y):
            next_fy = FLANGE_Y[i + 1]
            step(redis_client, State.TRANSIT_Y,
                 np.array([X_HOME, next_fy, WORKING_Z]), MOVE_TIME,
                 f"transit to next stitch at needle y={NEEDLE_Y_STITCHES[i + 1]:+.2f}")

    print("\n[DONE] All pierces complete.")


if __name__ == "__main__":
    main()
