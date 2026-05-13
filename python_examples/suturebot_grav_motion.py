"""
Suturebot V2 — Rizon4s + Grav gripper, with foam base + foam block scene.

Per-stitch motion (the gripper actually opens/closes at GRIP/RELEASE):
   1. GOTO_HOME        — over the -Y home, at puncture height, gripper open
   2. GRIP             — close gripper on (imaginary) needle
   3. PARTIAL_PIERCE   — move toward block a little (drive needle partway in)
   4. RELEASE          — open gripper
   5. GO_UP            — rise at the partial-pierce y
   6. REPOSITION_BACK  — translate back over the home y at lift height
   7. DESCEND_TO_HOME  — descend back at home y
   8. REGRIP           — close gripper (regrip back end of needle)
   9. FULL_PIERCE      — drive flush with the -Y block face
  10. RELEASE_OVER     — open gripper above the embedded needle
  11. LIFT_OVER        — rise above block
  12. CROSS            — translate to flush position on the +Y side
  13. DESCEND_OTHER    — descend on the +Y side
  14. REGRIP_TIP       — close gripper on the protruding needle tip
  15. MOVE_AWAY        — pull horizontally out to +Y
  16. RELEASE_DONE     — open gripper after the pull-through

Between stitches:
  17. LIFT_FOR_TRANSIT — rise from the +Y "away" pose
  18. TRANSIT_TO_NEXT  — translate to next stitch's -Y home at lift height
  19. DESCEND_FOR_NEXT — descend at next stitch's -Y home

The cartesian task is on Rizon4s `closed_fingers_tcp` (the center of the
closed jaws), so XYZ goals are in the same frame as where the gripped
needle sits. Gripper width is commanded via the `gripper_fingers` joint
task on `finger_width_joint` (0 m closed, 0.10 m open).

Usage (in two terminals, from the OpenSai/ directory):
    # terminal 1
    sh scripts/launch.sh suturebot.xml
    # terminal 2
    python python_examples/suturebot_grav_motion.py
"""

import json
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import redis


class State(Enum):
    GOTO_HOME        = auto()
    HOLD_AT_HEIGHT   = auto()
    GRIP             = auto()
    PARTIAL_PIERCE   = auto()
    RELEASE          = auto()
    GO_UP            = auto()
    REPOSITION_BACK  = auto()
    DESCEND_TO_HOME  = auto()
    REGRIP           = auto()
    FULL_PIERCE      = auto()
    RELEASE_OVER     = auto()
    LIFT_OVER        = auto()
    CROSS            = auto()
    DESCEND_OTHER    = auto()
    REGRIP_TIP       = auto()
    MOVE_AWAY        = auto()
    RELEASE_DONE     = auto()
    LIFT_FOR_TRANSIT = auto()
    TRANSIT_TO_NEXT  = auto()
    DESCEND_FOR_NEXT = auto()


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation"
    )
    gripper_task_goal_position: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::gripper_fingers::goal_position"
    )
    active_controller: str = "opensai::controllers::Rizon4s::active_controller_name"
    config_file_name: str  = "::sai-interfaces-webui::config_file_name"


REDIS_KEYS = RedisKeys()
CONFIG_FILE_FOR_THIS_SCRIPT = "suturebot.xml"
CONTROLLER_TO_USE = "cartesian_controller"

# --- Geometry ---------------------------------------------------------------
# FoamBlock (sits in FoamBase recess, extends above):
#   center (0.55, 0, 0.03), size 0.10 (X) x 0.05 (Y) x 0.06 (Z)
#   -Y face y=-0.025, +Y face y=+0.025, top z=0.06
WORKING_Z = 0.03   # flange at mid-block (no gripper offset)
LIFT_Z    = 0.15   # flange above block top (block top at 0.06)

# Y waypoints (TCP y positions). The gripper is camera-down and its
# fingers spread along world X (perpendicular to motion), so Y motion is
# clear of the gripper body.
Y_HOME       = -0.20   # -Y side, far from block (start / "back end of needle")
Y_PARTIAL    = -0.12   # partial-pierce stop
Y_FLUSH_LEFT = -0.055  # flush with -Y block face (face at y=-0.025)
Y_FLUSH_RIGHT = +0.055 # flush with +Y block face
Y_AWAY       = +0.20   # +Y side, far from block after the stitch is done

# Stitch X positions (length of foam base, 13 cm along world X).
# Block X range is [0.50, 0.60]; three stitches at 4 cm spacing.
STITCH_X_POSITIONS = [0.51, 0.55, 0.59]

# --- Orientation ------------------------------------------------------------
# Flange-Z aligned with world -Z (camera down); additionally rotated 90°
# about Z so the Grav gripper's grip axis (gripper-X) lies along world Y,
# i.e. a held needle is along the puncture direction.
ORI_DOWN = np.array([[ 0.0, -1.0,  0.0],
                     [-1.0,  0.0,  0.0],
                     [ 0.0,  0.0, -1.0]])

# --- Gripper widths (m) -----------------------------------------------------
# finger_width_joint range: 0 (fully closed) to 0.10 (fully open).
GRIP_OPEN   = np.array([0.05])    # 5 cm open — clear of any needle
GRIP_CLOSED = np.array([0.005])   # 5 mm — gripped on a thin needle

# --- Timing -----------------------------------------------------------------
MOVE_TIME    = 2.0   # standard motion segment
SHORT_MOVE   = 1.5   # shorter segment for small in-plane moves
GRIPPER_TIME = 1.0   # time for gripper to open/close
HEIGHT_HOLD  = 2.0   # initial pause at home before any horizontal motion


def set_goal(redis_client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    redis_client.set(REDIS_KEYS.cartesian_task_goal_position, json.dumps(pos.tolist()))
    redis_client.set(REDIS_KEYS.cartesian_task_goal_orientation, json.dumps(ori.tolist()))


def set_gripper(redis_client: redis.Redis, width: np.ndarray) -> None:
    # The Grav is currently loaded via Rizon4s_Grav_fixed.urdf with every
    # gripper joint welded fixed, so there is no gripper_fingers task to
    # write to. This call is a no-op; left in place so the choreography
    # reads the same once an actuated gripper variant is swapped back in.
    _ = width  # intentionally unused
    return


def move(redis_client: redis.Redis, state: State, pos: np.ndarray, dwell: float, msg: str) -> None:
    print(f"[{state.name}] {msg}")
    set_goal(redis_client, pos, ORI_DOWN)
    time.sleep(dwell)


def grip(redis_client: redis.Redis, state: State, closed: bool, msg: str) -> None:
    print(f"[{state.name}] {msg}")
    set_gripper(redis_client, GRIP_CLOSED if closed else GRIP_OPEN)
    time.sleep(GRIPPER_TIME)


def run_stitch(redis_client: redis.Redis, x: float, stitch_idx: int) -> None:
    """One stitch at a single X position: pierce in two stages with a
    regrip in between, cross over to the far side, regrip the tip, pull
    out, all with the gripper opening and closing at each grip point."""
    print(f"\n=== Stitch {stitch_idx + 1} at x = {x:.2f} ===")

    # 2. Grip the (imaginary) needle at home
    grip(redis_client, State.GRIP, closed=True,
         "closing gripper on needle")

    # 3. Partial pierce — move toward block a little
    move(redis_client, State.PARTIAL_PIERCE,
         np.array([x, Y_PARTIAL, WORKING_Z]), SHORT_MOVE,
         "partial pierce — driving needle a little into block")

    # 4. Release
    grip(redis_client, State.RELEASE, closed=False,
         "opening gripper to release needle")

    # 5. Go up
    move(redis_client, State.GO_UP,
         np.array([x, Y_PARTIAL, LIFT_Z]), MOVE_TIME,
         "going up to reposition")

    # 6. Reposition back to home y at lift height
    move(redis_client, State.REPOSITION_BACK,
         np.array([x, Y_HOME, LIFT_Z]), MOVE_TIME,
         "repositioning back over home y (end of imaginary needle)")

    # 7. Descend
    move(redis_client, State.DESCEND_TO_HOME,
         np.array([x, Y_HOME, WORKING_Z]), MOVE_TIME,
         "descending — back at the end of needle")

    # 8. Regrip
    grip(redis_client, State.REGRIP, closed=True,
         "closing gripper — fresh grip on back end of needle")

    # 9. Full pierce — flush with -Y block face
    move(redis_client, State.FULL_PIERCE,
         np.array([x, Y_FLUSH_LEFT, WORKING_Z]), MOVE_TIME,
         "full pierce — driving flush with -Y block face")

    # 10. Release before crossing over
    grip(redis_client, State.RELEASE_OVER, closed=False,
         "opening gripper — needle stays embedded in block")

    # 11. Lift over
    move(redis_client, State.LIFT_OVER,
         np.array([x, Y_FLUSH_LEFT, LIFT_Z]), MOVE_TIME,
         "lifting above block")

    # 12. Cross to +Y side
    move(redis_client, State.CROSS,
         np.array([x, Y_FLUSH_RIGHT, LIFT_Z]), MOVE_TIME,
         "crossing to +Y side")

    # 13. Descend on +Y side
    move(redis_client, State.DESCEND_OTHER,
         np.array([x, Y_FLUSH_RIGHT, WORKING_Z]), MOVE_TIME,
         "descending on +Y side, flush with block")

    # 14. Regrip the protruding needle tip
    grip(redis_client, State.REGRIP_TIP, closed=True,
         "closing gripper on protruding needle tip")

    # 15. Pull horizontally away from block
    move(redis_client, State.MOVE_AWAY,
         np.array([x, Y_AWAY, WORKING_Z]), MOVE_TIME,
         "pulling needle out to +Y side")

    # 16. Release after pull-through
    grip(redis_client, State.RELEASE_DONE, closed=False,
         "opening gripper — stitch complete")


def transit_to_next(redis_client: redis.Redis, x_curr: float, x_next: float) -> None:
    """Reposition to the next stitch's home pose along the foam base."""
    move(redis_client, State.LIFT_FOR_TRANSIT,
         np.array([x_curr, Y_AWAY, LIFT_Z]), MOVE_TIME,
         f"lifting for transit to next stitch at x={x_next:.2f}")
    move(redis_client, State.TRANSIT_TO_NEXT,
         np.array([x_next, Y_HOME, LIFT_Z]), MOVE_TIME,
         "transiting across the block top to next stitch's home")
    move(redis_client, State.DESCEND_FOR_NEXT,
         np.array([x_next, Y_HOME, WORKING_Z]), MOVE_TIME,
         "descending to home pose for next stitch")


def main() -> None:
    redis_client = redis.Redis()

    config_file_name = redis_client.get(REDIS_KEYS.config_file_name)
    if config_file_name is None or config_file_name.decode("utf-8") != CONFIG_FILE_FOR_THIS_SCRIPT:
        print(f"This script expects OpenSai_main running with {CONFIG_FILE_FOR_THIS_SCRIPT}.")
        return

    while redis_client.get(REDIS_KEYS.active_controller).decode("utf-8") != CONTROLLER_TO_USE:
        redis_client.set(REDIS_KEYS.active_controller, CONTROLLER_TO_USE)

    # 1. Go to first stitch's home pose with gripper open
    first_x = STITCH_X_POSITIONS[0]
    initial_home = np.array([first_x, Y_HOME, WORKING_Z])
    set_gripper(redis_client, GRIP_OPEN)
    move(redis_client, State.GOTO_HOME, initial_home, MOVE_TIME,
         "going to home pose with gripper open")
    move(redis_client, State.HOLD_AT_HEIGHT, initial_home, HEIGHT_HOLD,
         "holding at puncture height before first stitch")

    for i, x in enumerate(STITCH_X_POSITIONS):
        run_stitch(redis_client, x, i)
        if i + 1 < len(STITCH_X_POSITIONS):
            transit_to_next(redis_client, x, STITCH_X_POSITIONS[i + 1])

    print("\n[DONE] All stitches complete. Holding final pose.")


if __name__ == "__main__":
    main()
