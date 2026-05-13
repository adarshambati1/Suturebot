"""
Suturebot full-stitch choreography — Grav-with-hemostat variant.

Per stitch:
   1. GOTO_HOME           — needle tip 75 mm from foam's -Y face
   2. HOLD_AT_HEIGHT      — settle
   3. INITIAL_PUNCTURE    — move +8 mm in Y toward foam
   4. REGRIP_LIFT         — lift up (release implied)
   5. REGRIP_BACK         — translate back to home y at lift height
   6. REGRIP_DESCEND      — descend back to working z at home y
   7. FULL_PUNCTURE       — drive +Y until forceps jaws flush with -Y face
   8. LIFT_OVER           — lift to clear foam top
   9. CROSS_TO_OTHER      — translate to +Y side at lift height
  10. DESCEND_OTHER       — descend to working z (jaws flush with +Y face)
  11. MOVE_AWAY           — +15 mm in Y, pulling needle clear
  12. LIFT_FOR_TRANSIT    — lift before X transit
  13. TRANSIT_X           — translate to next stitch x at home y, lift z
  14. DESCEND_FOR_NEXT    — descend to working z; loop

Geometry (flange to feature in world frame, under the orientation used here):
    JAWS_OFFSET       = (-0.0652, -0.0071, -0.2647)
    NEEDLE_TIP_OFFSET = (-0.0652, +0.0627, -0.2647)   # jaws + 69.81 mm in +Y

ORI puts flange-X along world +Y so the needle (link +X from the jaws)
points in world +Y, matching the foam's puncture axis.
"""

import json
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import redis


class State(Enum):
    GOTO_HOME         = auto()
    HOLD_AT_HEIGHT    = auto()
    INITIAL_PUNCTURE  = auto()
    REGRIP_LIFT       = auto()
    REGRIP_BACK       = auto()
    REGRIP_DESCEND    = auto()
    FULL_PUNCTURE     = auto()
    LIFT_OVER         = auto()
    CROSS_TO_OTHER    = auto()
    DESCEND_OTHER     = auto()
    MOVE_AWAY         = auto()
    LIFT_FOR_TRANSIT  = auto()
    TRANSIT_X         = auto()
    DESCEND_FOR_NEXT  = auto()


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation"
    )
    cartesian_task_current_position: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        "opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::current_orientation"
    )
    active_controller: str = "opensai::controllers::Rizon4s::active_controller_name"
    config_file_name: str  = "::sai-interfaces-webui::config_file_name"


REDIS_KEYS = RedisKeys()
CONFIG_FILE_FOR_THIS_SCRIPT = "suturebot_grav.xml"
CONTROLLER_TO_USE = "cartesian_controller"

# --- Offsets from flange to features (in WORLD frame, under ORI below).
# X component (-0.0152) calibrated against observed behavior: at flange_x =
# 0.5652 the needle tip lands at world x = 0.55, so flange→needle world X
# offset is -0.0152, not the -0.0652 the simple geometry suggested.
JAWS_OFFSET       = np.array([-0.0015, -0.0085, -0.2826])
NEEDLE_TIP_OFFSET = np.array([-0.0015, +0.0613, -0.2826])

# --- Foam (from world_suturebot_grav.urdf) ---------------------------------
FOAM_NEG_Y     = -0.00285        # foam -Y face
FOAM_POS_Y     = +0.00285        # foam +Y face
FOAM_TOP_Z     =  0.0305         # foam top
NEEDLE_Z       =  0.027          # working needle height — mid foam exposed strip

# --- Distances from spec --------------------------------------------------
HOME_GAP        = 0.075          # needle tip 75 mm from foam -Y face
PUNCTURE_INIT   = 0.008          # initial puncture motion (8 mm in +Y)
MOVE_AWAY_DIST  = 0.015          # post-stitch pull away (15 mm in +Y)
LIFT_RISE       = 0.05           # extra Z to clear foam during regrip / crossover

# --- Stitch X positions along foam length (world frame) -------------------
NEEDLE_X_STITCHES = [0.50, 0.55, 0.60]
FLANGE_X = [nx - NEEDLE_TIP_OFFSET[0] for nx in NEEDLE_X_STITCHES]   # +0.0652

# --- Derived flange-frame waypoints in Y / Z ------------------------------
WORKING_Z = NEEDLE_Z - NEEDLE_TIP_OFFSET[2]                # 0.3096
LIFT_Z    = WORKING_Z + LIFT_RISE                          # 0.3596
# HOME sits at the puncture height — same Z as WORKING_Z.
HOME_Z    = WORKING_Z

# Y waypoints (flange). Needle tip y = flange_y + NEEDLE_TIP_OFFSET[1];
# jaws y     = flange_y + JAWS_OFFSET[1].
def flange_y_for_needle_tip(needle_y: float) -> float:
    return needle_y - NEEDLE_TIP_OFFSET[1]

def flange_y_for_jaws(jaws_y: float) -> float:
    return jaws_y - JAWS_OFFSET[1]

# HOME_GAP measures distance from the JAWS (forceps tip / red ball) to the
# foam's -Y face, not the needle tip — the needle protrudes 69.81 mm forward
# of the jaws, so when the jaws are 75 mm from foam, the needle tip is
# already ~5 mm from foam.
Y_HOME            = flange_y_for_jaws(FOAM_NEG_Y - HOME_GAP)         # -0.06935
Y_INITIAL_PIERCE  = Y_HOME + PUNCTURE_INIT                            # -0.13255
# Wiggle on both sides: jaws stop 5 mm shy of the foam faces (not exactly
# flush). Avoids gripper-body interaction with the foam during descent.
LEFT_DESCEND_BUFFER  = 0.005
RIGHT_DESCEND_BUFFER = 0.005
Y_FLUSH_LEFT      = flange_y_for_jaws(FOAM_NEG_Y - LEFT_DESCEND_BUFFER)   # +0.00065
Y_FLUSH_RIGHT     = flange_y_for_jaws(FOAM_POS_Y) + RIGHT_DESCEND_BUFFER  # +0.01495
Y_AWAY            = Y_FLUSH_RIGHT + MOVE_AWAY_DIST                       # +0.02995

# --- Orientation: flange-X along world +Y (needle along world +Y) ---------
ORI = np.array([[0.0, 1.0,  0.0],
                [1.0, 0.0,  0.0],
                [0.0, 0.0, -1.0]])

# --- Timing ---------------------------------------------------------------
MOVE_TIME   = 2.0
HEIGHT_HOLD = 2.0
SHORT_MOVE  = 1.0
DWELL       = 0.5


def set_goal(redis_client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    redis_client.set(REDIS_KEYS.cartesian_task_goal_position, json.dumps(pos.tolist()))
    redis_client.set(REDIS_KEYS.cartesian_task_goal_orientation, json.dumps(ori.tolist()))


def read_actual_pose(redis_client: redis.Redis):
    """Read flange's actual world pose from redis."""
    pos_str = redis_client.get(REDIS_KEYS.cartesian_task_current_position)
    ori_str = redis_client.get(REDIS_KEYS.cartesian_task_current_orientation)
    if pos_str is None or ori_str is None:
        return None, None
    return (np.array(json.loads(pos_str.decode("utf-8"))),
            np.array(json.loads(ori_str.decode("utf-8"))))


# Jaws-tip position in flange frame (recovered from the world offset by
# inverting the script orientation). Same as the red ball's anchor.
JAWS_IN_FLANGE   = ORI.T @ JAWS_OFFSET
NEEDLE_IN_FLANGE = ORI.T @ NEEDLE_TIP_OFFSET


def compute_world_from_flange(flange_pos: np.ndarray, flange_ori: np.ndarray,
                              point_in_flange: np.ndarray) -> np.ndarray:
    """Transform a flange-frame point to world using the actual flange pose."""
    return flange_pos + flange_ori @ point_in_flange


def compute_needle_tip_world(flange_pos: np.ndarray, flange_ori: np.ndarray) -> np.ndarray:
    return compute_world_from_flange(flange_pos, flange_ori, NEEDLE_IN_FLANGE)


def compute_jaws_world(flange_pos: np.ndarray, flange_ori: np.ndarray) -> np.ndarray:
    return compute_world_from_flange(flange_pos, flange_ori, JAWS_IN_FLANGE)


def step(redis_client: redis.Redis, state: State, pos: np.ndarray, dwell: float, msg: str) -> None:
    print(f"[{state.name:<18}] {msg}")
    print(f"  cmd  flange = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
    set_goal(redis_client, pos, ORI)
    time.sleep(dwell)
    actual_pos, actual_ori = read_actual_pose(redis_client)
    if actual_pos is not None:
        needle_tip = compute_needle_tip_world(actual_pos, actual_ori)
        ball_pos = compute_jaws_world(actual_pos, actual_ori)
        err = actual_pos - pos
        print(f"  actl flange = ({actual_pos[0]:+.4f}, {actual_pos[1]:+.4f}, {actual_pos[2]:+.4f})"
              f"  err=({err[0]:+.4f}, {err[1]:+.4f}, {err[2]:+.4f})")
        print(f"  red ball    = ({ball_pos[0]:+.4f}, {ball_pos[1]:+.4f}, {ball_pos[2]:+.4f})")
        print(f"  needle tip  = ({needle_tip[0]:+.4f}, {needle_tip[1]:+.4f}, {needle_tip[2]:+.4f})")


def run_stitch(redis_client: redis.Redis, fx: float, nx: float, idx: int) -> None:
    print(f"\n=== Stitch {idx + 1} at needle x = {nx:.2f} ===")

    step(redis_client, State.INITIAL_PUNCTURE,
         np.array([fx, Y_INITIAL_PIERCE, WORKING_Z]), SHORT_MOVE,
         "initial 8 mm puncture")
    time.sleep(DWELL)

    step(redis_client, State.REGRIP_LIFT,
         np.array([fx, Y_INITIAL_PIERCE, LIFT_Z]), SHORT_MOVE,
         "regrip — lift up")
    step(redis_client, State.REGRIP_BACK,
         np.array([fx, Y_HOME, LIFT_Z]), MOVE_TIME,
         "regrip — back to home y at lift height (end of needle)")
    step(redis_client, State.REGRIP_DESCEND,
         np.array([fx, Y_HOME, WORKING_Z]), SHORT_MOVE,
         "regrip — descend to working z")
    time.sleep(DWELL)

    step(redis_client, State.FULL_PUNCTURE,
         np.array([fx, Y_FLUSH_LEFT, WORKING_Z]), MOVE_TIME,
         "full puncture — drive until forceps flush with -Y face")
    time.sleep(DWELL)

    step(redis_client, State.LIFT_OVER,
         np.array([fx, Y_FLUSH_LEFT, LIFT_Z]), SHORT_MOVE,
         "lift over foam")
    step(redis_client, State.CROSS_TO_OTHER,
         np.array([fx, Y_FLUSH_RIGHT, LIFT_Z]), SHORT_MOVE,
         "cross to mirror flush position on +Y side")
    step(redis_client, State.DESCEND_OTHER,
         np.array([fx, Y_FLUSH_RIGHT, WORKING_Z]), SHORT_MOVE,
         "descend on +Y side — forceps flush with +Y face")
    time.sleep(DWELL)

    step(redis_client, State.MOVE_AWAY,
         np.array([fx, Y_AWAY, WORKING_Z]), SHORT_MOVE,
         "move 15 mm away from foam")


def transit_to_next(redis_client: redis.Redis, fx_curr: float, fx_next: float, nx_next: float) -> None:
    step(redis_client, State.LIFT_FOR_TRANSIT,
         np.array([fx_curr, Y_AWAY, LIFT_Z]), SHORT_MOVE,
         "lift for transit")
    step(redis_client, State.TRANSIT_X,
         np.array([fx_next, Y_HOME, LIFT_Z]), MOVE_TIME,
         f"transit to next stitch at needle x={nx_next:.2f}")
    step(redis_client, State.DESCEND_FOR_NEXT,
         np.array([fx_next, Y_HOME, HOME_Z]), SHORT_MOVE,
         "descend to home for next stitch")


def main() -> None:
    redis_client = redis.Redis()

    config_file_name = redis_client.get(REDIS_KEYS.config_file_name)
    if config_file_name is None or config_file_name.decode("utf-8") != CONFIG_FILE_FOR_THIS_SCRIPT:
        print(f"This script expects OpenSai_main running with {CONFIG_FILE_FOR_THIS_SCRIPT}.")
        return

    while redis_client.get(REDIS_KEYS.active_controller).decode("utf-8") != CONTROLLER_TO_USE:
        redis_client.set(REDIS_KEYS.active_controller, CONTROLLER_TO_USE)

    print(f"WORKING_Z={WORKING_Z:.4f}, LIFT_Z={LIFT_Z:.4f}")
    print(f"Y_HOME={Y_HOME:.4f}, Y_INITIAL_PIERCE={Y_INITIAL_PIERCE:.4f}, "
          f"Y_FLUSH_LEFT={Y_FLUSH_LEFT:.4f}, Y_FLUSH_RIGHT={Y_FLUSH_RIGHT:.4f}, "
          f"Y_AWAY={Y_AWAY:.4f}")

    first_fx = FLANGE_X[0]
    home = np.array([first_fx, Y_HOME, HOME_Z])
    step(redis_client, State.GOTO_HOME, home, MOVE_TIME,
         "going to home")
    step(redis_client, State.HOLD_AT_HEIGHT, home, HEIGHT_HOLD,
         "holding at home height")

    for i, (fx, nx) in enumerate(zip(FLANGE_X, NEEDLE_X_STITCHES)):
        run_stitch(redis_client, fx, nx, i)
        if i + 1 < len(FLANGE_X):
            transit_to_next(redis_client, fx, FLANGE_X[i + 1], NEEDLE_X_STITCHES[i + 1])

    print("\n[DONE] All stitches complete. Holding final pose.")


if __name__ == "__main__":
    main()
