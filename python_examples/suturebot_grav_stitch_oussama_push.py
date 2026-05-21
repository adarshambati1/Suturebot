"""
Stitch choreography using the Oussama-push orientation and offsets.

This script reuses the full-stitch state machine from
`suturebot_grav_stitch.py` but keeps the flange/needle orientation and
offsets from the Oussama Push variant so the needle axis and flange
geometry match that rotated setup.
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import redis

# Toggle: "Rizon4s" for sim, "Titania" for the real Flexiv driver.
ROBOT_NAME = "Rizon4s"

# Force logging is opt-in via --pf / --plot-forces on the CLI.
FORCE_LOG_DIR   = "log_files/force_logs"
FORCE_SAMPLE_HZ = 100

# Matching xml config that OpenSai_main must be running.
CONFIG_FILE_FOR_THIS_SCRIPT = (
    "suturebot_grav_real.xml" if ROBOT_NAME == "Titania" else "suturebot_grav_oussama_push.xml"
)


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
    set_gripper_mode: str = (
        "opensai::commands::Rizon4s::gripper::mode"
    )
    active_controller: str = "opensai::controllers::Rizon4s::active_controller_name"
    config_file_name: str  = "::sai-interfaces-webui::config_file_name"


REDIS_KEYS = RedisKeys()

CONTROLLER_TO_USE = "cartesian_controller"

# Force logging
FORCE_LOG_DIR   = "log_files/force_logs"
FORCE_SAMPLE_HZ = 100

# --- Offsets from flange to features: use the Oussama Push values ---------
JAWS_OFFSET       = np.array([-0.0085, +0.0015, -0.2826])
NEEDLE_TIP_OFFSET = np.array([+0.0613, +0.0015, -0.2826])

# --- Foam (keep stitch-style naming) ------------------------------------
FOAM_NEG_Y     = -0.00285
FOAM_POS_Y     = +0.00285
FOAM_TOP_Z     =  0.0305
NEEDLE_Z       =  0.032

# --- Distances -----------------------------------------------------------
HOME_GAP        = 0.075
PUNCTURE_INIT   = 0.008
MOVE_AWAY_DIST  = 0.015
LIFT_RISE       = 0.05

# --- Stitch positions along the foam (needle X positions in original)
NEEDLE_X_STITCHES = [0.50, 0.55, 0.60]
FLANGE_X = [nx - NEEDLE_TIP_OFFSET[0] for nx in NEEDLE_X_STITCHES]

# --- Derived flange-frame Z / waypoints ---------------------------------
WORKING_Z = NEEDLE_Z - NEEDLE_TIP_OFFSET[2]
LIFT_Z    = WORKING_Z + LIFT_RISE
HOME_Z    = WORKING_Z

def flange_y_for_needle_tip(needle_y: float) -> float:
    return needle_y - NEEDLE_TIP_OFFSET[1]

def flange_y_for_jaws(jaws_y: float) -> float:
    return jaws_y - JAWS_OFFSET[1]

# Home and waypoint Ys (using same semantics as stitch script)
Y_HOME            = flange_y_for_jaws(FOAM_NEG_Y - HOME_GAP)
Y_INITIAL_PIERCE  = Y_HOME + PUNCTURE_INIT
LEFT_DESCEND_BUFFER  = 0.005
RIGHT_DESCEND_BUFFER = 0.005
Y_FLUSH_LEFT      = flange_y_for_jaws(FOAM_NEG_Y - LEFT_DESCEND_BUFFER)
Y_FLUSH_RIGHT     = flange_y_for_jaws(FOAM_POS_Y) + RIGHT_DESCEND_BUFFER
Y_AWAY            = Y_FLUSH_RIGHT + MOVE_AWAY_DIST

# --- Orientation: use Oussama Push ORI ---------------------------------
ORI = np.array([[1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]])

# --- Timing ---------------------------------------------------------------
MOVE_TIME   = 2.0
HEIGHT_HOLD = 2.0
SHORT_MOVE  = 1.0
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


# Module-level handle so step() can mark state transitions when enabled
_FORCE_LOGGER: "ForceLogger | None" = None


class ForceLogger(threading.Thread):
    """Background poller for sensed_force. Stores (t, [Fx,Fy,Fz]) samples
    and state-transition markers; saves .npz + .png on stop."""

    def __init__(self, redis_client: redis.Redis, key: str, sample_hz: int = 100):
        super().__init__(daemon=True)
        self._redis = redis_client
        self._key = key
        self._period = 1.0 / sample_hz
        self._stop = threading.Event()
        self._t0 = None
        self.times = []
        self.forces = []
        self.markers = []   # list of (t, state_name)

    def start(self):
        self._t0 = time.monotonic()
        super().start()

    def run(self):
        while not self._stop.is_set():
            raw = self._redis.get(self._key)
            if raw is not None:
                try:
                    f = json.loads(raw.decode("utf-8"))
                    self.times.append(time.monotonic() - self._t0)
                    self.forces.append(f)
                except (ValueError, TypeError):
                    pass
            time.sleep(self._period)

    def mark(self, state_name: str):
        if self._t0 is not None:
            self.markers.append((time.monotonic() - self._t0, state_name))

    def stop(self):
        self._stop.set()
        self.join(timeout=1.0)

    def save(self, out_dir: str, tag: str):
        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(out_dir, f"{tag}.npz")
        png_path = os.path.join(out_dir, f"{tag}.png")
        t = np.array(self.times)
        f = np.array(self.forces) if self.forces else np.zeros((0, 3))
        mt = np.array([m[0] for m in self.markers], dtype=float)
        ml = np.array([m[1] for m in self.markers], dtype=object)
        np.savez(npz_path, t=t, forces=f, markers_t=mt, markers_label=ml)
        print(f"  saved {npz_path}  ({len(t)} samples)")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not installed; skipping .png")
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        if len(t) > 0:
            ax.plot(t, f[:, 0], label="Fx", color="C0", linewidth=1)
            ax.plot(t, f[:, 1], label="Fy", color="C1", linewidth=1)
            ax.plot(t, f[:, 2], label="Fz", color="C2", linewidth=1)
            ax.plot(t, np.linalg.norm(f, axis=1), label="|F|",
                    color="k", linewidth=0.8, alpha=0.5)
        ymin, ymax = (ax.get_ylim() if len(t) > 0 else (-1, 1))
        for mt_i, ml_i in self.markers:
            ax.axvline(mt_i, color="gray", linestyle="--", alpha=0.4, linewidth=0.5)
            ax.text(mt_i, ymax, str(ml_i), rotation=90, fontsize=7,
                    va="top", ha="right", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N, world frame)")
        ax.set_title(f"Sensed flange forces - {tag}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  saved {png_path}")


def set_gripper_mode(redis_client: redis.Redis, mode: str) -> None:
    redis_client.set(REDIS_KEYS.set_gripper_mode, mode)


def run_stitch(redis_client: redis.Redis, fx: float, nx: float, idx: int) -> None:
    print(f"\n=== Stitch {idx + 1} at needle x = {nx:.2f} ===")

    step(redis_client, State.INITIAL_PUNCTURE,
         np.array([fx, Y_INITIAL_PIERCE, WORKING_Z]), SHORT_MOVE,
         "initial 8 mm puncture")
    time.sleep(DWELL)


    step(redis_client, State.FULL_PUNCTURE,
         np.array([fx, Y_FLUSH_LEFT, WORKING_Z]), MOVE_TIME,
         "full puncture — drive until forceps flush with -Y face")
    time.sleep(DWELL)

    set_gripper_mode(redis_client, "o")
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

    set_gripper_mode(redis_client, "g")
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
    global _FORCE_LOGGER

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf", "--plot-forces", dest="plot_forces", action="store_true",
                        help="Log sensed force during the run and save .npz + .png")
    args = parser.parse_args()

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

    if args.plot_forces:
        _FORCE_LOGGER = ForceLogger(redis_client, REDIS_KEYS.cartesian_task_sensed_force,
                                    sample_hz=FORCE_SAMPLE_HZ)
        _FORCE_LOGGER.start()
        print(f"[force logger] sampling {REDIS_KEYS.cartesian_task_sensed_force} at {FORCE_SAMPLE_HZ} Hz")

    try:
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
    finally:
        if _FORCE_LOGGER is not None:
            _FORCE_LOGGER.stop()
            tag = f"stitch_oussama_{time.strftime('%Y%m%d_%H%M%S')}"
            _FORCE_LOGGER.save(FORCE_LOG_DIR, tag)


if __name__ == "__main__":
    main()
