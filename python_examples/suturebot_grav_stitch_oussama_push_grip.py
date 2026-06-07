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

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

import redis


# Toggle: "Rizon4s" for sim, "Titania" for the real Flexiv driver.
ROBOT_NAME = "Titania"

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
    cartesian_task_sensed_force: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::sensed_force"
    )
    set_gripper_mode: str = (
        f"opensai::commands::{ROBOT_NAME}::gripper::mode"
    )   
    ft_sensor_tcp_force: str  = f"opensai::sensors::{ROBOT_NAME}::ft_sensor::tcp::force"
    ft_sensor_tcp_moment: str = f"opensai::sensors::{ROBOT_NAME}::ft_sensor::tcp::moment"
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


FOAM_THICKNESS = 0.0057   # in X direction
FOAM_CENTER_X  =  0.738 - 0.165 # +0.30 m in EE X (EE X = world -X)
FOAM_NEG_X     = FOAM_CENTER_X - FOAM_THICKNESS / 2 #Jaws approach this side
FOAM_POS_X     = FOAM_CENTER_X + FOAM_THICKNESS / 2 #Needle pulled out from this side
NEEDLE_Z       =  0.018


APPROACH_BUFFER = 0.012   # jaws stop 5 mm shy of foam -X face
# --- Distances from spec --------------------------------------------------
HOME_GAP        = 0.055         # needle tip 75 mm from foam -x face
PUNCTURE_INIT   = 0.008          # initial puncture motion (8 mm in +x)
MOVE_AWAY_DIST  = 0.075          # post-stitch pull away (30 mm in +x)
LIFT_RISE       = 0.05           # extra Z to clear foam during regrip / crossover
TRANSIT_LIFT_RISE = 0.20         # extra Z for the pull-away lift after each stitch
# Individual post-stitch lift rises (above WORKING_Z) after the 1st, 2nd, 3rd stitch.
TRANSIT_LIFT_RISES = [0.3, 0.2, 0.15]
FINAL_LIFT_RISE   = 0.05         # lift rise after the last (4th) stitch
PIERCE_ADVANCE    = 0.005        # +X: advance 5 mm further in stitching dir before piercing
REGRIP_BACK       = 0.030        # -X: base regrip back-off along the needle
REGRIP_BACK_EXTRA = 0.003 # -X: back off 15 mm more so the needle is gripped nearer its end




NEEDLE_Y_STITCHES = [+0.15, +0.20, +0.25, +0.30]
TRAJECTORY_Y_OFFSET = 0.225  # shift the whole trajectory +235 mm in world +Y
FLANGE_Y = [ny - NEEDLE_TIP_OFFSET[1] - .5 + TRAJECTORY_Y_OFFSET for ny in NEEDLE_Y_STITCHES]

WORKING_Z       = NEEDLE_Z - NEEDLE_TIP_OFFSET[2]
LIFT_Z          = WORKING_Z + LIFT_RISE
TRANSIT_LIFT_Z  = WORKING_Z + TRANSIT_LIFT_RISE
HOME_Z          = WORKING_Z

def flange_x_for_jaws(jaws_x: float) -> float:
    return jaws_x - JAWS_OFFSET[0]

def flange_y_for_jaws(jaws_y: float) -> float:
    return jaws_y - JAWS_OFFSET[1]

# Approach from -X side; drive +X to pierce outward.
X_HOME        = flange_x_for_jaws(FOAM_NEG_X - HOME_GAP)
X_FLUSH_LEFT  = flange_x_for_jaws(FOAM_NEG_X - APPROACH_BUFFER)
X_INITIAL_PIERCE  = X_HOME + PUNCTURE_INIT
LEFT_DESCEND_BUFFER  = 0.000
RIGHT_DESCEND_BUFFER = 0.01
X_FLUSH_LEFT      = flange_x_for_jaws(FOAM_NEG_X - LEFT_DESCEND_BUFFER)
X_FLUSH_RIGHT     = flange_x_for_jaws(FOAM_POS_X) + RIGHT_DESCEND_BUFFER
X_AWAY            = X_FLUSH_RIGHT + MOVE_AWAY_DIST






ORI = np.array([[-1.0,  0.0,  0.0],
                [0.0, 1.0,  0.0],
                [0.0,  0.0, -1.0]])

MOVE_TIME   = 2.0
HEIGHT_HOLD = 2.0
SHORT_MOVE  = 1.0
DWELL       = 0.5


class ForceLogger(threading.Thread):
    """Background poller for one or more 3-vector Redis keys. Stores time-
    aligned samples and state-transition markers; saves .npz + .png."""

    def __init__(self, redis_client: redis.Redis, keys: dict, sample_hz: int = 100):
        """keys is a dict mapping label -> redis key (e.g.
        {"tcp_force": "opensai::sensors::...::ft_sensor::tcp::force", ...})"""
        super().__init__(daemon=True)
        self._redis = redis_client
        self._keys = keys
        self._period = 1.0 / sample_hz
        self._stop_event = threading.Event()
        self._t0 = None
        self.times = []
        self.series = {label: [] for label in keys}   # label -> list of [x,y,z]
        self.markers = []                              # list of (t, state_name)

    def start(self):
        self._t0 = time.monotonic()
        super().start()

    def run(self):
        nan_vec = [float("nan")] * 3
        while not self._stop_event.is_set():
            t = time.monotonic() - self._t0
            row = {}
            for label, key in self._keys.items():
                raw = self._redis.get(key)
                if raw is None:
                    row[label] = nan_vec
                    continue
                try:
                    v = json.loads(raw.decode("utf-8"))
                    row[label] = v if isinstance(v, list) and len(v) == 3 else nan_vec
                except (ValueError, TypeError):
                    row[label] = nan_vec
            self.times.append(t)
            for label, v in row.items():
                self.series[label].append(v)
            time.sleep(self._period)

    def mark(self, state_name: str):
        if self._t0 is not None:
            self.markers.append((time.monotonic() - self._t0, state_name))

    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)

    def save(self, out_dir: str, tag: str):
        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(out_dir, f"{tag}.npz")
        png_path = os.path.join(out_dir, f"{tag}.png")
        t = np.array(self.times)
        arrays = {label: (np.array(vals) if vals else np.zeros((0, 3)))
                  for label, vals in self.series.items()}
        mt = np.array([m[0] for m in self.markers], dtype=float)
        ml = np.array([m[1] for m in self.markers], dtype=object)
        np.savez(npz_path, t=t, markers_t=mt, markers_label=ml, **arrays)
        print(f"  saved {npz_path}  ({len(t)} samples, {len(self._keys)} series)")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not installed; skipping .png")
            return

        labels = list(self._keys.keys())
        n = len(labels)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, label in zip(axes, labels):
            data = arrays[label]
            unit = "N·m" if "moment" in label else "N"
            if len(t) > 0 and data.shape[0] == len(t):
                ax.plot(t, data[:, 0], label=f"{label}_x", color="C0", linewidth=1)
                ax.plot(t, data[:, 1], label=f"{label}_y", color="C1", linewidth=1)
                ax.plot(t, data[:, 2], label=f"{label}_z", color="C2", linewidth=1)
                mag = np.linalg.norm(np.nan_to_num(data), axis=1)
                ax.plot(t, mag, label=f"|{label}|", color="k", linewidth=0.8, alpha=0.4)
            ymin, ymax = ax.get_ylim()
            for mt_i, ml_i in self.markers:
                ax.axvline(mt_i, color="gray", linestyle="--", alpha=0.4, linewidth=0.5)
                ax.text(mt_i, ymax, str(ml_i), rotation=90, fontsize=7,
                        va="top", ha="right", alpha=0.6)
            ax.set_ylabel(f"{label} ({unit})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        axes[0].set_title(f"Force / moment traces - {tag}")
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  saved {png_path}")


# Module-level handle so step() can mark state transitions.
_FORCE_LOGGER: "ForceLogger | None" = None

# Position log — populated by step() when --pp is passed.
_POS_LOG: list = []          # list of dicts: t, state, cmd, actual, needle_tip, jaws
_POS_LOG_ENABLED: bool = False
_POS_LOG_T0: "float | None" = None


def _write_goal(redis_client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    """Raw goal write. Does NOT register the goal with the pause manager."""
    redis_client.set(REDIS_KEYS.cartesian_task_goal_position, json.dumps(pos.tolist()))
    redis_client.set(REDIS_KEYS.cartesian_task_goal_orientation, json.dumps(ori.tolist()))


def set_goal(redis_client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    _write_goal(redis_client, pos, ori)
    # Remember the last commanded goal so a pause can hold position and a
    # resume can re-issue this goal.
    if _PAUSE is not None:
        _PAUSE.set_current_goal(pos, ori)


def set_gripper_mode(redis_client: redis.Redis, mode: str) -> None:
    redis_client.set(REDIS_KEYS.set_gripper_mode, mode)


def read_actual_pose(redis_client: redis.Redis):
    pos_str = redis_client.get(REDIS_KEYS.cartesian_task_current_position)
    ori_str = redis_client.get(REDIS_KEYS.cartesian_task_current_orientation)
    if pos_str is None or ori_str is None:
        return None, None
    return (np.array(json.loads(pos_str.decode("utf-8"))),
            np.array(json.loads(ori_str.decode("utf-8"))))


# --- Pause / resume -------------------------------------------------------
# Press PAUSE_KEY in the terminal to pause; press it again to resume.
PAUSE_KEY = " "          # spacebar
PAUSE_KEY_LABEL = "SPACE"


class PauseManager(threading.Thread):
    """Background single-key listener that pauses/resumes the trajectory.

    On pause it commands the robot to hold its current *actual* pose (so it
    stops where it is); on resume it re-issues the goal that was in progress.
    The trajectory blocks via ``wait_if_paused`` (called from ``pausable_sleep``)
    so no new motion is commanded while paused.
    """

    def __init__(self, redis_client: redis.Redis, pause_key: str = PAUSE_KEY):
        super().__init__(daemon=True)
        self._redis = redis_client
        self._pause_key = pause_key
        self._paused = threading.Event()        # set => paused
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._current_goal = None               # (pos, ori) last commanded

    # -- goal bookkeeping --------------------------------------------------
    def set_current_goal(self, pos: np.ndarray, ori: np.ndarray) -> None:
        with self._lock:
            self._current_goal = (np.array(pos, copy=True), np.array(ori, copy=True))

    def is_paused(self) -> bool:
        return self._paused.is_set()

    # -- pause / resume transitions ---------------------------------------
    def toggle(self) -> None:
        if self._paused.is_set():
            self._resume()
        else:
            self._pause()

    def _pause(self) -> None:
        actual_pos, actual_ori = read_actual_pose(self._redis)
        if actual_pos is not None:
            _write_goal(self._redis, actual_pos, actual_ori)
        self._paused.set()
        print(f"\n[PAUSED] holding position — press {PAUSE_KEY_LABEL} to resume")

    def _resume(self) -> None:
        with self._lock:
            goal = self._current_goal
        if goal is not None:
            _write_goal(self._redis, goal[0], goal[1])
        self._paused.clear()
        print(f"[RESUMED] continuing — press {PAUSE_KEY_LABEL} to pause")

    def wait_if_paused(self) -> None:
        """Block while paused, returning once running again."""
        while self._paused.is_set() and not self._stop.is_set():
            time.sleep(0.05)

    # -- keyboard listener -------------------------------------------------
    def run(self) -> None:
        if not sys.stdin.isatty():
            print("[pause] stdin is not a TTY; pause/resume disabled.")
            return
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == self._pause_key:
                        self.toggle()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)

    def stop(self) -> None:
        self._stop.set()
        self._paused.clear()


# Module-level handle so set_goal()/pausable_sleep() can reach the manager.
_PAUSE: "PauseManager | None" = None


def pausable_sleep(duration: float) -> None:
    """Sleep ``duration`` seconds of *running* time, blocking while paused.

    Time spent paused does not count against ``duration``."""
    remaining = duration
    tick = 0.05
    while remaining > 0:
        if _PAUSE is not None:
            _PAUSE.wait_if_paused()
        dt = min(tick, remaining)
        time.sleep(dt)
        remaining -= dt


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
    global _POS_LOG_T0
    # Don't issue a new goal while paused — wait here until resumed.
    if _PAUSE is not None:
        _PAUSE.wait_if_paused()
    print(f"[{state.name:<18}] {msg}")
    print(f"  cmd  flange = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
    if _POS_LOG_ENABLED and _POS_LOG_T0 is None:
        _POS_LOG_T0 = time.monotonic()
    set_goal(redis_client, pos, ORI)
    pausable_sleep(dwell)
    actual_pos, actual_ori = read_actual_pose(redis_client)
    if actual_pos is not None:
        needle_tip = compute_needle_tip_world(actual_pos, actual_ori)
        ball_pos = compute_jaws_world(actual_pos, actual_ori)
        err = actual_pos - pos
        print(f"  actl flange = ({actual_pos[0]:+.4f}, {actual_pos[1]:+.4f}, {actual_pos[2]:+.4f})"
              f"  err=({err[0]:+.4f}, {err[1]:+.4f}, {err[2]:+.4f})")
        print(f"  red ball    = ({ball_pos[0]:+.4f}, {ball_pos[1]:+.4f}, {ball_pos[2]:+.4f})")
        print(f"  needle tip  = ({needle_tip[0]:+.4f}, {needle_tip[1]:+.4f}, {needle_tip[2]:+.4f})")
        if _POS_LOG_ENABLED:
            t = time.monotonic() - _POS_LOG_T0
            _POS_LOG.append({
                "t": t,
                "state": state.name,
                "cmd": pos.copy(),
                "actual": actual_pos.copy(),
                "needle_tip": needle_tip.copy(),
                "jaws": ball_pos.copy(),
            })


POS_LOG_DIR = "log_files/position_logs"

def save_position_log(out_dir: str, tag: str) -> None:
    if not _POS_LOG:
        print("  no position data to save")
        return
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{tag}.npz")
    png_path = os.path.join(out_dir, f"{tag}.png")

    t      = np.array([r["t"]          for r in _POS_LOG])
    states = np.array([r["state"]       for r in _POS_LOG], dtype=object)
    cmd    = np.array([r["cmd"]         for r in _POS_LOG])
    actual = np.array([r["actual"]      for r in _POS_LOG])
    needle = np.array([r["needle_tip"]  for r in _POS_LOG])
    jaws   = np.array([r["jaws"]        for r in _POS_LOG])

    np.savez(npz_path, t=t, states=states, cmd=cmd, actual=actual,
             needle_tip=needle, jaws=jaws)
    print(f"  saved {npz_path}  ({len(t)} steps)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping .png")
        return

    series = [("commanded flange", cmd), ("actual flange", actual),
              ("needle tip",       needle), ("jaws",         jaws)]
    axes_labels = ["X (m)", "Y (m)", "Z (m)"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = ["C0", "C1", "C2", "C3"]
    for (label, data), color in zip(series, colors):
        for ax_i, ax in enumerate(axes):
            ax.plot(t, data[:, ax_i], label=label, color=color, linewidth=1)

    # mark state transitions
    prev = None
    for rec in _POS_LOG:
        if rec["state"] != prev:
            for ax in axes:
                ax.axvline(rec["t"], color="gray", linestyle="--", alpha=0.4, linewidth=0.5)
            axes[0].text(rec["t"], axes[0].get_ylim()[1], rec["state"],
                         rotation=90, fontsize=7, va="top", ha="right", alpha=0.6)
            prev = rec["state"]

    for ax, ylabel in zip(axes, axes_labels):
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"End-effector trajectory — {tag}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"  saved {png_path}")




def run_stitch(redis_client: redis.Redis, fy: float, nx: float, idx: int) -> None:
    print(f"\n=== Stitch {idx + 1} at needle x = {nx:.2f} ===")
   

    if idx != 0:
        # If not the first stitch, we will regrip

        # initial puncture, advanced PIERCE_ADVANCE further in the stitching (+X) dir
        x_pierce  = X_INITIAL_PIERCE + 0.045 + PIERCE_ADVANCE
        x_regrip  = x_pierce - (REGRIP_BACK + REGRIP_BACK_EXTRA)

        #lift up
        step(redis_client, State.INITIAL_PUNCTURE,
             np.array([x_pierce, fy, WORKING_Z]), SHORT_MOVE,
             "initial puncture (advanced 5 mm in +X)")
        pausable_sleep(DWELL)

        set_gripper_mode(redis_client, "o")
        pausable_sleep(2)

        step(redis_client, State.LIFT_OVER,
         np.array([x_pierce, fy, LIFT_Z]), SHORT_MOVE,
         "lift over foam")

        #move back 15 mm more in -X so the needle is gripped nearer its end
        step(redis_client, State.REGRIP_LIFT,
            np.array([x_regrip, fy, LIFT_Z]), SHORT_MOVE,
            "regrip — back off along needle (-X)")
        pausable_sleep(DWELL)

        #move down
        step(redis_client, State.REGRIP_DESCEND,
            np.array([x_regrip, fy, WORKING_Z]), SHORT_MOVE,
            "regrip — descend to working z")
        pausable_sleep(DWELL)

        set_gripper_mode(redis_client, "g")
        pausable_sleep(2)

        #step(redis_client, State.REGRIP_BACK,
            #np.array([X_HOME, fy, LIFT_Z]), MOVE_TIME,
            #"regrip — back to home y at lift height (end of needle)")
        #time.sleep(DWELL)

        #step(redis_client, State.REGRIP_DESCEND,
            #np.array([X_HOME, fy,WORKING_Z]), SHORT_MOVE,
            #"regrip — descend to working z")
        #time.sleep(DWELL)

        #set_gripper_mode(redis_client, "g")
        #time.sleep(2)

    else:

        step(redis_client, State.INITIAL_PUNCTURE,
         np.array([X_INITIAL_PIERCE + PIERCE_ADVANCE, fy, WORKING_Z]), SHORT_MOVE,
         "initial 8 mm puncture (advanced 5 mm in +X)")
        pausable_sleep(DWELL)
        

    step(redis_client, State.FULL_PUNCTURE,
         np.array([X_FLUSH_LEFT, fy, WORKING_Z]), MOVE_TIME,
         "full puncture — drive until forceps flush with -X face")
    
    set_gripper_mode(redis_client, "o")
    pausable_sleep(2)

    step(redis_client, State.LIFT_OVER,
         np.array([X_FLUSH_LEFT, fy, LIFT_Z]), SHORT_MOVE,
         "lift over foam")
    step(redis_client, State.CROSS_TO_OTHER,
         np.array([X_FLUSH_RIGHT, fy, LIFT_Z]), SHORT_MOVE,
         "cross to mirror flush position on +Y side")
    step(redis_client, State.DESCEND_OTHER,
         np.array([X_FLUSH_RIGHT, fy, WORKING_Z]), SHORT_MOVE,
         "descend on +Y side — forceps flush with +Y face")
    pausable_sleep(DWELL)

    set_gripper_mode(redis_client, "g")
    pausable_sleep(2)

    step(redis_client, State.MOVE_AWAY,
         np.array([X_AWAY, fy, WORKING_Z]), SHORT_MOVE,
         "move 15 mm away from foam")


def transit_to_next(redis_client: redis.Redis, fy_curr: float, fy_next: float, nx_next: float, lift_z: float) -> None:
    step(redis_client, State.LIFT_FOR_TRANSIT,
         np.array([X_AWAY, fy_curr, lift_z]), SHORT_MOVE,
         "lift for transit")
    pausable_sleep(3.0)
    step(redis_client, State.TRANSIT_X,
         np.array([X_HOME, fy_next, LIFT_Z]), MOVE_TIME,
         f"transit to next stitch at needle x={nx_next:.2f}")
    step(redis_client, State.DESCEND_FOR_NEXT,
         np.array([X_HOME, fy_next, HOME_Z]), SHORT_MOVE,
         "descend to home for next stitch")


def main() -> None:
    global _POS_LOG_ENABLED
    global _PAUSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf", "--plot-forces",    action="store_true",
                        help="log force/moment and save .npz + .png")
    parser.add_argument("--pp", "--plot-positions", action="store_true",
                        help="log x,y,z positions and save .npz + .png")
    args = parser.parse_args()
    _POS_LOG_ENABLED = args.pp

    redis_client = redis.Redis()

    config_file_name = redis_client.get(REDIS_KEYS.config_file_name)
    if config_file_name is None or config_file_name.decode("utf-8") != CONFIG_FILE_FOR_THIS_SCRIPT:
        print(f"This script expects OpenSai_main running with {CONFIG_FILE_FOR_THIS_SCRIPT}.")
        return

    while redis_client.get(REDIS_KEYS.active_controller).decode("utf-8") != CONTROLLER_TO_USE:
        redis_client.set(REDIS_KEYS.active_controller, CONTROLLER_TO_USE)

    # Start the pause/resume key listener.
    _PAUSE = PauseManager(redis_client)
    _PAUSE.start()
    print(f"[pause] press {PAUSE_KEY_LABEL} at any time to pause/resume")

    print(f"WORKING_Z={WORKING_Z:.4f}, LIFT_Z={LIFT_Z:.4f}")
    print(f"X_HOME={X_HOME:.4f}, X_INITIAL_PIERCE={X_INITIAL_PIERCE:.4f}, "
          f"X_FLUSH_LEFT={X_FLUSH_LEFT:.4f}, X_FLUSH_RIGHT={X_FLUSH_RIGHT:.4f}, "
          f"X_AWAY={X_AWAY:.4f}")

    try:
        first_fy = FLANGE_Y[0]
        home = np.array([X_HOME, first_fy, HOME_Z])
        step(redis_client, State.GOTO_HOME, home, MOVE_TIME,
             "going to home")
        step(redis_client, State.HOLD_AT_HEIGHT, home, HEIGHT_HOLD,
             "holding at home height")

        # Per-stitch lift heights: explicit value after the 1st, 2nd, 3rd stitch.
        transit_lift_heights = [WORKING_Z + r for r in TRANSIT_LIFT_RISES]

        for i, (fy, ny) in enumerate(zip(FLANGE_Y, NEEDLE_Y_STITCHES)):
            run_stitch(redis_client, fy, ny, i)
            if i + 1 < len(FLANGE_Y):
                transit_to_next(redis_client, fy, FLANGE_Y[i + 1], NEEDLE_Y_STITCHES[i + 1],
                                transit_lift_heights[i])

        final_fy = FLANGE_Y[-1]
        step(redis_client, State.LIFT_FOR_TRANSIT,
             np.array([X_AWAY, final_fy, WORKING_Z + FINAL_LIFT_RISE]), SHORT_MOVE,
             "final lift after last stitch")
        pausable_sleep(3.0)

        print("\n[DONE] All stitches complete. Holding final pose.")
    finally:
        if _PAUSE is not None:
            _PAUSE.stop()

    if _POS_LOG_ENABLED:
        import time as _time
        tag = f"positions_{int(_time.time())}"
        print(f"\nSaving position log (tag={tag}) ...")
        save_position_log(POS_LOG_DIR, tag)


if __name__ == "__main__":
    main()
