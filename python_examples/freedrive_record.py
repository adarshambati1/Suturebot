"""Hand-guide the real Rizon4s and record the demo -- via the Flexiv RDK directly
(NOT OpenSai/Redis), so it works in free-float without auto mode.

This script:
  1. connects to the robot (Flexiv RDK),
  2. puts it in FLOATING / hand-guiding so you can move it by hand,
  3. records joint angles at 100 Hz,
  4. lets you open/close the gripper with keys (logged),
  5. saves in the SAME format playback_smooth.py expects (times, q, gripper).

    python3 python_examples/freedrive_record.py --sn Rizon4s-XXXXXX
        g = close gripper   b = open gripper   x = stop and save
    (--no-float: don't command floating; you enable hand-guiding on the pendant
     and this just records.)

!!! VERIFY against YOUR flexivrdk version -- the API changed across releases.
    The calls marked [RDK] below are the ones to check (mode/primitive names,
    states accessor, gripper class). And the gripper here assumes a Flexiv-
    recognized gripper; if your Robotiq is on its own driver, wire grip() to it.
"""
import argparse
import glob
import os
import select
import sys
import termios
import time
import tty

import numpy as np

try:
    import flexivrdk
except ImportError:
    sys.exit("flexivrdk not installed. Install the Flexiv RDK Python package first.")

DEMO_DIR = "log_files/demos"
RATE_HZ = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn", required=True, help="robot serial number, e.g. Rizon4s-062077")
    ap.add_argument("--no-float", action="store_true",
                    help="don't command floating; enable hand-guiding yourself, just record")
    args = ap.parse_args()
    if not sys.stdin.isatty():
        sys.exit("run in a real terminal (raw keypresses).")

    # --- connect + enable [RDK] ---
    robot = flexivrdk.Robot(args.sn)
    if robot.fault():
        robot.ClearFault(); time.sleep(2)
    robot.Enable()
    print("enabling...", end="", flush=True)
    while not robot.operational():
        time.sleep(0.5); print(".", end="", flush=True)
    print(" operational.")

    # --- floating / hand-guiding [RDK] ---
    if not args.no_float:
        robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("Floating", {})    # [RDK] verify primitive name/params
        print("FLOATING -- move the arm by hand.")
    else:
        print("enable hand-guiding on the pendant now; recording...")

    # --- gripper [RDK] (assumes Flexiv-recognized gripper; wire to Robotiq if not) ---
    try:
        gripper = flexivrdk.Gripper(robot)
    except Exception:
        gripper = None
        print("(no RDK gripper; g/b will only be LOGGED, not actuated -- wire grip() to your Robotiq)")

    def grip(close):
        if gripper is None:
            return
        try:
            if close:
                gripper.Grasp(20)          # [RDK] grasp force (N)
            else:
                gripper.Move(0.1, 0.1, 20) # [RDK] open width/speed/force
        except Exception as e:
            print(f"\n(gripper command failed: {e})")

    def read_q():
        # [RDK] joint positions: v1.x -> robot.states().q ; older -> read_robot_states()
        return list(robot.states().q)

    times, qs, gripper_state = [], [], []
    closed = False
    period = 1.0 / RATE_HZ
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    print("recording @100Hz... g/b gripper, x to save.")
    t0 = time.time()
    try:
        tty.setcbreak(fd)
        while True:
            q = read_q()
            times.append(time.time() - t0); qs.append(q); gripper_state.append(1 if closed else 0)
            sys.stdout.write(f"\r t={time.time()-t0:6.1f}s  samples={len(qs)}  grip={'CLOSED' if closed else 'open'}   ")
            sys.stdout.flush()
            if select.select([sys.stdin], [], [], period)[0]:
                c = sys.stdin.read(1)
                if c == 'g': closed = True; grip(True)
                elif c == 'b': closed = False; grip(False)
                elif c == 'x': break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old); print()
        try:
            robot.Stop()
        except Exception:
            pass

    if len(qs) < 2:
        print("nothing recorded."); return
    os.makedirs(DEMO_DIR, exist_ok=True)
    n = len(glob.glob(os.path.join(DEMO_DIR, "demo_*.npz"))) + 1
    path = os.path.join(DEMO_DIR, f"demo_{n}.npz")
    np.savez(path, times=np.array(times), q=np.array(qs), gripper=np.array(gripper_state), robot="Titania")
    print(f"saved {len(qs)} samples over {times[-1]:.1f}s -> {path}")


if __name__ == "__main__":
    main()
