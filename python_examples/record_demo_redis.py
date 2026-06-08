"""Record a hand-guided stitch demo over Redis (no Flexiv RDK, no serial number).

Reads the live joint stream that the Titania driver publishes to Redis and logs
it at a fixed rate while you hand-guide the arm. Saves in the SAME .npz format
playback_smooth.py expects (times, q, gripper, robot), so the recording replays
with no changes.

Use this instead of freedrive_record.py when you'd rather not touch the RDK:
  * enable hand-guiding / free-drive on the teach pendant yourself,
  * make sure the Titania driver is up and publishing
    opensai::sensors::Titania::joint_positions,
  * run this and move the arm by hand through the ideal stitch.

The clamp is the external clamp toggled over Redis (same key the other clients
use), so g/b here both ACTUATE it and log the state at that instant.

Keys (run in a REAL terminal — raw keypresses):
  g  close clamp (grip)      b  open clamp (release)      x  stop and save

    python3 python_examples/record_demo_redis.py
    python3 python_examples/record_demo_redis.py --robot Titania --rate 100

Saves to log_files/demos/demo_<n>.npz. Replay with:
    python3 python_examples/playback_smooth.py log_files/demos/demo_<n>.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import select
import sys
import termios
import time
import tty

import numpy as np
import redis

DEMO_DIR = "log_files/demos"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="Titania",
                    help="OpenSai/Redis robot name (default Titania)")
    ap.add_argument("--rate", type=float, default=100.0, help="record rate Hz (default 100)")
    ap.add_argument("--host", default="127.0.0.1", help="Redis host")
    ap.add_argument("--port", type=int, default=6379, help="Redis port")
    return ap.parse_args()


def get_key():
    """Non-blocking single keypress, or None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def parse_vec(raw):
    """Parse a Redis joint vector as a JSON array (OpenSai's format, e.g.
    "[0.1, 0.2, ...]") or whitespace-separated floats. Returns a list or None."""
    if raw is None:
        return None
    s = raw.decode().strip()
    try:
        v = json.loads(s)
        return [float(x) for x in v]
    except Exception:
        pass
    try:
        return [float(x) for x in s.split()]
    except Exception:
        return None


def main():
    args = parse_args()
    if not sys.stdin.isatty():
        sys.exit("Run in a real terminal (this reads raw keypresses).")

    r = redis.Redis(host=args.host, port=args.port)
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        sys.exit(f"Cannot reach Redis at {args.host}:{args.port}.")

    q_key = f"opensai::sensors::{args.robot}::joint_positions"
    grip_key = f"opensai::commands::{args.robot}::gripper::mode"

    # Verify the joint stream is actually being published (and parseable).
    first = parse_vec(r.get(q_key))
    if first is None:
        sys.exit(f"No parseable joint stream at '{q_key}'. Is the {args.robot} "
                 "driver running and publishing joint_positions?")
    n_joints = len(first)
    print(f"Recording {n_joints}-joint stream from '{q_key}' @ {args.rate:.0f} Hz.")
    print("Enable hand-guiding on the pendant, then move the arm by hand.")
    print("  g = close clamp,  b = open clamp,  x = stop and save")

    times, qs, gripper = [], [], []
    closed = False
    period = 1.0 / args.rate
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    t0 = time.monotonic()
    next_t = t0
    try:
        tty.setcbreak(fd)
        while True:
            k = get_key()
            if k == "g":
                closed = True
                r.set(grip_key, "g")
                print("  clamp -> CLOSED")
            elif k == "b":
                closed = False
                r.set(grip_key, "o")
                print("  clamp -> open")
            elif k == "x":
                break

            q = parse_vec(r.get(q_key))
            if q is not None and len(q) == n_joints:
                times.append(time.monotonic() - t0)
                qs.append(q)
                gripper.append(closed)
                if len(qs) % int(args.rate) == 0:
                    print(f"  {len(qs)} samples, {times[-1]:.1f}s, "
                          f"clamp={'closed' if closed else 'open'}", end="\r")

            next_t += period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()   # fell behind; resync
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    if not qs:
        sys.exit("\nNo samples recorded — was the joint stream live?")

    os.makedirs(DEMO_DIR, exist_ok=True)
    n = len(glob.glob(os.path.join(DEMO_DIR, "demo_*.npz"))) + 1
    path = os.path.join(DEMO_DIR, f"demo_{n}.npz")
    np.savez(path, times=np.array(times), q=np.array(qs),
             gripper=np.array(gripper), robot=args.robot)
    print(f"\nsaved {len(qs)} samples over {times[-1]:.1f}s -> {path}")
    print(f"replay: python3 python_examples/playback_smooth.py {path}")


if __name__ == "__main__":
    main()
