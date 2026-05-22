#!/usr/bin/env bash
# Tail the three teleop Redis keys at ~5 Hz so you can eyeball that the
# Unity bridge (or fake_quest_publisher.py) is writing what you expect.
#
# Usage:
#   ./monitor_teleop.sh            # localhost
#   ./monitor_teleop.sh 192.168.0.5  # remote redis (e.g. Rizon PC)

set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-6379}"

K_POS="opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position"
K_ORI="opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation"
K_GRIP="opensai::controllers::Rizon4s::cartesian_controller::gripper_fingers::goal_position"

if ! command -v redis-cli >/dev/null 2>&1; then
  echo "redis-cli not found. Install with: brew install redis" >&2
  exit 1
fi

echo "Polling Redis at ${HOST}:${PORT} every 200 ms. Ctrl-C to stop."
echo

while true; do
  pos=$(redis-cli -h "$HOST" -p "$PORT" GET "$K_POS"  || echo "<unset>")
  ori=$(redis-cli -h "$HOST" -p "$PORT" GET "$K_ORI"  || echo "<unset>")
  grip=$(redis-cli -h "$HOST" -p "$PORT" GET "$K_GRIP" || echo "<unset>")
  printf "\r\033[Kpos=%-40s grip=%s" "$pos" "$grip"
  sleep 0.2
done
