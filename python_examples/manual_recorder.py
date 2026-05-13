import redis
import time
import json
import csv

r = redis.Redis()

# adjust these key names to match your actual Redis keys
# you can discover them with: redis-cli KEYS "*flexiv*" or KEYS "*robot*"
JOINT_POS_KEY = "sai2::FlexivRizon::sensors::q"      # joint positions
JOINT_TORQUE_KEY = "sai2::FlexivRizon::sensors::tau"  # joint torques

records = []

print("Recording... press Ctrl+C to stop")
try:
    while True:
        q = r.get(JOINT_POS_KEY)
        tau = r.get(JOINT_TORQUE_KEY)
        
        if q and tau:
            # Redis values are typically space-separated floats
            q_vals = [float(x) for x in q.decode().strip().split()]
            tau_vals = [float(x) for x in tau.decode().strip().split()]
            
            record = {
                "time": time.time(),
                "joint_positions": q_vals,
                "joint_torques": tau_vals
            }
            records.append(record)
            print(f"q: {[f'{v:.4f}' for v in q_vals]}")
        
        time.sleep(0.01)  # 100 Hz sampling

except KeyboardInterrupt:
    print(f"\nRecorded {len(records)} samples")

# save to CSV
with open("recorded_trajectory.csv", "w", newline="") as f:
    writer = csv.writer(f)
    n_joints = len(records[0]["joint_positions"])
    header = ["time"] + [f"q{i}" for i in range(n_joints)] + [f"tau{i}" for i in range(n_joints)]
    writer.writerow(header)
    for rec in records:
        writer.writerow([rec["time"]] + rec["joint_positions"] + rec["joint_torques"])

print("Saved to recorded_trajectory.csv")
