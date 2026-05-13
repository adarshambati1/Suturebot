import redis
import time
import numpy as np
import matplotlib.pyplot as plt

r = redis.Redis()

# your actual keys
Q_KEY   = "opensai::sensors::Titania::joint_positions"
TAU_KEY = "opensai::sensors::Titania::joint_torques"

def read(key):
    val = r.get(key)
    if val is None:
        return None
    return [float(x) for x in val.decode().strip().split()]

# ── record ──
times, positions, torques = [], [], []

print("Recording... move the robot and pierce the needle.")
print("Press Ctrl+C to stop.\n")

t0 = time.time()
try:
    while True:
        q = read(Q_KEY)
        tau = read(TAU_KEY)
        if q and tau:
            times.append(time.time() - t0)
            positions.append(q)
            torques.append(tau)
        time.sleep(0.01)  # 100 Hz
except KeyboardInterrupt:
    pass

print(f"Recorded {len(times)} samples over {times[-1]:.1f}s")

times = np.array(times)
positions = np.array(positions)
torques = np.array(torques)
n_joints = positions.shape[1]

# ── save raw data ──
np.savez("needle_pierce_data.npz", times=times, positions=positions, torques=torques)
print("Saved to needle_pierce_data.npz")

# ── plot ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# joint positions
for j in range(n_joints):
    axes[0].plot(times, np.degrees(positions[:, j]), label=f"J{j}")
axes[0].set_ylabel("Joint Position (deg)")
axes[0].set_title("Joint Trajectories During Needle Pierce")
axes[0].legend(loc="upper right", ncol=n_joints)
axes[0].grid(True, alpha=0.3)

# joint torques
for j in range(n_joints):
    axes[1].plot(times, torques[:, j], label=f"J{j}")
axes[1].set_ylabel("Joint Torque (Nm)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Joint Torques — Look for Spike at Needle Insertion")
axes[1].legend(loc="upper right", ncol=n_joints)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("needle_pierce_plot.png", dpi=150)
plt.show()

print("Saved plot to needle_pierce_plot.png")