# vr_teleop

Scaffolding for Meta Quest 3 teleoperation of the Rizon4s, with an Intel RealSense D405 as the in-headset camera view. Designed so you can develop most of it on your MacBook now and finish on the Rizon PC later.

## Architecture

```
RealSense D405 ─USB─┐
                    ├─→ PC: Unity app ─Quest Link─→ Quest 3
Quest controllers ──┘            │
                                 ↓
                              Redis ──→ controller.cpp ──→ Rizon4s
```

The Unity app writes to the **same Redis keys** that [suturebot_grav_motion.py](../python_examples/suturebot_grav_motion.py) writes to today. From the controller's point of view nothing changes — a different process is just producing the goal stream.

| Key | Format |
| --- | --- |
| `opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_position` | JSON `[x, y, z]` (meters, robot base frame) |
| `opensai::controllers::Rizon4s::cartesian_controller::cartesian_task::goal_orientation` | JSON 3×3 row-major rotation matrix |
| `opensai::controllers::Rizon4s::cartesian_controller::gripper_fingers::goal_position` | JSON `[width]` (meters) |

## Folder layout

```
vr_teleop/
├── README.md                          ← you are here
├── docs/
│   └── SETUP_MAC.md                   ← Unity + Meta XR Simulator install walkthrough
├── unity_scripts/
│   └── RedisTeleopBridge.cs           ← drop into Assets/Scripts in the Unity project
├── test_harness/
│   ├── fake_quest_publisher.py        ← streams pose like the Quest would, no headset needed
│   └── monitor_teleop.sh              ← tail the three teleop keys in a terminal
└── SuturebotVR/                     ← created by Unity Hub (gitignored)
```

## Quickstart on the Mac (no headset, no robot)

```bash
# 1. Start a local Redis
brew install redis
brew services start redis

# 2. Stream fake controller poses
python vr_teleop/test_harness/fake_quest_publisher.py

# 3. In another terminal, watch the writes
vr_teleop/test_harness/monitor_teleop.sh
```

You should see `pos=[...]` updating at 60 Hz, with the gripper width toggling every 4 seconds.

When that works, run your existing SAI sim on top of the same Redis instance — the foam-block scene should follow the fake controller's lissajous motion. **This is the test that says "the pipeline is real" before any Unity work happens.**

## Next: Unity on the Mac

Follow [docs/SETUP_MAC.md](docs/SETUP_MAC.md). It walks through:

1. Installing Unity Hub + Unity 2022.3 LTS
2. Creating the `SuturebotVR/` and adding the Meta XR All-in-One SDK
3. Enabling the Meta XR Simulator (so you can "play" the scene without a headset)
4. Adding StackExchange.Redis via NuGetForUnity
5. (Optional) Adding librealsense + the Unity wrapper for D405 frames in the editor
6. Dropping [unity_scripts/RedisTeleopBridge.cs](unity_scripts/RedisTeleopBridge.cs) in and wiring up the inspector

## Porting to the Rizon PC

Once it works in the simulator on the Mac:

1. Copy `SuturebotVR/` to the PC (or push via git — Library/Temp are gitignored).
2. Install Unity 2022.3 LTS + the **Meta Quest Link** desktop app (Windows only).
3. Connect Quest 3 over USB-C, enable Link from inside the headset.
4. Plug the D405 into the PC.
5. Update `redisHost` on the bridge component to point at the Rizon PC's Redis.
6. Hit Play.

## Safety notes baked into the bridge

- **Deadman**: pose writes only happen while the right grip button is held. Release → robot freezes at the last commanded pose.
- **Motion scaling**: `motionScale` defaults to `0.3`, so 10 cm of hand motion → 3 cm of robot motion. Tune down for finer suturing work.
- **Workspace clamp**: hand-tunable XYZ box, defaults sized around the foam block. The bridge clamps before writing so a wild controller swing can't send the EE outside the table.
- **Orientation held fixed** at the camera-down `ORI_DOWN` matrix for the first pass. Adding wrist-orientation streaming is a follow-up — needs another calibration pass and is risky to add before the position loop is trusted.
