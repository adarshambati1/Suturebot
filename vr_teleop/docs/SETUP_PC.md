# Moving to the Rizon PC (Windows + Quest Link)

Once the Mac side is working (RedisTeleopBridge driving the robot in OpenSai sim — proven on the Mac with the Meta XR Simulator), this is what you do on the PC the Rizon4s is wired to.

## Prerequisites on the PC

1. **Windows 10/11** (Quest Link is Windows-only).
2. **Unity Hub** with **the same Unity 6 LTS** version you used on the Mac. Project files reference that exact version; a mismatch will force a re-import that can break.
   - Skip Android Build Support on the PC unless you also want to sideload APKs — Quest Link doesn't need it.
3. **Meta Quest Link desktop app** — <https://www.meta.com/quest/setup/> → download for Windows. Install + log in with your Meta account.
4. **Quest 3 cable**: any USB-C cable that supports data + power (the official Meta Link cable works best; most USB-C 3.0+ cables also work).
5. **OpenSai** built on the PC, same checkout as Mac. The launch flow is identical.
6. **Redis** running locally on the PC — OpenSai's `launch.sh` auto-starts it.

## Pull the repo

```cmd
cd path\to\where\you\want\it
git clone <your-repo-url> Suturebot
cd Suturebot
git pull
```

Or, if you already have a checkout, `git pull` in the existing directory.

## Open the project in Unity

1. Unity Hub → **Open** → navigate to `Suturebot\vr_teleop\SuturebotVR` → Open.
2. Unity rebuilds Library/ on first launch (the gitignored 3+ GB cache). This takes 2-10 min.
3. **NuGetForUnity packages** (StackExchange.Redis etc.) come in via `Assets/Packages/` — they're committed, so no extra restore.
4. **Meta XR SDK** packages come in via Unity Package Manager (`Packages/manifest.json` references them). They auto-fetch on first open.

## Wire up to the local Redis

1. Open `Assets/Scenes/SampleScene`.
2. Select **RedisBridge** in the Hierarchy.
3. In the Inspector, **Redis Host**: leave as `127.0.0.1` (OpenSai's Redis is on the same PC).
4. Scene preset, motion scale, workspace clamp — whatever you tuned on the Mac.

## Connect the Quest 3

1. Plug the Quest 3 into the PC via USB-C.
2. Put on the headset.
3. Inside the headset, accept the "Allow access to data" prompt that pops up on plug-in.
4. From the headset's quick settings: **Quest Link → Launch Quest Link**. (Or pre-launch via the Meta Quest Link desktop app on the PC.)
5. You should now see your PC desktop as a virtual screen inside the headset.

## Tell Unity to use the real headset, not the Simulator

The toolbar dropdown that you used on Mac to switch to "Meta XR Simulator" lives in the same place on PC:

1. Top toolbar, look for the Meta XR Simulator dropdown / button (same one you found on Mac).
2. Change it from **Simulator** to **Real Headset / OpenXR Runtime** (or whatever your version calls it).
3. Confirm in **Edit → Project Settings → XR Plug-in Management → OpenXR** that **OpenXR** is checked for Standalone — same as Mac.

## Run the integration test

In three windows on the PC, same pattern as Mac:

1. **OpenSai** with the oussama_push XML (or whichever scene you want to drive).
2. **redis-cli MONITOR** or `monitor_teleop.sh` if you want to see writes (`scripts/monitor_teleop.sh` works on Windows via Git Bash or WSL; otherwise just use redis-cli directly).
3. **Unity Editor** → hit Play.

When you put the headset on, you should see the OVRCameraRig view (the empty scene, just sky). Hold the right-hand **grip button** on the Touch controller (NOT a keyboard key — the physical button under your middle finger). The bridge engages:

- Robot in OpenSai snaps to the scene's home pose
- As you move your right hand in space, the robot follows at 0.3× scale (per the inspector's Motion Scale)
- Pulling the right index trigger → gripper width updates from 0.05 m to 0.005 m

Release the grip → robot freezes.

## What's different from Mac

- Real controller pose / buttons instead of keyboard fakes
- Quest Link adds ~20 ms latency vs. wired editor play — fine for teleop
- The Game tab in the editor now shows what you see in the headset (in real time)

## Common PC-specific snags

- **Quest Link won't activate** → close Meta Quest Link app, replug cable, re-launch. Sometimes the USB enumeration is flaky on first connect.
- **Headset shows black/no Unity content** → in the Meta Quest Link desktop app, confirm OpenXR runtime is set to "Meta Quest Link" (not SteamVR). The Meta Link app has a setting for this.
- **Controller pose drifts when you walk** → expected if Guardian boundary isn't set up. Set one in the headset (Quick Settings → Boundary).
- **Robot lurches violently on first grip-press** → orientation or home-pose mismatch. Confirm Scene preset in the inspector matches the XML OpenSai is running.
