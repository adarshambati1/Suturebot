# Setting up Unity + Meta XR Simulator on macOS

This is the dev environment you use **before** you have access to the Rizon PC. It lets you build and iterate on the VR-teleop Unity app without ever putting on the headset, then port to the Rizon PC + Quest Link later.

What you will NOT have on Mac:

- **Quest Link / Air Link** — Meta doesn't ship a macOS client. You can't drive a real Quest from your Mac as a tethered display.
- **OpenXR runtime that talks to a physical Quest** — same reason.

What you WILL have:

- Unity Editor running on macOS, with the Meta XR Simulator faking a Quest at the OpenXR layer so you can play the scene in the editor.
- Intel RealSense SDK (via Homebrew) — if you have a D405, you can plug it into the Mac and see frames inside the Unity scene.
- Local Redis — `controller.cpp`-side integration testable end-to-end using the existing SAI sim on the Mac.

---

## 1. Unity Hub + Editor

1. Download **Unity Hub** for macOS: <https://unity.com/download>
2. Open Unity Hub → **Installs** → **Install Editor** → pick **Unity 2022.3 LTS** (Apple Silicon if you're on M-series).
3. In the modules step, check:
   - **Android Build Support** (and its sub-modules: OpenJDK, Android SDK & NDK Tools) — you'll need this to build APKs later for direct-sideload-to-Quest workflows.
   - **Mac Build Support (Mono)** — for editor play mode.

   You don't need iOS support.

Install takes ~15 min depending on bandwidth.

## 2. Create the project

1. Unity Hub → **Projects** → **New project**.
2. Template: **3D (Built-in Render Pipeline)** — keep it simple; HDRP/URP can be added later.
3. Project name: `SuturebotVR`.
4. Location: `~/Documents/cs225a_project/Suturebot/vr_teleop/SuturebotVR` (so it lives next to this README — the `SuturebotVR/` folder is already created and gitignored).

## 3. Add the Meta XR packages

Inside the Unity Editor:

1. **Window → Package Manager**.
2. Top-left dropdown: **Unity Registry**.
3. Install:
   - **XR Plugin Management**
   - **OpenXR Plugin**
4. Top-left dropdown: **+ → Add package by name…**, enter `com.meta.xr.sdk.all` (the Meta XR All-in-One SDK). This pulls in Core SDK, Interaction SDK, and the Simulator.
   - If you prefer minimal: install `com.meta.xr.sdk.core` and `com.meta.xr.simulator` separately.
5. **Edit → Project Settings → XR Plug-in Management** → enable **OpenXR** for the Standalone tab. Under **OpenXR** → **Interaction Profiles**, add **Oculus Touch Controller Profile**.
6. **Edit → Project Settings → Meta XR** — run the "Fix all" button it offers; it will warn about any missing settings.

## 4. Enable the Meta XR Simulator

The Simulator lets you "play" the scene as if a Quest were attached.

1. **Meta → Meta XR Simulator → Activate** (menu item appears once the package is installed).
2. Hit **Play** in the editor. A second window opens showing a virtual headset view. Mouse + WASD moves the head, the on-screen UI exposes controller buttons.
3. Deactivate later via **Meta → Meta XR Simulator → Deactivate** if you want to play without it.

Reference: <https://developer.oculus.com/documentation/unity/xrsim-intro/>

## 5. Add the Redis client

Unity's package manager doesn't ship StackExchange.Redis directly. Two options:

**Option A: NuGetForUnity (easiest)**

1. Install NuGetForUnity: Package Manager → + → Add package from git URL → `https://github.com/GlitchEnzo/NuGetForUnity.git?path=/src/NuGetForUnity`
2. Once installed, menu **NuGet → Manage NuGet Packages** → search `StackExchange.Redis` → Install.

**Option B: Drop the DLL in**

1. Download `StackExchange.Redis` NuGet package, extract the .NET Standard 2.0 DLL.
2. Drop it into `Assets/Plugins/` in your Unity project.

## 6. Install RealSense SDK on macOS (optional, for camera-in-editor)

```bash
brew install librealsense
```

Then clone <https://github.com/IntelRealSense/librealsense> and copy `wrappers/unity/Assets/RealSenseSDK2.0` into your Unity project's `Assets/`. You'll get an `RsDevice` component you can drop on a GameObject and a stream texture you can put on a quad.

If you don't have a D405 on the Mac yet, skip this — you can stub it with a webcam (`WebCamTexture`) for now and swap in RealSense when you move to the Rizon PC.

## 7. Wire up Redis writes

Drop [unity_scripts/RedisTeleopBridge.cs](../unity_scripts/RedisTeleopBridge.cs) into `Assets/Scripts/` in your Unity project. Attach it to an empty GameObject. Set the inspector fields:

- **Redis Host**: `127.0.0.1` for local Mac dev, the Rizon PC's IP later.
- **Redis Port**: `6379`.
- **Right Controller Anchor**: drag the `RightControllerAnchor` from your OVRCameraRig.
- **Robot Base Transform**: drag an empty GameObject placed where the Rizon4s base sits in your Quest world (calibration target).

## 8. Test end-to-end without a headset

1. In one terminal: `brew install redis && brew services start redis` (one-time).
2. In another terminal: `python vr_teleop/test_harness/fake_quest_publisher.py` — this streams fake controller poses to the same Redis keys the Unity script would. Useful for confirming the SAI controller consumes them correctly *before* Unity is even running.
3. To watch the writes in real time: `vr_teleop/test_harness/monitor_teleop.sh`.

When you're ready to swap the fake publisher for the real Unity app, just stop the Python script and hit Play in Unity with the Simulator active.

---

## Porting to the Rizon PC

When you move to the PC:

1. Copy the `SuturebotVR/` folder over (or push it to git — the `.gitignore` excludes Library/Temp/Obj).
2. Install Unity 2022.3 LTS on the PC.
3. Install the **Meta Quest Link app** on the PC (Windows only — that's why this happens on the PC, not the Mac).
4. Connect Quest 3 via USB-C, enable Link from the headset.
5. Change Redis host in the inspector to `localhost` (or wherever Redis lives relative to the Rizon controller).
6. Plug the D405 into the PC. Hit Play. You should now see the RealSense feed inside the headset and your controller pose driving the robot.
