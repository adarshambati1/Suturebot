# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Suturebot is autonomous surgical suturing with a single 7-DOF arm and a **passive 3D-printed end-effector** (no actuated gripper on the robot wrist — needle re-grasping is done by an external clamp toggled over Redis). CS 225A final project, Stanford Spring 2026. See `README.md` for full motivation.

This repo contains only the **scene (URDFs), OpenSai configs (XMLs), and trajectory clients (Python)**. The simulator and controller (`OpenSai_main`) are a **separate project** — [manips-sai-org/OpenSai](https://github.com/manips-sai-org/OpenSai) — that you must have built and checked out at `../OpenSai` (sibling dir). Nothing here runs without it.

## Running

```bash
./scripts/setup_sim.sh                  # one-time: symlinks this repo's XMLs/URDFs/meshes into ../OpenSai/config_folder/
                                        #   override location with OPENSAI=/path/to/OpenSai ./scripts/setup_sim.sh
                                        #   idempotent — re-run after adding any new XML/world URDF/mesh

cd ../OpenSai                           # Terminal 1: launch sim + controller + web UI (auto-starts redis)
sh scripts/launch.sh config_folder/xml_config_files/suturebot_grav_oussama_push.xml

python3 python_examples/suturebot_grav_pierce_oussama_push.py   # Terminal 2: run a trajectory client
python3 python_examples/<client>.py --pf                        # add --pf to log forces and save .npz + .png
```

`pip install redis numpy` (plus `matplotlib` for `--pf` plots). `RUN_SIMULATION.md` has the full XML↔client compatibility table and troubleshooting. There is **no build/lint/test suite** — this is a research repo; "running" means launching the sim and a client.

## The XML ↔ client contract (critical)

Each Python client hardcodes the XML it requires in `CONFIG_FILE_FOR_THIS_SCRIPT` and **refuses to run against any other** (it reads the `::sai-interfaces-webui::config_file_name` Redis key and exits if it mismatches). You must launch the matching XML for the client you want. The variants differ by how the scene is rotated about world Z and which direction the needle pierces:

| XML | Pierce dir | Matching clients |
|---|---|---|
| `suturebot.xml` | n/a (no Grav end-effector) | `suturebot_motion.py`, `suturebot_stitch_sequence.py` |
| `suturebot_grav.xml` | +Y | `suturebot_grav_motion.py`, `suturebot_grav_stitch.py`, `suturebot_grav_pierce.py` |
| `suturebot_grav_negY.xml` | −Y | `suturebot_grav_pierce_negY.py` |
| `suturebot_grav_oussama_push.xml` | +X (outward) | `suturebot_grav_pierce_oussama_push*.py`, `suturebot_grav_stitch_oussama_push*.py` |

## Sim vs. real robot — single toggle

The same clients drive both sim and the real Flexiv arm. The switch is the **`ROBOT_NAME` constant** at the top of each client:
- `"Rizon4s"` → simulation. `CONFIG_FILE_FOR_THIS_SCRIPT` resolves to a `suturebot_grav_*.xml`.
- `"Titania"` → real Flexiv driver. Resolves to `suturebot_grav_real.xml`, which has **no `<simvizConfiguration>` block** — OpenSai runs only the controller and robot state comes from the Flexiv→Redis driver.

Redis keys and trajectories are identical across both paths; only `ROBOT_NAME` (and thus the XML) changes. All Redis keys are namespaced by `ROBOT_NAME` (e.g. `opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position`).

## The grasp patch (a tracked OpenSai fork)

To model a needle the arm **grips, carries, drops, and re-grasps**, the sim needs a free dynamic object that can be pinned to the hand and released. Vanilla OpenSai can't do this: object pose is publish-only over Redis in sim mode, the actuated gripper relies on URDF `<mimic>` (which SAI doesn't enforce, so the fingers are welded), and there's no friction grasp. So `scripts/opensai_patch.diff` patches `<OpenSai>/core/sai-interfaces/src/simviz/SimVizRedisInterface.{h,cpp}` to add two Redis receive keys per dynamic object:

- `opensai::commands::<obj>::pose` — JSON 4×4 row-major transform.
- `opensai::commands::<obj>::kinematic` — `"1"` pins the object to that pose each sim step (gripped); `"0"` is normal dynamics (released → falls).

`setup_sim.sh` applies the patch idempotently to the sibling OpenSai checkout. **A fresh apply requires rebuilding OpenSai** (`core/sai-interfaces/build` then the top-level `build`) — the core libs are standalone clones built in-tree, not submodules, and there's no `make install`. `python_examples/pin_test.py` + `pin_test.xml` are the self-test (pin → drive → release; expect all PASS). The patch is the foundation for the planned free-needle running-stitch FSM.

Interpreter gotcha: run clients with a `python3` that has `redis`+`numpy` (Homebrew python, not base conda).

## How a client works (architecture)

Clients are self-contained single files following a consistent pattern (the `*_oussama_push_grip.py` files are the most complete — full finite state machine with re-grip):

1. **Redis is the only interface.** A client never imports OpenSai — it `set`s goal-pose keys and `get`s current-pose / force-sensor keys. `RedisKeys` dataclass holds all key strings. On startup it verifies the right XML is loaded and forces `active_controller_name` = `cartesian_controller`.

2. **Everything is computed in the flange frame, commanded to the flange.** The controlled `linkName` is `flange`. The two points we actually care about — the **jaws** (where the clamp grips) and the **needle tip** — are constant offsets in the flange frame (`JAWS_OFFSET`, `NEEDLE_TIP_OFFSET`). Helper fns (`flange_x_for_jaws`, `compute_needle_tip_world`, etc.) convert between flange and world. When you rotate the scene about Z for a variant, you rotate these offsets too (the file headers document the `R_z` math, e.g. `R_z(-90°) @ (x,y,z) = (y,-x,z)`).

3. **Orientation is held fixed.** A constant `ORI` 3×3 matrix (camera-down) is commanded at every step; suturing motion is purely translational in these clients.

4. **The FSM is an explicit `State` enum walked by `step()`.** `step(redis, state, pos, dwell, msg)` writes the goal, sleeps `dwell`, reads back the actual pose, and prints commanded/actual/needle-tip/jaws positions. A stitch = pierce → re-grip (lift, release clamp, reposition, re-grip) → push through → move away; `transit_to_*` moves between stitch sites. Geometry constants (`FOAM_CENTER_X`, `HOME_GAP`, `LIFT_RISE`, the `*_STITCHES` arrays) are tuned per scene and **hardcoded** — recent commits note positions are hand-calibrated, not perceived.

5. **The clamp/gripper is toggled by mode string,** not width: `set_gripper_mode(redis, "g")` to grip, `"o"` to open, on key `opensai::commands::{ROBOT_NAME}::gripper::mode`, followed by a `time.sleep(2)` to let it actuate.

6. **`--pf` force logging.** `ForceLogger` (a daemon `threading.Thread`) polls force/moment Redis keys at 100 Hz, records state-transition markers, and on exit saves `.npz` + a per-key `.png` to `log_files/force_logs/`. `log_files/` is gitignored.

When editing a client, keep the offset/`R_z` conventions in the file header consistent with the constants — getting the frame math wrong silently drives the needle to the wrong place. The sim treats foam as a **rigid `<static_object>`** (it does not deform/pierce), so sim validates trajectory shape and Redis plumbing only — insertion force behavior must be validated on hardware.

## VR teleop (`vr_teleop/`)

Separate scaffolding for Meta Quest 3 teleoperation. The Unity bridge (`RedisTeleopBridge.cs`) writes the **same goal-pose Redis keys** the Python clients use, so the controller is agnostic to whether a script or the headset is producing goals. `SuturebotVR/` (the Unity project body — `Library/`, `Temp/` etc.) is gitignored. `vr_teleop/test_harness/fake_quest_publisher.py` streams fake poses for testing the pipeline without a headset or robot. See `vr_teleop/README.md`.

## Layout notes

- `config_folder/` — `xml_config_files/` (OpenSai launch configs), `world_files/` (scene URDFs), `robot_files/` (Rizon4s+Grav URDF + `rizon4s/` mesh tree + clamp STL).
- `reference/` — archived material, **not used at runtime** (CS 225A starter code, unused Flexiv URDFs).
- `python_examples/` — all trajectory clients. `manual_recorder.py` and `jointorientation.py` are utility/experiment scripts.
