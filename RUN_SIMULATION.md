# Running the Suturebot Simulation

This repo contains the scene (URDFs), OpenSai config (XMLs), and trajectory clients (Python). The simulator itself is **OpenSai**, a separate Stanford project. You need a working OpenSai checkout to run anything here.

## Prerequisites

1. **OpenSai** built and runnable: a binary at `<OpenSai>/bin/OpenSai_main` and `<OpenSai>/scripts/launch.sh`. See [manips-sai-org/OpenSai](https://github.com/manips-sai-org/OpenSai).
2. **redis-server**, **tmux** on your PATH (macOS: `brew install redis tmux`).
3. **Python deps**: `pip install redis numpy`.

## One-time setup

```bash
./scripts/setup_sim.sh
```

This symlinks Suturebot's XML configs, world URDFs, robot URDF, and the rizon4s mesh tree into OpenSai's `config_folder/` so `OpenSai_main` can find them. The script assumes OpenSai is at `../OpenSai` (sibling of this repo); override with:

```bash
OPENSAI=/path/to/your/OpenSai ./scripts/setup_sim.sh
```

It's idempotent — re-run any time you add a new world URDF or XML config.

## Run a simulation

**Terminal 1** — launch OpenSai (sim + controller + web UI):

```bash
cd ../OpenSai
sh scripts/launch.sh config_folder/xml_config_files/suturebot_grav.xml
```

A graphics window opens with the foam scene and the Rizon4s arm. Redis starts automatically if it isn't already.

**Terminal 2** — run a trajectory client:

```bash
python3 python_examples/suturebot_grav_pierce.py
```

The script connects to Redis, verifies OpenSai is running with the expected XML, activates the cartesian controller, and walks the flange through pierce waypoints.

## Available variants

| XML | World | Pierce direction | Python client |
|---|---|---|---|
| `suturebot.xml` | `world_suturebot.urdf` | n/a (no Grav) | `suturebot_motion.py`, `suturebot_stitch_sequence.py` |
| `suturebot_grav.xml` | `world_suturebot_grav.urdf` | +Y | `suturebot_grav_motion.py`, `suturebot_grav_stitch.py`, `suturebot_grav_pierce.py` |
| `suturebot_grav_negY.xml` | `world_suturebot_grav_negY.urdf` | −Y | `suturebot_grav_pierce_negY.py` |
| `suturebot_grav_oussama_push.xml` | `world_suturebot_grav_oussama_push.urdf` | +X (outward) | `suturebot_grav_pierce_oussama_push.py` |

Each Python client declares the XML it expects in `CONFIG_FILE_FOR_THIS_SCRIPT` and refuses to run against the wrong one.

## Sim vs real

The real-robot pipeline (Flexiv Elements Studio → `titania-4a_gripper_driver.sh` → `OpenSai_main` with `suturebot_grav_real.xml` → Python client) uses a different XML with **no `<simvizConfiguration>` block** — OpenSai launches only the controller, and robot state comes from the Flexiv→Redis driver. The Python clients in this repo are identical for both paths: same Redis keys, same trajectories. Only the XML changes.

## Troubleshooting

- **"Incorrect folder provided in the relative path of the config file"** — `OpenSai_main` was built with a different `CONFIG_FOLDER_PATH`. Rebuild OpenSai: `cd ../OpenSai && rm -rf build && sh scripts/build_OpenSai_main.sh`.
- **"This script expects OpenSai_main running with <xml>"** — XML / Python client mismatch. Match them per the table above.
- **`redis.exceptions.ConnectionError`** — `redis-server` didn't start. `launch.sh` should handle this; if not, run `redis-server &` manually.
- **`Couldn't load … link file`** — mesh file missing. Run `./scripts/setup_sim.sh` again to refresh the rizon4s symlink.

## Caveat about the foam

The foam slab, walls, and block are all `<static_object>` — rigid and immovable. The needle collides with them but won't actually pierce; the simulator resolves contact by pushing the arm back. The sim is good for trajectory shape, collision-free approach, gain tuning, and Redis-plumbing validation. It is **not** a faithful model of needle insertion through deformable tissue — that has to be validated on real hardware.
