# Suturebot

Single-arm autonomous surgical suturing with a Franka FR3 and a static 3D-printed end-effector. CS 225A final project, Stanford Spring 2026.

## Motivation

Surgical suturing is one of the most common procedures in medicine and one of the most tedious. The standard robotic platform for it is the da Vinci Surgical System, which uses three or four teleoperated arms and proprietary instruments. A new da Vinci unit runs around $1.5M before maintenance and per-procedure consumables, which puts robotic suturing out of reach of most hospitals outside large urban academic centers.

Most academic work on autonomous suturing inherits this hardware assumption. Berkeley's AUTOLAB uses the dVRK (the research version of da Vinci). Johns Hopkins' STAR uses a custom multi-arm platform with purpose-built suturing tools. Both lines of work are excellent and both are expensive to reproduce, which means the cost of the platform sits between the research and any kind of broad clinical deployment.

Suturebot asks a different question: how far can you get with one standard 7-DOF industrial arm and a few dollars of 3D-printed plastic for the end-effector? If you can do real suturing with hardware that costs roughly 1% of a da Vinci, the technology stops being something only a top-tier hospital can afford.

The novelty here is not the algorithms. It is the constraint. Single arm, no actuated gripper, off-the-shelf research robot, printed end-effector. Everything that makes the platform expensive is removed and the system has to work anyway.

## Related work

**Berkeley AUTOLAB (Goldberg et al.).** Berkeley's group showed a dVRK system that learns to plan the six motions of a stitch from dual-camera observations, and chained six consecutive stitches at its best on 2D phantom skin. Reported challenges include needle reflectivity confusing perception and modeling deformable tissue and thread. The October 2024 "Augmented Dexterity" paper in Science Robotics with Intuitive's CEO sketched a path toward AI-driven plans that human surgeons supervise, and STITCH 2.0 extended this with EKF-based needle state estimation. All of this work runs on da Vinci hardware.

**Johns Hopkins STAR (Krieger et al.).** The Smart Tissue Autonomous Robot demonstrated autonomous laparoscopic intestinal anastomosis on live pigs in 2022 and a more realistic supervised procedure in 2025. Recent work introduced a single-arm Suture Management Device that lets STAR tension and manage thread without a second arm or human assistant, matching dual-arm performance.

**How Suturebot differs.** STAR is single-arm in execution but still relies on a custom platform with infrared fiducials in tissue and purpose-built suturing tools. We are working with a generic 7-DOF research arm, no fiducials, and a passive printed end-effector instead of an actuated needle driver. Compared to the Berkeley line of work we trade dexterity for cost and reproducibility: no two arms, no surgical wrist, no proprietary instruments. The closest published prior work is the 2023 IEEE single-arm autonomous suturing system, which still used surgical-grade tooling. To our knowledge nothing has tried this with a static printed needle holder.

## System overview

The platform is one Franka FR3 with a custom static end-effector mounted at the flange. The needle is pre-loaded into the holder before the run. The arm carries the needle through the stitch trajectory and uses its 7 DOF to handle the orientation changes that a fixed gripper cannot.

![Franka Research 3](https://franka.de/hubfs/_Frank%20Robotics4386_resized-1.jpg)

*Franka Research 3. Image: Franka Robotics ([franka.de](https://franka.de/franka-research-3)).*

FR3 specs that matter for this project:

| Property | Value |
|---|---|
| DOF | 7 |
| Reach | 855 mm |
| Payload | 3 kg |
| Joint torque sensing | yes, all 7 joints |
| Pose repeatability | ±0.1 mm |
| Max TCP speed | ~1 m/s |

The torque sensors at every joint are the part we care about most. Suturing involves contact with deformable tissue, and we need to detect insertion forces and back off when the needle catches or hits something it shouldn't. The 855 mm reach is comfortable for a tabletop suturing setup. The 3 kg payload is more than enough for the printed end-effector and a needle.

## Phased approach

We are doing this in two stages so we can debug perception and control in a regime where everything is large and slow, before scaling down to anything resembling a real surgical workflow.

### V1: oversized everything

* **Workpiece:** custom-cast foam epidermis. We pour silicone or two-part foam into a printed mold so we can rebuild it cheaply between attempts.
* **Needle:** an oversized stitching needle, much larger than a real surgical needle. Bigger needle means easier perception, more forgiving force tolerances, and a wider error budget on the trajectory.
* **Goal:** complete one full suture (entry, arc, exit, pull-through) end to end on the foam, then chain three in a row.

V1 is about getting the loop working. We expect most of the time to go into perception (finding the needle tip and the entry point in the camera frame) and into the trajectory primitive that drives the needle through the tissue without snagging.

### V2: real suture practice kit

Once V1 chains stitches reliably we graduate to a standard suture practice pad ([Amazon link](https://www.amazon.com/Practice-Training-Include-Non-Absorbent-Surgical/dp/B07HQD6WRX)) and a real curved surgical needle. This is where the project starts to look like the published baselines. Smaller needle, more compliant tissue analog, tighter geometric tolerances.

We do not expect the V1 controller to transfer cleanly. The plan is to keep the same primitive structure but retune force thresholds and trajectory parameters, and probably upgrade the perception stack.

## End-effector

The end-effector is fully passive. No motor, no actuated gripper.

* **Why static.** An actuated needle driver adds weight, wiring, and a control channel we do not need. Real surgical needle drivers exist mostly to release and re-grasp the needle from the other side of the wound, which is what the second arm does on a da Vinci. We are betting we can do that with an FR3 trajectory instead. If it works, we save the cost of the gripper and the integration effort, and the rest of the system stays simpler.
* **Materials.**
  * SLA resin for the parts that hold the needle. Tight tolerances and a smooth surface finish so the needle seats predictably and does not wobble.
  * FDM PLA or PETG for the body and the FR3 flange adapter. Cheaper, faster to iterate, plenty stiff for the forces involved.
* **Pre-loading.** The needle is seated by hand before the run. A future revision could add passive snap retention (friction plus a light feature) but V1 just uses a press fit.

Total bill of materials for the end-effector should come in under $10.

## Software architecture

We are building on OpenSai, the Stanford robotics simulation and control framework. The four modules we use:

* **SaiModel.** Kinematics and dynamics of the FR3. Forward and inverse kinematics, Jacobians, and the mass matrix used by the operational-space controller.
* **SaiPrimitives.** The controller library. Suturing decomposes naturally into operational-space motion primitives (move to pre-insertion pose, drive along the needle arc, pull through, retract) and SaiPrimitives is built around that decomposition.
* **SaiSimulation.** Physics simulator. Everything gets developed and shaken out in sim before it touches the real arm. The foam epidermis is modeled as a deformable contact and the needle is a rigid body fixed in the end-effector frame.
* **SaiGraphics.** Visualizer for the simulation. Used for debugging trajectories and for the demo video.

**Sim-to-real.** The controller talks to both the simulator and the real FR3 over the same Redis interface OpenSai uses for hardware. Once the primitive sequence works in sim we swap the simulation backend for the FR3 driver and retune.

**Perception.** A wrist-mounted RGB camera plus a static overhead camera, with classical CV for needle tip localization in V1. Anything more elaborate (learned needle pose estimation in the spirit of STITCH 2.0) is a V2 stretch goal.

**Repo layout**:

```
config_folder/
  xml_config_files/   OpenSai launch configs (suturebot*.xml)
  world_files/        scene URDFs (world_suturebot*.urdf)
  robot_files/        Rizon4s + Grav URDF, mesh tree (rizon4s/), surgery clamp STL
python_examples/      trajectory clients (pierce, stitch, motion sequences)
scripts/              setup helpers (setup_sim.sh)
reference/            archived material — not used at runtime
  project_starter/    CS 225A starter code (Panda/OptiTrack examples)
  flexiv_archive/     unused Flexiv URDFs / meshes
RUN_SIMULATION.md     how to run the sim end-to-end
```

## Getting started

See [RUN_SIMULATION.md](RUN_SIMULATION.md) for sim setup and launch. The short version:

```bash
./scripts/setup_sim.sh    # one-time, assumes OpenSai at ../OpenSai
cd ../OpenSai
sh scripts/launch.sh config_folder/xml_config_files/suturebot_grav.xml
# in another terminal:
python3 python_examples/suturebot_grav_pierce.py
```

OpenSai itself lives at [manips-sai-org/OpenSai](https://github.com/manips-sai-org/OpenSai).

## Team

CS 225A, Stanford, Spring 2026.

* Adarsh Ambati
* Kimberly Nickerson
* Simon Casper
* Matthew Kim
Course staff: Prof. Oussama Khatib, Mentor: Enzo Andreacchio

## License

MIT. See [LICENSE](LICENSE).
