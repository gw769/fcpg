# FCPG on the SO-101 arm

FCPG (feedback-conditioned proprioceptive grounding) reads a robot's past camera pictures through the joint states at
the time of each picture and now, so that the policy keeps using what it saw after a camera loses the view. This
repository holds the system that runs on our SO-101 arm (November 2026), one version only:

- the **FCPG reader** and the Transformer **Actor**, which proposes the arm's next moves;
- the **LeWM** world model (built on LeWorldModel) on DINOv3-L patches of the wrist and the external camera, with a
  memory of the last 30 pictures per camera;
- the **CEM planner**, which scores the Actor's proposals with the LeWM, and the **robot service** around them.

## Layout

| Folder | Contents |
|---|---|
| `fcpg/` | the library. The system is the `v21_*` modules: `v21_core.py` (reader + Actor + LeWM), `v21_memory.py` (FCPG reader), `v21_full_patch_lewm.py` (LeWM), `v21_picture_memory.py` (the picture memory), `v21_alignment_cem.py` and `v21_grasp_cem.py` (CEM), `v21_runtime.py` and `v21_live_service.py` (robot runtime and service). The `prod_v13` ... `prod_v20` modules stay because the v21 code builds on them |
| `scripts/` | the entry points, see below; the other scripts here are modules the entry points import |
| `tests/` | pytest: `python -m pytest tests/` (tests that need our data or a GPU skip or fail without them) |
| `adapters/`, `interfaces/`, `perception/` | the robot hardware (Feetech bus, cameras, safety) and its interfaces |
| `configs/`, `config/` | camera layouts, camera types, task and robot settings |
| `v10/` | the task contract the data loader still reads |
| `third_party/lewm` | LeWorldModel snapshot (MIT, see its `LICENSE` and `UPSTREAM.md`) |
| `third_party/so_arm100` | SO-ARM100 simulation assets (Apache-2.0, see its `LICENSE`) |

| Entry point | What it does |
|---|---|
| `scripts/train_v21_horizon40.py` | trains the LeWM (`--lewm-only --full-patch-lewm`): an Archive stage on the public recordings, then the fish stage on the two demonstrations and the robot blocks |
| `scripts/train_v21_fcpg_actor.py` | trains the FCPG reader and the Actor: an Archive stage, then the fish stage |
| `scripts/package_v21_fcpg_actor_candidate.py` | packages one fish-stage checkpoint of the Actor |
| `scripts/package_v21_robot_lewm.py` | puts a LeWM into an Actor package and checks it the way the robot service loads it |
| `scripts/serve_so101_v21_research.py` | the robot service (HTTP); `scripts/canary_so101_v21_research.py` checks a service without moving the arm |
| `scripts/exam_v21_lewm_robot.py` | the LeWM's robot exam: forecasts on robot blocks never trained on, with one or both cameras unseen for 0.5 to 16 s |
| `scripts/eval_v21_lewm_demo3.py` | the LeWM's forecast errors on the held-out demonstration (demo 3) and the Archive development rows |
| `scripts/select_v21_camera_layout_sources.py` | the camera table: which recordings train with both cameras and which with the external camera only |
| `scripts/build_v21_dinov3_features.py` | the DINOv3-L features of the recordings (int8 tokens) |

## The release models

| Model | Recipe |
|---|---|
| LeWM | the last 30 pictures per camera (8 tokens each), balanced inputs, the arm's control setting as input, two passes over 1,481,603 Archive rows, then 1,000 fish-stage updates on demonstrations 0 and 1 and the robot blocks; loss = prediction + SIGReg (0.02) |
| Actor | three steps on the same camera table: from scratch (6 passes over the Archive rows), 8,000 more Archive updates, then the fish stage with the box paste; fish-stage step 375 is packaged |

Camera table: a recording keeps both cameras when its wrist camera turns like ours, and only its external camera
otherwise.

## Not included yet

- the data: the DINOv3-L feature store of the recordings, the camera and cue tables, the robot block packs, the parent
  checkpoint and the response pack the trainers read;
- the weights;
- configurable paths: some modules still default to paths on our machines (`/home/...`);
- a license (none yet: all rights reserved until one is added).
