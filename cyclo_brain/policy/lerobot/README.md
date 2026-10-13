# LeRobot Integration for Cyclo Intelligence

## Overview

LeRobot integration for Cyclo Intelligence. This folder contains the executor and Docker configuration for running LeRobot training and inference via Zenoh communication.

## Dream relative EEF Diffusion inference

Use the existing **Policy Path**, **Dataset FPS**, and Simulation/Robot controls.
The engine reads `config.json`: Diffusion checkpoints with `"state": "relative"`
use the EEF path; `"state": null` or an absent `state` keeps the joint-action path.
The representation is never inferred from the action dimension.

The EEF path reuses UMI's `get_real_umi_obs_dict` and `get_real_umi_action`:
the latest measured TCP anchors the complete observation/action window,
positions use meters, and rotations use UMI's row-based rot6d. Dream's copied
FK and gripper scaling match its training transform. After the saved policy
postprocessor, UMI restores absolute EE poses. The official Cyclo
`AIWorkerMoveLController` core coordinates both arm targets and their shared
lift joint through a small Python binding. The command groups are `arm_left`, `arm_right`,
and `lift`: eight values per arm (seven joints plus gripper), followed by one
lift position, for 17 values total. Arm angles use radians and lift uses meters.
The mobile base and head stay fixed; arm/lift URDF joint limits remain enabled.
Lift velocity damping is configurable with `CYCLO_IK_LIFT_WEIGHT` (default
`10.0`); arm damping stays `1e-4`. Increasing the lift weight penalizes lift
motion more strongly, shifting motion toward the arms when feasible. The
previous lift weight was `1e-6`, which strongly favored lift use. This weight
multiplies squared lift velocity in the QP; it is not a usage percentage or a
hard travel limit. Arm angles use radians and lift uses meters, so the raw
weight ratio does not directly measure their share of motion.

For example, set `CYCLO_IK_LIFT_WEIGHT=100` in `docker/.env` for stronger lift
suppression, or `1` for more lift use than the new default. Recreate the
LeRobot container so it receives the updated environment, then reload the
policy. Once the updated binding is built, tuning this value needs no rebuild.
The selected arm/lift weights are logged when an EEF policy is loaded.
In an offline bent-arm example commanding both TCPs 30 mm upward, lift moved
29.57 mm with the old `1e-6` weight, 5.93 mm with `10`, and 0.54 mm with `100`.
Both targets remained within 0.1 mm position error. These results depend on
posture and reachability; a weight is not a guarantee of a travel limit.

Each waypoint starts from the previous solution (the first uses measured
joints), applies up to 100 native QP updates with a 0.01 s internal step, and
clamps active joint positions to their URDF limits. The saved 20-dimensional EEF policy and processors
remain unchanged. This training representation supports **ffw_sg2_rev1**.

EEF targets do not uniquely identify the demonstration's lift trajectory;
the shared IK selects a compatible lift position. As in the earlier adapter,
the portable training URDF has visual/collision geometry removed, so this
inference bridge does not perform collision checking. The official Cyclo
submodule itself is unchanged; only its FK/MoveL/QP sources are compiled.
No ROS controller node or additional service is launched for IK.

Initialize the pinned submodule, build the image, then stop inference and
recreate `lerobot_server` through your existing deployment workflow before
reloading the model:

```bash
git submodule update --init --recursive cyclo_brain/sdk/cyclo_control
docker compose -f docker/docker-compose.yml build lerobot
```

For an existing container, the native module can also be built with
`cyclo_brain/sdk/cyclo_ik/build.sh /opt/cyclo_ik` using that container's Python
and SDK sources. Install `pybind11==2.13.6`, `cmeel-eigen==3.4.1`, and
`cmeel-urdfdom-headers==1.1.2` first. The build uses the policy environment's
Pinocchio, the submodule's vendored OsqpEigen, and OSQP 0.6.3. An unavailable
binding fails LOAD explicitly; it does not fall back to Placo.

For a transferred relative checkpoint, include the **original joint dataset's**
`meta/info.json` and `meta/stats.json` in `pretrained_model/dataset_meta/`:

```bash
POLICY_DIR=docker/workspace/model/lerobot/my_relative_policy/pretrained_model
DATASET_DIR=/path/to/original_joint_dataset
mkdir -p "$POLICY_DIR/dataset_meta"
cp "$DATASET_DIR/meta/info.json" "$DATASET_DIR/meta/stats.json" "$POLICY_DIR/dataset_meta/"
```

These two small files provide the named joints and the separate observation/action
gripper ranges. The relative normalizer files alone cannot recover those ranges.
If the original dataset is still accessible at the `train_config.json` dataset
root, the engine can read its metadata directly. Missing metadata fails LOAD
with a specific message. Include all the saved policy/processor files as usual.

The engine and copied helpers are bind-mounted. Both source Dockerfiles install
SciPy and build the native Cyclo binding against their Pinocchio installation.
The SDK build context is supplied by Compose's `additional_contexts` entry.

### Select Cyclo Control or NodeIK

In **Inference**, choose LeRobot / Diffusion and use **EEF IK** to select
**Cyclo Control** (default) or **NodeIK**. This applies to relative EEF policies
in both normal inference and dataset feedback simulation. Joint-action policies
keep their existing path. Unload dataset simulation before changing the solver;
normal inference reloads when the IK backend or model changes.

NodeIK is an unchanged Git submodule at `cyclo_brain/sdk/nodeik`, pinned to
`5273c0503bd2176f8a42c50e6e50f83f5103d53f`. The small `lerobot_engine/nodeik.py`
bridge calls its official `load_model`, `encode_poses`, and `predict` functions.
It reuses the Dream FK/math files already vendored for relative EEF policies,
so a separate Dream LeRobot installation is unnecessary.

```bash
git submodule update --init cyclo_brain/sdk/nodeik
docker compose -f docker/docker-compose.yml build lerobot
```

For an existing container with the submodule mounted at `/opt/cyclo_sdk/nodeik`,
install only `torchdiffeq==0.2.5` in the policy Python environment and restart
the LeRobot runtime. The source Dockerfiles install NodeIK and this dependency;
Compose also mounts the submodule for local development. Rebuild `interfaces`
and restart the orchestrator, ROS bridge, and policy runtime after updating
the service definitions (`ik_backend`, `ik_model_path`), then rebuild the UI.

**NodeIK Model** accepts a folder or `.pt` file and has the same folder browser
as Policy Path. A compare-run folder resolves to `pose/best.pt`; selecting
`pose/`, `vision/`, or an explicit checkpoint loads that model. The tested
local weights were copied to the container-visible folder:

```text
/workspace/model/nodeik/nodeik_20261010_e2e30k
```

The pose checkpoint is step 76000; the run name predates its resumed training.
The vision checkpoint requires all four recorded head/wrist RGB cameras.
Missing cameras fail explicitly. UMI first restores absolute left/right TCP
poses in the fixed URDF base frame; NodeIK returns left 7 / right 7 / lift in
radians/meters. One fixed latent (`0`) is retained across the trajectory. Joint
outputs are clipped to the training URDF limits, then combined with the existing
policy gripper outputs. FK residuals are measured after clipping. NodeIK is an
approximate learned solver and does not receive Cyclo QP refinement.

When **NodeIK** is selected, the normal control loop uses quintic joint-space
interpolation instead of linear interpolation, at the existing **Control Hz**.
It interpolates both arms, lift, and policy grippers after IK, and uses a
quintic weight for the existing chunk blend. The uploaded `quintic_spline` and
`quintic_spline_multi` functions (Suhan Park) are copied unchanged into
`cyclo_brain/sdk/action_chunk_processing/action_chunk_processing/spline.py`;
the NodeIK submodule is unchanged. Normal interpolation requires the existing
`POSTPROCESS_ACTIONS=true` setting (default).

Each interval has zero endpoint velocity and acceleration, giving a bounded,
rest-to-rest quintic curve without overshooting its joint endpoints. This does
not impose joint velocity/acceleration limits or a straight TCP path. Dataset
feedback simulation also displays these intermediate states from the current
state to each IK target. Only target endpoints enter the next observation
history, and images advance at dataset frame boundaries. Playback samples each
interval at Control Hz and also includes its exact endpoint. At 30 dataset FPS and
100 Control Hz, eight IK targets produce 32 displayed states over 8/30 seconds,
while the dataset frame still advances by eight. Cyclo Control keeps its
existing interpolation and dataset playback behavior.

### Dataset feedback simulation in Inference

Keep the existing Policy Path, select **Simulation** and the LeRobot backend,
then enable **Dataset feedback simulation** in the Inference sidebar. Enter a
LeRobot dataset path, episode, and initial frame, and click **Load / Reset**.
Both paths must be accessible inside the LeRobot container (normally under
`/workspace`). Checkpoint metadata retains the original gripper ranges. If that
metadata is missing, select the original joint training dataset so the engine
can read it there; another test dataset requires bundled training metadata.

**Step chunk** runs one prediction and displays every resulting joint state.
**Run** repeats prediction and playback; **Pause after chunk** lets the current
chunk finish. **Unload** releases the model and lets you change/reset the inputs.
Turning the checkbox off also finishes any in-flight chunk, unloads the dataset
session, and restores the normal inference controls in one action.
Unload normal inference before entering this mode. The normal lifecycle controls
are hidden during dataset simulation, and the engine rejects live commands until
the dataset session is unloaded.

After updating an existing deployment, rebuild the ROS `interfaces` package
(including `EngineCommand.srv`) and the UI, then restart the ROS bridge and
LeRobot runtime so both ends use the updated service definition.

The first observation window comes from the selected episode, ending at the
initial frame; frames before episode start are padded with its first sample.
Afterward, the existing UMI decoder and the selected EEF IK backend produce joint
commands. Each command replaces the simulated arms, grippers, and lift state;
the last `n_obs_steps` simulated states feed the next prediction. Head and base
retain their initial values. Camera images advance one recorded dataset frame
per applied command. The last frame ends the rollout. This tests state feedback
with recorded images, rather than rendering cameras or simulating task physics.

The existing 3D viewer shows the simulated robot and EEF targets with orientation
axes: **Observation** in blue, **Action** in orange. **Robot** shows restored
absolute poses; **Relative TCP** shows both relative to their latest observed
TCP anchors, with left/right origins separated by a display offset. This is the
same anchor used to decode each action chunk. Simulation creates no live sensor
subscriptions or robot publishers and sends no hardware commands.

Dataset LOAD/STEP/UNLOAD reuse the EngineCommand service with dedicated commands
and JSON simulation fields. After updating the source, rebuild the orchestrator's
ROS `interfaces` package and UI, and recreate the LeRobot backend so both sides
use the updated service definition. The existing policy and IK dependencies are
sufficient; the LeRobot submodule, UMI helpers, and IK core are unchanged.

Local smoke validation used the real Task 0 dataset (episodes 0–1), 100 training
steps, four 64×64 cameras, and a small Diffusion model. It checks UMI observations
against Dream's training transform, identity-pose IK, LOAD/GET_ACTION/cached
LOAD/UNLOAD, and RobotClient's arm-command slicing using replayed sensors and
captured commands. Re-run inference with a Python environment containing
LeRobot's dataset/diffusion/IK dependencies and SciPy:

```bash
python cyclo_brain/policy/lerobot/scripts/smoke_relative_eef.py \
  docker/workspace/model/lerobot/relative_eef_smoke_20261010/checkpoints/000100 \
  --dataset-root=/path/to/Task_0_tape_MCAP_lerobot_v30
```

This validation sends no commands to hardware and measures process compatibility,
not task success after 100 training steps.
The earlier arm-only validation of the 100-step checkpoint returned three
`(8, 16)` chunks and 48 captured arm commands. The existing non-relative
checkpoint 964000 returned `(8, 22)`;
switching joint → relative EEF → joint also passed. The targeted engine/runtime
and Dockerfile regression tests passed (78 tests, plus two subtests).
Local logs and results are under the smoke model's `validation/` directory.
The current replay smoke script checks `(n_action_steps, 17)` chunks and captures
both arm and lift commands without hardware publishers. Shared-lift IK tests
also cover stationary poses, common vertical motion, differing arm goals, and
lift joint limits:

```bash
docker exec -e PYTHONPATH=/app:/policy_runtime:/zenoh_sdk:/robot_client_sdk \
  lerobot_server python3 -m unittest discover \
  -s /app/lerobot_engine/tests -p test_relative_eef.py -v
```

## Diffusion Task 0 inference (checkpoint 964000)

This setup uses the existing Web GUI and the two-process LeRobot runtime.
Diffusion observations are captured at the GUI's **Dataset FPS** and stacked
as `(batch, n_obs_steps, ...)` before the saved processor pipelines run.
Other policies keep their existing observation format.

### Set up another PC

From a PC with Docker and the NVIDIA Container Toolkit:

```bash
git clone --recurse-submodules https://github.com/lab-dream/dream_cyclo_intelligence.git
cd dream_cyclo_intelligence
mkdir -p docker/workspace/model/lerobot/diffusion_task0_964000
# Replace /path/to/964000 with the manually transferred checkpoint directory.
cp -a /path/to/964000/pretrained_model docker/workspace/model/lerobot/diffusion_task0_964000/
bash docker/container.sh start
bash docker/container.sh build-ui
```

Start the ROS services from the Physical AI Manager as usual. For a standalone
PC without that manager, start them through the existing supervisor API:

```bash
curl -fsS -X POST http://127.0.0.1:7100/services/orchestrator/start
curl -fsS -X POST http://127.0.0.1:7100/services/cyclo_data/start
```

The configured Zenoh router must also be running. If there is no existing
router, run the following in another terminal and leave it running:

```bash
docker exec -it cyclo_intelligence bash -lc \
  'source /opt/ros/jazzy/setup.bash; ros2 run rmw_zenoh_cpp rmw_zenohd'
```

Transfer the entire `pretrained_model` directory, including `config.json`,
`model.safetensors`, both `policy_*processor.json` files, and both processor
statistics `.safetensors` files. Training state and the dataset are unnecessary
for inference. `docker/workspace` and model weights are already ignored by Git.
Keep a real copy inside this workspace; a symlink to a host-only path will not
resolve inside the policy container.

Open `http://<robot-pc>:7080`, select robot **ffw_sg2_rev1**, and open Inference:

1. Select **Diffusion** under Model.
2. Click **Use Task 0 · 964000 (30 FPS)**. It fills Policy Path,
   Dataset FPS `30`, Control Hz `100`, and async action requests.
3. Use the LeRobot Docker **Pull Image** / **Start** controls until both runtime
   services are ready. Both policy processes must use this checkout's runtime;
   restart `lerobot_server` when updating its internal protocol.
4. Select **Simulation** or **Robot** using the existing GUI mode controls,
   then start inference. Simulation uses robot sensor inputs and previews
   predictions; Robot publishes actions to the robot.

The container policy path is
`/workspace/model/lerobot/diffusion_task0_964000/pretrained_model`.
The folder browser can select a different manually copied checkpoint.
This policy consumes four RGB cameras at `96 × 96`, two observations, and
22 state values (arms, grippers, head, lift, mobile odometry). It returns eight
22-dimensional actions per request using the checkpoint's eight DDIM steps.
Dataset FPS denotes action waypoint timing and observation spacing, not the
number of neural-network evaluations per second.

When building the policy image from source instead of pulling it, use
`bash docker/container.sh start-lerobot --build`; the helper detects the PC
architecture. Both policy Dockerfiles explicitly install the diffusion extra.
The engine changes are bind-mounted, so a policy image rebuild is unnecessary
when the installed image already includes Diffusion dependencies.

## Folder Structure (legacy executor documentation)

```
cyclo_brain/policy/lerobot/
├── lerobot/                 # LeRobot repository (git submodule)
│   └── (HuggingFace LeRobot source code)
├── executor.py              # Zenoh communication + train/infer execution
├── Dockerfile               # Container build file
├── entrypoint.sh            # Container entrypoint
├── workspace/               # Dataset/Model/Results (gitignore)
├── test_executor.py         # Unit tests
├── README.md                # This document
└── INTEGRATION_REPORT.md    # Integration detail report
```

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU (CUDA 12.1+)
- Cyclo Intelligence repository

## Quick Start

### 1. Clone LeRobot (if not already done)

```bash
cd cyclo_intelligence/cyclo_brain/policy/lerobot
git clone https://github.com/huggingface/lerobot.git lerobot
```

### 2. Build Docker Image

```bash
cd cyclo_intelligence
docker compose -f docker/docker-compose.yml build lerobot
```

### 3. Download Test Dataset

```bash
# Create workspace directory
mkdir -p cyclo_brain/policy/lerobot/workspace

# Download lerobot/pusht dataset
huggingface-cli download lerobot/pusht \
  --local-dir cyclo_brain/policy/lerobot/workspace/lerobot/pusht \
  --repo-type dataset
```

### 4. Run Container

```bash
docker compose -f docker/docker-compose.yml up lerobot
```

## Architecture

```
Cyclo Intelligence Web UI (React UI)
        │
        ▼ WebSocket (7090)
Cyclo Intelligence Orchestrator (ROS2 + rmw_zenoh_cpp)
        │
        ▼ Zenoh Protocol (7447)
LeRobot Executor (Docker Container)
        │
        ▼
LeRobot Training/Inference APIs
```

## Supported Policies

| Policy | Category | Description |
|--------|----------|-------------|
| act | Imitation Learning | Action Chunking Transformer |
| diffusion | Imitation Learning | Diffusion Policy |
| vqbet | Imitation Learning | VQ-BeT |
| tdmpc | RL | TD-MPC |
| pi0 | VLA | Physical Intelligence VLA |
| pi0_fast | VLA | Optimized Pi0 |
| smolvla | VLA | SmolVLA |
| sac | RL | Soft Actor-Critic |

## ROS2 Services (via Zenoh)

| Service | Description |
|---------|-------------|
| /lerobot/train | Start training |
| /lerobot/infer | Start inference |
| /lerobot/stop | Stop current task |
| /lerobot/status | Get status |
| /lerobot/policy_list | List available policies |
| /lerobot/checkpoint_list | List checkpoints |
| /lerobot/model_list | List cached models |

## ROS2 Topics (via Zenoh)

| Topic | Direction | Description |
|-------|-----------|-------------|
| /lerobot/progress | Published | Training metrics (step, loss, epoch) |
| /lerobot/action | Published | Inference action outputs |

## Docker Configuration

### Volume Mappings

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| cyclo_brain/policy/lerobot/workspace | /workspace | Dataset/Model storage |
| cyclo_brain/sdk/zenoh_ros2_sdk | /zenoh_sdk | Zenoh SDK |

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| RMW_IMPLEMENTATION | rmw_zenoh_cpp | ROS2 Zenoh middleware |
| ROS_DOMAIN_ID | 30 | ROS2 domain |
| ZENOH_CONFIG_OVERRIDE | local shared-memory default or remote router example in `/root/.bashrc` | Zenoh client configuration |

At runtime, `lerobot_server` sources `/root/.bashrc` before starting
`main-runtime` and `engine-process`. Enter the container, edit the Cyclo
ROS/Zenoh block near the top of `/root/.bashrc`, and choose either the local
`ZENOH_CONFIG_OVERRIDE` line or the commented remote router example. Then restart
`lerobot_server`. `docker restart` preserves the edit;
recreating or updating the container resets `/root/.bashrc` to the image
default.

## Testing

### Unit Tests

```bash
cd cyclo_intelligence/cyclo_brain/policy/lerobot
python -m pytest test_executor.py -v
```

### Integration Test

```bash
# Start container
docker compose -f docker/docker-compose.yml up -d lerobot

# Enter container
docker exec -it lerobot_server bash

# Test import
python -c "from executor import LeRobotExecutor; print('OK')"
```

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Zenoh connection failed

```bash
# Ensure the externally managed Zenoh router is running on the endpoint in ZENOH_CONFIG_OVERRIDE.
ss -ltnp 'sport = :7447'
```

### LeRobot import error

```bash
# Check LeRobot is properly cloned
ls -la cyclo_brain/policy/lerobot/lerobot/

# Ensure PYTHONPATH includes lerobot
export PYTHONPATH="/app/lerobot:$PYTHONPATH"
```

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Documentation](https://huggingface.co/lerobot)
- [Zenoh ROS2 SDK](https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk)
- [Cyclo Intelligence Workflow](../../ai_system_agents/opensource_integration_workflow/)
