# Cyclo Intelligence

## Build or update this fork

On an NVIDIA Linux PC with Docker Engine, Compose v2, and NVIDIA Container
Toolkit installed, initialize the containers and apply this checkout with:

```bash
git clone https://github.com/lab-dream/dream_cyclo_intelligence.git
cd dream_cyclo_intelligence
./build.sh
```

For subsequent updates:

```bash
git pull --ff-only
./build.sh
```

`build.sh` initializes the pinned runtime submodules, provisions missing Cyclo
and LeRobot images/containers, builds the ROS interfaces and Web GUI, checks
the policy environment, restarts both containers, and starts the ROS services
needed by the GUI. Existing compatible
containers keep their Python environments and ROS settings. A local Zenoh
router is started if port 7447 is unused; set `CYCLO_START_LOCAL_ROUTER=0` when
using an external router. The script does not install host drivers or Docker.

Open **http://localhost:7080** and refresh the page after an update.
Models and datasets are user data under
`docker/workspace` and are preserved. They are not downloaded by the build.
Copy policies into `docker/workspace/model/lerobot` and the ChunkRL actor into
`docker/workspace/model/chunkrl`, then select their paths in the GUI.

Use `./build.sh --rebuild-images` when rebuilding image dependencies, and
`./build.sh --with-groot` to also provision the larger GR00T backend. A container
bound to a different checkout is reported explicitly so the script cannot
silently serve another repository's code or remove its data.

For relative Diffusion policies, select the trained hand, the IK backend, and
Async / Sync / ChunkRL before loading the model. ChunkRL accepts both 10-D
single-hand and 20-D bimanual policies and runs a decider only for enabled arms.
During inference, the 3D viewer overlays the measured observation and accepted
action TCP poses; live joint-state updates remain enabled. The timing panel
shows observation, preprocessing, policy, postprocessing, request, action-buffer,
and total durations, including warm-up samples and errors. Postprocessing
includes the separately displayed IK/ChunkRL times. GR00T uses the same timing
transport; EEF overlays are provided for relative EEF policies.

Live Diffusion observations use receive-timestamped rings for the policy cameras
and robot state. LOAD warms up the history; inference selects a fixed window at
the configured Dataset FPS without waiting for another frame. Missing, stale,
or misaligned history is rejected. The relative TCP anchor uses the latest state
in that same snapshot. Sampling age and alignment diagnostics accompany timing
telemetry; dataset feedback continues to use its recorded frame history.

Cyclo Intelligence is a Physical AI platform for data recording, dataset
conversion, policy training, inference, and robot execution with AI Worker.
For detailed usage instructions, please refer to the documentation below.
  - [Documentation for AI Worker Imitation Learning](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/)

To install Cyclo Intelligence:

```bash
curl -fsSL https://raw.githubusercontent.com/ROBOTIS-GIT/cyclo_intelligence/main/install.sh | bash
```

On ROBOTIS robot PCs whose host name starts with `ffw`, the installer places
the checkout under `/mnt/ssd/cyclo_intelligence` and bind-mounts it at
`~/cyclo_intelligence`. On other PCs, it installs directly under
`~/cyclo_intelligence`.

To learn more about the ROS 2 packages for the AI Worker, visit:
  - [AI Worker ROS 2 Packages](https://github.com/ROBOTIS-GIT/ai_worker)

To explore our open-source platforms in a simulation environment, visit:
  - [Simulation Models](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)

For usage instructions and demonstrations of the AI Worker, check out:
  - [Tutorial Videos](https://www.youtube.com/@ROBOTISOpenSourceTeam)

To access datasets and pre-trained models for our open-source platforms, see:
  - [AI Models & Datasets](https://huggingface.co/ROBOTIS)

To use Docker images for running Cyclo Intelligence, visit:
  - [Docker Images](https://hub.docker.com/u/robotis)

### Inference rosbag2 logs

In the Inference settings, enable **Record inference · rosbag2** before Start
(default: off). This applies to live Async, Sync, and ChunkRL in both Real Robot
and 3D Sim Deploy. The recorder starts before START, continues through Pause /
Resume, and saves after Clear has stopped the control loop, completed in-flight
Engine work, and unloaded the policy. A busy recorder blocks a logged Start
instead of silently running without the requested log. The GUI displays the bag
path. Use Clear to finish this recording; the Record page cannot discard an
active automatic inference log.

Logs use the existing dataset recording v2 layout:
`docker/workspace/rosbag2/Task_<UTC timestamp>_inference_MCAP/<episode>/`
on the host (`/workspace/rosbag2/...` in the container). The folder contains
rosbag2 `metadata.yaml` + MCAP, camera videos with frame timestamps, camera-info
snapshots, `episode_info.json`, and `inference_log.json` with the policy settings
and recording bounds. The normal robot state/action topics are preserved.
`/inference/trajectory_preview` also records the interpolated action in preview
mode, where robot command publishing is disabled.

`/<backend>/inference_events` is `std_msgs/msg/String` containing JSON. Each
event has `schema_version`, `session_id`, `process` (main/engine), `process_id`,
`clock_id`, increasing per-process `event_index`, `wall_time_ns` (Unix epoch),
and `monotonic_ns`. Requests and their outputs share `seq_id` across processes.

| Event | Meaning |
| --- | --- |
| `request_started` / `request_finished` | Main dispatch / response, including timeout or rejection |
| `engine_request_started` / `engine_request_finished` | Engine receipt / completion; final timing and observation-window diagnostics |
| `stage_started` / `stage_finished` | observation, preprocess, policy, postprocess, and nested IK / ChunkRL |
| `observation_joints` / `observation_state` | Selected joint snapshots and actual policy state input; no image arrays in JSON |
| `policy_action` / `action_chunk` | Denormalized policy predictions and final selected joint action chunk before Main interpolation |
| `buffer_started` / `buffer_finished` | Main decoding/alignment/buffer acceptance, with stale/rejected results marked |
| `control_tick` / `control_state` | Per-tick target, command publication result, buffer size, executing/empty/sync/paused state |
| `initial_pose_target`, `initial_pose_sync_finished`, `pose_hold_*` | Slow startup target, sync completion, and interrupted-sync hold results |
| `lifecycle_requested` / `lifecycle_finished`, `session_config` | Start, Pause, Resume, Stop, Unload, and policy/run configuration |

For stop-time analysis, use monotonic differences **only for matching
`clock_id`**; wall timestamps align ROS messages across machines and can jump
with clock adjustments. `control_state=buffer_empty` measures command
starvation. Determine physical rest from actual joint feedback (positions and,
when available, velocities) with an explicit motion threshold. A non-empty
buffer can also command a stationary pose. In async mode, interpolation/alignment
can combine chunks: a tick's `latest_buffer_seq_id` names the latest buffer
update, not exclusive ownership of that interpolated command. Missing event
indices or sensor gaps must be treated as missing data, not stationary time.
Disabled logging does not serialize state/action tensors or write bag files.
