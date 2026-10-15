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
