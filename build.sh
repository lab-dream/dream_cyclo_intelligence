#!/usr/bin/env bash
# Apply the checked-out source to the Cyclo UI/ROS and policy containers.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
REBUILD=0
WITH_GROOT=0
usage() {
    cat <<'EOF'
Usage: ./build.sh [--rebuild-images] [--with-groot]

Default: initialize pinned runtime submodules, create missing Cyclo/LeRobot
containers, rebuild ROS interfaces and UI, and restart with the current source.
Existing compatible containers are reused (including their Python environments).

  --rebuild-images   Rebuild selected Docker images instead of reusing/pulling them.
  --with-groot       Also provision/update GR00T (large image and submodule).

Requires Docker Engine + Compose v2, git, Python 3, and NVIDIA Container Toolkit
for GPU inference. Models/datasets remain under docker/workspace; weights are
not included in this repository and are not downloaded by this script.
EOF
}
for option in "$@"; do
    case "$option" in
        --rebuild-images) REBUILD=1 ;;
        --with-groot) WITH_GROOT=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$option" >&2; usage >&2; exit 2 ;;
    esac
done
for dependency in git docker python3; do
    command -v "$dependency" >/dev/null || { echo "Missing prerequisite: $dependency" >&2; exit 1; }
done
docker info >/dev/null
docker compose version >/dev/null
case "$(uname -m)" in
    x86_64) export ARCH=amd64 ;;
    aarch64|arm64) export ARCH=arm64 ;;
    *) echo 'Supported architectures: x86_64 and aarch64.' >&2; exit 1 ;;
esac
if [[ -z "${GPU_ARCH:-}" ]]; then
    GPU_ARCH=default
    if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q '^12\.'; then
        GPU_ARCH=blackwell
    fi
    export GPU_ARCH
fi
COMPOSE=(docker compose -f "$ROOT/docker/docker-compose.yml")
[[ ! -f "$ROOT/docker/docker-compose.override.yml" ]] || COMPOSE+=(-f "$ROOT/docker/docker-compose.override.yml")
SUBMODULES=(cyclo_brain/policy/lerobot/lerobot cyclo_brain/sdk/zenoh_ros2_sdk cyclo_brain/sdk/cyclo_control cyclo_brain/sdk/nodeik)
SERVICES=(cyclo_intelligence lerobot)
CONTAINERS=(cyclo_intelligence lerobot_server)
if (( WITH_GROOT )); then
    SUBMODULES+=(cyclo_brain/policy/groot/Isaac-GR00T)
    SERVICES+=(groot)
    CONTAINERS+=(groot_server)
fi
echo '[build] Initializing pinned runtime submodules (no remote branch upgrades).'
git submodule sync --recursive -- "${SUBMODULES[@]}"
git submodule update --init --recursive -- "${SUBMODULES[@]}"
mkdir -p docker/workspace/{dataset,model/lerobot,model/chunkrl,model/groot,rosbag2,lerobot,.build-state} docker/huggingface docker/zenoh_cache
if (( REBUILD )); then
    "${COMPOSE[@]}" build "${SERVICES[@]}"
fi
for i in "${!SERVICES[@]}"; do
    service="${SERVICES[i]}"
    container="${CONTAINERS[i]}"
    if docker container inspect "$container" >/dev/null 2>&1 && (( ! REBUILD )); then
        # Preserve working environments and local ROS router settings. A container
        # mounted from another checkout must be explicitly recreated, not silently
        # report that this checkout was applied while continuing to serve old code.
        actual="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$container")"
        if [[ "$(readlink -f "$actual")" != "$(readlink -f "$ROOT/docker/workspace")" ]]; then
            echo "Container $container mounts /workspace from $actual, not this checkout." >&2
            echo 'Back up its container-local configuration, remove that stale container, and rerun ./build.sh.' >&2
            exit 1
        fi
        docker start "$container" >/dev/null
    else
        if (( ! REBUILD )); then
            image="$("${COMPOSE[@]}" config --format json | python3 -c 'import json,sys;print(json.load(sys.stdin)["services"][sys.argv[1]]["image"])' "$service")"
            if ! docker image inspect "$image" >/dev/null 2>&1; then
                if ! "${COMPOSE[@]}" pull "$service"; then
                    echo "[build] Image pull failed; building $service from this checkout."
                    "${COMPOSE[@]}" build "$service"
                fi
            fi
        fi
        "${COMPOSE[@]}" up -d --no-deps "$service"
    fi
done

echo '[build] Building current ROS packages and generated service interfaces.'
docker exec cyclo_intelligence bash -c '
    set -eo pipefail
    source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
    cd /root/ros2_ws
    colcon build --symlink-install --paths \
        src/cyclo_intelligence/interfaces src/cyclo_intelligence/shared \
        src/cyclo_intelligence/orchestrator src/cyclo_intelligence/cyclo_data \
        src/cyclo_intelligence/cyclo_data/recorder/rosbag_recorder \
        src/robotis_interfaces \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
'
echo '[build] Installing UI dependencies from package-lock and rebuilding nginx assets.'
ui_hash="$(sha256sum orchestrator/ui/package{,-lock}.json | sha256sum | cut -d ' ' -f 1)"
if [[ "$(cat docker/workspace/.build-state/ui-dependencies.sha 2>/dev/null || true)" != "$ui_hash" ]] || \
   ! docker exec cyclo_intelligence test -x /root/ros2_ws/src/cyclo_intelligence/orchestrator/ui/node_modules/.bin/react-scripts; then
    docker exec -w /root/ros2_ws/src/cyclo_intelligence/orchestrator/ui \
        cyclo_intelligence npm ci --legacy-peer-deps --no-audit --no-fund
    printf '%s\n' "$ui_hash" > docker/workspace/.build-state/ui-dependencies.sha
fi
docker exec -w /root/ros2_ws/src/cyclo_intelligence/orchestrator/ui cyclo_intelligence npm run build
docker cp "$ROOT/orchestrator/ui/build/." cyclo_intelligence:/usr/share/nginx/html/
docker exec cyclo_intelligence nginx -s reload
# /root is private in the image. Build as root, then return only generated UI
# directories to the host user instead of making /root world-accessible.
docker exec cyclo_intelligence chown -R "$(id -u):$(id -g)" \
    /root/ros2_ws/src/cyclo_intelligence/orchestrator/ui/node_modules \
    /root/ros2_ws/src/cyclo_intelligence/orchestrator/ui/build

echo '[build] Checking the policy environment and GPU support.'
docker exec -e GPU_ARCH="$GPU_ARCH" lerobot_server bash -c '
    set -eo pipefail
    if ! python -c "import pinocchio" >/dev/null 2>&1; then
        uv pip install --python /lerobot/.venv/bin/python pin==3.4.0
    fi
    if ! python -c "import modern_robotics" >/dev/null 2>&1; then
        uv pip install --python /lerobot/.venv/bin/python modern_robotics==1.1.1
    fi
    if ! python -c "import torch; assert torch.cuda.is_available(); print(torch.ones(2,device=\"cuda\").sin().cpu())"; then
        if [ "$GPU_ARCH" != blackwell ]; then
            echo "GPU probe failed. Check the NVIDIA driver and Container Toolkit." >&2
            exit 1
        fi
        uv pip install --python /lerobot/.venv/bin/python --upgrade \
            --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchcodec
        python -c "import torch; print(torch.ones(2,device=\"cuda\").sin().cpu())"
    fi
    PYTHONPATH=/app:/policy_runtime:/zenoh_sdk:/robot_client_sdk:/action_chunk_processing_sdk:/opt/cyclo_sdk/dls_ik \
        python -c "from lerobot_engine import LeRobotEngine; from dls_ik import DLSIK; from engine_process.protocol import ENGINE_COMMAND_RESPONSE_DEF; assert \"string telemetry\" in ENGINE_COMMAND_RESPONSE_DEF"
'
# Restart after successful builds: Python imports, ROS type hashes, and the UI
# now all refer to the same checked-out protocol. User data volumes are retained.
echo '[build] Restarting containers to load current Python modules and ROS types.'
docker restart "${CONTAINERS[@]}" >/dev/null
for container in "${CONTAINERS[@]}"; do
    healthy=0
    for attempt in {1..60}; do
        status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container")"
        if [[ "$status" == healthy || "$status" == running ]]; then healthy=1; break; fi
        if [[ "$status" == exited || "$status" == dead ]]; then break; fi
        sleep 1
    done
    if (( ! healthy )); then
        echo "$container did not become healthy. Inspect: docker logs $container" >&2
        exit 1
    fi
    echo "[build] $container: $status"
done
if [[ "${CYCLO_START_LOCAL_ROUTER:-auto}" != 0 ]] && \
   ! python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",7447),timeout=1); s.close()' 2>/dev/null; then
    echo '[build] Starting a local Zenoh router for ROS discovery and policy services.'
    if docker container inspect cyclo_zenoh_router >/dev/null 2>&1; then
        docker start cyclo_zenoh_router >/dev/null
    else
        router_image="$(docker inspect -f '{{.Image}}' cyclo_intelligence)"
        docker run -d --name cyclo_zenoh_router --restart unless-stopped --network host \
            --entrypoint /bin/bash "$router_image" -c \
            'source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash; exec ros2 run rmw_zenoh_cpp rmw_zenohd' >/dev/null
    fi
    for attempt in {1..30}; do
        if python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",7447),timeout=1); s.close()' 2>/dev/null; then break; fi
        sleep 1
    done
    python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",7447),timeout=1); s.close()'
fi
echo '[build] Starting the ROS bridge, orchestrator, and data services for the Web GUI.'
docker exec cyclo_intelligence /command/s6-rc -u change orchestrator cyclo_data
echo '[build] Source update applied. Open http://localhost:7080 and refresh the browser.'
echo '[build] Select your robot, policy, and ChunkRL actor checkpoint in the Web GUI.'
