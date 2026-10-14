# AI Worker DLS IK

AI Worker (`ffw_sg2_rev1`)의 양팔과 공용 리프트를 함께 푸는 IK입니다.
Modern Robotics의 body pose 오차와 DLS·영공간 계산을 작은 Python API로 구성했습니다.
TCP 위치와 방향을 우선하고, 여유 자유도에서 시작 자세의 팔꿈치 위치와 굽힘을 유지합니다.

## 메소드 호출

```python
from dls_ik import DLSIK, DLSOptions

ik = DLSIK("ffw_sg2_follower.urdf", options=DLSOptions(
    damping=0.01,
    nullspace_gain=0.15,
    elbow_position_weight=4.0,
    elbow_angle_weight=1.0,
    lift_weight=10.0,
))

# measured: {joint_name: value}. 회전 관절은 rad, 리프트는 m.
ik.set_reference(measured)
left_tcp, right_tcp = ik.forward(measured)
left_tcp[2, 3] += 0.02
joints, position_error, angle_error = ik.solve(measured, left_tcp, right_tcp)
print(ik.last_diagnostics)
```

`solve()`는 왼팔 7개·오른팔 7개·리프트 1개의 관절값과 최대 위치 오차(m),
최대 방향 오차(rad)를 반환합니다. 그리퍼는 기존 정책 액션 어댑터에서 처리합니다.
다음 waypoint에는 직전 해를 seed로 사용합니다. 팔꿈치 기준은 세션 시작 자세로
유지하며, `set_reference()`를 생략하면 첫 solve의 seed를 기준으로 잡습니다.
새 기준 자세는 `set_reference()`로 지정합니다.

팔꿈치 보조 목표는 어깨에 대한 팔꿈치 중심의 상대 위치와 J4의 시작 각도입니다.
`nullspace_gain=0`이면 보조 목표를 끕니다. TCP 목표와 관절 한계를 우선하므로
팔꿈치 각도를 특정 값에 계속 고정하는 제약은 아닙니다.

## Diffusion 추론과 GUI

Inference에서 LeRobot의 상대 EEF Diffusion 체크포인트를 선택하고,
**EEF IK → DLS · Elbow Hold**를 선택합니다. 별도의 IK 체크포인트는 필요 없습니다.
모델이나 IK 모드를 변경할 때는 기존 세션을 Unload한 뒤 다시 Load합니다.

IK 이후 양팔·리프트·그리퍼 명령은 NodeIK와 동일한 5차 곡선으로 보간합니다.
실제 제어의 sync/async 청크와 ChunkRL 단일 waypoint에 적용되며,
기본 설정인 `POSTPROCESS_ACTIONS=true`에서 **Control Hz**를 따릅니다.

```python
u = t / duration
weight = 10 * u**3 - 15 * u**4 + 6 * u**5
q = q_start + weight * (q_target - q_start)
```

각 구간의 시작·끝 속도와 가속도를 0으로 놓습니다. `solve()`는 목표 q를 반환하고,
보간은 제어와 시뮬레이션 경로에서 처리합니다. 이 곡선은 관절의 양 끝 값 사이에서
움직이며, TCP 직선 경로나 관절 속도·가속도 제한을 보장하지 않습니다.

Dataset Feedback Simulation도 현재 q부터 다음 IK 결과까지 중간 자세를 표시합니다.
30 FPS 데이터셋·100 Hz 제어에서는 구간마다 10·20·30·33.333ms의 자세를 표시합니다.
다음 정책 관측 이력에는 최종 q만 넣고, 카메라는 데이터셋 프레임 경계에서 진행합니다.

## 초기 자세 설정

**Use initial frame state**를 켜면 선택한 데이터셋 프레임의 관절 state를 사용합니다.
끄면 `config/initial_state.json`의 값을 관절 이름으로 대응시켜 전체 초기 관측 이력을
채웁니다. 카메라는 두 경우 모두 선택한 데이터셋 프레임을 사용합니다.
API에서는 `simulation_config.use_initial_frame_state`로 선택하며 기본값은 `true`입니다.

기본 설정은 양손이 가슴 앞에 나란히 있고 손목 카메라가 아래를 바라보는 자세입니다.
TCP의 공통 x는 0.485632m, 공통 z는 1.084526m이며,
왼손 y는 +0.173552m, 오른손 y는 −0.194970m입니다.
팔 외 관절은 지정한 데이터셋의 첫 프레임 평균을 유지합니다.

- `config/initial_state.json`: 관절과 베이스를 포함한 실행용 초기 state.
- `config/ffw_sg2_follower_initial_positions.yaml`: 대응하는 로봇 home 설정.
- `config/initial_state_first_frame_means.json`: 초기 자세 생성용 기준 값 및 테스트 fixture.
- `config/zero_home.yaml`: home YAML 입출력 테스트용 원본 fixture.

`dls_ik.pose_constraints.floor_camera_initial_pose()`로 리프트와 팔 외 값을 유지하면서
가슴 앞 TCP·카메라 조건을 만족하는 팔 자세를 계산할 수 있습니다.
`dls_ik.initial_pose.read_home()`과 `write_home()`는 반복된 ROS wildcard 섹션을 보존해
home YAML을 읽고 씁니다. SDK 자체는 로봇 전송 연결이나 명령 발행을 하지 않습니다.

## Docker 테스트

```bash
docker exec -e PYTHONPATH=/opt/cyclo_sdk/dls_ik lerobot_server \
  python -m pytest /opt/cyclo_sdk/dls_ik/tests -q -o cache_dir=/tmp/dls_pytest_cache
```

보고서·실험 스크립트·벤치마크·검증 결과는 로컬 자료로 관리하며 Git에서 제외합니다.
회귀 테스트와 실행용 설정은 구현 코드와 함께 관리합니다.
