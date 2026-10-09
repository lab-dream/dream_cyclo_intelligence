# Copied relative EEF helpers

`dream/` copies `src/lerobot/state_free/{__init__,ai_worker,math,relative_eef_pose}.py`
and the single expanded SG2 Rev1 URDF from
`/media/son/Remember/AX_Humanoid/dream_lerobot`. The Python files and URDF are
unchanged. `LICENSE` and `AI_WORKER_LICENSE` preserve the upstream notices.
Source commit: `17c79a243a08ff1be8e585b17c50481a850a0759`.

`umi/` copies `umi/common/pose_util.py`, `umi/real_world/real_inference_util.py`,
`diffusion_policy/common/pose_repr_util.py`, and `diffusion_policy/common/cv2_util.py`
from `/media/son/Remember/AX_Humanoid/universal_manipulation_interface`.
Only `real_inference_util.py`'s imports use local relative paths, and its unused
`RotationTransformer` import is removed. All function bodies are unchanged.
The original MIT license is included.
Source commit: `d095ba9590df789df5189eea5ee7e431689038a6`.

`../relative_eef.py` supplies the Cyclo-specific joint/camera schema bridge,
Dream's gripper ranges, and the existing Cyclo IK call. UMI's hardware environment
and timing loop are handled by Cyclo's existing RobotClient and common runtime.
