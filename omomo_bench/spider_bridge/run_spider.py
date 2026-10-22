"""SPIDER run_mjwp.py with optional torque-space sampling noise for low-gain (real) actuators.

SPIDER samples position-target noise of the same size (joint_noise_scale rad) on every joint. With stiff actuators
(kp 500) that is 50 Nm of exploration; with the real G1/Wuji gains (arm kp 14, finger kp 0.6) it is ~1 Nm and the
sampler cannot find the torques the motion needs. TORQUE_NOISE=<c> sets the per-joint noise (before SPIDER's knot
annealing) to c * effort_limit / kp rad (capped at TORQUE_NOISE_MAX, default 0.5), i.e. ~c of the torque limit, like the
BeyondMimic action scale (0.25 * effort / kp). For reference, stiff kp 500 with 0.1 rad is ~2x effort on the arms and
~0.4x on the knees.

    cd ~/projects/spider && TORQUE_NOISE=0.25 MUJOCO_GL=egl ~/software/venvs/spider/bin/python \
        ~/projects/retarget/omomo_bench/spider_bridge/run_spider.py +override=g1_wuji_hiphi task=<tag> dataset_name=hiphi_real ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

SPIDER = Path(os.environ.get("SPIDER_ROOT", Path.home() / "projects" / "spider"))
sys.path.insert(0, str(SPIDER / "examples"))

import spider.config as C  # noqa: E402

_get_noise_scale = C.get_noise_scale


def get_noise_scale(config):
    ns = _get_noise_scale(config)
    c = float(os.environ.get("TORQUE_NOISE", "0") or 0)
    if c <= 0:
        return ns
    scene = Path(config.dataset_dir) / "processed" / config.dataset_name / config.robot_type / config.embodiment_type / config.task / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(scene))
    kp, effort = m.actuator_gainprm[:, 0], m.actuator_forcerange[:, 1]
    rad = np.minimum(c * effort / kp, float(os.environ.get("TORQUE_NOISE_MAX", "0.5")))
    mult = torch.as_tensor(rad / config.joint_noise_scale, dtype=ns.dtype, device=ns.device)  # replaces the uniform joint_noise_scale
    print(f"[run_spider] torque-space noise: c={c}, rad per joint min {rad.min():.3f} median {np.median(rad):.3f} max {rad.max():.3f}")
    ns[..., : m.nu] *= mult
    return ns


C.get_noise_scale = get_noise_scale

if __name__ == "__main__":
    import runpy

    sys.argv[0] = str(SPIDER / "examples" / "run_mjwp.py")  # run as __main__ so hydra finds examples/config
    runpy.run_path(sys.argv[0], run_name="__main__")
