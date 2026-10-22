"""Diagnostic preset: holosoma's STOCK G1 (29 DoF, sphere hands, stock box) tracking one of OUR HiPHI motion files.

The MotionLoader maps bodies / joints by name, and every stock G1 body and joint exists in our 103-body / 69-joint npz, so the
stock experiment can consume the file unchanged (finger columns are ignored). Separates "our data is hard" from "our robot /
hand setup is slow to learn".

    STOCK_MOTION=omomo_bench/wbt/data/motions/002_....npz python train_agent.py --import-file omomo_bench/wbt/stock_on_hiphi.py exp:stock-on-hiphi ...
"""
import os
from dataclasses import replace

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY, g1_29dof_wbt_w_object

base = g1_29dof_wbt_w_object
mc = base.command.setup_terms["motion_command"]
mcfg = replace(mc.params["motion_config"], motion_file=os.path.abspath(os.environ["STOCK_MOTION"]))
command = replace(base.command, setup_terms={**base.command.setup_terms, "motion_command": replace(mc, params={**mc.params, "motion_config": mcfg})})
stock_on_hiphi = replace(base, command=command, training=TrainingConfig(project="HiPHI_WBT_fast", name="stock_on_hiphi", num_envs=1024))
EXPERIMENT_REGISTRY.add("stock_on_hiphi", stock_on_hiphi)

# ---- bisection: the stock task config (rewards / obs / termination / action = default pose + action) with OUR robot.
# HIPHI_WBT_FROZEN_HAND=1 -> 29-DoF variant with the fingers fixed (hand mass kept), else the full 69-DoF G1 + Wuji hand.
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hiphi_wbt.robot import g1_wuji  # noqa: E402

_name = "stock_on_hiphi_wuji" + ("_frozen" if os.environ.get("HIPHI_WBT_FROZEN_HAND", "0") == "1" else "") + ("_collapsed" if os.environ.get("HIPHI_WBT_COLLAPSE", "0") == "1" else "")
stock_on_hiphi_wuji = replace(
    stock_on_hiphi,
    robot=replace(g1_wuji, asset=replace(g1_wuji.asset, enable_self_collisions=False)),
    training=TrainingConfig(project="HiPHI_WBT_fast", name=_name, num_envs=1024),
)
EXPERIMENT_REGISTRY.add("stock_on_hiphi_wuji", stock_on_hiphi_wuji)

# ---- bisection: stock robot whose sphere hands weigh 0.92 kg (= our wrist + Wuji hand) -> isolates the hand-mass effect
_heavy = replace(base.robot, asset=replace(base.robot.asset, urdf_file="g1/main_mesh_collision_halfspherehand_heavy.urdf"))
EXPERIMENT_REGISTRY.add("stock_on_hiphi_heavyhand", replace(stock_on_hiphi, robot=_heavy,
                                                           training=TrainingConfig(project="HiPHI_WBT_fast", name="stock_on_hiphi_heavyhand", num_envs=1024)))
