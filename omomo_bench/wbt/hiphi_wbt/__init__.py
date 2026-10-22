"""holosoma plugin: multi-sequence human-object interaction tracking for G1 + Wuji Hand 2 on the HiPHI retarget results.

Usage (venv with holosoma + IsaacSim/IsaacLab):
    PYTHONPATH=omomo_bench/wbt python holosoma/src/holosoma/holosoma/train_agent.py \
        --import-file omomo_bench/wbt/hiphi_wbt/presets.py exp:g1-wuji-hiphi-wbt logger:wandb_offline

Modules
  robot.py           RobotConfig for the 69-DoF G1 + Wuji Hand 2 (29 body + 40 finger joints), built from the URDF
  sim_multiasset.py  IsaacSim backend subclass (class name kept as IsaacSim): one object asset per environment (env i <- asset i % N)
  command.py         MotionCommand subclass: environment i always tracks motion i % N (its object matches), hand metrics
  action.py          joint-position action term whose finger targets are residuals on the reference finger pose
  rewards.py         fingertip / finger-joint tracking rewards
  presets.py         experiment presets (registered through --import-file)
"""
