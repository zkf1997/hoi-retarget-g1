"""WholeBodyTrackingManager that allows an action dimension smaller than the number of DoFs (finger synergies)."""
from __future__ import annotations

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.utils.torch_utils import to_torch


class HiPHIWBTManager(WholeBodyTrackingManager):
    def _load_assets(self):  # copy of BaseTask._load_assets without the num_dof == dim_actions assertion
        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self.simulator.num_dof, self.simulator.num_bodies, self.simulator.dof_names, self.simulator.body_names)
        if self.num_dof != self.dim_actions:
            print(f"[HiPHIWBTManager] {self.dim_actions} policy actions for {self.num_dof} DoFs (finger synergies)")
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        base_init_state_list = (self.robot_config.init_state.pos + self.robot_config.init_state.rot
                                + self.robot_config.init_state.lin_vel + self.robot_config.init_state.ang_vel)
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
