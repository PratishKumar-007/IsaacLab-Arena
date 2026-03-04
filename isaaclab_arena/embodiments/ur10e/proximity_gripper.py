from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ProximityGripperAction(BinaryJointPositionAction):
    """Gripper that closes automatically when the EEF is near the target object.

    Inherits from BinaryJointPositionAction but overrides process_actions
    to ignore the policy output and instead close/open based on EEF-to-object
    distance. The policy still outputs a dummy value for this action dimension
    (action_dim=1) but it is completely ignored.
    """

    cfg: ProximityGripperActionCfg

    def __init__(self, cfg: ProximityGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._close_threshold = cfg.close_threshold
        self._object_name = cfg.object_name
        self._ee_frame_name = cfg.ee_frame_name

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        ee_frame = self._env.scene[self._ee_frame_name]
        obj = self._env.scene[self._object_name]

        ee_pos = ee_frame.data.target_pos_w[..., 0, :]
        obj_pos = obj.data.root_pos_w
        distance = torch.norm(obj_pos - ee_pos, dim=1)

        should_close = (distance < self._close_threshold).unsqueeze(-1)
        self._processed_actions = torch.where(should_close, self._close_command, self._open_command)


@configclass
class ProximityGripperActionCfg(BinaryJointPositionActionCfg):
    """Configuration for proximity-based gripper action.

    The gripper closes when the end-effector frame is within close_threshold
    of the target object. The policy's gripper action output is ignored.
    """

    class_type: type[ActionTerm] = ProximityGripperAction
    object_name: str = "object"
    ee_frame_name: str = "ee_frame"
    close_threshold: float = 0.08
