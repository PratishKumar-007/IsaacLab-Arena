# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the Robotiq 2F-140 gripper position.

    The Robotiq 2F-140 gripper uses a single ``finger_joint`` to drive both fingers.
    We return the joint position normalized to [0, 1] where 0 is fully open and 1 is fully closed.
    The maximum travel of finger_joint is approximately 0.7 rad for the Robotiq 2F-140.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    # Normalize: Robotiq 2F-140 finger_joint range is approximately [0, 0.7] rad
    return joint_pos / 0.7

