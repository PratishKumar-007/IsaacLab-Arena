# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def approach_ee_above_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for positioning the EEF directly above the object.

    Encourages a top-down approach by rewarding small horizontal (XY) distance
    between the EEF and the object, but ONLY when the EEF is above the object.
    This prevents the lateral approach that causes finger-object collisions.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w

    xy_distance = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], dim=1)
    height_above = ee_pos[:, 2] - obj_pos[:, 2]

    is_above = (height_above > 0.02).float()
    xy_reward = 1 - torch.tanh(xy_distance / std)

    return is_above * xy_reward


def fingertips_close_to_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for having both fingertips close to the object.

    Both Franka and UR10e define tool_rightfinger (index 1) and tool_leftfinger
    (index 2) in their ee_frame target_frames, making this embodiment-agnostic.
    Uses a tanh kernel on the mean fingertip-to-object distance.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = obj.data.root_pos_w
    right_finger_pos = ee_frame.data.target_pos_w[..., 1, :]
    left_finger_pos = ee_frame.data.target_pos_w[..., 2, :]

    dist_right = torch.norm(object_pos_w - right_finger_pos, dim=1)
    dist_left = torch.norm(object_pos_w - left_finger_pos, dim=1)
    mean_dist = (dist_right + dist_left) / 2.0

    return 1 - torch.tanh(mean_dist / std)


def close_gripper_near_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for commanding the gripper to close when the EEF is near the object.

    Reads the raw gripper action (last dimension of the action buffer).
    For BinaryJointPositionActionCfg: action > 0 means "close".
    Combined with a tanh proximity kernel so the reward is only active near the object.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_pos, dim=1)
    near_score = 1 - torch.tanh(distance / std)

    gripper_action = env.action_manager.action[:, -1]
    is_closing = (gripper_action > 0).float()

    return near_score * is_closing
