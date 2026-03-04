# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, TiledCameraCfg  # noqa: F401
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.ur10e.observations import gripper_pos
from isaaclab_arena.embodiments.ur10e.proximity_gripper import ProximityGripperActionCfg
from isaaclab_arena.utils.pose import Pose

# Default camera offset for a wrist-mounted camera on the UR10E + Robotiq gripper
_DEFAULT_CAMERA_OFFSET = Pose(position_xyz=(0.05, -0.03, -0.07), rotation_wxyz=(-0.74896, 0.0, 0.0, -0.66262))

# UR10E with Robotiq 2F-140 gripper configuration
_UR10E_CFG = UR10e_ROBOTIQ_GRIPPER_CFG.copy()
# Enable contact sensors for pick-and-place tasks
_UR10E_CFG.spawn.activate_contact_sensors = True
# Increase actuator stiffness and damping for better IK tracking
# (similar to Franka's HIGH_PD_CFG which is recommended for differential IK control)
_UR10E_CFG.actuators["shoulder"].stiffness = 2000.0
_UR10E_CFG.actuators["shoulder"].damping = 100.0
_UR10E_CFG.actuators["elbow"].stiffness = 1000.0
_UR10E_CFG.actuators["elbow"].damping = 60.0
_UR10E_CFG.actuators["wrist"].stiffness = 500.0
_UR10E_CFG.actuators["wrist"].damping = 40.0

_UR10E_CFG.actuators["shoulder"].velocity_limit_sim = 2.0
_UR10E_CFG.actuators["elbow"].velocity_limit_sim = 2.5
_UR10E_CFG.actuators["wrist"].velocity_limit_sim = 3.0

_UR10E_CFG.actuators["gripper_drive"].stiffness = 2000.0
_UR10E_CFG.actuators["gripper_drive"].damping = 100.0
_UR10E_CFG.actuators["gripper_drive"].effort_limit_sim = 200.0
_UR10E_CFG.actuators["gripper_drive"].velocity_limit_sim = 10.0

_UR10E_CFG.actuators["gripper_finger"].stiffness = 100.0
_UR10E_CFG.actuators["gripper_finger"].damping = 10.0
_UR10E_CFG.actuators["gripper_finger"].effort_limit_sim = 50.0
_UR10E_CFG.actuators["gripper_finger"].velocity_limit_sim = 10.0

_UR10E_CFG.actuators["gripper_passive"].velocity_limit_sim = 10.0


@register_asset
class UR10eEmbodiment(EmbodimentBase):
    """Embodiment for the UR10E robot with Robotiq 2F-140 gripper."""

    name = "ur10e"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        camera_offset: Pose | None = _DEFAULT_CAMERA_OFFSET,
        is_tiled_camera: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = UR10eSceneCfg()
        self.action_config = UR10eActionsCfg()
        self.observation_config = UR10eObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = UR10eEventCfg()
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.reward_config = UR10eRewardsCfg()
        self.mimic_env = UR10eMimicEnv
        self.camera_config = UR10eCameraCfg()
        self.camera_config._is_tiled_camera = is_tiled_camera
        self.camera_config._camera_offset = camera_offset

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        self.event_config.init_ur10e_arm_pose.params["default_pose"] = initial_joint_pose

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name


@configclass
class UR10eSceneCfg:
    """Additions to the scene configuration coming from the UR10E embodiment."""

    # The robot (UR10E with Robotiq 2F-140 gripper)
    robot: ArticulationCfg = _UR10E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The end-effector frame marker
    # UR10E: ee_link is NOT a rigid body, so we use wrist_3_link as reference.
    # From the UR URDF, the ee_fixed_joint has rpy=(0,0,π/2) meaning:
    #   ee_link local X = wrist_3_link local Y
    #   ee_link local Z = wrist_3_link local Z
    # Tool direction = wrist_3_link LOCAL Z-axis (confirmed by UR10E reach config).
    # Robotiq 2F-140 TCP is ~0.24m from wrist_3_link along local Z
    #   (d6=0.0922m to flange + ~0.15m gripper to fingertip center)
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.24),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(-0.01, 0.0, 0.22),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.01, 0.0, 0.22),
                ),
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class UR10eActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        body_name="wrist_3_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.1,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.24)),
    )

    gripper_action: ActionTermCfg = ProximityGripperActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7},
        object_name="dex_cube",
        ee_frame_name="ee_frame",
        close_threshold=0.015,
    )


@configclass
class UR10eCameraCfg:
    """Configuration for cameras."""

    wrist_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled_camera = getattr(self, "_is_tiled_camera", False)
        camera_offset = getattr(self, "_camera_offset", _DEFAULT_CAMERA_OFFSET)

        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        common_kwargs = dict(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/wrist_cam",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8, focus_distance=28, horizontal_aperture=5.376, vertical_aperture=3.024
            ),
        )
        offset = OffsetClass(
            pos=camera_offset.position_xyz,
            rot=camera_offset.rotation_wxyz,
            convention="ros",
        )

        self.wrist_cam = CameraClass(offset=offset, **common_kwargs)


@configclass
class UR10eObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR10eEventCfg:
    """Configuration for UR10E reset events."""

    # Default joint pose for UR10E + Robotiq 2F-140 gripper
    # 6 arm joints + 8 gripper joints = 14 total
    # Arm extends forward with gripper oriented DOWNWARD toward the table,
    # ready for top-down manipulation. This avoids the lateral sweep that
    # occurs when starting from a compact/tool-up configuration.
    init_ur10e_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                0.0,          # shoulder_pan_joint (face +X toward table/cube)
                -2.3562,      # shoulder_lift_joint (-3π/4, upper arm angled ~45° below horizontal)
                1.5708,       # elbow_joint (π/2 forearm bend brings wrist toward table)
                -0.7854,      # wrist_1_joint (-π/4, orients gripper downward)
                -1.5708,      # wrist_2_joint (-π/2)
                0.0,          # wrist_3_joint
                0.0,          # finger_joint (Robotiq open)
                0.0,          # right_outer_knuckle_joint
                0.0,          # left_outer_finger_joint
                0.0,          # right_outer_finger_joint
                0.0,          # left_inner_finger_joint
                0.0,          # right_inner_finger_joint
                0.0,          # left_inner_finger_pad_joint
                0.0,          # right_inner_finger_pad_joint
            ],
        },
    )
    randomize_ur10e_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class UR10eRewardsCfg:
    """Reward specifications for the MDP."""

    action_rate = RewardTermCfg(func=mdp_isaac_lab.action_rate_l2, weight=-0.001)
    joint_vel = RewardTermCfg(
        func=mdp_isaac_lab.joint_vel_l2,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
            )
        },
    )


# Mimic environment for UR10E — same logic as Franka, adapted for UR10E action space
class UR10eMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic environment for UR10E with Robotiq gripper."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose.
        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action to a target pose for the end effector controller.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions.
        """
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object in the current scene.
        """
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)
        object_pose_matrix = get_rigid_and_articulated_object_poses(state, env_ids)

        return object_pose_matrix

