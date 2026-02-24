# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""H1 humanoid embodiment configured for VLN navigation.

This embodiment configures the Unitree H1 humanoid robot for
vision-language navigation tasks:
  - Joint-position actions for the full body (controlled by the low-level
    RSL-RL locomotion policy).
  - Observations: proprioception (base angular velocity, projected gravity,
    velocity commands, joint pos/vel, last action) + camera RGB.
  - An RGBD camera mounted on the pelvis for first-person navigation.
  - Contact sensors and lighting for Matterport indoor scenes.

The low-level locomotion policy converts high-level velocity commands
[vx, vy, yaw_rate] into joint-position targets.  This embodiment provides
the observation and action spaces that the low-level policy expects.

Reference: NaVILA-Bench ``h1_matterport_base_cfg.py``.
"""

from __future__ import annotations

import math
from dataclasses import MISSING  # noqa: F401

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import H1_MINIMAL_CFG

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose


# ========================================================================== #
# Scene configuration: H1 robot articulation                                 #
# ========================================================================== #


@configclass
class H1VlnSceneCfg:
    """Scene configuration for the H1 VLN embodiment.

    Uses the official ``H1_MINIMAL_CFG`` from ``isaaclab_assets`` to ensure
    the robot articulation (joint names, init state, actuator PD gains) is
    exactly the same as what NaVILA uses for low-level policy training.

    Source: isaaclab_assets/robots/unitree.py -> H1_CFG / H1_MINIMAL_CFG
    """

    # Use the official H1 config directly — guarantees joint names, init
    # positions, and actuator PD gains match the NaVILA training config.
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Contact sensor on all robot links (for foot contact detection)
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )


# ========================================================================== #
# Camera configuration                                                       #
# ========================================================================== #

# Default camera offset: mounted on the pelvis, facing forward
_DEFAULT_H1_CAMERA_OFFSET = Pose(
    position_xyz=(0.1, 0.0, 0.5),
    rotation_wxyz=(-0.5, 0.5, -0.5, 0.5),
)


@configclass
class H1VlnCameraCfg:
    """Camera configuration for the H1 VLN embodiment.

    Mounts an RGB camera on the pelvis link.  Supports both
    ``CameraCfg`` (single-env) and ``TiledCameraCfg`` (parallel eval).
    """

    robot_head_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled = getattr(self, "_is_tiled_camera", False)
        cam_offset = getattr(self, "_camera_offset", _DEFAULT_H1_CAMERA_OFFSET)

        CameraClass = TiledCameraCfg if is_tiled else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        offset = OffsetClass(
            pos=cam_offset.position_xyz,
            rot=cam_offset.rotation_wxyz,
            convention="ros",
        )

        self.robot_head_cam = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis/VlnCamera",
            offset=offset,
            update_period=0.0,
            height=512,
            width=512,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                horizontal_aperture=54.0,
                clipping_range=(0.1, 10.0),
            ),
        )


# ========================================================================== #
# Observations                                                               #
# ========================================================================== #
# The "policy" observation group provides the concatenated proprioceptive
# vector that the NaVILA low-level locomotion policy expects.
#
# Layout (concatenated, in order):
#   base_ang_vel          (3)
#   projected_gravity     (3)
#   velocity_commands     (3)  <- [vx, vy, yaw_rate]
#   joint_pos_rel         (19)
#   joint_vel_rel         (19)
#   last_action           (19)
# Total: ~66 (depends on exact H1 DOF count)
#
# The high-level VLM policy only needs the camera RGB, which is served
# through the camera observation group added by ``enable_cameras=True``.


@configclass
class H1VlnObservationsCfg:
    """Observation groups for the H1 VLN embodiment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive observations consumed by the low-level locomotion policy."""

        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        """Duplicate proprio group for the history wrapper (NaVILA compatibility)."""

        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()


# ========================================================================== #
# Actions                                                                    #
# ========================================================================== #


@configclass
class H1VlnActionCfg:
    """Joint-position action space for the H1 robot.

    The low-level locomotion policy outputs target joint positions for all
    joints at 50 Hz (sim dt=0.005, decimation=4 -> 50 Hz control).
    """

    joint_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


# ========================================================================== #
# Commands                                                                   #
# ========================================================================== #


@configclass
class H1VlnCommandsCfg:
    """Velocity command generator.

    This provides the ``base_velocity`` command term that the proprioceptive
    observation ``velocity_commands`` reads from.  During VLN evaluation the
    actual velocity command is injected by the :class:`VLNEnvWrapper`
    directly into the observation buffer (indices 6:9 of the proprio vector).

    We set the sampling range to zero so the command generator does not
    randomly override the injected command.
    """

    # Ranges match the NaVILA training config.  During VLN evaluation the
    # actual velocity command is injected by VLNEnvWrapper directly into the
    # observation buffer, so the sampled command here is effectively unused.
    base_velocity: base_mdp.UniformVelocityCommandCfg = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ========================================================================== #
# Events                                                                     #
# ========================================================================== #


@configclass
class H1VlnEventCfg:
    """Reset events for the H1 embodiment."""

    # On reset, write the default joint positions from the ArticulationCfg
    reset_robot_joints: EventTerm = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ========================================================================== #
# Embodiment class                                                           #
# ========================================================================== #


@register_asset
class H1VlnEmbodiment(EmbodimentBase):
    """H1 humanoid embodiment for Vision-Language Navigation.

    Provides:
      - Full-body joint-position action space.
      - Proprioceptive observations matching the NaVILA low-level policy.
      - An optional pelvis-mounted RGB camera for VLM input.
      - Velocity command generator (values injected by VLNEnvWrapper).
    """

    name = "h1_vln"

    def __init__(
        self,
        enable_cameras: bool = True,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_H1_CAMERA_OFFSET,
        use_tiled_camera: bool = False,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)

        self.scene_config = H1VlnSceneCfg()
        self.camera_config = H1VlnCameraCfg()
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset

        self.action_config = H1VlnActionCfg()
        self.observation_config = H1VlnObservationsCfg()
        self.event_config = H1VlnEventCfg()
        self.command_config = H1VlnCommandsCfg()
