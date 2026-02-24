# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLN benchmark environment builder.

This module defines the ``VLNBenchmarkEnvironment`` that integrates:
  - A Matterport 3D background scene.
  - The H1 humanoid embodiment configured for VLN.
  - The VLN navigation task with R2R episode management.

It follows the ``ExampleEnvironmentBase`` pattern used by all IsaacLab Arena
environments, so it plugs into the CLI and ``ArenaEnvBuilder`` seamlessly.

Usage (CLI)::

    python -m isaaclab_arena.evaluation.policy_runner \\
        --policy_type isaaclab_arena_vln.policy.vln_client_side_policy.VlnClientSidePolicy \\
        --remote_host localhost --remote_port 5555 \\
        --num_episodes 10 \\
        VLN_Benchmark \\
        --usd_path /path/to/matterport.usd \\
        --r2r_dataset_path /path/to/vln_ce_isaac_v1.json.gz
"""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class VLNBenchmarkEnvironment(ExampleEnvironmentBase):
    """IsaacLab Arena environment for VLN benchmarking."""

    name: str = "VLN_Benchmark"

    def get_env(self, args_cli: argparse.Namespace):
        """Build and return the VLN environment.

        Multi-env note:
            When ``num_envs > 1``, each env gets a full copy of the Matterport
            scene.  The ``env_spacing`` should be large enough that scenes don't
            overlap (Matterport houses are typically 20-50m wide).  The default
            is overridden to 100m if the user hasn't set it explicitly.
        """
        # Matterport scenes are large — override default env_spacing if user
        # hasn't set a custom value (the Arena default is 30m, too small).
        if not hasattr(args_cli, "_env_spacing_set_by_user"):
            if getattr(args_cli, "num_envs", 1) > 1 and args_cli.env_spacing < 100.0:
                print(
                    f"[VLN] Overriding env_spacing from {args_cli.env_spacing}m to 100m "
                    f"for Matterport scenes (num_envs={args_cli.num_envs})."
                )
                args_cli.env_spacing = 100.0

        # Delayed imports — require simulation app to be running
        import isaaclab.sim as sim_utils
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        from isaaclab_arena_vln.assets.matterport_background import MatterportBackground
        from isaaclab_arena_vln.embodiments.h1_vln import H1VlnEmbodiment
        from isaaclab_arena_vln.tasks.vln_task import VlnNavTask

        # 1) Background: Matterport 3D scene
        background = MatterportBackground(usd_path=args_cli.usd_path)

        # 2) Embodiment: H1 humanoid with camera
        use_tiled = getattr(args_cli, "use_tiled_camera", False)
        embodiment = H1VlnEmbodiment(
            enable_cameras=True,
            use_tiled_camera=use_tiled,
        )

        # 3) Task: VLN navigation with R2R episodes
        episode_indices = None
        if hasattr(args_cli, "episode_start") and args_cli.episode_start is not None:
            end = getattr(args_cli, "episode_end", args_cli.episode_start + 1)
            episode_indices = list(range(args_cli.episode_start, end))

        task = VlnNavTask(
            robot=embodiment,
            r2r_dataset_path=args_cli.r2r_dataset_path,
            episode_indices=episode_indices,
            episode_length_s=getattr(args_cli, "episode_length_s", 60.0),
            success_radius=getattr(args_cli, "success_radius", 3.0),
        )

        # 4) Scene: Matterport background
        scene = Scene(assets=[background])

        # 5) Simulation parameters callback
        # These MUST match the low-level locomotion policy training config.
        # Default values come from NaVILA-Bench:
        #   h1_matterport_base_cfg.py -> H1MatterportBaseCfg.__post_init__()
        sim_dt = getattr(args_cli, "sim_dt", 0.005)
        decimation = getattr(args_cli, "sim_decimation", 4)

        def vln_sim_cfg_callback(env_cfg):
            env_cfg.sim.dt = sim_dt                # 200 Hz physics
            env_cfg.decimation = decimation         # 50 Hz control (200/4)
            env_cfg.sim.render_interval = decimation
            env_cfg.sim.disable_contact_processing = True
            env_cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                friction_combine_mode="max",
                restitution_combine_mode="max",
            )
            return env_cfg

        # 6) Compose the Arena environment
        arena_env = IsaacLabArenaEnvironment(
            name=self.name,
            scene=scene,
            embodiment=embodiment,
            task=task,
            env_cfg_callback=vln_sim_cfg_callback,
        )
        return arena_env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add VLN-specific CLI arguments."""
        group = parser.add_argument_group("VLN Benchmark", "VLN benchmark environment arguments")
        group.add_argument(
            "--usd_path",
            type=str,
            required=True,
            help="Path to the Matterport USD scene file.",
        )
        group.add_argument(
            "--r2r_dataset_path",
            type=str,
            required=True,
            help="Path to the R2R VLN dataset (e.g. vln_ce_isaac_v1.json.gz).",
        )
        group.add_argument(
            "--episode_start",
            type=int,
            default=None,
            help="Starting episode index (inclusive).  If None, use all episodes.",
        )
        group.add_argument(
            "--episode_end",
            type=int,
            default=None,
            help="Ending episode index (exclusive).  Used with --episode_start.",
        )
        group.add_argument(
            "--episode_length_s",
            type=float,
            default=60.0,
            help="Maximum episode duration in seconds (default: 60).",
        )
        group.add_argument(
            "--success_radius",
            type=float,
            default=3.0,
            help="Distance threshold for goal success (default: 3.0m).",
        )
        group.add_argument(
            "--use_tiled_camera",
            action="store_true",
            default=False,
            help="Use TiledCamera for parallel evaluation (default: False).",
        )

        # Simulation parameters — must match the low-level policy training config
        sim_group = parser.add_argument_group(
            "Simulation", "Physics simulation parameters (must match low-level policy training)"
        )
        sim_group.add_argument(
            "--sim_dt",
            type=float,
            default=0.005,
            help="Physics simulation timestep in seconds (default: 0.005 = 200Hz).",
        )
        sim_group.add_argument(
            "--sim_decimation",
            type=int,
            default=4,
            help="Number of physics steps per policy step (default: 4, giving 50Hz control).",
        )

        # Low-level policy arguments (for VLNEnvWrapper)
        ll_group = parser.add_argument_group(
            "Low-Level Policy", "Arguments for the NaVILA low-level locomotion policy"
        )
        ll_group.add_argument(
            "--ll_log_root_path",
            type=str,
            default=None,
            help="Root directory of RSL-RL training logs for the low-level policy.",
        )
        ll_group.add_argument(
            "--ll_agent_cfg_yaml",
            type=str,
            default=None,
            help="Path to the agent.yaml used during low-level policy training.",
        )
        ll_group.add_argument(
            "--ll_policy_run_name",
            type=str,
            default=None,
            help="Run folder name under the RSL-RL log root.",
        )
        ll_group.add_argument(
            "--ll_policy_checkpoint_id",
            type=int,
            default=0,
            help="Checkpoint index for the low-level policy (default: 0).",
        )
