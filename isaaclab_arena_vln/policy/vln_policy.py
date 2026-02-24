# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Composite VLN policy that integrates VLM + low-level locomotion.

This policy is a ``ClientSidePolicy`` that can be used directly with
Arena's ``policy_runner.py``.  Internally it:

  1. Queries a remote VLM server for velocity commands (via ZeroMQ).
  2. Injects the velocity command into the proprioceptive observation.
  3. Runs a pre-trained RSL-RL locomotion policy to produce joint actions.
  4. Returns joint actions to the environment.

Usage with Arena policy_runner::

    python -m isaaclab_arena.evaluation.policy_runner \\
        --policy_type isaaclab_arena_vln.policy.vln_policy.VlnPolicy \\
        --remote_host localhost --remote_port 5555 \\
        --ll_checkpoint_path /path/to/rsl_rl/model.pt \\
        --ll_agent_cfg /path/to/agent.yaml \\
        --num_episodes 10 \\
        VLN_Benchmark \\
        --usd_path /path/to/scene.usd \\
        --r2r_dataset_path /path/to/dataset.json.gz
"""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.client_side_policy import ClientSidePolicy
from isaaclab_arena.remote_policy.action_protocol import VlnVelocityActionProtocol
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class VlnPolicy(ClientSidePolicy):
    """Composite VLN policy: remote VLM + local RSL-RL locomotion.

    Inherits from ``ClientSidePolicy`` to reuse the ZeroMQ handshake,
    observation packing, and remote lifecycle management.  Overrides
    ``get_action()`` to add the low-level locomotion policy layer.
    """

    def __init__(
        self,
        remote_config: RemotePolicyConfig,
        ll_checkpoint_path: str,
        ll_agent_cfg: str,
        device: str = "cuda",
        vel_cmd_obs_indices: tuple[int, int] = (6, 9),
        warmup_steps: int = 200,
    ):
        super().__init__(
            config=None,
            remote_config=remote_config,
            protocol_cls=VlnVelocityActionProtocol,
        )
        self._device = device
        self._vel_cmd_indices = vel_cmd_obs_indices
        self._warmup_steps = warmup_steps

        # RSL-RL low-level policy (loaded lazily in first get_action)
        self._ll_checkpoint_path = ll_checkpoint_path
        self._ll_agent_cfg = ll_agent_cfg
        self._ll_policy = None
        self._ll_obs: torch.Tensor | None = None
        self._ll_vec_env = None

        # VLM scheduling state
        self._step_count: int = 0
        self._target_step: int = 0
        self._last_vel_cmd = np.zeros(self.action_dim, dtype=np.float32)
        self._env_dt: float | None = None

        # Track current instruction to detect episode changes
        self._current_instruction: str | None = None

    # ------------------------------------------------------------------ #
    # PolicyBase interface                                                #
    # ------------------------------------------------------------------ #

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Return joint-position actions for the environment.

        Internally: VLM query → velocity command → inject into obs →
        RSL-RL forward pass → joint actions.
        """
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        # Lazy init
        if self._ll_policy is None:
            self._load_low_level_policy(env)
        if self._env_dt is None:
            try:
                self._env_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
            except Exception:
                self._env_dt = 0.02

        # Detect per-episode instruction changes from env.extras
        self._check_instruction_update(unwrapped)

        # Query VLM if scheduling says it's time
        if self._step_count >= self._target_step:
            packed_obs = self.pack_observation_for_server(observation)
            resp = self.remote_client.get_action(observation=packed_obs)

            vel_cmd = np.asarray(
                resp.get("action", np.zeros(self.action_dim)), dtype=np.float32
            )
            duration = float(resp.get("duration", self.protocol.default_duration))
            self._last_vel_cmd = vel_cmd

            if self._env_dt > 0.0 and duration > 0.0:
                steps_to_hold = max(1, int(duration / self._env_dt))
            else:
                steps_to_hold = 1
            self._target_step = self._step_count + steps_to_hold

            # STOP: VLM returns zero velocity + zero duration
            if np.allclose(vel_cmd, 0.0) and duration <= 0.0:
                extras = getattr(unwrapped, "extras", {})
                if "vln_stop_called" in extras:
                    extras["vln_stop_called"][:] = True

        # Get the latest proprioceptive observation from the RSL-RL wrapper.
        # After policy_runner calls env.step(), the underlying ManagerBasedEnv
        # updates its observation buffers.  RslRlVecEnvWrapper reads from
        # those same buffers, so get_observations() returns fresh data.
        self._ll_obs, _ = self._ll_vec_env.get_observations()

        # Inject velocity command into the proprioceptive observation
        i, j = self._vel_cmd_indices
        cmd_tensor = torch.tensor(self._last_vel_cmd, device=self._device, dtype=torch.float32)
        self._ll_obs[:, i:j] = cmd_tensor

        # Run low-level policy → joint actions
        with torch.inference_mode():
            joint_actions = self._ll_policy(self._ll_obs)

        self._step_count += 1
        return joint_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset VLM scheduling state and notify server."""
        super().reset(env_ids)
        self._step_count = 0
        self._target_step = 0
        self._last_vel_cmd[:] = 0.0
        self._current_instruction = None

    def set_task_description(self, task_description: str | None) -> str:
        """Forward task description to the VLM server."""
        self.task_description = task_description
        if task_description is not None:
            self.remote_client.call_endpoint(
                "set_task_description",
                data={"task_description": task_description},
                requires_input=True,
            )
        return self.task_description or ""

    # ------------------------------------------------------------------ #
    # Low-level policy loading                                            #
    # ------------------------------------------------------------------ #

    def _load_low_level_policy(self, env) -> None:
        """Load the RSL-RL locomotion policy and do warmup."""
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import load_yaml

        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        except ImportError:
            from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        if isinstance(env, RslRlVecEnvWrapper):
            vec_env = env
        else:
            vec_env = RslRlVecEnvWrapper(unwrapped)
        self._ll_vec_env = vec_env

        agent_cfg_dict = load_yaml(self._ll_agent_cfg)
        device = agent_cfg_dict.get("device", "cuda")

        runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=device)
        runner.load(self._ll_checkpoint_path)
        self._ll_policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        # Warmup
        obs, _ = vec_env.get_observations()
        self._ll_obs = obs
        zero_cmd = torch.zeros(self.action_dim, device=self._device)
        i, j = self._vel_cmd_indices

        print(f"[VlnPolicy] Warming up ({self._warmup_steps} steps)...")
        for step in range(self._warmup_steps):
            self._ll_obs[:, i:j] = zero_cmd
            with torch.inference_mode():
                actions = self._ll_policy(self._ll_obs)
            self._ll_obs, _, _, _ = vec_env.step(actions)
        print("[VlnPolicy] Warmup complete.")

    # ------------------------------------------------------------------ #
    # Instruction tracking                                                #
    # ------------------------------------------------------------------ #

    def _check_instruction_update(self, unwrapped) -> None:
        """Detect per-episode instruction changes from env.extras."""
        extras = getattr(unwrapped, "extras", {})
        instruction = extras.get("current_instruction")
        if instruction is None:
            return
        if isinstance(instruction, list):
            instruction = instruction[0]

        if instruction != self._current_instruction:
            self._current_instruction = instruction
            self.remote_client.call_endpoint(
                "set_task_description",
                data={"task_description": instruction},
                requires_input=True,
            )
            self._step_count = 0
            self._target_step = 0
            self._last_vel_cmd[:] = 0.0

    # ------------------------------------------------------------------ #
    # CLI helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = ClientSidePolicy.add_remote_args_to_parser(parser)

        ll = parser.add_argument_group("Low-Level Locomotion Policy")
        ll.add_argument(
            "--ll_checkpoint_path", type=str, required=True,
            help="Path to the RSL-RL checkpoint (e.g. model_0.pt).",
        )
        ll.add_argument(
            "--ll_agent_cfg", type=str, required=True,
            help="Path to the RSL-RL agent config YAML.",
        )
        ll.add_argument(
            "--warmup_steps", type=int, default=200,
            help="Low-level policy warmup steps (default: 200).",
        )
        ll.add_argument(
            "--policy_device", type=str, default="cuda",
            help="Device for policy inference (default: cuda).",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> VlnPolicy:
        remote_config = ClientSidePolicy.build_remote_config_from_args(args)
        return VlnPolicy(
            remote_config=remote_config,
            ll_checkpoint_path=args.ll_checkpoint_path,
            ll_agent_cfg=args.ll_agent_cfg,
            device=getattr(args, "policy_device", "cuda"),
            warmup_steps=getattr(args, "warmup_steps", 200),
        )
