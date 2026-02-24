# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab Arena VLN Benchmark extension.

This package provides Vision-Language Navigation (VLN) benchmark support for
IsaacLab Arena, including:
  - VLN task with R2R-style episode management
  - VLN metrics (SPL, Success, PathLength, DistanceToGoal)
  - H1 humanoid embodiment for navigation
  - Matterport 3D scene background
  - Remote VLM policy (server + client) via the Arena remote-policy framework
  - Low-level locomotion policy wrapper (RSL-RL based)
"""
