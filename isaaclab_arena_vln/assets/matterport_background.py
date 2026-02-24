# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Matterport 3D scene background for VLN tasks.

Follows the same pattern as ``LightwheelKitchenBackground`` in
``background_library.py`` — the USD path is determined at runtime
(from a CLI argument) and set in ``__init__`` before calling super.
"""

from __future__ import annotations

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class MatterportBackground(LibraryBackground):
    """A Matterport 3D scene loaded from a USD file.

    Usage::

        bg = MatterportBackground(usd_path="/path/to/scene.usd")
        scene = Scene(assets=[bg])
    """

    name = "matterport"
    # Tags for AssetRegistry.get_assets_by_tag() queries.
    # "background" — standard category, same as kitchen/galileo/table.
    # NOTE: If you want to query VLN-specific backgrounds separately in the
    # future (e.g. asset_registry.get_assets_by_tag("vln")), you can add
    # extra tags like "vln" or "matterport" here.
    tags = ["background"]
    usd_path = None  # Set at runtime in __init__
    initial_pose = Pose.identity()
    object_min_z = -0.5

    def __init__(self, usd_path: str):
        self.usd_path = usd_path
        super().__init__()
