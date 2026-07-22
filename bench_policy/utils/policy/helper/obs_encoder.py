#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Shared map CNN + state MLP condition encoder for BC / ACT / DP / FM."""

from __future__ import annotations

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """Encode occupancy map + proprio state into a fused condition vector.

    Map tower (256×256 input typical)::

        (B,1,256,256) --s2→ (B,16,128,128)
                  --s2→ (B,32,64,64)
                  --s2→ (B,64,32,32)
                  --s2→ (B,64,16,16)
                  pool→ (B,64,8,8) → flatten 4096

    State tower: ``obs_horizon * state_dim → 256 → 256``.
    """

    MAP_CHANNELS: int = 64
    MAP_POOL: tuple[int, int] = (8, 8)
    STATE_FEAT_DIM: int = 256

    def __init__(self, obs_horizon: int, state_dim: int):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.state_dim = int(state_dim)

        self.map_encoder = nn.Sequential(
            # (B, 1, 256, 256) -> (B, 16, 128, 128)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # (B, 16, 128, 128) -> (B, 32, 64, 64)
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # (B, 32, 64, 64) -> (B, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # (B, 64, 32, 32) -> (B, 64, 16, 16)
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # (B, 64, 16, 16) -> (B, 64, 8, 8)
            nn.AdaptiveAvgPool2d(self.MAP_POOL),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_horizon * self.state_dim, self.STATE_FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.STATE_FEAT_DIM, self.STATE_FEAT_DIM),
            nn.ReLU(inplace=True),
        )

    @property
    def map_feat_dim(self) -> int:
        h, w = self.MAP_POOL
        return self.MAP_CHANNELS * h * w

    @property
    def cond_dim(self) -> int:
        return self.map_feat_dim + self.STATE_FEAT_DIM

    @staticmethod
    def normalize_map(maps: torch.Tensor) -> torch.Tensor:
        """Accept ``(B, H, W)`` or ``(B, 1, H, W)`` → ``(B, 1, H, W)``."""
        if maps.ndim == 3:
            return maps.unsqueeze(1)
        if maps.ndim == 4:
            return maps
        raise ValueError(
            f"map tensor must be 3D or 4D, got shape={tuple(maps.shape)}")

    def forward(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return fused cond ``(B, cond_dim)`` from map + state."""
        maps = self.normalize_map(maps)
        batch_size = maps.shape[0]
        map_feat = self.map_encoder(maps)
        state_feat = self.state_encoder(state.reshape(batch_size, -1))
        return torch.cat([map_feat, state_feat], dim=-1)
