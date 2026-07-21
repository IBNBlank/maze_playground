#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""CNN + MLP regressor for maze behavior cloning."""

from __future__ import annotations

import torch
import torch.nn as nn


class BcModel(nn.Module):
    """Predict ``pred_horizon`` pixel ``(dx, dy)`` actions from map + state."""

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

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
            # (B, 64, 32, 32) -> (B, 64, 4, 4)
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_horizon * self.state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.pred_horizon * self.action_dim),
            nn.Tanh(),
        )

    @staticmethod
    def _normalize_map(maps: torch.Tensor) -> torch.Tensor:
        """Accept ``(B, H, W)`` or ``(B, 1, H, W)`` -> ``(B, 1, H, W)``."""
        if maps.ndim == 3:
            return maps.unsqueeze(1)
        if maps.ndim == 4:
            return maps
        raise ValueError(
            f"map tensor must be 3D or 4D, got shape={tuple(maps.shape)}")

    def forward(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        maps = self._normalize_map(maps)
        batch_size = maps.shape[0]
        map_feat = self.map_encoder(maps)
        state_feat = self.state_encoder(state.reshape(batch_size, -1))
        pred = self.head(torch.cat([map_feat, state_feat], dim=-1))
        return pred.view(batch_size, self.pred_horizon, self.action_dim)
