#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Map CNN + DETR-VAE action chunker for maze ACT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.policy.helper.detr.detr_vae import DETRVAE, build_encoder
from utils.policy.helper.detr.transformer import build_transformer


@dataclass
class ActModelConfig:
    """Transformer / CVAE hyper-parameters (ManiSkill ACT defaults)."""

    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 4
    dim_feedforward: int = 512
    enc_layers: int = 2
    dec_layers: int = 4
    pre_norm: bool = False


class ActModel(nn.Module):
    """Predict ``pred_horizon`` actions via CVAE transformer from map + state."""

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        cfg: ActModelConfig | None = None,
    ):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg or ActModelConfig()

        # Same map / state towers as BC; fused cond feeds DETRVAE.
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_horizon * self.state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        cond_dim = 64 * 4 * 4 + 128

        transformer = build_transformer(self.cfg)
        cvae_encoder = build_encoder(self.cfg)
        self.detr_vae = DETRVAE(
            transformer,
            cvae_encoder,
            state_dim=cond_dim,
            action_dim=self.action_dim,
            num_queries=self.pred_horizon,
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

    def encode_cond(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        maps = self._normalize_map(maps)
        batch_size = maps.shape[0]
        map_feat = self.map_encoder(maps)
        state_feat = self.state_encoder(state.reshape(batch_size, -1))
        return torch.cat([map_feat, state_feat], dim=-1)

    def forward(
        self,
        maps: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            maps: ``(B, H, W)`` or ``(B, 1, H, W)``
            state: ``(B, obs_horizon, state_dim)``
            actions: optional ``(B, pred_horizon, action_dim)`` for CVAE train

        Returns:
            a_hat ``(B, pred_horizon, action_dim)``, ``[mu, logvar]``
        """
        cond = self.encode_cond(maps, state)
        return self.detr_vae(cond, actions)
