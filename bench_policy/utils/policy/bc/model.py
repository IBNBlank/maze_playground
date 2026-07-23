#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Map CNN + ConditionalUnet1D action chunker (learnable query; no CVAE)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.policy.helper.conditional_unet1d import ConditionalUnet1D
from utils.policy.helper.obs_encoder import ObsEncoder


@dataclass
class BcModelConfig:
    """UNet hyper-parameters (aligned with DP / FM backbone defaults)."""

    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    kernel_size: int = 5


class BcModel(nn.Module):
    """Predict ``pred_horizon`` actions via FiLM-conditioned UNet.

    UNet input is a shared learnable action query; global condition is the
    ``ObsEncoder`` output. Timestep is fixed to 0 (no diffusion).
    """

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        cfg: BcModelConfig | None = None,
    ):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg or BcModelConfig()

        self.obs_encoder = ObsEncoder(self.obs_horizon, self.state_dim)
        self.action_query = nn.Parameter(
            torch.zeros(1, self.pred_horizon, self.action_dim), )
        nn.init.normal_(self.action_query, std=0.02)
        self.action_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_encoder.cond_dim,
            diffusion_step_embed_dim=self.cfg.diffusion_step_embed_dim,
            down_dims=self.cfg.unet_dims,
            kernel_size=self.cfg.kernel_size,
            n_groups=self.cfg.n_groups,
        )

    def forward(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        cond = self.obs_encoder(maps, state)
        batch_size = cond.shape[0]
        sample = self.action_query.expand(batch_size, -1, -1)
        timesteps = torch.zeros(batch_size,
                                dtype=torch.long,
                                device=cond.device)
        return self.action_net(sample, timesteps, global_cond=cond)
