#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Map CNN + state MLP conditioned DDPM action chunker (Diffusion Policy)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.policy.helper.conditional_unet1d import ConditionalUnet1D
from utils.policy.helper.ddpm_scheduler import build_ddpm_scheduler


@dataclass
class DpModelConfig:
    """UNet / DDPM hyper-parameters (ManiSkill Diffusion Policy defaults)."""

    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    num_diffusion_iters: int = 50
    kernel_size: int = 5


class DpModel(nn.Module):
    """Predict ``pred_horizon`` actions via conditional DDPM from map + state."""

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        cfg: DpModelConfig | None = None,
    ):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg or DpModelConfig()

        # Same map / state towers as BC / ACT; fused cond FiLM-conditions the UNet.
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

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=self.cfg.diffusion_step_embed_dim,
            down_dims=self.cfg.unet_dims,
            kernel_size=self.cfg.kernel_size,
            n_groups=self.cfg.n_groups,
        )
        self.num_diffusion_iters = int(self.cfg.num_diffusion_iters)
        self.noise_scheduler = build_ddpm_scheduler(self.num_diffusion_iters)

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

    def predict_noise(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        maps: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon for ``noisy_actions`` at ``timesteps``."""
        cond = self.encode_cond(maps, state)
        return self.noise_pred_net(
            noisy_actions, timesteps, global_cond=cond)

    @torch.no_grad()
    def sample(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Reverse DDPM sampling → ``(B, pred_horizon, action_dim)``."""
        batch_size = maps.shape[0] if maps.ndim >= 3 else state.shape[0]
        device = state.device
        cond = self.encode_cond(maps, state)
        noisy_action_seq = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=device,
        )
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=noisy_action_seq,
                timestep=k,
                global_cond=cond,
            )
            noisy_action_seq = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action_seq,
            ).prev_sample
        return noisy_action_seq
