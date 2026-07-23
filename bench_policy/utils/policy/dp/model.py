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

from utils.policy.dp.ddim_scheduler import build_ddim_scheduler
from utils.policy.dp.ddpm_scheduler import build_ddpm_scheduler
from utils.policy.helper.conditional_unet1d import ConditionalUnet1D
from utils.policy.helper.obs_encoder import ObsEncoder


@dataclass
class DpModelConfig:
    """UNet / DDPM hyper-parameters (ManiSkill Diffusion Policy defaults)."""

    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    num_diffusion_iters: int = 100
    num_inference_iters: int = 20
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

        self.obs_encoder = ObsEncoder(self.obs_horizon, self.state_dim)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_encoder.cond_dim,
            diffusion_step_embed_dim=self.cfg.diffusion_step_embed_dim,
            down_dims=self.cfg.unet_dims,
            kernel_size=self.cfg.kernel_size,
            n_groups=self.cfg.n_groups,
        )
        self.num_diffusion_iters = int(self.cfg.num_diffusion_iters)
        self.num_inference_iters = int(self.cfg.num_inference_iters)
        # Train: full DDPM schedule; infer: shorter DDIM chain.
        self.noise_scheduler = build_ddpm_scheduler(self.num_diffusion_iters)
        self.infer_scheduler = build_ddim_scheduler(
            num_train_timesteps=self.num_diffusion_iters,
            num_inference_steps=self.num_inference_iters,
        )

    def predict_noise(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        maps: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon for ``noisy_actions`` at ``timesteps``."""
        cond = self.obs_encoder(maps, state)
        return self.noise_pred_net(
            noisy_actions, timesteps, global_cond=cond)

    @torch.no_grad()
    def sample(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Reverse DDIM sampling → ``(B, pred_horizon, action_dim)``."""
        batch_size = maps.shape[0] if maps.ndim >= 3 else state.shape[0]
        device = state.device
        cond = self.obs_encoder(maps, state)
        noisy_action_seq = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=device,
        )
        for k in self.infer_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=noisy_action_seq,
                timestep=k,
                global_cond=cond,
            )
            noisy_action_seq = self.infer_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action_seq,
            ).prev_sample
        return noisy_action_seq
