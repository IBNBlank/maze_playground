#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Map CNN + state MLP conditioned flow-matching action chunker."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.policy.fm.fm_scheduler import build_fm_scheduler
from utils.policy.helper.conditional_unet1d import ConditionalUnet1D
from utils.policy.helper.obs_encoder import ObsEncoder


@dataclass
class FmModelConfig:
    """UNet / FM hyper-parameters (same backbone as DP; fewer ODE steps)."""

    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    num_inference_steps: int = 20
    kernel_size: int = 5


class FmModel(nn.Module):
    """Predict ``pred_horizon`` actions via conditional flow matching."""

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        cfg: FmModelConfig | None = None,
    ):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg or FmModelConfig()

        self.obs_encoder = ObsEncoder(self.obs_horizon, self.state_dim)
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_encoder.cond_dim,
            diffusion_step_embed_dim=self.cfg.diffusion_step_embed_dim,
            down_dims=self.cfg.unet_dims,
            kernel_size=self.cfg.kernel_size,
            n_groups=self.cfg.n_groups,
        )
        self.num_inference_steps = int(self.cfg.num_inference_steps)
        self.fm_scheduler = build_fm_scheduler(self.num_inference_steps)

    def predict_velocity(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        maps: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity ``v̂`` for ``x_t`` at continuous ``t ∈ [0,1]``."""
        cond = self.obs_encoder(maps, state)
        t_embed = self.fm_scheduler.time_for_net(t).to(device=xt.device)
        return self.velocity_net(xt, t_embed, global_cond=cond)

    @torch.no_grad()
    def sample(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Euler ODE from noise (t=0) → action (t=1)."""
        batch_size = maps.shape[0] if maps.ndim >= 3 else state.shape[0]
        device = state.device
        cond = self.obs_encoder(maps, state)
        x = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=device,
        )
        for t in self.fm_scheduler.timesteps:
            t_batch = t.to(device=device).expand(batch_size)
            t_embed = self.fm_scheduler.time_for_net(t_batch)
            v_pred = self.velocity_net(
                sample=x,
                timestep=t_embed,
                global_cond=cond,
            )
            x = self.fm_scheduler.step(
                model_output=v_pred,
                timestep=t,
                sample=x,
            ).prev_sample
        return x
