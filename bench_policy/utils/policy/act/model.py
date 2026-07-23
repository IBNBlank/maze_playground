#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Map CNN + MLP-CVAE latent + ConditionalUnet1D action chunker for maze ACT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.policy.helper.conditional_unet1d import ConditionalUnet1D
from utils.policy.helper.obs_encoder import ObsEncoder


@dataclass
class ActModelConfig:
    """UNet + MLP-CVAE hyper-parameters (UNet aligned with DP / FM / BC)."""

    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    kernel_size: int = 5
    latent_dim: int = 32
    cvae_hidden_dim: int = 256


class ActModel(nn.Module):
    """Predict ``pred_horizon`` actions via CVAE-conditioned UNet.

    Same UNet layout as BC: learnable action query + ``ObsEncoder`` as
    ``global_cond``. CVAE ``z`` is projected into the diffusion-step
    embedding slot (BC keeps that slot at ``t=0``).

    Train: run CVAE on ``(cond, actions)``, use ``μ`` as ``z`` (no
    reparam sampling). Infer: ``z = 0``.
    """

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
        self.latent_dim = int(self.cfg.latent_dim)
        dsed = int(self.cfg.diffusion_step_embed_dim)
        cvae_hidden = int(self.cfg.cvae_hidden_dim)

        self.obs_encoder = ObsEncoder(self.obs_horizon, self.state_dim)
        self.action_query = nn.Parameter(
            torch.zeros(1, self.pred_horizon, self.action_dim),
        )
        nn.init.normal_(self.action_query, std=0.02)

        # MLP-CVAE: cat(cond, flat actions) → (μ, logσ²).
        cvae_in = self.obs_encoder.cond_dim + self.pred_horizon * self.action_dim
        self.latent_encoder = nn.Sequential(
            nn.Linear(cvae_in, cvae_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cvae_hidden, cvae_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cvae_hidden, self.latent_dim * 2),
        )

        # z → step channel (same width as diffusion_step_embed_dim).
        self.latent_to_step = nn.Sequential(
            nn.Linear(self.latent_dim, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # Same global_cond_dim as BC (obs only; z goes through step_embed).
        self.action_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_encoder.cond_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=self.cfg.unet_dims,
            kernel_size=self.cfg.kernel_size,
            n_groups=self.cfg.n_groups,
        )

    def _encode_latent(
        self,
        cond: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``(cond, actions)`` → ``(mu, logvar)``."""
        flat_actions = actions.reshape(actions.shape[0], -1)
        latent_info = self.latent_encoder(torch.cat([cond, flat_actions], dim=-1))
        mu = latent_info[:, :self.latent_dim]
        logvar = latent_info[:, self.latent_dim:]
        return mu, logvar

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
        cond = self.obs_encoder(maps, state)
        batch_size = cond.shape[0]

        if actions is not None:
            mu, logvar = self._encode_latent(cond, actions)
            z = mu
        else:
            mu = logvar = None
            z = torch.zeros(
                batch_size, self.latent_dim, dtype=torch.float32, device=cond.device)

        sample = self.action_query.expand(batch_size, -1, -1)
        step_embed = self.latent_to_step(z)
        a_hat = self.action_net(
            sample,
            timestep=0,
            global_cond=cond,
            step_embed=step_embed,
        )
        return a_hat, [mu, logvar]
