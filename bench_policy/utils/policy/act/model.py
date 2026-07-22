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
from utils.policy.helper.obs_encoder import ObsEncoder


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

        self.obs_encoder = ObsEncoder(self.obs_horizon, self.state_dim)
        transformer = build_transformer(self.cfg)
        cvae_encoder = build_encoder(self.cfg)
        self.detr_vae = DETRVAE(
            transformer,
            cvae_encoder,
            state_dim=self.obs_encoder.cond_dim,
            action_dim=self.action_dim,
            num_queries=self.pred_horizon,
        )

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
        return self.detr_vae(cond, actions)
