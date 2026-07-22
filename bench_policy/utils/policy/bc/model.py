#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Map CNN + DETR action chunker (latent z fixed to 0; no CVAE)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.policy.helper.detr.detr_vae import DETRVAE
from utils.policy.helper.detr.transformer import build_transformer
from utils.policy.helper.obs_encoder import ObsEncoder


@dataclass
class BcModelConfig:
    """Match ACT transformer defaults so BC vs ACT isolates the CVAE."""

    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 4
    dim_feedforward: int = 512
    enc_layers: int = 2
    dec_layers: int = 4
    pre_norm: bool = False


class BcModel(nn.Module):
    """Same towers + DETRVAE decoder as ACT; ``encoder=None`` → always ``z=0``."""

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
        transformer = build_transformer(self.cfg)
        self.detr_vae = DETRVAE(
            transformer,
            encoder=None,
            state_dim=self.obs_encoder.cond_dim,
            action_dim=self.action_dim,
            num_queries=self.pred_horizon,
        )

    def forward(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        cond = self.obs_encoder(maps, state)
        a_hat, _ = self.detr_vae(cond, actions=None)
        return a_hat
