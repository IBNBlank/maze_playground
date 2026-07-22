#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""ACT losses: MSE reconstruction + KL on the CVAE latent."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Batch-mean total KL; matches ManiSkill ACT ``total_kld[0]``."""
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean()


def act_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 5.0,
) -> torch.Tensor:
    """``MSE(pred, target) + kl_weight * KL(mu, logvar)``."""
    mse = F.mse_loss(pred, target)
    kld = kl_divergence(mu, logvar)
    return mse + float(kl_weight) * kld
