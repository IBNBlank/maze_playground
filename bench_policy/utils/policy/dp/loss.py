#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Diffusion Policy losses: epsilon-prediction MSE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dp_noise_mse_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and sampled noise over the action chunk."""
    return F.mse_loss(noise_pred, noise)
