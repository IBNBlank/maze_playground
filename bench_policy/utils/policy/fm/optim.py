#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Optimizers for Flow Matching Policy."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim


def build_fm_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    """AdamW aligned with Diffusion Policy defaults for fair comparison."""
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )
