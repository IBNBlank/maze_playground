#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-22
################################################################
"""Shared optimizers for maze IL policies."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim


def build_adamw_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    """AdamW used by both BC and ACT (transformer-friendly, aligned)."""
    return optim.AdamW(model.parameters(), lr=lr)
