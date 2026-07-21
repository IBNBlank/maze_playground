#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    # device
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, set cudnn.deterministic=True and cudnn.benchmark=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # algo / data
    algo: str = "bc"
    """policy algorithm: bc | act | dp | fm"""
    state_dim: int = 4
    """state dimension: [x, y, goal_x, goal_y] in normalized coords"""
    action_dim: int = 2
    """action dimension: [dx, dy] in pixels"""
    dataset_name: str = "genplan256_mix"
    """subdir under ../datasets/"""

    # train
    epochs: int = 100
    """number of training epochs (= number of idx/epoch_*.npy perms to use)"""
    batch_size: int = 64
    """samples per training batch (last batch of an epoch may be shorter)"""
    lr: float = 3e-4
    """learning rate"""
    log_freq: int = 50
    """tensorboard loss log frequency in steps (0 = epoch end only)"""

    # eval
    eval_freq: int = 5
    """evaluation frequency in epochs (0 = only final)"""
    num_eval: int = 100
    """number of episodes to evaluate"""
    goal_tol: float = 1.0
    """pixel L2 distance threshold for success (error < goal_tol)"""


@dataclass
class EvalArgs:
    # device
    seed: int = 42
    """training seed used to resolve runs/{run_name}/"""
    torch_deterministic: bool = True
    """if toggled, set cudnn.deterministic=True and cudnn.benchmark=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # path
    ckpt_name: str = "best_success_ckpt.pt"
    """checkpoint filename under runs/{run_name}/"""

    # algo / data
    algo: str = "bc"
    """policy algorithm: bc | act | dp | fm"""
    state_dim: int = 4
    """state dimension: [x, y, goal_x, goal_y] in normalized coords"""
    action_dim: int = 2
    """action dimension: [dx, dy] in pixels"""
    dataset_name: str = "genplan256_mix"
    """subdir under ../datasets/"""

    # eval knobs
    num_eval: int = 100
    """number of episodes to evaluate"""
    goal_tol: float = 1.0
    """pixel L2 distance threshold for success (error < goal_tol)"""
    capture_preview: bool = False
    """save a small collage of rollout overlays under runs/{run_name}/"""
