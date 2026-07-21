#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

from dataclasses import dataclass


@dataclass
class _SharedArgs:
    # device
    seed: int = 42
    """seed of the experiment / run_name resolution"""
    torch_deterministic: bool = True
    """if toggled, set cudnn.deterministic=True and cudnn.benchmark=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # algo / data
    algo: str = "bc"
    """policy algorithm: bc | act | dp | fm"""
    dataset_name: str = "genplan256_mix"
    """subdir under ../datasets/"""

    # eval knobs shared by train mid-eval and eval.py
    num_eval: int = 100
    """number of episodes to evaluate"""
    goal_tol: float = 1.0
    """pixel L2 distance threshold for success (error < goal_tol)"""


@dataclass
class TrainArgs(_SharedArgs):
    # train
    epochs: int = 100
    """number of training epochs (= number of idx/epoch_*.npy perms to use)"""
    batch_size: int = 256
    """samples per training batch (last batch of an epoch may be shorter)"""
    lr: float = 3e-4
    """learning rate"""

    # eval
    eval_freq: int = 5
    """evaluation frequency in epochs (0 = only final)"""
    num_eval: int = 100
    """number of episodes to evaluate"""
    goal_tol: float = 1.0
    """pixel L2 distance threshold for success (error < goal_tol)"""


@dataclass
class EvalArgs(_SharedArgs):
    # path
    ckpt_name: str = "best_success_ckpt.pt"
    """checkpoint filename under runs/{run_name}/"""

    # eval
    capture_preview: bool = True
    """save a small collage of rollout overlays under runs/{run_name}/"""
