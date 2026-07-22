#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Notify Feishu after a full ``run_train.sh`` sweep."""

import os, sys, tyro
from dataclasses import dataclass

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils.feishu import send_feishu_train_sweep_notification


@dataclass
class NotifyTrainArgs:
    seeds: list[str]
    """training seeds used in the sweep"""
    algos: list[str]
    """policy algorithms used in the sweep"""
    datasets: list[str]
    """dataset names under ../datasets/"""
    use_class: bool = False
    """match train --use-class run dirs (priv_*)"""


def main():
    args = tyro.cli(NotifyTrainArgs)
    send_feishu_train_sweep_notification(
        REPO_DIR,
        seeds=args.seeds,
        algos=args.algos,
        dataset_names=args.datasets,
        use_class=args.use_class,
    )


if __name__ == "__main__":
    main()
