#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Notify Feishu after a full ``run_eval.sh`` sweep (aggregate summary)."""

import os, sys, tyro
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils.feishu import (
    collect_eval_results,
    send_feishu_eval_sweep_notification,
)


@dataclass
class NotifyEvalArgs:
    seeds: list[str]
    """training seeds used in the sweep"""
    algos: list[str]
    """policy algorithms used in the sweep"""
    datasets: list[str]
    """dataset names under ../datasets/"""
    runs_dir: str = "runs"
    """directory containing [priv_]seed{seed}_{dataset}_{algo}/eval/eval_result.json"""


def main():
    args = tyro.cli(NotifyEvalArgs)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / runs_dir

    results = collect_eval_results(
        runs_dir,
        args.seeds,
        args.datasets,
        args.algos,
    )
    send_feishu_eval_sweep_notification(
        REPO_DIR,
        seeds=args.seeds,
        algos=args.algos,
        dataset_names=args.datasets,
        results=results,
    )


if __name__ == "__main__":
    main()
