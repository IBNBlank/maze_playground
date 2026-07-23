#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################
"""Collect eval results and push a Feishu sweep summary.

Reads ``runs/[priv_]seed{seed}_{dataset}_{algo}/eval/eval_result.json``.
"""

import os, sys, tyro
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils.feishu import (
    collect_eval_results,
    format_eval_sweep_markdown,
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
    """directory containing [priv_]seed{seed}_{dataset}_{algo}/eval/"""


def main():
    args = tyro.cli(NotifyEvalArgs)
    if not args.seeds or not args.algos or not args.datasets:
        print("[notify_eval] no seeds/algos/datasets; nothing to summarize.")
        sys.exit(0)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / runs_dir

    results = collect_eval_results(
        runs_dir, args.seeds, args.datasets, args.algos)
    markdown, _ = format_eval_sweep_markdown(
        args.seeds, args.algos, args.datasets, results)
    print("[notify_eval] summary:")
    print(markdown)

    send_feishu_eval_sweep_notification(
        REPO_DIR,
        seeds=args.seeds,
        algos=args.algos,
        dataset_names=args.datasets,
        results=results,
    )


if __name__ == "__main__":
    main()
