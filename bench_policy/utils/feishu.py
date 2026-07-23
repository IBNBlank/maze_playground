#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-07-21
################################################################

import json
import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .common import make_run_name

if TYPE_CHECKING:
    from .common import Metrics


def _load_feishu_config(repo_dir: str) -> Optional[dict]:
    config_path = os.path.join(repo_dir, "feishu.json")
    if not os.path.isfile(config_path):
        print(f"[feishu] config not found at {config_path}, skip")
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[feishu] failed to read {config_path}: {e}")
        return None


def _feishu_template_from_success(success_rate: float) -> str:
    if success_rate != success_rate:  # NaN
        return "grey"
    if success_rate <= 0.5:
        return "red"
    if success_rate <= 0.8:
        return "yellow"
    return "green"


def _post_feishu_card(
    webhook_url: str,
    *,
    title: str,
    elements: Sequence[dict],
    template: str = "green",
) -> bool:
    payload = json.dumps(
        {
            "msg_type": "interactive",
            "card": {
                "schema": "2.0",
                "header": {
                    "title": {
                        "tag": "plain_text",
                        # Feishu bot custom keyword must appear in the message.
                        "content": title,
                    },
                    "template": template,
                },
                "body": {"elements": list(elements)},
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        result = json.loads(body)
    except Exception as e:
        print(f"[feishu] notification error: {e}")
        return False
    if result.get("code") == 0:
        print("[feishu] notification sent")
        return True
    print(f"[feishu] notification rejected: {body}")
    return False


def send_feishu_notification(
    repo_dir: str,
    *,
    mode: str = "train",
    title: Optional[str] = None,
    markdown: Optional[str] = None,
    elements: Optional[Sequence[dict]] = None,
    success_rate: Optional[float] = None,
    template: Optional[str] = None,
) -> bool:
    """Wrap content into a Feishu interactive card and POST it."""
    mode = mode.lower()
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

    config = _load_feishu_config(repo_dir)
    if config is None:
        return False
    key = "train_webhook_url" if mode == "train" else "eval_webhook_url"
    webhook_url = config.get(key)
    if not webhook_url:
        print(f"[feishu] {key} missing in feishu.json, skip")
        return False

    if title is None:
        title = ("Maze Training Finished"
                 if mode == "train" else "Maze Evaluation Finished")
    if template is None:
        template = (_feishu_template_from_success(float(success_rate))
                    if success_rate is not None else "green")

    if elements is not None:
        body = list(elements)
    elif markdown is not None:
        body = [{"tag": "markdown", "content": markdown}]
    else:
        raise ValueError("either markdown or elements must be provided")

    return _post_feishu_card(
        webhook_url,
        title=title,
        elements=body,
        template=template,
    )


def send_feishu_train_notification(
    repo_dir: str,
    *,
    algo: str,
    dataset_name: str,
    seed: int,
    epochs: int,
    metrics: "Metrics",
    run_name: Optional[str] = None,
) -> bool:
    """Notify after one training run finishes."""
    md_lines = [
        f"- **algo:** {algo}",
        f"- **dataset:** {dataset_name}",
        f"- **seed:** {seed}",
        f"- **epochs:** {epochs}",
        f"- **best_success:** {metrics.best_success_rate * 100:.2f}%",
        f"- **best_success_average_steps:** "
        f"{metrics.best_success_average_steps:.2f}",
    ]
    if run_name:
        md_lines.insert(0, f"- **run:** {run_name}")
    return send_feishu_notification(
        repo_dir,
        mode="train",
        markdown="\n".join(md_lines),
        success_rate=metrics.best_success_rate,
    )


def send_feishu_train_sweep_notification(
    repo_dir: str,
    *,
    seeds: Sequence[Any],
    algos: Sequence[str],
    dataset_names: Sequence[str],
) -> bool:
    """Notify after a full ``run_train.sh`` sweep finishes."""
    if not seeds or not algos or not dataset_names:
        print("[feishu] empty train sweep; nothing to notify")
        return False
    md = (f"- **datasets:** {' '.join(str(d) for d in dataset_names)}\n"
          f"- **seeds:** {' '.join(str(s) for s in seeds)}\n"
          f"- **algos:** {' '.join(str(a) for a in algos)}")
    return send_feishu_notification(
        repo_dir,
        mode="train",
        title="Maze Training Finished",
        markdown=md,
        template="green",
    )


def collect_eval_results(
    runs_dir: str | Path,
    seeds: Sequence[Any],
    dataset_names: Sequence[str],
    algos: Sequence[str],
) -> list[dict]:
    """Load ``eval_result.json`` for the sweep grid (obs + priv)."""
    results: list[dict] = []
    root = Path(runs_dir)
    for seed in seeds:
        for dataset in dataset_names:
            for algo in algos:
                for use_class in (False, True):
                    run = make_run_name(seed,
                                        dataset,
                                        algo,
                                        use_class=use_class)
                    path = root / run / "eval" / "eval_result.json"
                    if not path.is_file():
                        continue
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        results.append({
                            "run_name": run,
                            "algo": str(data.get("algo", algo)),
                            "dataset_name": str(
                                data.get("dataset_name", dataset)),
                            "use_class": bool(
                                data.get("use_class", use_class)),
                            "seed": data.get("train_seed", seed),
                            "success_rate": float(data["success_rate"]),
                            "collision_rate": float(
                                data.get("collision_rate", float("nan"))),
                            "success_average_steps": data.get(
                                "success_average_steps"),
                            "num_episodes": data.get("num_episodes"),
                        })
                    except (OSError, json.JSONDecodeError, KeyError, TypeError,
                            ValueError):
                        continue
    return results


def _fmt_rate(rate: Optional[float]) -> str:
    if rate is None:
        return "-"
    return f"{rate * 100:.1f}%"


def _rate_color(rate: Optional[float]) -> str:
    if rate is None:
        return "grey"
    if rate <= 0.5:
        return "red"
    if rate <= 0.8:
        return "yellow"
    return "green"


def _option_cell(rate: Optional[float]) -> list[dict]:
    return [{"text": _fmt_rate(rate), "color": _rate_color(rate)}]


def _mean(rates) -> Optional[float]:
    rates = [r for r in rates if r is not None]
    if not rates:
        return None
    return sum(rates) / len(rates)


def _index_results(results: Sequence[dict]) -> dict[tuple, dict]:
    return {
        (str(r["seed"]), str(r["dataset_name"]), str(r["algo"]),
         bool(r["use_class"])): r
        for r in results
    }


def _success_rate(index, algo, dataset, seed, use_class) -> Optional[float]:
    row = index.get((str(seed), str(dataset), str(algo), bool(use_class)))
    return None if row is None else float(row["success_rate"])


def _row_specs(algos: Sequence[str]) -> list[tuple[str, str, str, bool]]:
    """(row_key, display_label, algo, use_class) — labels follow run_name."""
    specs = []
    for algo in algos:
        specs.append((f"row_{algo}", algo, algo, False))
        specs.append((f"row_priv_{algo}", f"priv_{algo}", algo, True))
    return specs


def _feishu_table(columns: list[dict], rows: list[dict]) -> dict:
    return {
        "tag": "table",
        "row_height": "low",
        "header_style": {"background_style": "grey", "bold": True},
        "columns": columns,
        "rows": rows,
    }


def _text_col(name: str, display_name: str) -> dict:
    return {
        "name": name,
        "display_name": display_name,
        "data_type": "text",
        "horizontal_align": "left",
    }


def _opt_col(name: str, display_name: str) -> dict:
    return {
        "name": name,
        "display_name": display_name,
        "data_type": "options",
        "horizontal_align": "center",
    }


def _cell_rate(index, algo, dataset, use_class, seeds) -> Optional[float]:
    """Mean over ``seeds`` (pass a single-seed list for per-seed tables)."""
    return _mean(
        _success_rate(index, algo, dataset, s, use_class) for s in seeds)


def _algo_dataset_table(index, algos, datasets, seeds):
    """Rows=algo (bc/priv_bc/...), cols=dataset; values mean over ``seeds``."""
    specs = _row_specs(algos)
    columns = [_text_col("algo", "algo \\ dataset")]
    for ds in datasets:
        columns.append(_opt_col(f"ds_{ds}", str(ds)))

    rows = []
    for _, label, algo, use_class in specs:
        row: dict[str, Any] = {"algo": label}
        for ds in datasets:
            rate = _cell_rate(index, algo, ds, use_class, seeds)
            row[f"ds_{ds}"] = _option_cell(rate)
        rows.append(row)
    return _feishu_table(columns, rows)


def _algo_dataset_text(index, algos, datasets, seeds):
    specs = _row_specs(algos)
    lines = ["algo \\ dataset | " + " | ".join(str(d) for d in datasets)]
    for _, label, algo, use_class in specs:
        cells = [
            _fmt_rate(_cell_rate(index, algo, ds, use_class, seeds))
            for ds in datasets
        ]
        lines.append(f"{label} | " + " | ".join(cells))
    return "\n".join(lines)


def format_eval_sweep_markdown(
    seeds: Sequence[Any],
    algos: Sequence[str],
    dataset_names: Sequence[str],
    results: Sequence[dict],
) -> tuple[str, Optional[float]]:
    """Build console-friendly eval-sweep markdown + mean success."""
    index = _index_results(results)
    expected = len(seeds) * len(dataset_names) * len(algos) * 2
    mean_rate = _mean(float(r["success_rate"]) for r in results)
    overall_text = _algo_dataset_text(index, algos, dataset_names, seeds)

    parts = [
        f"- **datasets:** {' '.join(str(d) for d in dataset_names)}",
        f"- **seeds:** {' '.join(str(s) for s in seeds)}",
        f"- **algos:** {' '.join(str(a) for a in algos)}",
        f"- **completed:** {len(results)}/{expected}",
        f"- **mean_success:** {_fmt_rate(mean_rate)}",
        "",
        "**1. Overall** (mean over seeds)",
        overall_text,
    ]
    for seed in seeds:
        text = _algo_dataset_text(index, algos, dataset_names, [seed])
        parts.append("")
        parts.append(f"**2. Seed {seed}**")
        parts.append(text)
    return "\n".join(parts), mean_rate


def _build_eval_sweep_elements(
    seeds: Sequence[Any],
    algos: Sequence[str],
    dataset_names: Sequence[str],
    results: Sequence[dict],
) -> tuple[list[dict], Optional[float]]:
    """Build Feishu card: overall table + one table per seed."""
    index = _index_results(results)
    expected = len(seeds) * len(dataset_names) * len(algos) * 2
    mean_rate = _mean(float(r["success_rate"]) for r in results)

    summary = (f"- **datasets:** {' '.join(str(d) for d in dataset_names)}\n"
               f"- **seeds:** {' '.join(str(s) for s in seeds)}\n"
               f"- **algos:** {' '.join(str(a) for a in algos)}\n"
               f"- **completed:** {len(results)}/{expected}\n"
               f"- **mean_success:** {_fmt_rate(mean_rate)}")

    elements: list[dict] = [
        {"tag": "markdown", "content": summary},
        {
            "tag": "markdown",
            "content": "**1. Overall** (mean over seeds)",
        },
        _algo_dataset_table(index, algos, dataset_names, seeds),
    ]
    for seed in seeds:
        elements.append({
            "tag": "markdown",
            "content": f"**2. Seed {seed}**",
        })
        elements.append(
            _algo_dataset_table(index, algos, dataset_names, [seed]))
    return elements, mean_rate


def send_feishu_eval_sweep_notification(
    repo_dir: str,
    *,
    seeds: Sequence[Any],
    algos: Sequence[str],
    dataset_names: Sequence[str],
    results: Sequence[dict],
) -> bool:
    """Notify after a full ``run_eval.sh`` sweep from collected results."""
    if not seeds or not algos or not dataset_names:
        print("[feishu] empty eval sweep; nothing to notify")
        return False
    elements, mean_rate = _build_eval_sweep_elements(
        seeds, algos, dataset_names, results)
    return send_feishu_notification(
        repo_dir,
        mode="eval",
        title="Maze Evaluation Finished",
        elements=elements,
        success_rate=mean_rate,
    )
