# maze_playground

Multimodal planning-map tools: dataset generation under `bench_data/`, imitation learning under `bench_policy/`.

## Batch generation

```bash
cd bench_data
./run_gen.sh
# or
../.venv/bin/python data_gen.py \
  --num-maps 500 --size 256 --num-routes 4 --seed 56 \
  --output-dir ../dataset/genplan256_r4
```

`run_gen.sh` loops over `NUM_ROUTES_LIST` (default `2 3 4 5 6`), writing each dataset to `${OUTPUT_DIR}_r${num_routes}` with `seed = num_routes * 14`.

Outputs per dataset:

| Path | Content |
|------|---------|
| `shard_XXXXX.npz` | Batched sample tensors |
| `preview.png` | Collage of leading maps + routes |
| `manifest.json` | Shard list and encoding notes |
| `config.json` | Resolved `GenCfg` |

### Sample tensors (`shard_*.npz`)

| Key | Meaning |
|-----|---------|
| `maps` | Occupancy (`1` = occupied) |
| `planning_maps` | Occupancy dilated by `robot_radius` |
| `starts_rc` / `goals_rc` | Start / goal in `[row, col]` |
| `waypoints_xy` | `(K, H+1, 2)` normalized absolute `xy` |
| `action_chunks` | `(K, H, 2)` pixel `(dx, dy)` with `|dx|,|dy| <= max_abs_delta` (default 5); L∞-maximal steps, then zeros after goal |
| `raw_paths_rc` | Grid expert paths (padded) |
| `route_lengths` / `optimal_lengths` | Path length stats |

Decode: `pixel[i+1] = pixel[i] + action[i]`, `q = pixel / (size - 1)`.

## Single-map debug

Dump every intermediate stage for one accepted map:

```bash
cd bench_data
../.venv/bin/python data_gen_single.py \
  --output-dir ../dataset/genplan256_single \
  --size 256 --num-routes 4 --seed 7
```

| Path | Content |
|------|---------|
| `images/*.png` | Colored overlay per stage |
| `arrays/*.npy` | Raw masks / paths / costs |
| `steps.json` | Ordered stage list |
| `sample_single.npz` | Final sample |

## Generation pipeline

```mermaid
flowchart TD
    A[sample_start_goal] --> B[build_route_polylines]
    B --> C[build_free_space]
    C --> D[seal_occupancy]
    D --> E1["inflate → search_map<br/>radius = robot_radius + search_inflate_extra"]
    D --> E2["inflate → planning_map<br/>radius = robot_radius"]
    E1 --> F[generate_expert_routes on search_map]
    F --> G[shortcut_path on planning_map]
    G --> H[L-inf max-step action_chunks + zero pad]
    H --> I[validate_waypoints on planning_map]
    I --> J[final sample]

    A -.- A1["start / goal + tangent / normal"]
    B -.- B1["K control polylines"]
    C -.- C1["free_mask + per-route guide_mask"]
    D -.- D1["occupancy = maps"]
    E1 -.- E1a["extra clearance for search"]
    E2 -.- E2a["nominal clearance for labels"]
    F -.- F1["K diverse MCP routes"]
    G -.- G1["guide-constrained string-pull"]
    H -.- H1["waypoints_xy / action_chunks"]
    J -.- J1["npz shards or sample_single.npz"]
```

### Stage notes

1. **start / goal** — Opposite-side endpoints; tangent / normal for lateral route offset.
2. **route polylines** — `K` shared-endpoint control curves with sinusoidal envelope and jitter.
3. **free space** — Carve corridors (width ~ `2–3 × robot_radius`), rooms, cross-links, side branches; build per-route guide masks. No protected core / pillars.
4. **seal occupancy** — `occupancy = ~free_mask`, force border, keep start/goal free → stored as `maps`.
5. **dual inflate**
   - `search_map`: `robot_radius + search_inflate_extra` (default extra `3`) — expert search stays off walls.
   - `planning_map`: `robot_radius` — shortcut and waypoint validation; stored as `planning_maps`.
6. **expert routes** — On `search_map`, per guide: smooth random cost + guide penalty → MCP path; reject on length / buffered IoU.
7. **shortcut** — On `planning_map`, farthest free chord inside each guide (mode-preserving simplify).
8. **waypoints / actions** — Walk the shortcut path with L∞-maximal pixel steps (`|dx|,|dy| <= max_abs_delta`); pad to fixed `action_horizon` with zeros after the goal.
9. **validate** — Consecutive waypoint chords must be free on `planning_map`.
10. **accept / retry** — Any failed check returns `None`; outer loop retries up to `max_map_attempts`.

## bench_policy

IL training / evaluation on `datasets/{dataset_name}` (build via `bench_data/`).
`run_name = seed{seed}_{dataset_name}_{algo}`.

```mermaid
flowchart TB
    dataset_data["../datasets/{dataset_name}"]

    dataset_data --> train
    subgraph train["train"]
        train_bash["run_train.sh"]
        train_bash --> |seed, dataset_name, algo, epochs| train_python["train.py"]

        train_python --> train_log
        subgraph train_log["logs"]
            train_log_tensorboard["runs/{run_name}/ (tensorboard)"]
            train_log_latest["runs/{run_name}/latest.json"]
            train_log_success["runs/{run_name}/best_success.json"]
        end

        train_python --> train_ckpt
        subgraph train_ckpt["ckpts"]
            train_ckpt_mid["runs/{run_name}/ckpt_*.pt"]
            train_ckpt_final["runs/{run_name}/final_ckpt.pt"]
            train_ckpt_success["runs/{run_name}/best_success_ckpt.pt"]
        end

        train_python --> train_notify
        subgraph train_notify["notify"]
            train_notify_job["send_feishu_train_notification"]
        end

        train_bash --> train_sweep["notify_train.py → sweep"]
    end

    train_ckpt --> eval
    subgraph eval["eval"]
        eval_bash["run_eval.sh"]
        eval_bash --> |seed, dataset_name, algo, ckpt_name| eval_python["eval.py"]

        eval_python --> eval_log
        subgraph eval_log["logs"]
            eval_log_tb["runs/{run_name}/eval/ (tensorboard)"]
            eval_log_result["runs/{run_name}/eval_result.json"]
            eval_log_preview["runs/{run_name}/eval_preview.png"]
        end

        eval_python --> eval_notify
        subgraph eval_notify["notify"]
            eval_notify_job["send_feishu_eval_notification"]
        end

        eval_bash --> eval_sweep["notify_eval.py → sweep"]
    end
```

### train.py

```mermaid
flowchart TB
    start["tyro.cli TrainArgs"] --> run_name["run_name = seed{seed}_{dataset}_{algo}"]
    run_name --> early{"latest.json already finished?"}
    early -->|yes| done_early["early exit"]
    early -->|no| init["device + TB + MazeWindowDataset + eval_episodes"]
    init --> policy["build_policy"]
    policy --> resume{"latest.json exists?"}
    resume -->|yes| load["load ckpt_name + restore metrics / start_epoch"]
    resume -->|no| pick["sample epoch_ids from idx/ perms"]
    load --> pick
    pick --> loop

    subgraph loop["for epoch_idx in epochs"]
        set_epoch["dataset.set_epoch(epoch_ids[epoch_idx])"]
        set_epoch --> batches["policy.update_batch until epoch done"]
        batches --> eval_gate{"epoch % eval_freq == 0 and not last?"}
        eval_gate -->|yes| mid_eval["evaluate → ckpt_*.pt + latest.json + best_*"]
        eval_gate -->|no| next_ep["next epoch"]
        mid_eval --> next_ep
    end

    loop --> final_eval["evaluate → final_ckpt.pt + latest.json iteration=-1"]
    final_eval --> feishu["send_feishu_train_notification"]
    feishu --> done["close writer"]
```

### eval.py

```mermaid
flowchart TB
    start["tyro.cli EvalArgs"] --> run_name["run_name = seed{seed}_{dataset}_{algo}"]
    run_name --> init["device + TB eval/ + MazeWindowDataset"]
    init --> episodes["build_eval_episodes(num_eval)"]
    episodes --> policy["build_policy + load runs/{run_name}/{ckpt_name}"]
    policy --> roll["evaluate open-loop rollouts"]
    roll --> metrics["success_rate / success_average_steps / collision_rate"]
    metrics --> log["log_eval_summary → tensorboard"]
    log --> json["runs/{run_name}/eval_result.json"]
    roll -.-> preview["runs/{run_name}/eval_preview.png"]
    json --> feishu["send_feishu_eval_notification"]
    feishu --> done["close writer"]
```

### Quick start

```bash
cd bench_policy
./run_train.sh
# or
../.venv/bin/python train.py \
  --algo bc --dataset-name genplan256_mix --seed 42 --epochs 50

./run_eval.sh
# or
../.venv/bin/python eval.py \
  --algo bc --dataset-name genplan256_mix --seed 42 \
  --ckpt-name best_success_ckpt.pt
```
