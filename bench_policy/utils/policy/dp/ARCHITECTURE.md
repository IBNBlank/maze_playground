# Diffusion Policy (DP) 框架

条件 DDPM 动作分块策略：观测侧与 BC / ACT 共用 map CNN + state MLP，动作侧为 Conditional 1D UNet + DDPM。

## 模块分层

```mermaid
flowchart TB
  subgraph iface["PolicyBase"]
    IB["infer_batch(obs)"]
    UB["update_batch(batch)"]
  end

  subgraph dp_pkg["dp/"]
    Pol["policy.py · DpPolicy"]
    Mod["model.py · DpModel"]
    Loss["loss.py · dp_noise_mse_loss"]
    Opt["optim.py · AdamW"]
  end

  subgraph helper["helper/"]
    UNet["ConditionalUnet1D"]
    Sched["build_ddpm_scheduler"]
  end

  IB --> Pol
  UB --> Pol
  Pol --> Mod
  Pol --> Loss
  Pol --> Opt
  Mod --> UNet
  Mod --> Sched
```

| 文件 | 职责 |
|------|------|
| `policy.py` | `DpPolicy`：实现 `infer_batch` / `update_batch` |
| `model.py` | `DpModel`：条件编码 + UNet 噪声预测 + DDPM 采样 |
| `loss.py` | `dp_noise_mse_loss`：ε̂ 与 ε 的 MSE |
| `optim.py` | AdamW（ManiSkill DP 默认 betas / weight decay） |
| `helper/conditional_unet1d.py` | FiLM 条件 1D UNet |
| `helper/ddpm_scheduler.py` | 自实现 `DDPMScheduler` / `build_ddpm_scheduler`（无 diffusers） |

## 数据流（训练 / 推理）

```mermaid
flowchart LR
  subgraph enc["共享条件塔"]
    Map["map → CNN"]
    State["state → MLP"]
    Cond["cond = cat(map_feat, state_feat)"]
    Map --> Cond
    State --> Cond
  end

  subgraph train["update_batch"]
    A["action"]
    N["noise ~ N(0,I)"]
    T["t ~ Uniform"]
    Add["scheduler.add_noise"]
    Pred["UNet: ε̂"]
    MSE["MSE(ε̂, ε)"]
    A --> Add
    N --> Add
    T --> Add
    Add --> Pred
    Cond --> Pred
    Pred --> MSE
    N --> MSE
  end

  subgraph infer["infer_batch / sample"]
    X0["x_T ~ N(0,I)"]
    Loop["for t in timesteps"]
    Step["UNet → scheduler.step"]
    Out["action chunk"]
    X0 --> Loop
    Cond --> Loop
    Loop --> Step
    Step --> Loop
    Loop --> Out
  end
```

- **训练**：对 GT action 加噪 → UNet 预测噪声 → MSE；单次前向。
- **推理**：从高斯噪声出发，按 scheduler `timesteps` 迭代去噪，得到 `(B, pred_horizon, action_dim)`。

## DpModel 内部

```mermaid
flowchart TB
  In["map + state"] --> Enc["encode_cond"]
  Enc --> Cond["cond_dim = 1024 + 128"]
  XA["noisy_actions (B, H, A)"] --> UNet["ConditionalUnet1D"]
  Cond -->|"FiLM global_cond"| UNet
  TS["timestep"] -->|"sinusoidal embed"| UNet
  UNet --> Eps["ε̂"]
  Sched["build_ddpm_scheduler · 100 steps · ε-pred"] -.->|"train: add_noise / infer: step"| XA
```

默认超参见 `DpModelConfig`：`unet_dims=(64,128,256)`，`num_diffusion_iters=100`；调度器由 `helper.ddpm_scheduler.build_ddpm_scheduler` 提供（ε-pred，`squaredcos_cap_v2`，`clip_sample`）。
