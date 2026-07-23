# Diffusion Policy (DP) 框架

条件 DDPM / DDIM 动作分块策略：观测侧与 BC / ACT / FM 共用 map CNN + state MLP，动作侧为 Conditional 1D UNet；训练用 DDPM 加噪，推理用较短 DDIM 链。

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
    Sched["ddpm / ddim scheduler"]
  end

  subgraph helper["helper/"]
    Obs["ObsEncoder"]
    UNet["ConditionalUnet1D"]
    Ema["EMAModel"]
  end

  IB --> Pol
  UB --> Pol
  Pol --> Mod
  Pol --> Loss
  Pol --> Opt
  Pol --> Ema
  Mod --> Obs
  Mod --> UNet
  Mod --> Sched
```

| 文件 | 职责 |
|------|------|
| `policy.py` | `DpPolicy`：实现 `infer_batch` / `update_batch` + EMA |
| `model.py` | `DpModel`：条件编码 + UNet 噪声预测 + DDPM 训练 / DDIM 采样 |
| `loss.py` | `dp_noise_mse_loss`：ε̂ 与 ε 的 MSE |
| `optim.py` | AdamW（ManiSkill DP 默认 betas / weight decay） |
| `ddpm_scheduler.py` | 自实现 `DDPMScheduler` / `build_ddpm_scheduler`（无 diffusers） |
| `ddim_scheduler.py` | 自实现 `DDIMScheduler` / `build_ddim_scheduler`（推理） |
| `helper/obs_encoder.py` | map CNN + state MLP → `cond` |
| `helper/conditional_unet1d.py` | FiLM 条件 1D UNet |
| `helper/ema.py` | 权重 EMA |

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
    Add["noise_scheduler.add_noise"]
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
    Loop["for t in infer_scheduler.timesteps"]
    Step["UNet → infer_scheduler.step"]
    Out["action chunk"]
    X0 --> Loop
    Cond --> Loop
    Loop --> Step
    Step --> Loop
    Loop --> Out
  end
```

- **训练**：对 GT action 用 DDPM `add_noise` → UNet 预测噪声 → MSE；单次前向。
- **推理**：从高斯噪声出发，按 DDIM `infer_scheduler.timesteps` 迭代去噪，得到 `(B, pred_horizon, action_dim)`。

## DpModel 内部

```mermaid
flowchart TB
  In["map + state"] --> Enc["ObsEncoder"]
  Enc --> Cond["cond_dim = 1024 + 128"]
  XA["noisy_actions (B, H, A)"] --> UNet["ConditionalUnet1D"]
  Cond -->|"FiLM global_cond"| UNet
  TS["timestep"] -->|"sinusoidal embed"| UNet
  UNet --> Eps["ε̂"]
  TrainSched["build_ddpm_scheduler · 100 steps · ε-pred"] -.->|"train: add_noise"| XA
  InferSched["build_ddim_scheduler · 20 steps · ε-pred"] -.->|"infer: step"| XA
```

默认超参见 `DpModelConfig`：`unet_dims=(64,128,256)`，`num_diffusion_iters=100`，`num_inference_iters=20`；训练调度器由 `dp.ddpm_scheduler.build_ddpm_scheduler` 提供（ε-pred，`squaredcos_cap_v2`，`clip_sample`），推理用 `dp.ddim_scheduler.build_ddim_scheduler`。
