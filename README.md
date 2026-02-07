# eggroll_v2

Text-to-pose（手语姿态）两阶段训练项目：

1. VQ-VAE：`pose -> discrete motion tokens`
2. GPT：`text -> motion tokens -> VQ-VAE decode`

## 项目重构说明

本仓库已重构为“核心代码 + 分层命令入口”模式：

- `src/`：核心模型、数据、工具模块（可复用）
- `src/commands/`：所有可执行入口（按功能分层）
- `run.py`：统一命令调度器

不再在根目录平铺训练/评估脚本。

## 目录结构

```text
src/
  config.py
  dataloader.py
  model_vqvae.py
  model_gpt.py
  train_utils.py
  cli_args.py
  latent/
  latent2d/
    data_utils.py
  commands/
    train/
      vqvae.py
      gpt.py
      ae2d.py
      flow2d.py
    infer/
      t2m.py
      flow2d_text.py
    eval/
      vqvae.py
      gpt_bleu.py
    data/
      extract_displacements.py
      avg_displacements.py
    viz/
      flow2d_samples.py
      gt_text.py
      p0_points.py
      reconstruct_from_deltas.py
    verify/
      latent_step1.py
      latent2d_step1.py
    debug/
      memory_leak.py
run.py
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据目录

默认从 `./datasets/ASL_gloss` 读取：

```text
datasets/ASL_gloss/
  train/<sid>/{text.txt, pose.json}
  dev/<sid>/{text.txt, pose.json}
  test/<sid>/{text.txt, pose.json}
```

## 统一命令（推荐）

通过 `run.py` 执行：

```bash
python run.py train-vqvae --session_epochs 3
python run.py train-gpt
python run.py infer-t2m --text "apple"
python run.py eval-vqvae --checkpoint ./checkpoints/vqvae_model.pth
python run.py eval-gpt-bleu --vqvae_checkpoint ./checkpoints/vqvae_model.pth --gpt_checkpoint ./checkpoints/t2m_gpt_model.pth
```

也可以直接运行模块：

```bash
python -m src.commands.train.vqvae --session_epochs 3
python -m src.commands.infer.t2m --text "hello"
```

## 常用流程

1. 训练 VQ-VAE

```bash
python run.py train-vqvae --session_epochs 3
```

2. 训练 GPT

```bash
python run.py train-gpt
```

3. 文本推理并导出 GIF

```bash
python run.py infer-t2m \
  --vqvae_checkpoint ./checkpoints/vqvae_model.pth \
  --gpt_checkpoint ./checkpoints/t2m_gpt_model.pth \
  --text "apple" \
  --output_dir ./outputs
```

## 其他工具命令

- 位移提取与模板
  - `python run.py data-extract-displacements ...`
  - `python run.py data-avg-displacements ...`
- 2D latent 系列
  - `python run.py train-ae2d ...`
  - `python run.py train-flow2d ...`
  - `python run.py infer-flow2d-text ...`
- 可视化
  - `python run.py viz-flow2d-samples ...`
  - `python run.py viz-gt-text ...`
  - `python run.py viz-p0-points ...`
  - `python run.py viz-reconstruct-deltas ...`
- 验证与调试
  - `python run.py verify-latent-step1`
  - `python run.py verify-latent2d-step1`
  - `python run.py debug-memory-leak`

## 说明

- `cfg`（`src/config.py`）仍是训练默认参数单一来源。
- 新结构下脚本间不再互相依赖入口文件，公共逻辑统一在 `src` 模块中。
