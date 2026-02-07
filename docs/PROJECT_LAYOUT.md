# Project Layout

This repository now follows a command-layer architecture:

- `src/`: reusable core library code
- `src/commands/`: executable command modules grouped by purpose
- `run.py`: single entrypoint that dispatches subcommands

## Command Groups

- `src/commands/train/`
  - `vqvae.py`
  - `gpt.py`
  - `ae2d.py`
  - `flow2d.py`
- `src/commands/infer/`
  - `t2m.py`
  - `flow2d_text.py`
- `src/commands/eval/`
  - `vqvae.py`
  - `gpt_bleu.py`
- `src/commands/data/`
  - `extract_displacements.py`
  - `avg_displacements.py`
- `src/commands/viz/`
  - `flow2d_samples.py`
  - `gt_text.py`
  - `p0_points.py`
  - `reconstruct_from_deltas.py`
- `src/commands/verify/`
  - `latent_step1.py`
  - `latent2d_step1.py`
- `src/commands/debug/`
  - `memory_leak.py`

## Shared Utilities

- `src/latent2d/data_utils.py`
  - shared filelist cache builder
  - shared `pose.json` loader
  - shared sample-sequence loading helpers

This removes command-to-command imports and keeps shared logic in reusable modules.

## Recommended Usage

Prefer the unified launcher:

```bash
python run.py <command> [args...]
```

Example:

```bash
python run.py train-vqvae --session_epochs 3
python run.py infer-t2m --text "hello"
```
