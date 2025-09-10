# Repository Guidelines

## Fork Goals & Policy
- Purpose: finetune a Video2World “world model” that takes a text description of a robot stroke plus a frame sequence and predicts future frames.
- Minimal-diff fork: prefer adding configs and example scripts over touching core code. Keep changes under `cosmos_predict2/configs/**/experiment/` and `examples/`.
- Track upstream: regularly sync with NVIDIA’s main repo; resolve conflicts only in local configs/examples.

## Project Structure
- `cosmos_predict2/`: Core library (configs, datasets, models, pipelines).
- `imaginaire/`: Training/runtime utilities.
- `examples/`: Runnable demos; add `video2world_robotstroke.py` here if needed.
- `documentations/`: Inference and post‑training guides (see Video2World post‑training docs).
- `scripts/`: Training entrypoints (`scripts/train.py`) and utilities.

## Build & Dev Commands
- `just setup` → install pre‑commit; `just install` → deps via `uv` (CUDA 12.6).
- `just lint` → add license headers + `ruff` fix/format.
- Run examples: `python examples/video2world.py`.
- Docker (GPU): `just docker` or build then run with an env file to pass secrets.

## Cloud & Auth (HF + W&B)
- Prefer `.env` (not committed) with `HUGGINGFACE_HUB_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`.
- Example run: `docker run --gpus all --rm --env-file .env -v $PWD:/workspace -it $(docker build -q .)`.
- Inside container: dataset pull e.g. `huggingface-cli download <dataset> --repo-type dataset --local-dir datasets/robotstroke`.

## Fine‑tuning Workflow (Video2World)
1) Prepare data: follow `documentations/post-training_video2world.md`. Layout:
   `datasets/<name>/{metas/*.txt,videos/*.mp4}` then `python -m scripts.get_t5_embeddings --dataset_path datasets/<name>`.
2) Create experiment config: add under `cosmos_predict2/configs/base/experiment/` (e.g., `robotstroke.py`) pointing `dataset_dir="datasets/<name>"`, set frames/resolution.
3) Launch training: `torchrun --nproc_per_node=8 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_<name>`.
   Use `job.project/group/name` for W&B; add `model.config.train_architecture=lora` for LoRA.
4) Inference on finetuned ckpt: use `examples/video2world.py --model_size 2B --dit_path checkpoints/.../iter_xxx.pt [--load_ema]`.

## Style & PRs
- Python 3.10; `ruff` (line length 120). Run `just lint` before pushing.
- Commits: imperative; PRs include description, commands, and sample output.

## Upstream Sync
- `git remote add upstream https://github.com/nvidia-cosmos/cosmos-predict2.git`
- `git fetch upstream && git checkout main && git merge upstream/main` (or `rebase`). Keep custom changes in configs/examples to minimize conflicts.
