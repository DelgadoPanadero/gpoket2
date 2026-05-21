#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$REPO_ROOT/scripts/upload_checkpoint.py" iamthinbaker/GPokeT2 \
  --checkpoint-path /workspace/train/checkpoints/latest/checkpoint-4200 \
  --version v0.1-wip-4200 \
  --context-length 4096 \
  --n-embd 512 \
  --n-layer 12 \
  --n-head 8 \
  --model-code-path "$REPO_ROOT/src/application/gym/model/conditioned_gpt2.py"
