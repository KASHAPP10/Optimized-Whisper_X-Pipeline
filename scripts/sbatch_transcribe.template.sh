#!/bin/bash
#SBATCH --job-name=whisperx_transcribe
#SBATCH --output=logs/transcribe_%A_%a.out
#SBATCH --error=logs/transcribe_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1   # uncomment if you want a GPU

# Usage: submit with SLURM_ARRAY_TASK_ID set and CHUNKS_DIR defined
# Example: sbatch --array=0-9 scripts/sbatch_transcribe.template.sh

CHUNKS_DIR="/path/to/chunks"
OUTDIR="/path/to/results"
VENV_ACTIVATE="/path/to/venv/bin/activate"

# determine chunk for this array task
CHUNK=$(ls "$CHUNKS_DIR"/*.wav | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")

mkdir -p logs
source "$VENV_ACTIVATE"
python run_chunk.py "$CHUNK" "$OUTDIR" \
  --model small --device cpu
