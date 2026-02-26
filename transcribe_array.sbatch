#!/bin/bash
# SLURM array job template to transcribe chunk files in parallel
# Edit CHUNKS_DIR, OUTDIR, VENV_PATH, MODEL_NAME as appropriate before submitting

#SBATCH --job-name=whisperx_trans
#SBATCH --output=logs/transcribe_%A_%a.out
#SBATCH --error=logs/transcribe_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-<N_CHUNKS_MINUS1>

# USER CONFIG: update these paths
CHUNKS_DIR="/scratch/your_user/chunks"      # directory containing chunk wav files
OUTDIR="/scratch/your_user/whisperx_results" # where CSVs will be written
VENV_PATH="/scratch/your_user/WhisperX/venv" # path to the python venv to activate
MODEL_NAME="small"
DEVICE="cpu"  # or 'cuda' if you request GPU resources and have access

# load module / env if your cluster uses modules (optional)
# module load python/3.10

# activate venv
source "${VENV_PATH}/bin/activate"

# build file list
FILES=("${CHUNKS_DIR}"/*.{wav,mp3,m4a})
# guard for bash globbing
FILES=( ${FILES[@]} )

IDX=${SLURM_ARRAY_TASK_ID}
CHUNK=${FILES[$IDX]}

if [ -z "$CHUNK" ]; then
  echo "No chunk found for index ${IDX}. Exiting."
  exit 2
fi

mkdir -p "$OUTDIR"

echo "Task ${SLURM_ARRAY_TASK_ID} processing ${CHUNK}"
python run_chunk.py --audio "$CHUNK" --outdir "$OUTDIR" --model "$MODEL_NAME" --device "$DEVICE"
