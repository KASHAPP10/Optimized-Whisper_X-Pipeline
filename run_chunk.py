#!/usr/bin/env python3
"""Small runner to transcribe a single audio chunk using the optimized pipeline.
Usage:
    python run_chunk.py --audio /path/to/chunk.wav --outdir /path/to/results --model small --device cpu
This script is intentionally simple so it can be called from SLURM array jobs.
"""
import argparse
import os
from pathlib import Path

from transcribe_optimized import load_whisper_model, transcribe_file_with_vad_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Path to chunk audio file")
    p.add_argument("--outdir", required=True, help="Directory to write CSV output")
    p.add_argument("--model", default="small", help="whisperx model name")
    p.add_argument("--device", default=None, help="cpu or cuda")
    p.add_argument("--vad_top_db", type=int, default=30)
    p.add_argument("--vad_pad_ms", type=int, default=100)
    p.add_argument("--min_voiced_ms", type=int, default=100)
    p.add_argument("--chunk_duration", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    audio = args.audio
    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)
    base = Path(audio).stem
    out_csv = os.path.join(outdir, f"transcript_{base}.csv")

    # load model once per run
    try:
        model = load_whisper_model(args.model, device=args.device)
    except Exception as e:
        print("Failed to load model:", e)
        raise

    print(f"Transcribing {audio} -> {out_csv} on device={args.device}")
    try:
        transcribe_file_with_vad_params(
            audio,
            out_csv,
            model=model,
            device=args.device,
            vad_top_db=args.vad_top_db,
            vad_pad_ms=args.vad_pad_ms,
            min_voiced_ms=args.min_voiced_ms,
            chunk_duration=args.chunk_duration,
        )
        print("Done", out_csv)
    except Exception as e:
        print("Transcription failed:", e)
        raise


if __name__ == '__main__':
    main()
