#!/usr/bin/env python3
"""Aggregate per-chunk transcript CSVs into a single combined CSV and export suspicious segments.

Usage:
  python scripts/aggregate_transcripts.py --input-dir results --out combined.csv --suspicious suspicious.csv
"""
import argparse
import glob
import pandas as pd
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suspicious", required=False, default="suspicious_segments.csv")
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.input_dir}/transcript_*.csv"))
    if not files:
        print("No transcript files found in", args.input_dir)
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as exc:
            print("Skipping", f, "due to read error:", exc)

    combined = pd.concat(dfs, ignore_index=True)
    # try to sort by numeric start seconds if present
    if 'Start Seconds' in combined.columns:
        combined = combined.sort_values('Start Seconds')
    combined.to_csv(args.out, index=False)
    print('Wrote combined:', args.out)

    # export suspicious segments
    if 'flags' in combined.columns:
        suspicious = combined[combined['flags'].notna() & (combined['flags'].str.strip() != '')]
        suspicious = suspicious.sort_values('hallucination_score', ascending=False) if 'hallucination_score' in suspicious.columns else suspicious
        suspicious.to_csv(args.suspicious, index=False)
        print('Wrote suspicious segments:', args.suspicious)


if __name__ == '__main__':
    main()
