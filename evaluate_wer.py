#!/usr/bin/env python3
"""WER evaluation skeleton.

This is a minimal, clear structure for computing WER between ground truth
CSV(s) and generated transcript CSV(s). Fill in the normalization and
scoring details as needed.
"""
import argparse
from pathlib import Path
import pandas as pd


def load_transcripts(path: str):
    """Load a transcript CSV and return a list of strings (utterances).

    Expected column names: 'transcript' or 'Transcript' or 'Transcript'
    """
    df = pd.read_csv(path)
    for col in ['transcript', 'Transcript', 'Transcript']:
        if col in df.columns:
            return df[col].astype(str).tolist()
    # fallback: take first text-like column
    return df.iloc[:, 0].astype(str).tolist()


def normalize_text(s: str) -> str:
    """Normalize text for fair WER computation (lowercase, strip, remove punctuation).
    Implement per-project rules here.
    """
    return s.lower().strip()


def compute_wer(ref: str, hyp: str) -> float:
    """Compute WER between reference and hypothesis (token-level).
    Replace with your preferred implementation (numpy, python Levenshtein, etc.).
    """
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    # simple DP
    import numpy as np
    d = np.zeros((len(r)+1, len(h)+1), dtype=int)
    for i in range(len(r)+1): d[i,0]=i
    for j in range(len(h)+1): d[0,j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[len(r),len(h)]/max(1,len(r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='Ground truth CSV')
    p.add_argument('--hyp', required=True, help='Hypothesis CSV')
    p.add_argument('--out', default='wer_report.csv')
    args = p.parse_args()

    refs = load_transcripts(args.ref)
    hyps = load_transcripts(args.hyp)

    n = min(len(refs), len(hyps))
    rows = []
    total_err = 0.0
    total_ref_words = 0
    for i in range(n):
        r = refs[i]
        h = hyps[i]
        wer = compute_wer(r, h)
        rows.append({'ref': r, 'hyp': h, 'wer': wer})
        total_err += wer * len(r.split())
        total_ref_words += len(r.split())

    overall = total_err / max(1, total_ref_words)
    print('Aggregate WER:', overall)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
