#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path


def read_transcript_csv(path):
    df = pd.read_csv(path)
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str)
        return df['text'].tolist()
    for c in ['Transcript','transcript','Text','TEXT']:
        if c in df.columns:
            return df[c].astype(str).tolist()
    return df.astype(str).agg(' '.join, axis=1).tolist()


def simple_wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    import numpy as _np
    d = _np.zeros((len(r)+1, len(h)+1), dtype=int)
    for i in range(len(r)+1): d[i,0]=i
    for j in range(len(h)+1): d[0,j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            if r[i-1]==h[j-1]: cost=0
            else: cost=1
            d[i,j]=min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[len(r),len(h)]/max(1,len(r))


def compare(refs, hyps):
    rows=[]
    total_err=0.0
    total_ref_words=0
    n=min(len(refs), len(hyps))
    for i in range(n):
        r=refs[i]
        h=hyps[i]
        err = simple_wer(r,h)
        rows.append({'ref':r,'hyp':h,'wer':err})
        total_err += err*len(r.split())
        total_ref_words += len(r.split())
    overall = total_err/max(1,total_ref_words)
    return overall, pd.DataFrame(rows)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='Path to human CSV')
    p.add_argument('--hyp', required=True, help='Path to generated CSV to compare')
    p.add_argument('--out', default='compare_out.csv', help='Output CSV with alignment')
    return p.parse_args()


def main():
    args=parse_args()
    refs = read_transcript_csv(args.ref)
    hyps = read_transcript_csv(args.hyp)
    overall, df = compare(refs, hyps)
    print(f'Aggregate WER: {overall:.3f}')
    df.to_csv(args.out, index=False)
    print(f'Wrote detailed alignment to {args.out}')


if __name__=='__main__':
    main()
