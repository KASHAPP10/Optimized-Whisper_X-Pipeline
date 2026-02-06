#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--compare_csv', required=True)
    p.add_argument('--out', default='compare_plot.png')
    return p.parse_args()


def main():
    args=parse_args()
    df=pd.read_csv(args.compare_csv)
    # expecting columns: ref, hyp, wer
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['wer'], marker='o')
    plt.xlabel('segment index')
    plt.ylabel('WER')
    plt.title('Per-segment WER (reference vs hypothesis)')
    plt.grid(True)
    plt.savefig(args.out, dpi=150)
    print('Saved plot to', args.out)

if __name__=='__main__':
    main()
