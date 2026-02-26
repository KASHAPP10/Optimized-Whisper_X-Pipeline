#!/usr/bin/env python3
"""Time-aligned evaluator: match ASR segments to human utterances by time and compute WER.
Outputs two CSVs: detailed per-human-utterance WER and ASR hallucination flags.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def simple_wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    d = np.zeros((len(r)+1, len(h)+1), dtype=int)
    for i in range(len(r)+1): d[i,0]=i
    for j in range(len(h)+1): d[0,j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[len(r),len(h)]/max(1,len(r))


def parse_time_ms(s):
    # expect form 'start_end' in ms
    try:
        a,b = s.strip().split('_')
        return int(a)/1000.0, int(b)/1000.0
    except Exception:
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--human', required=True)
    p.add_argument('--asr', required=True)
    p.add_argument('--out-align', default='/tmp/compare_time_aligned.csv')
    p.add_argument('--out-halluc', default='/tmp/hallucination_flags.csv')
    args = p.parse_args()

    human = pd.read_csv(args.human)
    asr = pd.read_csv(args.asr)

    # normalize columns
    human_cols = [c.lower() for c in human.columns]
    if 'transcript' in human_cols:
        human_text = human.iloc[:, human_cols.index('transcript')].astype(str).tolist()
        human_times = human.iloc[:, human_cols.index('time_ms')].astype(str).tolist()
    else:
        raise RuntimeError('human CSV must have transcript and time_ms columns')

    asr_cols = [c.lower() for c in asr.columns]
    if 'start' in asr_cols and 'end' in asr_cols and 'text' in asr_cols:
        start_idx = asr_cols.index('start')
        end_idx = asr_cols.index('end')
        text_idx = asr_cols.index('text')
        asr_segs = []
        for r in asr.itertuples(index=False):
            st = float(r[start_idx])
            et = float(r[end_idx])
            tx = str(r[text_idx])
            asr_segs.append({'start': st, 'end': et, 'text': tx})
    else:
        raise RuntimeError('ASR CSV must have start,end,text columns')

    # build index to mark ASR segments overlapped by any human utt
    asr_used = [False]*len(asr_segs)

    rows = []
    for i,(ht,tt) in enumerate(zip(human_times, human_text)):
        s,e = parse_time_ms(ht)
        if s is None:
            continue
        # collect ASR segments with overlap > 0
        overlapping = [seg for seg in asr_segs if not (seg['end'] <= s or seg['start'] >= e)]
        if overlapping:
            for j,seg in enumerate(asr_segs):
                if not (seg['end'] <= s or seg['start'] >= e):
                    asr_used[j]=True
            hyp = ' '.join([seg['text'] for seg in overlapping]).strip()
        else:
            hyp = ''
        wer = simple_wer(str(tt).strip(), hyp)
        rows.append({'human_start': s, 'human_end': e, 'ref': tt, 'hyp': hyp, 'wer': wer, 'overlap_count': len(overlapping)})

    df_align = pd.DataFrame(rows)
    df_align.to_csv(args.out_align, index=False)

    # hallucination flags: ASR segments with no overlap
    halluc_rows = []
    for i,seg in enumerate(asr_segs):
        if not asr_used[i]:
            halluc_rows.append({'start': seg['start'], 'end': seg['end'], 'text': seg['text']})
    pd.DataFrame(halluc_rows).to_csv(args.out_halluc, index=False)

    print('Wrote align:', args.out_align)
    print('Wrote hallucination flags:', args.out_halluc)


if __name__=='__main__':
    main()
