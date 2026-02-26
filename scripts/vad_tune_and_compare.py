#!/usr/bin/env python3
"""Sweep VAD params, run original and optimized decoding, and record WERs."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import os
from pathlib import Path
import re
from transcribe_optimized import load_whisper_model, transcribe_file_with_vad_params, CHUNK_DURATION, DECODE_OPTIONS

def normalize_text(s: str):
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\[.*?\]|\<.*?\>|\(.*?\)", " ", s)   # remove bracketed annotations
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(text):
    return [t for t in normalize_text(text).split() if t]

def concatenated_wer(ref_rows, hyp_rows):
    ref_text = " ".join(ref_rows)
    hyp_text = " ".join(hyp_rows)
    r = tokens(ref_text)
    h = tokens(hyp_text)
    if len(r) == 0:
        return None
    n = len(r)
    m = len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    wer = d[n][m] / float(n)
    return wer

def read_human_csv(path):
    import csv
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(r['transcript'])
    return out

def read_asr_csv(path):
    import csv
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(r.get('text') or r.get('Transcript') or "")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Single audio file to test")
    p.add_argument("--human", required=True, help="Human CSV (same format: transcript,time_ms)")
    p.add_argument("--outdir", default="experiments/vad_tune", help="output dir")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model_opt = load_whisper_model(device=device)

    baseline_decode = {"temperature": 0.2, "beam_size": 5}

    top_db_list = [20, 30, 40]
    pad_ms_list = [0, 50, 100]
    min_voiced_list = [100, 250]

    human_rows = read_human_csv(args.human)

    results = []
    for top_db in top_db_list:
        for pad_ms in pad_ms_list:
            for min_voiced in min_voiced_list:
                label = f"td{top_db}_pad{pad_ms}_min{min_voiced}"
                print("Running:", label)
                # optimized transcribe with VAD params
                opt_out = outdir / f"opt_{label}.csv"
                transcribe_file_with_vad_params(
                    args.audio, str(opt_out),
                    model=model_opt, device=device,
                    vad_top_db=top_db, vad_pad_ms=pad_ms, min_voiced_ms=min_voiced,
                    chunk_duration=CHUNK_DURATION, decode_options=DECODE_OPTIONS
                )
                # baseline: run optimized helper but with baseline decode options and a looser VAD
                base_out = outdir / f"base_{label}.csv"
                transcribe_file_with_vad_params(
                    args.audio, str(base_out),
                    model=model_opt, device=device,
                    vad_top_db=top_db, vad_pad_ms=pad_ms, min_voiced_ms=min_voiced,
                    chunk_duration=CHUNK_DURATION, decode_options=baseline_decode
                )
                ref = human_rows
                hyp_opt = read_asr_csv(str(opt_out))
                hyp_base = read_asr_csv(str(base_out))
                wer_opt = concatenated_wer(ref, hyp_opt) or 1.0
                wer_base = concatenated_wer(ref, hyp_base) or 1.0
                print(f"  base WER={wer_base:.3f}  opt WER={wer_opt:.3f}")
                results.append({"label": label, "original": wer_base, "optimized": wer_opt})
    # save results
    csv_out = outdir / "vad_tune_results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "original", "optimized"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("Saved results to", csv_out)

if __name__ == "__main__":
    main()
