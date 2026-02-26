#!/usr/bin/env python3
"""Batch experiment runner.
Processes audio files in data/raw/dataset/, uses a sliding-window approach to safely
produce transcripts for logical 5-minute windows while keeping model inputs small.
Writes transcripts and evaluation artifacts to experiments/results/.
"""
import argparse
import os
from pathlib import Path
import soundfile as sf
import math
import re
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw/dataset', help='Unpacked dataset dir')
    p.add_argument('--out-dir', default='experiments/results', help='Where to save outputs')
    p.add_argument('--model', default='small')
    p.add_argument('--device', default=None)
    p.add_argument('--logical-window', type=int, default=300, help='Logical window (sec) e.g. 300')
    p.add_argument('--physical-chunk', type=int, default=30, help='Physical chunk size in seconds')
    p.add_argument('--stride', type=int, default=15, help='Stride in seconds for overlapping chunks')
    p.add_argument('--max-windows', type=int, default=0, help='Max logical windows to process per file (0 = all)')
    p.add_argument('--merge-tol', type=float, default=0.5, help='Merge tolerance (s) for joining adjacent segments')
    p.add_argument('--max-files', type=int, default=0, help='Max number of files to process (0 = all)')
    p.add_argument('--beam-size', type=int, default=5)
    p.add_argument('--enhance', action='store_true')
    return p.parse_args()


def sliding_ranges(start, end, chunk, stride):
    t = start
    while t < end:
        yield t, min(t + chunk, end)
        t += stride


def merge_segments(segments, merge_tol=0.5):
    """Simple greedy merge of overlapping/adjacent segments into a coherent transcript."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s['start'])
    out = []
    cur = segs[0].copy()
    for s in segs[1:]:
        if s['start'] <= cur['end'] + merge_tol:
            # join
            cur['end'] = max(cur['end'], s['end'])
            if s.get('text'):
                if cur.get('text') and cur['text'].strip():
                    cur['text'] = (cur['text'].strip() + ' ' + s['text'].strip()).strip()
                else:
                    cur['text'] = s['text']
            cur['confidence'] = np.nanmean([x for x in [cur.get('confidence'), s.get('confidence')] if x is not None])
        else:
            out.append(cur)
            cur = s.copy()
    out.append(cur)
    return out


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # lazy import whisperx to avoid heavy deps until needed
    import whisperx

    device = args.device
    try:
        import torch
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        device = device or 'cpu'

    print('Loading model', args.model, 'on', device)
    # Use a NoopVad similar to runner to avoid pyannote unpickle
    import whisperx.vads as _vads
    from types import SimpleNamespace

    class NoopVad(_vads.Vad):
        def __init__(self, chunk_size=30, vad_onset=0.5):
            super().__init__(vad_onset)
            self.chunk_size = float(chunk_size)

        @staticmethod
        def preprocess_audio(audio):
            return audio

        def __call__(self, inputs):
            waveform = inputs.get('waveform')
            sr = inputs.get('sample_rate', 16000)
            length_sec = float(len(waveform)) / float(sr) if waveform is not None else 0.0
            nchunks = max(1, math.ceil(length_sec / self.chunk_size))
            segs = []
            for i in range(nchunks):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, length_sec)
                segs.append(SimpleNamespace(start=start, end=end, speaker=None))
            return segs

    vad_model = NoopVad(chunk_size=args.physical_chunk)
    asr_options = {
        'beam_size': args.beam_size,
        'word_timestamps': False,
        'condition_on_previous_text': True,
        'temperatures': [0.0],
    }

    model = whisperx.load_model(args.model, compute_type='float32', device=device, vad_model=vad_model, vad_options={'use_vad': False}, asr_options=asr_options, language='en')

    files = sorted([p for p in data_dir.rglob('*') if p.suffix.lower() in ['.wav','.mp3','.m4a']])
    if not files:
        print('No audio files found in', data_dir)
        return

    if args.max_files > 0:
        files = files[:args.max_files]

    for audio_path in files:
        print('Processing', audio_path)
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if getattr(audio, 'ndim', 1) > 1:
            audio = audio.mean(axis=1)
        total_dur = len(audio) / sr

        base = audio_path.stem
        all_segments = []

        # iterate logical windows
        logical_step = args.logical_window
        phys_chunk = args.physical_chunk
        stride = args.stride
        window_count = 0
        for wstart in range(0, math.ceil(total_dur / logical_step) * logical_step, logical_step):
            wend = min(wstart + logical_step, total_dur)
            # honor max_windows if set
            if args.max_windows > 0 and window_count >= args.max_windows:
                break
            window_count += 1
            # create overlapping physical chunks inside this logical window
            for (s,e) in sliding_ranges(wstart, wend, phys_chunk, stride):
                sidx = int(s * sr)
                eidx = int(e * sr)
                chunk_audio = audio[sidx:eidx]
                # enhance optionally
                if args.enhance:
                    try:
                        import noisereduce as nr
                        import librosa
                        chunk_audio = nr.reduce_noise(y=chunk_audio, sr=sr, stationary=True, prop_decrease=0.75)
                        chunk_audio = librosa.util.normalize(chunk_audio)
                    except Exception:
                        pass

                # transcribe chunk
                try:
                    result = model.transcribe(chunk_audio, chunk_size=phys_chunk)
                except Exception as err:
                    print('Error transcribing chunk', err)
                    continue

                for seg in result.get('segments', []):
                    # adjust relative times to absolute; skip empty text segments
                    text = seg.get('text', '')
                    if text is None:
                        continue
                    text = str(text).strip()
                    if not text:
                        continue
                    all_segments.append({
                        'start': float(s + seg.get('start', 0.0)),
                        'end': float(s + seg.get('end', 0.0)),
                        'text': text,
                        'confidence': seg.get('avg_logprob', None)
                    })

            merged = merge_segments(all_segments, merge_tol=args.merge_tol)
            # write transcript CSV compatible with evaluator (use column 'Transcript')
            out_csv = out_dir / f"{base}_transcript.csv"
            pd.DataFrame([{'Transcript': s.get('text',''), 'start': s.get('start'), 'end': s.get('end')} for s in merged]).to_csv(out_csv, index=False)
            print('Wrote', out_csv)

        # find matching human transcript (search by stem)
        human = None
        human_csv = None
        for ex in data_dir.rglob('*'):
            if ex.stem == base:
                if ex.suffix.lower() in ['.csv', '.txt']:
                    human = ex
                    break
                if ex.suffix.lower() == '.cha':
                    human = ex
                    break

        if human and human.suffix.lower() == '.cha':
            # parse .cha CHAT file into a simple CSV of utterances for evaluation
            def parse_cha_to_csv(cha_path, out_csv_path):
                utterances = []
                with open(cha_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        # speaker lines typically start with '*' (e.g., '*CHI:')
                        if not line:
                            continue
                        if line.startswith('*'):
                            # remove speaker tier marker
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                text = parts[1].strip()
                                # remove CHAT annotations in brackets or parens, and tags
                                text = re.sub(r'\[.*?\]', '', text)
                                text = re.sub(r'\(.*?\)', '', text)
                                text = re.sub(r'<.*?>', '', text)
                                text = re.sub(r'[^\S\r\n]+', ' ', text).strip()
                                if text:
                                    utterances.append({'Transcript': text})
                if utterances:
                    pd.DataFrame(utterances).to_csv(out_csv_path, index=False)
                    return out_csv_path
                return None

            human_csv = out_dir / f"{base}_human.csv"
            parsed = parse_cha_to_csv(human, human_csv)
            if parsed:
                human = parsed
                print('Parsed .cha to', human)
            else:
                print('Failed to parse .cha for', base)

        if human:
            # evaluate
            cmp_out = out_dir / f"{base}_compare.csv"
            # call local evaluator
            import subprocess
            subprocess.run([os.path.expanduser('~/WhisperX/venv/bin/python3'), str(Path(__file__).parent / 'evaluate_transcripts.py'), '--ref', str(human), '--hyp', str(out_csv), '--out', str(cmp_out)])
            print('Wrote compare', cmp_out)
            # plot
            plot_out = out_dir / f"{base}_wer.png"
            subprocess.run([os.path.expanduser('~/WhisperX/venv/bin/python3'), str(Path(__file__).parent / 'visualize_comparison.py'), '--compare_csv', str(cmp_out), '--out', str(plot_out)])
            print('Wrote plot', plot_out)
        else:
            print('No human transcript found for', base)


if __name__ == '__main__':
    main()
