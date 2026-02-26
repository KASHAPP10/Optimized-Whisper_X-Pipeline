#!/usr/bin/env python3
"""Transcribe only voiced regions after trimming silence with librosa.
Saves a CSV of segments with start/end/text/confidence.

Usage example:
  python src/run_transcribe_silence_trim.py \
    --audio /tmp/A787_001107_30s.wav \
    --output /tmp/opt_silencetrim_30s.csv \
    --model small --device cpu --beam-size 1 --top-db 30 --pad 0.1
"""
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='small')
    p.add_argument('--device', default=None)
    p.add_argument('--beam-size', type=int, default=1)
    p.add_argument('--chunk-size', type=int, default=30)
    p.add_argument('--top-db', type=int, default=30, help='librosa.effects.split top_db silence threshold')
    p.add_argument('--pad', type=float, default=0.0, help='pad seconds to add around voiced segments')
    return p.parse_args()


def main():
    args = parse_args()
    import soundfile as sf

    # device selection
    try:
        import torch
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        device = args.device or 'cpu'

    print(f"Preparing silence-trim transcription of {args.audio} (device={device})")

    # Prefer OpenAI whisper for direct segment transcription (if installed).
    whisper_base = None
    try:
        import whisper as _whisper_base
        whisper_base = _whisper_base.load_model(args.model, device=device)
        print('Using OpenAI whisper for segment transcription')
    except Exception:
        whisper_base = None

    # Also try to import whisperx but avoid letting it instantiate pyannote VAD
    whisperx_model = None
    try:
        import whisperx
        try:
            # monkeypatch whisperx pyannote loader to avoid torch.load/unpickle issues
            import whisperx.vads.pyannote as _pyvad

            class _NoopPyannote:
                def __init__(self, *a, **k):
                    pass

                def get_timeline(self, *a, **k):
                    return None

            def _noop_load_vad_model(*a, **k):
                return _NoopPyannote()

            _pyvad.load_vad_model = _noop_load_vad_model
            _pyvad.Pyannote = _NoopPyannote
        except Exception:
            # different whisperx version or internals -- ignore
            pass

        asr_options = {
            'beam_size': args.beam_size,
            'word_timestamps': False,
            'condition_on_previous_text': True,
            'temperatures': [0.0],
        }
        try:
            whisperx_model = whisperx.load_model(args.model, compute_type='float32', device=device, asr_options=asr_options, language='en')
            print('Loaded whisperx model (for optional alignment)')
        except Exception:
            whisperx_model = None
    except Exception:
        whisperx_model = None

    print('Loading audio:', args.audio)
    audio, sr = sf.read(args.audio, dtype='float32')
    if getattr(audio, 'ndim', 1) > 1:
        audio = audio.mean(axis=1)

    try:
        import librosa
    except Exception:
        print('librosa required for silence trimming. Install librosa in venv.'); raise

    print('Detecting voiced intervals (top_db=', args.top_db, ')')
    intervals = librosa.effects.split(audio, top_db=args.top_db)
    pad_samples = int(args.pad * sr)

    rows = []
    for i, (s, e) in enumerate(intervals):
        s0 = max(0, s - pad_samples)
        e0 = min(len(audio), e + pad_samples)
        seg = audio[s0:e0]
        start_sec = s0 / sr
        end_sec = e0 / sr
        print(f'Transcribing voiced segment {i}: {start_sec:.2f}-{end_sec:.2f} s')

        # Try OpenAI whisper first
        if whisper_base is not None:
            try:
                seg_res = seg
                if sr != 16000:
                    seg_res = librosa.resample(seg.astype('float32'), orig_sr=sr, target_sr=16000)
                # whisper accepts numpy audio and will return segments
                res = whisper_base.transcribe(seg_res, beam_size=args.beam_size, language='en', temperature=0.0)
            except Exception as exc:
                print('whisper transcription failed for segment', i, exc)
                res = None
        else:
            res = None

        # fallback to whisperx model transcribe if available
        if res is None and whisperx_model is not None:
            try:
                res = whisperx_model.transcribe(seg, chunk_size=args.chunk_size)
            except Exception as exc:
                print('whisperx transcription failed for segment', i, exc)
                res = None

        if not res:
            print('no transcription result for segment', i)
            continue

        for segd in res.get('segments', []):
            t0 = start_sec + segd.get('start', 0.0)
            t1 = start_sec + segd.get('end', 0.0)
            txt = segd.get('text', '').strip()
            if not txt:
                continue
            rows.append({'start': t0, 'end': t1, 'text': txt, 'confidence': segd.get('avg_logprob', None)})

    import pandas as pd
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print('Saved', outp)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Transcribe only voiced regions after trimming silence with librosa.
Saves a CSV of segments with start/end/text/confidence.
"""
import argparse
from pathlib import Path
#!/usr/bin/env python3
"""Transcribe only voiced regions after trimming silence with librosa.
Saves a CSV of segments with start/end/text/confidence.
"""
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='small')
    p.add_argument('--device', default=None)
    p.add_argument('--beam-size', type=int, default=1)
    p.add_argument('--chunk-size', type=int, default=30)
    p.add_argument('--top-db', type=int, default=30, help='librosa.effects.split top_db silence threshold')
    p.add_argument('--pad', type=float, default=0.0, help='pad seconds to add around voiced segments')
    return p.parse_args()


def main():
    args = parse_args()
    import whisperx
    import numpy as np
    import soundfile as sf

    # Monkeypatch whisperx pyannote VAD loading to avoid torch.load/unpickle issues
    try:
        import whisperx.vads.pyannote as _pyvad

        class _NoopPyannote:
            def __init__(self, *a, **k):
                pass

            def get_timeline(self, *a, **k):
                return None

        def _noop_load_vad_model(*a, **k):
            return _NoopPyannote()

        _pyvad.load_vad_model = _noop_load_vad_model
        _pyvad.Pyannote = _NoopPyannote
    except Exception:
        # if whisperx internals differ or not installed, ignore
        pass

    # device selection
    try:
        import torch
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        device = args.device or 'cpu'

    print(f"Loading model {args.model} on {device} (conservative decoding)...")
    asr_options = {
        'beam_size': args.beam_size,
        'word_timestamps': False,
        'condition_on_previous_text': True,
        'temperatures': [0.0],
    }
    # Load whisperx model for any downstream alignment if desired, but we'll
    # use the OpenAI `whisper` model to transcribe our pre-split segments to
    # avoid whisperx internal VAD calls.
    try:
        model = whisperx.load_model(args.model, compute_type='float32', device=device, asr_options=asr_options, language='en')
    except Exception:
        model = None

    # load OpenAI whisper as a direct transcriber for numpy segments
    try:
        import whisper as _whisper_base
        whisper_base = _whisper_base.load_model(args.model, device=device)
    except Exception:
        whisper_base = None

    print('Loading audio:', args.audio)
    audio, sr = sf.read(args.audio, dtype='float32')
    if getattr(audio, 'ndim', 1) > 1:
        audio = audio.mean(axis=1)

    # lazy import librosa for splitting
    try:
        import librosa
    except Exception:
        print('librosa required for silence trimming. Install librosa in venv.'); raise

    print('Detecting voiced intervals (top_db=', args.top_db, ')')
    intervals = librosa.effects.split(audio, top_db=args.top_db)
    pad_samples = int(args.pad * sr)

    rows = []
    for i, (s, e) in enumerate(intervals):
        s0 = max(0, s - pad_samples)
        e0 = min(len(audio), e + pad_samples)
        seg = audio[s0:e0]
        start_sec = s0 / sr
        end_sec = e0 / sr
        print(f'Transcribing voiced segment {i}: {start_sec:.2f}-{end_sec:.2f} s')
        # resample to 16 kHz for whisper if needed
        try:
            if whisper_base is None:
                raise RuntimeError('whisper base model not available')

            seg_res = seg
            if sr != 16000:
                seg_res = librosa.resample(seg.astype('float32'), orig_sr=sr, target_sr=16000)

            # whisper expects float32 numpy array at 16 kHz
            res = whisper_base.transcribe(seg_res, beam_size=args.beam_size, language='en', temperature=0.0)
        except Exception as exc:
            print('transcribe failed for segment', i, exc)
            continue

        for segd in res.get('segments', []):
            t0 = start_sec + segd.get('start', 0.0)
            t1 = start_sec + segd.get('end', 0.0)
            txt = segd.get('text', '').strip()
            if not txt:
                continue
            # OpenAI whisper does not provide avg_logprob in same key; use None
            rows.append({'start': t0, 'end': t1, 'text': txt, 'confidence': segd.get('avg_logprob', None)})

    import pandas as pd
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print('Saved', outp)


if __name__ == '__main__':
    main()
