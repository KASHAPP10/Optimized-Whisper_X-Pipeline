#!/usr/bin/env python3
"""Transcription runner using whisperx.
Usage examples:
  python run_transcribe.py --audio /path/to/lena.wav --output /path/to/out.csv --model small --use-vad false
  python run_transcribe.py --audio file.wav --output out.csv --model small --beam-size 5

This script is defensive: it checks for dependencies and prints actionable errors.
"""
import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Path to audio file to transcribe")
    p.add_argument("--output", required=True, help="Path to output CSV")
    p.add_argument("--model", default="small", help="Model id to load (small/base/large-v2)")
    p.add_argument("--device", default=None, help="Device to use: cpu or cuda (auto by default)")
    p.add_argument("--use-vad", default=False, action="store_true", help="Enable VAD (pyannote); may require auth and extra deps")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=30, help="VAD chunk size in seconds for splitting long audio")
    p.add_argument("--word-timestamps", action="store_true", default=False)
    p.add_argument("--enhance", action="store_true", default=False, help="Apply simple enhance_audio before transcription")
    return p.parse_args()


def ensure_deps():
    try:
        import whisperx
        import pandas as pd
        import numpy as np
    except Exception as e:
        print("Missing required dependencies. Run the workspace requirements or install manually:")
        print("pip install -r requirements.txt  # then optionally pip install git+https://github.com/m-bain/whisperX.git")
        raise


def enhance_audio(audio_array, sample_rate=16000):
    # small in-script enhancer to match original pipeline (no external side-effects)
    try:
        import numpy as _np
        import noisereduce as nr
        import librosa
    except Exception:
        return audio_array
    if hasattr(audio_array, 'ndim') and audio_array.ndim > 1:
        audio_array = _np.mean(audio_array, axis=0)
    reduced = nr.reduce_noise(y=audio_array, sr=sample_rate, stationary=True, prop_decrease=0.75)
    normalized = librosa.util.normalize(reduced)
    compressed = _np.tanh(normalized * 2) * 0.8
    return compressed


def run_transcription(args):
    ensure_deps()
    import whisperx
    import pandas as pd
    import numpy as np

    device = args.device or ("cuda" if ("cuda" in __import__('torch').cuda.get_device_name(0).lower() if __import__('torch').cuda.is_available() else False) else "cpu")
    # Fallback simple device choice
    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = args.device or "cpu"

    # Only pass vad_options to whisperx if VAD is explicitly requested.
    # Passing None avoids initializing pyannote/other VAD backends which may
    # require extra dependencies and can trigger unsafe checkpoint unpickling.
    if args.use_vad:
        vad_options = {"use_vad": True}
        vad_model = None
    else:
        # Create a no-op VAD object that implements the minimal Vad interface
        # required by whisperx.FasterWhisperPipeline to avoid importing pyannote.
        # Build a no-op VAD that subclasses the whisperx Vad base so the pipeline
        # uses it without initializing pyannote/silero. It returns a single
        # full-length segment covering the entire waveform.
        import whisperx.vads as _vads
        from types import SimpleNamespace

        class NoopVad(_vads.Vad):
            def __init__(self, chunk_size=30, vad_onset=0.5):
                super().__init__(vad_onset)
                self.chunk_size = float(chunk_size)

            @staticmethod
            def preprocess_audio(audio):
                # whisperx expects waveform in shape (n_samples,)
                return audio

            def __call__(self, inputs):
                # inputs is a dict with 'waveform' and 'sample_rate'
                waveform = inputs.get('waveform')
                sr = inputs.get('sample_rate', 16000)
                length_sec = float(len(waveform)) / float(sr) if waveform is not None else 0.0
                # split into fixed-size segments of chunk_size seconds
                import math
                nchunks = max(1, math.ceil(length_sec / self.chunk_size))
                segs = []
                for i in range(nchunks):
                    start = i * self.chunk_size
                    end = min((i + 1) * self.chunk_size, length_sec)
                    segs.append(SimpleNamespace(start=start, end=end, speaker=None))
                return segs

    vad_options = {"use_vad": False}
    vad_model = NoopVad(chunk_size=args.chunk_size, vad_onset=0.5)
    # Map runner flags into whisperx ASR options
    # ASR options map to whisperx TranscriptionOptions fields (do NOT include 'task' or 'language')
    asr_options = {
        "beam_size": args.beam_size,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": True,
        # useful defaults for temperature sweep and thresholds
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }

    print(f"Loading model {args.model} on {device} (VAD={args.use_vad})...")
    # pass language explicitly to load_model; default ASR options are provided via asr_options
    model = whisperx.load_model(args.model, compute_type="float32", device=device, vad_model=vad_model, vad_options=vad_options, asr_options=asr_options, language="en")

    print(f"Loading audio: {args.audio}")
    # Prefer pure-Python loading for local WAV files to avoid ffmpeg dependency.
    if str(args.audio).lower().endswith('.wav'):
        try:
            import soundfile as sf
            audio, sr = sf.read(args.audio, dtype='float32')
            # convert to mono if needed
            if getattr(audio, 'ndim', 1) > 1:
                audio = audio.mean(axis=1)
            # resample to 16k if necessary
            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except Exception:
                    print(f"Warning: unable to resample from {sr} to 16000; continuing with original sample rate")
            sample_rate = 16000
        except Exception as e:
            print(f"soundfile failed ({e}), falling back to whisperx.load_audio")
            audio = whisperx.load_audio(args.audio)
            sample_rate = 16000
    else:
        audio = whisperx.load_audio(args.audio)
        sample_rate = 16000

    # Run enhancement when requested (may be slower for long audio).
    if args.enhance:
        audio = enhance_audio(audio, sample_rate)

    print("Transcribing...")
    # split into chunks to avoid excessively large inputs
    result = model.transcribe(audio, chunk_size=args.chunk_size)

    segments = result.get("segments", [])
    rows = []
    for seg in segments:
        rows.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", "").strip(),
            "confidence": seg.get("avg_logprob", None),
        })

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved transcript to {out_path}")


def main():
    args = parse_args()
    run_transcription(args)


if __name__ == '__main__':
    main()
