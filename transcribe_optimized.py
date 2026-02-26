# Public function: transcribe a single file with VAD and decode params
def transcribe_file_with_vad_params(
    audio_path: str,
    output_csv: str,
    model=None,
    device: str = None,
    vad_top_db: int = 30,
    vad_pad_ms: int = 100,
    min_voiced_ms: int = 100,
    chunk_duration: int = None,
    decode_options: dict = None,
):
    """
    Transcribe one file with specific VAD parameters and decoding options.
    Uses librosa VAD to detect voiced intervals, then transcribes with model.
    Returns: path to output CSV (same as output_csv)
    """
    import soundfile as sf
    import librosa
    import pandas as pd
    from pathlib import Path

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # load or reuse model
    mdl = model or load_whisper_model(device=device)
    if chunk_duration is None:
        chunk_duration = CHUNK_DURATION

    # load audio
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
    except Exception:
        audio, sr = librosa.load(audio_path, sr=16000)

    # detect voiced intervals using librosa VAD (energy-based)
    intervals = librosa.effects.split(audio, top_db=vad_top_db)
    pad_samples = int((vad_pad_ms / 1000.0) * sr)
    rows = []
    
    for s, e in intervals:
        s0 = max(0, s - pad_samples)
        e0 = min(len(audio), e + pad_samples)
        dur_ms = (e0 - s0) / sr * 1000
        if dur_ms < min_voiced_ms:
            continue
        seg = audio[s0:e0]
        
        # optionally split long voiced segments into chunk_duration subchunks
        seg_len_s = (e0 - s0) / sr
        if seg_len_s > chunk_duration * 1.5:
            # split into subchunks
            num_sub = max(1, int(seg_len_s // chunk_duration) + 1)
            for j in range(num_sub):
                ss = int(j * chunk_duration * sr)
                ee = int(min(len(seg), (j + 1) * chunk_duration * sr))
                sub = seg[ss:ee]
                # Transcribe using fallback openai-whisper (handles decode_options)
                try:
                    res = mdl.transcribe(sub)
                except Exception:
                    # fallback to openai-whisper with decode options
                    try:
                        import whisper as _whisper_base
                        if not hasattr(transcribe_file_with_vad_params, '_whisper_base'):
                            transcribe_file_with_vad_params._whisper_base = _whisper_base.load_model('tiny', device=device)
                        whisper_model = transcribe_file_with_vad_params._whisper_base
                        res = whisper_model.transcribe(sub, **(decode_options or {}))
                    except Exception as exc:
                        print(f'fallback whisper transcription failed: {exc}')
                        continue
                
                for segd in res.get("segments", []):
                    raw = segd.get("text", "").strip()
                    cleaned = postprocess_transcript(raw)
                    flags, score = generate_hallucination_flags(cleaned)
                    rows.append({
                        "start": (s0 + ss) / sr + segd.get("start", 0.0),
                        "end": (s0 + ss) / sr + segd.get("end", 0.0),
                        "text": cleaned,
                        "flags": ";".join(flags) if flags else "",
                        "hallucination_score": score,
                    })
        else:
            # Transcribe single voiced segment
            try:
                res = mdl.transcribe(seg)
            except Exception:
                # fallback to openai-whisper with decode options
                try:
                    import whisper as _whisper_base
                    if not hasattr(transcribe_file_with_vad_params, '_whisper_base'):
                        transcribe_file_with_vad_params._whisper_base = _whisper_base.load_model('tiny', device=device)
                    whisper_model = transcribe_file_with_vad_params._whisper_base
                    res = whisper_model.transcribe(seg, **(decode_options or {}))
                except Exception as exc:
                    print(f'fallback whisper transcription failed: {exc}')
                    continue
            
            for segd in res.get("segments", []):
                raw = segd.get("text", "").strip()
                cleaned = postprocess_transcript(raw)
                flags, score = generate_hallucination_flags(cleaned)
                rows.append({
                    "start": s0 / sr + segd.get("start", 0.0),
                    "end": s0 / sr + segd.get("end", 0.0),
                    "text": cleaned,
                    "flags": ";".join(flags) if flags else "",
                    "hallucination_score": score,
                })

    # collapse consecutive duplicates and save
    def collapse_consecutive_duplicates_df(df):
        if df.empty:
            return df
        out = [df.iloc[0].to_dict()]
        for i in range(1, len(df)):
            prev = out[-1]
            cur = df.iloc[i].to_dict()
            if prev['text'].strip().lower() == cur['text'].strip().lower():
                prev['end'] = cur['end']
            else:
                out.append(cur)
        return pd.DataFrame(out)
    
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["start", "end", "text"])
    df = collapse_consecutive_duplicates_df(df)
    outp = Path(output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    return str(outp)
#!/usr/bin/env python3
"""Optimized WhisperX transcription pipeline

Purpose (what changed vs the baseline `transcribe_original.py`):
- Structured code into small functions: model loader, directory processor,
  chunked audio processor, and post-processing.
- Added conservative decoding options to discourage repetition (low
  temperature, small beam_size).
- Added a post-processing step `collapse_repeated_words` that detects
  runs of repeated tokens (e.g. "brown brown brown") and collapses
  them to a single or limited number of repetitions.
- Exposes parameters at the top so you can tune chunking and repetition
  thresholds without digging into code.

This file intentionally keeps the same high-level behavior as the
baseline (load model once, process files in chunks, write CSVs) but
applies small changes to reduce hallucinated repetition and make the
flow easier to follow.
"""
import os
import argparse
from datetime import timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import torch

try:
    import whisperx
except Exception:
    whisperx = None

# ---------------------- Tunable parameters ----------------------
CHUNK_DURATION = 300              # seconds per chunk (same as baseline)
MIN_REPEAT_RUN = 3                # run length considered hallucination
MAX_REPEAT_KEEP = 1               # collapse runs to this many tokens
DECODE_OPTIONS = {
    "temperature": 0.0,
    "beam_size": 1,
    "best_of": 1,
}
VAD_TOP_DB = 30                    # librosa.effects.split threshold
VAD_PAD = 0.1                      # seconds to pad around voiced regions

# ---------------------- Utilities --------------------------------
def format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def collapse_repeated_words(text: str, min_run: int = MIN_REPEAT_RUN, keep: int = MAX_REPEAT_KEEP) -> str:
    """Collapse repeated token runs.

    Example: 'brown brown brown green' -> 'brown green' when min_run=3, keep=1
    """
    if not text:
        return text
    toks = text.split()
    out = []
    i = 0
    n = len(toks)
    while i < n:
        # count run length of identical token starting at i
        j = i + 1
        while j < n and toks[j].lower() == toks[i].lower():
            j += 1
        run_len = j - i
        if run_len >= min_run:
            # keep at most `keep` repetitions
            out.extend([toks[i]] * keep)
        else:
            out.extend(toks[i:j])
        i = j
    return " ".join(out)


def postprocess_transcript(text: str) -> str:
    """Apply post-processing heuristics to suppress likely hallucinations.

    Current steps:
    - Collapse long runs of exact repeated tokens.
    - Trim whitespace and normalize simple punctuation spacing.
    Additional heuristics (confidence filtering, profanity removal, etc.)
    can be added here later.
    """
    if text is None:
        return ""
    txt = text.strip()
    txt = collapse_repeated_words(txt)
    # collapse multiple spaces
    txt = " ".join(txt.split())
    return txt


def _detect_non_latin(text: str) -> bool:
    """Return True if text contains non-ASCII / non-Latin characters."""
    if not text:
        return False
    # A simple heuristic: flag any character with codepoint > 127
    return any(ord(ch) > 127 for ch in text)


def _long_word_present(text: str, length_threshold: int = 30) -> bool:
    """Flag if any token exceeds length_threshold characters."""
    if not text:
        return False
    return any(len(tok) >= length_threshold for tok in text.split())


def _nonalpha_ratio(text: str) -> float:
    """Return fraction of characters that are non-alphanumeric (excluding whitespace)."""
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    nonalpha = sum(1 for c in chars if not c.isalnum())
    return nonalpha / len(chars)


def generate_hallucination_flags(text: str) -> tuple[list[str], float]:
    """Return (flags, score) for a transcript text.

    Flags: list of short tags (e.g. 'repeated_words', 'non_latin', ...)
    Score: 0..1 numeric indicating how suspicious the segment is.
    """
    flags: list[str] = []
    score = 0.0

    # repeated words heuristic: run-length collapse changes text
    collapsed = collapse_repeated_words(text)
    if collapsed.strip().lower() != (text or "").strip().lower():
        flags.append("repeated_words")
        score += 1.0

    if _detect_non_latin(text):
        flags.append("non_latin_chars")
        score += 1.0

    if _long_word_present(text):
        flags.append("long_token")
        score += 0.5

    if _nonalpha_ratio(text) > 0.5:
        flags.append("high_nonalpha_ratio")
        score += 0.75

    # normalize score to [0,1]
    max_possible = 1.0 + 1.0 + 0.5 + 0.75
    score = min(1.0, score / max_possible)
    return flags, float(score)


# ---------------------- Audio / VAD helpers ----------------------
def _load_audio_array(path: str):
    """Load audio as a mono numpy array at 16 kHz. Uses soundfile and librosa as fallbacks."""
    import tempfile
    import shutil

    # Try soundfile first (works for WAV/FLAC/others)
    try:
        import soundfile as sf
        a, sr = sf.read(path, dtype='float32')
        if getattr(a, 'ndim', 1) > 1:
            a = a.mean(axis=1)
    except Exception:
        a = None
        sr = None

    # Try librosa as a fallback
    if a is None:
        try:
            import librosa
            a, sr = librosa.load(path, sr=16000, mono=True)
        except Exception:
            a = None
            sr = None

    # Try torchaudio next (can handle some formats if compiled with ffmpeg)
    if a is None:
        try:
            import torchaudio
            wav, sr_t = torchaudio.load(path)
            import numpy as _np
            if wav.ndim > 1:
                wav = wav.mean(dim=0)
            a = wav.numpy().astype('float32')
            sr = int(sr_t)
        except Exception:
            a = None
            sr = None

    # Final fallback: invoke a local ffmpeg (if present) to convert to 16k WAV
    if a is None:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            tmp_wav = None
            try:
                fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                cmd = [ffmpeg_path, '-y', '-i', path, '-ar', '16000', '-ac', '1', tmp_wav]
                import subprocess
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                import soundfile as sf
                a, sr = sf.read(tmp_wav, dtype='float32')
                if getattr(a, 'ndim', 1) > 1:
                    a = a.mean(axis=1)
            except Exception as exc:
                raise RuntimeError(f"Unable to load audio via ffmpeg fallback: {exc}")
            finally:
                try:
                    if tmp_wav and os.path.exists(tmp_wav):
                        os.remove(tmp_wav)
                except Exception:
                    pass
        else:
            # If whisperx is available, try its loader as an additional fallback
            try:
                a = whisperx.load_audio(path)
                sr = 16000
            except Exception as exc:
                raise RuntimeError(f"Unable to load audio: no suitable loader found (install ffmpeg or use WAV input). Original error: {exc}")

    # ensure sample rate is 16000
    if sr != 16000:
        try:
            import librosa
            a = librosa.resample(a.astype('float32'), orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception:
            pass
    return a, sr


def _detect_voiced_intervals(audio: np.ndarray, sr: int, top_db: int = 30, pad: float = 0.1):
    """Return list of (start_sec, end_sec) voiced intervals using librosa.effects.split."""
    try:
        import librosa
    except Exception:
        raise RuntimeError('librosa required for VAD splitting; install librosa in the venv')

    intervals = librosa.effects.split(audio, top_db=top_db)
    pad_samples = int(pad * sr)
    out = []
    for s, e in intervals:
        s0 = max(0, s - pad_samples)
        e0 = min(len(audio), e + pad_samples)
        out.append((s0 / sr, e0 / sr))
    return out


# ---------------------- Model / transcription --------------------
def load_whisper_model(model_name: str = "small", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if whisperx is None:
        raise RuntimeError("whisperx not available in this environment")

    print(f"Loading WhisperX model '{model_name}' on {device} with decode options: {DECODE_OPTIONS}")
    asr_options = {
        "beam_size": DECODE_OPTIONS.get("beam_size", 1),
        "word_timestamps": False,
        "condition_on_previous_text": True,
        # whisperx expects temperatures as a list for sampling schedule
        "temperatures": [DECODE_OPTIONS.get("temperature", 0.0)],
    }

    # Defensive: monkeypatch whisperx pyannote loader to avoid torch.load/unpickle issues
    try:
        import whisperx.vads.pyannote as _pyvad

        class _NoopPyannote:
            class _EmptyTimeline:
                def __init__(self):
                    self.data = []

            def __init__(self, *a, **k):
                pass

            def get_timeline(self, *a, **k):
                return _NoopPyannote._EmptyTimeline()

            # whisperx sometimes calls the pyannote pipeline as a callable
            # (e.g., vad_pipeline(audio)). Provide a callable that returns
            # an empty timeline / no-speech result.
            def __call__(self, *a, **k):
                return _NoopPyannote._EmptyTimeline()

        def _noop_load_vad_model(*a, **k):
            return _NoopPyannote()

        _pyvad.load_vad_model = _noop_load_vad_model
        _pyvad.Pyannote = _NoopPyannote
    except Exception:
        pass

    try:
        model = whisperx.load_model(model_name, compute_type="float32", device=device, asr_options=asr_options, language='en')
        model._is_whisper_fallback = False
        return model
    except Exception as e:
        # Graceful fallback to openai-whisper if whisperx/model loading fails
        print(f"whisperx.load_model failed: {type(e).__name__} {e}. Falling back to openai-whisper model.")
        try:
            import whisper as _whisper_base
            _whisper_m = _whisper_base.load_model(model_name, device=device)

            class _WhisperFallbackAdapter:
                def __init__(self, whisper_model):
                    self._wm = whisper_model
                    self._is_whisper_fallback = True

                def transcribe(self, audio, **kwargs):
                    # openai-whisper accepts file paths or numpy arrays
                    return self._wm.transcribe(audio, **kwargs)

            adapter = _WhisperFallbackAdapter(_whisper_m)
            return adapter
        except Exception as e2:
            raise RuntimeError(f"Both whisperx and fallback whisper failed: {e2}")


def process_audio_in_chunks_optimized(audio_path: str, model, chunk_duration: int = CHUNK_DURATION):
    print(f"\nProcessing (optimized) file: {os.path.basename(audio_path)}")
    # load audio array (with soundfile fallback)
    audio, sr = _load_audio_array(audio_path)

    # detect voiced intervals using librosa (VAD via energy threshold)
    intervals = _detect_voiced_intervals(audio, sr, top_db=VAD_TOP_DB, pad=VAD_PAD)

    all_transcripts = []
    # process each voiced interval; split long voiced regions into subchunks
    for seg_i, (s_sec, e_sec) in enumerate(intervals):
        seg_dur = e_sec - s_sec
        # split into subchunks of at most chunk_duration
        n_sub = max(1, int(np.ceil(seg_dur / chunk_duration)))
        for k in range(n_sub):
            sub_s = s_sec + k * chunk_duration
            sub_e = min(e_sec, sub_s + chunk_duration)
            start_idx = int(sub_s * sr)
            end_idx = int(sub_e * sr)
            audio_chunk = audio[start_idx:end_idx]

            # transcribe chunk using model, fallback to openai-whisper if needed
            try:
                res = model.transcribe(audio_chunk)
            except Exception:
                try:
                    import whisper as _whisper_base
                    if not hasattr(process_audio_in_chunks_optimized, '_whisper_base'):
                        process_audio_in_chunks_optimized._whisper_base = _whisper_base.load_model('tiny', device='cpu')
                    whisper_model = process_audio_in_chunks_optimized._whisper_base
                    res = whisper_model.transcribe(audio_chunk, beam_size=DECODE_OPTIONS.get('beam_size', 1), language='en', temperature=DECODE_OPTIONS.get('temperature', 0.0))
                except Exception as exc:
                    print('fallback whisper transcription failed:', exc)
                    res = None

            if not res:
                continue

            for segment in res.get("segments", []):
                # segment times are relative to the chunk
                adjusted_start = sub_s + segment.get("start", 0.0)
                adjusted_end = sub_s + segment.get("end", 0.0)
                raw_text = segment.get("text", "").strip()
                cleaned = postprocess_transcript(raw_text)

                if not cleaned:
                    continue

                flags, score = generate_hallucination_flags(cleaned)

                all_transcripts.append({
                    "Chunk": seg_i + 1,
                    "Chunk Start": format_timestamp(s_sec),
                    "Chunk End": format_timestamp(e_sec),
                    "Start Time": format_timestamp(adjusted_start),
                    "End Time": format_timestamp(adjusted_end),
                    "Start Seconds": adjusted_start,
                    "End Seconds": adjusted_end,
                    "Transcript": cleaned,
                    "flags": ";".join(flags) if flags else "",
                    "hallucination_score": score,
                })

    return all_transcripts


def process_directory(parent_dir: str, output_dir: str, model_name: str = "small"):
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    model = load_whisper_model(model_name)

    for filename in sorted(os.listdir(parent_dir)):
        if not filename.lower().endswith((".wav", ".mp3", ".m4a")):
            continue

        file_path = os.path.join(parent_dir, filename)
        base_name = os.path.splitext(filename)[0]

        try:
            transcript_data = process_audio_in_chunks_optimized(file_path, model)

            # Merge consecutive identical transcripts to reduce repeated lines
            def collapse_consecutive_duplicates(rows):
                if not rows:
                    return rows
                out = []
                prev = rows[0].copy()
                for cur in rows[1:]:
                    prev_txt = (prev.get('Transcript') or '').strip().lower()
                    cur_txt = (cur.get('Transcript') or '').strip().lower()
                    if prev_txt and cur_txt and prev_txt == cur_txt:
                        # extend prev end time to cur end time
                        prev['Chunk End'] = cur.get('Chunk End', prev.get('Chunk End'))
                        prev['End Time'] = cur.get('End Time', prev.get('End Time'))
                        prev['End Seconds'] = cur.get('End Seconds', prev.get('End Seconds'))
                    else:
                        out.append(prev)
                        prev = cur.copy()
                out.append(prev)
                return out

            transcript_data = collapse_consecutive_duplicates(transcript_data)
            df = pd.DataFrame(transcript_data)
            df = df.sort_values(by="Start Seconds")
            df.drop(columns=[c for c in ["Start Seconds", "End Seconds"] if c in df.columns], inplace=True)
            output_csv = outp / f"transcriptions_{base_name}_optimized.csv"
            df.to_csv(output_csv, index=False)
            print(f"Saved optimized transcript: {output_csv}")
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="small")
    return p.parse_args()


def main():
    args = parse_args()
    process_directory(args.input_dir, args.output_dir, model_name=args.model)


if __name__ == "__main__":
    main()
