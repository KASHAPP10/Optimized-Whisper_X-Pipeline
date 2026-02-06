This workspace contains helper scripts to run and evaluate WhisperX-based transcriptions.

Files:
- `run_transcribe.py` - run a single transcription with options (model size, VAD, beam size, enhancement)
- `evaluate_transcripts.py` - compare generated CSV to a human CSV (00017.csv) and compute WER
- `visualize_comparison.py` - plot per-segment WER from the compare CSV
- `requirements.txt` - minimal dependencies

Quick start:
1) Activate your venv and install requirements:
   source /Users/malhaarkashyap/WhisperX/venv/bin/activate
   pip install -r /Users/malhaarkashyap/WhisperX/requirements.txt

2) Run baseline (small model, no VAD):
   python /Users/malhaarkashyap/WhisperX/run_transcribe.py --audio /path/to/lena.wav --output /tmp/baseline.csv --model small

3) Run a tuned config (enable enhance, increase beam size):
   python /Users/malhaarkashyap/WhisperX/run_transcribe.py --audio /path/to/lena.wav --output /tmp/tuned.csv --model small --enhance --beam-size 8

4) Evaluate against human transcript `00017.csv`:
   python /Users/malhaarkashyap/WhisperX/evaluate_transcripts.py --ref /path/to/00017.csv --hyp /tmp/tuned.csv --out /tmp/compare.csv

5) Visualize:
   python /Users/malhaarkashyap/WhisperX/visualize_comparison.py --compare_csv /tmp/compare.csv --out /tmp/compare.png

Notes:
- For quick iterations set `--model small` or `--model base`. `large-v2` downloads are multi-GB and slow on CPU.
- If you want VAD using pyannote, set `--use-vad` but be aware it may need extra auth and dependencies.
