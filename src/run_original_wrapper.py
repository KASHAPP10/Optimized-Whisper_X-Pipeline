#!/usr/bin/env python3
import subprocess
import sys
import os

ROOT = Path = None
try:
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
except Exception:
    ROOT = None

AUDIO = os.path.expanduser('~/Downloads/Sample_Lena_Audio.wav')
OUT = '/tmp/lena_original.csv'
RUNNER = os.path.expanduser('~/WhisperX/src/run_transcribe.py')
PY = os.path.expanduser('~/WhisperX/venv/bin/python3')

cmd = [PY, RUNNER, '--audio', AUDIO, '--output', OUT, '--model', 'small', '--chunk-size', '10']
print('Running original wrapper:', ' '.join(cmd))
res = subprocess.run(cmd)
if res.returncode != 0:
    print('Original wrapper failed with code', res.returncode)
    sys.exit(res.returncode)
print('Original wrapper completed, output at', OUT)
