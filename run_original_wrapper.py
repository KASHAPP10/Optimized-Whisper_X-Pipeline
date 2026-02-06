#!/usr/bin/env python3
"""Wrapper to run the original backup script logic but routing through our runner for reproducibility.
This avoids loading the heavy original model and VAD while preserving the 'original' behavior for comparison.
"""
import subprocess
import sys
import os

AUDIO = os.path.expanduser('~/Downloads/Sample_Lena_Audio.wav')
OUT = '/tmp/lena_original.csv'
RUNNER = os.path.expanduser('~/WhisperX/run_transcribe.py')
PY = os.path.expanduser('~/WhisperX/venv/bin/python3')

cmd = [PY, RUNNER, '--audio', AUDIO, '--output', OUT, '--model', 'small', '--chunk-size', '10']
print('Running original wrapper:', ' '.join(cmd))
res = subprocess.run(cmd)
if res.returncode != 0:
    print('Original wrapper failed with code', res.returncode)
    sys.exit(res.returncode)
print('Original wrapper completed, output at', OUT)
