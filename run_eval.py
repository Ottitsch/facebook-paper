#!/usr/bin/env python
"""
Wrapper script to run eval with proper environment setup.
"""

import os
import subprocess
import sys

# Print Python environment info
print("Python executable:", sys.executable)
print("Python version:")
os.system(f'"{sys.executable}" --version')
print("\nInstalled packages:")
os.system(f'"{sys.executable}" -m pip list')

# Run the experiment
script = "experiments/eval_rag_flan_t5.py"
if len(sys.argv) > 1:
    script = sys.argv[1]
    args = sys.argv[2:]
else:
    args = []

print(f"\nRunning: python {script} {' '.join(args)}\n")
result = subprocess.run([sys.executable, script] + args)
sys.exit(result.returncode)