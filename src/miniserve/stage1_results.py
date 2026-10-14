import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
import transformers

from miniserve.hf_baseline import generate_naive, load
from miniserve.utils import get_device

PROMPT_LENS = [16, 128, 256]
NEW_TOKENS = 60
TRIALS = 3
OUT = 'results/stage1_naive_mps_fp32.json'


def prompt_of_len(tok, n):
    text = 'The quick brown fox jumps over the lazy dog. ' * 200
    return tok.decode(tok(text).input_ids[:n])


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return 'unknown'


def main():
    device = get_device()
    tok, model, _ = load()
    generate_naive(model, tok, 'Hello', max_new_tokens=5, device=device, stop=set())   # warmup
    runs = []
    for n in PROMPT_LENS:
        prompt = prompt_of_len(tok, n)
        actual = len(tok(prompt).input_ids)
        for trial in range(TRIALS):
            times = []
            generate_naive(model, tok, prompt, NEW_TOKENS, device, stop=set(), times=times)
            ms = [t * 1e3 for t in times]
            runs.append({
                'prompt_tokens': actual, 'trial': trial, 'step_ms': ms,
                'early_ms': float(np.median(ms[1:11])),    # steps 2 to 11
                'late_ms': float(np.median(ms[-10:])),     # last 10 steps
            })
            print(actual, trial, 'early %.1f ms | late %.1f ms' % (runs[-1]['early_ms'], runs[-1]['late_ms']))
    meta = {
        'device': device, 'dtype': 'fp32', 'new_tokens': NEW_TOKENS, 'trials': TRIALS,
        'chip': sh(['sysctl', '-n', 'machdep.cpu.brand_string']),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'commit': sh(['git', 'rev-parse', '--short', 'HEAD']),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'naive loop, no KV cache, EOS ignored, warmup run discarded',
    }
    os.makedirs('results', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'meta': meta, 'runs': runs}, f)
    print('saved', OUT)

    import matplotlib.pyplot as plt
    for n in sorted({r['prompt_tokens'] for r in runs}):
        steps = np.mean([r['step_ms'] for r in runs if r['prompt_tokens'] == n], axis=0)
        plt.plot(range(1, len(steps) + 1), steps, label='%d token prompt' % n)
    plt.ylim(0, None)
    plt.xlabel('generation step')
    plt.ylabel('ms per step')
    plt.title('Naive loop, no KV cache (%s, fp32)' % device)
    plt.legend()
    plt.savefig(OUT.replace('.json', '.png'), dpi=150)


if __name__ == '__main__':
    main()