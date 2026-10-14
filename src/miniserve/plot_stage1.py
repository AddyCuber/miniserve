import json

import matplotlib.pyplot as plt
import numpy as np

d = json.load(open('results/stage1_naive_mps_fp32.json'))
runs = d['runs']
for n in sorted({r['prompt_tokens'] for r in runs}):
    steps = np.median([r['step_ms'] for r in runs if r['prompt_tokens'] == n], axis=0)
    plt.plot(range(1, len(steps) + 1), steps, label='%d token prompt' % n)
plt.ylim(0, None)
plt.xlabel('generation step')
plt.ylabel('ms per step (median of 3 trials)')
plt.title('Naive loop, no KV cache (MPS, fp32)')
plt.legend()
plt.savefig('results/stage1_naive_mps_fp32.png', dpi=150)

