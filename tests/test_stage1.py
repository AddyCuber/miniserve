import pytest

from miniserve.hf_baseline import generate_naive, load

PROMPTS = [
    'The capital of France is',
    'def fibonacci(n):',
    'Once upon a time',
    'Inference engineering is about making language models',
    '1 + 1 =',
]


@pytest.fixture(scope='module')
def parts():
    return load(device='cpu')


@pytest.mark.parametrize('prompt', PROMPTS)
def test_naive_matches_hf_generate(parts, prompt):
    tok, model, _ = parts
    ours = generate_naive(model, tok, prompt, max_new_tokens=40, device='cpu')
    ids = tok(prompt, return_tensors='pt').input_ids
    ref = model.generate(ids, max_new_tokens=40, do_sample=False, repetition_penalty=1.0)
    assert ours.tolist() == ref.tolist()
