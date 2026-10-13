from miniserve.hf_baseline import load, generate_naive


def test_naive_matches_hf_generate():
    tok, model, device = load(device="cpu")

    prompt = "The capital of France is"

    ours = generate_naive(
        model,
        tok,
        prompt,
        max_new_tokens=20,
        device="cpu",
    )

    ids = tok(prompt, return_tensors="pt").input_ids

    ref = model.generate(
        ids,
        max_new_tokens=20,
        do_sample=False,
        repetition_penalty=1.0,
    )

    assert ours.tolist() == ref.tolist()