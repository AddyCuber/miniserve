import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniserve.sampling import pick
from miniserve.utils import get_device, now, stop_ids

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load(device=None, dtype=torch.float32):
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(dtype).to(device).eval()
    return tok, model, device


@torch.no_grad()
def generate_naive(
    model,
    tok,
    prompt,
    max_new_tokens=50,
    device="cpu",
    temperature=0.0,
    stop=None,
    times=None,
):
    stop = stop_ids(tok) if stop is None else stop
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    for _ in range(max_new_tokens):
        t0 = now(device)
        logits = model(ids).logits
        nxt = pick(logits[:, -1, :], temperature)
        ids = torch.cat([ids, nxt], dim=1)

        if times is not None:
            times.append(now(device) - t0)

        if int(nxt) in stop:
            break

    return ids


if __name__ == "__main__":
    tok, model, device = load()

    msgs = [
        {
            "role": "user",
            "content": "Explain a KV cache in two sentences.",
        }
    ]

    prompt = tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    times = []
    out = generate_naive(
        model,
        tok,
        prompt,
        max_new_tokens=60,
        device=device,
        times=times,
    )

    print(tok.decode(out[0], skip_special_tokens=True))
    print("device:", device, "| steps:", len(times))
    print(
        "step 1: %.1f ms | step 30: %.1f ms | last: %.1f ms"
        % (
            times[0] * 1e3,
            times[min(29, len(times) - 1)] * 1e3,
            times[-1] * 1e3,
        )
    )