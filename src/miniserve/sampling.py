import torch

# we are using greedy decoding, picking the single most likely next word
def pick(logits, temperature=0.0, top_k=0, top_p=1.0):
    # logits: (B, vocab) -> returns (B, 1) token ids
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    #temperature scaling
    logits = logits / temperature

    if top_k > 0:
        kth = torch.topk(logits, top_k).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = (cum - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)