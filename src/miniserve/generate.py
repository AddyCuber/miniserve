import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model():
    """Load the tokenizer and language model onto the best available device."""

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading {MODEL_NAME}...")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model = model.to(device)
    model.eval()

    print("Model class:", type(model).__name__)
    print("Device:", next(model.parameters()).device)
    print("Parameter dtype:", next(model.parameters()).dtype)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    print("Model loaded successfully.")

    return model, tokenizer, device


if __name__ == "__main__":
    model, tokenizer, device = load_model()