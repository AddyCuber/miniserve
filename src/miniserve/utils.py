import time
import torch

def get_device() -> str:
#checking the device type. Im using an m1 mac so mps

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
#since gpu can execute asynchronously, python can contiue before the gpu is done. so we measure how quickly the work was submitted and not how long it took
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def now(device: str) -> float:
    sync(device)
    return time.perf_counter()


def stop_ids(tok) -> set:
    return {
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"),
    }