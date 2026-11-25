"""
Autoregressive Text Generation using Native LittleBit-2 Forward Pass.
Supports single-turn and multi-turn conversation history.
"""

from typing import List, Optional, Tuple
import torch
from src.architecture.model import ARCLB01Model
from src.tokenizer.tokenizer import ARCTokenizer


def build_prompt_ids(
    tokenizer: ARCTokenizer,
    prompt: str,
    history: Optional[List[Tuple[str, str]]] = None,
) -> List[int]:
    ids = [tokenizer.bos_id]
    if history:
        for u_msg, a_msg in history:
            ids += [tokenizer.user_id] + tokenizer.encode(u_msg) + [tokenizer.sep_id, tokenizer.assistant_id]
            ids += tokenizer.encode(a_msg) + [tokenizer.eos_id]
    ids += [tokenizer.user_id] + tokenizer.encode(prompt) + [tokenizer.sep_id, tokenizer.assistant_id]
    return ids


@torch.no_grad()
def generate_response(
    model: ARCLB01Model,
    tokenizer: ARCTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    model.eval()
    device = next(model.parameters(), torch.tensor(0)).device
    for buf in model.buffers():
        device = buf.device
        break

    input_ids = build_prompt_ids(tokenizer, prompt, history=history)
    # Keep tail within model context window leaving room for generation
    max_prompt_len = max(16, model.config.max_seq_len - min(max_new_tokens, 32))
    if len(input_ids) > max_prompt_len:
        input_ids = [tokenizer.bos_id] + input_ids[-(max_prompt_len - 1) :]

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.max_seq_len :]
        logits = model(x_cond)
        next_logits = logits[:, -1, :]
        if temperature > 0.0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(next_logits.argmax(dim=-1).item())

        if next_id == tokenizer.eos_id or next_id == tokenizer.pad_id:
            break
        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)
