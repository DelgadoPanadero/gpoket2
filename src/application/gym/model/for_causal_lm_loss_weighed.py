from typing import Optional

import torch
import transformers


def ForCausalLMLossWeighed(  # based on ForCausalLMLoss from transformers.loss.loss_utils.py
    logits: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    weight_token_id=None,
    token_weight=0.3,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:

    logits = logits.logits.float().view(-1, vocab_size)
    labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)

    weights = torch.ones((vocab_size,), device=logits.device)
    if weight_token_id is not None:
        weights[weight_token_id] = token_weight

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        input=logits,
        target=shift_labels,
        weight=weights,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    loss = torch.nan_to_num(loss, nan=0.0)

    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch

    return loss
