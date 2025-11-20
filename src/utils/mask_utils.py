import torch
from typing import Optional

__all__ = ["get_last_valid_token_index"]


def get_last_valid_token_index(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.LongTensor:
    """
    通用工具函数：根据 attention_mask（支持 2D padding mask 与 4D additive mask）
    计算每个样本最后一个非 PAD token 的索引。
    """
    if attention_mask is None:
        return torch.full(
            (batch_size,),
            seq_len - 1,
            dtype=torch.long,
            device=device,
        )

    if attention_mask.dim() == 4:
        last_row = attention_mask[:, 0, -1, :]
        valid_mask = (last_row == 0)
    elif attention_mask.dim() == 2:
        valid_mask = (attention_mask != 0)
    else:
        raise ValueError(
            f"Unexpected attention_mask.dim={attention_mask.dim()}, expected 2 or 4."
        )

    has_valid = valid_mask.any(dim=-1)
    flipped = torch.flip(valid_mask.to(dtype=torch.long), dims=[1])
    inv_idx = flipped.argmax(dim=-1)
    last_idx = (seq_len - 1) - inv_idx

    last_idx = torch.where(
        has_valid,
        last_idx,
        torch.zeros_like(last_idx),
    )

    return last_idx

