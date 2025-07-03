# dataloader_utils.py
"""
Utility: dynamic padding + mask collator
"""

import torch
from typing import List, Tuple


def collate_pose_batch(
    batch: List[Tuple[str, torch.Tensor]]
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Args
    ----
    batch : [(text, pose_seq<T_i,150>), ...], len == B

    Returns
    -------
    texts : list[str]                长度 B
    poses : FloatTensor   (B, T_max, 150)
    mask  : BoolTensor    (B, T_max)   1 = valid, 0 = pad
    """
    texts, seqs = zip(*batch)
    lens = [s.size(0) for s in seqs]
    T_max = max(lens)

    padded, masks = [], []
    for seq, L in zip(seqs, lens):
        pad = T_max - L
        padded.append(
            torch.cat([seq, seq.new_zeros(pad, seq.size(-1))], dim=0)
        )
        masks.append(
            torch.cat([
                torch.ones(L,  dtype=torch.bool),
                torch.zeros(pad, dtype=torch.bool)
            ])
        )
    return list(texts), torch.stack(padded), torch.stack(masks)
