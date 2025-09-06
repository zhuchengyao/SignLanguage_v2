from __future__ import annotations

from dataclasses import dataclass
from typing import List

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


@dataclass
class TextEncConfig:
    model_name: str = "distilbert-base-uncased"  # hidden size 768, lighter
    max_length: int = 77


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextEncConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.cfg.max_length)
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.model(**toks)
        # use CLS token embedding if exists; else mean pool
        hidden = out.last_hidden_state  # (B, L, H)
        feat = hidden[:, 0, :] if hidden.size(1) > 0 else hidden.mean(dim=1)
        return feat


