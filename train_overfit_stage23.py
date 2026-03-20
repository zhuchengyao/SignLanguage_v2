"""Resume overfit training from Stage 2 (Stage 1 checkpoint already exists)."""

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.model_unified import HLC_NAR_Model
from train_overfit_15cases import (
    CASE_IDS, DATA_ROOT, make_overfit_cfg,
    OverfitDataset, collate_fn,
    train_stage2, train_stage3,
)


def main():
    cfg = make_overfit_cfg()
    device = cfg.get_device()

    dataset = OverfitDataset(CASE_IDS, DATA_ROOT, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.stage1_batch_size,
                        shuffle=True, collate_fn=collate_fn,
                        num_workers=cfg.num_workers, drop_last=False)

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

    model = HLC_NAR_Model(cfg).to(device)
    model.pose_mean = torch.from_numpy(dataset.pose_mean).float().to(device)
    model.pose_std  = torch.from_numpy(dataset.pose_std).float().to(device)

    train_stage2(model, loader, cfg, device, tokenizer)
    train_stage3(model, loader, cfg, device, tokenizer)

    print("\nStage 2+3 done. Run infer_overfit_15cases.py to generate GIFs.")


if __name__ == "__main__":
    main()
