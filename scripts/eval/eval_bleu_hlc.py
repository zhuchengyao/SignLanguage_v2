"""
Evaluate HLC-NAR model with BLEU-1 ~ BLEU-4.

Compares ground-truth HLC token indices (from HLC encoder on GT poses)
against Token Predictor outputs (from text), reporting per-codebook and
combined BLEU scores.

Usage:
    python scripts/eval/eval_bleu_hlc.py \
        --ckpt ./checkpoints/hlc_nar/stage3_best.pth \
        --split test --max_samples 0
"""
import os
import sys
import math
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BertTokenizer

from src.config_hlc import HLC_NAR_Config
from src.model_unified import HLC_NAR_Model
from src.dataloader import ASLPoseDataset, collate_hlc_batch


CODEBOOK_NAMES = ["global_shape", "local_shape", "global_motion", "local_motion"]


def count_ngrams(tokens: List[int], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(candidate: List[int], reference: List[int], n: int) -> Tuple[int, int]:
    cand_counts = count_ngrams(candidate, n)
    ref_counts = count_ngrams(reference, n)
    clipped = sum(min(cnt, ref_counts.get(ng, 0)) for ng, cnt in cand_counts.items())
    total = max(len(candidate) - n + 1, 0)
    return clipped, total


def corpus_bleu_detailed(
    candidates: List[List[int]],
    references: List[List[int]],
    max_n: int = 4,
) -> Dict[str, float]:
    precisions: List[float] = []
    for n in range(1, max_n + 1):
        total_match, total_count = 0, 0
        for cand, ref in zip(candidates, references):
            m, c = modified_precision(cand, ref, n)
            total_match += m
            total_count += c
        if total_count == 0:
            precisions.append(0.0)
        else:
            precisions.append((total_match + 1.0) / (total_count + 1.0))

    ref_len = sum(len(r) for r in references)
    cand_len = sum(len(c) for c in candidates)
    if cand_len == 0:
        bp = 0.0
    elif cand_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / cand_len)

    result: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        log_avg = 0.0
        valid = True
        for k in range(n):
            if precisions[k] <= 0:
                valid = False
                break
            log_avg += math.log(precisions[k])
        result[f"BLEU-{n}"] = bp * math.exp(log_avg / n) if valid else 0.0

    log_sum = sum(math.log(p) for p in precisions if p > 0)
    all_valid = all(p > 0 for p in precisions)
    result["BLEU"] = bp * math.exp(log_sum / max_n) if all_valid else 0.0
    result["BP"] = bp
    result["ref_len"] = ref_len
    result["cand_len"] = cand_len
    result["precisions"] = precisions
    return result


@torch.no_grad()
def evaluate(ckpt_path: str, split: str, max_samples: int, batch_size: int, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = HLC_NAR_Config()
    model = HLC_NAR_Model(cfg).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    data_path = os.path.join(cfg.data_root, f"ASL_gloss/{split}")
    train_path = os.path.join(cfg.data_root, "ASL_gloss/train")

    train_data = ASLPoseDataset(data_paths=[train_path], split="train", max_seq_len=cfg.max_seq_len)
    dataset = ASLPoseDataset(
        data_paths=[data_path], split=split, max_seq_len=cfg.max_seq_len,
        extern_mean=train_data.pose_mean, extern_std=train_data.pose_std,
    )

    model.pose_mean = torch.from_numpy(train_data.pose_mean).float().to(device)
    model.pose_std = torch.from_numpy(train_data.pose_std).float().to(device)

    if 0 < max_samples < len(dataset):
        import random
        random.seed(42)
        indices = sorted(random.sample(range(len(dataset)), max_samples))
        dataset = Subset(dataset, indices)
        print(f"Subsampled to {max_samples} from {split}")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_hlc_batch, num_workers=0,
    )

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

    all_refs = {k: [] for k in CODEBOOK_NAMES}
    all_cands = {k: [] for k in CODEBOOK_NAMES}
    combined_refs: List[List[int]] = []
    combined_cands: List[List[int]] = []
    per_sample: List[dict] = []

    for batch in tqdm(loader, desc=f"Eval HLC-NAR BLEU ({split})"):
        texts, pose_seq, masks = batch
        if texts is None:
            continue
        pose_seq = pose_seq.to(device)
        masks = masks.to(device)

        tok = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        text_features, text_cls, text_mask = model.encode_text(tok)

        hlc_out = model.hlc.encode(pose_seq, masks)
        gt_indices = hlc_out["indices"]

        B, T = pose_seq.shape[:2]
        valid_lens = masks.sum(dim=1).long().cpu()

        tok_logits = model.token_predictor(text_features, T, text_mask)
        pred_indices = {k: v.argmax(dim=-1) for k, v in tok_logits.items()}

        for b in range(B):
            L = int(valid_lens[b].item())
            sample_info = {"text": texts[b], "ref_len": L}
            combined_ref = []
            combined_cand = []

            for cb_name in CODEBOOK_NAMES:
                ref_seq = gt_indices[cb_name][b, :L].cpu().tolist()
                cand_seq = pred_indices[cb_name][b, :L].cpu().tolist()
                all_refs[cb_name].append(ref_seq)
                all_cands[cb_name].append(cand_seq)
                combined_ref.extend(ref_seq)
                combined_cand.extend(cand_seq)

            combined_refs.append(combined_ref)
            combined_cands.append(combined_cand)
            per_sample.append(sample_info)

    print(f"\nEvaluated {len(combined_refs)} samples")

    report = {"split": split, "num_samples": len(combined_refs)}

    print("\n" + "=" * 65)
    print(f"  HLC-NAR Token BLEU — {split} ({len(combined_refs)} samples)")
    print("=" * 65)

    for cb_name in CODEBOOK_NAMES:
        bleu = corpus_bleu_detailed(all_cands[cb_name], all_refs[cb_name])
        report[cb_name] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in bleu.items()
            if k != "precisions"
        }
        report[cb_name]["precisions"] = [round(p, 6) for p in bleu["precisions"]]
        print(f"\n  [{cb_name}]")
        print(f"    BLEU-1={bleu['BLEU-1']:.4f}  BLEU-2={bleu['BLEU-2']:.4f}  "
              f"BLEU-3={bleu['BLEU-3']:.4f}  BLEU-4={bleu['BLEU-4']:.4f}  "
              f"BP={bleu['BP']:.4f}")

    combined_bleu = corpus_bleu_detailed(combined_cands, combined_refs)
    report["combined"] = {
        k: round(v, 6) if isinstance(v, float) else v
        for k, v in combined_bleu.items()
        if k != "precisions"
    }
    report["combined"]["precisions"] = [round(p, 6) for p in combined_bleu["precisions"]]
    print(f"\n  [COMBINED (all 4 codebooks concatenated)]")
    print(f"    BLEU-1={combined_bleu['BLEU-1']:.4f}  BLEU-2={combined_bleu['BLEU-2']:.4f}  "
          f"BLEU-3={combined_bleu['BLEU-3']:.4f}  BLEU-4={combined_bleu['BLEU-4']:.4f}")
    print(f"    BLEU={combined_bleu['BLEU']:.4f}  BP={combined_bleu['BP']:.4f}")
    print("=" * 65)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HLC-NAR with token-level BLEU")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/hlc_nar/stage3_best.pth")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="./outputs/eval_bleu_hlc/bleu_results.json")
    args = parser.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
