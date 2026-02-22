import os
import argparse
import math
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
from src.model_gpt import T2M_GPT
from src.dataloader import ASLPoseDataset, collate_pose_batch


def ngram_counts(tokens: List[int], n: int):
    return {tuple(tokens[i:i+n]): 1 + 0 for i in range(max(0, len(tokens) - n + 1))}


def clipped_precision(candidate: List[int], reference: List[int], n: int) -> Tuple[float, int, int]:
    cand_ngrams = ngram_counts(candidate, n)
    ref_ngrams = ngram_counts(reference, n)
    match = 0
    total = max(0, len(candidate) - n + 1)
    if total == 0:
        return 0.0, 0, 0
    for ng, c in cand_ngrams.items():
        match += min(c, ref_ngrams.get(ng, 0))
    return match / total, match, total


def corpus_bleu(cands: List[List[int]], refs: List[List[int]], max_n: int = 4, smooth: bool = True) -> float:
    weights = [1.0 / max_n] * max_n
    precisions = []
    for n in range(1, max_n + 1):
        num = 0
        den = 0
        for c, r in zip(cands, refs):
            p, m, t = clipped_precision(c, r, n)
            num += m
            den += t
        if den == 0:
            precisions.append(0.0)
        else:
            if smooth:
                # add-1 smoothing
                precisions.append((num + 1.0) / (den + 1.0))
            else:
                precisions.append(num / den)
    # brevity penalty
    ref_len = sum(len(r) for r in refs)
    cand_len = sum(len(c) for c in cands)
    if cand_len == 0:
        return 0.0
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
    score = bp * math.exp(sum(w * math.log(max(p, 1e-12)) for w, p in zip(weights, precisions)))
    return score


@torch.no_grad()
def evaluate_bleu(vqvae_ckpt: str, gpt_ckpt: str, split: str, max_batches: int, batch_size: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load VQ-VAE (cfg source of truth)
    vq_ck = torch.load(vqvae_ckpt, map_location='cpu', weights_only=False)
    cfg: T2M_Config = vq_ck['cfg']
    vq = VQ_VAE(cfg).to(device)
    vq.load_state_dict(vq_ck['model_state_dict'], strict=False)
    vq.eval()

    # Load GPT
    gpt = T2M_GPT(cfg).to(device)
    gpt_ck = torch.load(gpt_ckpt, map_location='cpu', weights_only=False)
    if 'model_state_dict' in gpt_ck:
        gpt.load_state_dict(gpt_ck['model_state_dict'], strict=False)
    else:
        gpt.load_state_dict(gpt_ck, strict=False)
    gpt.eval()

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

    # Data
    dataset = ASLPoseDataset(data_paths=[os.path.join(cfg.data_root, getattr(cfg, 'dataset_name', 'ASL_gloss'), split)], split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pose_batch, num_workers=0)

    refs: List[List[int]] = []
    cands: List[List[int]] = []

    processed = 0
    for texts, poses, masks in tqdm(loader, desc=f"BLEU on {split}"):
        if texts is None:
            continue
        poses = poses.to(device)
        masks = masks.to(device)
        # ground-truth tokens
        _, gt_indices, _ = vq.encode(poses, masks)  # (B, T_down)
        B, Tdown = gt_indices.shape
        num_valid_poses = masks.sum(dim=1)
        num_valid_tokens = torch.ceil(num_valid_poses / cfg.downsample_rate).long().cpu()
        gt_list = []
        for b in range(B):
            L = int(num_valid_tokens[b].item())
            gt_list.append(gt_indices[b, :L].detach().cpu().tolist())

        # predict tokens (tokenize per-sample to match generation batch size=1)
        sos = cfg.codebook_size
        eos = cfg.codebook_size + 1 if getattr(cfg, 'use_eos_token', True) else None
        pred_list: List[List[int]] = []
        for b in range(B):
            tok = tokenizer([texts[b]], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            max_len = int(num_valid_tokens[b].item())
            gen = torch.full((1, 1), sos, device=device, dtype=torch.long)
            for _ in range(max_len):
                logits = gpt(tok, gen, torch.ones_like(gen, dtype=torch.bool))
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                gen = torch.cat([gen, nxt], dim=1)
                if eos is not None and int(nxt.item()) == eos:
                    break
            seq = gen[0, 1:].detach().cpu().tolist()
            if eos is not None and eos in seq:
                seq = seq[:seq.index(eos)]
            pred_list.append(seq)

        refs.extend(gt_list)
        cands.extend(pred_list)
        processed += 1
        if 0 < max_batches <= processed:
            break

    bleu4 = corpus_bleu(cands, refs, max_n=4, smooth=True)
    print(f"BLEU-4 (corpus): {bleu4:.4f}  on {len(cands)} samples")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT motion tokens with BLEU')
    parser.add_argument('--vqvae_checkpoint', type=str, default='./checkpoints/vqvae_model.pth')
    parser.add_argument('--gpt_checkpoint', type=str, default='./checkpoints/t2m_gpt_model.pth')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    evaluate_bleu(args.vqvae_checkpoint, args.gpt_checkpoint, args.split, args.max_batches, args.batch_size)


if __name__ == '__main__':
    main()


