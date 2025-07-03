# data_loader.py
from __future__ import annotations
import os, json
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader_utils import collate_pose_batch


# ════════════════════════════════════════════════════════════════════════
#                       Dataset: ASLPoseDataset
# ════════════════════════════════════════════════════════════════════════
class ASLPoseDataset(Dataset):
    """
    ASL Pose 数据集

    目录结构::
        data_root/
            train/  {sample}/text.txt,  pose.json
            dev/    ...
            test/   ...

    pose.json schema::
        {
          "poses": [
            { "pose_keypoints_2d": [...24],
              "hand_left_keypoints_2d": [...63],
              "hand_right_keypoints_2d": [...63] },
            ...
          ]
        }
    """
    def __init__(
        self,
        data_root: str,
        split: str,
        max_samples: Optional[int] = None,
        pose_normalize: bool = True,
        pose_clip_range: float = 8.0,
        truncate_len: Optional[int] = None,     # 可选：截断过长序列
        extern_mean: Optional[Sequence[float]] = None,
        extern_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.data_root       = data_root
        self.split           = split
        self.pose_normalize  = pose_normalize
        self.pose_clip_range = pose_clip_range
        self.truncate_len    = truncate_len

        self.extern_mean = None if extern_mean is None else np.asarray(extern_mean)
        self.extern_std  = None if extern_std  is None else np.asarray(extern_std)

        self.texts:  List[str]             = []
        self.poses:  List[List[List[float]]] = []   # 每条 (T,150)

        self._load_samples(max_samples)

        if self.extern_mean is not None and self.extern_std is not None:
            self.pose_mean, self.pose_std = self.extern_mean, self.extern_std
            self._apply_norm("external μ/σ")
        elif self.pose_normalize and self.poses:
            self._compute_and_apply_norm()

    # ───────────────────────────────────────────────
    # internal helpers
    # ───────────────────────────────────────────────
    def _load_samples(self, max_samples: Optional[int]) -> None:
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"❌ Path not found: {split_dir}")

        ids = sorted(p for p in os.listdir(split_dir)
                     if os.path.isdir(os.path.join(split_dir, p)))
        print(f"🔍 {self.split}: {len(ids)} samples detected")

        for sid in ids:
            base = os.path.join(split_dir, sid)
            txt_f  = os.path.join(base, "text.txt")
            pose_f = os.path.join(base, "pose.json")
            if not (os.path.exists(txt_f) and os.path.exists(pose_f)):
                continue

            try:
                text = open(txt_f, "r", encoding="utf-8").read().strip()
                js   = json.load(open(pose_f, "r", encoding="utf-8"))
            except Exception:
                continue  # skip corrupt

            frames = js.get("poses", [])
            if not frames:
                continue

            seq: List[List[float]] = []
            for fr in frames:
                pose = (fr["pose_keypoints_2d"]
                        + fr["hand_right_keypoints_2d"]
                        + fr["hand_left_keypoints_2d"])
                if len(pose) == 150:
                    seq.append(pose)
                if self.truncate_len and len(seq) >= self.truncate_len:
                    break
            if not seq:
                continue

            self.texts.append(text)
            self.poses.append(seq)
            if max_samples and len(self.texts) >= max_samples:
                break

        print(f"✅ {self.split}: loaded {len(self.texts)} samples")

    def _compute_and_apply_norm(self) -> None:
        all_frames = np.concatenate([np.array(p) for p in self.poses], axis=0)
        self.pose_mean = all_frames.mean(0)
        self.pose_std  = all_frames.std(0)
        self._apply_norm("auto μ/σ")

    def _apply_norm(self, tag: str) -> None:
        out = []
        for seq in self.poses:
            arr = (np.asarray(seq) - self.pose_mean) / (self.pose_std + 1e-8)
            arr = np.clip(arr, -self.pose_clip_range, self.pose_clip_range)
            out.append(arr.tolist())
        self.poses = out
        m, s = self.pose_mean.mean(), self.pose_std.mean()
        print(f"  Normalize ({tag}): μ≈{m:.3f}, σ≈{s:.3f}")

    # ───────────────────────────────────────────────
    # dataset interface
    # ───────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], torch.tensor(self.poses[idx], dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════════
#                      factory: create_data_loaders
# ════════════════════════════════════════════════════════════════════════
def create_data_loaders(cfg):
    train_set = ASLPoseDataset(cfg.data_root, "train",
                               pose_normalize=cfg.pose_normalize,
                               pose_clip_range=cfg.pose_clip_range)

    dev_set   = ASLPoseDataset(cfg.data_root, "dev",
                               pose_normalize=False,
                               pose_clip_range=cfg.pose_clip_range,
                               extern_mean=train_set.pose_mean,
                               extern_std=train_set.pose_std)

    test_set  = ASLPoseDataset(cfg.data_root, "test",
                               pose_normalize=False,
                               pose_clip_range=cfg.pose_clip_range,
                               extern_mean=train_set.pose_mean,
                               extern_std=train_set.pose_std)

    kwargs = dict(batch_size=cfg.batch_size,
                  num_workers=0, pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                              collate_fn=collate_pose_batch, **kwargs)
    dev_loader   = DataLoader(dev_set,   shuffle=False,
                              collate_fn=collate_pose_batch, **kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False,
                              collate_fn=collate_pose_batch, **kwargs)
    return train_loader, dev_loader, test_loader
