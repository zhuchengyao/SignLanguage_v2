import os
import json
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ASLPoseDataset(Dataset):
    """ASL Pose 数据集 (支持外部均值/方差)

    数据根目录结构::
        data_root/
            train/  {sample_id}/text.txt & pose.json
            dev/    ...
            test/   ...

    `pose.json` 每帧字段::
        {
          "poses": [
              {"pose_keypoints_2d": [...24],
               "hand_left_keypoints_2d": [...63],
               "hand_right_keypoints_2d": [...63]},
              ...
          ]
        }

    参数
    ------
    split : str                'train' | 'dev' | 'test'
    max_samples : int | None    仅加载前 N 个样本 (调试用)
    pose_normalize : bool       是否根据当前 split 重新计算均值/方差
    extern_mean/std : Sequence  若给定, 直接用外部 μ/σ 做归一化, 不再自动计算
    clip_len : int              固定帧长, 不足用最后一帧补齐
    pose_clip_range : float     归一化后截断区间 [-k, k]
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        pose_normalize: bool = True,
        pose_clip_range: float = 3.0,
        clip_len: int = 32,
        extern_mean: Optional[Sequence[float]] = None,
        extern_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.pose_normalize = pose_normalize
        self.pose_clip_range = pose_clip_range
        self.clip_len = clip_len

        # 将外部 μ/σ 存成 ndarray 方便广播
        self.extern_mean = None if extern_mean is None else np.asarray(extern_mean)
        self.extern_std = None if extern_std is None else np.asarray(extern_std)

        self.texts: List[str] = []
        self.poses: List[List[List[float]]] = []  # (T, 150)

        self._load_data(max_samples)

        # 归一化策略
        if self.extern_mean is not None and self.extern_std is not None:
            self.pose_mean = self.extern_mean
            self.pose_std = self.extern_std
            self._apply_norm("外部 μ/σ")
        elif self.pose_normalize and len(self.poses):
            self._compute_and_apply_norm()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_data(self, max_samples: Optional[int]) -> None:
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"❌ 数据目录不存在: {split_dir}")

        sample_dirs = sorted(
            d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
        )
        print(f"🔍 {self.split}: 发现 {len(sample_dirs)} 个样本")

        for d in sample_dirs:
            sample_path = os.path.join(split_dir, d)
            text_file = os.path.join(sample_path, "text.txt")
            pose_file = os.path.join(sample_path, "pose.json")
            if not (os.path.exists(text_file) and os.path.exists(pose_file)):
                continue

            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                with open(pose_file, "r", encoding="utf-8") as f:
                    js = json.load(f)
            except Exception:
                continue  # skip corrupt

            frames = js.get("poses", [])
            if not frames:
                continue

            seq: List[List[float]] = []
            for fr in frames[: self.clip_len]:
                pose = (
                    fr["pose_keypoints_2d"]
                    + fr["hand_right_keypoints_2d"]
                    + fr["hand_left_keypoints_2d"]
                )
                if len(pose) == 150:
                    seq.append(pose)
            if not seq:
                continue

            while len(seq) < self.clip_len:
                seq.append(seq[-1].copy())  # pad last frame
            self.texts.append(text)
            self.poses.append(seq)

            if max_samples and len(self.texts) >= max_samples:
                break
        print(f"✅ {self.split}: 成功加载 {len(self.texts)} 个样本")

    def _compute_and_apply_norm(self) -> None:
        all_frames = np.array(self.poses).reshape(-1, 150)  # (N,150)
        self.pose_mean = all_frames.mean(0)
        self.pose_std = all_frames.std(0)
        self._apply_norm("自动 μ/σ")
        print(f"  数据归一化 (自动): μ≈{self.pose_mean.mean():.3f}, σ≈{self.pose_std.mean():.3f}")

    def _apply_norm(self, tag: str) -> None:
        normed = []
        for seq in self.poses:
            arr = np.asarray(seq)
            arr = (arr - self.pose_mean) / (self.pose_std + 1e-8)
            arr = np.clip(arr, -self.pose_clip_range, self.pose_clip_range)
            normed.append(arr.tolist())
        self.poses = normed
        print(f"  数据归一化 ({tag}): 均值=0, 标准差=1")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        text = self.texts[idx]
        pose_seq = torch.tensor(self.poses[idx], dtype=torch.float32)  # (T,150)
        return text, pose_seq


# ----------------------------------------------------------------------
# Convenience loader factory
# ----------------------------------------------------------------------

def create_data_loaders(config):
    """构建 train / dev / test 的 DataLoader, 保证 μ/σ 一致"""

    # 1⃣  先加载 train, 自动归一化
    train_set = ASLPoseDataset(
        data_root=config.data_root,
        split="train",
        pose_normalize=config.pose_normalize,
        pose_clip_range=config.pose_clip_range,
        clip_len=config.clip_len,
    )

    # 2⃣  dev / test 复用 train 的 μ/σ
    dev_set = ASLPoseDataset(
        data_root=config.data_root,
        split="dev",
        pose_normalize=False,            # 关闭内部均值计算
        pose_clip_range=config.pose_clip_range,
        clip_len=config.clip_len,
        extern_mean=train_set.pose_mean,
        extern_std=train_set.pose_std,
    )

    test_set = ASLPoseDataset(
        data_root=config.data_root,
        split="test",
        pose_normalize=False,
        pose_clip_range=config.pose_clip_range,
        clip_len=config.clip_len,
        extern_mean=train_set.pose_mean,
        extern_std=train_set.pose_std,
    )

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=0,      # Win / 简化
        pin_memory=True,
    )

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    dev_loader   = DataLoader(dev_set,   shuffle=False, drop_last=False, **loader_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, dev_loader, test_loader
