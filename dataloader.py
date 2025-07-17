# data_loader.py (Final fix for config passing)
from __future__ import annotations
import os, json
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import random

# ... [collate_pose_batch, BucketSampler, ASLPoseDataset 类的代码完全不变, 此处省略] ...
def collate_pose_batch(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None];
    if not batch: return None, None, None
    texts, seqs = zip(*batch); lens = [s.size(0) for s in seqs]; T_max = max(lens) if lens else 0
    padded, masks = [], []
    for seq, L in zip(seqs, lens):
        pad = T_max - L
        padded.append(torch.cat([seq, seq.new_zeros(pad, seq.size(-1))], dim=0))
        masks.append(torch.cat([torch.ones(L, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]))
    return list(texts), torch.stack(padded), torch.stack(masks)
class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, boundaries, shuffle=True, cache_path: str = "sampler_cache.pt"):
        super().__init__(dataset); self.dataset = dataset; self.batch_size = batch_size; self.boundaries = boundaries; self.shuffle = shuffle; self.cache_path = cache_path
        self.buckets = self._prepare_buckets()
    def _get_seq_length(self, idx):
        item = self.dataset[idx]; return len(item[1]) if item is not None and item[1] is not None else 0
    def _prepare_buckets(self):
        if os.path.exists(self.cache_path):
            print(f"✅ Loading sampler buckets from cache: {self.cache_path}"); return torch.load(self.cache_path)
        print("🪣 Preparing buckets for sampling (first run)..."); lengths = [self._get_seq_length(i) for i in tqdm(range(len(self.dataset)), desc="  Fetching lengths")]
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for i, length in enumerate(lengths):
            if length == 0: continue
            bucket_idx = 0
            for boundary in self.boundaries:
                if length <= boundary: break
                bucket_idx += 1
            buckets[bucket_idx].append(i)
        buckets = [b for b in buckets if b]; print(f"✅ Buckets prepared. Found {len(buckets)} non-empty buckets. Caching..."); torch.save(buckets, self.cache_path)
        return buckets
    def __iter__(self):
        all_batches = [];
        for bucket in self.buckets:
            if self.shuffle: random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size: all_batches.append(batch)
        if self.shuffle: random.shuffle(all_batches)
        for batch in all_batches: yield batch
    def __len__(self):
        total_len = 0
        for bucket in self.buckets: total_len += len(bucket) // self.batch_size
        return total_len
class ASLPoseDataset(Dataset):
    def __init__( self, data_paths: Union[str, List[str]], split: str, pose_normalize: bool = True, pose_clip_range: float = 8.0, max_seq_len: Optional[int] = None, extern_mean: Optional[np.ndarray] = None, extern_std: Optional[np.ndarray] = None):
        super().__init__(); self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]; self.split = split; self.pose_normalize = pose_normalize; self.pose_clip_range = pose_clip_range; self.max_seq_len = max_seq_len
        self.sample_paths: List[Tuple[str, str]] = []; self._scan_and_build_paths()
        self.pose_mean: np.ndarray; self.pose_std: np.ndarray
        stats_cache_path = os.path.join(os.path.dirname(self.data_paths[0]), f".cache/{self.split}_stats.npz")
        if extern_mean is not None and extern_std is not None:
            print(f"  Using external μ/σ for {self.split} set."); self.pose_mean, self.pose_std = extern_mean, extern_std
        elif self.pose_normalize and self.sample_paths:
            if os.path.exists(stats_cache_path):
                print(f"  ⚡ Loading cached stats for '{self.split}' from {stats_cache_path}"); stats = np.load(stats_cache_path); self.pose_mean, self.pose_std = stats['mean'], stats['std']
            else:
                print(f"  Computing μ/σ for {self.split} set..."); self._compute_stats(); os.makedirs(os.path.dirname(stats_cache_path), exist_ok=True); np.savez(stats_cache_path, mean=self.pose_mean, std=self.pose_std); print(f"   -> Stats cached to {stats_cache_path}")
    def _scan_and_build_paths(self):
        print(f"🔍 Scanning samples for split: {self.split}")
        for data_root in self.data_paths:
            if not os.path.isdir(data_root): print(f"  ⚠️ Path not found, skipping: {data_root}"); continue
            for sid in sorted(os.listdir(data_root)):
                base = os.path.join(data_root, sid); txt_f, pose_f = os.path.join(base, "text.txt"), os.path.join(base, "pose.json")
                if os.path.exists(txt_f) and os.path.exists(pose_f):
                    try:
                        with open(pose_f, 'r') as f: data = json.load(f)
                        if data.get("poses"): self.sample_paths.append((txt_f, pose_f))
                    except (json.JSONDecodeError, KeyError): continue
        print(f"✅ {self.split}: found a total of {len(self.sample_paths)} valid samples.")
    def _compute_stats(self):
        all_frames = []
        for _, pose_path in tqdm(self.sample_paths, desc="  Computing stats"):
            try:
                with open(pose_path, "r", encoding="utf-8") as f: js = json.load(f)
                frames = js.get("poses", []);
                if not frames: continue
                seq = [ (fr.get("pose_keypoints_2d", []) + fr.get("hand_right_keypoints_2d", []) + fr.get("hand_left_keypoints_2d", [])) for fr in frames ]
                seq_valid = [p for p in seq if len(p) == 150]
                if seq_valid: all_frames.append(np.array(seq_valid, dtype=np.float32))
            except Exception: continue
        if not all_frames: print("  ⚠️ No valid frames found to compute stats..."); self.pose_mean = np.zeros(150, dtype=np.float32); self.pose_std = np.ones(150, dtype=np.float32); return
        all_frames_np = np.concatenate(all_frames, axis=0); self.pose_mean = all_frames_np.mean(0); self.pose_std = all_frames_np.std(0)
        m, s = self.pose_mean.mean(), self.pose_std.mean(); print(f"  Stats computed: μ≈{m:.3f}, σ≈{s:.3f}")
    def __len__(self) -> int: return len(self.sample_paths)
    def __getitem__(self, idx: int) -> Optional[Tuple[str, torch.Tensor]]:
        text_path, pose_path = self.sample_paths[idx]
        try:
            with open(text_path, 'r', encoding='utf-8') as f: text = f.read().strip()
            with open(pose_path, 'r', encoding='utf-8') as f: js = json.load(f)
            frames = js.get("poses", [])
            seq_list = [ p for p in [ (fr.get("pose_keypoints_2d", []) + fr.get("hand_right_keypoints_2d", []) + fr.get("hand_left_keypoints_2d", [])) for fr in frames ] if len(p) == 150 ]
            if not seq_list: return None
            pose_tensor = torch.tensor(seq_list, dtype=torch.float32)
            if self.max_seq_len and pose_tensor.size(0) > self.max_seq_len: pose_tensor = pose_tensor[:self.max_seq_len]
            if self.pose_normalize:
                mean = torch.from_numpy(self.pose_mean).to(pose_tensor.device); std = torch.from_numpy(self.pose_std).to(pose_tensor.device)
                pose_tensor = (pose_tensor - mean) / (std + 1e-8); pose_tensor = torch.clip(pose_tensor, -self.pose_clip_range, self.pose_clip_range)
            return text, pose_tensor
        except Exception: return None


# ✨ CHANGED: 函数签名现在只接收一个统一的配置对象 cfg
def create_data_loaders(cfg):
    # ✨ CHANGED: 所有路径现在都从统一的 cfg 对象中读取
    train_paths = [os.path.join(cfg.data_root, "ASL_gloss/train")]
    dev_paths =   [os.path.join(cfg.data_root, "ASL_gloss/dev")]
    test_paths =  [os.path.join(cfg.data_root, "ASL_gloss/test")]

    max_len = getattr(cfg, 'vae_max_seq_len', None)
    
    train_set = ASLPoseDataset(train_paths, "train", 
                               pose_normalize=cfg.pose_normalize, 
                               pose_clip_range=cfg.pose_clip_range,
                               max_seq_len=max_len)
    
    if len(train_set) == 0: raise ValueError("Training set is empty. Please check your data paths.")

    dev_set = ASLPoseDataset(dev_paths, "dev", 
                             pose_normalize=cfg.pose_normalize, 
                             pose_clip_range=cfg.pose_clip_range,
                             max_seq_len=max_len,
                             extern_mean=train_set.pose_mean, 
                             extern_std=train_set.pose_std)
    
    test_set = ASLPoseDataset(test_paths, "test", 
                              pose_normalize=cfg.pose_normalize, 
                              pose_clip_range=cfg.pose_clip_range,
                              max_seq_len=max_len,
                              extern_mean=train_set.pose_mean, 
                              extern_std=train_set.pose_std)
    
    bucket_boundaries = getattr(cfg, 'bucket_boundaries', [50, 100, 150, 200])
    # ✨ CHANGED: 缓存路径也从统一的 cfg 对象中读取
    cache_path = os.path.join(cfg.data_root, "ASL_gloss/.cache", f"sampler_cache_train.pt")
    train_sampler = BucketSampler(train_set, batch_size=cfg.batch_size, boundaries=bucket_boundaries, shuffle=True, cache_path=cache_path)
    
    kwargs = dict(num_workers=getattr(cfg, 'num_workers', 0), pin_memory=True)
    
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_pose_batch, **kwargs)
    
    dev_loader = DataLoader(dev_set, shuffle=False, batch_size=cfg.batch_size, collate_fn=collate_pose_batch, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=cfg.batch_size, collate_fn=collate_pose_batch, **kwargs)
    
    return train_loader, dev_loader, test_loader