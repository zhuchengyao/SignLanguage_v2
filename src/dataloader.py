import os
import json
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm
import random

def collate_pose_batch(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[Optional[List[str]], Optional[torch.Tensor], Optional[torch.Tensor]]:
    # ... This function remains the same ...
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None
    texts, seqs = zip(*batch)
    lens = [s.size(0) for s in seqs]
    T_max = max(lens) if lens else 0
    padded = []
    masks = []
    for seq, L in zip(seqs, lens):
        pad = T_max - L
        padded.append(torch.cat([seq, seq.new_zeros(pad, seq.size(-1))], dim=0))
        masks.append(torch.cat([torch.ones(L, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]))
    return list(texts), torch.stack(padded), torch.stack(masks)

# BucketSampler is not used for the fix, but we keep it in the file for future use if needed.
class BucketSampler(Sampler):
    # ... This class remains the same ...
    def __init__(self, dataset, batch_size, boundaries, shuffle=True, cache_path: str = "sampler_cache.pt"):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.shuffle = shuffle
        self.cache_path = cache_path
        self.buckets = self._prepare_buckets()
    def _get_seq_length(self, idx):
        item = self.dataset[idx]
        return len(item[1]) if item is not None and item[1] is not None else 0
    def _prepare_buckets(self):
        if os.path.exists(self.cache_path):
            print(f"Loading sampler buckets from cache: {self.cache_path}")
            return torch.load(self.cache_path)
        print("Preparing buckets for sampling (first run)...")
        lengths = [self._get_seq_length(i) for i in tqdm(range(len(self.dataset)), desc="Fetching lengths")]
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for i, length in enumerate(lengths):
            if length == 0: continue
            bucket_idx = 0
            for boundary in self.boundaries:
                if length <= boundary: break
                bucket_idx += 1
            buckets[bucket_idx].append(i)
        buckets = [b for b in buckets if b]
        print(f"Buckets prepared. Found {len(buckets)} non-empty buckets. Caching...")
        torch.save(buckets, self.cache_path)
        return buckets
    def __iter__(self):
        all_batches = []
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
    def __init__(self, data_paths: Union[str, List[str]], split: str, pose_normalize: bool = True, pose_clip_range: tuple = (-2.0, 2.0), max_seq_len: Optional[int] = None, extern_mean: Optional[np.ndarray] = None, extern_std: Optional[np.ndarray] = None, cache_in_memory: bool = True, cache_max_items: int = 1024):
        super().__init__()
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.split = split
        self.pose_normalize = pose_normalize
        self.pose_clip_range = pose_clip_range
        self.max_seq_len = max_seq_len
        self.sample_paths: List[Tuple[str, str]] = []
        self._scan_and_build_paths()
        
        self.pose_mean: np.ndarray = np.zeros(150, dtype=np.float32)
        self.pose_std: np.ndarray = np.ones(150, dtype=np.float32)
        
        # --- FIX: Convert to tensors ONCE during initialization ---
        self.mean_tensor: Optional[torch.Tensor] = None
        self.std_tensor: Optional[torch.Tensor] = None
        
        if self.pose_normalize:
            cache_dir = os.path.join(os.path.dirname(self.data_paths[0]), ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            stats_cache_path = os.path.join(cache_dir, f"{self.split}_stats.npz")
            
            if extern_mean is not None and extern_std is not None:
                print(f"  Using external μ/σ for {self.split} set.")
                self.pose_mean, self.pose_std = extern_mean, extern_std
            elif self.sample_paths:
                if os.path.exists(stats_cache_path):
                    print(f"Loading cached stats for '{self.split}' from {stats_cache_path}")
                    stats = np.load(stats_cache_path)
                    self.pose_mean, self.pose_std = stats['mean'], stats['std']
                else:
                    print(f"Computing mean/std for {self.split} set...")
                    self._compute_stats()
                    np.savez(stats_cache_path, mean=self.pose_mean, std=self.pose_std)
                    print(f"Stats cached to {stats_cache_path}")

            # --- FIX: Create tensor versions here ---
            self.mean_tensor = torch.from_numpy(self.pose_mean)
            self.std_tensor = torch.from_numpy(self.pose_std)

        # --- simple LRU-ish cache ---
        self.cache_in_memory = cache_in_memory
        self.cache_max_items = cache_max_items
        self._cache: dict[int, Tuple[str, torch.Tensor]] = {}
        self._cache_order: List[int] = []

    # ... _scan_and_build_paths and _compute_stats methods remain the same ...
    def _scan_and_build_paths(self):
        print(f"Scanning samples for split: {self.split}")
        for data_root in self.data_paths:
            if not os.path.isdir(data_root):
                print(f"Path not found, skipping: {data_root}")
                continue
            for sid in sorted(os.listdir(data_root)):
                base = os.path.join(data_root, sid)
                txt_f, pose_f = os.path.join(base, "text.txt"), os.path.join(base, "pose.json")
                if os.path.exists(txt_f) and os.path.exists(pose_f):
                    self.sample_paths.append((txt_f, pose_f))
        print(f"{self.split}: found a total of {len(self.sample_paths)} valid samples.")
    def _compute_stats(self):
        all_frames = []
        for _, pose_path in tqdm(self.sample_paths, desc="Computing stats"):
            try:
                with open(pose_path, "r", encoding="utf-8") as f: js = json.load(f)
                frames = js.get("poses", [])
                if not frames: continue
                seq = [(fr.get("pose_keypoints_2d", []) + fr.get("hand_right_keypoints_2d", []) + fr.get("hand_left_keypoints_2d", [])) for fr in frames]
                seq_valid = [p for p in seq if len(p) == 150]
                if seq_valid: all_frames.append(np.array(seq_valid, dtype=np.float32))
            except Exception: continue
        if not all_frames:
            print("No valid frames found to compute stats...")
            return
        all_frames_np = np.concatenate(all_frames, axis=0)
        self.pose_mean = all_frames_np.mean(axis=0)
        self.pose_std = all_frames_np.std(axis=0)
        self.pose_std[self.pose_std < 1e-8] = 1e-8
        m, s = self.pose_mean.mean(), self.pose_std.mean()
        print(f"Stats computed: mean~{m:.3f}, std~{s:.3f}")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Optional[Tuple[str, torch.Tensor]]:
        # Memory cache
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        text_path, pose_path = self.sample_paths[idx]
        try:
            with open(text_path, 'r', encoding='utf-8') as f: text = f.read().strip()
            with open(pose_path, 'r', encoding='utf-8') as f: js = json.load(f)
            frames = js.get("poses", [])
            seq_list = [p for p in [(fr.get("pose_keypoints_2d", []) + fr.get("hand_right_keypoints_2d", []) + fr.get("hand_left_keypoints_2d", [])) for fr in frames] if len(p) == 150]
            if not seq_list: return None
            
            pose_tensor = torch.tensor(seq_list, dtype=torch.float32)
            if self.max_seq_len and pose_tensor.size(0) > self.max_seq_len:
                pose_tensor = pose_tensor[:self.max_seq_len]
            
            # --- FIX: Use the pre-converted tensors for normalization ---
            if self.pose_normalize and self.mean_tensor is not None and self.std_tensor is not None:
                # No new tensors are created here, just using existing ones.
                pose_tensor = (pose_tensor - self.mean_tensor) / self.std_tensor
                pose_tensor = torch.clip(pose_tensor, self.pose_clip_range[0], self.pose_clip_range[1])
            item = (text, pose_tensor)

            if self.cache_in_memory:
                self._cache[idx] = item
                self._cache_order.append(idx)
                if len(self._cache_order) > self.cache_max_items:
                    old = self._cache_order.pop(0)
                    self._cache.pop(old, None)

            return item
        except Exception:
            return None