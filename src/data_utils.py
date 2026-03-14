"""共享的数据加载工具函数。

原位于 train_ae2d.py 顶部，因被多个脚本引用，提取至此。
"""

import os
import json
import torch
from tqdm import tqdm


def build_pose_filelist(root: str, splits: list[str], max_files: int, cache_path: str) -> list[str]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f if line.strip()]
        if max_files > 0:
            files = files[:max_files]
        if files:
            print(f"Loaded filelist cache: {len(files)} files from {cache_path}")
            return files
    files = []
    for sp in splits:
        base = os.path.join(root, sp)
        if not os.path.isdir(base):
            continue
        for sid in sorted(os.listdir(base)):
            jf = os.path.join(base, sid, 'pose.json')
            if os.path.exists(jf):
                files.append(jf)
                if max_files > 0 and len(files) >= max_files:
                    break
        if max_files > 0 and len(files) >= max_files:
            break
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(files))
    print(f"Saved filelist cache: {len(files)} files to {cache_path}")
    return files


def load_first_n_samples(root: str, max_samples: int = 64, max_T: int = 64,
                         splits: list[str] | None = None, cache_path: str | None = None):
    if splits is None:
        splits = ['train']
    if cache_path is None:
        cache_path = os.path.join(root, '.cache', f'pose_filelist_{"_".join(splits)}.txt')
    filelist = build_pose_filelist(root, splits, max_samples, cache_path)
    samples = []
    for jf in tqdm(filelist, desc='Loading JSONs'):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                js = json.load(f)
            frames = js.get('poses', [])
            if not frames:
                continue
            arr = []
            for fr in frames[:max_T]:
                pose = fr.get('pose_keypoints_2d', []) + fr.get('hand_right_keypoints_2d', []) + fr.get('hand_left_keypoints_2d', [])
                if len(pose) == 150:
                    arr.append(pose)
            if arr:
                samples.append(torch.tensor(arr, dtype=torch.float32))
        except KeyboardInterrupt:
            raise
        except Exception:
            continue
    return samples
