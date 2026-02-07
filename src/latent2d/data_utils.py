import json
import os
from typing import List

import torch


def default_filelist_cache_path(data_root: str, splits: List[str]) -> str:
    split_tag = "_".join(splits)
    return os.path.join(data_root, ".cache", f"pose_filelist_{split_tag}.txt")


def build_pose_filelist(
    data_root: str,
    splits: List[str],
    max_files: int = -1,
    cache_path: str | None = None,
) -> List[str]:
    cache_path = cache_path or default_filelist_cache_path(data_root, splits)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = [line.strip() for line in f if line.strip()]
        if max_files > 0:
            cached = cached[:max_files]
        if cached:
            print(f"Loaded filelist cache: {len(cached)} files from {cache_path}")
            return cached

    files: List[str] = []
    for split in splits:
        base = os.path.join(data_root, split)
        if not os.path.isdir(base):
            continue
        for sid in sorted(os.listdir(base)):
            pose_json = os.path.join(base, sid, "pose.json")
            if os.path.exists(pose_json):
                files.append(pose_json)
                if max_files > 0 and len(files) >= max_files:
                    break
        if max_files > 0 and len(files) >= max_files:
            break

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write("\n".join(files))
    print(f"Saved filelist cache: {len(files)} files to {cache_path}")
    return files


def load_pose150_from_json(pose_json_path: str, max_frames: int | None = None) -> torch.Tensor | None:
    try:
        with open(pose_json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception:
        return None

    frames = js.get("poses", [])
    if max_frames is not None:
        frames = frames[:max_frames]

    seq = []
    for fr in frames:
        pose = (
            fr.get("pose_keypoints_2d", [])
            + fr.get("hand_right_keypoints_2d", [])
            + fr.get("hand_left_keypoints_2d", [])
        )
        if len(pose) == 150:
            seq.append(pose)

    if not seq:
        return None
    return torch.tensor(seq, dtype=torch.float32)


def load_first_n_pose_sequences(
    data_root: str,
    splits: List[str],
    max_samples: int,
    max_frames: int = 64,
    cache_path: str | None = None,
) -> List[torch.Tensor]:
    filelist = build_pose_filelist(
        data_root=data_root,
        splits=splits,
        max_files=max_samples,
        cache_path=cache_path,
    )
    samples: List[torch.Tensor] = []
    for pose_json in filelist:
        seq = load_pose150_from_json(pose_json, max_frames=max_frames)
        if seq is not None:
            samples.append(seq)
    return samples
