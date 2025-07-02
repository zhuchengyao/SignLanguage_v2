import os, json, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

def reshape_body(lst: list[float]) -> np.ndarray:
    """body list 长度 24 或 72 → 返回 (8,3)"""
    arr = np.array(lst, dtype=np.float32)
    if len(arr) == 24:          # 已是 8×3
        return arr.reshape(8, 3)
    if len(arr) == 72:          # 24×3 取 0,1,2,5,6,7,8,11 (8 joints)
        arr = arr.reshape(24, 3)
        idx = [0, 1, 2, 5, 8, 11, 14, 17]
        return arr[idx]
    raise ValueError("body len neither 24 nor 72")

def reshape_hand(lst: list[float]) -> np.ndarray:
    """hand list 长度 63 或 189 → 前 21 joints (21,3)"""
    arr = np.array(lst, dtype=np.float32)
    if len(arr) == 63:
        return arr.reshape(21, 3)
    if len(arr) == 189:
        return arr.reshape(63, 3)[:21]
    # 其他情况补零
    if len(arr) == 0:
        return np.zeros((21, 3), dtype=np.float32)
    raise ValueError("unexpected hand length")

class ASLPoseDataset(Dataset):
    def __init__(self, root: str, split="train", clip_len=32):
        self.root, self.split, self.clip_len = root, split, clip_len
        self.texts, self.poses = [], []; self._load()

    def _load(self):
        pdir = os.path.join(self.root, self.split)
        dirs = sorted(d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d)))
        keep = 0
        for d in dirs:
            t_path = os.path.join(pdir, d, "text.txt")
            j_path = os.path.join(pdir, d, "pose.json")
            if not (os.path.exists(t_path) and os.path.exists(j_path)):
                continue
            text = open(t_path, "r", encoding="utf-8").read().strip()
            frames = json.load(open(j_path, "r", encoding="utf-8")).get("poses", [])
            seq = []
            for fr in frames[: self.clip_len]:
                try:
                    body  = reshape_body(fr.get("pose_keypoints_2d",  []))
                    right = reshape_hand(fr.get("hand_right_keypoints_2d", []))
                    left  = reshape_hand(fr.get("hand_left_keypoints_2d",  []))
                except ValueError:
                    continue
                p50 = np.vstack([body, right, left])       # (50,3)
                seq.append(p50.reshape(150).tolist())
            if not seq: continue
            while len(seq) < self.clip_len: seq.append(seq[-1].copy())
            self.texts.append(text); self.poses.append(seq); keep += 1
        print(f"✅ {self.split}: kept {keep} clips")

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.poses[idx], dtype=torch.float32)

def create_data_loaders(cfg):
    kw = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
    tr = ASLPoseDataset(cfg.data_root, "train", cfg.clip_len)
    if len(tr)==0:
        raise RuntimeError("train split empty ― 检查数据目录！")
    dv = ASLPoseDataset(cfg.data_root, "dev", cfg.clip_len)
    te = ASLPoseDataset(cfg.data_root, "test", cfg.clip_len)
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(dv, **kw) if len(dv) else None,
            DataLoader(te, **kw) if len(te) else None)
