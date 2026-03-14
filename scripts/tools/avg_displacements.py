import os
import re
import json
import argparse
from datetime import datetime
from typing import List, Tuple

import numpy as np


def list_word_displacements(in_dir: str, word: str, max_samples: int = 8) -> List[Tuple[str, str]]:
    """
    Return list of (npy_path, json_path) for given word, up to max_samples.
    We rely on filename pattern disp_{word}_*.npy and verify with json metadata if exists.
    """
    files = []
    pattern = re.compile(rf"^disp_{re.escape(word.lower())}_.*\.npy$")
    for fn in sorted(os.listdir(in_dir)):
        if pattern.match(fn):
            npy_path = os.path.join(in_dir, fn)
            js_path = npy_path[:-4] + ".json"
            if os.path.exists(js_path):
                try:
                    meta = json.load(open(js_path, 'r', encoding='utf-8'))
                    if str(meta.get('word', '')).lower() != word.lower():
                        continue
                except Exception:
                    pass
            files.append((npy_path, js_path if os.path.exists(js_path) else ''))
            if len(files) >= max_samples:
                break
    return files


def resample_by_phase(deltas: np.ndarray, T_star: int) -> np.ndarray:
    """
    deltas: (T-1, J, d)
    Returns: (T_star, J, d), resampled along cumulative path-length phase in [0,1].
    """
    if deltas.ndim != 3:
        raise ValueError(f"Expected deltas shape (T-1, J, d), got {deltas.shape}")
    Tm1, J, d = deltas.shape
    if Tm1 <= 0:
        return np.zeros((T_star, J, d), dtype=np.float32)

    flat = deltas.reshape(Tm1, -1)
    step_norm = np.linalg.norm(flat, axis=1)
    total = float(step_norm.sum())
    if total <= 1e-8:
        # all-zero motion; just return zeros
        return np.zeros((T_star, J, d), dtype=np.float32)

    phase = np.cumsum(step_norm)
    phase = phase / phase[-1]
    target = np.linspace(0.0, 1.0, T_star, dtype=np.float32)

    res = np.zeros((T_star, flat.shape[1]), dtype=np.float32)
    # interpolate each dimension independently
    for dim in range(flat.shape[1]):
        res[:, dim] = np.interp(target, phase, flat[:, dim])
    return res.reshape(T_star, J, d).astype(np.float32)


def robust_average_median(stack: np.ndarray) -> np.ndarray:
    """
    stack: (K, T_star, J, d)
    Returns: (T_star, J, d) median across K (robust to outliers)
    """
    if stack.ndim != 4:
        raise ValueError(f"Expected stack shape (K, T, J, d), got {stack.shape}")
    return np.median(stack, axis=0).astype(np.float32)


def integrate_from_p0(deltas: np.ndarray, p0: np.ndarray = None) -> np.ndarray:
    """
    deltas: (T_star, J, d)
    p0: (J, d) or None (defaults to zeros)
    Returns positions: (T_star+1, J, d)
    """
    T_star, J, d = deltas.shape
    if p0 is None:
        p0 = np.zeros((J, d), dtype=np.float32)
    pos = np.zeros((T_star + 1, J, d), dtype=np.float32)
    pos[0] = p0
    pos[1:] = p0 + np.cumsum(deltas, axis=0)
    return pos


def save_template(out_dir: str, word: str, T_star: int, deltas: np.ndarray, positions: np.ndarray, sources: List[str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f"template_{word}_{ts}_T{T_star}"
    np.save(os.path.join(out_dir, base + "_deltas.npy"), deltas)
    np.save(os.path.join(out_dir, base + "_positions.npy"), positions)
    meta = {
        'word': word,
        'T_star': int(T_star),
        'deltas_shape': list(deltas.shape),
        'positions_shape': list(positions.shape),
        'sources': sources,
        'note': 'deltas are robust-median after phase resampling; positions integrated from zeros'
    }
    with open(os.path.join(out_dir, base + ".json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return base


def main():
    parser = argparse.ArgumentParser(description='Build robust averaged displacement templates per gloss')
    parser.add_argument('--in_dir', type=str, default='./outputs/displacements', help='directory containing disp_*.npy/json')
    parser.add_argument('--out_dir', type=str, default='./outputs/displacements/templates', help='directory to save templates')
    parser.add_argument('--words', type=str, nargs='+', required=True, help='gloss words, e.g., hello apple')
    parser.add_argument('--T', type=int, default=64, help='target length after phase resampling')
    parser.add_argument('--max_samples_per_word', type=int, default=8, help='max sequences per word to use')
    args = parser.parse_args()

    manifest = []
    for word in args.words:
        pairs = list_word_displacements(args.in_dir, word, max_samples=args.max_samples_per_word)
        if len(pairs) < 2:
            print(f"[Skip] word='{word}': need at least 2 samples, found {len(pairs)}")
            continue
        resampled = []
        used = []
        for npy_path, js_path in pairs:
            try:
                d = np.load(npy_path)
                # expect (T-1, 50, 2) or (T-1, 50, 3) -> use first 2 dims
                if d.ndim != 3 or d.shape[1] != 50 or d.shape[2] < 2:
                    print(f"  [Warn] Unexpected shape {d.shape} in {npy_path}; skipping")
                    continue
                d_xy = d[..., :2].astype(np.float32)
                r = resample_by_phase(d_xy, args.T)
                resampled.append(r)
                used.append(npy_path)
            except Exception as e:
                print(f"  [Warn] Failed to load {npy_path}: {e}")
        if len(resampled) < 2:
            print(f"[Skip] word='{word}': usable samples < 2 after filtering")
            continue
        stack = np.stack(resampled, axis=0)  # (K, T, 50, 2)
        bar = robust_average_median(stack)    # (T, 50, 2)
        pos = integrate_from_p0(bar, p0=None)  # (T+1, 50, 2)
        base = save_template(args.out_dir, word, args.T, bar, pos, used)
        manifest.append({'word': word, 'base': base})
        print(f"[OK] word='{word}': template saved as base='{base}'")

    # Save a small overall manifest
    if manifest:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(args.out_dir, f"manifest_{ts}.json"), 'w', encoding='utf-8') as f:
            json.dump({'items': manifest}, f, ensure_ascii=False, indent=2)
        print(f"Wrote manifest for {len(manifest)} word(s).")


if __name__ == '__main__':
    main()


