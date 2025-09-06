import os
import json
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from src.config import T2M_Config
from src.pose_normalizer import normalize_sequence_xy


def scan_gloss_samples(data_root: str, splits: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Return mapping: word -> list of tuples (split, sid, sample_dir)
    A valid sample_dir contains text.txt and pose.json
    """
    word_to_samples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for sp in splits:
        base = os.path.join(data_root, sp)
        if not os.path.isdir(base):
            continue
        for sid in sorted(os.listdir(base)):
            d = os.path.join(base, sid)
            txt_f = os.path.join(d, 'text.txt')
            pose_f = os.path.join(d, 'pose.json')
            try:
                if os.path.exists(txt_f) and os.path.exists(pose_f):
                    t = open(txt_f, 'r', encoding='utf-8').read().strip()
                    if t:
                        word_to_samples[t.lower()].append((sp, sid, d))
            except Exception:
                continue
    return word_to_samples


def load_pose150_sequence(pose_json_path: str) -> np.ndarray:
    js = json.load(open(pose_json_path, 'r', encoding='utf-8'))
    frames = js.get('poses', [])
    seq = []
    for fr in frames:
        pose = fr.get('pose_keypoints_2d', []) + fr.get('hand_right_keypoints_2d', []) + fr.get('hand_left_keypoints_2d', [])
        if len(pose) == 150:
            seq.append(pose)
    if not seq:
        return np.zeros((0, 150), dtype=np.float32)
    return np.asarray(seq, dtype=np.float32)


def compute_displacements_xy(sequence_150: np.ndarray, include_delta_conf: bool = False) -> np.ndarray:
    """
    Input: (T, 150)  -> Output:
      - if include_delta_conf == False: (T-1, 50, 2)  (dx, dy for 50 points)
      - if include_delta_conf == True:  (T-1, 50, 3)  (dx, dy, dc)
    The 150 dims are ordered as: body(8*3), right(21*3), left(21*3).
    """
    if sequence_150.shape[0] < 2:
        return np.zeros((0, 50, 3 if include_delta_conf else 2), dtype=np.float32)

    T = sequence_150.shape[0]
    # reshape per frame to (50,3)
    seq_50x3 = sequence_150.reshape(T, 50, 3)
    cur = seq_50x3[:-1]
    nxt = seq_50x3[1:]
    dxdy = nxt[..., :2] - cur[..., :2]
    if not include_delta_conf:
        return dxdy.astype(np.float32)
    dc = (nxt[..., 2:3] - cur[..., 2:3]).astype(np.float32)
    return np.concatenate([dxdy.astype(np.float32), dc], axis=-1)


def select_words(word_to_samples: Dict[str, List[Tuple[str, str, str]]], requested_words: List[str], samples_per_word: int, max_words: int) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    """
    Return list of (word, picked_samples)
    If requested_words is provided, filter and pick from them first.
    Otherwise, auto-pick words that have at least samples_per_word, up to max_words.
    """
    selections: List[Tuple[str, List[Tuple[str, str, str]]]] = []
    picked_words = set()

    def pick_for_word(w: str):
        nonlocal selections
        if w in picked_words:
            return
        samples = word_to_samples.get(w.lower(), [])
        if len(samples) >= samples_per_word:
            selections.append((w, samples[:samples_per_word]))
            picked_words.add(w)

    if requested_words:
        for w in requested_words:
            pick_for_word(w)
    # auto-fill if not enough
    if len(selections) < max_words:
        # sort words by frequency (descending)
        sorted_words = sorted(word_to_samples.items(), key=lambda kv: len(kv[1]), reverse=True)
        for w, samples in sorted_words:
            if len(selections) >= max_words:
                break
            if w in picked_words:
                continue
            if len(samples) >= samples_per_word:
                selections.append((w, samples[:samples_per_word]))
                picked_words.add(w)
    return selections


def save_output(displacements: np.ndarray, meta: dict, out_dir: str, basename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"{basename}.npy")
    json_path = os.path.join(out_dir, f"{basename}.json")
    np.save(npy_path, displacements)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npy_path


def main():
    parser = argparse.ArgumentParser(description='Extract per-frame displacement vectors for selected gloss samples')
    parser.add_argument('--words', type=str, nargs='*', default=None, help='specific words to include, e.g., hello apple able')
    parser.add_argument('--samples_per_word', type=int, default=2, help='number of samples per word')
    parser.add_argument('--max_words', type=int, default=10, help='max number of words to include (auto-selected if not provided)')
    parser.add_argument('--split', type=str, default='', help='optional: limit to a single split: train/dev/test')
    parser.add_argument('--include_delta_conf', action='store_true', help='include delta of confidence as third channel')
    parser.add_argument('--out_dir', type=str, default='./outputs/displacements', help='output directory')
    parser.add_argument('--save_raw', action='store_true', help='save raw (unnormalized) displacements and p0')
    parser.add_argument('--save_norm', action='store_true', help='save normalized displacements and normalized p0')
    args = parser.parse_args()

    cfg = T2M_Config()
    data_root = os.path.join(cfg.data_root, 'ASL_gloss')
    splits = [args.split] if args.split in ('train', 'dev', 'test') else ['train', 'dev', 'test']

    print(f"Scanning dataset at: {data_root} splits={splits}")
    word_to_samples = scan_gloss_samples(data_root, splits)
    print(f"Total distinct words: {len(word_to_samples)}")

    requested_words = args.words or []
    selections = select_words(word_to_samples, requested_words, args.samples_per_word, args.max_words)
    if not selections:
        print("No words with enough samples were found.")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []
    total_saved = 0
    for word, samples in selections:
        for sp, sid, d in samples:
            pose_f = os.path.join(d, 'pose.json')
            seq = load_pose150_sequence(pose_f)
            if seq.shape[0] < 2:
                continue
            base = f"disp_{word}_{ts}_{sp}_{sid}"

            # Save RAW variants
            if args.save_raw or (not args.save_raw and not args.save_norm):
                disp = compute_displacements_xy(seq, include_delta_conf=args.include_delta_conf)
                p0 = seq[0].reshape(50, 3)[:, :2].astype(np.float32)
                p0_path = os.path.join(args.out_dir, f"{base}_p0.npy")
                os.makedirs(args.out_dir, exist_ok=True)
                np.save(p0_path, p0)
                meta = {
                    'word': word,
                    'split': sp,
                    'sid': sid,
                    'sample_dir': d,
                    'input_frames': int(seq.shape[0]),
                    'disp_shape': list(disp.shape),
                    'include_delta_conf': bool(args.include_delta_conf),
                    'p0_path': p0_path,
                    'p0_shape': list(p0.shape),
                    'normalized': False,
                }
                npy_path = save_output(disp, meta, args.out_dir, base)
                summary.append({'word': word, 'split': sp, 'sid': sid, 'npy': npy_path, 'p0': p0_path, 'disp_shape': meta['disp_shape']})
                total_saved += 1

            # Save NORMALIZED variants
            if args.save_norm:
                seq_xy = seq.reshape(seq.shape[0], 50, 3)[..., :2]
                seq_norm, _ = normalize_sequence_xy(seq_xy)
                disp_norm = (seq_norm[1:] - seq_norm[:-1]).astype(np.float32)
                p0_norm = seq_norm[0].astype(np.float32)
                disp_norm_path = os.path.join(args.out_dir, f"{base}_norm.npy")
                p0_norm_path = os.path.join(args.out_dir, f"{base}_p0_norm.npy")
                np.save(disp_norm_path, disp_norm)
                np.save(p0_norm_path, p0_norm)
                norm_meta = {
                    'word': word,
                    'split': sp,
                    'sid': sid,
                    'sample_dir': d,
                    'input_frames': int(seq.shape[0]),
                    'disp_norm_shape': list(disp_norm.shape),
                    'p0_norm_path': p0_norm_path,
                    'p0_norm_shape': list(p0_norm.shape),
                    'normalized': True,
                }
                with open(disp_norm_path.replace('.npy', '.json'), 'w', encoding='utf-8') as f:
                    json.dump(norm_meta, f, ensure_ascii=False, indent=2)
                summary.append({'word': word, 'split': sp, 'sid': sid, 'npy_norm': disp_norm_path, 'p0_norm': p0_norm_path, 'disp_norm_shape': norm_meta['disp_norm_shape']})

    # Save a small manifest for convenience
    manifest_path = os.path.join(args.out_dir, f"manifest_{ts}.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'items': summary}, f, ensure_ascii=False, indent=2)

    print(f"Completed. Saved {total_saved} displacement arrays. Manifest: {manifest_path}")


if __name__ == '__main__':
    main()


