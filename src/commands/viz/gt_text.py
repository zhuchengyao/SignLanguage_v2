import os
import json
import argparse
from datetime import datetime
from typing import List, Tuple

import torch

from src.config import T2M_Config
from src.asl_visualizer import ASLVisualizer


def find_samples_with_text(data_root: str, target: str, splits: List[str]) -> List[Tuple[str, str, str]]:
    results = []
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
                    if t.lower() == target.lower():
                        results.append((sp, sid, d))
            except Exception:
                continue
    return results


def load_pose150_sequence(pose_json_path: str) -> List[list]:
    js = json.load(open(pose_json_path, 'r', encoding='utf-8'))
    frames = js.get('poses', [])
    seq = []
    for fr in frames:
        pose = fr.get('pose_keypoints_2d', []) + fr.get('hand_right_keypoints_2d', []) + fr.get('hand_left_keypoints_2d', [])
        if len(pose) == 150:
            seq.append(pose)
    return seq


def main():
    parser = argparse.ArgumentParser(description='Visualize ground-truth pose for a given text')
    parser.add_argument('--text', type=str, required=True, help='target text to search, e.g., able')
    parser.add_argument('--split', type=str, default='', help='optional: limit to one split (train/dev/test)')
    parser.add_argument('--max', type=int, default=1, help='max number of samples to visualize')
    parser.add_argument('--out_dir', type=str, default='./outputs', help='output directory')
    args = parser.parse_args()

    cfg = T2M_Config()
    data_root = os.path.join(cfg.data_root, 'ASL_gloss')
    splits = [args.split] if args.split in ('train', 'dev', 'test') else ['train', 'dev', 'test']

    print(f"Searching '{args.text}' in {data_root} splits={splits}")
    matches = find_samples_with_text(data_root, args.text, splits)
    if not matches:
        print('No sample found.')
        return
    print(f"Found {len(matches)} matches. Visualizing up to {args.max}...")

    os.makedirs(args.out_dir, exist_ok=True)
    viz = ASLVisualizer()
    count = 0
    for sp, sid, d in matches:
        pose_f = os.path.join(d, 'pose.json')
        seq_list = load_pose150_sequence(pose_f)
        # convert to numpy array (T,150) for visualizer
        import numpy as np
        seq = np.array(seq_list, dtype=np.float32)
        if seq.size == 0:
            continue
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(args.out_dir, f"real_{args.text}_{ts}_{sp}_{sid}.gif")
        print(f"Saving {out_path} ...")
        viz.create_animation(pose_sequence=seq, output_path=out_path, title=f"GT: {args.text}")
        count += 1
        if count >= args.max:
            break
    print(f"Done. Created {count} GIF(s).")


if __name__ == '__main__':
    main()


