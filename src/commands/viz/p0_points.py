import os
import argparse
import numpy as np

from src.asl_visualizer import ASLVisualizer


def pose150_from_p0(p0_50x2: np.ndarray, conf: float = 1.0) -> np.ndarray:
    assert p0_50x2.shape == (50, 2)
    out = np.zeros((1, 50, 3), dtype=np.float32)
    out[0, :, :2] = p0_50x2
    out[0, :, 2] = float(conf)
    return out.reshape(1, 150)


def main():
    parser = argparse.ArgumentParser(description='Visualize p0 or p0_norm (50x2) as a single-frame GIF')
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help='paths to *_p0.npy or *_p0_norm.npy')
    parser.add_argument('--out_dir', type=str, default='./outputs/displacements/p0_viz', help='output directory')
    parser.add_argument('--no_invert_y', action='store_true', help='do not invert Y axis in visualization')
    parser.add_argument('--invert_x', action='store_true', help='horizontally mirror (invert X) in visualization')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    viz = ASLVisualizer(invert_y=(not args.no_invert_y), invert_x=args.invert_x)
    for p in args.inputs:
        try:
            arr = np.load(p)
            if arr.shape != (50, 2):
                print(f"[Skip] {p}: unexpected shape {arr.shape}")
                continue
            pose150_seq = pose150_from_p0(arr)
            name = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(args.out_dir, f"p0_{name}.gif")
            viz.create_animation(pose_sequence=pose150_seq, output_path=out_path, title=name)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"[Error] {p}: {e}")


if __name__ == '__main__':
    main()


