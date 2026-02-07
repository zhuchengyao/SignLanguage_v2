import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

class ASLVisualizer:
    def __init__(self, invert_y: bool = True, invert_x: bool = False):
        self.setup_skeleton_connections()
        self.invert_y = invert_y
        self.invert_x = invert_x
        self.colors = {
            'body': {'point': '#FF6B6B', 'line': '#FF6B6B'},
            'left_hand': {'point': '#4ECDC4', 'line': '#4ECDC4'},
            'right_hand': {'point': '#45B7D1', 'line': '#45B7D1'},
            'background': '#FFFFFF',
            'grid': '#E5E5E5'
        }
        self.point_sizes = {'body': 50, 'left_hand': 25, 'right_hand': 25}
        self.line_widths = {'body': 2.5, 'left_hand': 1.5, 'right_hand': 1.5}

    def setup_skeleton_connections(self):
        self.body_connections = [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7)]
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

    def parse_pose_150d(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pose) != 150: raise ValueError(f"Expected 150-dim pose, got {len(pose)}")
        body = pose[:24].reshape(-1, 3)
        right_hand = pose[24:87].reshape(-1, 3)
        left_hand = pose[87:150].reshape(-1, 3)
        return body, right_hand, left_hand

    def filter_valid_points(self, points: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        confidence = points[:, 2]
        # 仅基于置信度过滤，避免因坐标接近0而被整体判无效
        valid = (confidence > threshold)
        return valid

    def interpolate_missing_points(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        对置信度为 0 的关键点做简单插帧：取前后最近的非零帧的对应点坐标求平均。
        边界（无前或后有效点）保持原值以避免外推。
        """
        if pose_sequence.ndim != 2 or pose_sequence.shape[1] != 150:
            raise ValueError(f"Expected pose_sequence shape (T,150), got {pose_sequence.shape}")
        seq = np.array(pose_sequence, copy=True, dtype=np.float32)
        T = seq.shape[0]
        if T == 0:
            return seq
        joints = 50
        seq_view = seq.reshape(T, joints, 3)
        coords = seq_view[..., :2]
        conf = seq_view[..., 2]
        missing_mask = conf <= 0.0
        for j in range(joints):
            missing_idx = np.where(missing_mask[:, j])[0]
            if missing_idx.size == 0:
                continue
            valid_idx = np.where(~missing_mask[:, j])[0]
            if valid_idx.size == 0:
                continue
            for t in missing_idx:
                prev_candidates = valid_idx[valid_idx < t]
                next_candidates = valid_idx[valid_idx > t]
                if prev_candidates.size == 0 or next_candidates.size == 0:
                    # 边界：缺少前或后的有效帧，直接跳过
                    continue
                prev_t = prev_candidates[-1]
                next_t = next_candidates[0]
                coords[t, j] = 0.5 * (coords[prev_t, j] + coords[next_t, j])
                conf[t, j] = 0.5 * (conf[prev_t, j] + conf[next_t, j])
        return seq_view.reshape(T, -1)

    def draw_skeleton_part(self, ax: plt.Axes, points: np.ndarray, connections: List[Tuple], color_config: dict, size: int, line_width: float):
        valid_mask = self.filter_valid_points(points)
        if not valid_mask.any(): return
        coords = points[:, :2]
        valid_coords = coords[valid_mask]
        sx = -1.0 if self.invert_x else 1.0
        sy = -1.0 if self.invert_y else 1.0
        ax.scatter(sx * valid_coords[:, 0], sy * valid_coords[:, 1], c=color_config['point'], s=size, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=3)
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points) and valid_mask[start_idx] and valid_mask[end_idx]:
                start_point, end_point = coords[start_idx], coords[end_idx]
                ax.plot([sx * start_point[0], sx * end_point[0]], [sy * start_point[1], sy * end_point[1]], color=color_config['line'], linewidth=line_width, alpha=0.8, zorder=2)

    def draw_pose(self, pose: np.ndarray, ax: Optional[plt.Axes] = None, title: str = ""):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        body, right_hand, left_hand = self.parse_pose_150d(pose)
        self.draw_skeleton_part(ax, body, self.body_connections, self.colors['body'], self.point_sizes['body'], self.line_widths['body'])
        self.draw_skeleton_part(ax, right_hand, self.hand_connections, self.colors['right_hand'], self.point_sizes['right_hand'], self.line_widths['right_hand'])
        self.draw_skeleton_part(ax, left_hand, self.hand_connections, self.colors['left_hand'], self.point_sizes['left_hand'], self.line_widths['left_hand'])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])

    def create_animation(
        self,
        pose_sequence: np.ndarray,
        output_path: str,
        title: str = "手语动画",
        fps: int = 15,
        interpolate_missing: bool = True,
    ):
        if len(pose_sequence) == 0: raise ValueError("Pose sequence is empty")
        pose_sequence = np.array(pose_sequence, dtype=np.float32, copy=False)
        if interpolate_missing:
            pose_sequence = self.interpolate_missing_points(pose_sequence)
        fig, ax = plt.subplots(figsize=(8, 8))
        all_coords = np.concatenate([self.parse_pose_150d(p)[i][:, :2] for p in pose_sequence for i in range(3)])
        # 放宽可视窗范围，避免坐标过小导致看似空白
        x_min, x_max = np.nanmin(all_coords[:, 0]), np.nanmax(all_coords[:, 0])
        y_min, y_max = np.nanmin(all_coords[:, 1]), np.nanmax(all_coords[:, 1])
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            x_min, x_max, y_min, y_max = -1, 1, -1, 1
        if x_max - x_min < 1e-6 and y_max - y_min < 1e-6:
            # 全部几乎重合，给一个默认范围
            x_min, x_max, y_min, y_max = -1, 1, -1, 1
        margin = max(x_max - x_min, y_max - y_min, 1.0) * 0.2

        def animate(frame_idx):
            ax.clear()
            ax.set_xlim(x_min - margin, x_max + margin)
            if self.invert_y:
                ax.set_ylim(-(y_max + margin), -(y_min - margin))
            else:
                ax.set_ylim((y_min - margin), (y_max + margin))
            self.draw_pose(pose_sequence[frame_idx], ax, title=f"{title}\nFrame {frame_idx + 1}/{len(pose_sequence)}")
        
        anim = animation.FuncAnimation(fig, animate, frames=len(pose_sequence), interval=1000/fps, blit=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close(fig)