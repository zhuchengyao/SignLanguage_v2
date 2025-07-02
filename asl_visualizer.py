#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASL 统一可视化工具（适配“方案 B”时序扩散模型）
功能1: 可视化真实数据集中的 ASL 样例
功能2: 使用 checkpoints/best.pth 推理生成 ASL 动画并可视化
"""

import os, json
from datetime import datetime
from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ───────── 模型 & 配置 ─────────
from model import TextToPoseDiffusion
from config import ModelConfig           # ⬅️ NEW

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ASLVisualizer:
    """统一可视化工具，兼容优化后模型。"""

    BODY_JOINTS = 8
    HAND_JOINTS = 21

    def __init__(self, checkpoint_path: str = "./checkpoints/best.pth"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TextToPoseDiffusion | None = None
        self.model_cfg: ModelConfig | None = None        # ⬅️ NEW

        # —— 连接拓扑 ——（与训练数据一致）
        self.pose_connections = [
            (0, 1),
            (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
        ]
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        print("🚀 ASL 可视化工具初始化完毕 (Optimised Model)")
        print("📱 使用设备:", self.device)

    # ------------------------------------------------------------------
    # 功能 1: 可视化真实数据集样例
    # ------------------------------------------------------------------
    def _load_real_sample(self, folder: str):
        txt_path = os.path.join(folder, "text.txt")
        pose_path = os.path.join(folder, "pose.json")
        if not (os.path.exists(txt_path) and os.path.exists(pose_path)):
            raise FileNotFoundError("text.txt 或 pose.json 缺失")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        with open(pose_path, "r", encoding="utf-8") as f:
            poses = json.load(f)["poses"]
        return text, poses

    def visualize_real_sample(self, folder: str,
                              max_frames: int = 32, interval: int = 250):
        text, poses = self._load_real_sample(folder)
        poses = poses[:max_frames]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in text if c.isalnum() or c in ("_", " ")).replace(" ", "_")
        out_path = f"real_{safe}_{timestamp}.gif"
        self._create_animation(poses, text, out_path, interval, "Real Data")

    # ------------------------------------------------------------------
    # 功能 2: 推理并可视化
    # ------------------------------------------------------------------
    def _load_model(self):
        if self.model is not None:
            return self.model

        print("📦 正在加载模型…")
        # ── 1) 读取 checkpoint
        ckpt = None
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            print(f"🔑 从 {self.checkpoint_path} 读取权重")

        # ── 2) 确定 ModelConfig
        if ckpt and "model_cfg" in ckpt:
            self.model_cfg = ModelConfig(**ckpt["model_cfg"])
            print("📝 使用 checkpoint 内保存的 model_cfg")
        else:
            self.model_cfg = ModelConfig()
            print("📝 使用默认 ModelConfig()")

        # ── 3) 构建模型并加载权重
        self.model = TextToPoseDiffusion(self.model_cfg).to(self.device)
        if ckpt:
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print("⚠️  缺少参数:", missing)
            if unexpected:
                print("⚠️  多余参数:", unexpected)
        self.model.eval()
        print("✅ 模型加载完成")
        return self.model

    @torch.no_grad()
    def infer_and_visualize(self, text: str,
                            steps: int = 300, interval: int = 100):
        model = self._load_model()
        print(f"🤖 推理 '{text}' (steps={steps})…")
        seq = model.sample([text], num_steps=steps).cpu().numpy()[0]   # (T, pose_dim)

        pose_dim = seq.shape[1]
        assert pose_dim == self.BODY_JOINTS * 3 + 2 * self.HAND_JOINTS * 3, \
            f"pose_dim ({pose_dim}) 与 50×3 不符"

        body_end = self.BODY_JOINTS * 3
        lh_end   = body_end + self.HAND_JOINTS * 3

        body = seq[:, :body_end]
        lh   = seq[:, body_end:lh_end]
        rh   = seq[:, lh_end:]

        T = seq.shape[0]
        poses: List[dict] = []
        for i in range(T):
            frame = {
                "pose_keypoints_2d":      body[i].tolist(),
                "hand_left_keypoints_2d": lh[i].tolist(),
                "hand_right_keypoints_2d": rh[i].tolist(),
            }
            poses.append(frame)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in text if c.isalnum() or c in ("_", " ")).replace(" ", "_")
        out_path = f"infer_{safe}_{timestamp}.gif"
        self._create_animation(poses, text, out_path, interval, "Inference")

    # ------------------------------------------------------------------
    # 绘制 / 动画 通用方法
    # ------------------------------------------------------------------
    @staticmethod
    def _np_points(kpts):
        return np.array(kpts).reshape(-1, 3)

    def _draw_edges(self, ax, pts, conns, p_col, l_col, p_size, l_w):
        valid = (np.abs(pts[:, 0]) > 1e-6) | (np.abs(pts[:, 1]) > 1e-6)
        if valid.any():
            ax.scatter(pts[valid, 0], -pts[valid, 1],
                       c=p_col, s=p_size, alpha=0.85)
        for a, b in conns:
            if a < len(pts) and b < len(pts) and valid[a] and valid[b]:
                ax.plot([pts[a, 0], pts[b, 0]],
                        [-pts[a, 1], -pts[b, 1]],
                        color=l_col, linewidth=l_w, alpha=0.8)

    def _create_animation(self, poses, text, out_path,
                          interval, tag):
        print(f"🎬 生成 {tag} 动画，共 {len(poses)} 帧 → {out_path}")
        fig, ax = plt.subplots(figsize=(7, 7))

        # 计算可视区域
        coords = []
        for fr in poses:
            for key in ("pose_keypoints_2d",
                        "hand_left_keypoints_2d",
                        "hand_right_keypoints_2d"):
                coords.append(self._np_points(fr[key])[:, :2])
        coords = np.concatenate(coords, axis=0)
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        margin = 0.1 * max(xmax - xmin, ymax - ymin, 1)
        xlim = (xmin - margin, xmax + margin)
        ylim = (-(ymax + margin), -(ymin - margin))

        def _update(i):
            ax.clear()
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            fr = poses[i]
            body = self._np_points(fr["pose_keypoints_2d"])
            lh   = self._np_points(fr["hand_left_keypoints_2d"])
            rh   = self._np_points(fr["hand_right_keypoints_2d"])

            self._draw_edges(ax, body, self.pose_connections,
                             "red", "blue", 40, 2)
            self._draw_edges(ax, lh, self.hand_connections,
                             "green", "green", 20, 1.5)
            self._draw_edges(ax, rh, self.hand_connections,
                             "orange", "orange", 20, 1.5)

            ax.set_title(f'{tag}: "{text}" | Frame {i+1}/{len(poses)}')

        ani = animation.FuncAnimation(fig, _update,
                                      frames=len(poses),
                                      interval=interval, blit=False)
        ani.save(out_path, writer="pillow", fps=1000 / interval)
        print("✅ 动画保存成功")
        plt.show()


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------
def main():
    vis = ASLVisualizer()
    while True:
        print("\n=== 选择功能 ===\n1) 可视化真实数据样例\n2) 推理生成并可视化\n3) 退出")
        choice = input("请输入选项 (1/2/3): ").strip()
        if choice == "1":
            folder = input("数据文件夹路径: ").strip()
            vis.visualize_real_sample(folder) if os.path.isdir(folder) \
                else print("❌ 路径不存在")
        elif choice == "2":
            text = input("输入要生成的 ASL 词语: ").strip()
            vis.infer_and_visualize(text) if text else print("❌ 请输入有效文本")
        elif choice == "3":
            print("👋 再见！"); break
        else:
            print("❌ 无效选择")


if __name__ == "__main__":
    main()
