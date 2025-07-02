import torch
import torch.nn as nn
import math
from skeletalModel import getSkeletalModelStructure


class PoseFeatureExtractor(nn.Module):
    """
    把 150-D (50 关节 × xyz) 原坐标扩展成多尺度几何特征：
    - 距离  |direction|：关节连接向量的模与方向
    - 角度  ∠p1-p2-p3
    - 比例  ：重要骨段长度比
    - 手部  ：手掌展开度、手腕-掌心距离
    全部拼接成 `combined`，其 150 原始维永远排在最前。
    """
    def __init__(self):
        super().__init__()

        # ── 骨架拓扑 ──────────────────────────────
        self.connections = []        # [(j1, j2, bone_id), ...]
        self.bones, self.joints = set(), set()
        for s, e, bone in getSkeletalModelStructure():
            self.connections.append((s, e, bone))
            self.bones.add(bone)
            self.joints.update([s, e])

        self.num_joints       = len(self.joints)         # 50
        self.num_connections  = len(self.connections)    # 25

        # 预存连接索引 → 向量化计算
        idx_start = torch.tensor([s for s, _, _ in self.connections])
        idx_end   = torch.tensor([e for _, e, _ in self.connections])
        self.register_buffer('idx_start', idx_start, persistent=False)
        self.register_buffer('idx_end',   idx_end,   persistent=False)

        # 特殊关节分组
        self.body_joints       = [0, 1, 2, 3, 4, 5, 6, 7]
        self.left_hand_joints  = list(range(8, 29))
        self.right_hand_joints = list(range(29, 50))

        print(f"骨骼统计 ▶ joints={self.num_joints}  bones={len(self.bones)}  "
              f"connections={self.num_connections}")

    # ──────────────────────── 基础工具 ────────────────────────
    @staticmethod
    def _euclidean(a, b):
        return torch.norm(a - b, dim=-1)

    # ──────────────────────── 特征计算 ────────────────────────
    def extract_pose_coordinates(self, pose):
        """(B,150) → (B,50,3)"""
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        return pose.view(pose.size(0), 50, 3)

    def compute_distance_features(self, coords):
        """向量化：一次性 gather 计算 25 条连线的欧氏距离"""
        start = coords[:, self.idx_start, :2]      # (B,C,2)
        end   = coords[:, self.idx_end,   :2]
        return self._euclidean(start, end)         # (B,C)

    def compute_direction_features(self, coords):
        """连接向量与 x 轴夹角 θ"""
        start = coords[:, self.idx_start, :2]
        end   = coords[:, self.idx_end,   :2]
        v     = end - start                        # (B,C,2)
        return torch.atan2(v[..., 1], v[..., 0])   # (B,C)

    def compute_angle_features(self, coords):
        """∠p1-p2-p3；数量依骨架而定"""
        triples = []
        adj = {j: [] for j in range(50)}
        for s, e, _ in self.connections:
            adj[s].append(e)
            adj[e].append(s)
        for j, nbrs in adj.items():
            n = len(nbrs)
            if n < 2:
                continue
            for i in range(n - 1):
                for k in range(i + 1, n):
                    triples.append((nbrs[i], j, nbrs[k]))

        if not triples:
            return coords.new_zeros(coords.size(0), 0)

        p1, p2, p3 = zip(*triples)
        v1 = coords[:, p1, :2] - coords[:, p2, :2]
        v2 = coords[:, p3, :2] - coords[:, p2, :2]

        cos = (v1 * v2).sum(-1) / (torch.norm(v1, dim=-1) *
                                   torch.norm(v2, dim=-1) + 1e-8)
        return torch.acos(torch.clamp(cos, -1, 1))   # (B,#angles)

    def compute_ratio_features(self, coords):
        """几段骨长比例，共 4 维"""
        b = {
            'l_up'  : (2, 3),    # 左上臂
            'l_lo'  : (3, 4),    # 左前臂
            'r_up'  : (5, 6),
            'r_lo'  : (6, 7),
            'should': (2, 5)     # 肩宽
        }
        lens = {k: self._euclidean(coords[:, s, :2], coords[:, e, :2]) for k, (s, e) in b.items()}
        sw   = lens['should']
        ratios = torch.stack([
            lens['l_up'] / (sw + 1e-8),
            lens['r_up'] / (sw + 1e-8),
            lens['l_lo'] / (lens['l_up'] + 1e-8),
            lens['r_lo'] / (lens['r_up'] + 1e-8)
        ], dim=1)
        return ratios                                               # (B,4)

    def compute_hand_features(self, coords):
        """左右手指张开度 & 腕→掌心距离，共 4 维"""
        def hand_stats(joint_idx, wrist_idx):
            pts = coords[:, joint_idx, :2]                          # (B,21,2)
            ctr = pts.mean(dim=1)                                   # (B,2)
            spread = torch.norm(pts - ctr.unsqueeze(1), dim=2).mean(1)
            wrist = coords[:, wrist_idx, :2]
            wrist2ctr = self._euclidean(wrist, ctr)
            return spread, wrist2ctr

        ls, lw = hand_stats(self.left_hand_joints, 4)
        rs, rw = hand_stats(self.right_hand_joints, 7)
        return torch.stack([ls, rs, lw, rw], dim=1)                 # (B,4)

    # ───────────────────────── forward ─────────────────────────
    def forward(self, pose):
        coords = self.extract_pose_coordinates(pose)
        feats = [
            pose,                                    # 原始150
            self.compute_distance_features(coords),
            self.compute_angle_features(coords),
            self.compute_direction_features(coords),
            self.compute_ratio_features(coords),
            self.compute_hand_features(coords)
        ]
        return {
            'original':   feats[0],
            'distances':  feats[1],
            'angles':     feats[2],
            'directions': feats[3],
            'ratios':     feats[4],
            'hands':      feats[5],
            'combined':   torch.cat(feats, dim=1)
        }

    # — meta info —
    def get_feature_dims(self):
        return {
            'original': 150,
            'distances': self.num_connections,
            'angles': 'dynamic',
            'directions': self.num_connections,
            'ratios': 4,
            'hands': 4,
        }
