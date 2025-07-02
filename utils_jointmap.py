import numpy as np

# 150-D 向量中的分段索引
BODY24_START  = 0        # 0‥23  -> 24*3 = 72 floats
R_HAND_START  = 24       # 24‥86 -> 63 floats
L_HAND_START  = 24 + 63  # 87‥149-> 63 floats

# 8-keypoint 上半身（按 OpenPose BODY_25 顺序）
BODY_8 = [
    0,  # 0 nose
    1,  # 1 neck
    2, 3, 4,      # 右肩/肘/腕
    5, 6, 7,      # 左肩/肘/腕
]

# 21-hand 索引 0‥20 原样使用
R_HAND_21 = [R_HAND_START + i for i in range(21)]
L_HAND_21 = [L_HAND_START + i for i in range(21)]

# 拼接得到模型 0‥49 的映射
OPENPOSE_IDX = BODY_8 + R_HAND_21 + L_HAND_21     # len = 8+21+21 = 50
assert len(OPENPOSE_IDX) == 50, "映射必须正好 50 个关节"

def reorder(pose150: np.ndarray) -> np.ndarray:
    """
    pose150: np.ndarray, shape (50,3) 或 (150,)
    将 OpenPose 顺序 → 模型顺序
    返回 (50,3) numpy
    """
    p = pose150.reshape(50, 3)
    return p[np.asarray(OPENPOSE_IDX)]
