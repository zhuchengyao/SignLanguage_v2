# ASL 统一可视化工具使用指南 🎯

## 概述
这是一个统一的ASL可视化工具，整合了两个主要功能：
1. **真实数据可视化** - 可视化数据集中的真实ASL样例
2. **推理生成可视化** - 使用训练好的模型推理生成ASL动画

## 文件结构
```
eggroll_v2/
├── asl_visualizer.py          # 🎯 主要可视化工具
├── checkpoints/best.pth       # 训练好的模型
├── model.py                   # 模型定义
├── config.py                  # 配置文件
├── data_loader.py             # 数据加载器
├── train.py                   # 训练脚本
└── ...
```

## 快速开始

### 方法1: 交互式使用
```bash
python asl_visualizer.py
```

然后根据提示选择功能：
- 输入 `1` - 可视化真实数据
- 输入 `2` - 推理生成可视化
- 输入 `3` - 退出

### 方法2: 编程使用

#### 真实数据可视化
```python
from asl_visualizer import ASLVisualizer

# 创建可视化器
visualizer = ASLVisualizer()

# 可视化真实数据样例
data_folder = "../openpose/datasets/ASL_gloss/dev/69213"
visualizer.visualize_real_data(data_folder)
```

#### 推理生成可视化
```python
from asl_visualizer import ASLVisualizer

# 创建可视化器
visualizer = ASLVisualizer()

# 推理生成并可视化
word = "hello"
visualizer.inference_and_visualize(word)
```

## 功能详解

### 🔍 功能1: 真实数据可视化
- **输入**: 数据文件夹路径（包含text.txt和pose.json）
- **处理**: 加载真实的手语动作数据
- **输出**: 生成动画文件（如：real_apple_20250628_102030.gif）

**示例数据路径**:
```
../openpose/datasets/ASL_gloss/dev/69213/
├── text.txt     # 包含词语 "apple"
└── pose.json    # OpenPose格式的姿态数据
```

### 🤖 功能2: 推理生成可视化
- **输入**: 英文单词/短语
- **处理**: 使用Flow-Matching模型生成32帧手语动作
- **输出**: 生成动画文件（如：inference_hello_20250628_102030.gif）

**支持的词语**: 训练数据集中的任何词语（如：hello, apple, thank_you等）

## 技术参数

### 模型规格
- **架构**: Flow-Matching + Text Encoder (BERT) + Pose GCN
- **输入**: 文本字符串
- **输出**: 32帧 × 150维姿态序列
- **维度分解**: 24(身体) + 63(左手) + 63(右手)

### 可视化规格
- **格式**: GIF动画
- **帧数**: 32帧（推理）/ 可配置（真实数据）
- **颜色编码**: 
  - 🔴 身体关键点（红色点）
  - 🔵 身体连接线（蓝色线）
  - 🟢 左手（绿色）
  - 🟠 右手（橙色）

## 自定义选项

### 调整动画参数
```python
# 自定义帧间隔和输出路径
visualizer.inference_and_visualize(
    text="your_word",
    output_path="custom_name.gif",
    frame_interval=200,  # 毫秒
    num_inference_steps=50
)
```

### 真实数据限制帧数
```python
# 限制显示帧数
visualizer.visualize_real_data(
    data_folder_path="path/to/data",
    max_frames=32,  # 最多显示32帧
    frame_interval=250
)
```

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   ❌ Checkpoint文件不存在: ./checkpoints/best.pth
   ```
   **解决**: 确保已完成模型训练，或下载预训练模型

2. **找不到真实数据**
   ```
   ❌ 路径不存在: ../openpose/datasets/ASL_gloss/dev/69213
   ```
   **解决**: 检查数据集路径是否正确

3. **CUDA内存不足**
   **解决**: 在config.py中降低batch_size或使用CPU推理

### 性能优化
- **GPU推理**: 自动检测并使用CUDA加速
- **推理速度**: 单次推理约10秒（RTX GPU）
- **内存使用**: 约2-4GB GPU显存

## 依赖环境
- Python 3.8+
- PyTorch 1.12+
- matplotlib
- numpy
- transformers (BERT)
- torch_geometric (可选，用于GCN)

## 输出示例
```
🎯 ASL统一可视化工具
==================================================
🚀 ASL统一可视化工具初始化完成
📱 使用设备: cuda

请选择功能:
1. 可视化真实数据集样例
2. 推理生成并可视化  
3. 退出

请输入选择 (1/2/3): 2

=== 推理生成并可视化 ===
请输入要生成的ASL词语: hello
🤖 推理生成手语动画: 'hello'
📦 加载模型: ./checkpoints/best.pth
✅ 模型加载成功 (epoch: 48)
📹 模型生成了 32 帧，每帧 150 维
🎬 创建 Inference 动画: 'hello' (共 32 帧)
💾 保存动画到: inference_hello_20250628_102315.gif
✅ 动画保存成功
✅ 推理生成可视化完成!
```

---
*工具版本: v1.0 | 更新时间: 2025-06-28* 