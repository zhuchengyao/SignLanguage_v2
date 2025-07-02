# ASL 数据可视化器使用说明

## 概述

`ASLDataVisualizer` 是一个支持两种ASL数据集的可视化工具：
- **ASL_gloss**: 单词级别的手语数据 (如 "water", "home")
- **ASL_sentence**: 句子级别的手语数据 (如 "You have used waterproof mascara...")

## 快速开始

### 1. 基本使用

```python
from asl_gloss_visualizer import ASLDataVisualizer

# 创建可视化器 - 支持两种数据集类型
visualizer_gloss = ASLDataVisualizer(dataset_type="gloss")      # 单词数据集
visualizer_sentence = ASLDataVisualizer(dataset_type="sentence") # 句子数据集

# 加载数据
visualizer_gloss.load_sample("00000", "dev")  # 加载单词 "water"
visualizer_sentence.load_sample("dev__0fO5ETSwyg_0-5-rgb_front", "dev")  # 加载句子

# 可视化
visualizer_gloss.create_static_plot(0, "word_static.png")
visualizer_sentence.create_animation("sentence_animation.gif")
```

### 2. 数据集对比

| 特性 | ASL_gloss | ASL_sentence |
|------|-----------|--------------|
| **数据类型** | 单词手语 | 句子手语 |
| **样本ID格式** | 数字 (00000, 00001...) | 长字符串 (dev__xxx...) |
| **文本内容** | 单词 ("water", "home") | 完整句子 |
| **帧数** | 20-50帧 | 100-400帧 |
| **路径结构** | `datasets/ASL_gloss/dev/` | `datasets/ASL_sentence/ASL/dev/` |

## 详细使用方法

### 1. 初始化可视化器

```python
# 方式1: 自动检测路径
visualizer = ASLDataVisualizer(dataset_type="gloss")

# 方式2: 手动指定路径
visualizer = ASLDataVisualizer(
    dataset_type="sentence", 
    dataset_root="custom/path/to/ASL_sentence"
)
```

### 2. 浏览数据

```python
# 列出可用样本
gloss_samples = visualizer_gloss.list_available_samples("dev", max_samples=10)
sentence_samples = visualizer_sentence.list_available_samples("dev", max_samples=5)

print("Gloss samples:", gloss_samples)
# 输出: ['00000', '00001', '00002', ...]

print("Sentence samples:", sentence_samples[:2])  # 只显示前2个，因为ID很长
# 输出: ['dev__0fO5ETSwyg_0-5-rgb_front', 'dev__0fO5ETSwyg_1-5-rgb_front']
```

### 3. 加载和可视化

```python
# 加载单词数据
visualizer_gloss.load_sample("00000", "dev")
print(f"单词: {visualizer_gloss.text_data}")  # "water"
print(f"帧数: {len(visualizer_gloss.pose_data['poses'])}")  # 32

# 加载句子数据
visualizer_sentence.load_sample("dev__0fO5ETSwyg_0-5-rgb_front", "dev")
print(f"句子: {visualizer_sentence.text_data}")  # "You have used waterproof..."
print(f"帧数: {len(visualizer_sentence.pose_data['poses'])}")  # 365

# 创建静态图
visualizer_gloss.create_static_plot(0, "word_frame1.png")
visualizer_sentence.create_static_plot(50, "sentence_frame50.png")

# 创建动画
visualizer_gloss.create_animation("word_animation.gif", max_frames=20)
visualizer_sentence.create_animation("sentence_animation.gif", max_frames=30)
```

### 4. 批量处理

```python
def process_gloss_samples(max_samples=5):
    """批量处理单词样本"""
    visualizer = ASLDataVisualizer(dataset_type="gloss")
    samples = visualizer.list_available_samples("dev", max_samples)
    
    for sample_id in samples:
        visualizer.load_sample(sample_id, "dev")
        visualizer.create_static_plot(0, f"gloss_{sample_id}.png")
        print(f"处理完成: {sample_id} - {visualizer.text_data}")

def process_sentence_samples(max_samples=3):
    """批量处理句子样本"""
    visualizer = ASLDataVisualizer(dataset_type="sentence")
    samples = visualizer.list_available_samples("dev", max_samples)
    
    for sample_id in samples:
        visualizer.load_sample(sample_id, "dev")
        short_name = sample_id[:10]  # 缩短文件名
        visualizer.create_static_plot(0, f"sentence_{short_name}.png")
        print(f"处理完成: {short_name} - {visualizer.text_data[:50]}...")
```

## 高级功能

### 1. 数据分析

```python
# 分析关键点数据
frame_data = visualizer.pose_data['poses'][0]
pose_points, left_hand, right_hand = visualizer.extract_keypoints(frame_data)

print(f"身体关键点: {pose_points.shape}")    # (8, 3)
print(f"左手关键点: {left_hand.shape}")     # (21, 3)
print(f"右手关键点: {right_hand.shape}")    # (21, 3)
```

### 2. 自定义可视化

```python
# 指定帧创建静态图
middle_frame = len(visualizer.pose_data['poses']) // 2
visualizer.create_static_plot(middle_frame, "middle_frame.png")

# 自定义动画参数
visualizer.create_animation(
    "custom_animation.gif",
    frame_interval=200,  # 每帧200ms
    max_frames=25        # 最多25帧
)
```

## 文件输出

### 静态图
- **格式**: PNG
- **内容**: 三个子图（身体姿态、左手、右手）
- **命名**: `{dataset_type}_{sample_id}_static.png`

### 动画
- **格式**: GIF
- **帧率**: 10 FPS
- **命名**: `{dataset_type}_{sample_id}_animation.gif`

## 运行演示

```bash
# 在 eggroll_v2 目录下运行
python demo_visualize.py
```

演示脚本会：
1. 展示 ASL_gloss 数据集的使用
2. 展示 ASL_sentence 数据集的使用
3. 生成示例图片和动画

## 注意事项

1. **句子数据集的样本ID很长**，建议使用缩短名称保存文件
2. **句子数据集的帧数较多**，建议使用 `max_frames` 参数限制动画长度
3. **路径会自动检测**，但可以手动指定自定义路径
4. **两种数据集的pose.json格式相同**，关键点提取方式一致

## 故障排除

### 路径问题
```python
# 如果自动检测失败，手动指定路径
visualizer = ASLDataVisualizer(
    dataset_type="gloss",
    dataset_root="你的/自定义/路径/ASL_gloss"
)
```

### 样本不存在
```python
# 先列出可用样本
samples = visualizer.list_available_samples("dev")
print(f"可用样本: {samples}")

# 然后使用实际存在的样本ID
visualizer.load_sample(samples[0], "dev")
``` 