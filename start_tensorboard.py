import subprocess
import sys
import time
import os
from improved_config import config

def start_tensorboard():
    """启动Tensorboard查看训练日志"""
    print("🚀 启动Tensorboard")
    print("=" * 40)
    
    log_dir = config.log_dir
    if not os.path.exists(log_dir):
        print(f"❌ 日志目录不存在: {log_dir}")
        return
    
    print(f"📊 日志目录: {log_dir}")
    print("🌐 启动Tensorboard服务器...")
    
    try:
        # 启动Tensorboard
        cmd = [
            sys.executable, "-m", "tensorboard.main", 
            "--logdir", log_dir,
            "--host", "localhost",
            "--port", "6006"
        ]
        
        print("执行命令:", " ".join(cmd))
        print("\n" + "="*50)
        print("📈 Tensorboard正在启动中...")
        print("🌐 访问地址: http://localhost:6006")
        print("❗ 按 Ctrl+C 停止服务")
        print("="*50)
        
        # 运行Tensorboard
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tensorboard启动失败: {e}")
        print("💡 请确保已安装tensorboard: pip install tensorboard")
    except KeyboardInterrupt:
        print("\n⏹️ Tensorboard已停止")
    except Exception as e:
        print(f"❌ 启动过程中出错: {e}")

def show_training_metrics():
    """显示可查看的训练指标"""
    print("\n📊 Tensorboard中可查看的训练指标:")
    print("-" * 40)
    print("📈 Loss/Train - 训练损失曲线")
    print("📉 Loss/Validation - 验证损失曲线") 
    print("📏 Learning_Rate - 学习率变化")
    print("🔄 训练进度和收敛情况")
    print("⏱️ 各轮次的时间消耗")

if __name__ == "__main__":
    show_training_metrics()
    
    print("\n启动Tensorboard吗? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice in ['y', 'yes', '是', '启动']:
        start_tensorboard()
    else:
        print("已取消启动")
        print("你可以手动启动: tensorboard --logdir logs_improved") 