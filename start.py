#!/usr/bin/env python3
"""
小车二级摆项目快速启动脚本
Cart Double Pendulum Project Quick Start Script

这个脚本提供一个用户友好的界面来使用小车二级摆强化学习项目。
This script provides a user-friendly interface for the cart-double pendulum RL project.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


def check_python_version():
    """检查Python版本 / Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print("   Error: Python 3.8 or higher is required")
        print(f"   当前版本 / Current version: {sys.version}")
        return False
    return True


def check_dependencies():
    """检查依赖包是否安装 / Check if dependencies are installed"""
    required_packages = [
        'pygame', 'numpy', 'torch', 'stable_baselines3', 
        'gymnasium', 'matplotlib', 'tensorboard'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包 / Missing dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 解决方案 / Solution:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装 / All dependencies installed")
    return True


def install_dependencies():
    """安装依赖包 / Install dependencies"""
    print("🔄 正在安装依赖包... / Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装成功 / Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖包安装失败 / Failed to install dependencies")
        return False


def create_directories():
    """创建必要的目录 / Create necessary directories"""
    dirs = ["models", "logs", "tensorboard_logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("📁 目录创建完成 / Directories created")


def show_welcome():
    """显示欢迎信息 / Show welcome message"""
    print("=" * 70)
    print("🚗 小车二级摆强化学习项目 / Cart Double Pendulum Reinforcement Learning")
    print("=" * 70)
    print("📚 本项目使用pygame实现小车二级摆仿真，并通过PPO算法训练智能体")
    print("📚 This project simulates a cart with double pendulum and trains an agent using PPO")
    print("=" * 70)


def show_menu():
    """显示主菜单 / Show main menu"""
    print("\n🎯 请选择操作 / Choose an option:")
    print("1. 🎮 演示环境 (Demo Environment)")
    print("2. 🧠 训练模型 (Train Model)")
    print("3. 🧪 测试模型 (Test Trained Model)")
    print("4. 📊 查看训练结果 (View Training Results)")
    print("5. 📖 查看帮助 (Help)")
    print("6. 🚪 退出 (Exit)")


def run_demo():
    """运行演示 / Run demo"""
    print("\n🎮 启动演示程序...")
    try:
        subprocess.run([sys.executable, "demo.py"])
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"❌ 运行演示时出错: {e}")


def run_training():
    """运行训练 / Run training"""
    print("\n🧠 开始训练模型...")
    print("注意: 训练可能需要很长时间 (建议30分钟到几小时)")
    print("Note: Training may take a long time (30 minutes to several hours)")
    
    confirm = input("\n确认开始训练? (y/N): ").strip().lower()
    if confirm == 'y':
        try:
            subprocess.run([sys.executable, "train.py", "--mode", "train"])
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        except Exception as e:
            print(f"❌ 训练时出错: {e}")
    else:
        print("训练已取消")


def run_testing():
    """运行测试 / Run testing"""
    model_path = "models/cart_double_pendulum_ppo_final.zip"
    
    if not os.path.exists(model_path):
        print("❌ 未找到训练好的模型")
        print("   请先运行训练或确保模型文件存在于 models/ 目录")
        return
    
    print(f"\n🧪 测试模型: {model_path}")
    episodes = input("请输入测试回合数 (默认5): ").strip()
    episodes = episodes if episodes.isdigit() else "5"
    
    try:
        subprocess.run([sys.executable, "train.py", "--mode", "test", "--episodes", episodes])
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"❌ 测试时出错: {e}")


def show_results():
    """显示训练结果 / Show training results"""
    if not os.path.exists("logs"):
        print("❌ 未找到训练日志")
        print("   请先完成训练")
        return
    
    print("\n📊 显示训练结果...")
    try:
        subprocess.run([sys.executable, "train.py", "--mode", "plot"])
    except KeyboardInterrupt:
        print("\n显示结果被用户中断")
    except Exception as e:
        print(f"❌ 显示结果时出错: {e}")


def show_help():
    """显示帮助信息 / Show help information"""
    print("\n📖 帮助信息 / Help Information")
    print("=" * 50)
    print("🎮 演示环境:")
    print("   - 随机控制: 观看小车在随机动作下的表现")
    print("   - PD控制器: 使用经典控制方法尝试平衡二级摆")
    print("   - 键盘控制: 手动控制小车 (←→方向键)")
    print()
    print("🧠 训练模型:")
    print("   - 使用PPO算法训练智能体")
    print("   - 默认训练50万步 (约需30分钟-2小时)")
    print("   - 支持GPU加速 (如果可用)")
    print()
    print("🧪 测试模型:")
    print("   - 加载训练好的模型进行测试")
    print("   - 可视化智能体的控制策略")
    print("   - 显示性能统计信息")
    print()
    print("📊 查看结果:")
    print("   - 显示训练过程中的奖励曲线")
    print("   - 分析学习进度和收敛情况")
    print("=" * 50)
    print("💡 提示:")
    print("   - 首次使用建议先运行演示了解环境")
    print("   - 训练前确保有足够的计算资源和时间")
    print("   - 可以使用 Ctrl+C 中断长时间运行的程序")
    print()
    print("🚗 系统说明:")
    print("   - 小车可在轨道上左右移动")
    print("   - 小车上安装两级连接的摆杆")
    print("   - 目标是通过移动小车来保持摆杆平衡")


def main():
    """主函数 / Main function"""
    # 检查Python版本
    if not check_python_version():
        return
    
    # 显示欢迎信息
    show_welcome()
    
    # 检查依赖
    if not check_dependencies():
        print("\n🔧 是否自动安装依赖包? / Auto install dependencies? (y/N): ", end="")
        if input().strip().lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("请手动安装依赖包: pip install -r requirements.txt")
            return
    
    # 创建目录
    create_directories()
    
    # 主循环
    while True:
        show_menu()
        choice = input("\n请输入选择 / Enter choice (1-6): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            run_training()
        elif choice == '3':
            run_testing()
        elif choice == '4':
            show_results()
        elif choice == '5':
            show_help()
        elif choice == '6':
            print("\n👋 感谢使用！再见！/ Thank you! Goodbye!")
            break
        else:
            print("❌ 无效选择，请重新输入 / Invalid choice, please try again")
        
        # 等待用户按键继续
        if choice in ['1', '2', '3', '4', '5']:
            input("\n按回车键继续... / Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断 / Program interrupted by user")
    except Exception as e:
        print(f"\n❌ 程序出现错误 / Program error: {e}")
        print("请检查环境配置或联系技术支持") 