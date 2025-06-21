#!/usr/bin/env python3
"""
小车二级摆演示脚本
Cart Double Pendulum Demo Script

这个脚本演示了不同的控制策略：
1. 随机控制
2. PD控制器
3. 键盘控制

This script demonstrates different control strategies:
1. Random control
2. PD controller  
3. Keyboard control
"""

import numpy as np
import pygame
import time
from double_pendulum_env import CartDoublePendulumEnv
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


class CartPDController:
    """
    Simple PD controller for cart-double pendulum system.
    Controls the cart position to balance the double pendulum.
    简单的PD控制器用于小车二级摆系统。
    """
    
    def __init__(self, kp_pos=10, kd_pos=5, kp1=30, kd1=8, kp2=15, kd2=4):
        self.kp_pos = kp_pos  # Proportional gain for cart position
        self.kd_pos = kd_pos  # Derivative gain for cart velocity
        self.kp1 = kp1  # Proportional gain for first pendulum
        self.kd1 = kd1  # Derivative gain for first pendulum
        self.kp2 = kp2  # Proportional gain for second pendulum
        self.kd2 = kd2  # Derivative gain for second pendulum
    
    def get_action(self, state):
        """
        Calculate control action based on current state.
        """
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        
        # PD control for pendulums (trying to keep them upright)
        # Target angles are 0 (vertical position)
        pendulum_control = -(self.kp1 * theta1 + self.kd1 * theta1_dot + 
                           self.kp2 * theta2 + self.kd2 * theta2_dot)
        
        # PD control for cart position (trying to keep cart near center)
        position_control = -(self.kp_pos * x + self.kd_pos * x_dot)
        
        # Combine both controls
        total_force = pendulum_control + position_control
        
        # Normalize to [-1, 1] range
        return np.clip(total_force / 20.0, -1.0, 1.0)


def demo_random_control():
    """演示随机控制 / Demonstrate random control"""
    print("🎲 随机控制演示 / Random Control Demo")
    print("小车将使用随机动作进行控制...")
    
    env = CartDoublePendulumEnv(render_mode="human", max_steps=1000)
    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    try:
        while not done and step_count < 500:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
            
            # Random action
            action = np.random.uniform(-1, 1)
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            env.render()
            time.sleep(0.02)
            
            if done:
                print(f"随机控制结束: {step_count} 步, 总奖励: {total_reward:.2f}")
                break
    
    except KeyboardInterrupt:
        print("演示被用户中断")
    
    finally:
        env.close()
        time.sleep(1)


def demo_pd_control():
    """演示PD控制器 / Demonstrate PD controller"""
    print("🎯 PD控制器演示 / PD Controller Demo")
    print("小车将使用PD控制器尝试平衡二级摆...")
    
    env = CartDoublePendulumEnv(render_mode="human", max_steps=2000)
    controller = CartPDController()
    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    try:
        while not done and step_count < 1500:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
            
            # Get action from PD controller
            action = controller.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            env.render()
            time.sleep(0.02)
            
            if done:
                print(f"PD控制结束: {step_count} 步, 总奖励: {total_reward:.2f}")
                break
    
    except KeyboardInterrupt:
        print("演示被用户中断")
    
    finally:
        env.close()
        time.sleep(1)


def demo_keyboard_control():
    """演示键盘控制 / Demonstrate keyboard control"""
    print("⌨️  键盘控制演示 / Keyboard Control Demo")
    print("使用方向键控制小车:")
    print("← 左箭头: 小车向左移动")
    print("→ 右箭头: 小车向右移动")
    print("ESC: 退出演示")
    print("空格: 无控制输入")
    
    env = CartDoublePendulumEnv(render_mode="human", max_steps=3000)
    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    action = 0.0
    
    try:
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key == pygame.K_LEFT:
                        action = -0.5
                    elif event.key == pygame.K_RIGHT:
                        action = 0.5
                    elif event.key == pygame.K_SPACE:
                        action = 0.0
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        action = 0.0
            
            # Apply action
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            env.render()
            time.sleep(0.02)
            
            if done:
                print(f"键盘控制结束: {step_count} 步, 总奖励: {total_reward:.2f}")
                break
    
    except KeyboardInterrupt:
        print("演示被用户中断")
    
    finally:
        env.close()
        time.sleep(1)


def main():
    """主演示函数 / Main demo function"""
    print("=" * 60)
    print("🎪 小车二级摆演示程序 / Cart Double Pendulum Demo")
    print("=" * 60)
    
    print("\n选择演示模式 / Choose demo mode:")
    print("1. 随机控制 / Random Control")
    print("2. PD控制器 / PD Controller")
    print("3. 键盘控制 / Keyboard Control")
    print("4. 全部演示 / All Demos")
    print("q. 退出 / Quit")
    
    while True:
        choice = input("\n请输入选择 / Enter your choice (1-4, q): ").strip().lower()
        
        if choice == '1':
            demo_random_control()
            break
        elif choice == '2':
            demo_pd_control()
            break
        elif choice == '3':
            demo_keyboard_control()
            break
        elif choice == '4':
            print("\n🚀 开始全部演示...")
            demo_random_control()
            print("\n" + "="*40)
            demo_pd_control()
            print("\n" + "="*40)
            demo_keyboard_control()
            break
        elif choice == 'q':
            print("再见! / Goodbye!")
            return
        else:
            print("无效选择，请重新输入 / Invalid choice, please try again")
    
    print("\n✨ 演示完成！")
    print("💡 提示:")
    print("- 运行 'python train.py --mode train' 开始训练强化学习模型")
    print("- 运行 'python train.py --mode test' 测试训练好的模型")
    print("- 查看 README.md 了解更多信息")


if __name__ == "__main__":
    main() 