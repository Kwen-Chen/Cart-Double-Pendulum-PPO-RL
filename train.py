import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
from double_pendulum_env import CartDoublePendulumEnv


class CartDoublePendulumGymEnv(gym.Env):
    """
    Gymnasium wrapper for the cart-double pendulum environment to make it compatible with stable-baselines3.
    """
    
    def __init__(self, render_mode="human", max_steps=1000):
        super().__init__()
        
        # Initialize the custom environment
        self.env = CartDoublePendulumEnv(render_mode=render_mode, max_steps=max_steps)
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # State space: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        high = np.array([
            4.0,  # x (cart position)
            10.0,  # x_dot (cart velocity)
            np.pi,  # theta1
            8 * np.pi,  # theta1_dot
            np.pi,  # theta2
            8 * np.pi,  # theta2_dot
        ], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=-high, high=high, dtype=np.float32
        )
        
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        observation = self.env.reset()
        info = {}
        return observation, info
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action[0])
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


def train_cart_double_pendulum():
    """
    Train the cart-double pendulum using PPO algorithm.
    """
    print("开始训练小车二级摆智能体...")
    
    log_dir = "./logs/"
    
    # Create training environment
    def make_env():
        env = CartDoublePendulumGymEnv(render_mode=None, max_steps=1000)
        # The Monitor wrapper will be added by make_vec_env
        return env
    
    # Create vectorized training environment and wrap it with Monitor
    train_env = make_vec_env(make_env, n_envs=4, monitor_dir=log_dir)
    
    # Create evaluation environment
    # The Monitor wrapper will be added by the EvalCallback
    eval_env = CartDoublePendulumGymEnv(render_mode=None, max_steps=1000)
    
    # Define PPO model
    model = PPO(
        "MlpPolicy", 
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/",
        device="auto"
    )
    
    print(f"使用设备: {model.device}")
    print(f"网络架构: {model.policy}")
    
    # Set up callbacks
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=5000, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )
    
    # Train the model
    total_timesteps = 1000000
    print(f"开始训练，总时间步数: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("models/cart_double_pendulum_ppo_final")
    print("训练完成！模型已保存。")
    
    return model, eval_env


def test_trained_model(model_path="models/cart_double_pendulum_ppo_final.zip", episodes=5):
    """
    Test the trained model with visualization.
    """
    print(f"加载已训练的模型: {model_path}")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create test environment with rendering
    test_env = CartDoublePendulumGymEnv(render_mode="human", max_steps=2000)
    
    print(f"开始测试，运行 {episodes} 个回合...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        print(f"\n第 {episode + 1} 回合开始")
        
        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Render the environment
            test_env.render()
            
            # Handle pygame events
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"第 {episode + 1} 回合结束")
        print(f"回合奖励: {episode_reward:.2f}")
        print(f"回合长度: {episode_length} 步")
        
        # Wait a bit before next episode
        import time
        time.sleep(1)
    
    test_env.close()
    
    # Print statistics
    print(f"\n测试统计:")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"最高奖励: {np.max(episode_rewards):.2f}")
    print(f"最长长度: {np.max(episode_lengths)} 步")


def plot_training_results():
    """
    Plot training results from the logs.
    """
    try:
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        
        log_dir = "./logs/"
        if os.path.exists(log_dir):
            results = load_results(log_dir)
            x, y = ts2xy(results, 'timesteps')
            
            plt.figure(figsize=(12, 4))
            
            # Plot episode rewards
            plt.subplot(1, 2, 1)
            plt.plot(x, y)
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title('Episode Rewards During Training')
            plt.grid(True)
            
            # Plot moving average
            window_size = 100
            if len(y) > window_size:
                moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                plt.plot(x[window_size-1:], moving_avg, 'r-', label=f'{window_size}-step Moving Average')
                plt.legend()
            
            # Plot episode lengths
            plt.subplot(1, 2, 2)
            episode_lengths = results['l'].values
            plt.plot(episode_lengths)
            plt.xlabel('Episode Number')
            plt.ylabel('Episode Length')
            plt.title('Episode Lengths During Training')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("训练结果图表已保存为 training_results.png")
        else:
            print("未找到训练日志文件夹")
            
    except Exception as e:
        print(f"绘制训练结果时出错: {e}")


# Keep backward compatibility
train_double_pendulum = train_cart_double_pendulum


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="小车二级摆强化学习训练和测试")
    parser.add_argument("--mode", choices=["train", "test", "plot"], default="train",
                       help="运行模式: train(训练), test(测试), plot(绘图)")
    parser.add_argument("--model_path", default="models/cart_double_pendulum_ppo_final.zip",
                       help="模型文件路径 (仅测试模式使用)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="测试回合数 (仅测试模式使用)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    if args.mode == "train":
        model, eval_env = train_cart_double_pendulum()
        eval_env.close()
        print("\n训练完成！")
        print("运行 'python train.py --mode test' 来测试训练好的模型")
        print("运行 'python train.py --mode plot' 来查看训练结果图表")
        
    elif args.mode == "test":
        if os.path.exists(args.model_path):
            test_trained_model(args.model_path, args.episodes)
        else:
            print(f"错误: 找不到模型文件 {args.model_path}")
            print("请先运行训练模式: python train.py --mode train")
            
    elif args.mode == "plot":
        plot_training_results() 