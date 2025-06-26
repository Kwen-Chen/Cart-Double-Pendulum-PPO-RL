import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
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


def train_cart_double_pendulum(algorithm="PPO", total_timesteps=1000000):
    """
    Train the cart-double pendulum using specified algorithm.
    
    Args:
        algorithm (str): Training algorithm to use ("PPO", "SAC", or "DDPG")
        total_timesteps (int): Total number of training timesteps
    """
    print(f"开始使用 {algorithm} 算法训练小车二级摆智能体...")
    
    log_dir = f"./logs/{algorithm.lower()}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment
    def make_env():
        env = CartDoublePendulumGymEnv(render_mode=None, max_steps=1000)
        return env
    
    # Create evaluation environment
    eval_env = CartDoublePendulumGymEnv(render_mode=None, max_steps=1000)
    
    # Configure model based on algorithm
    if algorithm.upper() == "PPO":
        # Create vectorized training environment for PPO (supports multiple environments)
        train_env = make_vec_env(make_env, n_envs=4, monitor_dir=log_dir)
        
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
            tensorboard_log=f"./tensorboard_logs/{algorithm}/",
            device="auto"
        )
        
    elif algorithm.upper() == "SAC":
        # SAC can work with vectorized environments
        train_env = make_vec_env(make_env, n_envs=4, monitor_dir=log_dir)
        
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            tensorboard_log=f"./tensorboard_logs/{algorithm}/",
            device="auto"
        )
        
    elif algorithm.upper() == "DDPG":
        # DDPG can work with vectorized environments
        train_env = make_vec_env(make_env, n_envs=4, monitor_dir=log_dir)
        
        # Add action noise for DDPG exploration
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        model = DDPG(
            "MlpPolicy",
            train_env,
            action_noise=action_noise,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=f"./tensorboard_logs/{algorithm}/",
            device="auto"
        )
        
    else:
        raise ValueError(f"不支持的算法: {algorithm}. 请选择 'PPO', 'SAC', 或 'DDPG'.")
    
    print(f"使用设备: {model.device}")
    print(f"算法: {algorithm}")
    print(f"网络架构: {model.policy}")
    
    # Set up callbacks
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=5000, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{algorithm.lower()}/",
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )
    
    # Train the model
    print(f"开始训练，总时间步数: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    os.makedirs(f"./models/{algorithm.lower()}/", exist_ok=True)
    model.save(f"models/{algorithm.lower()}/cart_double_pendulum_{algorithm.lower()}_final")
    print("训练完成！模型已保存。")
    
    return model, eval_env


def test_trained_model(model_path="models/ppo/cart_double_pendulum_ppo_final.zip", episodes=5, algorithm="PPO"):
    """
    Test the trained model with visualization.
    
    Args:
        model_path (str): Path to the trained model
        episodes (int): Number of episodes to test
        algorithm (str): Algorithm used for training
    """
    print(f"加载已训练的模型: {model_path}")
    
    # Load the trained model based on algorithm
    if algorithm.upper() == "PPO":
        model = PPO.load(model_path)
    elif algorithm.upper() == "SAC":
        model = SAC.load(model_path)
    elif algorithm.upper() == "DDPG":
        model = DDPG.load(model_path)
    else:
        raise ValueError(f"不支持的算法: {algorithm}. 请选择 'PPO', 'SAC', 或 'DDPG'.")
    
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


def plot_training_results(algorithm="PPO"):
    """
    Plot training results from the logs.
    
    Args:
        algorithm (str): Algorithm to plot results for
    """
    try:
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        
        log_dir = f"./logs/{algorithm.lower()}/"
        if os.path.exists(log_dir):
            results = load_results(log_dir)
            x, y = ts2xy(results, 'timesteps')
            
            plt.figure(figsize=(12, 4))
            
            # Plot episode rewards
            plt.subplot(1, 2, 1)
            plt.plot(x, y)
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title(f'{algorithm} - Episode Rewards During Training')
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
            plt.title(f'{algorithm} - Episode Lengths During Training')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{algorithm.lower()}_training_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"{algorithm} 训练结果图表已保存为 {algorithm.lower()}_training_results.png")
        else:
            print(f"未找到 {algorithm} 训练日志文件夹: {log_dir}")
            
    except Exception as e:
        print(f"绘制训练结果时出错: {e}")


# Keep backward compatibility
train_double_pendulum = train_cart_double_pendulum


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="小车二级摆强化学习训练和测试")
    parser.add_argument("--mode", choices=["train", "test", "plot"], default="train",
                       help="运行模式: train(训练), test(测试), plot(绘图)")
    parser.add_argument("--algorithm", choices=["PPO", "SAC", "DDPG"], default="PPO",
                       help="强化学习算法: PPO, SAC, DDPG (默认: PPO)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="训练时间步数 (默认: 1000000)")
    parser.add_argument("--model_path", default=None,
                       help="模型文件路径 (测试模式使用，如果未指定则自动推断)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="测试回合数 (测试模式使用，默认: 5)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Create algorithm-specific directories
    for alg in ["ppo", "sac", "ddpg"]:
        os.makedirs(f"models/{alg}", exist_ok=True)
        os.makedirs(f"logs/{alg}", exist_ok=True)
        os.makedirs(f"tensorboard_logs/{alg}", exist_ok=True)
    
    if args.mode == "train":
        model, eval_env = train_cart_double_pendulum(
            algorithm=args.algorithm, 
            total_timesteps=args.timesteps
        )
        eval_env.close()
        print(f"\n{args.algorithm} 训练完成！")
        print(f"运行 'python train.py --mode test --algorithm {args.algorithm}' 来测试训练好的模型")
        print(f"运行 'python train.py --mode plot --algorithm {args.algorithm}' 来查看训练结果图表")
        
    elif args.mode == "test":
        # Auto-infer model path if not provided
        if args.model_path is None:
            args.model_path = f"models/{args.algorithm.lower()}/cart_double_pendulum_{args.algorithm.lower()}_final.zip"
        
        if os.path.exists(args.model_path):
            test_trained_model(args.model_path, args.episodes, args.algorithm)
        else:
            print(f"错误: 找不到模型文件 {args.model_path}")
            print(f"请先运行训练模式: python train.py --mode train --algorithm {args.algorithm}")
            
    elif args.mode == "plot":
        plot_training_results(args.algorithm) 