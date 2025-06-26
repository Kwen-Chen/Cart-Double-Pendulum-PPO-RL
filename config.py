"""
Cart Double Pendulum Multi-Algorithm RL Configuration - 精简版
"""

import os

# ==================== 基础目录 ====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 算法配置 ====================
ALGORITHMS = {
    'PPO': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'clip_range': 0.2,
        'n_envs': 4,  # 并行环境数
        'tensorboard_log': './tensorboard_logs/PPO/'
    },
    
    'SAC': {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'ent_coef': 'auto',
        'n_envs': 1,  # 单环境
        'tensorboard_log': './tensorboard_logs/SAC/'
    },
    
    'DDPG': {
        'learning_rate': 1e-3,
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'batch_size': 128,
        'tau': 0.005,
        'gamma': 0.99,
        'action_noise_sigma': 0.1,
        'n_envs': 1,  # 单环境
        'tensorboard_log': './tensorboard_logs/DDPG/'
    }
}

# ==================== 环境配置 ====================
ENV_CONFIG = {
    'max_steps': 1000,
    'max_force': 20.0,
    'gravity': 9.81,
    'cart_mass': 0.1,
    'pendulum1_mass': 2.0,
    'pendulum2_mass': 0.5,
    'pendulum1_length': 0.5,
    'pendulum2_length': 1.2,
}

# ==================== 训练配置 ====================
TRAINING_CONFIG = {
    'total_timesteps': 1000000,
    'eval_freq': 10000,
    'device': 'auto',
    'verbose': 1
}

# ==================== 简单函数 ====================
def create_dirs():
    """创建必要的目录"""
    dirs = ['models/ppo', 'models/sac', 'models/ddpg', 
            'logs/ppo', 'logs/sac', 'logs/ddpg',
            'tensorboard_logs/PPO', 'tensorboard_logs/SAC', 'tensorboard_logs/DDPG']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("目录创建完成！")

def get_model_path(algorithm):
    """获取模型路径"""
    return f"models/{algorithm.lower()}/cart_double_pendulum_{algorithm.lower()}_final.zip"

if __name__ == "__main__":
    create_dirs()
    print("支持的算法:", list(ALGORITHMS.keys())) 