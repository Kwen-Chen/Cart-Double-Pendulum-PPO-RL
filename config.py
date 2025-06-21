"""
Configuration file for Cart Double Pendulum PPO Reinforcement Learning

This file contains all hyperparameters and environment parameters
used in the project for easy configuration and reproducibility.
"""

import os


class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # ==================== Directory Settings ====================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    TENSORBOARD_LOGS_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")
    ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
    
    # ==================== Environment Parameters ====================
    class Environment:
        # Physical parameters
        GRAVITY = 9.81          # m/s²
        CART_MASS = 0.1         # kg
        PENDULUM1_MASS = 2.0    # kg
        PENDULUM2_MASS = 0.5    # kg
        PENDULUM1_LENGTH = 0.5  # m
        PENDULUM2_LENGTH = 1.2  # m
        
        # Control parameters
        MAX_FORCE = 20.0        # N
        MAX_STEPS = 1000        # Maximum steps per episode
        
        # Track parameters
        TRACK_LENGTH = 8.0      # m
        X_THRESHOLD = 4.0       # m (track boundary)
        
        # Simulation parameters
        TIME_STEP = 0.01        # s
        
        # State bounds
        MAX_POSITION = 4.0      # m
        MAX_VELOCITY = 10.0     # m/s
        MAX_ANGLE = 3.14159     # radians (π)
        MAX_ANGULAR_VELOCITY = 8 * 3.14159  # rad/s
        
        # Rendering parameters
        RENDER_MODE = "human"   # "human" or None
        SCREEN_WIDTH = 1000     # pixels
        SCREEN_HEIGHT = 600     # pixels
        FPS = 50               # frames per second
    
    # ==================== PPO Training Parameters ====================
    class PPO:
        # Algorithm parameters
        LEARNING_RATE = 3e-4
        N_STEPS = 2048
        BATCH_SIZE = 64
        N_EPOCHS = 10
        GAMMA = 0.99           # Discount factor
        GAE_LAMBDA = 0.95      # GAE parameter
        CLIP_RANGE = 0.2       # PPO clipping parameter
        ENT_COEF = 0.01        # Entropy coefficient
        
        # Training parameters
        TOTAL_TIMESTEPS = 500000
        N_ENVS = 4             # Number of parallel environments
        EVAL_FREQ = 10000      # Evaluation frequency
        
        # Model architecture
        POLICY_TYPE = "MlpPolicy"
        NET_ARCH = None        # Use default MLP architecture
        
        # Device settings
        DEVICE = "auto"        # "auto", "cuda", or "cpu"
        
        # Logging
        VERBOSE = 1
        TENSORBOARD_LOG = "./tensorboard_logs/"
    
    # ==================== Reward Function Parameters ====================
    class Reward:
        # Reward weights
        UPRIGHT_WEIGHT = 1.0       # Weight for upright reward
        POSITION_PENALTY = 0.1     # Weight for position penalty
        VELOCITY_PENALTY = 0.001   # Weight for velocity penalty
        SURVIVAL_REWARD = 0.1      # Reward for each step
        
        # Reward calculation parameters
        UPRIGHT_SCALE = 3.0        # Scale factor for upright reward
    
    # ==================== Evaluation Parameters ====================
    class Evaluation:
        N_EVAL_EPISODES = 10
        EVAL_RENDER = True
        EVAL_DETERMINISTIC = True
        EVAL_MAX_STEPS = 2000
        
        # Performance thresholds
        SUCCESS_THRESHOLD = 900    # Minimum reward for success
        BALANCE_THRESHOLD = 1000   # Minimum steps for balance success
    
    # ==================== File Paths ====================
    class Paths:
        # Model files
        BEST_MODEL = os.path.join(MODELS_DIR, "best_model.zip")
        FINAL_MODEL = os.path.join(MODELS_DIR, "cart_double_pendulum_ppo_final.zip")
        
        # Asset files
        TRAINING_RESULTS_PNG = os.path.join(ASSETS_DIR, "training_results.png")
        BEFORE_TRAINING_GIF = os.path.join(ASSETS_DIR, "训练前.gif")
        AFTER_TRAINING_GIF = os.path.join(ASSETS_DIR, "训练后.gif")
    
    # ==================== Logging Configuration ====================
    class Logging:
        LOG_LEVEL = "INFO"
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Log file names
        TRAINING_LOG = "training.log"
        EVALUATION_LOG = "evaluation.log"
    
    # ==================== Experimental Settings ====================
    class Experiment:
        # Random seed for reproducibility
        RANDOM_SEED = 42
        
        # Experiment tracking
        EXPERIMENT_NAME = "CartDoublePendulumPPO"
        PROJECT_NAME = "Cart-Double-Pendulum-RL"
        
        # Model checkpointing
        SAVE_FREQ = 50000      # Save model every N steps
        KEEP_N_CHECKPOINTS = 5 # Number of checkpoints to keep


# Create config instance
config = Config()


# Convenience functions for directory creation
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        config.MODELS_DIR,
        config.LOGS_DIR,
        config.TENSORBOARD_LOGS_DIR,
        config.ASSETS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Created necessary directories.")


def print_config():
    """Print current configuration settings."""
    print("=" * 60)
    print("Cart Double Pendulum PPO Configuration")
    print("=" * 60)
    
    print("\nEnvironment Parameters:")
    env_params = vars(config.Environment)
    for key, value in env_params.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    print("\nPPO Parameters:")
    ppo_params = vars(config.PPO)
    for key, value in ppo_params.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    print("\nReward Parameters:")
    reward_params = vars(config.Reward)
    for key, value in reward_params.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    print("=" * 60)


if __name__ == "__main__":
    create_directories()
    print_config() 