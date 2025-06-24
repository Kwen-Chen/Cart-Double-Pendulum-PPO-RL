import pygame
import numpy as np
import math
from typing import Tuple, Optional


class CartDoublePendulumEnv:
    """
    Cart with double pendulum environment using pygame for visualization and physics simulation.
    A cart with two wheels that can move horizontally, with a double pendulum mounted on top.
    The goal is to balance the double pendulum by moving the cart left and right.
    """
    
    def __init__(self, render_mode: str = "human", max_steps: int = 1000):
        # Environment parameters
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Physical parameters
        self.g = 9.81  # gravity
        self.M = 0.1   # mass of cart
        self.m1 = 2  # mass of first pendulum
        self.m2 = 0.5  # mass of second pendulum
        self.l1 = 0.5  # length of first pendulum
        self.l2 = 1.2  # length of second pendulum
        self.dt = 0.01  # time step
        
        # Cart parameters
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.wheel_radius = 0.1
        
        # Control parameters
        self.max_force = 20.0  # maximum horizontal force applied to cart
        
        # Track boundaries
        self.track_length = 8.0  # total track length
        self.x_threshold = self.track_length / 2  # cart position limits
        
        # State space: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        # x: cart position
        # theta1: angle of first pendulum from vertical (radians)
        # theta2: angle of second pendulum from vertical (radians)
        self.state_dim = 6
        self.action_dim = 1
        
        # State bounds
        self.max_position = self.x_threshold
        self.max_velocity = 10.0
        self.max_angle = np.pi
        self.max_angular_velocity = 8 * np.pi
        
        # Pygame settings
        self.screen_width = 1000
        self.screen_height = 600
        self.scale = 80  # pixels per meter
        self.origin = (self.screen_width // 2, self.screen_height * 3 // 4)
        
        # Initialize pygame
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Cart with Double Pendulum")
            self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Initialize with small random perturbations around upright position
        self.x = np.random.uniform(-0.05, 0.05)  # cart position (smaller range)
        self.x_dot = 0.0  # cart starts at rest
        
        # Both pendulums start nearly upright (vertical)
        self.theta1 = np.random.uniform(-0.05, 0.05)  # small angle around vertical
        self.theta1_dot = 0.0  # first pendulum starts at rest
        self.theta2 = np.random.uniform(-0.05, 0.05)  # small angle around vertical  
        self.theta2_dot = 0.0  # second pendulum starts at rest
        
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        return np.array([
            self.x,
            self.x_dot,
            self.theta1,
            self.theta1_dot,
            self.theta2,
            self.theta2_dot
        ], dtype=np.float32)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Horizontal force applied to the cart (normalized between -1 and 1)
            
        Returns:
            observation: Next state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Clip and scale action
        force = np.clip(action, -1.0, 1.0) * self.max_force
        
        # Update physics
        self._update_physics(force)
        
        # Get reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        self.step_count += 1
        
        info = {
            'x': self.x,
            'theta1': self.theta1,
            'theta2': self.theta2,
            'step': self.step_count
        }
        
        return self._get_state(), reward, done, info
    
    def _update_physics(self, force: float):
        """Update the physics simulation using Runge-Kutta integration."""
        # Current state
        state = np.array([self.x, self.x_dot, self.theta1, self.theta1_dot, self.theta2, self.theta2_dot])
        
        # Runge-Kutta 4th order integration
        k1 = self._dynamics(state, force)
        k2 = self._dynamics(state + 0.5 * self.dt * k1, force)
        k3 = self._dynamics(state + 0.5 * self.dt * k2, force)
        k4 = self._dynamics(state + self.dt * k3, force)
        
        new_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Update state variables
        self.x = new_state[0]
        self.x_dot = new_state[1]
        self.theta1 = new_state[2]
        self.theta1_dot = new_state[3]
        self.theta2 = new_state[4]
        self.theta2_dot = new_state[5]
        
        # Normalize angles to [-pi, pi]
        self.theta1 = self._normalize_angle(self.theta1)
        self.theta2 = self._normalize_angle(self.theta2)
    
    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        """
        Compute the dynamics of the cart with double pendulum system.
        Returns the derivatives [x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
        """
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        
        # Precompute trigonometric functions
        s1, c1 = np.sin(theta1), np.cos(theta1)
        s2, c2 = np.sin(theta2), np.cos(theta2)
        s12 = np.sin(theta1 - theta2)
        c12 = np.cos(theta1 - theta2)
        
        # Mass matrix elements for the system
        # State vector: [x, theta1, theta2]
        # Mass matrix M * [x_ddot, theta1_ddot, theta2_ddot] = F
        
        M11 = self.M + self.m1 + self.m2
        M12 = (self.m1 + self.m2) * self.l1 * c1
        M13 = self.m2 * self.l2 * c2
        
        M21 = (self.m1 + self.m2) * self.l1 * c1
        M22 = (self.m1 + self.m2) * self.l1**2
        M23 = self.m2 * self.l1 * self.l2 * c12
        
        M31 = self.m2 * self.l2 * c2
        M32 = self.m2 * self.l1 * self.l2 * c12
        M33 = self.m2 * self.l2**2
        
        # Force vector elements
        F1 = (force + (self.m1 + self.m2) * self.l1 * theta1_dot**2 * s1 
              + self.m2 * self.l2 * theta2_dot**2 * s2)
        
        F2 = -(-(self.m1 + self.m2) * self.g * self.l1 * s1 
              + self.m2 * self.l1 * self.l2 * theta2_dot**2 * s12)
        
        F3 = -(-self.m2 * self.g * self.l2 * s2 
              - self.m2 * self.l1 * self.l2 * theta1_dot**2 * s12)
        
        # Construct mass matrix and force vector
        M = np.array([[M11, M12, M13],
                      [M21, M22, M23],
                      [M31, M32, M33]])
        
        F = np.array([F1, F2, F3])
        
        # Solve for accelerations: M * accelerations = F
        try:
            accelerations = np.linalg.solve(M, F)
            x_ddot, theta1_ddot, theta2_ddot = accelerations
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            x_ddot = theta1_ddot = theta2_ddot = 0.0
        
        return np.array([x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on the current state."""
        # Reward for keeping pendulums upright
        upright_reward = np.exp(-3 * (self.theta1**2 + self.theta2**2))
        
        # Penalty for cart being far from center
        position_penalty = 0.1 * self.x**2
        
        # Penalty for high velocities
        velocity_penalty = 0.001 * (self.x_dot**2 + self.theta1_dot**2 + self.theta2_dot**2)
        
        # Small positive reward for surviving
        survival_reward = 0.1
        
        total_reward = upright_reward + survival_reward - position_penalty - velocity_penalty
        
        return total_reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Episode ends if:
        # 1. Maximum steps reached
        # 2. Cart goes out of bounds
        # 3. Pendulums fall too far
        # 4. Velocities become too large
        # 5. Any pendulum touches the ground
        
        if self.step_count >= self.max_steps:
            return True
        
        if abs(self.x) > self.x_threshold:
            return True
        
        if abs(self.theta1) > self.max_angle or abs(self.theta2) > self.max_angle:
            return True
        
        if (abs(self.x_dot) > self.max_velocity or 
            abs(self.theta1_dot) > self.max_angular_velocity or 
            abs(self.theta2_dot) > self.max_angular_velocity):
            return True
        
        # Check if any pendulum touches the ground
        if self._check_ground_collision():
            return True
        
        return False
    
    def _check_ground_collision(self) -> bool:
        """Check if any pendulum touches the ground."""
        # Ground level is at the cart's wheel level (origin[1])
        ground_y = self.origin[1]
        
        # Calculate cart position on screen (same as in render function)
        cart_screen_x = self.origin[0] + self.x * self.scale
        cart_screen_y = self.origin[1]
        cart_top_y = cart_screen_y - self.cart_height * self.scale
        
        # Calculate pendulum positions (corrected coordinate system)
        # When theta=0 (upright), pendulum extends upward (smaller y values)
        # First pendulum
        x1 = cart_screen_x + self.l1 * self.scale * np.sin(self.theta1)
        y1 = cart_top_y - self.l1 * self.scale * np.cos(self.theta1)
        
        # Second pendulum
        x2 = x1 + self.l2 * self.scale * np.sin(self.theta2)
        y2 = y1 - self.l2 * self.scale * np.cos(self.theta2)
        
        # Check if any pendulum tip is at or below ground level
        # Note: in pygame, larger y values are lower on screen
        if y1 >= ground_y or y2 >= ground_y:
            return True
            
        return False
    
    def render(self):
        """Render the environment using pygame."""
        if self.render_mode != "human":
            return
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw track
        track_y = self.origin[1]
        track_start = self.origin[0] - self.track_length * self.scale // 2
        track_end = self.origin[0] + self.track_length * self.scale // 2
        pygame.draw.line(self.screen, self.BLACK, 
                        (track_start, track_y), (track_end, track_y), 8)
        
        # Calculate cart position on screen
        cart_screen_x = self.origin[0] + self.x * self.scale
        cart_screen_y = self.origin[1]
        
        # Draw cart
        cart_rect = pygame.Rect(
            cart_screen_x - self.cart_width * self.scale // 2,
            cart_screen_y - self.cart_height * self.scale,
            self.cart_width * self.scale,
            self.cart_height * self.scale
        )
        pygame.draw.rect(self.screen, self.GRAY, cart_rect)
        pygame.draw.rect(self.screen, self.BLACK, cart_rect, 3)
        
        # Draw wheels
        wheel_y = cart_screen_y
        wheel1_x = cart_screen_x - self.cart_width * self.scale // 4
        wheel2_x = cart_screen_x + self.cart_width * self.scale // 4
        wheel_radius = self.wheel_radius * self.scale
        
        pygame.draw.circle(self.screen, self.DARK_GRAY, 
                          (int(wheel1_x), int(wheel_y)), int(wheel_radius))
        pygame.draw.circle(self.screen, self.BLACK, 
                          (int(wheel1_x), int(wheel_y)), int(wheel_radius), 2)
        pygame.draw.circle(self.screen, self.DARK_GRAY, 
                          (int(wheel2_x), int(wheel_y)), int(wheel_radius))
        pygame.draw.circle(self.screen, self.BLACK, 
                          (int(wheel2_x), int(wheel_y)), int(wheel_radius), 2)
        
        # Calculate pendulum positions (corrected coordinate system)
        cart_top_y = cart_screen_y - self.cart_height * self.scale
        
        # First pendulum (theta=0 means upright, extending upward)
        x1 = cart_screen_x + self.l1 * self.scale * np.sin(self.theta1)
        y1 = cart_top_y - self.l1 * self.scale * np.cos(self.theta1)
        
        # Second pendulum
        x2 = x1 + self.l2 * self.scale * np.sin(self.theta2)
        y2 = y1 - self.l2 * self.scale * np.cos(self.theta2)
        
        # Draw pendulums
        # First pendulum
        pygame.draw.line(self.screen, self.RED, 
                        (cart_screen_x, cart_top_y), (int(x1), int(y1)), 6)
        pygame.draw.circle(self.screen, self.RED, (int(x1), int(y1)), 15)
        
        # Second pendulum
        pygame.draw.line(self.screen, self.BLUE, 
                        (int(x1), int(y1)), (int(x2), int(y2)), 6)
        pygame.draw.circle(self.screen, self.BLUE, (int(x2), int(y2)), 15)
        
        # Draw reference lines (vertical)
        pygame.draw.line(self.screen, self.GREEN, 
                        (cart_screen_x, cart_top_y), 
                        (cart_screen_x, cart_top_y - 2 * self.scale), 2)
        
        # Display information
        font = pygame.font.Font(None, 36)
        text_lines = [
            f"Step: {self.step_count}",
            f"Cart X: {self.x:.3f} m",
            f"Theta1: {self.theta1:.3f} rad",
            f"Theta2: {self.theta2:.3f} rad",
            f"Reward: {self._calculate_reward():.3f}"
        ]
        
        for i, line in enumerate(text_lines):
            text = font.render(line, True, self.BLACK)
            self.screen.blit(text, (10, 10 + i * 30))
        
        # Draw track boundaries
        pygame.draw.line(self.screen, self.RED,
                        (track_start, track_y - 20), (track_start, track_y + 20), 4)
        pygame.draw.line(self.screen, self.RED,
                        (track_end, track_y - 20), (track_end, track_y + 20), 4)
        
        pygame.display.flip()
        self.clock.tick(50)  # 50 FPS
    
    def close(self):
        """Close the pygame window."""
        if self.render_mode == "human":
            pygame.quit()


# Keep backward compatibility
DoublePendulumEnv = CartDoublePendulumEnv


# Test the environment
if __name__ == "__main__":
    env = CartDoublePendulumEnv(render_mode="human")
    
    # Test with random actions
    state = env.reset()
    done = False
    total_reward = 0
    
    try:
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            # Random action
            action = np.random.uniform(-1, 1)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            env.render()
            
            if done:
                print(f"Episode finished. Total reward: {total_reward:.2f}")
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        env.close() 