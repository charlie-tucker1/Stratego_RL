"""
PPO Training Script for Stratego RL Environment

This script trains a PPO agent to play Stratego against a scripted bot opponent.
Uses action masking to handle the large (3600) action space efficiently.

The agent learns from:
- Material rewards (piece captures/losses)
- Strategic bonuses (spy kills marshal, miner defuses bomb)
- Territorial rewards (advancing into enemy territory)
- Terminal rewards (win/loss)
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch
import torch.nn as nn
import gymnasium as gym

from stratego_logic import StrategoEnv


# Training Configuration
TOTAL_TIMESTEPS = 2_000_000  # Total training steps
SAVE_FREQ = 50_000           # Save model every N steps
LOG_DIR = "./logs"           # TensorBoard logs
MODEL_DIR = "./models"       # Saved model checkpoints
EVAL_FREQ = 10_000           # Evaluate every N steps
N_EVAL_EPISODES = 10         # Episodes per evaluation

# PPO Hyperparameters
LEARNING_RATE = 3e-4         # Learning rate
N_STEPS = 2048               # Steps per update
BATCH_SIZE = 64              # Minibatch size
N_EPOCHS = 10                # Optimization epochs per update
GAMMA = 0.99                 # Discount factor
GAE_LAMBDA = 0.95            # GAE lambda
CLIP_RANGE = 0.2             # PPO clipping parameter


class StrategoCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Stratego's 27-channel observation.

    Architecture:
    - Conv2D: 27 channels -> 64 channels (3x3 kernel)
    - Conv2D: 64 channels -> 64 channels (3x3 kernel)
    - Conv2D: 64 channels -> 128 channels (3x3 kernel)
    - Flatten
    - FC: -> 512 features

    This should preserve spatial structure while extracting meaningful features.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 27 channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class MetricsCallback(BaseCallback):
    """
    Custom callback for logging training metrics.

    Tracks:
    - Win rate against bot
    - Average episode reward
    - Average episode length
    - Total games played
    """

    def __init__(self, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if evaluation should run
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()

        # Log episode metrics when episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        return True

    def _evaluate(self):
        """Run evaluation episodes and log metrics"""
        print(f"\n{'='*50}")
        print(f"Evaluation at step {self.n_calls}")
        print(f"{'='*50}")

        wins = 0
        total_reward = 0
        total_length = 0

        # Create evaluation environment
        eval_env = StrategoEnv()

        for episode in range(self.n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # Get action mask
                action_mask = info.get("action_mask", np.ones(3600))

                # Predict action
                action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)

                # Step environment
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1

                if done or truncated:
                    break

            # Track results
            total_reward += episode_reward
            total_length += episode_length

            if eval_env.game.winner == 'red':
                wins += 1

        # Calculate metrics
        win_rate = wins / self.n_eval_episodes
        avg_reward = total_reward / self.n_eval_episodes
        avg_length = total_length / self.n_eval_episodes

        # Log to tensorboard
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/avg_reward", avg_reward)
        self.logger.record("eval/avg_length", avg_length)

        # Print results
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Avg Reward: {avg_reward:.2f}")
        print(f"Avg Length: {avg_length:.1f} steps")
        print(f"{'='*50}\n")


def mask_fn(env):
    """
    Extract action mask from environment.
    Required for MaskablePPO to handle invalid actions.
    """
    # Unwrap to get the base StrategoEnv (Monitor wrapper doesn't expose action_mask)
    return env.unwrapped.action_mask


def make_env():
    """Create and wrap the environment"""
    env = StrategoEnv()
    env = Monitor(env)  # Monitor wrapper for logging
    env = ActionMasker(env, mask_fn)  # Add action masking
    return env


def train():
    """Main training function"""

    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Initializing Stratego RL Training")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # Create environment (action masking already in make_env)
    env = DummyVecEnv([make_env])

    # Policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=StrategoCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # Initialize PPO agent
    model = MaskablePPO(
        "CnnPolicy",                    # Use CNN policy with custom feature extractor
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto"                   # Automatically use GPU if available
    )

    print("\nModel Architecture:")
    print(model.policy)
    print()

    # Setup callbacks
    metrics_callback = MetricsCallback(
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="stratego_ppo"
    )

    # Train the agent
    print("Starting training")
    print("Monitor progress with: tensorboard --logdir ./logs\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[metrics_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except ValueError as e:
        if "Simplex()" in str(e):
            print("\n\nTraining stopped due to numerical stability issue in action masking")
            print("This is a rare edge case - your checkpoints are saved!")
            print(f"Error: {e}")
        else:
            raise

    # Save final model
    final_path = os.path.join(MODEL_DIR, "stratego_ppo_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    return model


def evaluate_model(model_path, n_episodes=100):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
    """
    print(f"Evaluating model: {model_path}")
    print(f"Running {n_episodes} episodes\n")

    # Load model
    model = MaskablePPO.load(model_path)

    # Create environment
    env = StrategoEnv()

    wins = 0
    total_reward = 0
    total_length = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action_mask = info.get("action_mask", np.ones(3600))
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if done or truncated:
                break

        total_reward += episode_reward
        total_length += episode_length

        if env.game.winner == 'red':
            wins += 1

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - Win rate: {wins/(episode+1)*100:.1f}%")

    # Print final statistics
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"Win Rate: {wins/n_episodes*100:.1f}%")
    print(f"Avg Reward: {total_reward/n_episodes:.2f}")
    print(f"Avg Episode Length: {total_length/n_episodes:.1f} steps")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate Stratego PPO agent")
    parser.add_argument("--eval", type=str, help="Path to model to evaluate")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes for evaluation")

    args = parser.parse_args()

    if args.eval:
        # Evaluation mode
        evaluate_model(args.eval, args.episodes)
    else:
        # Training mode
        train()
