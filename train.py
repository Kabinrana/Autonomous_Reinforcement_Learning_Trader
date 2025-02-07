"""
train.py

This is the main training script for the RL trading model using Stable Baselines3.
It loads configuration parameters from config.yaml, creates the custom trading environment,
and trains a PPO model. The script supports fine-tuning an existing model,
timestamped model saving, and detailed logging of performance metrics.
It also respects the render_mode setting from the config so that in "human" mode,
the environment renders detailed candlestick charts, while in "fast" mode no rendering occurs.
"""

import os
import random
import argparse
from datetime import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
from time import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Import the custom trading environment.
from trading_env import TradingDataEnv


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    If any required key is missing, fill it with a default value.
    """
    default_config = {
        "data_path": "data.csv",
        "trading": {
            "initial_balance": 500.0,
            "target_balance": 5000.0,
            "margin_requirement": 100.0,
            "commission": 1.00
        },
        "environment": {
            "observation_space_dynamic": True,
            "timeframes": ["1m", "5m", "1h"],
            "enable_logging": True,
            "render_mode": "fast"  # Options: "human", "fast", or None
        },
        "reward": {
            "clipping": False,
            "max_pnl_factor": 10
        },
        "ppo": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 500000
        },
        "logging": {
            "experiment_name": "default_experiment",
            "model_dir": "models/",
            "eval_dir": "evals/",
            "log_dir": "logs/",
            "save_model_freq": 10000,
            "eval_freq": 5000
        }
    }
    try:
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)
        loaded_config = validate_config(loaded_config, default_config)
        return loaded_config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default settings.")
        return default_config


def validate_config(config, default_config):
    """
    Validate the loaded config by ensuring all required top-level and nested keys exist.
    If a key is missing, print a warning and insert the default value.
    """
    for key, default_value in default_config.items():
        if key not in config:
            print(f"Warning: Missing '{key}' in config.yaml. Using default value.")
            config[key] = default_value
        elif isinstance(default_value, dict):
            for sub_key, sub_default in default_value.items():
                if sub_key not in config[key]:
                    print(f"Warning: Missing '{key}.{sub_key}' in config.yaml. Using default value.")
                    config[key][sub_key] = sub_default
    return config


def plot_reward_distribution(rewards, save_path):
    """
    Plot and save a histogram of the reward distribution.

    Args:
        rewards (list): List of rewards from evaluation episodes.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(rewards, bins=20, alpha=0.75, color="blue", edgecolor="black")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def evaluate_additional_metrics(model, env, eval_dir):
    """
    Run multiple evaluation episodes to compute additional performance metrics.
    This includes maximum drawdown and generating a reward distribution plot.

    Args:
        model: The trained RL model.
        env: The evaluation environment.
        eval_dir (str): Directory where evaluation plots will be saved.

    Returns:
        float: Maximum drawdown observed across episodes.
    """
    equity_drawdowns = []
    all_rewards = []
    # Run 10 evaluation episodes.
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        equity_trajectory = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)
            equity_trajectory.append(info.get("equity", 0))
            if done or truncated:
                break
        all_rewards.extend(episode_rewards)
        if equity_trajectory:
            peak = max(equity_trajectory)
            drawdown = peak - min(equity_trajectory)
            equity_drawdowns.append(drawdown)
    max_drawdown = max(equity_drawdowns) if equity_drawdowns else 0
    plot_reward_distribution(all_rewards, os.path.join(eval_dir, "reward_distribution.png"))
    return max_drawdown


def main():
    # --- Parse CLI arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (overrides config)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to an existing model for fine-tuning")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Name of experiment (used for logging)")
    args = parser.parse_args()

    # --- Load configuration ---
    config = load_config("config.yaml")

    # Use experiment name from CLI if provided; otherwise, use the default from config.
    if args.experiment is None:
        args.experiment = config["logging"]["experiment_name"]

    # Data configuration.
    data_path = config["data_path"]

    # Trading parameters.
    trading_params = config["trading"]

    # Environment settings.
    env_settings = config["environment"]

    # Reward settings.
    reward_settings = config["reward"]

    # PPO Hyperparameters.
    ppo_params = config["ppo"]

    # Logging & checkpoint settings.
    logging_params = config["logging"]
    experiment = args.experiment
    model_dir = os.path.join(logging_params["model_dir"], experiment)
    eval_dir = os.path.join(logging_params["eval_dir"], experiment)
    log_dir = os.path.join(logging_params["log_dir"], experiment)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    total_timesteps = args.timesteps if args.timesteps is not None else ppo_params["total_timesteps"]

    # --- Create the environment ---
    env = TradingDataEnv(
        csv_path=data_path,
        obs_dim=10,
        initial_balance=trading_params["initial_balance"],
        render_mode=env_settings.get("render_mode", "fast")
    )
    # Update additional trading parameters.
    env.margin_requirement = trading_params["margin_requirement"]

    if env_settings.get("enable_logging", False):
        print("Environment logging enabled.")

    # Validate environment integrity.
    check_env(env, warn=True)

    # Create an evaluation environment.
    eval_env = TradingDataEnv(
        csv_path=data_path,
        obs_dim=10,
        initial_balance=trading_params["initial_balance"],
        render_mode=env_settings.get("render_mode", "fast")
    )

    # --- Fine-Tuning / Model Loading ---
    model_path = args.model_path
    default_model_path = os.path.join(model_dir, "final_ppo_model.zip")
    if model_path is None and os.path.exists(default_model_path):
        model_path = default_model_path

    if model_path is not None and os.path.exists(model_path):
        print(f"Loading existing model for fine-tuning from {model_path}...")
        loaded_model = PPO.load(model_path)
        # Ensure the policy is of type MlpPolicy.
        if loaded_model.policy_class.__name__ != "MlpPolicy":
            raise ValueError(f"Model was trained with {loaded_model.policy_class.__name__}, expected MlpPolicy.")
        model = PPO.load(model_path, env=env)
    else:
        print("No existing model found. Training from scratch...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=ppo_params["learning_rate"],
            n_steps=ppo_params["n_steps"],
            batch_size=ppo_params["batch_size"],
            gamma=ppo_params["gamma"],
            gae_lambda=ppo_params["gae_lambda"],
            clip_range=ppo_params["clip_range"],
            ent_coef=ppo_params["ent_coef"],
            vf_coef=ppo_params["vf_coef"],
            max_grad_norm=ppo_params["max_grad_norm"],
        )

    # --- Setup Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=logging_params["save_model_freq"],
        save_path=model_dir,
        name_prefix="ppo_model"
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_dir,
        log_path=eval_dir,
        eval_freq=logging_params["eval_freq"],
        deterministic=True,
        render=False
    )

    # --- Begin Training ---
    print("Starting training...")
    start_time = time()
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    end_time = time()
    execution_time = end_time - start_time

    # --- Evaluate the Model ---
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    sharpe_ratio = mean_reward / std_reward if std_reward > 0 else 0
    max_drawdown = evaluate_additional_metrics(model, eval_env, eval_dir)
    
    print(f"Evaluation Results - Mean Reward: {mean_reward:.2f}, Std Dev: {std_reward:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}")
    
    training_summary_path = os.path.join(eval_dir, "training_summary.txt")
    with open(training_summary_path, "w") as f:
        f.write(f"Total Timesteps: {total_timesteps}\n")
        f.write(f"PPO Hyperparameters: {ppo_params}\n")
        f.write(f"Mean Reward: {mean_reward:.2f}, Std Dev: {std_reward:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}\n")
        f.write(f"Training Execution Time: {execution_time:.2f} seconds\n")
    
    # --- Save the Final Model with a Timestamp ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_model_path = os.path.join(model_dir, f"final_ppo_model_{timestamp}.zip")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
