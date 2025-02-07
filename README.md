# Autonomous_Reinforcement_Learning_Trader
Autonomous Reinforcement Learning Trader (ARLT) is a high-performance, PPO-based reinforcement learning system designed to consistently achieve 10x profit growth within a single trading day. Utilizes a custom built gymnasium reinforcement learning envrionment to train a PPO model to aim to achieve the desired target balance. 

Welcome to the **RL Trading Environment and PPO Training** project. This repository implements a custom Gymnasium trading environment along with a training pipeline using Stable Baselines3's PPO algorithm. The environment is designed to simulate realistic trading on a per-day basis using 1-minute OHLC data and supports both detailed ("human") and fast (non-rendering) modes. The model is trained on randomly sampled trading days to promote learning across diverse market conditions and to avoid overfitting to sequential data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training Instructions](#training-instructions)
- [Evaluation and Logging](#evaluation-and-logging)
- [Rendering Modes](#rendering-modes)
- [Logical Flow and Expected Outcomes](#Logical-Flow-and-Expected-Outcomes)
- [License](#license)

## Overview

This project implements a reinforcement learning (RL) framework for trading. Key components include:

- **Custom Trading Environment** (`trading_env.py`):  
  - Loads and processes 1-minute OHLC market data.
  - Computes technical indicators (EMA, Bollinger Bands, ATR).
  - Supports realistic trade management, including profit target enforcement and margin-based position sizing.
  - Randomly samples trading days to expose the agent to varied market conditions.
  - Provides two render modes: 
    - `"human"` for detailed candlestick charts with annotations.
    - `"fast"` (or `None`) for training without rendering.

- **Training Script** (`train.py`):  
  - Loads parameters from a configurable `config.yaml`.
  - Supports fine-tuning of an existing model with policy validation.
  - Logs extensive training and evaluation metrics (reward distribution, Sharpe Ratio, maximum drawdown, etc.).
  - Saves model checkpoints and final models with timestamped filenames.

- **Configuration** (`config.yaml`):  
  - Centralizes all project parameters (data paths, trading parameters, PPO hyperparameters, logging settings, etc.).
  - Allows easy switching between experiments via an experiment name.

- **Dependencies**:  
  - Listed in `requirements.txt` to ensure compatibility and smooth execution.

## Features

- **Random Day Sampling:**  
  Each episode uses a randomly selected trading day (from 35 days of pre-purchased data) to ensure varied market conditions and to avoid overfitting.

- **Profit Target Enforcement:**  
  Episodes terminate early with a bonus reward if the account equity reaches or exceeds a set profit target.

- **Detailed and Fast Rendering:**  
  The environment can render detailed candlestick charts (human mode) or disable rendering (fast mode) to speed up training.

- **Modular Configuration and Logging:**  
  All parameters are externalized in `config.yaml`. Experiment-specific logs, models, and evaluations are organized into separate directories.

- **Fine-Tuning Support:**  
  The training script automatically detects and loads existing models for further training (with policy compatibility checks).

## File Structure
```
trading-rl-project/ 
├── LICENSE # Open-source license file. 
├── README.md # This file. 
├── config.yaml # Configuration file for parameters and hyperparameters. 
├── requirements.txt # List of Python dependencies. 
├── train.py # Main training script. 
├── trading_env.py # Custom Gymnasium environment for trading. 
└── utils/ # (Optional) Additional utility modules. 
        ├── init.py
        └── evaluation_utils.py # (Optional) Functions for advanced evaluation metrics.
```
## Installation

1. **Clone the Repository:**
   ```bash
     git clone https://github.com/your_username/trading-rl-project.git
     cd trading-rl-project
  
2. Create and Activate a Virtual Environment:
   
   On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ``` 
   On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
4. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Configuration
All configurable parameters are located in the config.yaml file. Key sections include:

 * Data Configuration:
   - Set the path to your market data CSV file.

 * Trading Parameters:
   - Define the initial account balance, profit target, margin requirements, and commission.

 * Environment Settings:
   - Configure the render mode ("human" for detailed charts or "fast" for no rendering), observation space dynamics, and supported timeframes.

 * PPO Hyperparameters:
   - Adjust learning rate, batch size, total training timesteps, etc.

 * Logging Settings:
   - Set the default experiment name and directories for models, evaluations, and logs.

## Training Instructions
To begin training the model, navigate to the repository root and run one of the following commands:

 * Default Training (using settings in config.yaml):
     ```bash
     python train.py
     ```
 * Override Total Timesteps (for quick tests or debugging):
     ```bash
     python train.py --timesteps 10000
     ```
 * Fine-Tune an Existing Model:
    ```bash
    python train.py --model_path models/default_experiment/final_ppo_model.zip
    ```
 * Specify a Custom Experiment Name:
    ```bash
    python train.py --experiment "my_experiment"
    ```
## Evaluation and Logging
During training, checkpoints and evaluation logs will be saved under:
 * ```models/<experiment_name>/```
 * ```evals/<experiment_name>/```
 * ```logs/<experiment_name>/```

After training, the script outputs key metrics such as Mean Reward, Standard Deviation, Sharpe Ratio, and Maximum Drawdown. A reward distribution plot is saved to help diagnose training stability.

## Rendering Modes
 * Human Mode:
    If "render_mode": "human" is set in config.yaml (or passed via environment arguments), the environment will continuously render a detailed candlestick chart with technical indicators and annotated trade points during     each step. This mode is useful for demonstration and debugging but may slow training.

 * Fast Mode:
    If "render_mode": "fast" (or left as None), no rendering occurs. This mode is recommended for efficient training.


## Logical Flow and Expected Outcomes
1. Training Flow:
     * **Data Loading & Preprocessing:**
        * The environment loads a CSV file containing 1-minute OHLC data. The "Date" column is converted to datetime objects (handling formats like "MM/DD/YYYY HH:MM:SS"). Technical indicators (EMA, Bollinger Bands, ATR)           are computed, and normalization statistics are calculated.
     * **Episode Reset:**
        * Each episode randomly selects one trading day from the dataset (ensuring non-sequential training). Trading parameters (balance, positions) are reset.
     * **Step Function:**
        * For every time step (minute), the environment processes the agent’s action:
          * Buy/Sell: Closes any opposing position, then opens a new position using available free margin.
          * Hold: Maintains the current position.
          * Close All: Liquidates any open position. The environment calculates realized and unrealized PnL, updates the balance, and adds bonus rewards if the target balance is reached.
     * **Termination:**
        * The episode terminates when the end of the day is reached or when the profit target is achieved.
2. Rendering Flow (Human Mode):
     * When render_mode is set to "human", the environment uses Matplotlib to render a candlestick chart:
          * The x-axis displays actual time converted from the "Date" column (formatted as hour:minute).
          * Each candlestick represents one minute of trading data.
          * Trade markers are plotted at the exact time and price where trades were executed.
          * The chart is updated at each step.
     * In "fast" mode (or if render_mode is None), no rendering occurs—ensuring training speed is maximized.
3. Expected Outcomes:
     * During Training:
        The RL agent receives observations that include normalized market data and account metrics. Episodes end at the end of a trading day or upon reaching the profit target. In human mode, you should see an updating           candlestick chart that accurately reflects the time from your CSV file, with trade markers at the correct time positions.
     * Post-Training:
        The model should generalize across randomly sampled days, continuously transferring learning between episodes. Evaluation metrics (such as profit target attainment, reward distribution, etc.) will be computed and         logged.

## License
This project is licensed under the terms specified in the LICENSE file.

## Final Notes
Ensure that your market data CSV file is formatted correctly with at least the following columns: Date, Open, High, Low, Close. For any questions or further assistance, please refer to the project documentation or contact the repository maintainer.

# Happy Training!
