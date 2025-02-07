"""
trading_env.py

This module implements the custom Gymnasium environment for trading, named TradingDataEnv.
It loads 1-minute OHLC market data from a CSV file, computes technical indicators,
and simulates trading on a per-day basis.

Key Features:
    - Randomly selects a trading day (from the dataset) for each episode.
    - Processes actions to simulate buying, selling, holding, and closing positions.
    - Enforces a daily profit target (if reached, the episode terminates with bonus reward).
    - Renders a candlestick chart using the actual time from the CSV's "Date" column as the x-axis.
      * In "human" render mode, detailed charts are displayed (with x-axis showing hour:minute).
      * In "fast" mode or if render_mode is None, rendering is disabled.
    - Uses proper DataFrame indexing (.loc) to avoid SettingWithCopyWarning.
    
Author: Advanced RL Trading Team
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces

class TradingDataEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "fast", None],
        "render_fps": 30
    }

    def __init__(self, csv_path, obs_dim=10, initial_balance=10000.0, target_balance=None, render_mode="fast"):
        """
        Initialize the trading environment.

        Args:
            csv_path (str): Path to CSV file with market data.
            obs_dim (int): Dimension of the observation space.
            initial_balance (float): Starting account balance.
            target_balance (float): Daily profit target; if None, defaults to 2x initial_balance.
            render_mode (str): Render mode: "human" for detailed candlestick charts, "fast" for no rendering.
        """
        super(TradingDataEnv, self).__init__()

        # Set render mode.
        self.render_mode = render_mode

        # Define the discrete action space: 
        # 0: Buy Max, 1: Sell Max, 2: Hold, 3: Close All.
        self.action_space = spaces.Discrete(4)

        # Define the observation space as a continuous box.
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_dim,), dtype=np.float32)

        # Trading parameters.
        self.initial_balance = initial_balance
        self.balance = initial_balance
        # If no target_balance provided, default to 2x the initial balance.
        self.target_balance = target_balance if target_balance is not None else initial_balance * 2
        self.position = 0              # Positive for long, negative for short.
        self.entry_price = None        # Price when the current position was opened.
        self.time_in_position = 0      # Number of steps the current position is held.
        self.trade_history = []        # List to record trade events (with time and price).

        # Tick parameters (for profit/loss calculations).
        self.tick_size = 0.25
        self.tick_value = 2.50

        # Margin requirement per contract.
        self.margin_requirement = 100.0

        # Load market data and compute technical indicators.
        self.data = self._load_data(csv_path)
        self.data = self._calculate_indicators(self.data)
        self.feature_means, self.feature_stds = self._calculate_feature_scaling(self.data)

        # Retrieve the list of unique trading days.
        self.trading_days = self.data['Date'].dt.date.unique()
        self.day_index = 0  # Will be randomized during reset.

        # Get data for the current trading day.
        self.day_data = self._get_day_data()
        # Use .loc to update the 'ATR_14' column to avoid SettingWithCopyWarning.
        self.day_data.loc[:, 'ATR_14'] = np.clip(self.day_data['ATR_14'], 0.1, 5.0)
        self.current_step = 0

        # If in human render mode, initialize rendering objects.
        if self.render_mode == "human":
            self._init_rendering()

    def _init_rendering(self):
        """
        Initialize Matplotlib objects (figure and axes) for rendering a candlestick chart.
        Sets up date conversion so that the x-axis displays actual time (hour:minute).
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.ion()  # Enable interactive mode.
        self.fig.show()
        self.fig.canvas.draw()

    def _load_data(self, csv_path):
        """
        Load market data from a CSV file and convert the "Date" column.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            DataFrame: Sorted DataFrame with "Date" as datetime objects.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")
        # Convert "Date" column to datetime; handles formats like "MM/DD/YYYY HH:MM:SS".
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _calculate_indicators(self, df):
        """
        Compute technical indicators and add them as new columns.

        Args:
            df (DataFrame): Raw market data.

        Returns:
            DataFrame: DataFrame with added indicator columns.
        """
        # 10-period Exponential Moving Average (EMA).
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

        # 20-period Moving Average and Bollinger Bands.
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['MA_20'] + 2 * df['STD_20']
        df['BB_lower'] = df['MA_20'] - 2 * df['STD_20']

        # 14-period Average True Range (ATR).
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        df['ATR_14'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()

        # Fill any NaN values.
        df.fillna(method='bfill', inplace=True)
        return df

    def _calculate_feature_scaling(self, df):
        """
        Calculate mean and standard deviation for normalization of selected features.

        Args:
            df (DataFrame): Data with technical indicators.

        Returns:
            tuple: (means, stds) for features ['Close', 'EMA_10', 'BB_upper', 'MA_20', 'BB_lower', 'ATR_14'].
        """
        features = ['Close', 'EMA_10', 'BB_upper', 'MA_20', 'BB_lower', 'ATR_14']
        means = df[features].mean()
        stds = df[features].std().replace(0, 1)
        return means, stds

    def _get_day_data(self):
        """
        Randomly select a trading day from the dataset and return its data.

        Returns:
            DataFrame: Data for the randomly selected trading day.
        """
        self.day_index = np.random.randint(0, len(self.trading_days))
        current_day = self.trading_days[self.day_index]
        day_data = self.data[self.data['Date'].dt.date == current_day].copy()
        day_data.reset_index(drop=True, inplace=True)
        return day_data

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new trading day.

        Returns:
            observation (np.array): The initial observation vector.
            info (dict): Additional info (empty in this implementation).
        """
        self.day_data = self._get_day_data()
        # Ensure ATR values are clipped using .loc to avoid warnings.
        self.day_data.loc[:, 'ATR_14'] = np.clip(self.day_data['ATR_14'], 0.1, 5.0)
        self.current_step = 0

        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.time_in_position = 0
        self.trade_history = []
        return self._next_observation(), {}

    def _next_observation(self):
        """
        Construct the observation vector by combining normalized market features and account metrics.

        Returns:
            np.array: The observation vector.
        """
        # Ensure we do not exceed day_data.
        if self.current_step >= len(self.day_data):
            self.current_step = len(self.day_data) - 1
        row = self.day_data.iloc[self.current_step]
        feature_list = []
        for feature in ['Close', 'EMA_10', 'BB_upper', 'MA_20', 'BB_lower', 'ATR_14']:
            norm_val = (row[feature] - self.feature_means[feature]) / self.feature_stds[feature]
            feature_list.append(norm_val)
        features = np.array(feature_list, dtype=np.float32)

        # Account metrics: balance, position, unrealized PnL, free margin.
        account_metrics = np.array([
            self.balance,
            self.position,
            self._get_unrealized_pnl(),
            self._get_free_margin()
        ], dtype=np.float32)
        observation = np.concatenate([features, account_metrics])
        return observation

    def _get_equity(self):
        """
        Compute the current equity (balance plus unrealized PnL).

        Returns:
            float: Total account equity.
        """
        return self.balance + self._get_unrealized_pnl()

    def _get_free_margin(self):
        """
        Compute the free margin available for new positions.

        Returns:
            float: Free margin (equity minus used margin).
        """
        used_margin = abs(self.position) * self.margin_requirement
        free_margin = self._get_equity() - used_margin
        return max(free_margin, 0)

    def _get_unrealized_pnl(self):
        """
        Compute the unrealized profit and loss of the current open position.

        Returns:
            float: Unrealized PnL (negative for loss; positive profits are not added here).
        """
        if self.position == 0 or self.entry_price is None:
            return 0.0
        current_price = self.day_data.iloc[self.current_step]['Close']
        pnl = (current_price - self.entry_price) if self.position > 0 else (self.entry_price - current_price)
        ticks = pnl / self.tick_size
        pnl_value = ticks * self.tick_value * abs(self.position)
        return pnl_value if pnl_value < 0 else 0.0

    def step(self, action):
        """
        Execute one time step (typically 1 minute) of trading.

        Actions:
            0: Buy Max (go long)
            1: Sell Max (go short)
            2: Hold (maintain current position)
            3: Close All (liquidate any open position)

        Returns:
            observation (np.array): Next observation vector.
            reward (float): Reward obtained from the action.
            terminated (bool): True if the episode ends (end of day or profit target reached).
            truncated (bool): True if the episode is truncated.
            info (dict): Additional environment info.
        """
        reward = 0.0
        terminated = False
        truncated = False

        row = self.day_data.iloc[self.current_step]
        current_price = row['Close']
        current_time = row['Date']  # This is a datetime object.

        # Determine contract size based on free margin.
        free_margin = self._get_free_margin()
        contract_size = int(free_margin // self.margin_requirement)
        if contract_size < 1:
            contract_size = 1

        # Process action.
        if action == 0:  # Buy Max
            if self.position < 0:
                # Close short position first.
                realized = (self.entry_price - current_price) / self.tick_size * self.tick_value * abs(self.position)
                self.balance += realized
                reward += realized
                # Record trade with time and price.
                self.trade_history.append({
                    "action": "CloseShort",
                    "price": current_price,
                    "time": current_time,
                    "realized": realized,
                    "step_index": self.current_step
                })
                self.position = 0
                self.entry_price = None
            if self.position == 0:
                self.position = contract_size
                self.entry_price = current_price
                # Record trade entry.
                self.trade_history.append({
                    "action": "Buy",
                    "price": current_price,
                    "time": current_time,
                    "step_index": self.current_step
                })

        elif action == 1:  # Sell Max
            if self.position > 0:
                # Close long position first.
                realized = (current_price - self.entry_price) / self.tick_size * self.tick_value * self.position
                self.balance += realized
                reward += realized
                self.trade_history.append({
                    "action": "CloseLong",
                    "price": current_price,
                    "time": current_time,
                    "realized": realized,
                    "step_index": self.current_step
                })
                self.position = 0
                self.entry_price = None
            if self.position == 0:
                self.position = -contract_size
                self.entry_price = current_price
                self.trade_history.append({
                    "action": "Sell",
                    "price": current_price,
                    "time": current_time,
                    "step_index": self.current_step
                })

        elif action == 2:  # Hold
            # No trade; optionally add time-based penalty/bonus.
            pass

        elif action == 3:  # Close All
            if self.position != 0 and self.entry_price is not None:
                if self.position > 0:
                    realized = (current_price - self.entry_price) / self.tick_size * self.tick_value * self.position
                else:
                    realized = (self.entry_price - current_price) / self.tick_size * self.tick_value * abs(self.position)
                self.balance += realized
                reward += realized
                self.trade_history.append({
                    "action": "CloseAll",
                    "price": current_price,
                    "time": current_time,
                    "realized": realized,
                    "step_index": self.current_step
                })
                self.position = 0
                self.entry_price = None

        # Update time_in_position.
        if self.position != 0:
            self.time_in_position += 1
        else:
            self.time_in_position = 0

        self.current_step += 1
        if self.current_step >= len(self.day_data):
            terminated = True

        # Check if profit target is reached.
        if self._get_equity() >= self.target_balance:
            terminated = True
            reward += 100.0  # Bonus reward for achieving profit target.

        observation = self._next_observation()

        # Render if in human mode.
        if self.render_mode == "human":
            self.render()

        info = {
            "balance": self.balance,
            "position": self.position,
            "equity": self._get_equity(),
            "time_in_position": self.time_in_position
        }
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the current state of the environment.
        In human mode, display a candlestick chart with the x-axis representing the actual time.
        Trade markers are annotated with the time and price of the trade.
        """
        if self.render_mode != "human":
            return None

        # Clear axes.
        self.ax.cla()

        # Prepare data for plotting.
        data = self.day_data.copy()
        # Convert the "Date" column to Matplotlib date numbers.
        data["DateNum"] = data["Date"].apply(mdates.date2num)

        # Plot candlestick chart.
        for idx, row in data.iterrows():
            # Choose color based on price movement.
            color = "green" if row["Close"] >= row["Open"] else "red"
            # Plot the high-low line.
            self.ax.plot([row["DateNum"], row["DateNum"]], [row["Low"], row["High"]], color="black", linewidth=1)
            # Draw the candle body as a rectangle.
            open_price, close_price = row["Open"], row["Close"]
            lower = min(open_price, close_price)
            height = abs(open_price - close_price)
            # Use width in days (e.g., 0.0007 days ~ 1 minute width relative to date numbers).
            width = 0.0007
            self.ax.add_patch(plt.Rectangle((row["DateNum"] - width/2, lower), width, height, color=color))

        # Annotate trades with markers at the correct time.
        for trade in self.trade_history:
            trade_time = trade.get("time", None)
            if trade_time is not None:
                trade_time_num = mdates.date2num(trade_time)
                # Get corresponding price from data (if available) or use trade price.
                trade_price = trade.get("price", None)
                if trade_price is not None:
                    self.ax.plot(trade_time_num, trade_price, marker="o", color="blue", markersize=8)
                    # Optionally annotate with the action label.
                    action_label = trade.get("action", "")
                    self.ax.text(trade_time_num, trade_price, action_label, color="black", fontsize=8)

        # Format x-axis to show time (hour:minute).
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.fig.autofmt_xdate()

        # Set labels and title.
        day_str = self.day_data.iloc[0]["Date"].strftime("%m/%d/%Y")
        self.ax.set_title(f"Candlestick Chart for {day_str} | Step: {self.current_step}")
        self.ax.set_xlabel("Time (HH:MM)")
        self.ax.set_ylabel("Price")

        self.fig.canvas.draw()
        plt.pause(0.001)

    def close(self):
        """
        Close the environment and rendering windows if in human mode.
        """
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.fig)
