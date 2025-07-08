#!/usr/bin/env python3
"""
BITCOIN-FOCUSED PPO TRADING SYSTEM
- Allows up to 80% Bitcoin allocation
- Larger position sizes
- Optimized for Bitcoin bull markets
- Can compete with Bitcoin buy & hold
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import warnings
warnings.filterwarnings('ignore')

# Import from existing system
from ppo_trading_v2 import (
    get_device, ASSETS, ASSET_NAMES, create_robust_features, 
    create_cross_asset_features, load_simple_data, test_simple_model
)

class BitcoinFocusedEnv(gym.Env):
    """Bitcoin-focused multi-asset environment"""
    
    def __init__(self, data_dict, initial_balance=1000000, transaction_cost=0.002):
        super().__init__()
        
        self.data_dict = {}
        self.feature_columns = {}
        
        # Process data
        for asset_name, asset_data in data_dict.items():
            processed_data, features = create_robust_features(asset_data, asset_name)
            self.data_dict[asset_name] = processed_data.dropna()
            self.feature_columns[asset_name] = features
        
        # Cross-asset features
        self.cross_features = create_cross_asset_features(self.data_dict)
        
        # Align data
        self.align_data()
        
        # Parameters - MORE AGGRESSIVE
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # LOWER transaction costs
        self.num_assets = len(ASSET_NAMES)
        
        # Observation space
        total_features = 1 + self.num_assets
        for features in self.feature_columns.values():
            total_features += len(features)
        total_features += len(self.cross_features.columns)
        
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(total_features,), dtype=np.float32
        )
        
        # Action space: MORE AGGRESSIVE ACTIONS
        self.action_space = spaces.MultiDiscrete([7] * self.num_assets)  # 7 actions instead of 5
        self.action_mapping = {
            0: -1.0,    # Strong Sell
            1: -0.5,    # Moderate Sell
            2: -0.25,   # Light Sell
            3: 0.0,     # Hold
            4: 0.25,    # Light Buy
            5: 0.5,     # Moderate Buy
            6: 1.0      # Strong Buy
        }
        
        self.reset()
    
    def align_data(self):
        """Align all data to common timeframe"""
        # Get common date range
        common_start = max(data.index[0] for data in self.data_dict.values())
        common_end = min(data.index[-1] for data in self.data_dict.values())
        
        # Ensure cross_features alignment
        common_start = max(common_start, self.cross_features.index[0])
        common_end = min(common_end, self.cross_features.index[-1])
        
        # Align all data
        for asset_name in self.data_dict:
            self.data_dict[asset_name] = self.data_dict[asset_name][
                (self.data_dict[asset_name].index >= common_start) & 
                (self.data_dict[asset_name].index <= common_end)
            ]
        
        self.cross_features = self.cross_features[
            (self.cross_features.index >= common_start) &
            (self.cross_features.index <= common_end)
        ]
        
        # Ensure same length
        min_length = min(len(data) for data in self.data_dict.values())
        min_length = min(min_length, len(self.cross_features))
        
        for asset_name in self.data_dict:
            self.data_dict[asset_name] = self.data_dict[asset_name].iloc[:min_length]
        
        self.cross_features = self.cross_features.iloc[:min_length]
        self.data_length = min_length
        
        print(f"📊 Aligned data: {self.data_length} days")
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares = {asset: 0.0 for asset in ASSET_NAMES}
        self.net_worth_history = [self.initial_balance]
        self.returns_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= self.data_length:
            self.current_step = self.data_length - 1
        
        # Balance (normalized)
        obs = [self.balance / self.initial_balance]
        
        # Shares value (normalized)
        for asset in ASSET_NAMES:
            price = float(self.data_dict[asset].iloc[self.current_step]['Close'])
            shares_value = self.shares[asset] * price
            obs.append(shares_value / self.initial_balance)
        
        # Asset features
        for asset in ASSET_NAMES:
            for col in self.feature_columns[asset]:
                value = float(self.data_dict[asset].iloc[self.current_step][col])
                if not np.isfinite(value):
                    value = 0.0
                obs.append(value)
        
        # Cross-asset features
        for col in self.cross_features.columns:
            value = float(self.cross_features.iloc[self.current_step][col])
            if not np.isfinite(value):
                value = 0.0
            obs.append(value)
        
        obs = np.array(obs, dtype=np.float32)
        obs = np.where(np.isfinite(obs), obs, 0.0)
        
        return obs
    
    def step(self, action):
        if self.current_step >= self.data_length - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Execute trades - MORE AGGRESSIVE POSITION SIZING
        actions_taken = {}
        current_portfolio_value = self.balance
        for asset in ASSET_NAMES:
            price = float(self.data_dict[asset].iloc[self.current_step]['Close'])
            current_portfolio_value += self.shares[asset] * price
        
        for i, asset in enumerate(ASSET_NAMES):
            trade_signal = self.action_mapping[action[i]]
            actions_taken[asset] = trade_signal
            
            if trade_signal == 0:
                continue
            
            price = float(self.data_dict[asset].iloc[self.current_step]['Close'])
            
            if trade_signal > 0:  # Buy - MUCH MORE AGGRESSIVE
                # Allow up to 25% of portfolio per trade (vs 10% original)
                trade_amount = current_portfolio_value * abs(trade_signal) * 0.25
                max_shares = int(trade_amount / (price * (1 + self.transaction_cost)))
                
                if max_shares > 0 and trade_amount <= self.balance:
                    cost = max_shares * price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares[asset] += max_shares
                    
            elif trade_signal < 0:  # Sell - MORE AGGRESSIVE
                # Allow up to 50% of shares per trade (vs 20% original)
                sell_percentage = abs(trade_signal) * 0.5
                shares_to_sell = int(self.shares[asset] * sell_percentage)
                
                if shares_to_sell > 0:
                    revenue = shares_to_sell * price * (1 - self.transaction_cost)
                    self.balance += revenue
                    self.shares[asset] -= shares_to_sell
        
        # Update step
        self.current_step += 1
        
        # Calculate net worth
        net_worth = self.balance
        for asset in ASSET_NAMES:
            new_price = float(self.data_dict[asset].iloc[self.current_step]['Close'])
            net_worth += self.shares[asset] * new_price
        
        self.net_worth_history.append(net_worth)
        
        # Calculate return and reward
        daily_return = (net_worth - self.net_worth_history[-2]) / self.net_worth_history[-2]
        self.returns_history.append(daily_return)
        
        # BITCOIN-FOCUSED REWARD FUNCTION
        if len(self.returns_history) > 21:
            recent_returns = self.returns_history[-21:]
            avg_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 0:
                reward = avg_return / std_return
            else:
                reward = avg_return
        else:
            reward = daily_return
        
        # BITCOIN-FOCUSED ALLOCATION BONUS
        bitcoin_bonus = 0
        total_asset_value = sum(self.shares[asset] * float(self.data_dict[asset].iloc[self.current_step]['Close']) 
                              for asset in ASSET_NAMES)
        
        if total_asset_value > 0:
            # Calculate Bitcoin allocation
            btc_price = float(self.data_dict['BTC'].iloc[self.current_step]['Close'])
            btc_value = self.shares['BTC'] * btc_price
            btc_allocation = btc_value / total_asset_value
            
            # REWARD HIGH BITCOIN ALLOCATION (opposite of original)
            if btc_allocation > 0.6:  # Reward 60%+ Bitcoin allocation
                bitcoin_bonus = (btc_allocation - 0.6) * 10  # Up to 4 bonus points
            elif btc_allocation > 0.4:  # Moderate reward for 40%+ Bitcoin
                bitcoin_bonus = (btc_allocation - 0.4) * 5   # Up to 1 bonus point
        
        reward = reward * 100 + bitcoin_bonus
        
        done = self.current_step >= self.data_length - 1
        
        # Current prices for info
        current_prices = {}
        for asset in ASSET_NAMES:
            current_prices[asset] = float(self.data_dict[asset].iloc[self.current_step]['Close'])
        
        info = {
            'net_worth': net_worth,
            'balance': self.balance,
            'shares': self.shares.copy(),
            'prices': current_prices,
            'actions_taken': actions_taken
        }
        
        return self._get_observation(), reward, done, False, info

def main():
    print("🚀 BITCOIN-FOCUSED PPO TRADING SYSTEM")
    print("=" * 70)
    print("✅ Allows up to 80% Bitcoin allocation")
    print("✅ Larger position sizes (25% buy, 50% sell)")
    print("✅ Lower transaction costs (0.1% vs 0.2%)")
    print("✅ Bitcoin-focused reward function")
    print("✅ Designed to compete with Bitcoin buy & hold")
    print("=" * 70)
    
    # Load data
    result = load_simple_data()
    if result is None:
        print("❌ Failed to load data")
        return
    
    train_data, val_data, test_data = result
    
    # Get device
    device = get_device()
    
    # Create Bitcoin-focused environment
    env = BitcoinFocusedEnv(train_data)
    env = DummyVecEnv([lambda: env])
    
    # Create model with more aggressive parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher entropy for more exploration
        verbose=1,
        device=device
    )
    
    # Train
    print("\n🔥 Training Bitcoin-focused model (500,000 steps)...")
    model.learn(total_timesteps=500_000, progress_bar=True)
    
    # Save
    model.save("trained_models/bitcoin_focused_ppo_500k")
    print("✅ Model saved")
    
    # Test on unseen data
    print("\n🧪 Testing on test set (2019-2024)...")
    test_env = BitcoinFocusedEnv(test_data)
    results = test_simple_model(model, test_env)
    
    print(f"\n🏆 BITCOIN-FOCUSED RESULTS (2019-2024):")
    print(f"📈 Total Return: {results['return']:.2f}%")
    print(f"📊 Annual Return: {results['annual_return']:.2f}%")
    print(f"⚡ Sharpe Ratio: {results['sharpe']:.3f}")
    print(f"📉 Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"🌊 Volatility: {results['volatility']:.2f}%")
    
    # Portfolio allocation analysis
    print(f"\n💼 PORTFOLIO ALLOCATION ANALYSIS:")
    print(f"\n📊 Final Portfolio Allocation:")
    for asset in ASSET_NAMES:
        print(f"   {asset}: {results['final_allocations'][asset]:.1f}%")
    print(f"   CASH: {results['final_cash']:.1f}%")
    
    print(f"\n📊 Average Portfolio Allocation:")
    for asset in ASSET_NAMES:
        print(f"   {asset}: {results['avg_allocations'][asset]:.1f}%")
    print(f"   CASH: {results['avg_cash']:.1f}%")
    
    print(f"\n✅ Bitcoin-focused PPO training complete!")
    print(f"🎯 Designed to compete with Bitcoin buy & hold!")
    print(f"💾 Model saved as 'trained_models/bitcoin_focused_ppo'")

if __name__ == "__main__":
    main() 