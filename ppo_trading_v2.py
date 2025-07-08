#!/usr/bin/env python3
"""
MULTI-ASSET PPO TRADING SYSTEM - SAME DATES AS PPO_TRADING.PY
- Multi-asset trading: BTC, S&P 500, US Bonds
- Same train/val/test splits as ppo_trading.py (2010-2016/2017-2018/2019-2024)
- Robust feature handling for fair comparison
- UPDATED: Uses improved Sharpe-like reward and metrics (same as v3)
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Device configuration for Mac GPU support
def get_device():
    """Get the best available device (MPS for Mac, CUDA for NVIDIA, CPU fallback)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🚀 Using Mac GPU (MPS): {device}")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using NVIDIA GPU (CUDA): {device}")
        return device
    else:
        device = torch.device("cpu")
        print(f"💻 Using CPU: {device}")
        return device

# Assets
ASSETS = {
    'BTC': 'BTC-USD',      # Bitcoin
    'SP500': '^GSPC',      # S&P 500
    'BONDS': 'TLT'         # US Treasury Bonds
}

ASSET_NAMES = list(ASSETS.keys())

def create_robust_features(data, asset_name):
    """Create robust features that won't generate NaN"""
    df = data.copy()
    
    # Basic returns
    df[f'returns_{asset_name}'] = df['Close'].pct_change().fillna(0)
    df[f'log_returns_{asset_name}'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    
    # Momentum indicators (multiple timeframes)
    df[f'momentum_5d_{asset_name}'] = df['Close'].pct_change(periods=5).fillna(0)
    df[f'momentum_21d_{asset_name}'] = df['Close'].pct_change(periods=21).fillna(0)
    
    # Volatility indicators
    returns = df[f'returns_{asset_name}']
    df[f'volatility_5d_{asset_name}'] = returns.rolling(5).std().fillna(0)
    df[f'volatility_21d_{asset_name}'] = returns.rolling(21).std().fillna(0)
    
    # RSI (simple calculation)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)  # Avoid division by zero
    df[f'rsi_{asset_name}'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # Moving averages
    ma_20 = df['Close'].rolling(20).mean()
    ma_50 = df['Close'].rolling(50).mean()
    df[f'ma_ratio_{asset_name}'] = (ma_20 / ma_50).fillna(1)
    
    # Volume ratio
    volume_ma = df['Volume'].rolling(20).mean()
    df[f'volume_ratio_{asset_name}'] = (df['Volume'] / (volume_ma + 1)).fillna(1)
    
    # Trend strength (simple linear regression slope)
    def calculate_trend(prices):
        if len(prices) < 10:
            return 0
        x = np.arange(len(prices))
        y = prices.values
        if np.std(y) == 0:
            return 0
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope if not np.isnan(slope) else 0
    
    df[f'trend_strength_{asset_name}'] = df['Close'].rolling(21).apply(calculate_trend).fillna(0)
    
    # Feature list
    features = [
        f'returns_{asset_name}', f'log_returns_{asset_name}',
        f'momentum_5d_{asset_name}', f'momentum_21d_{asset_name}',
        f'volatility_5d_{asset_name}', f'volatility_21d_{asset_name}',
        f'rsi_{asset_name}', f'ma_ratio_{asset_name}',
        f'volume_ratio_{asset_name}', f'trend_strength_{asset_name}'
    ]
    
    # Robust cleaning - ensure no NaN or extreme values
    for col in features:
        if col in df.columns:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill any remaining NaN with 0
            df[col] = df[col].fillna(0)
            # Replace infinite values
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            # Clip extreme values
            df[col] = np.clip(df[col], -10, 10)
            # Final check - ensure all values are finite
            df[col] = np.where(np.isfinite(df[col]), df[col], 0)
    
    return df, features

def create_cross_asset_features(data_dict):
    """Create cross-asset correlation features"""
    # Get common dates
    common_dates = None
    for asset_data in data_dict.values():
        if common_dates is None:
            common_dates = asset_data.index
        else:
            common_dates = common_dates.intersection(asset_data.index)
    
    cross_features = pd.DataFrame(index=common_dates)
    
    # Get returns for correlation
    returns_data = {}
    for asset_name, asset_data in data_dict.items():
        returns_data[asset_name] = asset_data[f'returns_{asset_name}'].reindex(common_dates)
    
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate rolling correlations
    for i, asset1 in enumerate(ASSET_NAMES):
        for j, asset2 in enumerate(ASSET_NAMES):
            if i < j:
                corr = returns_df[asset1].rolling(21).corr(returns_df[asset2])
                cross_features[f'corr_{asset1}_{asset2}'] = corr.fillna(0)
    
    # Market volatility regime
    market_vol = returns_df.rolling(21).std().mean(axis=1)
    cross_features['market_volatility'] = market_vol.fillna(0)
    
    # Risk-on/Risk-off indicator
    if 'BTC' in returns_df.columns and 'SP500' in returns_df.columns:
        risk_on_corr = returns_df['BTC'].rolling(21).corr(returns_df['SP500'])
        cross_features['risk_on_regime'] = risk_on_corr.fillna(0)
    
    # Ensure all cross features are finite
    for col in cross_features.columns:
        cross_features[col] = cross_features[col].fillna(0)
        cross_features[col] = cross_features[col].replace([np.inf, -np.inf], 0)
        cross_features[col] = np.clip(cross_features[col], -5, 5)
    
    return cross_features

class SimpleMultiAssetEnv(gym.Env):
    """Simple but sophisticated multi-asset environment"""
    
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
        
        # Parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.num_assets = len(ASSET_NAMES)
        
        # Observation space: balance + shares + all features
        total_features = 1 + self.num_assets
        for features in self.feature_columns.values():
            total_features += len(features)
        total_features += len(self.cross_features.columns)
        
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(total_features,), dtype=np.float32
        )
        
        # Action space: {-1, -0.5, 0, 0.5, 1} for each asset
        self.action_space = spaces.MultiDiscrete([5] * self.num_assets)
        self.action_mapping = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
        
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
        """Get current observation with robust handling"""
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
                # Ensure finite value
                if not np.isfinite(value):
                    value = 0.0
                obs.append(value)
        
        # Cross-asset features
        for col in self.cross_features.columns:
            value = float(self.cross_features.iloc[self.current_step][col])
            # Ensure finite value
            if not np.isfinite(value):
                value = 0.0
            obs.append(value)
        
        # Final check for any remaining NaN or inf
        obs = np.array(obs, dtype=np.float32)
        obs = np.where(np.isfinite(obs), obs, 0.0)
        
        return obs
    
    def get_all_observations(self):
        """Get all observations for LSTM training"""
        all_obs = []
        
        for step in range(self.data_length):
            self.current_step = step
            obs = self._get_observation()
            all_obs.append(obs)
        
        return np.array(all_obs)
    
    def step(self, action):
        if self.current_step >= self.data_length - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Execute trades
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
            
            if trade_signal > 0:  # Buy
                trade_amount = current_portfolio_value * abs(trade_signal) * 0.1
                max_shares = int(trade_amount / (price * (1 + self.transaction_cost)))
                
                if max_shares > 0 and trade_amount <= self.balance:
                    cost = max_shares * price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares[asset] += max_shares
                    
            elif trade_signal < 0:  # Sell
                sell_percentage = abs(trade_signal) * 0.2
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
        
        # Enhanced reward function (Sharpe-like) - SAME AS V3
        if len(self.returns_history) > 21:  # Need some history
            recent_returns = self.returns_history[-21:]
            avg_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 0:
                reward = avg_return / std_return  # Sharpe-like reward
            else:
                reward = avg_return
        else:
            reward = daily_return
        
        # Diversification bonus
        diversification_bonus = 0
        total_asset_value = sum(self.shares[asset] * float(self.data_dict[asset].iloc[self.current_step]['Close']) 
                              for asset in ASSET_NAMES)
        
        if total_asset_value > 0:
            allocations = []
            for asset in ASSET_NAMES:
                price = float(self.data_dict[asset].iloc[self.current_step]['Close'])
                asset_value = self.shares[asset] * price
                allocation = asset_value / total_asset_value
                allocations.append(allocation)
            
            # Reward balanced allocation
            allocations = np.array(allocations)
            ideal_allocation = np.array([0.35, 0.40, 0.25])  # BTC, SP500, BONDS
            allocation_distance = np.sum(np.abs(allocations - ideal_allocation))
            diversification_bonus = max(0, 2 - allocation_distance * 5)
        
        reward = reward * 100 + diversification_bonus
        
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

def load_simple_data():
    """Load data with robust handling - SAME DATES AS PPO_TRADING.PY"""
    print("📊 Loading data (2010-2024) - Same dates as ppo_trading.py...")
    
    data_dict = {}
    
    for asset_name, ticker in ASSETS.items():
        print(f"  Loading {asset_name}...")
        try:
            data = yf.download(ticker, start='2010-01-01', end='2024-12-01', progress=False)
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            
            data = data.dropna()
            data_dict[asset_name] = data
            print(f"    ✅ {len(data)} days")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return None
    
    # Split data - SAME AS PPO_TRADING.PY (ICML Paper style)
    train_end = '2016-12-31'
    val_end = '2018-12-31'
    
    train_data = {}
    val_data = {}
    test_data = {}
    
    for asset_name, data in data_dict.items():
        train_data[asset_name] = data[data.index <= train_end].copy()
        val_data[asset_name] = data[(data.index > train_end) & (data.index <= val_end)].copy()
        test_data[asset_name] = data[data.index > val_end].copy()
    
    print(f"📈 Training: {len(train_data[list(ASSETS.keys())[0]])} days (2010-01-01 to 2016-12-31)")
    print(f"📊 Validation: {len(val_data[list(ASSETS.keys())[0]])} days (2017-01-01 to 2018-12-31)")
    print(f"📉 Testing: {len(test_data[list(ASSETS.keys())[0]])} days (2019-01-01 to 2024-12-01)")
    
    return train_data, val_data, test_data

def test_simple_model(model, env):
    """Test the model"""
    obs, _ = env.reset()
    portfolio_values = [env.initial_balance]
    actions_taken = {asset: [] for asset in ASSET_NAMES}
    portfolio_allocations = {asset: [] for asset in ASSET_NAMES}
    cash_allocations = []
    
    for _ in range(env.data_length - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        portfolio_values.append(info['net_worth'])
        
        # Calculate allocations
        total_value = info['net_worth']
        cash_percentage = (info['balance'] / total_value) * 100
        cash_allocations.append(cash_percentage)
        
        for asset in ASSET_NAMES:
            actions_taken[asset].append(info['actions_taken'][asset])
            asset_value = info['shares'][asset] * info['prices'][asset]
            asset_percentage = (asset_value / total_value) * 100
            portfolio_allocations[asset].append(asset_percentage)
        
        if done:
            break
    
    # Calculate metrics
    returns = np.array(portfolio_values)
    daily_returns = np.diff(returns) / returns[:-1]
    
    total_return = (returns[-1] - returns[0]) / returns[0] * 100
    annual_return = (((returns[-1] / returns[0]) ** (252 / len(returns))) - 1) * 100
    
    volatility = np.std(daily_returns) * np.sqrt(252) * 100
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    max_drawdown = np.min((returns - np.maximum.accumulate(returns)) / np.maximum.accumulate(returns)) * 100
    
    # Calculate final and average allocations
    final_allocations = {}
    avg_allocations = {}
    
    for asset in ASSET_NAMES:
        final_allocations[asset] = portfolio_allocations[asset][-1] if portfolio_allocations[asset] else 0
        avg_allocations[asset] = np.mean(portfolio_allocations[asset]) if portfolio_allocations[asset] else 0
    
    final_cash = cash_allocations[-1] if cash_allocations else 0
    avg_cash = np.mean(cash_allocations) if cash_allocations else 0
    
    return {
        'return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'actions': actions_taken,
        'final_allocations': final_allocations,
        'avg_allocations': avg_allocations,
        'final_cash': final_cash,
        'avg_cash': avg_cash,
        'portfolio_allocations': portfolio_allocations
    }

def main():
    print("🚀 MULTI-ASSET PPO WITH SAME DATES AS PPO_TRADING.PY")
    print("=" * 70)
    print("✅ Multi-asset trading: BTC, S&P 500, US Bonds")
    print("✅ Advanced technical indicators")
    print("✅ Cross-asset correlation analysis")
    print("✅ Robust feature handling (no NaN)")
    print("✅ Improved Sharpe-like reward function")
    print("✅ Enhanced metrics (same as v3 for fair comparison)")
    print("✅ SAME DATES AS PPO_TRADING.PY (2010-2024)")
    print("=" * 70)
    
    # Load data
    result = load_simple_data()
    if result is None:
        print("❌ Failed to load data")
        return
    
    train_data, val_data, test_data = result
    
    # Get best available device
    device = get_device()
    
    # Create environment
    env = SimpleMultiAssetEnv(train_data)
    env = DummyVecEnv([lambda: env])
    
    # Create model
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
        ent_coef=0.01,
        verbose=1,
        device=device
    )
    
    # Train
    print("\n🔥 Training (100,000 steps)...")
    model.learn(total_timesteps=100_000, progress_bar=True)
    
    # Save
    model.save("trained_models/multi_asset_ppo_same_dates")
    print("✅ Model saved")
    
    # Validate
    print("\n📊 Validating on validation set (2017-2018)...")
    val_env = SimpleMultiAssetEnv(val_data)
    val_results = test_simple_model(model, val_env)
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"📈 Total Return: {val_results['return']:.2f}%")
    print(f"📊 Annual Return: {val_results['annual_return']:.2f}%")
    print(f"⚡ Sharpe Ratio: {val_results['sharpe']:.3f}")
    print(f"📉 Max Drawdown: {val_results['max_drawdown']:.2f}%")
    print(f"🌊 Volatility: {val_results['volatility']:.2f}%")
    
    # Test on unseen data
    print("\n🧪 Testing on unseen test set (2019-2024)...")
    test_env = SimpleMultiAssetEnv(test_data)
    results = test_simple_model(model, test_env)
    
    print(f"\n🏆 FINAL TEST RESULTS (2019-2024):")
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
    
    # Action analysis
    print(f"\n🎬 Action Distribution:")
    for asset in ASSET_NAMES:
        action_counts = {}
        for action in results['actions'][asset]:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"\n   {asset}:")
        for action, count in sorted(action_counts.items()):
            pct = count / len(results['actions'][asset]) * 100
            if action < 0:
                action_name = f"Strong Sell" if action == -1 else f"Moderate Sell"
            elif action > 0:
                action_name = f"Strong Buy" if action == 1 else f"Moderate Buy"
            else:
                action_name = "Hold"
            print(f"     {action_name}: {pct:.1f}%")
    
    print(f"\n✅ Multi-asset PPO training complete!")
    print(f"🎯 Now uses SAME DATES as ppo_trading.py for fair comparison!")
    print(f"📊 UPDATED: Improved Sharpe-like reward and metrics (same as v3)")
    print(f"📊 Train: 2010-2016 | Val: 2017-2018 | Test: 2019-2024")
    print(f"💾 Model saved as 'trained_models/multi_asset_ppo_same_dates'")

if __name__ == "__main__":
    main() 