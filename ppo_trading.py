#!/usr/bin/env python3
"""
ENHANCED PPO BASED ON ICML 2025 PAPER
- Proper train/validation/test splits (2010-2016/2017-2018/2019-2024)
- Segmented actions {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
- Fama-French features
- Transaction costs 0.2%
- Sharpe ratio reward
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

def get_fama_french_features(data):
    """Calculate Fama-French factors and additional features"""
    df = data.copy()
    
    # Basic returns
    df['returns'] = df['Close'].pct_change().fillna(0)
    
    # Momentum (12-month return, use 60 days for shorter sequences)
    df['momentum'] = df['Close'].pct_change(periods=60).fillna(0)
    
    # Size factor (market cap proxy using volume * price)
    market_value = df['Volume'] * df['Close']
    market_value_log = np.log(market_value.replace(0, 1))  # Avoid log(0)
    market_mean = market_value_log.rolling(60).mean()
    market_std = market_value_log.rolling(60).std()
    df['size_factor'] = ((market_value_log - market_mean) / market_std).fillna(0)
    
    # Value factor (P/E proxy using price momentum)
    df['value_factor'] = (-df['Close'].pct_change(periods=21)).fillna(0)  # 3-week reverse
    
    # Profitability factor (price efficiency)
    profit_raw = (df['High'] - df['Close']) / df['Close']
    df['profitability'] = profit_raw.rolling(21).mean().fillna(0)
    
    # Volatility
    df['volatility'] = (df['returns'].rolling(21).std() * np.sqrt(252)).fillna(0)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)  # Avoid division by zero
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # Moving averages
    ma_20 = df['Close'].rolling(20).mean()
    ma_50 = df['Close'].rolling(50).mean()
    df['ma_ratio'] = (ma_20 / ma_50).fillna(1)
    
    # Volume indicators
    volume_ma = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = (df['Volume'] / (volume_ma + 1)).fillna(1)  # Avoid division by zero
    
    # Final cleanup - ensure all features are float
    features = ['returns', 'momentum', 'size_factor', 'value_factor', 
                'profitability', 'volatility', 'rsi', 'ma_ratio', 'volume_ratio']
    
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Clip extreme values
        df[col] = np.clip(df[col], -10, 10)
    
    return df

class PaperTradingEnv(gym.Env):
    """Enhanced Trading Environment based on ICML Paper"""
    
    def __init__(self, data, initial_balance=1000000, transaction_cost=0.002, max_shares=1000):
        super(PaperTradingEnv, self).__init__()
        
        # Data and features
        self.data = get_fama_french_features(data).dropna()
        self.feature_columns = [
            'returns', 'momentum', 'size_factor', 'value_factor', 
            'profitability', 'volatility', 'rsi', 'ma_ratio', 'volume_ratio'
        ]
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # 0.2% as in paper
        self.max_shares = max_shares  # K = 1000 as in paper
        
        # State space: balance + shares + features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(2 + len(self.feature_columns),), 
            dtype=np.float32
        )
        
        # Action space: Segmented actions {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
        self.action_space = spaces.Discrete(11)
        self.action_mapping = {
            0: -5, 1: -4, 2: -3, 3: -2, 4: -1, 
            5: 0,   # Hold
            6: 1, 7: 2, 8: 3, 9: 4, 10: 5
        }
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares = 0.0
        self.net_worth_history = [self.initial_balance]
        self.returns_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        # Normalize balance and shares
        balance_norm = self.balance / self.initial_balance
        price = float(self.data.iloc[self.current_step]['Close'])
        shares_norm = (self.shares * price) / self.initial_balance
        
        # Get features
        features = []
        for col in self.feature_columns:
            features.append(float(self.data.iloc[self.current_step][col]))
        
        obs = np.array([balance_norm, shares_norm] + features, dtype=np.float32)
        return obs
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Map action to trade size
        trade_multiplier = self.action_mapping[action]
        
        # Get current price
        price = float(self.data.iloc[self.current_step]['Close'])
        
        # Calculate trade amount
        if trade_multiplier > 0:  # Buy
            max_affordable = int(self.balance / (price * (1 + self.transaction_cost)))
            shares_to_buy = min(max_affordable, trade_multiplier * 200)  # Scale up trades
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares += shares_to_buy
                    
        elif trade_multiplier < 0:  # Sell
            shares_to_sell = min(self.shares, abs(trade_multiplier) * 200)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares -= shares_to_sell
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new net worth
        new_price = float(self.data.iloc[self.current_step]['Close'])
        net_worth = self.balance + self.shares * new_price
        self.net_worth_history.append(net_worth)
        
        # Calculate return
        daily_return = (net_worth - self.net_worth_history[-2]) / self.net_worth_history[-2]
        self.returns_history.append(daily_return)
        
        # Enhanced reward function (Sharpe-like)
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
        
        # Scale reward
        reward *= 100
        
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'net_worth': net_worth,
            'balance': self.balance,
            'shares': self.shares,
            'price': new_price,
            'action_taken': trade_multiplier
        }
        
        return self._get_observation(), reward, done, False, info

def load_sp500_data():
    """Load S&P 500 data with proper splits"""
    print("📊 Loading S&P 500 data (2010-2024)...")
    
    # Load extended period
    data = yf.download('^GSPC', start='2010-01-01', end='2024-12-01', progress=False)
    data = data.dropna()
    
    # Split according to paper methodology
    train_end = '2016-12-31'
    val_end = '2018-12-31'
    
    train_data = data[data.index <= train_end].copy()
    val_data = data[(data.index > train_end) & (data.index <= val_end)].copy()
    test_data = data[data.index > val_end].copy()
    
    print(f"📈 Training: {len(train_data)} days ({train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"📊 Validation: {len(val_data)} days ({val_data.index[0].strftime('%Y-%m-%d')} to {val_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"📉 Testing: {len(test_data)} days ({test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')})")
    
    return train_data, val_data, test_data

def train_enhanced_ppo():
    """Train enhanced PPO model"""
    print("🚀 Training Enhanced PPO (Based on ICML Paper)...")
    
    # Load data
    train_data, val_data, test_data = load_sp500_data()
    
    # Create environment
    env = PaperTradingEnv(train_data)
    env = DummyVecEnv([lambda: env])
    
    # Enhanced PPO parameters (closer to paper)
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
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_enhanced_ppo/"
    )
    
    # Train model
    print("🔥 Training PPO with segmented actions and Fama-French features...")
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    model_path = "trained_models/enhanced_ppo_paper"
    model.save(model_path)
    print(f"✅ Model saved as {model_path}")
    
    # Validate model
    print("📊 Validating model...")
    val_env = PaperTradingEnv(val_data)
    val_results = test_model(model, val_env)
    
    print(f"🎯 Validation Results:")
    print(f"   Return: {val_results['return']:.2f}%")
    print(f"   Sharpe: {val_results['sharpe']:.3f}")
    print(f"   Max Drawdown: {val_results['max_drawdown']:.2f}%")
    
    return model, train_data, val_data, test_data

def test_model(model, env):
    """Test model performance"""
    obs, _ = env.reset()
    portfolio_values = [env.initial_balance]
    actions_taken = []
    
    for _ in range(len(env.data) - 1):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)  # Convert numpy array to int
        obs, reward, done, truncated, info = env.step(action)
        
        portfolio_values.append(info['net_worth'])
        actions_taken.append(env.action_mapping[action])
        
        if done:
            break
    
    # Calculate metrics
    returns = np.array(portfolio_values)
    daily_returns = np.diff(returns) / returns[:-1]
    
    total_return = (returns[-1] - returns[0]) / returns[0] * 100
    annual_return = ((returns[-1] / returns[0]) ** (252 / len(returns))) - 1
    annual_return *= 100
    
    volatility = np.std(daily_returns) * np.sqrt(252) * 100
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    peak = np.maximum.accumulate(returns)
    drawdown = (returns - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    
    # Activity
    hold_actions = sum(1 for a in actions_taken if a == 0)
    activity = (1 - hold_actions / len(actions_taken)) * 100
    
    return {
        'return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'activity': activity,
        'portfolio_values': portfolio_values,
        'actions': actions_taken
    }

def main():
    print("🏆 ENHANCED PPO TRAINING (ICML 2025 Paper Implementation)")
    print("=" * 80)
    print("📋 Improvements:")
    print("✅ Extended period: 2010-2024 (14 years)")
    print("✅ Proper splits: Train(2010-2016) / Val(2017-2018) / Test(2019-2024)")
    print("✅ Segmented actions: {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}")
    print("✅ Fama-French features: momentum, size, value, profitability")
    print("✅ Transaction costs: 0.2%")
    print("✅ Sharpe-like reward function")
    print("=" * 80)
    
    # Train model
    model, train_data, val_data, test_data = train_enhanced_ppo()
    
    # Test on test set
    print("\n🧪 Testing on unseen data (2019-2024)...")
    test_env = PaperTradingEnv(test_data)
    test_results = test_model(model, test_env)
    
    print(f"\n🏆 FINAL TEST RESULTS:")
    print(f"📈 Total Return: {test_results['return']:.2f}%")
    print(f"📊 Annual Return: {test_results['annual_return']:.2f}%")
    print(f"⚡ Sharpe Ratio: {test_results['sharpe']:.3f}")
    print(f"📉 Max Drawdown: {test_results['max_drawdown']:.2f}%")
    print(f"🎯 Activity: {test_results['activity']:.1f}%")
    
    # Compare with benchmark
    sp500_return = ((test_data['Close'].iloc[-1] / test_data['Close'].iloc[0]) - 1) * 100
    print(f"\n📊 S&P 500 Benchmark: {sp500_return:.2f}%")
    print(f"🚀 Outperformance: {test_results['return'] - sp500_return:.2f}%")
    
    # Action analysis
    action_counts = {}
    for action in test_results['actions']:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"\n🎬 Action Distribution:")
    for action, count in sorted(action_counts.items()):
        pct = count / len(test_results['actions']) * 100
        action_name = f"Sell {abs(action)}" if action < 0 else f"Buy {action}" if action > 0 else "Hold"
        print(f"   {action_name}: {pct:.1f}% ({count} times)")
    
    print(f"\n✅ Enhanced PPO training complete!")
    print(f"💾 Model saved as 'trained_models/enhanced_ppo_paper'")

if __name__ == "__main__":
    main() 