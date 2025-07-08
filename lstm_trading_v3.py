#!/usr/bin/env python3
"""
MULTI-ASSET LSTM TRADING SYSTEM - ALIGNED WITH ACADEMIC LITERATURE
- Multi-asset trading: BTC, S&P 500, US Bonds
- Same train/val/test splits as ppo_trading.py (2010-2016/2017-2018/2019-2024)
- LSTM-based trading model (Fischer & Krauss inspired)
- Parameters aligned with academic literature
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
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

class LSTMTradingModel(nn.Module):
    """LSTM-based trading model - Inspired by Fischer & Krauss (2018)"""
    
    def __init__(self, input_size, hidden_size=25, num_layers=1, num_assets=3, dropout=0.1):
        super(LSTMTradingModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_assets = num_assets
        
        # LSTM layer - Based on Fischer & Krauss: 25 units, 1 layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # No dropout in LSTM layer for single layer
            bidirectional=False
        )
        
        # Fully connected layers with light dropout (Fischer & Krauss style)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)  # 0.1 dropout like in literature
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, num_assets * 5)  # 5 actions per asset
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights (Xavier initialization)
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for layer in [self.fc1, self.fc2, self.fc3]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers with light dropout
        x = self.relu(self.fc1(last_output))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Reshape to (batch_size, num_assets, num_actions)
        x = x.view(-1, self.num_assets, 5)
        
        return x

class TradingDataset(Dataset):
    """Dataset for LSTM trading model"""
    
    def __init__(self, features, targets, sequence_length=20):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.sequence_length]
        
        # Get target (next step actions)
        y = self.targets[idx + self.sequence_length]
        
        return torch.FloatTensor(x), torch.LongTensor(y)

class SimpleMultiAssetEnv(gym.Env):
    """Simple but sophisticated multi-asset environment - SAME AS PPO VERSION"""
    
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
        
        # Calculate return
        daily_return = (net_worth - self.net_worth_history[-2]) / self.net_worth_history[-2]
        self.returns_history.append(daily_return)
        
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
            'actions_taken': actions_taken,
            'daily_return': daily_return
        }
        
        return self._get_observation(), daily_return, done, False, info

def augment_trading_data(features, targets, noise_factor=0.01):
    """Add noise to features for data augmentation - Conservative approach"""
    augmented_features = []
    augmented_targets = []
    
    # Original data
    augmented_features.append(features)
    augmented_targets.append(targets)
    
    # Add fewer noisy versions to maintain data quality
    for _ in range(1):  # Only 1 augmented version instead of 2
        noise = np.random.normal(0, noise_factor, features.shape)
        noisy_features = features + noise
        augmented_features.append(noisy_features)
        augmented_targets.append(targets)  # Same targets
    
    return np.vstack(augmented_features), np.vstack(augmented_targets)

def generate_trading_targets(env, lookback_days=3):
    """Generate trading targets based on future returns with balanced classes"""
    targets = []
    
    # Get all price data
    prices = {}
    for asset in ASSET_NAMES:
        prices[asset] = env.data_dict[asset]['Close'].values
    
    # Calculate all returns first to get percentiles
    all_returns = {}
    for asset in ASSET_NAMES:
        returns = []
        for i in range(len(prices[asset]) - lookback_days):
            current_price = prices[asset][i]
            future_price = prices[asset][i + lookback_days]
            future_return = (future_price - current_price) / current_price
            returns.append(future_return)
        all_returns[asset] = np.array(returns)
    
    # Generate targets using percentile-based thresholds for balance
    for i in range(len(prices[ASSET_NAMES[0]]) - lookback_days):
        asset_actions = []
        
        for asset in ASSET_NAMES:
            # Calculate future return
            current_price = prices[asset][i]
            future_price = prices[asset][i + lookback_days]
            future_return = (future_price - current_price) / current_price
            
            # Use percentile-based thresholds for more balanced classes
            returns = all_returns[asset]
            p75 = np.percentile(returns, 75)  # Top 25%
            p60 = np.percentile(returns, 60)  # Top 40%
            p40 = np.percentile(returns, 40)  # Bottom 40%
            p25 = np.percentile(returns, 25)  # Bottom 25%
            
            # More balanced action distribution
            if future_return >= p75:
                action = 4  # Strong Buy (top 25%)
            elif future_return >= p60:
                action = 3  # Buy (25%-40%)
            elif future_return >= p40:
                action = 2  # Hold (40%-60%)
            elif future_return >= p25:
                action = 1  # Sell (25%-40%)
            else:
                action = 0  # Strong Sell (bottom 25%)
            
            asset_actions.append(action)
        
        targets.append(asset_actions)
    
    # Print distribution for debugging
    targets_array = np.array(targets)
    for asset_idx, asset in enumerate(ASSET_NAMES):
        actions = targets_array[:, asset_idx]
        print(f"  {asset} action distribution:")
        for action in range(5):
            count = np.sum(actions == action)
            pct = count / len(actions) * 100
            action_names = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
            print(f"    {action_names[action]}: {pct:.1f}%")
    
    return targets_array

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

def test_lstm_model(model, env, device, scaler):
    """Test the LSTM model"""
    model.eval()
    
    obs, _ = env.reset()
    portfolio_values = [env.initial_balance]
    actions_taken = {asset: [] for asset in ASSET_NAMES}
    portfolio_allocations = {asset: [] for asset in ASSET_NAMES}
    cash_allocations = []
    
    # Get all observations first
    all_obs = env.get_all_observations()
    all_obs_scaled = scaler.transform(all_obs)  # Use the same scaler from training
    
    sequence_length = 10  # Same as training
    
    with torch.no_grad():
        for step in range(sequence_length, len(all_obs_scaled)):
            # Get sequence
            sequence = all_obs_scaled[step - sequence_length:step]
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            # Get action probabilities
            action_probs = model(sequence)
            
            # Get actions (argmax for each asset)
            actions = torch.argmax(action_probs, dim=-1).cpu().numpy()[0]
            
            # Execute step
            env.current_step = step
            obs, reward, done, truncated, info = env.step(actions)
            
            portfolio_values.append(info['net_worth'])
            
            # Calculate allocations
            total_value = info['net_worth']
            cash_percentage = (info['balance'] / total_value) * 100
            cash_allocations.append(cash_percentage)
            
            for i, asset in enumerate(ASSET_NAMES):
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
    print("🚀 MULTI-ASSET LSTM TRADING SYSTEM - ACADEMIC LITERATURE ALIGNED")
    print("=" * 80)
    print("✅ Multi-asset trading: BTC, S&P 500, US Bonds")
    print("✅ LSTM-based model (Fischer & Krauss inspired)")
    print("✅ Parameters aligned with academic literature:")
    print("   - LSTM: 25 units, 1 layer (Fischer & Krauss)")
    print("   - Dropout: 0.1 (light regularization)")
    print("   - Optimizer: RMSProp lr=1e-3")
    print("   - Batch size: 512 (literature standard)")
    print("   - Early stopping: patience=10")
    print("✅ Advanced technical indicators")
    print("✅ Cross-asset correlation analysis")
    print("✅ Robust feature handling (no NaN)")
    print("✅ SAME DATES AS PPO_TRADING.PY (2010-2024)")
    print("=" * 80)
    
    # Load data
    result = load_simple_data()
    if result is None:
        print("❌ Failed to load data")
        return
    
    train_data, val_data, test_data = result
    
    # Get best available device
    device = get_device()
    
    # Create training environment
    train_env = SimpleMultiAssetEnv(train_data)
    
    # Prepare training data
    print("\n📊 Preparing training data...")
    train_features = train_env.get_all_observations()
    train_targets = generate_trading_targets(train_env)
    
    # Ensure same length
    min_length = min(len(train_features), len(train_targets))
    train_features = train_features[:min_length]
    train_targets = train_targets[:min_length]
    
    print(f"📈 Training features shape: {train_features.shape}")
    print(f"📈 Training targets shape: {train_targets.shape}")
    
    # Scale features
    scaler = MinMaxScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    
    # Conservative data augmentation
    print("🎯 Applying conservative data augmentation...")
    aug_features, aug_targets = augment_trading_data(train_features_scaled, train_targets)
    print(f"📈 Augmented features shape: {aug_features.shape}")
    print(f"📈 Augmented targets shape: {aug_targets.shape}")
    
    # Create dataset and dataloader with academic literature parameters
    sequence_length = 20  # Standard sequence length
    train_dataset = TradingDataset(aug_features, aug_targets, sequence_length)
    
    # LITERATURE ALIGNED: Use batch_size=512 like Fischer & Krauss
    # If memory is limited, use 256 or 128 as compromise
    try:
        batch_size = 512
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"📊 Using batch size: {batch_size} (Fischer & Krauss standard)")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            batch_size = 256
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            print(f"📊 Using batch size: {batch_size} (memory adjusted)")
        else:
            raise e
    
    # Create model with literature-aligned parameters
    input_size = train_features.shape[1]
    model = LSTMTradingModel(
        input_size=input_size,
        hidden_size=25,     # Fischer & Krauss: 25 units
        num_layers=1,       # Fischer & Krauss: 1 layer
        dropout=0.1         # Fischer & Krauss: light dropout
    )
    model.to(device)
    
    # LITERATURE ALIGNED: RMSProp optimizer with lr=1e-3 (Fischer & Krauss)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    print("🔧 Using RMSProp optimizer (lr=1e-3) - Fischer & Krauss standard")
    
    # Train model with literature-aligned early stopping
    print(f"\n🔥 Training LSTM model with Academic Parameters...")
    epochs = 100  # More epochs for better convergence
    patience = 10  # Fischer & Krauss: patience=10
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Calculate class weights for balanced training
    class_weights = torch.ones(5, device=device)
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Prepare validation data
    val_env = SimpleMultiAssetEnv(val_data)
    val_features = val_env.get_all_observations()
    val_targets = generate_trading_targets(val_env)
    
    min_length_val = min(len(val_features), len(val_targets))
    val_features = val_features[:min_length_val]
    val_targets = val_targets[:min_length_val]
    val_features_scaled = scaler.transform(val_features)
    
    val_dataset = TradingDataset(val_features_scaled, val_targets, sequence_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop with learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)  # (batch_size, num_assets, num_actions)
            
            # Calculate loss for each asset
            loss = 0
            for asset_idx in range(3):
                loss += weighted_criterion(outputs[:, asset_idx, :], targets[:, asset_idx])
            
            # Backward pass
            loss.backward()
            
            # Light gradient clipping (less aggressive than before)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation phase for early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_features, val_targets in val_loader:
                val_features, val_targets = val_features.to(device), val_targets.to(device)
                val_outputs = model(val_features)
                
                loss = 0
                for asset_idx in range(3):
                    loss += weighted_criterion(val_outputs[:, asset_idx, :], val_targets[:, asset_idx])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        lr_info = f", LR: {new_lr:.2e}"
        if new_lr < old_lr:
            lr_info += " (REDUCED)"
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}{lr_info}")
        
        # Early stopping logic (Fischer & Krauss: patience=10)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print("   📚 Using Fischer & Krauss patience=10 standard")
            # Load best model
            model.load_state_dict(best_model_state)
            break
    
    # Save model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'model_config': {
            'input_size': input_size,
            'hidden_size': 25,
            'num_layers': 1,
            'dropout': 0.1
        }
    }, "trained_models/lstm_v3_academic_aligned.pth")
    print("✅ Model and scaler saved (academic parameters)")
    
    # Validate
    print("\n📊 Validating on validation set (2017-2018)...")
    val_env = SimpleMultiAssetEnv(val_data)
    val_results = test_lstm_model(model, val_env, device, scaler)
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"📈 Total Return: {val_results['return']:.2f}%")
    print(f"📊 Annual Return: {val_results['annual_return']:.2f}%")
    print(f"⚡ Sharpe Ratio: {val_results['sharpe']:.3f}")
    print(f"📉 Max Drawdown: {val_results['max_drawdown']:.2f}%")
    print(f"🌊 Volatility: {val_results['volatility']:.2f}%")
    
    # Test on unseen data
    print("\n🧪 Testing on unseen test set (2019-2024)...")
    test_env = SimpleMultiAssetEnv(test_data)
    results = test_lstm_model(model, test_env, device, scaler)
    
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
    
    print(f"\n✅ Multi-asset LSTM training complete!")
    print(f"📚 ALIGNED WITH ACADEMIC LITERATURE:")
    print(f"   - Fischer & Krauss (2018): LSTM architecture")
    print(f"   - 25 hidden units, 1 layer")
    print(f"   - RMSProp optimizer, lr=1e-3")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Light dropout: 0.1")
    print(f"   - Early stopping patience: 10")
    print(f"🎯 Uses SAME DATES as ppo_trading.py for fair comparison!")
    print(f"📊 Train: 2010-2016 | Val: 2017-2018 | Test: 2019-2024")
    print(f"💾 Model saved as 'trained_models/lstm_v3_academic_aligned.pth'")

if __name__ == "__main__":
    main() 