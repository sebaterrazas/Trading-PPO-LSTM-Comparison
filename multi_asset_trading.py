import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm.auto import tqdm
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class MultiAssetEnvironment(gym.Env):
    def __init__(self, 
                 assets: List[str], 
                 start_date: str, 
                 end_date: str, 
                 initial_balance: float = 100000,
                 transaction_cost: float = 0.001,  # 0.1% transaction cost
                 lookback_window: int = 30,
                 verbose: bool = False,  # Add verbose flag
                 render_mode: Optional[str] = None):  # Add render_mode parameter
        
        super(MultiAssetEnvironment, self).__init__()
        
        # Store initialization parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.verbose = verbose  # Store verbose flag
        self.render_mode = render_mode  # Store render_mode
        
        # Initialize data storage
        self.asset_data = {}
        self.original_data = {}  # Store original data before normalization
        self.scalers = {}
        self.portfolio_value = []
        self.positions = {asset: 0.0 for asset in assets}
        self.balance = initial_balance
        
        self.assets = assets
        self.dates = []
        self.current_step = 0
        
        # Initialize action space
        # For each asset: -1 (short), 0 (neutral), 1 (long)
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(len(assets),),  # (4,) para 4 activos
            dtype=np.float32
        )
        
        # Initialize with None, will be set after data is loaded
        self.observation_space = None
        
        # Download and prepare data
        self._prepare_data(start_date, end_date)
        
        # Get a sample observation to set the correct shape
        obs_sample = self._get_observation()
        
        # Calculate the number of features per asset
        num_features = obs_sample.shape[1] // len(self.assets)
        
        # Set the observation space with the correct shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, num_features * len(self.assets)),
            dtype=np.float32
        )
        
        print(f"Observation space shape: {self.observation_space.shape}")
        print(f"Number of features per asset: {num_features}")
        
    def _prepare_data(self, start_date: str, end_date: str):
        print(f"Downloading data for {', '.join(self.assets)}...")
        
        raw_data = {}
        for asset in self.assets:
            df = yf.download(asset, start=start_date, end=end_date)
            if df.empty:
                print(f"Warning: No data found for {asset}")
                continue
            df = self._add_technical_indicators(df)
            raw_data[asset] = df
        
        if not raw_data:
            raise ValueError("No valid data was downloaded for any asset")
        
        # Find common columns across all assets (case insensitive)
        common_columns = None
        for asset, df in raw_data.items():
            # Get column names in lowercase for comparison
            cols = set(col.lower() for col in df.columns)
            if common_columns is None:
                common_columns = cols
            else:
                common_columns = common_columns.intersection(cols)
        
        if not common_columns:
            raise ValueError("No common columns found across assets")
            
        # Convert back to list and sort for consistency
        common_columns = sorted(list(common_columns))
        print(f"Common columns: {common_columns}")
        
        # Find common dates
        common_dates = None
        for df in raw_data.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(df.index)
        
        if not common_dates:
            raise ValueError("No common dates found across assets")
            
        self.dates = sorted(list(common_dates))
        print(f"Common dates: {len(self.dates)} trading days")
        
        # Process each asset's data
        for asset in raw_data.keys():
            # Get only the common columns and dates (case insensitive)
            df = raw_data[asset].copy()
            df.columns = df.columns.str.lower()
            df = df.loc[self.dates, common_columns].copy()
            
            # Drop any remaining NaN values
            df = df.dropna()
            
            if df.empty:
                print(f"Warning: No valid data remaining for {asset} after processing")
                continue
                
            # Store the original data before normalization
            self.original_data[asset] = df.copy()
            
            # Normalize the data
            self.scalers[asset] = StandardScaler()
            scaled_values = self.scalers[asset].fit_transform(df.values)
            
            # Store the scaled data
            self.asset_data[asset] = pd.DataFrame(
                data=scaled_values,
                index=df.index,
                columns=df.columns
            )
        
        # Verify that all assets have the same number of features
        num_features = [self.asset_data[asset].shape[1] for asset in self.asset_data]
        if len(set(num_features)) != 1:
            raise ValueError(f"Inconsistent number of features across assets: {dict(zip(self.asset_data.keys(), num_features))}")
            
        print(f"All assets have {num_features[0]} features")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            # For MultiIndex, we'll work with the first level only
            df.columns = df.columns.get_level_values(0).str.lower()
        else:
            # For regular Index
            df.columns = df.columns.str.lower()
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the DataFrame")
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _get_observation(self) -> np.ndarray:
        obs = []
        current_date = self.dates[self.current_step]
        
        # Get the number of features per asset (should be the same for all assets)
        num_features = None
        
        for asset in self.assets:
            # Get lookback window of data
            idx = self.asset_data[asset].index.get_loc(current_date)
            start_idx = max(0, idx - self.lookback_window + 1)
            asset_data = self.asset_data[asset].iloc[start_idx:idx+1]
            
            # Store the number of features (should be the same for all assets)
            if num_features is None:
                num_features = len(asset_data.columns)
            
            # Convert to numpy array
            asset_values = asset_data.values
            
            # Pad with zeros if not enough history
            if len(asset_values) < self.lookback_window:
                padding = np.zeros((self.lookback_window - len(asset_values), num_features))
                asset_values = np.vstack([padding, asset_values])
            
            obs.append(asset_values)
        
        # Stack all asset data along the feature dimension
        if not obs:
            raise ValueError("No observation data available")
            
        obs = np.concatenate(obs, axis=1)
        
        # Ensure the shape is correct
        expected_shape = (self.lookback_window, num_features * len(self.assets))
        if obs.shape != expected_shape:
            obs = np.zeros(expected_shape, dtype=np.float32)
            print(f"Warning: Observation shape {obs.shape} does not match expected shape {expected_shape}")
        
        return obs.astype(np.float32)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Get current prices
        current_prices = {}
        current_date = self.dates[self.current_step]
        
        if self.verbose and self.current_step % 100 == 0:  # Only print every 100 steps
            print(f"Step {self.current_step}/{len(self.dates)} - Date: {current_date.strftime('%Y-%m-%d')}")
            print(f"Available columns for {self.assets[0]}: {self.asset_data[self.assets[0]].columns.tolist()}")
        
        for i, asset in enumerate(self.assets):
            try:
                current_prices[asset] = self.asset_data[asset].loc[current_date, 'close']
            except KeyError as e:
                raise KeyError(f"Error accessing 'close' price for {asset} at {current_date}. Available columns: {self.asset_data[asset].columns.tolist()}") from e
        
        # Calculate portfolio value before rebalancing
        portfolio_value = self.balance
        for asset, shares in self.positions.items():
            portfolio_value += shares * current_prices[asset]
        
        # Execute trades based on actions
        self._execute_trades(actions, current_prices)
        
        # Calculate new portfolio value
        new_portfolio_value = self.balance
        for asset, shares in self.positions.items():
            new_portfolio_value += shares * current_prices[asset]
        
        # Calculate reward (profit/loss)
        reward = (new_portfolio_value - portfolio_value) / portfolio_value if portfolio_value > 0 else 0
        
        # Update portfolio value history
        self.portfolio_value.append(new_portfolio_value)
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.dates) - 1
        truncated = False  # We don't have a truncation condition
        
        # Get new observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': float(new_portfolio_value),
            'balance': float(self.balance),
            'positions': {k: float(v) for k, v in self.positions.items()},
            'prices': {k: float(v) for k, v in current_prices.items()},
            'current_step': self.current_step,
            'current_date': self.dates[min(self.current_step, len(self.dates)-1)].strftime('%Y-%m-%d'),
            'total_steps': len(self.dates)
        }
        
        return obs, reward, terminated, truncated, info
    
    def _execute_trades(self, actions: np.ndarray, current_prices: Dict[str, float]):
        # Calculate current portfolio value
        total_value = self.balance
        for asset, shares in self.positions.items():
            total_value += shares * current_prices[asset]
        
        # Calculate target allocation for each asset
        target_values = {}
        for i, asset in enumerate(self.assets):
            # Scale action from [-1, 1] to [0, 1] for allocation
            allocation = (np.clip(actions[i], -1, 1) + 1) / 2  # Ensure action is in [-1, 1] then scale to [0, 1]
            target_values[asset] = total_value * allocation
        
        # Normalize target values to ensure they sum to total_value
        total_target = sum(target_values.values())
        if total_target > 0:  # Avoid division by zero
            target_values = {k: (v / total_target) * total_value for k, v in target_values.items()}
        
        # Execute trades
        for asset in self.assets:
            current_value = self.positions[asset] * current_prices[asset]
            target_value = target_values[asset]
            
            # Skip if no change needed
            if np.isclose(current_value, target_value, rtol=1e-5):
                continue
                
            # Calculate how much to buy/sell
            if target_value > current_value:
                # Buy more
                amount_to_buy = (target_value - current_value) / current_prices[asset]
                cost = amount_to_buy * current_prices[asset] * (1 + self.transaction_cost)
                
                # Ensure we have enough balance
                if cost > self.balance:
                    # Adjust to what we can afford
                    cost = self.balance
                    amount_to_buy = (cost / (1 + self.transaction_cost)) / current_prices[asset]
                
                if amount_to_buy > 0:
                    self.positions[asset] += amount_to_buy
                    self.balance -= cost
                    
            else:
                # Sell
                amount_to_sell = (current_value - target_value) / current_prices[asset]
                amount_to_sell = min(amount_to_sell, self.positions[asset])
                
                if amount_to_sell > 0:
                    revenue = amount_to_sell * current_prices[asset] * (1 - self.transaction_cost)
                    self.positions[asset] -= amount_to_sell
                    self.balance += revenue
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window  # Start after enough history
        self.balance = self.initial_balance
        self.positions = {asset: 0.0 for asset in self.assets}
        self.portfolio_value = []  # Track portfolio value over time
        
        # Initial observation and info
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'positions': self.positions.copy(),
            'portfolio_value': self.balance
        }
        
        return obs, info
    
    def render(self, mode='human'):
        if mode == 'human':
            current_date = self.dates[self.current_step]
            portfolio_value = self.balance
            
            # Calculate total portfolio value
            for asset, shares in self.positions.items():
                if shares > 0:
                    price = self.original_data[asset].loc[current_date, 'close']
                    portfolio_value += shares * price
            
            print(f"\nDate: {current_date}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Cash: ${self.balance:,.2f}")
            print("Positions:")
            for asset, shares in self.positions.items():
                if shares > 0:
                    price = self.original_data[asset].loc[current_date, 'close']


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe.require_grad = False
        
        # Use sin for even indices and cos for odd indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it's not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to the input
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        # Get dimensions from observation space
        self.sequence_length = observation_space.shape[0]  # 30 time steps
        self.num_features = observation_space.shape[1]     # 60 features per time step
        
        # Hyperparameters
        self.d_model = 64  # Match features_dim
        self.nhead = 8
        self.num_layers = 2  # Reduced from 3 to 2 for better stability
        self.dropout = 0.1
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(self.num_features, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
            max_len=self.sequence_length
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, features_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, num_features)
        x = self.input_proj(observations)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)            # Add positional encoding
        x = self.transformer_encoder(x)    # Process through transformer
        x = x.mean(dim=1)                  # Mean pooling over sequence
        return self.output_proj(x)         # Project to features_dim


class MultiAssetLSTMPolicy(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 features_dim: int = 256,  # Default output dimension
                 **kwargs):
        # Get number of assets from kwargs
        self.num_assets = kwargs.get('num_assets', 4)  # Default to 4 assets
        
        # Calculate the total number of features
        self.seq_len = observation_space.shape[0]
        self.feature_dim = observation_space.shape[1]
        
        # Initialize the base class with the observation space and features_dim
        super(MultiAssetLSTMPolicy, self).__init__(observation_space, features_dim)
        
        # Set hidden size based on features_dim
        hidden_size = features_dim // (2 * self.num_assets)
        
        # Enhanced input projection with better initialization
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim // self.num_assets, hidden_size * 2),
            nn.BatchNorm1d(self.seq_len),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # First LSTM layer with more capacity
        self.lstm1 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Second LSTM layer with residual connection
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,  # Bidirectional
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Enhanced self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.2,
            batch_first=True,
            bias=False
        )
        
        # Enhanced normalization and projection
        self.norm1 = nn.LayerNorm(hidden_size * 2)  # After LSTM1 (bidirectional)
        self.norm2 = nn.LayerNorm(hidden_size)       # After LSTM2
        self.norm3 = nn.LayerNorm(hidden_size)       # After attention
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_size * 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    for name, param in self.lstm1.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            param.data.fill_(0)
                elif 'attention' in name:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0.0)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape if needed (batch_size, seq_len * features) -> (batch_size, seq_len, features)
        if len(observations.shape) == 2:
            observations = observations.reshape(-1, self.seq_len, self.feature_dim)
        
        batch_size = observations.size(0)
        features_per_asset = self.feature_dim // self.num_assets
        all_features = []
        
        # Process each asset independently
        for i in range(self.num_assets):
            # Extract features for this asset
            start_idx = i * features_per_asset
            end_idx = (i + 1) * features_per_asset
            asset_data = observations[:, :, start_idx:end_idx]
            
            # Project to higher dimension
            projected = self.input_proj(asset_data)
            
            # First bidirectional LSTM layer
            lstm1_out, _ = self.lstm1(projected)
            lstm1_out = self.norm1(lstm1_out)
            
            # Process with second LSTM
            lstm2_out, _ = self.lstm2(lstm1_out)
            
            # Residual connection from LSTM1 to LSTM2 output
            lstm2_out = self.norm2(lstm2_out + lstm1_out[:, :, :lstm2_out.size(2)])
            
            # Apply self-attention
            attn_out, _ = self.self_attention(
                lstm2_out, lstm2_out, lstm2_out,
                need_weights=False
            )
            
            # Residual connection and layer norm
            attn_out = self.norm3(attn_out + lstm2_out)
            
            # Get the last time step's output and apply feature projection
            last_out = attn_out[:, -1, :]
            projected_features = self.feature_proj(last_out)
            all_features.append(projected_features)
        
        # Concatenate all asset features
        return torch.cat(all_features, dim=1)


class TransformerPolicy(nn.Module):
    """Policy network with Transformer feature extractor"""
    
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, **kwargs):
        super(TransformerPolicy, self).__init__()
        
        # Feature extractor
        self.features_extractor = TransformerFeaturesExtractor(
            observation_space,
            features_dim=128,
            nhead=4,
            num_layers=3
        )
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.shape[0]),
            nn.Tanh()
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Project input to hidden size
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Get the last time step's output for policy and value
        last_out = transformer_out[:, -1, :]
        
        # Get actions for each asset
        actions = self.policy_net(last_out)
        
        # Get state value
        value = self.value_net(last_out)
        
        return actions, value


def train_multi_asset_agent(model_type='transformer', timesteps=100000):
    # Define assets to trade (stocks, bonds, crypto)
    assets = [
        'SPY',  # S&P 500 ETF
        'TLT',  # 20+ Year Treasury Bond ETF
        'BTC-USD',  # Bitcoin
        'GLD'   # Gold ETF
    ]
    
    # Create environment
    env = MultiAssetEnvironment(
        assets=assets,
        start_date='2015-01-01',
        end_date='2023-01-01',
        initial_balance=100000,
        transaction_cost=0.001,  # 0.1% transaction cost
        lookback_window=30,
        render_mode=None
    )
    
    # Set up policy kwargs based on model type
    if model_type == 'transformer':
        policy_kwargs = {
            'features_extractor_class': TransformerFeaturesExtractor,
            'features_extractor_kwargs': {'features_dim': 64},
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
            'activation_fn': nn.Tanh,
            'ortho_init': True
        }
        policy = 'MlpPolicy'
    elif model_type == 'lstm':
        policy_kwargs = {
            'features_extractor_class': MultiAssetLSTMPolicy,
            'features_extractor_kwargs': {
                'features_dim': 256,  # Set the output dimension
                'num_assets': len(assets)  # Pass number of assets as a keyword argument
            },
            'net_arch': [],
            'activation_fn': nn.Tanh,
            'ortho_init': True
        }
        policy = 'MlpPolicy'
    else:  # Simple MLP baseline
        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
            'activation_fn': nn.Tanh,
            'ortho_init': True
        }
        policy = 'MlpPolicy'
    
    # Create model with appropriate hyperparameters for PPO
    model = PPO(
        policy=policy,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,       # Number of steps to run for each environment per update
        batch_size=64,      # Minibatch size for training
        n_epochs=10,        # Number of epochs when optimizing the surrogate loss
        gamma=0.99,         # Discount factor
        gae_lambda=0.95,    # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,     # Clipping parameter for the policy
        clip_range_vf=None, # Clipping parameter for the value function
        ent_coef=0.0,       # Entropy coefficient
        vf_coef=0.5,        # Value function coefficient in the loss
        max_grad_norm=0.5,  # Maximum norm for the gradient clipping
        use_sde=False,      # Whether to use generalized State Dependent Exploration
        sde_sample_freq=-1, # Sample a new noise matrix every n steps
        target_kl=0.1,      # Limit the KL divergence between updates
        tensorboard_log=f"./tensorboard_logs/{model_type}",
        verbose=1,
        device='auto'       # Automatically select GPU if available
    )
    
    # Train the model
    print(f"Training {model_type.upper()} model for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        progress_bar= False
    )
    
    # Save the trained model
    os.makedirs("trained_models", exist_ok=True)
    model_path = f"trained_models/{model_type}_asset_trader"
    model.save(model_path)
    print(f"Model saved to '{model_path}'")
    
    return model


def compare_models():
    import os
    import shutil
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print("\n=== Model Comparison ===")
    
    # Create directory if it doesn't exist
    os.makedirs("trained_models", exist_ok=True)
    
    results = {}
    all_returns = {}
    timesteps = 5000
    
    # Train and evaluate LSTM model
    print("\n=== Training LSTM Model ===")
    lstm_path = "trained_models/lstm_asset_trader"
    if os.path.exists(lstm_path):
        print("Removing existing LSTM model...")
        shutil.rmtree(lstm_path)
    
    print("Training LSTM model from scratch...")
    train_multi_asset_agent(model_type='lstm', timesteps=timesteps)
    
    print("\nEvaluating LSTM model...")
    lstm_returns, lstm_final = evaluate_multi_asset_agent(lstm_path)
    results['lstm'] = {"returns": lstm_returns, "final_value": lstm_final}
    all_returns['lstm'] = lstm_returns
    
    # Train and evaluate Transformer model
    print("\n=== Training Transformer Model ===")
    transformer_path = "trained_models/transformer_asset_trader"
    if os.path.exists(transformer_path):
        print("Removing existing Transformer model...")
        shutil.rmtree(transformer_path)
    
    print("Training Transformer model from scratch...")
    train_multi_asset_agent(model_type='transformer', timesteps=timesteps)
    
    print("\nEvaluating Transformer model...")
    transformer_returns, transformer_final = evaluate_multi_asset_agent(transformer_path)
    results['transformer'] = {"returns": transformer_returns, "final_value": transformer_final}
    all_returns['transformer'] = transformer_returns
    
    # Show final comparison
    print("\n=== Final Model Comparison ===")
    for model_type, result in results.items():
        print(f"\n{model_type.upper()} Model:")
        print(f"  - Final Portfolio Value: ${result['final_value']:,.2f}")
        print(f"  - Total Return: {result['returns']:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(14, 7))
    
    # Plot LSTM performance
    lstm_dates = pd.date_range(start='2023-01-01', periods=len(all_returns['lstm']), freq='D')
    plt.plot(lstm_dates, all_returns['lstm'], label='LSTM', linewidth=2)
    
    # Plot Transformer performance
    transformer_dates = pd.date_range(start='2023-01-01', periods=len(all_returns['transformer']), freq='D')
    plt.plot(transformer_dates, all_returns['transformer'], label='Transformer', linewidth=2)
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_plot_path = "model_comparison.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    
    print(f"\nComparison plot saved as '{comparison_plot_path}'")
    
    # Return results for both models
    return (
        (results['lstm']['returns'], results['lstm']['final_value']),
        (results['transformer']['returns'], results['transformer']['final_value'])
    )


def evaluate_multi_asset_agent(model_path: str = "trained_models/transformer_asset_trader"):
    # Define assets
    assets = [
        'SPY',  # S&P 500 ETF
        'TLT',  # 20+ Year Treasury Bond ETF
        'BTC-USD',  # Bitcoin
        'GLD'   # Gold ETF
    ]
    
    # Create evaluation environment with test data
    eval_env = MultiAssetEnvironment(
        assets=assets,
        start_date='2023-01-01',  # Test period
        end_date='2023-12-31',
        initial_balance=100000,
        transaction_cost=0.001,
        lookback_window=30,
        verbose=False
    )
    
    # Load the trained model
    model = PPO.load(model_path, env=eval_env)
    
    # Store portfolio values for plotting
    portfolio_values = [eval_env.initial_balance]
    dates = [eval_env.dates[0]]
    
    # Run evaluation
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = eval_env.step(action)
        
        # Calculate current portfolio value
        current_value = eval_env.balance
        for asset, shares in eval_env.positions.items():
            current_value += shares * eval_env.original_data[asset].loc[eval_env.dates[eval_env.current_step-1], 'close']
        
        portfolio_values.append(current_value)
        dates.append(eval_env.dates[eval_env.current_step-1])
    
    # Get final portfolio value and calculate metrics
    final_value = portfolio_values[-1]
    initial_balance = eval_env.initial_balance
    returns = ((final_value - initial_balance) / initial_balance) * 100
    
    # Calculate daily returns for risk metrics
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.sqrt(252) * (np.mean(daily_returns) / np.std(daily_returns)) if len(daily_returns) > 1 else 0
    
    # Plot portfolio value over time
    plt.figure(figsize=(14, 6))
    
    # Plot portfolio value
    plt.subplot(1, 2, 1)
    plt.plot(dates, portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot daily returns distribution
    plt.subplot(1, 2, 2)
    plt.hist(daily_returns, bins=30, alpha=0.7, color='green')
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot with model name
    model_name = os.path.basename(model_path)
    plot_path = f'portfolio_performance_{model_name}.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Print performance metrics
    print("\n=== Performance Metrics ===")
    print(f"Initial balance: ${initial_balance:,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {returns:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Number of trades: {len(dates) - 1}")
    print(f"Portfolio performance plot saved as '{plot_path}'")
    
    return returns, final_value


def main():
    try:
        # Compare LSTM and Transformer models
        results = compare_models()
        
        # Print final summary
        print("\n=== Training Complete ===")
        print("Both models have been trained and saved to the 'trained_models' directory.")
        print("\nTo evaluate the models, you can now run:")
        print("1. evaluate_multi_asset_agent(model_path='trained_models/lstm_asset_trader')")
        print("2. evaluate_multi_asset_agent(model_path='trained_models/transformer_asset_trader')")
        
        return results
        
    except Exception as e:
        print(f"\nAn error occurred during model comparison: {str(e)}")
        print("Trying to run a single model training instead...")
        # Train and evaluate a single model as fallback
        model = train_multi_asset_agent(model_type='transformer', timesteps=10000)
        evaluate_multi_asset_agent(model_path='trained_models/transformer_asset_trader')
        return {'model': model}

if __name__ == "__main__":
    main()
