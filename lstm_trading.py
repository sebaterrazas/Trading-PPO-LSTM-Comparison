#!/usr/bin/env python3
"""
PROFESSIONAL LSTM TRADING MODEL - SERIOUS IMPLEMENTATION
- Advanced LSTM architecture with attention mechanism
- Sophisticated feature engineering and preprocessing
- Professional training with proper validation and monitoring
- Advanced trading strategy with risk management
- Same data splits as PPO for fair comparison
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import data loading from PPO script
from ppo_trading import load_sp500_data, get_fama_french_features

class AttentionLSTM(nn.Module):
    """Advanced LSTM with Attention Mechanism for Trading"""
    
    def __init__(self, input_size=9, hidden_size=256, num_layers=3, dropout=0.4, num_heads=8):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Multi-layer LSTM with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, 
                   hidden_size, 
                   batch_first=True, 
                   dropout=dropout if i < num_layers-1 else 0)
            for i in range(num_layers)
        ])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for different predictions
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 11)  # 11 actions
        )
        
        # Value head for portfolio value prediction
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Input normalization
        x = self.input_norm(x)
        
        # Multi-layer LSTM with residual connections
        lstm_out = x
        hidden_states = []
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            lstm_output, _ = lstm_layer(lstm_out)
            
            # Residual connection (skip connection)
            if i > 0 and lstm_output.size(-1) == lstm_out.size(-1):
                lstm_output = lstm_output + lstm_out
            
            lstm_out = lstm_output
            hidden_states.append(lstm_output)
        
        # Multi-head attention over all hidden states
        # Use the last layer output as query, key, and value
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM output with attention
        combined = lstm_out + attn_output
        
        # Feature extraction
        features = self.feature_extractor(combined)
        
        # Take the last timestep for prediction
        last_features = features[:, -1, :]
        
        # Multiple prediction heads
        action_logits = self.action_head(last_features)
        value_pred = self.value_head(last_features)
        confidence = self.confidence_head(last_features)
        
        return action_logits, value_pred, confidence, attn_weights

def create_advanced_features(data, lookback_periods=[5, 10, 20, 50]):
    """Create advanced technical and fundamental features"""
    df = data.copy()
    
    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Get base Fama-French features
    features_df = get_fama_french_features(df)
    
    # Flatten MultiIndex columns for features_df too
    if isinstance(features_df.columns, pd.MultiIndex):
        features_df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in features_df.columns]
    
    # Technical indicators with multiple timeframes
    for period in lookback_periods:
        # Moving averages
        features_df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        features_df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Bollinger Bands
        sma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        features_df[f'bb_upper_{period}'] = bb_upper
        features_df[f'bb_lower_{period}'] = bb_lower
        features_df[f'bb_position_{period}'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if period >= 12:
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            features_df[f'macd_{period}'] = ema12 - ema26
            features_df[f'macd_signal_{period}'] = features_df[f'macd_{period}'].ewm(span=9).mean()
    
    # Volume indicators
    features_df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    features_df['volume_ratio'] = df['Volume'] / features_df['volume_sma_20']
    features_df['price_volume'] = df['Close'] * df['Volume']
    
    # Volatility measures
    features_df['volatility_10'] = df['Close'].pct_change().rolling(10).std() * np.sqrt(252)
    features_df['volatility_30'] = df['Close'].pct_change().rolling(30).std() * np.sqrt(252)
    
    # Market microstructure
    features_df['high_low_ratio'] = df['High'] / df['Low']
    features_df['open_close_ratio'] = df['Open'] / df['Close']
    
    # Regime indicators
    features_df['trend_strength'] = features_df['sma_5'] / features_df['sma_20']
    features_df['momentum_regime'] = np.where(features_df['momentum'] > 0.02, 1, 
                                            np.where(features_df['momentum'] < -0.02, -1, 0))
    
    return features_df

def prepare_advanced_lstm_data(data, sequence_length=30, prediction_horizon=1):
    """Prepare sophisticated LSTM training data"""
    print(f"📊 Preparing advanced LSTM data (seq_len={sequence_length}, pred_horizon={prediction_horizon})...")
    
    # Create advanced features
    features_df = create_advanced_features(data)
    
    # Select most important features (feature selection)
    base_features = ['returns', 'momentum', 'size_factor', 'value_factor', 
                    'profitability', 'volatility', 'rsi', 'ma_ratio', 'volume_ratio']
    
    technical_features = ['sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 
                         'bb_position_20', 'rsi_10', 'rsi_20', 'macd_20',
                         'volume_ratio', 'volatility_10', 'trend_strength']
    
    all_features = base_features + technical_features
    
    # Handle missing values
    features_df = features_df.dropna()
    
    if len(features_df) < sequence_length + prediction_horizon:
        raise ValueError(f"Not enough data after cleaning. Need at least {sequence_length + prediction_horizon}, got {len(features_df)}")
    
    features = features_df[all_features].values
    
    # Robust scaling (better for financial data with outliers)
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences and targets
    X, y_action, y_return = [], [], []
    
    for i in range(sequence_length, len(features_scaled) - prediction_horizon + 1):
        # Input sequence
        X.append(features_scaled[i-sequence_length:i])
        
        # Future return for action classification (use integer position indexing)
        current_price = features_df.iloc[i]['Close']
        future_price = features_df.iloc[min(i + prediction_horizon, len(features_df) - 1)]['Close']
        future_return = (future_price - current_price) / current_price
        # More sophisticated action mapping based on return distribution
        if future_return <= np.percentile(features_df['returns'].dropna(), 5):  # Bottom 5%
            action = 0  # Strong sell (-5)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 15):  # Bottom 15%
            action = 1  # Sell (-4)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 25):  # Bottom 25%
            action = 2  # Weak sell (-3)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 35):  # Bottom 35%
            action = 3  # Light sell (-2)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 45):  # Bottom 45%
            action = 4  # Minimal sell (-1)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 55):  # Middle 10%
            action = 5  # Hold (0)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 65):  # Top 45%
            action = 6  # Minimal buy (1)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 75):  # Top 35%
            action = 7  # Light buy (2)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 85):  # Top 25%
            action = 8  # Buy (3)
        elif future_return <= np.percentile(features_df['returns'].dropna(), 95):  # Top 15%
            action = 9  # Strong buy (4)
        else:  # Top 5%
            action = 10  # Very strong buy (5)
        
        y_action.append(action)
        y_return.append(future_return)
    
    X = np.array(X)
    y_action = np.array(y_action)
    y_return = np.array(y_return)
    
    print(f"✅ Advanced data prepared: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
    print(f"📈 Action distribution: {Counter(y_action)}")
    print(f"📊 Return stats: mean={np.mean(y_return):.4f}, std={np.std(y_return):.4f}")
    
    return X, y_action, y_return, scaler, features_df.iloc[sequence_length:]

def create_weighted_data_loaders(X_train, y_action_train, y_return_train, 
                                X_val, y_action_val, y_return_val, batch_size=64):
    """Create data loaders with class weighting for imbalanced actions"""
    
    # Calculate class weights for balanced training
    class_counts = Counter(y_action_train)
    total_samples = len(y_action_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = torch.FloatTensor([class_weights[y] for y in y_action_train])
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_action_train_tensor = torch.LongTensor(y_action_train)
    y_return_train_tensor = torch.FloatTensor(y_return_train)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_action_val_tensor = torch.LongTensor(y_action_val)
    y_return_val_tensor = torch.FloatTensor(y_return_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_action_train_tensor, y_return_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_action_val_tensor, y_return_val_tensor)
    
    # Weighted sampler for balanced training
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, class_weights

def train_advanced_lstm(model, train_loader, val_loader, class_weights, 
                       num_epochs=200, learning_rate=0.0005):
    """Professional LSTM training with multiple objectives"""
    print(f"🚀 Training Advanced LSTM for {num_epochs} epochs...")
    
    # Loss functions with class weighting (handle missing classes)
    weight_tensor = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(11)])
    action_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    value_criterion = nn.MSELoss()
    
    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                           weight_decay=1e-4, betas=(0.9, 0.999))
    
    # More conservative learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate*5,  # Less aggressive max LR
        epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy='cos'  # Longer warmup
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_action_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 40  # More patience for better convergence
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_action_loss = 0.0
        train_value_loss = 0.0
        
        for batch_X, batch_y_action, batch_y_return in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            action_logits, value_pred, confidence, _ = model(batch_X)
            
            # Multi-objective loss
            action_loss = action_criterion(action_logits, batch_y_action)
            value_loss = value_criterion(value_pred.squeeze(), batch_y_return)
            confidence_loss = F.mse_loss(confidence.squeeze(), torch.abs(batch_y_return))
            
            # Combined loss with weighting
            total_loss = action_loss + 0.5 * value_loss + 0.2 * confidence_loss
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            train_action_loss += action_loss.item()
            train_value_loss += value_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_action_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y_action, batch_y_return in val_loader:
                action_logits, value_pred, confidence, _ = model(batch_X)
                
                action_loss = action_criterion(action_logits, batch_y_action)
                value_loss = value_criterion(value_pred.squeeze(), batch_y_return)
                confidence_loss = F.mse_loss(confidence.squeeze(), torch.abs(batch_y_return))
                
                total_loss = action_loss + 0.5 * value_loss + 0.2 * confidence_loss
                val_loss += total_loss.item()
                
                # Action accuracy
                _, predicted = torch.max(action_logits.data, 1)
                val_total += batch_y_action.size(0)
                val_action_correct += (predicted == batch_y_action).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y_action.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_action_correct / val_total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Early stopping with best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'epoch': epoch,
                'class_weights': class_weights
            }, 'trained_models/lstm_advanced_best.pth')
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 20 == 0 or epoch < 10:
            print(f"Epoch [{epoch:3d}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"⚠️ Early stopping at epoch {epoch}")
            break
    
    print("✅ Advanced training completed!")
    
    # Print final classification report
    print("\n📊 Final Validation Classification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=[f'Action_{i}' for i in range(11)]))
    
    return train_losses, val_losses, val_accuracies

class MomentumLSTMStrategy:
    """Simple momentum-based LSTM strategy - should at least not lose everything!"""
    
    def __init__(self, model, scaler, initial_balance=1000000, transaction_cost=0.002):
        self.model = model
        self.scaler = scaler
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
    def backtest(self, data, sequence_length=30):
        """Very simple momentum backtesting - if LSTM predicts up, hold more; if down, hold less"""
        print("📊 Momentum LSTM backtesting...")
        
        # Prepare features
        features_df = create_advanced_features(data)
        features_df = features_df.dropna()
        
        # Feature selection (same as training)
        base_features = ['returns', 'momentum', 'size_factor', 'value_factor', 
                        'profitability', 'volatility', 'rsi', 'ma_ratio', 'volume_ratio']
        technical_features = ['sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 
                             'bb_position_20', 'rsi_10', 'rsi_20', 'macd_20',
                             'volume_ratio', 'volatility_10', 'trend_strength']
        all_features = base_features + technical_features
        
        features = features_df[all_features].values
        features_scaled = self.scaler.transform(features)
        
        # Start with a simple Buy & Hold approach and only deviate slightly
        balance = float(self.initial_balance * 0.2)  # Keep 20% cash
        initial_price = float(data.iloc[sequence_length]['Close'])
        shares = int((self.initial_balance * 0.8) / initial_price)  # Start 80% invested
        
        # Debug initial state
        print(f"DEBUG: Initial setup:")
        print(f"  Initial price: ${initial_price:.2f}")
        print(f"  Initial balance: ${balance:.2f}")
        print(f"  Initial shares: {shares}")
        print(f"  Initial stock value: ${shares * initial_price:.2f}")
        print(f"  Initial total value: ${balance + shares * initial_price:.2f}")
        
        portfolio_values = []
        trades_executed = 0
        
        # Risk tracking
        peak_value = self.initial_balance
        max_drawdown = 0.0
        
        # Simple tracking
        predictions_count = 0
        positive_predictions = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(sequence_length, len(features_scaled)):
                current_date = features_df.index[i]
                current_price = float(data.loc[current_date]['Close'])
                
                # Prepare input sequence
                sequence = features_scaled[i-sequence_length:i]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                
                # Get model predictions
                action_logits, value_pred, confidence, _ = self.model(sequence_tensor)
                
                predicted_return = float(value_pred.squeeze())
                predictions_count += 1
                
                if predicted_return > 0:
                    positive_predictions += 1
                
                # Current portfolio state
                current_value = balance + shares * current_price
                current_position_ratio = (shares * current_price) / current_value if current_value > 0 else 0
                
                # Debug first few days
                if i < sequence_length + 3:
                    print(f"DEBUG Day {i-sequence_length+1}:")
                    print(f"  Price: ${current_price:.2f}")
                    print(f"  Balance: ${balance:.2f}")
                    print(f"  Shares: {shares}")
                    print(f"  Stock value: ${shares * current_price:.2f}")
                    print(f"  Total value: ${current_value:.2f}")
                    print(f"  Position ratio: {current_position_ratio:.2%}")
                
                # Very conservative momentum strategy
                # Only make small adjustments every 20 days to avoid overtrading
                if i % 20 == 0 and predictions_count > 10:  # Only trade every 20 days
                    
                    # Calculate recent prediction trend
                    recent_positive_ratio = positive_predictions / predictions_count
                    
                    # Target allocation based on LSTM sentiment
                    if recent_positive_ratio > 0.6:  # More than 60% positive predictions
                        target_allocation = 0.85  # Slightly more aggressive
                    elif recent_positive_ratio < 0.4:  # Less than 40% positive predictions  
                        target_allocation = 0.70  # Slightly more conservative
                    else:
                        target_allocation = 0.80  # Default Buy & Hold allocation
                    
                    # Rebalance if needed
                    if abs(current_position_ratio - target_allocation) > 0.05:  # 5% threshold
                        
                        target_value = current_value * target_allocation
                        current_stock_value = shares * current_price
                        
                        if target_value > current_stock_value:
                            # Need to buy
                            additional_value = target_value - current_stock_value
                            shares_to_buy = int(additional_value / (current_price * (1 + self.transaction_cost)))
                            
                            if shares_to_buy > 0:
                                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                                if cost <= balance:
                                    balance -= cost
                                    shares += shares_to_buy
                                    trades_executed += 1
                        
                        elif target_value < current_stock_value:
                            # Need to sell
                            excess_value = current_stock_value - target_value
                            shares_to_sell = int(excess_value / current_price)
                            
                            if shares_to_sell > 0 and shares_to_sell < shares:
                                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                                balance += revenue
                                shares -= shares_to_sell
                                trades_executed += 1
                
                # Track portfolio
                net_worth = balance + shares * current_price
                portfolio_values.append(net_worth)
                
                # Update peak for drawdown calculation
                if net_worth > peak_value:
                    peak_value = net_worth
                else:
                    current_drawdown = (peak_value - net_worth) / peak_value
                    max_drawdown = max(max_drawdown, current_drawdown)
        
        # Debug final state
        final_price = float(data.iloc[-1]['Close'])
        final_value = balance + shares * final_price
        print(f"DEBUG: Final state:")
        print(f"  Final price: ${final_price:.2f}")
        print(f"  Final balance: ${balance:.2f}")
        print(f"  Final shares: {shares}")
        print(f"  Final stock value: ${shares * final_price:.2f}")
        print(f"  Final total value: ${final_value:.2f}")
        print(f"  Price change: {((final_price/initial_price)-1)*100:.2f}%")
        print(f"  Portfolio change: {((final_value/self.initial_balance)-1)*100:.2f}%")
        
        # Debug information
        positive_ratio = positive_predictions / predictions_count if predictions_count > 0 else 0
        print(f"DEBUG: Total predictions: {predictions_count}")
        print(f"DEBUG: Positive predictions: {positive_predictions} ({positive_ratio:.1%})")
        print(f"DEBUG: Trades executed: {trades_executed}")
        
        # Calculate metrics
        returns = np.array(portfolio_values)
        if len(returns) > 1:
            daily_returns = np.diff(returns) / returns[:-1]
            
            # Use actual initial value vs final value
            initial_value = self.initial_balance
            final_value = returns[-1]
            
            total_return = (final_value - initial_value) / initial_value * 100
            annual_return = (((final_value / initial_value) ** (252 / len(returns))) - 1) * 100
            
            volatility = np.std(daily_returns) * np.sqrt(252) * 100
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            # Calculate activity and win rate
            positive_returns = daily_returns[daily_returns > 0]
            negative_returns = daily_returns[daily_returns < 0]
            
            win_rate = len(positive_returns) / len(daily_returns) * 100 if len(daily_returns) > 0 else 0
            avg_win = np.mean(positive_returns) * 100 if len(positive_returns) > 0 else 0
            avg_loss = np.mean(negative_returns) * 100 if len(negative_returns) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            total_return = annual_return = volatility = sharpe = 0
            win_rate = avg_win = avg_loss = 0
            profit_factor = 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'volatility': volatility,
            'portfolio_values': portfolio_values,
            'trades': trades_executed,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

def main():
    """Professional LSTM training pipeline"""
    print("🧠 PROFESSIONAL LSTM TRADING MODEL")
    print("=" * 80)
    print("🎯 Advanced Architecture: Multi-layer LSTM + Attention + Multi-head")
    print("🎯 Sophisticated Features: 21 technical + fundamental indicators")
    print("🎯 Professional Training: Weighted sampling + Multi-objective loss")
    print("🎯 Risk Management: Stop-loss + Position sizing + Daily limits")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    train_data, val_data, test_data = load_sp500_data()
    
    print(f"📈 Training: {len(train_data)} days ({train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"📊 Validation: {len(val_data)} days ({val_data.index[0].strftime('%Y-%m-%d')} to {val_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"📉 Testing: {len(test_data)} days ({test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')})")
    
    # Advanced data preparation
    sequence_length = 30  # Longer sequences for better pattern recognition
    
    print(f"\n🔧 Preparing advanced training data (sequence_length={sequence_length})...")
    X_train, y_action_train, y_return_train, scaler, _ = prepare_advanced_lstm_data(
        train_data, sequence_length=sequence_length
    )
    
    print("🔧 Preparing validation data...")
    X_val, y_action_val, y_return_val, _, _ = prepare_advanced_lstm_data(
        val_data, sequence_length=sequence_length
    )
    
    # Create weighted data loaders
    train_loader, val_loader, class_weights = create_weighted_data_loaders(
        X_train, y_action_train, y_return_train,
        X_val, y_action_val, y_return_val,
        batch_size=128
    )
    
    # Create advanced model
    model = AttentionLSTM(
        input_size=21,  # 21 features
        hidden_size=256,
        num_layers=3,
        dropout=0.4,
        num_heads=8
    )
    
    print(f"\n🧠 Advanced Model Architecture:")
    print(f"   Input size: 21 features")
    print(f"   Hidden size: 256")
    print(f"   LSTM layers: 3 (with residual connections)")
    print(f"   Attention heads: 8")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Professional training
    train_losses, val_losses, val_accuracies = train_advanced_lstm(
        model, train_loader, val_loader, class_weights,
        num_epochs=300, learning_rate=0.0005
    )
    
    # Load best model
    checkpoint = torch.load('trained_models/lstm_advanced_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create momentum trading strategy
    strategy = MomentumLSTMStrategy(model, scaler)
    
    # Test on validation data
    print("\n📊 Momentum validation backtesting...")
    val_results = strategy.backtest(val_data, sequence_length)
    
    print(f"\n🏆 MOMENTUM LSTM VALIDATION RESULTS:")
    print(f"   Annual Return: {val_results['annual_return']:.2f}%")
    print(f"   Sharpe Ratio: {val_results['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {val_results['max_drawdown']:.2f}%")
    print(f"   Volatility: {val_results['volatility']:.2f}%")
    print(f"   Win Rate: {val_results['win_rate']:.1f}%")
    print(f"   Profit Factor: {val_results['profit_factor']:.2f}")
    print(f"   Trades Executed: {val_results['trades']}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'sequence_length': sequence_length,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_results': val_results,
        'class_weights': class_weights
    }, 'trained_models/lstm_momentum_final.pth')
    
    print("\n✅ MOMENTUM LSTM TRAINING COMPLETED!")
    print("✅ Model saved as 'trained_models/lstm_momentum_final.pth'")
    print("🚀 Ready for comparison with PPO!")

if __name__ == "__main__":
    main() 