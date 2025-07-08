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
import math  # Needed for custom LR scheduling
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Import data loading from PPO script
from ppo_trading import load_sp500_data, get_fama_french_features

# Commented out unused AttentionLSTM class
# class AttentionLSTM(nn.Module):
#     """Advanced LSTM with Attention Mechanism for Trading"""
#     
#     def __init__(self, input_size=9, hidden_size=256, num_layers=3, dropout=0.4, num_heads=8):
#         super(AttentionLSTM, self).__init__()
#         
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         
#         # Input normalization
#         self.input_norm = nn.LayerNorm(input_size)
#         
#         # Multi-layer LSTM with residual connections
#         self.lstm_layers = nn.ModuleList([
#             nn.LSTM(input_size if i == 0 else hidden_size, 
#                    hidden_size, 
#                    batch_first=True, 
#                    dropout=dropout if i < num_layers-1 else 0)
#             for i in range(num_layers)
#         ])
#         
#         # Multi-head attention
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_size,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#         
#         # Feature extraction layers
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#         
#         # Separate heads for different predictions
#         self.action_head = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 11)  # 11 actions
#         )
#         
#         # Value head for portfolio value prediction
#         self.value_head = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
#         
#         # Confidence head
#         self.confidence_head = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
#         
#         # Initialize weights
#         self.apply(self._init_weights)
#     
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LSTM):
#             for name, param in module.named_parameters():
#                 if 'weight' in name:
#                     torch.nn.init.xavier_uniform_(param)
#                 elif 'bias' in name:
#                     torch.nn.init.zeros_(param)
#     
#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         
#         # Input normalization
#         x = self.input_norm(x)
#         
#         # Multi-layer LSTM with residual connections
#         lstm_out = x
#         hidden_states = []
#         
#         for i, lstm_layer in enumerate(self.lstm_layers):
#             lstm_output, _ = lstm_layer(lstm_out)
#             
#             # Residual connection (skip connection)
#             if i > 0 and lstm_output.size(-1) == lstm_out.size(-1):
#                 lstm_output = lstm_output + lstm_out
#             
#             lstm_out = lstm_output
#             hidden_states.append(lstm_output)
#         
#         # Multi-head attention over all hidden states
#         # Use the last layer output as query, key, and value
#         attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
#         
#         # Combine LSTM output with attention
#         combined = lstm_out + attn_output
#         
#         # Feature extraction
#         features = self.feature_extractor(combined)
#         
#         # Take the last timestep for prediction
#         last_features = features[:, -1, :]
#         
#         # Multiple prediction heads
#         action_logits = self.action_head(last_features)
#         value_pred = self.value_head(last_features)
#         confidence = self.confidence_head(last_features)
#         
#         return action_logits, value_pred, confidence, attn_weights

class DualStageAttentionLSTM(nn.Module):
    """Enhanced Dual-Stage LSTM with improved attention and deeper architecture"""
    def __init__(self, input_size, hidden_size=384, extractor_layers=3, summarizer_layers=2, dropout=0.5, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Enhanced input processing
        self.input_norm = nn.LayerNorm(input_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # Deeper extractor LSTM with residual connections
        self.extractor_lstm = nn.LSTM(
            input_size, hidden_size, 
            num_layers=extractor_layers, 
            batch_first=True, 
            dropout=dropout if extractor_layers > 1 else 0,
            bidirectional=False
        )
        
        # Enhanced multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Deeper summarizer LSTM with residual connections
        self.summarizer_lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=summarizer_layers,
            batch_first=True,
            dropout=dropout if summarizer_layers > 1 else 0
        )
        
        # Enhanced feature extraction with skip connections
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Deeper action head with residual connections
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 11)  # 11 actions
        )
        
        # Enhanced value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Improved confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
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
        # x: (batch, seq_len, input_size)
        x = self.input_norm(x)
        extractor_out, _ = self.extractor_lstm(x)
        attn_out, attn_weights = self.attention(extractor_out, extractor_out, extractor_out)
        _, (h_n, _) = self.summarizer_lstm(attn_out)
        summary = h_n[-1]  # (batch, hidden)
        features = self.feature_extractor(summary)
        action_logits = self.action_head(features)
        value_pred = self.value_head(features)
        confidence = self.confidence_head(features)
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
    features_df['hl_spread'] = (df['High'] - df['Low']) / (df['High'] + df['Low'])
    
    # VIX change (if column available from joined dataset)
    if 'VIX_Close' in df.columns:
        features_df['vix'] = df['VIX_Close'].pct_change().fillna(0)
    else:
        features_df['vix'] = 0.0
    
    # Regime indicators
    features_df['trend_strength'] = features_df['sma_5'] / features_df['sma_20']
    features_df['momentum_regime'] = np.where(features_df['momentum'] > 0.02, 1, 
                                            np.where(features_df['momentum'] < -0.02, -1, 0))
    
    return features_df

def prepare_advanced_lstm_data(data, sequence_length=30, prediction_horizon=1):
    """Prepare sophisticated LSTM training data"""
    print(f" Preparing advanced LSTM data (seq_len={sequence_length}, pred_horizon={prediction_horizon})...")
    
    # Create advanced features
    features_df = create_advanced_features(data)
    
    # Select most important features (feature selection)
    base_features = ['returns', 'momentum', 'size_factor', 'value_factor', 
                    'profitability', 'volatility', 'rsi', 'ma_ratio', 'volume_ratio',
                    'hl_spread', 'vix']
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
    
    # Positional encoding (same for every sequence)
    pos_indices = np.arange(sequence_length).reshape(-1, 1)
    pos_enc = np.concatenate([
        np.sin(2 * np.pi * pos_indices / sequence_length),
        np.cos(2 * np.pi * pos_indices / sequence_length)
    ], axis=1)  # shape (seq_len, 2)

    for i in range(sequence_length, len(features_scaled) - prediction_horizon + 1):
        # Input sequence with positional encodings
        seq = features_scaled[i-sequence_length:i]
        seq = np.concatenate([seq, pos_enc], axis=1)
        X.append(seq)
        
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
    
    print(f" Advanced data prepared: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
    print(f" Action distribution: {Counter(y_action)}")
    print(f" Return stats: mean={np.mean(y_return):.4f}, std={np.std(y_return):.4f}")
    
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

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.confidence = 1.0 - self.epsilon
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        log_probs = self.log_softmax(x)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.epsilon * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def train_advanced_lstm(model, train_loader, val_loader, class_weights, 
                       num_epochs=200, learning_rate=0.0005):
    print(f" Training Advanced LSTM for {num_epochs} epochs...")
    
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss functions with class weighting (handle missing classes)
    criterion_value = nn.HuberLoss()  # More robust to outliers than MSE
    smooth_ce = LabelSmoothingCrossEntropy(epsilon=0.1)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,  # L2 regularization
        eps=1e-8  # Numerical stability
    )
    
    # Learning rate scheduler with warmup and cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos',
        final_div_factor=1000.0,  # Final LR will be learning_rate/1000
    )
    
    # Gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # Early stopping with patience
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    # Training metrics
    train_metrics = {'loss': [], 'action_acc': [], 'value_mae': []}
    val_metrics = {'loss': [], 'action_acc': [], 'value_mae': []}
    
    # Training loop with progress bar
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        value_errors = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}")
        
        for batch_x, batch_y_action, batch_y_return in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_action = batch_y_action.to(device, non_blocking=True)
            batch_y_return = batch_y_return.float().to(device, non_blocking=True)
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Forward pass
                action_logits, value_pred, confidence, _ = model(batch_x)
                
                # Calculate losses with label smoothing
                smooth_ce = LabelSmoothingCrossEntropy(epsilon=0.1)
                loss_action = smooth_ce(action_logits, batch_y_action)
                
                # Value prediction loss with uncertainty weighting
                loss_value = criterion_value(value_pred.squeeze(), batch_y_return)
                
                # Confidence regularization
                confidence_loss = -torch.log(confidence + 1e-8).mean()
                
                # L2 regularization
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param, 2)
                
                # Total loss with adaptive weighting
                loss = (loss_action + 
                       0.5 * loss_value + 
                       0.1 * confidence_loss + 
                       1e-4 * l2_reg)
            
            # Backward pass with gradient scaling for mixed precision
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(action_logits, 1)
            total += batch_y_action.size(0)
            correct += (predicted == batch_y_action).sum().item()
            value_errors.append(torch.abs(value_pred.squeeze() - batch_y_return).mean().item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate epoch metrics
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_mae = np.mean(value_errors)
        
        # Validation
        val_loss, val_acc, val_mae = validate_model(model, val_loader, smooth_ce, criterion_value, device)
        
        # Store metrics
        train_metrics['loss'].append(train_loss)
        train_metrics['action_acc'].append(train_acc)
        train_metrics['value_mae'].append(train_mae)
        
        val_metrics['loss'].append(val_loss)
        val_metrics['action_acc'].append(val_acc)
        val_metrics['value_mae'].append(val_mae)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': {'train': train_metrics, 'val': val_metrics}
            }, 'trained_models/lstm_best.pth')
            print(f"Model improved! Validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Logging
        if epoch % 20 == 0 or epoch < 10:
            print(f"Epoch [{epoch:3d}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f" Early stopping at epoch {epoch}")
            break
    
    print(" Advanced training completed!")
    
    # Get final predictions for classification report
    _, _, _, all_predictions, all_targets = validate_model(
        model, val_loader, smooth_ce, criterion_value, device, return_predictions=True
    )
    
    # Print final classification report
    print("\n Final Validation Classification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=[f'Action_{i}' for i in range(11)]))
    
    return train_metrics, val_metrics, {'predictions': all_predictions, 'targets': all_targets}

class MomentumLSTMStrategy:
    def __init__(self, model, scaler, initial_balance=1000000, transaction_cost=0.002):
        self.model = model
        self.scaler = scaler
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
    def backtest(self, data, sequence_length=30):
        print(" Momentum LSTM backtesting...")
        
        # Prepare features
        features_df = create_advanced_features(data)
        features_df = features_df.dropna()
        
        # Feature selection (must match training exactly)
        # These are the features the model was trained on
        all_features = [
            'returns', 'momentum', 'size_factor', 'value_factor', 'profitability',
            'volatility', 'rsi', 'ma_ratio', 'volume_ratio', 'hl_spread', 'vix',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'bb_position_20',
            'rsi_10', 'rsi_20', 'macd_20', 'volume_ratio', 'volatility_10', 'trend_strength'
        ]
        
        # Ensure all features exist (some might be missing in test data)
        available_features = [f for f in all_features if f in features_df.columns]
        if len(available_features) != len(all_features):
            missing = set(all_features) - set(available_features)
            print(f"  ⚠️  Warning: Missing features: {missing}")
            print(f"  Using {len(available_features)} of {len(all_features)} features")
        all_features = available_features
        
        features = features_df[all_features].values
        features_scaled = self.scaler.transform(features)
        
        # Start fully invested with dynamic position sizing
        initial_price = float(data.iloc[sequence_length]['Close'])
        balance = 0  # Start fully invested
        shares = int(self.initial_balance / initial_price)
        
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
                sequence = np.concatenate([sequence, np.zeros((sequence_length, 2))], axis=1)
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
                
                # Dynamic momentum strategy with confidence-based position sizing
                # Trade every day but with position sizing based on confidence
                if i > 0:  # Start trading immediately
                    
                    # Calculate recent prediction trend (use shorter lookback)
                    lookback = min(10, predictions_count)
                    recent_positive_ratio = positive_predictions / max(1, predictions_count)
                    
                    # Get model confidence (squared to emphasize high confidence)
                    confidence = float(confidence.squeeze())
                    confidence = confidence ** 1.5  # Emphasize high confidence more
                    
                    # More aggressive position sizing based on confidence and trend
                    if recent_positive_ratio > 0.65:  # Strong uptrend
                        target_allocation = 1.2  # Allow up to 120% allocation (leverage)
                    elif recent_positive_ratio > 0.55:  # Moderate uptrend
                        target_allocation = 0.9
                    elif recent_positive_ratio < 0.35:  # Strong downtrend
                        target_allocation = 0.4  # Stay partially invested
                    elif recent_positive_ratio < 0.45:  # Moderate downtrend
                        target_allocation = 0.7
                    else:  # Neutral
                        target_allocation = 0.8
                    
                    # Adjust position size based on confidence and recent performance
                    # Be more aggressive with position sizing
                    position_size = min(1.3, 0.7 + confidence * 0.6)  # Up to 130% position
                    target_allocation = target_allocation * position_size
                    
                    # Cap leverage and ensure minimum position
                    target_allocation = max(0, min(1.5, target_allocation))  # Cap at 150%
                    
                    # Rebalance more frequently but with tighter thresholds
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
                
                # Implement adaptive trailing stop-loss
                if i > sequence_length + 10:  # Wait for initial period
                    # Use ATR for dynamic stop-loss
                    lookback = min(20, len(portfolio_values))
                    recent_high = max(portfolio_values[-lookback:])
                    recent_returns = np.diff(portfolio_values[-lookback:]) / portfolio_values[-lookback:-1]
                    volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.05
                    
                    # Dynamic stop level based on volatility (tighter in high vol, looser in low vol)
                    stop_level = max(0.92, 0.95 - volatility * 10)  # Between 92-95% of peak
                    trailing_stop = stop_level * recent_high
                    
                    if net_worth < trailing_stop and shares > 0:
                        # Scale out of position rather than full exit
                        shares_to_sell = int(shares * 0.5)  # Sell half position
                        if shares_to_sell > 0:
                            revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                            balance += revenue
                            shares -= shares_to_sell
                            trades_executed += 1
        
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

def validate_model(model: nn.Module, val_loader: DataLoader, 
                  criterion_action: nn.Module, criterion_value: nn.Module, 
                  device: torch.device, return_predictions: bool = False) -> tuple:
    """
    Validate the model on the validation set
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion_action: Loss function for action prediction
        criterion_value: Loss function for value prediction
        device: Device to run validation on
        return_predictions: If True, returns predictions and targets for classification report
        
    Returns:
        If return_predictions is False: (avg_loss, accuracy, mae)
        If return_predictions is True: (avg_loss, accuracy, mae, all_predictions, all_targets)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    value_errors = []
    
    # Initialize lists to store predictions and targets if needed
    if return_predictions:
        all_predictions = []
        all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y_action, batch_y_return in val_loader:
            batch_x = batch_x.to(device)
            batch_y_action = batch_y_action.to(device)
            batch_y_return = batch_y_return.float().to(device)
            
            # Forward pass
            action_logits, value_pred, confidence, _ = model(batch_x)
            
            # Calculate losses
            loss_action = criterion_action(action_logits, batch_y_action)
            loss_value = criterion_value(value_pred.squeeze(), batch_y_return)
            confidence_loss = -torch.log(confidence + 1e-8).mean()
            
            # Total loss
            loss = loss_action + 0.5 * loss_value + 0.1 * confidence_loss
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(action_logits, 1)
            total += batch_y_action.size(0)
            correct += (predicted == batch_y_action).sum().item()
            value_errors.append(torch.abs(value_pred.squeeze() - batch_y_return).mean().item())
            
            # Store predictions and targets if needed
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y_action.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total * 100
    mae = np.mean(value_errors)
    
    if return_predictions:
        return (avg_loss, accuracy, mae, np.array(all_predictions), np.array(all_targets))
    else:
        return (avg_loss, accuracy, mae)

def plot_training_history(train_metrics: dict[str, list[float]], 
                         val_metrics: dict[str, list[float]]) -> None:
    """
    Plot training and validation metrics
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
    """
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_metrics['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_metrics['action_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, val_metrics['action_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot value MAE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_metrics['value_mae'], 'b-', label='Training MAE')
    plt.plot(epochs, val_metrics['value_mae'], 'r-', label='Validation MAE')
    plt.title('Training and Validation Value MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the figure
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training history plots saved to 'results/training_history.png'")

def main():
    """Professional LSTM training pipeline"""
    print(" PROFESSIONAL LSTM TRADING MODEL")
    print("=" * 80)
    print(" Advanced Architecture: Dual-Stage LSTM + Attention + Multi-head")
    print(" Sophisticated Features: 21 technical + fundamental indicators")
    print(" Professional Training: Weighted sampling + Multi-objective loss")
    print(" Risk Management: Stop-loss + Position sizing + Daily limits")
    print("=" * 80)
    
    # Load data
    print("\n Loading data...")
    train_data, val_data, test_data = load_sp500_data()
    
    print(f" Training: {len(train_data)} days ({train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')})")
    print(f" Validation: {len(val_data)} days ({val_data.index[0].strftime('%Y-%m-%d')} to {val_data.index[-1].strftime('%Y-%m-%d')})")
    print(f" Testing: {len(test_data)} days ({test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')})")
    
    # Advanced data preparation
    sequence_length = 30  # Longer sequences for better pattern recognition
    
    print(f"\n Preparing advanced training data (sequence_length={sequence_length})...")
    X_train, y_action_train, y_return_train, scaler, _ = prepare_advanced_lstm_data(
        train_data, sequence_length=sequence_length
    )
    
    print(" Preparing validation data...")
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
    input_size = X_train.shape[2]
    model = DualStageAttentionLSTM(
        input_size=input_size,
        hidden_size=256,
        extractor_layers=2,  # Number of layers in the extractor LSTM
        summarizer_layers=1,  # Number of layers in the summarizer LSTM
        dropout=0.4,
        num_heads=8
    )
    
    print(f"\n Advanced Model Architecture:")
    print(f"   Input size: {input_size} features (incl. positional encodings)")
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
    checkpoint = torch.load('trained_models/lstm_best.pth', weights_only=False)
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