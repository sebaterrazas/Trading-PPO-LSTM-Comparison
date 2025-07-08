#!/usr/bin/env python3
"""
MULTI-ASSET MODEL COMPARISON SYSTEM
- Compares multi-asset models trading BTC, S&P 500, and Treasury bonds
- Tests across all periods: Train (2010-2016), Val (2017-2018), Test (2019-2024)
- Shows portfolio allocation analysis
- Key metrics: Total Return, Annual Return, Sharpe Ratio, Max Drawdown, Volatility
"""

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import from multi-asset system
from ppo_trading_v2 import (
    SimpleMultiAssetEnv, 
    load_simple_data, 
    test_simple_model, 
    ASSETS, 
    ASSET_NAMES,
    get_device
)

def create_buy_hold_portfolios(data_dict):
    """Create different buy & hold strategies for comparison"""
    portfolios = {}
    
    # Strategy 1: S&P 500 only
    portfolios['S&P 500 Only'] = {
        'BTC': 0.0,
        'SP500': 1.0,
        'BONDS': 0.0
    }
    
    # # Strategy 2: Equal Weight
    # portfolios['Equal Weight'] = {
    #     'BTC': 0.333,
    #     'SP500': 0.333,
    #     'BONDS': 0.334
    # }
    
    # # Strategy 3: Conservative
    # portfolios['Conservative'] = {
    #     'BTC': 0.1,
    #     'SP500': 0.4,
    #     'BONDS': 0.5
    # }
    
    # # Strategy 4: Aggressive
    # portfolios['Aggressive'] = {
    #     'BTC': 0.3,
    #     'SP500': 0.7,
    #     'BONDS': 0.0
    # }

    # # Strategy 5: Bitcoin Only  
    # portfolios['Bitcoin Only'] = {
    #     'BTC': 1.0,
    #     'SP500': 0.0,
    #     'BONDS': 0.0
    # }
    
    return portfolios

def test_buy_hold_strategy(data_dict, allocation, name="Buy & Hold", 
                          initial_balance=1000000, transaction_cost=0.002):
    """Test a buy & hold strategy with given allocation"""
    print(f"📊 Testing {name}...")
    
    # Get aligned data
    common_dates = None
    for asset_data in data_dict.values():
        if common_dates is None:
            common_dates = asset_data.index
        else:
            common_dates = common_dates.intersection(asset_data.index)
    
    if len(common_dates) == 0:
        print(f"❌ No common dates found for {name}")
        return None
    
    # Initial allocation
    portfolio_values = [initial_balance]
    shares = {}
    remaining_cash = initial_balance
    
    # Buy initial shares
    for asset, weight in allocation.items():
        if weight > 0:
            allocation_amount = initial_balance * weight
            initial_price = data_dict[asset].loc[common_dates[0], 'Close']
            shares[asset] = int(allocation_amount / (initial_price * (1 + transaction_cost)))
            cost = shares[asset] * initial_price * (1 + transaction_cost)
            remaining_cash -= cost
        else:
            shares[asset] = 0
    
    # Calculate portfolio value over time
    for date in common_dates[1:]:
        total_value = remaining_cash
        for asset, num_shares in shares.items():
            if num_shares > 0:
                current_price = data_dict[asset].loc[date, 'Close']
                total_value += num_shares * current_price
        portfolio_values.append(total_value)
    
    # Calculate metrics
    returns = np.array(portfolio_values)
    daily_returns = np.diff(returns) / returns[:-1]
    
    total_return = (returns[-1] - returns[0]) / returns[0] * 100
    annual_return = (((returns[-1] / returns[0]) ** (252 / len(returns))) - 1) * 100
    
    volatility = np.std(daily_returns) * np.sqrt(252) * 100
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    max_drawdown = np.min((returns - np.maximum.accumulate(returns)) / np.maximum.accumulate(returns)) * 100
    
    # Calculate final allocations
    final_allocations = {}
    total_value = returns[-1]
    cash_percentage = (remaining_cash / total_value) * 100
    
    for asset in ASSET_NAMES:
        if shares[asset] > 0:
            final_date = common_dates[-1]
            final_price = data_dict[asset].loc[final_date, 'Close']
            asset_value = shares[asset] * final_price
            final_allocations[asset] = (asset_value / total_value) * 100
        else:
            final_allocations[asset] = 0.0
    
    return {
        'return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'portfolio_values': portfolio_values,
        'final_allocations': final_allocations,
        'final_cash': cash_percentage,
        'avg_allocations': allocation  # For buy & hold, avg = final (approximately)
    }

def test_multi_asset_model(data_dict, model_path, model_name="Multi-Asset Model"):
    """Test a multi-asset model (PPO or LSTM)"""
    print(f"🚀 Testing {model_name}...")
    
    try:
        # Check if it's an LSTM model
        if model_path.endswith('.pth'):
            return test_lstm_model_comparison(data_dict, model_path, model_name)
        else:
            return test_ppo_model_comparison(data_dict, model_path, model_name)
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None

def test_ppo_model_comparison(data_dict, model_path, model_name):
    """Test a PPO model"""
    # Load model (handles .zip format automatically)
    model = PPO.load(model_path)
    
    # Check if it's a Bitcoin-focused model (7 actions)
    if 'bitcoin_focused' in model_path.lower():
        # Import BitcoinFocusedEnv
        try:
            from ppo_bitcoin_focused import BitcoinFocusedEnv
            env = BitcoinFocusedEnv(data_dict)
        except ImportError:
            print(f"⚠️ Bitcoin-focused model detected but BitcoinFocusedEnv not available")
            # Fallback to regular environment
            env = SimpleMultiAssetEnv(data_dict)
    else:
        # Create regular environment
        env = SimpleMultiAssetEnv(data_dict)
    
    # Test model with compatible environment
    if 'bitcoin_focused' in model_path.lower():
        results = test_bitcoin_focused_model(model, env)
    else:
        results = test_simple_model(model, env)
    
    # Add portfolio values for plotting if not already included
    if 'portfolio_values' not in results:
        portfolio_values = [env.initial_balance]
        obs, _ = env.reset()
        
        for _ in range(env.data_length - 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info['net_worth'])
            if done:
                break
        
        results['portfolio_values'] = portfolio_values
    
    return results

def test_bitcoin_focused_model(model, env):
    """Test Bitcoin-focused model with 7 actions"""
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
        'portfolio_allocations': portfolio_allocations,
        'portfolio_values': portfolio_values
    }

def test_lstm_model_comparison(data_dict, model_path, model_name):
    """Test an LSTM model"""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    
    # Define LSTM model class (UPDATED to match improved lstm_trading_v2.py)
    class LSTMTradingModel(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=1, num_assets=3, dropout=0.5):
            super(LSTMTradingModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_assets = num_assets
            
            # LSTM layers (SMALLER for less overfitting)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0,
                bidirectional=False  # Simpler model
            )
            
            # Fully connected layers with MORE regularization
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(hidden_size // 4, num_assets * 5)  # 5 actions per asset
            
            # Activation functions
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            
        def forward(self, x):
            # x shape: (batch_size, sequence_length, input_size)
            
            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use the last output
            last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
            
            # Fully connected layers with multiple dropout layers
            x = self.relu(self.fc1(last_output))
            x = self.dropout1(x)
            x = self.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            
            # Reshape to (batch_size, num_assets, num_actions)
            x = x.view(-1, self.num_assets, 5)
            
            # Don't apply softmax here - let CrossEntropyLoss handle it
            return x
    
    # Load model and scaler
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create environment to get input size
    env = SimpleMultiAssetEnv(data_dict)
    input_size = len(env._get_observation())
    
    # Create and load model
    model = LSTMTradingModel(input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get scaler
    scaler = checkpoint['scaler']
    
    # Test LSTM model
    obs, _ = env.reset()
    portfolio_values = [env.initial_balance]
    actions_taken = {asset: [] for asset in ASSET_NAMES}
    portfolio_allocations = {asset: [] for asset in ASSET_NAMES}
    cash_allocations = []
    
    # Get all observations first
    all_obs = env.get_all_observations()
    all_obs_scaled = scaler.transform(all_obs)
    
    sequence_length = 10
    
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
        'portfolio_allocations': portfolio_allocations,
        'portfolio_values': portfolio_values
    }

def compare_across_periods(models_to_test):
    """Compare models across all three periods"""
    print("🏆 MULTI-PERIOD COMPARISON")
    print("=" * 80)
    
    # Load data for all periods
    result = load_simple_data()
    if result is None:
        print("❌ Failed to load data")
        return
    
    train_data, val_data, test_data = result
    periods = {
        'Train (2010-2016)': train_data,
        'Validation (2017-2018)': val_data,
        'Test (2019-2024)': test_data
    }
    
    all_results = {}
    
    for period_name, period_data in periods.items():
        print(f"\n📊 {period_name}")
        print("-" * 50)
        
        period_results = {}
        
        # Test buy & hold strategies
        buy_hold_strategies = create_buy_hold_portfolios(period_data)
        for strategy_name, allocation in buy_hold_strategies.items():
            result = test_buy_hold_strategy(period_data, allocation, strategy_name)
            if result:
                period_results[strategy_name] = result
        
        # Test PPO models
        for model_name, model_path in models_to_test.items():
            result = test_multi_asset_model(period_data, model_path, model_name)
            if result:
                period_results[model_name] = result
        
        all_results[period_name] = period_results
        
        # Print results for this period
        print_period_results(period_results, period_name)
    
    return all_results

def print_period_results(results, period_name):
    """Print results for a specific period"""
    print(f"\n📈 {period_name} RESULTS")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Total Ret':<10} {'Annual Ret':<11} {'Sharpe':<8} {'Max DD':<8} {'Vol':<8}")
    print("-" * 60)
    
    # Sort by Sharpe ratio
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    for strategy, result in sorted_results:
        print(f"{strategy:<20} {result['return']:>7.1f}% {result['annual_return']:>8.1f}% "
              f"{result['sharpe']:>6.2f} {result['max_drawdown']:>6.1f}% {result['volatility']:>6.1f}%")

def create_comprehensive_visualization(all_results, save_path="results/multi_asset_comparison.png"):
    """Create comprehensive visualization of all results - Academic Paper Style"""
    
    # Create figure with subplots (expanded layout)
    fig = plt.figure(figsize=(18, 14))
    
    # 1. MAIN PLOT: Continuous Performance (Academic Style)
    ax1 = plt.subplot(3, 3, (1, 3))
    
    # Extract strategies (models only, not buy & hold)
    ppo_strategies = [s for s in all_results.get('Test (2019-2024)', {}).keys() if 'PPO' in s or 'LSTM' in s]
    benchmark_strategies = [s for s in all_results.get('Test (2019-2024)', {}).keys() if 'PPO' not in s and 'LSTM' not in s]
    
    # Colors for different strategy types
    ppo_colors = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22']  # Red, Blue, Purple, Orange
    benchmark_colors = ['#95a5a6', '#7f8c8d', '#34495e', '#2c3e50']  # Grays
    
    # Combine portfolio values across all periods for each strategy
    def get_continuous_performance(strategy_name):
        """Get continuous portfolio values across all periods"""
        combined_values = []
        period_boundaries = []
        total_days = 0
        
        for period_name in ['Train (2010-2016)', 'Validation (2017-2018)', 'Test (2019-2024)']:
            if period_name in all_results and strategy_name in all_results[period_name]:
                values = all_results[period_name][strategy_name]['portfolio_values']
                if not combined_values:  # First period
                    combined_values.extend(values)
                else:  # Subsequent periods - continue from last value
                    # Normalize to continue from previous end
                    scale_factor = combined_values[-1] / values[0]
                    scaled_values = [v * scale_factor for v in values[1:]]  # Skip first (duplicate)
                    combined_values.extend(scaled_values)
                
                period_boundaries.append(total_days)
                total_days += len(values) - (1 if combined_values else 0)
        
        return combined_values, period_boundaries[1:]  # Skip first boundary (0)
    
    # Plot PPO models
    for i, strategy in enumerate(ppo_strategies):
        values, boundaries = get_continuous_performance(strategy)
        if values:
            # Normalize to percentage returns
            normalized = [(v / values[0] - 1) * 100 for v in values]
            ax1.plot(normalized, label=strategy, color=ppo_colors[i % len(ppo_colors)], 
                    linewidth=2.5, alpha=0.9)
    
    # Plot ALL buy & hold benchmarks with dashed lines
    benchmark_colors = ['#2c3e50', '#7f8c8d', '#34495e', '#95a5a6']  # Dark colors for benchmarks
    benchmark_styles = ['--', '-.', ':', (0, (3, 1, 1, 1))]  # Different line styles
    
    for i, benchmark in enumerate(benchmark_strategies):
        values, boundaries = get_continuous_performance(benchmark)
        if values:
            normalized = [(v / values[0] - 1) * 100 for v in values]
            ax1.plot(normalized, label=f'{benchmark} (Buy & Hold)', 
                    color=benchmark_colors[i % len(benchmark_colors)], 
                    linewidth=2, linestyle=benchmark_styles[i % len(benchmark_styles)], 
                    alpha=0.8)
    
    # Add period dividers (Academic Standard)
    if ppo_strategies:  # Get boundaries from any strategy
        _, boundaries = get_continuous_performance(ppo_strategies[0])
        for boundary in boundaries:
            ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        
        # Add period labels
        ax1.text(boundaries[0]/3, ax1.get_ylim()[1]*0.9, 'TRAIN\n(2010-2016)', 
                ha='center', va='top', fontsize=10, alpha=0.7, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
        ax1.text((boundaries[0] + boundaries[1])/2, ax1.get_ylim()[1]*0.9, 'VALIDATION\n(2017-2018)', 
                ha='center', va='top', fontsize=10, alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
        ax1.text((boundaries[1] + len(normalized))/2, ax1.get_ylim()[1]*0.9, 'TEST\n(2019-2024)', 
                ha='center', va='top', fontsize=10, alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
    
    ax1.set_title('📈 Cumulative Portfolio Performance (Academic Style)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Days (Continuous Timeline)')
    ax1.set_ylabel('Cumulative Return (%)')
    
    # Create custom legend with line style distinction
    legend_elements = []
    # Add ML models first
    for i, strategy in enumerate(ppo_strategies):
        model_type = "PPO" if "PPO" in strategy else "LSTM"
        legend_elements.append(plt.Line2D([0], [0], color=ppo_colors[i % len(ppo_colors)], 
                                        linewidth=2.5, label=f'{strategy} ({model_type})', alpha=0.9))
    # Add Buy & Hold strategies
    for i, strategy in enumerate(benchmark_strategies):
        legend_elements.append(plt.Line2D([0], [0], color=benchmark_colors[i % len(benchmark_colors)], 
                                        linewidth=2, linestyle=benchmark_styles[i % len(benchmark_styles)], 
                                        label=f'{strategy} (Buy & Hold)', alpha=0.8))
    
    ax1.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # 2. NEW: TEST SET ONLY PERFORMANCE
    ax2 = plt.subplot(3, 3, (4, 6))  # Middle row, all columns
    
    # Plot only TEST SET results with all models starting from 0%
    test_results = all_results.get('Test (2019-2024)', {})
    
    if test_results:
        # Colors for different strategy types
        ppo_colors = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22']
        benchmark_colors = ['#2c3e50', '#7f8c8d', '#34495e', '#95a5a6']
        benchmark_styles = ['--', '-.', ':', (0, (3, 1, 1, 1))]
        
        ppo_count = 0
        benchmark_count = 0
        
        for strategy, result in test_results.items():
            if 'portfolio_values' in result:
                values = result['portfolio_values']
                # Normalize so all start at 0%
                normalized = [(v / values[0] - 1) * 100 for v in values]
                
                # Choose color and style based on strategy type
                if 'PPO' in strategy or 'LSTM' in strategy:
                    color = ppo_colors[ppo_count % len(ppo_colors)]
                    linestyle = '-'
                    linewidth = 2.5
                    alpha = 0.9
                    label = strategy
                    ppo_count += 1
                else:
                    color = benchmark_colors[benchmark_count % len(benchmark_colors)]
                    linestyle = benchmark_styles[benchmark_count % len(benchmark_styles)]
                    linewidth = 2
                    alpha = 0.8
                    label = f'{strategy} (Buy & Hold)'
                    benchmark_count += 1
                
                ax2.plot(normalized, label=label, color=color, 
                        linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    
    ax2.set_title('🧪 Test Set Performance (2019-2024) - All Start Equal', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Add zero line for reference
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # 3. Risk-Return Scatter (Test Period Only - Academic Standard)
    ax3 = plt.subplot(3, 3, 7)
    test_results = all_results.get('Test (2019-2024)', {})
    
    # Define colors and markers for different strategy types
    ppo_colors = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22']  # PPO models
    benchmark_colors = ['#2c3e50', '#7f8c8d', '#34495e', '#95a5a6']  # Buy & Hold
    
    ppo_count = 0
    benchmark_count = 0
    
    for strategy, result in test_results.items():
        if 'PPO' in strategy or 'LSTM' in strategy:
            color = ppo_colors[ppo_count % len(ppo_colors)]
            marker = 'o'
            size = 140
            ppo_count += 1
        else:
            color = benchmark_colors[benchmark_count % len(benchmark_colors)]
            marker = 's'
            size = 100
            benchmark_count += 1
            
        ax3.scatter(result['volatility'], result['annual_return'], 
                   c=color, s=size, alpha=0.8, marker=marker, edgecolors='white', linewidth=1.5)
        ax3.annotate(strategy, (result['volatility'], result['annual_return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax3.set_title('🎯 Risk-Return Profile (Test Period)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Volatility (%)')
    ax3.set_ylabel('Annual Return (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add efficient frontier reference
    ax3.plot([10, 50], [5, 50], 'k--', alpha=0.3, linewidth=1, label='Reference Line')
    
    # Add legend to distinguish ML Models vs Buy & Hold
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', 
                   markersize=10, label='ML Models (PPO/LSTM)', markeredgecolor='white', markeredgewidth=1.5),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f8c8d', 
                   markersize=8, label='Buy & Hold', markeredgecolor='white', markeredgewidth=1.5)
    ]
    ax3.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # 4. Performance Metrics Comparison (Test Period)
    ax4 = plt.subplot(3, 3, 8)
    
    # Focus only on ML models for cleaner comparison
    ppo_results = {k: v for k, v in test_results.items() if 'PPO' in k or 'LSTM' in k}
    
    metrics = ['Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    x_pos = np.arange(len(ppo_results))
    width = 0.25
    
    # Normalize data for comparison
    annual_returns = [result['annual_return'] for result in ppo_results.values()]
    sharpe_ratios = [result['sharpe'] * 10 for result in ppo_results.values()]  # Scale for visibility
    max_drawdowns = [abs(result['max_drawdown']) for result in ppo_results.values()]
    
    ax4.bar(x_pos - width, annual_returns, width, label='Annual Return', color='#3498db', alpha=0.8)
    ax4.bar(x_pos, sharpe_ratios, width, label='Sharpe Ratio (×10)', color='#e74c3c', alpha=0.8)
    ax4.bar(x_pos + width, max_drawdowns, width, label='Max Drawdown', color='#f39c12', alpha=0.8)
    
    ax4.set_title('📊 Performance Metrics (ML Models)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(list(ppo_results.keys()), rotation=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Portfolio Allocation Analysis (PPO models only)
    ax5 = plt.subplot(3, 3, 9)
    
    if ppo_results:
        # Create stacked bar chart of allocations
        bottom = np.zeros(len(ppo_results))
        asset_colors = {'BTC': '#f39c12', 'SP500': '#3498db', 'BONDS': '#27ae60', 'CASH': '#95a5a6'}
        
        for asset in ASSET_NAMES + ['CASH']:
            values = []
            for model_name, result in ppo_results.items():
                if asset == 'CASH':
                    values.append(result.get('avg_cash', 0))
                else:
                    values.append(result.get('avg_allocations', {}).get(asset, 0))
            
            ax5.bar(range(len(ppo_results)), values, bottom=bottom, label=asset, 
                   color=asset_colors.get(asset, 'gray'), alpha=0.8)
            bottom += values
        
        ax5.set_title('💼 Portfolio Allocation (Test Period)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('ML Models')
        ax5.set_ylabel('Allocation (%)')
        ax5.set_xticks(range(len(ppo_results)))
        ax5.set_xticklabels(list(ppo_results.keys()), rotation=15)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive visualization saved as {save_path}")

def create_summary_table(all_results):
    """Create a summary table of the best performing strategies"""
    print("\n🏆 PERFORMANCE SUMMARY TABLE")
    print("=" * 120)
    
    test_results = all_results.get('Test (2019-2024)', {})
    
    if not test_results:
        print("❌ No test results available")
        return
    
    # Create summary DataFrame
    summary_data = []
    
    for strategy, result in test_results.items():
        summary_data.append({
            'Strategy': strategy,
            'Total Return (%)': result['return'],
            'Annual Return (%)': result['annual_return'],
            'Sharpe Ratio': result['sharpe'],
            'Max Drawdown (%)': result['max_drawdown'],
            'Volatility (%)': result['volatility']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Sharpe Ratio', ascending=False)
    
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Highlight best performers
    best_return = df.loc[df['Total Return (%)'].idxmax()]
    best_sharpe = df.loc[df['Sharpe Ratio'].idxmax()]
    best_drawdown = df.loc[df['Max Drawdown (%)'].idxmax()]  # Least negative
    
    print(f"\n🏅 BEST PERFORMERS (Test Period 2019-2024):")
    print(f"📈 Best Total Return: {best_return['Strategy']} ({best_return['Total Return (%)']:.1f}%)")
    print(f"⚡ Best Sharpe Ratio: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.2f})")
    print(f"🛡️ Best Max Drawdown: {best_drawdown['Strategy']} ({best_drawdown['Max Drawdown (%)']:.1f}%)")

def main():
    """Main comparison function"""
    print("🏆 MULTI-ASSET MODEL COMPARISON SYSTEM")
    print("=" * 80)
    print("✅ Multi-asset trading: BTC, S&P 500, Treasury Bonds")
    print("✅ Tests across all periods: Train/Val/Test")
    print("✅ Comprehensive portfolio allocation analysis")
    print("✅ Key metrics: Total Return, Annual Return, Sharpe, Max Drawdown, Volatility")
    print("=" * 80)
    
    # Define models to test
    models_to_test = {
        # 'PPO-T 30k': 'trained_models/multi_asset_transformer_ppo',
        # 'PPO Bitcoin': 'trained_models/bitcoin_focused_ppo_30k',
        # 'LSTM': 'trained_models/lstm_multi_asset.pth',
        'PPO 100k': 'trained_models/ppo_100k_training_steps',
        # 'PPO 30k': 'trained_models/multi_asset_ppo_same_dates',
    }
    
    # Run comparison
    all_results = compare_across_periods(models_to_test)
    
    if not all_results:
        print("❌ No results to compare")
        return
    
    # Create visualizations
    create_comprehensive_visualization(all_results)
    
    # Create summary table
    create_summary_table(all_results)
    
    print(f"\n🎊 MULTI-ASSET COMPARISON COMPLETE!")
    print(f"📊 Results saved to 'results/multi_asset_comparison.png'")
    print(f"🎯 All models tested across Train (2010-2016), Val (2017-2018), Test (2019-2024)")

if __name__ == "__main__":
    main() 