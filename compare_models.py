#!/usr/bin/env python3
"""
COMPARISON FINAL - PPO vs LSTM vs Buy & Hold
Usa los modelos ya entrenados para comparación directa
"""

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import environments and models
from scripts.train_ppo_final import FinalTradingEnv
from scripts.train_lstm_ultra import UltraAggressiveLSTM, UltraAggressiveStrategy, prepare_ultra_data

def test_buy_hold(data):
    """Test simple buy and hold strategy"""
    initial_price = float(data.iloc[0]['Close'])
    final_price = float(data.iloc[-1]['Close'])
    return_pct = (final_price - initial_price) / initial_price * 100
    
    return {
        'return': return_pct,
        'trades': 0,
        'activity': 0.0,
        'actions': ['Hold'] * len(data),
        'portfolio_values': [1000000 * (1 + (float(data.iloc[i]['Close']) - initial_price) / initial_price) 
                           for i in range(len(data))]
    }

def test_ppo_model(data, model_path="trained_models/ppo_final_anti_hold_v1"):
    """Test trained PPO model"""
    print("🔥 Testing PPO Model...")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Error loading PPO model: {e}")
        return None
    
    env = FinalTradingEnv(data)
    obs, _ = env.reset()
    
    actions_taken = []
    portfolio_values = []
    trades_executed = 0
    
    action_names = ['Hold', 'Buy 10%', 'Buy 25%', 'Buy 50%', 'Sell 10%', 'Sell 25%', 'Sell 50%']
    
    for i in range(len(data) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        actions_taken.append(action_names[action])
        portfolio_values.append(info['net_worth'])
        
        if info['action_executed']:
            trades_executed += 1
        
        if done:
            break
    
    final_value = portfolio_values[-1]
    initial_value = 1000000
    return_pct = (final_value - initial_value) / initial_value * 100
    
    # Calculate activity
    hold_count = sum(1 for action in actions_taken if action == 'Hold')
    activity = (1 - hold_count / len(actions_taken)) * 100
    
    return {
        'return': return_pct,
        'trades': trades_executed,
        'activity': activity,
        'actions': actions_taken,
        'portfolio_values': portfolio_values
    }

def test_lstm_model(data, model_path="trained_models/lstm_ultra_aggressive_v1.pth"):
    """Test trained LSTM model"""
    print("💀 Testing LSTM Model...")
    
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        model = UltraAggressiveLSTM(input_size=6, hidden_size=64, num_layers=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")
        return None
    
    # Prepare data for LSTM
    X_data, y_data, scaler, df = prepare_ultra_data(data, sequence_length=20)
    X_tensor = torch.FloatTensor(X_data)
    
    trader = UltraAggressiveStrategy()
    actions_taken = []
    portfolio_values = []
    
    action_names = ['Hold', 'Buy 10%', 'Buy 25%', 'Buy 50%', 'Sell 10%', 'Sell 25%', 'Sell 50%']
    
    with torch.no_grad():
        for i in range(len(X_tensor)):
            current_price = float(y_data[i])
            price_pred, confidence, momentum = model(X_tensor[i:i+1])
            predicted_price = float(price_pred.squeeze())
            conf_score = float(confidence.squeeze())
            momentum_score = float(momentum.squeeze())
            
            action = trader.make_decision(current_price, predicted_price, conf_score, momentum_score)
            trader.execute_trade(action, current_price)
            
            actions_taken.append(action_names[action])
            portfolio_values.append(trader.get_portfolio_value(current_price))
    
    final_value = portfolio_values[-1]
    return_pct = (final_value - trader.initial_cash) / trader.initial_cash * 100
    
    # Calculate activity
    hold_count = sum(1 for action in actions_taken if action == 'Hold')
    activity = (1 - hold_count / len(actions_taken)) * 100
    
    return {
        'return': return_pct,
        'trades': trader.total_trades,
        'activity': activity,
        'actions': actions_taken,
        'portfolio_values': portfolio_values
    }

def create_comparison_plot(results, data, save_path="results/model_comparison_final.png"):
    """Create comparison visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🏆 PPO vs LSTM vs Buy & Hold - FINAL COMPARISON', fontsize=16, fontweight='bold')
    
    # Plot 1: Portfolio values over time
    ax1.plot(results['Buy & Hold']['portfolio_values'], label='Buy & Hold', color='blue', alpha=0.7)
    ax1.plot(results['PPO']['portfolio_values'], label='PPO Final', color='red', alpha=0.8)
    ax1.plot(results['LSTM']['portfolio_values'], label='LSTM Ultra', color='green', alpha=0.8)
    ax1.set_title('📈 Portfolio Value Over Time')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Returns comparison
    models = list(results.keys())
    returns = [results[model]['return'] for model in models]
    colors = ['blue', 'red', 'green']
    
    bars = ax2.bar(models, returns, color=colors, alpha=0.7)
    ax2.set_title('💰 Total Returns Comparison')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, return_val in zip(bars, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{return_val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Activity comparison
    activities = [results[model]['activity'] for model in models]
    bars = ax3.bar(models, activities, color=colors, alpha=0.7)
    ax3.set_title('🎯 Trading Activity Comparison')
    ax3.set_ylabel('Activity (%)')
    ax3.grid(True, alpha=0.3)
    
    for bar, activity in zip(bars, activities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{activity:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Trades executed
    trades = [results[model]['trades'] for model in models]
    bars = ax4.bar(models, trades, color=colors, alpha=0.7)
    ax4.set_title('🔄 Total Trades Executed')
    ax4.set_ylabel('Number of Trades')
    ax4.grid(True, alpha=0.3)
    
    for bar, trade_count in zip(bars, trades):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{trade_count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as {save_path}")
    
    return fig

def analyze_action_distribution(results):
    """Analyze action distribution for each model"""
    
    print(f"\n🎬 ACTION DISTRIBUTION ANALYSIS:")
    print("=" * 80)
    
    for model_name, result in results.items():
        actions = result['actions']
        
        if model_name == 'Buy & Hold':
            print(f"\n📈 {model_name}:")
            print(f"  Hold: 100.0% ({len(actions)} days)")
            continue
        
        # Count actions
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"\n🎯 {model_name}:")
        total_actions = len(actions)
        for action, count in sorted(action_counts.items()):
            percentage = count / total_actions * 100
            print(f"  {action}: {percentage:.1f}% ({count} times)")

def calculate_advanced_metrics(results):
    """Calculate advanced performance metrics"""
    
    print(f"\n📊 ADVANCED PERFORMANCE METRICS:")
    print("=" * 80)
    
    for model_name, result in results.items():
        portfolio_values = result['portfolio_values']
        
        # Convert to numpy array and calculate returns
        values = np.array(portfolio_values)
        daily_returns = np.diff(values) / values[:-1]
        
        # Calculate metrics
        annual_return = result['return']
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized volatility
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        print(f"\n🎯 {model_name}:")
        print(f"  📈 Annual Return: {annual_return:.2f}%")
        print(f"  📊 Volatility: {volatility:.2f}%")
        print(f"  ⚡ Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"  🔄 Total Trades: {result['trades']}")
        print(f"  🎯 Activity: {result['activity']:.1f}%")

def main():
    print("🏆 COMPARACIÓN FINAL: PPO vs LSTM vs Buy & Hold")
    print("=" * 80)
    
    # Load data
    print("📊 Loading market data...")
    data = yf.download('^GSPC', start='2020-01-01', end='2024-01-01', progress=False)
    data = data.dropna()
    
    # Split data (use test portion for fair comparison)
    split_point = int(len(data) * 0.8)
    test_data = data[split_point:].copy()
    
    print(f"📉 Test data: {len(test_data)} days")
    print(f"📅 Period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
    
    # Test all models
    results = {}
    
    # Buy & Hold
    print(f"\n📈 Testing Buy & Hold...")
    results['Buy & Hold'] = test_buy_hold(test_data)
    
    # PPO
    results['PPO'] = test_ppo_model(test_data)
    if results['PPO'] is None:
        print("❌ PPO testing failed")
        return
    
    # LSTM
    results['LSTM'] = test_lstm_model(test_data)
    if results['LSTM'] is None:
        print("❌ LSTM testing failed")
        return
    
    # Results summary
    print(f"\n🏆 FINAL RESULTS SUMMARY:")
    print("=" * 80)
    
    # Sort by returns
    sorted_results = sorted(results.items(), key=lambda x: x[1]['return'], reverse=True)
    
    print(f"{'Rank':<6} {'Model':<12} {'Return':<10} {'Activity':<10} {'Trades':<8} {'Status'}")
    print("-" * 60)
    
    for i, (model, result) in enumerate(sorted_results, 1):
        emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉"
        status = "WINNER" if i == 1 else "RUNNER-UP" if i == 2 else "THIRD"
        
        print(f"{emoji} {i:<3} {model:<12} {result['return']:>7.2f}% {result['activity']:>8.1f}% {result['trades']:>6} {status}")
    
    # Analysis
    analyze_action_distribution(results)
    calculate_advanced_metrics(results)
    
    # Determine champion
    print(f"\n🏆 CHAMPION ANALYSIS:")
    print("=" * 80)
    
    best_return = sorted_results[0]
    most_active = max(results.items(), key=lambda x: x[1]['activity'])
    most_trades = max(results.items(), key=lambda x: x[1]['trades'])
    
    print(f"🏆 Best Performance: {best_return[0]} ({best_return[1]['return']:.2f}%)")
    print(f"⚡ Most Active: {most_active[0]} ({most_active[1]['activity']:.1f}% activity)")
    print(f"🔄 Most Trades: {most_trades[0]} ({most_trades[1]['trades']} trades)")
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT:")
    print("=" * 80)
    
    if results['PPO']['activity'] > 50 and results['LSTM']['activity'] > 50:
        print("🎉 ¡BOTH MODELS SUCCEEDED!")
        print("✅ Hold Bias COMPLETELY ELIMINATED in both PPO and LSTM")
        print("✅ Both models now do ACTIVE TRADING instead of passive holding")
    elif results['PPO']['activity'] > 40 or results['LSTM']['activity'] > 40:
        print("🔥 SIGNIFICANT PROGRESS!")
        print("✅ Hold Bias significantly reduced")
        print("✅ Models now trade actively")
    
    # Create visualization
    print(f"\n📊 Creating comparison visualization...")
    create_comparison_plot(results, test_data)
    
    print(f"\n🎊 COMPARISON COMPLETE!")
    print("📈 Check 'results/model_comparison_final.png' for detailed visual analysis")

if __name__ == "__main__":
    main() 