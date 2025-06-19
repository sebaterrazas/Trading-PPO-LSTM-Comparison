#!/usr/bin/env python3
"""
PPO FINAL - Solución DEFINITIVA al problema Buy & Hold Bias
Basado en research de trading RL para forzar trading activo
"""

import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

class FinalTradingEnv(gym.Env):
    def __init__(self, df, initial_amount=1000000):
        super(FinalTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_amount = initial_amount
        
        # SOLUTION 1: Más acciones granulares
        # 0=hold, 1=buy_10%, 2=buy_25%, 3=buy_50%, 4=sell_10%, 5=sell_25%, 6=sell_50%
        self.action_space = spaces.Discrete(7)
        
        # SOLUTION 2: State space más rico para mejor exploración
        # [price_norm, returns, volatility, rsi, momentum, cash_ratio, stock_ratio, 
        #  portfolio_return, days_since_trade, trade_penalty_score]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_amount
        self.shares_held = 0
        self.net_worth = self.initial_amount
        self.max_net_worth = self.initial_amount
        
        # SOLUTION 3: Anti-inactivity tracking
        self.days_since_trade = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.trade_penalty_accumulator = 0
        
        # Historia para calcular momentum y volatility
        self.price_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        current_price = float(self.df.loc[self.current_step, 'Close'])
        
        # Construir historia de precios
        start_idx = max(0, self.current_step - 10)
        recent_prices = self.df.loc[start_idx:self.current_step, 'Close'].values
        self.price_history = [float(x) for x in recent_prices]
        
        # 1. Precio normalizado
        price_norm = (current_price - self.df['Close'].mean()) / self.df['Close'].std()
        
        # 2. Returns recientes
        if len(self.price_history) >= 2:
            returns = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
        else:
            returns = 0
        
        # 3. Volatilidad reciente (rolling 5 days)
        if len(self.price_history) >= 5:
            recent_returns = [(self.price_history[i] - self.price_history[i-1])/self.price_history[i-1] 
                            for i in range(1, len(self.price_history))]
            volatility = np.std(recent_returns)
        else:
            volatility = 0
        
        # 4. RSI-like momentum (simplified)
        if len(self.price_history) >= 10:
            gains = [max(0, self.price_history[i] - self.price_history[i-1]) 
                    for i in range(1, len(self.price_history))]
            losses = [max(0, self.price_history[i-1] - self.price_history[i]) 
                     for i in range(1, len(self.price_history))]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            momentum = (rsi - 50) / 50  # Normalize to [-1, 1]
        else:
            momentum = 0
        
        # 5. Portfolio ratios
        total_value = self.balance + (self.shares_held * current_price)
        cash_ratio = self.balance / total_value
        stock_ratio = (self.shares_held * current_price) / total_value
        
        # 6. Portfolio performance
        portfolio_return = (total_value - self.initial_amount) / self.initial_amount
        
        # 7. Days since last trade (SOLUTION 3: Track inactivity)
        days_since_trade_norm = min(self.days_since_trade / 20, 1.0)  # Cap at 20 days
        
        # 8. Trade penalty score (accumulates with inactivity)
        trade_penalty_score = min(self.trade_penalty_accumulator / 10, 1.0)
        
        observation = np.array([
            float(price_norm),
            float(returns),
            float(volatility),
            float(momentum),
            float(cash_ratio),
            float(stock_ratio),
            float(portfolio_return),
            float(days_since_trade_norm),
            float(trade_penalty_score),
            float(current_price / 1000)  # Scale price for NN
        ], dtype=np.float32)
        
        # Clip to bounds
        observation = np.clip(observation, -10, 10)
        
        return observation
    
    def step(self, action):
        current_price = float(self.df.loc[self.current_step, 'Close'])
        prev_net_worth = self.net_worth
        reward = 0
        
        # Calculate quantities for each action
        total_value = self.balance + (self.shares_held * current_price)
        action_executed = False
        
        # SOLUTION 1: Granular actions
        if action == 0:  # HOLD
            # Do nothing
            self.days_since_trade += 1
            # SOLUTION 3: Exponentially increasing penalty for inactivity
            inactivity_penalty = -0.001 * (1.1 ** self.days_since_trade)
            reward += inactivity_penalty
            self.trade_penalty_accumulator += 0.1
            
        elif action == 1:  # BUY 10%
            buy_amount = total_value * 0.10
            if self.balance >= buy_amount:
                shares_to_buy = buy_amount / current_price
                self.shares_held += shares_to_buy
                self.balance -= buy_amount
                action_executed = True
                
        elif action == 2:  # BUY 25%
            buy_amount = total_value * 0.25
            if self.balance >= buy_amount:
                shares_to_buy = buy_amount / current_price
                self.shares_held += shares_to_buy
                self.balance -= buy_amount
                action_executed = True
                
        elif action == 3:  # BUY 50%
            buy_amount = total_value * 0.50
            if self.balance >= buy_amount:
                shares_to_buy = buy_amount / current_price
                self.shares_held += shares_to_buy
                self.balance -= buy_amount
                action_executed = True
                
        elif action == 4:  # SELL 10%
            shares_to_sell = self.shares_held * 0.10
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                action_executed = True
                
        elif action == 5:  # SELL 25%
            shares_to_sell = self.shares_held * 0.25
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                action_executed = True
                
        elif action == 6:  # SELL 50%
            shares_to_sell = self.shares_held * 0.50
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                action_executed = True
        
        # Update net worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # SOLUTION 2: Sophisticated reward function
        if action_executed:
            self.total_trades += 1
            self.days_since_trade = 0
            self.trade_penalty_accumulator *= 0.5  # Reduce penalty after trading
            
            # Base reward: Portfolio return
            portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth
            reward += portfolio_return * 100  # Scale up
            
            # SOLUTION 3: Bonus for trading activity
            activity_bonus = 0.01  # Small bonus for any trade
            reward += activity_bonus
            
            # SOLUTION 3: Bonus for good diversification (having both cash and stocks)
            cash_ratio = self.balance / self.net_worth
            if 0.1 <= cash_ratio <= 0.9:  # Good diversification
                diversification_bonus = 0.005
                reward += diversification_bonus
            
            # SOLUTION 3: Extra bonus for contrarian moves (buy low, sell high indicators)
            if len(self.price_history) >= 5:
                recent_trend = (current_price - self.price_history[-5]) / self.price_history[-5]
                if action in [1, 2, 3] and recent_trend < -0.02:  # Buying after a dip
                    contrarian_bonus = 0.01
                    reward += contrarian_bonus
                elif action in [4, 5, 6] and recent_trend > 0.02:  # Selling after a rise
                    contrarian_bonus = 0.01
                    reward += contrarian_bonus
            
            # Track profitable trades
            if (self.net_worth - prev_net_worth) > 0:
                self.profitable_trades += 1
        
        # SOLUTION 3: Alpha bonus - beat market performance
        if self.current_step > 0:
            # Simple market return (buy and hold)
            market_start = float(self.df.loc[0, 'Close'])
            market_current = current_price
            market_return = (market_current - market_start) / market_start
            
            agent_return = (self.net_worth - self.initial_amount) / self.initial_amount
            alpha = agent_return - market_return
            
            if alpha > 0:
                alpha_bonus = alpha * 0.1  # 10% of the alpha as bonus
                reward += alpha_bonus
        
        # SOLUTION 3: Progressive trading target
        # Expect at least 1 trade per 10 days
        expected_trades = max(1, self.current_step // 10)
        if self.total_trades >= expected_trades:
            activity_target_bonus = 0.001
            reward += activity_target_bonus
        
        # Update max net worth
        if float(self.net_worth) > float(self.max_net_worth):
            self.max_net_worth = float(self.net_worth)
        
        # Check if done
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Additional info
        info = {
            'net_worth': self.net_worth,
            'total_trades': self.total_trades,
            'days_since_trade': self.days_since_trade,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'action_executed': action_executed
        }
        
        return self._get_observation(), reward, done, False, info

def train_final_ppo():
    print("🔥 ENTRENANDO PPO FINAL - SOLUCIÓN DEFINITIVA")
    print("=" * 60)
    
    # Descargar datos
    print("📊 Descargando datos...")
    data = yf.download('^GSPC', start='2020-01-01', end='2024-01-01', progress=False)
    data = data.dropna()
    
    # Split data
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point].copy()
    test_data = data[split_point:].copy()
    
    print(f"📈 Datos de entrenamiento: {len(train_data)} días")
    print(f"📉 Datos de prueba: {len(test_data)} días")
    
    # Create environment
    train_env = FinalTradingEnv(train_data)
    
    print("\n🚀 Configuración PPO FINAL:")
    print("- 7 acciones granulares (Hold, Buy 10%/25%/50%, Sell 10%/25%/50%)")
    print("- Penalizaciones exponenciales por inactividad")
    print("- Bonuses por diversificación y trading contrarian")
    print("- Alpha tracking vs market")
    print("- 100,000 timesteps de entrenamiento intensivo")
    
    # SOLUTION 4: Hyperparameters optimized for trading
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=0.0005,  # Higher LR for more aggressive learning
        n_steps=2048,          # Larger batch for more stable learning
        batch_size=128,        # Bigger batches
        n_epochs=15,           # More epochs per update
        gamma=0.95,            # Lower gamma for more immediate rewards
        gae_lambda=0.9,        # GAE for better advantage estimation
        clip_range=0.25,       # Slightly higher clip range
        ent_coef=0.05,         # High entropy for exploration
        vf_coef=0.5,           # Balance value function
        max_grad_norm=0.5,     # Gradient clipping
        verbose=1,
        device='auto'
    )
    
    print("\n🔥 INICIO DEL ENTRENAMIENTO...")
    print("Este proceso puede tomar varios minutos...\n")
    
    # Train the model with more timesteps
    model.learn(total_timesteps=100000)
    
    print("\n💾 Guardando modelo...")
    model.save("ppo_final_anti_hold_v1")
    
    print("\n🧪 PROBANDO EN DATOS DE ENTRENAMIENTO...")
    # Test on training data
    obs, _ = train_env.reset()
    total_trades = 0
    actions_taken = []
    net_worths = []
    
    for i in range(len(train_data) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = train_env.step(action)
        
        actions_taken.append(action)
        net_worths.append(info['net_worth'])
        
        if info['action_executed']:
            total_trades += 1
        
        if done:
            break
    
    final_value = net_worths[-1]
    initial_value = 1000000
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate buy and hold return
    buy_hold_return = (float(train_data.iloc[-1]['Close']) - float(train_data.iloc[0]['Close'])) / float(train_data.iloc[0]['Close']) * 100
    
    print(f"\n📊 RESULTADOS ENTRENAMIENTO:")
    print(f"💰 Return PPO Final: {total_return:.2f}%")
    print(f"📈 Return Buy & Hold: {buy_hold_return:.2f}%")
    print(f"🎯 Alpha (diferencia): {total_return - buy_hold_return:.2f}%")
    print(f"🔄 Total trades ejecutados: {total_trades}")
    
    # Action distribution
    unique, counts = np.unique(actions_taken, return_counts=True)
    action_names = ['Hold', 'Buy 10%', 'Buy 25%', 'Buy 50%', 'Sell 10%', 'Sell 25%', 'Sell 50%']
    
    print(f"\n🎬 DISTRIBUCIÓN DE ACCIONES:")
    for action_idx, count in zip(unique, counts):
        percentage = count / len(actions_taken) * 100
        print(f"  {action_names[action_idx]}: {percentage:.1f}% ({count} veces)")
    
    print("\n🧪 PROBANDO EN DATOS DE PRUEBA...")
    # Test on test data
    test_env = FinalTradingEnv(test_data)
    obs, _ = test_env.reset()
    test_trades = 0
    test_actions = []
    test_net_worths = []
    
    for i in range(len(test_data) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        
        test_actions.append(action)
        test_net_worths.append(info['net_worth'])
        
        if info['action_executed']:
            test_trades += 1
        
        if done:
            break
    
    test_final_value = test_net_worths[-1]
    test_return = (test_final_value - initial_value) / initial_value * 100
    test_buy_hold = (float(test_data.iloc[-1]['Close']) - float(test_data.iloc[0]['Close'])) / float(test_data.iloc[0]['Close']) * 100
    
    print(f"\n📊 RESULTADOS PRUEBA (DATOS NO VISTOS):")
    print(f"💰 Return PPO Final: {test_return:.2f}%")
    print(f"📈 Return Buy & Hold: {test_buy_hold:.2f}%")
    print(f"🎯 Alpha (diferencia): {test_return - test_buy_hold:.2f}%")
    print(f"🔄 Total trades ejecutados: {test_trades}")
    
    # Test action distribution
    test_unique, test_counts = np.unique(test_actions, return_counts=True)
    print(f"\n🎬 DISTRIBUCIÓN DE ACCIONES (PRUEBA):")
    for action_idx, count in zip(test_unique, test_counts):
        percentage = count / len(test_actions) * 100
        print(f"  {action_names[action_idx]}: {percentage:.1f}% ({count} veces)")
    
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
    print(f"📁 Modelo guardado como: ppo_final_anti_hold_v1.zip")
    
    # Performance analysis
    print(f"\n📈 ANÁLISIS DE PERFORMANCE:")
    hold_percentage = (test_counts[test_unique == 0][0] if 0 in test_unique else 0) / len(test_actions) * 100
    trade_percentage = 100 - hold_percentage
    
    print(f"  🎯 Trading Activity: {trade_percentage:.1f}%")
    print(f"  😴 Hold Activity: {hold_percentage:.1f}%")
    
    if test_return > test_buy_hold:
        print(f"  🏆 ¡VICTORIA! PPO beats Buy & Hold por {test_return - test_buy_hold:.2f}%")
    else:
        print(f"  📉 PPO underperforms por {test_buy_hold - test_return:.2f}%")
    
    if test_trades > 5:
        print(f"  ✅ ¡TRADING ACTIVO! {test_trades} trades ejecutados")
    else:
        print(f"  ⚠️  Trading aún conservador: solo {test_trades} trades")

if __name__ == "__main__":
    train_final_ppo() 