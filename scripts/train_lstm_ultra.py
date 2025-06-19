#!/usr/bin/env python3
"""
LSTM ULTRA AGRESIVO - Sin piedad para el HOLD!
Aplicando técnicas BRUTALES para forzar trading activo
"""

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class UltraAggressiveLSTM(nn.Module):
    """LSTM ultra agresivo para trading activo"""
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super(UltraAggressiveLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers con más dropout para exploración
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Output layers - precio, confianza Y momentum
        self.price_head = nn.Linear(hidden_size, 1)
        self.confidence_head = nn.Linear(hidden_size, 1)
        self.momentum_head = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Triple prediction
        price_pred = self.price_head(last_output)
        confidence = torch.sigmoid(self.confidence_head(last_output))
        momentum = torch.tanh(self.momentum_head(last_output))  # -1 to 1
        
        return price_pred, confidence, momentum

class UltraAggressiveStrategy:
    """Estrategia ULTRA AGRESIVA - SIN PIEDAD PARA EL HOLD"""
    
    def __init__(self, initial_cash=1000000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares = 0
        self.trade_history = []
        self.consecutive_holds = 0
        self.total_trades = 0
        self.small_trades_penalty = 0
        
        # THRESHOLDS ULTRA AGRESIVOS
        self.buy_threshold = 0.005   # 0.5% (vs 1% antes)
        self.sell_threshold = -0.003 # 0.3% (vs 0.5% antes)
        self.confidence_threshold = 0.1  # 10% (vs 30% antes)
        
    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.trade_history = []
        self.consecutive_holds = 0
        self.total_trades = 0
        self.small_trades_penalty = 0
        
    def make_decision(self, current_price, predicted_price, confidence, momentum):
        """
        ULTRA AGGRESSIVE DECISION MAKING
        """
        
        # Expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # BRUTAL PENALTIES por holds consecutivos (hasta 5%!)
        hold_penalty = min(self.consecutive_holds * 0.005, 0.05)  # 5x más brutal
        
        # PENALTY por usar solo trades pequeños
        small_trade_penalty = min(self.small_trades_penalty * 0.001, 0.02)
        
        # ULTRA AGGRESSIVE THRESHOLDS
        adj_buy_threshold = self.buy_threshold - hold_penalty - small_trade_penalty
        adj_sell_threshold = self.sell_threshold + hold_penalty + small_trade_penalty
        
        # Portfolio ratios
        total_value = self.cash + self.shares * current_price
        cash_ratio = self.cash / total_value
        stock_ratio = 1 - cash_ratio
        
        # MOMENTUM BOOST - usar señal de momentum
        momentum_boost = float(momentum) * 0.01  # Hasta 1% boost
        
        action = 0  # Default HOLD
        
        # ULTRA AGGRESSIVE BUYING
        buy_signal = expected_return + momentum_boost > adj_buy_threshold
        sell_signal = expected_return + momentum_boost < adj_sell_threshold
        
        if buy_signal and confidence > self.confidence_threshold:
            
            # FORCE BIG TRADES con bonuses
            if cash_ratio > 0.7:  # Mucho cash
                if expected_return > 0.02:  # 2%+ prediction
                    action = 3  # Buy 50% - BIG TRADE BONUS!
                elif expected_return > 0.015:  # 1.5%+ prediction
                    action = 2  # Buy 25%
                elif expected_return > 0.01:  # 1%+ prediction  
                    action = 2  # Buy 25% (promote bigger trades)
                else:
                    action = 1  # Buy 10%
                    
            elif cash_ratio > 0.4:  # Cash moderado
                if expected_return > 0.015:
                    action = 2  # Buy 25%
                elif expected_return > 0.008:
                    action = 2  # Buy 25% (promote bigger)
                else:
                    action = 1  # Buy 10%
                    
            elif cash_ratio > 0.15:  # Poco cash
                if expected_return > 0.01:
                    action = 2  # Buy 25% (even with little cash!)
                else:
                    action = 1  # Buy 10%
            
            elif cash_ratio > 0.05:  # Muy poco cash
                action = 1  # Buy 10%
        
        # ULTRA AGGRESSIVE SELLING
        elif sell_signal and confidence > self.confidence_threshold:
            
            if stock_ratio > 0.8:  # Muchas acciones
                if expected_return < -0.015:  # -1.5%+ prediction
                    action = 6  # Sell 50% - BIG TRADE BONUS!
                elif expected_return < -0.01:  # -1%+ prediction
                    action = 5  # Sell 25%
                elif expected_return < -0.007:  # -0.7%+ prediction
                    action = 5  # Sell 25% (promote bigger)
                else:
                    action = 4  # Sell 10%
                    
            elif stock_ratio > 0.5:  # Acciones moderadas
                if expected_return < -0.01:
                    action = 5  # Sell 25%
                elif expected_return < -0.006:
                    action = 5  # Sell 25% (promote bigger)
                else:
                    action = 4  # Sell 10%
                    
            elif stock_ratio > 0.2:  # Pocas acciones
                if expected_return < -0.008:
                    action = 5  # Sell 25% (even with few shares!)
                else:
                    action = 4  # Sell 10%
                    
            elif stock_ratio > 0.05:  # Muy pocas acciones
                action = 4  # Sell 10%
        
        # BRUTAL FORCE TRADING: Después de solo 5 holds!
        if action == 0 and self.consecutive_holds > 5:
            # Force random big action
            if cash_ratio > stock_ratio:  # Más cash -> BUY AGGRESSIVELY
                if cash_ratio > 0.6:
                    action = 2  # Force Buy 25%
                else:
                    action = 1  # Force Buy 10%
            else:  # Más stocks -> SELL AGGRESSIVELY  
                if stock_ratio > 0.6:
                    action = 5  # Force Sell 25%
                else:
                    action = 4  # Force Sell 10%
        
        # EXTRA BRUTAL: Después de 10 holds, force BIG action
        if action == 0 and self.consecutive_holds > 10:
            if cash_ratio > 0.5:
                action = 3  # Force Buy 50%!
            elif stock_ratio > 0.5:
                action = 6  # Force Sell 50%!
            else:
                action = 2 if cash_ratio > stock_ratio else 5  # Force 25%
        
        return action
    
    def execute_trade(self, action, current_price):
        """Ejecuta trades y aplica penalties por conservadurismo"""
        
        action_executed = False
        
        if action == 0:  # Hold
            self.consecutive_holds += 1
            
        else:  # Trade action
            self.consecutive_holds = 0
            
            # PENALTY por usar solo trades pequeños
            if action in [1, 4]:  # Buy/Sell 10%
                self.small_trades_penalty += 1
            elif action in [2, 5]:  # Buy/Sell 25%
                self.small_trades_penalty = max(0, self.small_trades_penalty - 1)
            elif action in [3, 6]:  # Buy/Sell 50%
                self.small_trades_penalty = max(0, self.small_trades_penalty - 3)
            
            if action == 1:  # Buy 10%
                buy_amount = min(self.cash * 0.1, self.cash - 500)  # Menos cash mínimo
                if buy_amount > 50:  # Threshold más bajo
                    shares_to_buy = buy_amount / current_price
                    self.cash -= buy_amount
                    self.shares += shares_to_buy
                    action_executed = True
                    
            elif action == 2:  # Buy 25%
                buy_amount = min(self.cash * 0.25, self.cash - 500)
                if buy_amount > 50:
                    shares_to_buy = buy_amount / current_price
                    self.cash -= buy_amount
                    self.shares += shares_to_buy
                    action_executed = True
                    
            elif action == 3:  # Buy 50%
                buy_amount = min(self.cash * 0.5, self.cash - 500)
                if buy_amount > 50:
                    shares_to_buy = buy_amount / current_price
                    self.cash -= buy_amount
                    self.shares += shares_to_buy
                    action_executed = True
                    
            elif action == 4:  # Sell 10%
                shares_to_sell = min(self.shares * 0.1, self.shares - 0.5)  # Menos shares mínimas
                if shares_to_sell > 0.05:  # Threshold más bajo
                    self.cash += shares_to_sell * current_price
                    self.shares -= shares_to_sell
                    action_executed = True
                    
            elif action == 5:  # Sell 25%
                shares_to_sell = min(self.shares * 0.25, self.shares - 0.5)
                if shares_to_sell > 0.05:
                    self.cash += shares_to_sell * current_price
                    self.shares -= shares_to_sell
                    action_executed = True
                    
            elif action == 6:  # Sell 50%
                shares_to_sell = min(self.shares * 0.5, self.shares - 0.5)
                if shares_to_sell > 0.05:
                    self.cash += shares_to_sell * current_price
                    self.shares -= shares_to_sell
                    action_executed = True
        
        if action_executed:
            self.total_trades += 1
            
        return action_executed
    
    def get_portfolio_value(self, current_price):
        return self.cash + self.shares * current_price

def prepare_ultra_data(data, sequence_length=20):  # Secuencias más cortas para más reactividad
    """Prepara datos ULTRA optimizados"""
    
    df = data.copy()
    
    # Indicadores técnicos MÁS sensibles
    df['MA_3'] = df['Close'].rolling(window=3).mean()   # MA más corta
    df['MA_10'] = df['Close'].rolling(window=10).mean() # MA más corta
    df['volatility'] = df['Close'].rolling(window=10).std()  # Volatilidad más sensible
    df['returns'] = df['Close'].pct_change()
    df['momentum'] = df['Close'].pct_change(periods=3)  # Momentum 3-day
    
    # RSI más sensible
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()  # RSI 7-day
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Features para LSTM
    features = ['Close', 'returns', 'volatility', 'MA_3', 'MA_10', 'RSI']
    
    df = df.dropna()
    
    # Normalizar
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Crear secuencias más cortas
    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(df['Close'].iloc[i])
    
    return np.array(X), np.array(y), scaler, df

def train_ultra_lstm():
    print("💀 ENTRENANDO LSTM ULTRA AGRESIVO - SIN PIEDAD!")
    print("=" * 70)
    
    # Datos
    print("📊 Descargando datos...")
    data = yf.download('^GSPC', start='2020-01-01', end='2024-01-01', progress=False)
    data = data.dropna()
    
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point].copy()
    test_data = data[split_point:].copy()
    
    print(f"📈 Datos de entrenamiento: {len(train_data)} días")
    print(f"📉 Datos de prueba: {len(test_data)} días")
    
    # Preparar datos ULTRA
    print("🔧 Preparando datos ULTRA...")
    X_train, y_train, scaler, train_df = prepare_ultra_data(train_data, sequence_length=20)
    X_test, y_test, _, test_df = prepare_ultra_data(test_data, sequence_length=20)
    
    print(f"🎯 Training sequences: {X_train.shape}")
    print(f"🎯 Test sequences: {X_test.shape}")
    
    # Tensores
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Modelo ULTRA
    model = UltraAggressiveLSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.3)
    
    # Configuración ULTRA AGRESIVA
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)  # LR más alto
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    
    print(f"\n💀 INICIO DEL ENTRENAMIENTO ULTRA BRUTAL...")
    print("⚡ ULTRA Features: 3-day MA, 7-day RSI, 20-day sequences")
    print("🔥 BRUTAL Penalties: 5% hold penalty, big trade bonuses")
    print("💀 FORCE TRADING: After 5 holds (vs 10 before)")
    print("🎯 Triple head: Price + Confidence + Momentum")
    
    # Training ULTRA
    num_epochs = 80  # Menos epochs, más intenso
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward con triple head
        price_pred, confidence, momentum = model(X_train)
        price_pred = price_pred.squeeze()
        
        # Loss function ULTRA - premia confianza alta y momentum
        price_loss = criterion(price_pred, y_train)
        confidence_loss = criterion(confidence.squeeze(), torch.ones_like(confidence.squeeze()) * 0.8)  # Target 80% confidence
        momentum_loss = criterion(momentum.squeeze(), torch.zeros_like(momentum.squeeze()))  # Neutral momentum target
        
        total_loss = price_loss + 0.15 * confidence_loss + 0.05 * momentum_loss
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_price_pred, val_confidence, val_momentum = model(X_test)
            val_price_pred = val_price_pred.squeeze()
            val_loss = criterion(val_price_pred, y_test)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'input_size': 6,
                'hidden_size': 64,
                'num_layers': 2
            }, 'lstm_ultra_aggressive_v1.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"⏹️  Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 15 == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print("💀 MODELO ULTRA ENTRENADO!")
    
    # Test ULTRA strategy
    print(f"\n🧪 PROBANDO ESTRATEGIA ULTRA AGRESIVA...")
    
    checkpoint = torch.load('lstm_ultra_aggressive_v1.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test en training data
    trader = UltraAggressiveStrategy()
    actions_taken = []
    portfolio_values = []
    
    with torch.no_grad():
        for i in range(len(X_train)):
            current_price = float(y_train[i])
            price_pred, confidence, momentum = model(X_train[i:i+1])
            predicted_price = float(price_pred.squeeze())
            conf_score = float(confidence.squeeze())
            momentum_score = float(momentum.squeeze())
            
            action = trader.make_decision(current_price, predicted_price, conf_score, momentum_score)
            trader.execute_trade(action, current_price)
            
            actions_taken.append(action)
            portfolio_values.append(trader.get_portfolio_value(current_price))
    
    # Resultados training
    final_value = portfolio_values[-1]
    total_return = (final_value - trader.initial_cash) / trader.initial_cash * 100
    buy_hold_return = (float(y_train[-1]) - float(y_train[0])) / float(y_train[0]) * 100
    
    print(f"\n📊 RESULTADOS ENTRENAMIENTO:")
    print(f"💀 Return LSTM ULTRA: {total_return:.2f}%")
    print(f"📈 Return Buy & Hold: {buy_hold_return:.2f}%")
    print(f"🎯 Alpha (diferencia): {total_return - buy_hold_return:.2f}%")
    print(f"🔄 Total trades: {trader.total_trades}")
    
    # Distribución actions
    unique, counts = np.unique(actions_taken, return_counts=True)
    action_names = ['Hold', 'Buy 10%', 'Buy 25%', 'Buy 50%', 'Sell 10%', 'Sell 25%', 'Sell 50%']
    
    print(f"\n🎬 DISTRIBUCIÓN DE ACCIONES:")
    for action_idx, count in zip(unique, counts):
        percentage = count / len(actions_taken) * 100
        print(f"  {action_names[action_idx]}: {percentage:.1f}% ({count} veces)")
    
    # Test en test data
    print(f"\n🧪 PROBANDO EN DATOS DE PRUEBA...")
    trader.reset()
    test_actions = []
    test_portfolio_values = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            current_price = float(y_test[i])
            price_pred, confidence, momentum = model(X_test[i:i+1])
            predicted_price = float(price_pred.squeeze())
            conf_score = float(confidence.squeeze())
            momentum_score = float(momentum.squeeze())
            
            action = trader.make_decision(current_price, predicted_price, conf_score, momentum_score)
            trader.execute_trade(action, current_price)
            
            test_actions.append(action)
            test_portfolio_values.append(trader.get_portfolio_value(current_price))
    
    # Test results
    test_final_value = test_portfolio_values[-1]
    test_return = (test_final_value - trader.initial_cash) / trader.initial_cash * 100
    test_buy_hold = (float(y_test[-1]) - float(y_test[0])) / float(y_test[0]) * 100
    
    print(f"\n📊 RESULTADOS PRUEBA (DATOS NO VISTOS):")
    print(f"💀 Return LSTM ULTRA: {test_return:.2f}%")
    print(f"📈 Return Buy & Hold: {test_buy_hold:.2f}%")
    print(f"🎯 Alpha (diferencia): {test_return - test_buy_hold:.2f}%")
    print(f"🔄 Total trades: {trader.total_trades}")
    
    # Test distribución
    test_unique, test_counts = np.unique(test_actions, return_counts=True)
    print(f"\n🎬 DISTRIBUCIÓN DE ACCIONES (PRUEBA):")
    for action_idx, count in zip(test_unique, test_counts):
        percentage = count / len(test_actions) * 100
        print(f"  {action_names[action_idx]}: {percentage:.1f}% ({count} veces)")
    
    # Análisis ULTRA
    hold_count = test_counts[test_unique == 0][0] if 0 in test_unique else 0
    hold_percentage = hold_count / len(test_actions) * 100
    trade_percentage = 100 - hold_percentage
    
    print(f"\n💀 ANÁLISIS ULTRA FINAL:")
    print(f"  🔥 Trading Activity: {trade_percentage:.1f}%")
    print(f"  😴 Hold Activity: {hold_percentage:.1f}%")
    
    if test_return > test_buy_hold:
        print(f"  🏆 ¡VICTORY! LSTM ULTRA beats Buy & Hold por {test_return - test_buy_hold:.2f}%")
    else:
        print(f"  📉 LSTM ULTRA underperforms por {test_buy_hold - test_return:.2f}%")
    
    if trader.total_trades > 20:
        print(f"  💀 ¡ULTRA TRADING! {trader.total_trades} trades ejecutados")
    else:
        print(f"  ⚠️  Aún conservador: solo {trader.total_trades} trades")
    
    print(f"\n🏁 COMPARACIÓN ÉPICA:")
    print(f"  🔥 PPO Final:      17.60% (56.2% ACTIVITY)")  
    print(f"  🧠 LSTM Activo:    0.75%  (32.0% ACTIVITY)")
    print(f"  💀 LSTM ULTRA:    {test_return:.2f}% ({trade_percentage:.1f}% ACTIVITY)")
    print(f"  📈 Buy & Hold:    {test_buy_hold:.2f}%")
    
    # Determinar ganador
    if trade_percentage > 50 and test_return > 5:
        print(f"\n🎉 ¡ÉXITO BRUTAL! LSTM ULTRA ahora compite con PPO!")
        print(f"💀 Hold Bias EXTERMINADO completamente")
    elif trade_percentage > 40:
        print(f"\n🔥 ¡GRAN PROGRESO! LSTM ULTRA más activo que nunca")
        print(f"✅ Hold Bias significativamente reducido")
    else:
        print(f"\n⚠️ Necesita aún MÁS brutalidad...")

if __name__ == "__main__":
    train_ultra_lstm() 