# 🚀 Active Trading Models: PPO & LSTM

## Breaking the Hold Bias in Algorithmic Trading

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io)

## 🎯 Project Overview

This project successfully **eliminates the Hold Bias** in algorithmic trading models, transforming passive holders into **active traders**. We implemented and optimized two advanced models:

- **🔥 PPO Final**: Proximal Policy Optimization with anti-hold techniques
- **💀 LSTM Ultra**: Ultra-aggressive LSTM with momentum prediction
- **📈 Comparison System**: Comprehensive benchmark against Buy & Hold

### 🏆 Key Achievement

**Transformed models from 100% Hold → 33-44% Hold, with 56-67% active trading!**

---

## 📊 Performance Results

| Model             | Return     | Activity  | Trades  | Hold %    | Status          |
| ----------------- | ---------- | --------- | ------- | --------- | --------------- |
| 🏆 **Buy & Hold** | **21.70%** | 0.0%      | 0       | 100%      | **WINNER**      |
| 🥈 **PPO Final**  | **17.60%** | **56.2%** | **102** | **43.8%** | **ACTIVE**      |
| 🥉 **LSTM Ultra** | 1.68%      | **67.1%** | **116** | **32.9%** | **MOST ACTIVE** |

### 🎯 Mission Accomplished

- ✅ **Hold Bias ELIMINATED** in both models
- ✅ **Active trading achieved** (50%+ activity)
- ✅ **Multiple trade sizes** (10%, 25%, 50% positions)
- ✅ **Balanced buy/sell behavior**

---

## 🏗️ Project Structure

```
📁 Proyecto/
├── 📁 scripts/                 # Training scripts
│   ├── train_ppo_final.py     # 🔥 PPO Final (17.60% return, 56% activity)
│   └── train_lstm_ultra.py    # 💀 LSTM Ultra (67% activity, 116 trades)
├── 📁 trained_models/         # Pre-trained model weights
│   ├── ppo_final_anti_hold_v1.zip    # PPO model ready to use
│   └── lstm_ultra_aggressive_v1.pth  # LSTM model ready to use
├── 📁 results/               # Analysis results
│   └── model_comparison_final.png    # Performance visualization
├── compare_models.py         # 🏆 Comprehensive comparison tool
├── requirements.txt          # Dependencies
├── .venv/                   # Virtual environment
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate
git clone <repo-url>
cd Proyecto

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Comparison (Recommended)

```bash
python compare_models.py
```

**Output**: Complete comparison with visualization saved as `results/model_comparison_final.png`

### 3. Train Individual Models

```bash
# Train PPO model (5-10 minutes)
python scripts/train_ppo_final.py

# Train LSTM model (3-5 minutes)
python scripts/train_lstm_ultra.py
```

---

## 🔬 Technical Details

### 🔥 PPO Final Model

**Anti-Hold Reinforcement Learning Agent**

- **Algorithm**: Proximal Policy Optimization with custom anti-hold environment
- **Actions**: 7 granular actions (Hold, Buy/Sell 10%/25%/50%)
- **Features**: Price, momentum, volatility, portfolio ratios, trade penalties
- **Training**: 100,000 timesteps with aggressive reward shaping
- **Innovation**: Exponential penalties for consecutive holds + alpha tracking

```python
# Key anti-hold techniques
hold_penalty = min(consecutive_holds * 0.002, 0.02)  # Up to 2% penalty
force_trading_after = 10  # consecutive holds
alpha_bonus = agent_return - market_return  # Beat the market
```

### 💀 LSTM Ultra Model

**Ultra-Aggressive Price Prediction + Trading Strategy**

- **Architecture**: 2-layer LSTM (64 hidden, 30% dropout) + triple output heads
- **Outputs**: Price prediction + Confidence + Momentum
- **Features**: Technical indicators (MA_3, MA_10, RSI_7, volatility)
- **Strategy**: Ultra-aggressive thresholds with brutal hold penalties
- **Innovation**: 5% hold penalty + forced big trades + momentum signals

```python
# Ultra-aggressive settings
buy_threshold = 0.005    # 0.5% (vs 1% standard)
sell_threshold = -0.003  # 0.3% (vs 0.5% standard)
confidence_threshold = 0.1  # 10% (vs 30% standard)
force_trading_after = 5  # holds (vs 10 standard)
```

### 📊 Advanced Metrics

**Risk-Adjusted Performance Analysis**

- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum loss from peak
- **Volatility**: Annualized price volatility
- **Activity %**: Percentage of time spent trading (vs holding)
- **Trade Distribution**: Analysis of action patterns

---

## 🎯 Research Contributions

### 1. 🔬 Hold Bias Problem Identification

**Problem**: Traditional trading models default to 100% Hold behavior

- PPO agents learn to hold instead of trade
- LSTM models predict but don't act
- Buy & Hold dominates due to sparse rewards

### 2. 💡 Anti-Hold Solutions Developed

**Exponential Hold Penalties**:

```python
penalty = min(consecutive_holds * penalty_rate, max_penalty)
reward = base_reward - penalty
```

**Forced Trading Mechanisms**:

```python
if consecutive_holds > threshold:
    action = force_random_trade()
```

**Portfolio Diversification Bonuses**:

```python
optimal_cash_ratio = 0.2  # to 0.6
diversity_bonus = 1 - abs(current_ratio - optimal_ratio)
```

### 3. 🏆 Validation Results

**Before**: Both models → 100% Hold, 0 trades  
**After**: PPO (56% activity), LSTM (67% activity)

---

## 📈 Live Trading Considerations

### ⚠️ Important Disclaimers

```
🚨 RESEARCH PURPOSES ONLY
   This code is for educational and research purposes.
   NOT financial advice. NOT ready for live trading.
   Past performance ≠ future results.
```

### 🔧 Production Requirements

For live trading, you would need:

- [ ] **Transaction costs** modeling (commissions, slippage)
- [ ] **Risk management** (position sizing, stop losses)
- [ ] **Real-time data** feeds and execution
- [ ] **Regulatory compliance** and monitoring
- [ ] **Extensive backtesting** on multiple timeframes/assets
- [ ] **Paper trading** validation before live deployment

---

## 🛠️ Development

### Dependencies

```
torch>=2.0.0              # Deep learning framework
stable-baselines3>=2.0.0  # RL algorithms
gymnasium>=1.0.0          # RL environments
yfinance>=0.2.0           # Market data
pandas>=2.0.0             # Data manipulation
numpy>=1.20.0             # Numerical computing
matplotlib>=3.5.0         # Plotting
scikit-learn>=1.0.0       # ML utilities
```

### Configuration

**PPO Hyperparameters**:

```python
PPO_CONFIG = {
    "learning_rate": 0.0005,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 15,
    "clip_range": 0.25,
    "ent_coef": 0.1,  # High entropy for exploration
}
```

**LSTM Hyperparameters**:

```python
LSTM_CONFIG = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,  # High dropout for regularization
    "sequence_length": 20,  # Shorter for reactivity
    "learning_rate": 0.002,
}
```

---

## 🔮 Future Work

### Immediate Improvements

- [ ] **Multi-asset trading** (stocks, ETFs, crypto)
- [ ] **Ensemble methods** (combine PPO + LSTM)
- [ ] **Alternative algorithms** (SAC, TD3, Transformers)
- [ ] **Real transaction costs** integration

### Advanced Features

- [ ] **Sentiment analysis** integration
- [ ] **Options and derivatives** trading
- [ ] **Portfolio optimization** across multiple assets
- [ ] **Dynamic position sizing** based on volatility

---

## 📞 Contact & Contributing

### Issues & Improvements

- 🐛 **Bug reports**: Create GitHub issues
- 💡 **Feature requests**: Open discussions
- 🔧 **Pull requests**: Always welcome

### Research Collaboration

Interested in:

- 🎓 **Academic collaboration** on trading ML
- 🏢 **Industry applications** of anti-hold techniques
- 📊 **Alternative asset classes** (crypto, forex, commodities)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🎉 Acknowledgments

**Research inspired by**:

- FinRL-Meta framework for financial RL
- Stable-Baselines3 community
- PyTorch deep learning ecosystem

---

**🚀 From passive holding to active trading - Mission Accomplished!**
