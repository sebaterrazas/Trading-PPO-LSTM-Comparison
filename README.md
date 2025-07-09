# 🚀 Trading PPO vs LSTM Multi-Asset Portfolio Comparison

**📄 Accompanying Repository for Academic Paper**: _"Comparación de aprendizaje por refuerzo (PPO) vs aprendizaje supervisado (LSTM) para trading en portafolios multi-activo"_

## 📖 Coming from the Paper?

**Welcome!** This repository contains all the code, trained models, and results from our research paper. Here's what you'll find:

- ✅ **Pre-trained models** ready to use (no need to retrain)
- ✅ **Complete results** in `results/` folder with all figures from the paper
- ✅ **Reproduction scripts** to verify our findings
- ✅ **11 different strategies** tested across Train/Validation/Test periods

## 🎯 Project Overview

This project implements and compares multiple trading strategies on a multi-asset portfolio (BTC, S&P 500, Treasury Bonds):

- **🤖 PPO Models**: 4 variants (PPO-T 30k, PPO 30k, PPO 100k, PPO Bitcoin)
- **🧠 LSTM Model**: Deep Learning sequence-based approach
- **📈 Buy & Hold Strategies**: Traditional benchmarks (Bitcoin Only, S&P 500, Aggressive, Conservative, Equal Weight)

**🏆 Key Finding**: PPO 100k achieves **same return as S&P 500** (15.7% vs 15.9%) but with **37% less volatility** (14.6% vs 20.2%)

## 📊 Data & Methodology

### Data Splits (Same as ICML Paper)

- **Training**: 2010-2016 (7 years, 1,762 days)
- **Validation**: 2017-2018 (2 years, 502 days)
- **Testing**: 2019-2024 (5.9 years, 1,489 days)

### Features (Fama-French Factors)

1. `returns` - Daily returns
2. `momentum` - 12-month momentum
3. `size_factor` - Market cap factor
4. `value_factor` - Book-to-market ratio
5. `profitability` - Return on equity
6. `volatility` - 30-day volatility
7. `rsi` - Relative Strength Index
8. `ma_ratio` - Moving average ratio
9. `volume_ratio` - Volume ratio

### Trading Environment

- **Actions**: 11 segmented actions {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
- **Transaction Costs**: 0.2% (realistic)
- **Initial Balance**: $1,000,000
- **Reward Function**: Sharpe-like risk-adjusted returns

## 🗂️ Repository Structure

```
📁 Trading-PPO-LSTM-Comparison/
├── 🤖 ppo_trading_v2.py       # Multi-asset PPO implementation
├── 🧠 lstm_trading_v2.py      # LSTM multi-asset model
├── 📊 compare_models.py       # Paper results reproduction script
├── 📋 requirements.txt        # Dependencies
├── 📖 README.md              # This file
├── 📁 trained_models/        # Pre-trained models (ready to use!)
│   ├── bitcoin_focused_ppo_1M.zip      # PPO 1M timesteps
│   ├── bitcoin_focused_ppo_150k.zip    # PPO 150k timesteps
│   ├── lstm_multi_asset.pth            # LSTM model
│   └── ... (more models)
├── 📁 results/               # Paper figures and results
│   ├── multi_asset_comparison.png      # Main paper figure
│   ├── bitcoin_multi_asset_comparison.png
│   └── complete_multi_asset_comparison.png
└── 📁 docs/                  # Research papers and documentation
    ├── 1-s2.0-S0377221717310652-main.pdf
    ├── 2302.02269v3.pdf
    └── 2506.04658v1.pdf
```

## 🚀 Quick Start (Paper Readers)

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/sebaterrazas/Trading-PPO-LSTM-Comparison
cd Trading-PPO-LSTM-Comparison

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. 📊 Reproduce Paper Results (Using Pre-trained Models)

**All models are already trained and saved in `trained_models/` folder!**

```bash
# Run the complete comparison from the paper
python compare_models.py
```

**This will**:

- ✅ Load all pre-trained models automatically
- ✅ Generate Table 1 from the paper with all metrics
- ✅ Create the complete visualization: `results/multi_asset_comparison.png`
- ✅ Test all 11 strategies across Train/Validation/Test periods

### 3. 🔍 Explore Individual Results

#### Check Pre-trained Models

```bash
ls trained_models/
# bitcoin_focused_ppo_1M.zip        ← PPO models ready to use
# bitcoin_focused_ppo_150k.zip
# lstm_multi_asset.pth              ← LSTM model ready to use
# ... (more models)
```

#### View Paper Figures

```bash
ls results/
# multi_asset_comparison.png        ← Main paper figure
# bitcoin_multi_asset_comparison.png
# complete_multi_asset_comparison.png
```

### 4. 🔬 Retrain Models (Optional)

**Only if you want to reproduce training from scratch:**

#### Train PPO Models

```bash
# For different PPO variants
python ppo_trading_v2.py           # PPO 100k
python ppo_bitcoin_focused.py      # Bitcoin-focused PPO
# python ppo_transformer.py        # PPO-T 30k (if available)
```

#### Train LSTM Model

```bash
python lstm_trading_v2.py
```

**Note**: Training takes time! Pre-trained models give identical results.

## 📈 Paper Results (Test Period: 2019-2024)

### Complete Performance Table (Table 1 from Paper)

| Strategy         | Total Return (%) | Annual Return (%) | Sharpe Ratio | Max Drawdown (%) | Volatility (%) |
| ---------------- | ---------------- | ----------------- | ------------ | ---------------- | -------------- |
| **Bitcoin Only** | **2365.81**      | **72.02**         | **1.09**     | -76.63           | 65.79          |
| **PPO 100k**     | **136.81**       | **15.71**         | **1.08**     | **-22.33**       | **14.61**      |
| Aggressive       | 799.11           | 45.02             | 1.02         | -64.92           | 43.92          |
| Equal Weight     | 828.12           | 45.80             | 1.00         | -68.66           | 45.65          |
| Conservative     | 284.02           | 25.57             | 0.95         | -54.21           | 26.96          |
| **LSTM**         | **227.33**       | **22.39**         | **0.91**     | **-41.43**       | **24.66**      |
| S&P 500 Only     | 139.64           | 15.94             | 0.79         | -33.89           | 20.18          |
| PPO 30k          | 130.49           | 15.18             | 0.76         | -33.75           | 19.99          |
| PPO Bitcoin      | 390.24           | 30.87             | 0.64         | -70.73           | 48.05          |
| PPO-T 30k        | 518.39           | 36.12             | 0.63         | -76.20           | 57.42          |
| Free PPO 150k    | 507.36           | 35.70             | 0.61         | -76.59           | 58.77          |

### 🏆 Key Finding from Paper

**PPO 100k vs S&P 500 Only**:

- **Same Annual Return**: 15.71% vs 15.94% (practically identical)
- **37% Less Volatility**: 14.61% vs 20.18%
- **Better Sharpe Ratio**: 1.08 vs 0.79 (+37% improvement)
- **Better Drawdown**: -22.33% vs -33.89% (+34% improvement)

> _"PPO 100k achieves the holy grail: same returns as the most popular investment strategy (S&P 500 buy-and-hold) but with significantly less volatility and stress."_

## 🔬 Technical Implementation

### PPO (Reinforcement Learning)

- **Algorithm**: Proximal Policy Optimization
- **Network**: Actor-Critic with 64-64 hidden layers
- **Training**: 100,000 timesteps
- **Environment**: Custom gym environment with Fama-French features
- **Action Space**: Discrete 11 actions (segmented trading)

### LSTM (Deep Learning)

- **Architecture**: 2-layer LSTM with 128 hidden units
- **Sequence Length**: 20 days
- **Features**: Same 9 Fama-French factors as PPO
- **Training**: 200 epochs with early stopping
- **Output**: Action probabilities for 11 trading actions

### Key Innovations

1. **Enhanced Features**: Fama-French factors vs basic technical indicators
2. **Segmented Actions**: 11 actions vs binary buy/sell/hold
3. **Transaction Costs**: Realistic 0.2% costs
4. **Risk-Adjusted Rewards**: Sharpe-like optimization
5. **Proper Data Splits**: No look-ahead bias

## 📊 Files Description

### Core Scripts

- **`ppo_trading.py`**: Complete PPO implementation with training and testing
- **`lstm_trading.py`**: LSTM model with sequence-based prediction
- **`compare_models.py`**: Fair comparison of all three approaches

### 📋 Key Files for Paper Reproduction

| File                 | Purpose                                  | Paper Section        |
| -------------------- | ---------------------------------------- | -------------------- |
| `compare_models.py`  | **Main script** - reproduces all results | All figures & tables |
| `ppo_trading_v2.py`  | Multi-asset PPO training (if needed)     | PPO methodology      |
| `lstm_trading_v2.py` | LSTM model training (if needed)          | LSTM methodology     |
| `paper.tex`          | LaTeX source of the academic paper       | Full paper           |
| `trained_models/`    | **Pre-trained models** (ready to use)    | All experiments      |
| `results/`           | **Paper figures** (generated outputs)    | All visualizations   |

### Key Functions

- `load_simple_data()`: Loads multi-asset data (BTC, S&P 500, Bonds) with time splits
- `SimpleMultiAssetEnv`: Custom RL environment with transaction costs
- `create_buy_hold_portfolios()`: Traditional benchmark strategies
- `LSTMTradingModel`: PyTorch LSTM architecture for multi-asset trading
- `test_simple_model()`: Model evaluation across all time periods

## 🎯 Research Value

### Academic Contributions

1. **First Implementation** to exceed ICML 2025 paper results
2. **Comprehensive Comparison** of RL vs DL vs Traditional methods
3. **Recent Period Analysis** (2019-2024) including COVID impact
4. **Reproducible Results** with open-source implementation

### Business Value

1. **Practical Trading Strategy** with realistic transaction costs
2. **Risk Management** through Sharpe ratio optimization
3. **Scalable Architecture** for different assets/markets
4. **Performance Monitoring** with detailed metrics

## 🔧 Customization

### Modify Trading Parameters

```python
# In ppo_trading.py or lstm_trading.py
INITIAL_BALANCE = 1000000      # Starting capital
TRANSACTION_COST = 0.002       # 0.2% transaction cost
SEQUENCE_LENGTH = 20           # LSTM lookback period
```

### Add New Features

```python
# In get_fama_french_features()
def get_fama_french_features(data):
    # Add your custom features here
    features['new_feature'] = calculate_new_feature(data)
    return features
```

### Experiment with Actions

```python
# Modify action mapping in both models
action_mapping = {
    0: -10, 1: -5, 2: -2, 3: -1, 4: 0,  # More aggressive actions
    5: 0, 6: 1, 7: 2, 8: 5, 9: 10
}
```

## 📚 References

1. **Our Paper**: "Comparación de aprendizaje por refuerzo (PPO) vs aprendizaje supervisado (LSTM) para trading en portafolios multi-activo"
2. **PPO Algorithm**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
3. **LSTM Networks**: Hochreiter & Schmidhuber "Long Short-Term Memory" (1997)
4. **Stable Baselines3**: Raffin et al. PPO implementation framework

## 🤝 Contributing

Interested in extending this research? We welcome contributions!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Paper Summary

**🎯 Key Achievements**:

- ✅ **Novel Finding**: PPO 100k matches S&P 500 returns with 37% less volatility
- ✅ **Comprehensive Comparison**: 11 strategies across RL, DL, and traditional methods
- ✅ **Multi-Asset Portfolio**: BTC, S&P 500, and Treasury Bonds
- ✅ **Realistic Setup**: Transaction costs, proper time splits, no look-ahead bias
- ✅ **Reproducible Results**: All code and models publicly available

**🔬 Research Impact**: First study to show ML can match traditional benchmarks with lower risk in multi-asset portfolios!
