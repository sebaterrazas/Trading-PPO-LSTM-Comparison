#!/usr/bin/env python3
"""
MULTI-ASSET TRADING FRAMEWORK  –  v0.3
=====================================
• Gestiona S&P 500, Bitcoin y un proxy de bono libre de riesgo (ETF BIL).
• Usa columnas planas (`sp500_Close`, `btc_Close`, `bond_Close`) para
  evitar el bug de pandas ≥ 2.2 con MultiIndex.
• Incluye entorno Gym, entrenamiento PPO con tu extractor Transformer,
  ejemplo de LSTM allocator y comparativa Buy&Hold.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

# ───────────────────────────  Librerías estándar  ──────────────────────────
import os
from typing import List

# ───────────────────────────  Ciencia de datos  ───────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────  PyTorch / RL  ─────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.type_aliases import GymStepReturn

# Helpers de tu proyecto original
from ppo_trading import get_fama_french_features, TransformerFeaturesExtractor

def plot_equity_curves(histories: dict[str, list[float]]):
    import matplotlib.pyplot as plt
    plt.figure()
    for name, curve in histories.items():
        plt.plot(curve, label=name)
    plt.title("Evolución del portafolio")
    plt.xlabel("Días")                               # ← eje X
    plt.ylabel("Valor de la cartera")                # ← eje Y
    plt.legend()
    plt.tight_layout()
    plt.savefig("equity_curves.png", dpi=150)
    plt.close()


def plot_metric_bars(metrics: dict[str, dict[str, float]],
                     key: str, title: str, ylabel: str, fname: str):
    import matplotlib.pyplot as plt
    names = list(metrics.keys())
    vals  = [metrics[n][key] for n in names]
    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.xlabel("Estrategia")                         # ← eje X
    plt.ylabel(ylabel)                              # ← eje Y
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def summarize_weights(weights: list[np.ndarray],
                      asset_names=("S&P500", "Bitcoin", "Bono"), tag="ppo"):
    """
    Imprime y devuelve stats de los pesos de cartera.
    """
    import numpy as np
    w = np.vstack(weights)                # shape = (T, 3)

    mean_w   = w.mean(axis=0)
    med_w    = np.median(w, axis=0)
    max_w    = w.max(axis=0)
    gt_half  = (w > 0.5).sum(axis=0) / len(w) * 100   # % de días con >50 %

    print(f"── Tendencias de asignación {tag} ──")
    for i, name in enumerate(asset_names):
        print(f"{name:8s} | media {mean_w[i]:.2%} | mediana {med_w[i]:.2%} "
              f"| pico {max_w[i]:.2%} | >50 % {gt_half[i]:.1f}% días")

    return dict(mean=mean_w, median=med_w, gthalf=gt_half)

# ──────────────────────────────  Parámetros  ──────────────────────────────
TRANSACTION_COST = 0.002           # 0.2 %
INITIAL_BALANCE  = 1_000_000
TRAIN_START, TRAIN_END = "2010-01-01", "2016-12-31"
VAL_END                 = "2018-12-31"         # test ≈ 2019-actual
PRICE_COLS = ["sp500_Close", "btc_Close", "bond_Close"]

# ─────────────────────────────  Utilidades datos  ─────────────────────────
def load_multi_asset_data(start: str = "2010-01-01",
                          end: str   = "2024-12-01") -> pd.DataFrame:
    tickers = {"sp500": "^GSPC", "btc": "BTC-USD", "bond": "BIL"}
    dfs = []
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end,
                         progress=False).dropna()
        dfs.append(df.add_prefix(f"{name}_"))          # columnas planas
    return pd.concat(dfs, axis=1).dropna()

def split_data(data: pd.DataFrame):
    train = data[data.index <= TRAIN_END]
    val   = data[(data.index > TRAIN_END) & (data.index <= VAL_END)]
    test  = data[data.index > VAL_END]
    return train, val, test

# ─────────────────────────  Ingeniería de features  ───────────────────────
def build_feature_frame(data: pd.DataFrame, ref_cols=None) -> pd.DataFrame:
    """
    Devuelve las features Fama-French por activo, asegurando que la
    matriz final tenga **exactamente** las mismas columnas (y orden)
    que 'ref_cols', rellenando NaN con forward/back-fill y ceros.
    """
    # 1️⃣ Aplana MultiIndex → columnas planas
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.map(
            lambda t: "_".join(map(str, t)) if isinstance(t, tuple) else str(t)
        )

    feats_per_asset = []
    for asset in ["sp500", "btc", "bond"]:
        raw = data.filter(like=f"{asset}_").copy()
        raw.columns = raw.columns.str.replace(f"{asset}_", "", regex=False)

        # Close
        if "close" not in [c.lower() for c in raw.columns]:
            cand = [c for c in raw.columns if "close" in c.lower()]
            if cand:
                raw["Close"] = raw[cand[0]]
            else:
                raise KeyError(f"{asset}: falta columna Close")

        # Open / High / Low
        for col in ["Open", "High", "Low"]:
            if col not in raw.columns:
                raw[col] = raw["Close"]

        # Volume constante
        if "Volume" not in raw.columns:
            raw["Volume"] = 1.0

        feats_per_asset.append(
            get_fama_french_features(raw).add_prefix(f"{asset}_")
        )

    out = pd.concat(feats_per_asset, axis=1)

    # 2️⃣  Fuerza mismas columnas que en entrenamiento
    if ref_cols is not None:
        out = out.reindex(columns=ref_cols)

    # 3️⃣  Rellena NaNs: ffill→bfill→0  (ya no se hace dropna)
    out = out.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return out



# ─────────────────────────────  Entorno Gym  ──────────────────────────────
class MultiAssetTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 price_data: pd.DataFrame,
                 feature_data: pd.DataFrame,
                 initial_balance: float = INITIAL_BALANCE,
                 transaction_cost: float = TRANSACTION_COST):
        super().__init__()

        # 1️⃣  Aplana y simplifica nombres de precios
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data = price_data.copy()
            price_data.columns = price_data.columns.map(
                lambda t: "_".join(map(str, t)) if isinstance(t, tuple) else str(t)
            )
        price_data.columns = price_data.columns.map(
            lambda c: "_".join(c.split("_")[:2])  # activo_métrica
        )
        self.weights_history: list[np.ndarray] = []
        

        # 2️⃣  Precios Close de cada activo
        self.prices = price_data[PRICE_COLS].copy()

        # 3️⃣  Features alineadas
        self.features = feature_data.loc[self.prices.index]      # alineación solo temporal (filas)
        self.features = self.features.fillna(method="ffill").fillna(method="bfill")

        # 4️⃣  Contadores de transacción
        self.trade_count    = 0
        self.volume_traded  = 0.0
        self.cost_paid      = 0.0

        # 5️⃣  Espacios Gym
        self.initial_balance  = float(initial_balance)
        self.net_worth = self.initial_balance            # ← valor actual
        self.net_worth_history = [self.net_worth]        # ya estaba, la dejamos igual
        self.transaction_cost = transaction_cost
        self.action_space     = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_len = 1 + 3 + self.features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_len,), dtype=np.float32)
        self.reset()
    
    # Gym API
    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        self.step_idx = 0
        self.balance  = self.initial_balance
        self.shares   = np.zeros(3)
        self.net_worth = self.initial_balance
        self.net_worth_history = [self.net_worth]
        self.returns_history:   List[float] = []
        return self._get_obs(), {}

    # ───────────────────────────────  STEP  ────────────────────────────────
    def step(self, action: np.ndarray):
        """
        • action ∈ [0,1]^3  → pesos objetivo para sp500, btc, bond
        • Rebalancea la cartera al peso deseado.
        • Devuelve obs, reward, done, truncated(False), info.
        """
        # ── 1. Clip y normaliza acción  →  pesos deseados ───────────────────
        w = np.clip(action, 0.0, 1.0).astype(float)
        if w.sum() < 1e-6:              # vector casi nulo → fallback
            w[:] = 1 / 3
        else:
            w /= w.sum()                # ahora w ≥0 y suma 1

        # ── 2. Datos de mercado del paso actual ────────────────────────────
        price_vec = self.prices.iloc[self.step_idx].values.astype(float)

        # ── 3. Valores actuales y deseados de cada activo ($)  ─────────────
        position_values = self.shares * price_vec            # $ actuales
        portfolio_value = self.balance + position_values.sum()

        desired_values  = w * portfolio_value                # $ objetivo
        trade_values    = desired_values - position_values   # + compra | – venta

        # ── 4. Ejecuta rebalanceo activo por activo ────────────────────────
        for i in range(3):
            if abs(trade_values[i]) < 1e-8:
                continue

            trade_abs = abs(trade_values[i])

            if trade_values[i] > 0:                          # COMPRA
                cost = trade_abs * (1 + self.transaction_cost)
                cost = min(cost, self.balance)               # límite por cash
                shares_bought = cost / price_vec[i]
                self.balance -= cost
                self.shares[i] += shares_bought

            else:                                            # VENTA
                shares_to_sell = min(self.shares[i],
                                    trade_abs / price_vec[i])
                revenue = shares_to_sell * price_vec[i] * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares[i] -= shares_to_sell

            # contadores
            self.trade_count   += 1
            self.volume_traded += trade_abs
            self.cost_paid     += trade_abs * self.transaction_cost

        # ── 5. Avanza al siguiente día ─────────────────────────────────────
        self.step_idx += 1
        next_price_vec = self.prices.iloc[self.step_idx].values.astype(float)

        # valor neto después de rebalancear
        position_values = self.shares * next_price_vec
        new_net_worth   = self.balance + position_values.sum()
        reward          = (new_net_worth - self.net_worth) / self.net_worth
        self.net_worth  = new_net_worth
        self.net_worth_history.append(new_net_worth)

        # pesos efectivos tras el paso (para análisis)
        weights_now = position_values / new_net_worth if new_net_worth > 0 else np.zeros(3)
        self.weights_history.append(weights_now.copy())

        done = self.step_idx >= len(self.prices) - 2        # -2 porque ya miramos next

        info = {
            "net_worth":      new_net_worth,
            "trade_count":    self.trade_count,
            "volume_traded":  self.volume_traded,
            "cost_paid":      self.cost_paid,
            "weights":        weights_now,
        }

        return self._get_obs(), float(reward), done, False, info


    def _get_obs(self):
        price_vec = self.prices.iloc[self.step_idx].values.astype(float)
        net_worth = self.balance + np.dot(self.shares, price_vec)
        weights_now  = (self.shares * price_vec) / net_worth \
                       if net_worth > 0 else np.zeros(3)
        self.weights_history.append(weights_now.copy())

        balance_norm = self.balance / net_worth if net_worth > 0 else 0.0
        feat_vec     = self.features.iloc[self.step_idx].values.astype(float)
        return np.concatenate(([balance_norm], weights_now, feat_vec)
                              ).astype(np.float32)
# ──────────────────────────  Métricas genéricas  ──────────────────────────
def compute_metrics(net_worth: list[float],
                    trade_count: int,
                    volume_traded: float,
                    cost_paid: float) -> dict[str, float]:
    arr   = np.asarray(net_worth)
    daily = np.diff(arr) / arr[:-1]

    total_ret = arr[-1] / arr[0] - 1
    ann_ret   = (1 + total_ret) ** (252 / len(arr)) - 1
    vol       = np.std(daily) * np.sqrt(252)
    sharpe    = ann_ret / vol if vol > 0 else 0
    max_dd    = np.min((arr - np.maximum.accumulate(arr))
                       / np.maximum.accumulate(arr))

    return dict(total_return   = total_ret*100,
                annual_return  = ann_ret*100,
                sharpe_ratio   = sharpe,
                max_drawdown   = max_dd*100,
                volatility     = vol*100,
                trades         = trade_count,
                volume_traded  = volume_traded,
                cost_paid      = cost_paid)
def rollout_sb3(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
    return compute_metrics(env.net_worth_history,
                           info["trade_count"],
                           info["volume_traded"],
                           info["cost_paid"])




# ─────────────────────────────  Entrenamiento PPO  ────────────────────────
def train_ppo(train_df: pd.DataFrame, total_timesteps: int = 10_000):
    train_env = DummyVecEnv([lambda:
        MultiAssetTradingEnv(train_df,
                             build_feature_frame(train_df))])

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128, nhead=4, num_layers=3)
    )
    model = PPO("MlpPolicy", train_env,
                learning_rate=3e-5, n_steps=2048, batch_size=512,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=1e-4, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=policy_kwargs, verbose=1,
                tensorboard_log="runs/ppo_multi_asset")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs("trained_models", exist_ok=True)
    model.save("trained_models/ppo_multi_asset")
    return model

# ─────────────────────────────  LSTM Allocator  ───────────────────────────
class LSTMAllocator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Softmax(dim=-1))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

# ─────────────────────────  ENTRENAMIENTO LSTM  ──────────────────────────
def train_lstm(train_df: pd.DataFrame,
               seq_len: int = 30,
               epochs: int = 300,
               patience_max: int = 10,
               val_split: float = 0.2) -> nn.Module:
    """
    Entrena el LSTM con early-stopping:
    • patience_max   = nº de épocas sin mejora antes de parar.
    • val_split      = fracción del set para validación (shuffle estratificado).
    """
    feats  = build_feature_frame(train_df)
    prices = train_df[PRICE_COLS].values.astype(float)

    # ── Construye secuencias y targets
    X, y = [], []
    for i in range(seq_len, len(feats)):
        X.append(feats.iloc[i-seq_len:i].values)
        r = (prices[i] - prices[i-1]) / prices[i-1]
        w = np.maximum(r, 0)
        if w.sum() == 0:
            w[2] = 1.0
        y.append(w / w.sum())

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.stack(y), dtype=torch.float32)

    # ── Split train / val ──────────────────────────────────────────────
    n = len(X)
    idx = torch.randperm(n)
    val_size = int(val_split * n)
    train_idx, val_idx = idx[val_size:], idx[:val_size]

    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]),
                              batch_size=512, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X[val_idx],   y[val_idx]),
                              batch_size=1024, shuffle=False)

    # ── Modelo y optimizador ───────────────────────────────────────────
    model = LSTMAllocator(input_size=X.shape[2])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, epochs + 1):
        # —— Train step
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            opt.step()

        # —— Validation step
        model.eval()
        with torch.no_grad():
            val_losses = [F.mse_loss(model(xb), yb).item() for xb, yb in val_loader]
        val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch:3d}: val_loss = {val_loss:.6f}")

        # —— Early-stopping check
        if val_loss < best_val - 1e-4:          # mejora mínima 0.0001
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), "trained_models/lstm_best.pth")
        else:
            patience += 1
            if patience > patience_max:
                print(f"🛑  Early stop at epoch {epoch} — best val_loss {best_val:.6f}")
                break

    # carga el mejor peso antes de devolver
    model.load_state_dict(torch.load("trained_models/lstm_best.pth"))
    return model



def rollout_lstm(model: nn.Module, env: MultiAssetTradingEnv,
                 seq_len: int = 30):
    """
    Simula el LSTM usando las features almacenadas en env.features
    (misma dimensionalidad que en entrenamiento).
    """
    obs, _ = env.reset()
    done = False
    feat_dim = env.features.shape[1]
    feats_hist = []

    while not done:
        # vector de features del día actual (dim = feat_dim)
        feat_vec = env.features.iloc[env.step_idx].values.astype(np.float32)
        feats_hist.append(feat_vec)

        if len(feats_hist) < seq_len:
            # aún no hay historial suficiente → cartera igual-ponderada
            action = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        else:
            x = torch.tensor(np.stack(feats_hist[-seq_len:])[None],
                             dtype=torch.float32)
            with torch.no_grad():
                action = model(x).squeeze().numpy()

        # avanzamos un paso
        _, _, done, _, info = env.step(action)

    return compute_metrics(env.net_worth_history,
                           info["trade_count"],
                           info["volume_traded"],
                           info["cost_paid"])




# ────────────────────────  Back-test y métricas  ─────────────────────────
# ─────────────────  Buy & Hold (1/3-1/3-1/3)  ──────────────────
def buy_hold_equal_weight(data: pd.DataFrame,
                          initial_balance: float = INITIAL_BALANCE):
    """
    Calcula métricas de un portafolio buy-and-hold igual-ponderado.
    """
    # 1️⃣  Aplana MultiIndex y simplifica nombres a 'activo_Close'
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.map(
            lambda tup: "_".join(map(str, tup)) if isinstance(tup, tuple) else str(tup)
        )
    data.columns = data.columns.map(lambda c: "_".join(c.split("_")[:2]))

    # 2️⃣  Asegura que las columnas Close existan
    missing = [c for c in PRICE_COLS if c not in data.columns]
    if missing:
        raise KeyError(f"Buy&Hold: columnas faltantes {missing}")

    prices = data[PRICE_COLS].values.astype(float)
    shares = (initial_balance / 3) / prices[0]          # 1/3,1/3,1/3
    portfolio = (shares * prices).sum(axis=1)

    total_ret = portfolio[-1] / portfolio[0] - 1
    ann_ret   = (1 + total_ret) ** (252 / len(portfolio)) - 1
    daily     = np.diff(portfolio) / portfolio[:-1]
    vol       = np.std(daily) * np.sqrt(252)
    sharpe    = ann_ret / vol if vol > 0 else 0
    max_dd    = np.min((portfolio - np.maximum.accumulate(portfolio))
                       / np.maximum.accumulate(portfolio))

    return dict(total_return   = total_ret*100,
                annual_return  = ann_ret*100,
                sharpe_ratio   = sharpe,
                max_drawdown   = max_dd*100,
                volatility     = vol*100,
                trades         = 0)


def test_model(model: PPO, env: gym.Env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
    final = info["net_worth"]
    days  = len(env.net_worth_history)
    total_ret = final / INITIAL_BALANCE - 1
    ann_ret   = (1 + total_ret) ** (252 / days) - 1
    daily     = np.diff(env.net_worth_history) / env.net_worth_history[:-1]
    vol       = np.std(daily) * np.sqrt(252)
    sharpe    = ann_ret / vol if vol > 0 else 0
    max_dd    = np.min((np.array(env.net_worth_history)
                        - np.maximum.accumulate(env.net_worth_history))
                       / np.maximum.accumulate(env.net_worth_history))
    return dict(total_return  = total_ret*100,
                annual_return = ann_ret*100,
                sharpe_ratio  = sharpe,
                max_drawdown  = max_dd*100,
                volatility    = vol*100)
def plot_weight_share(weights: list[np.ndarray],
                      asset_names=("S&P500", "Bitcoin", "Bono"),
                      tag="ppo"):
    """
    Dibuja la serie temporal de los pesos asignados a cada activo.
    Guarda weights_<tag>.png.
    """
    import matplotlib.pyplot as plt, numpy as np
    w = np.vstack(weights)                       # shape (T, 3)

    plt.figure(figsize=(10, 4))
    for i, name in enumerate(asset_names):
        plt.plot(w[:, i], label=name)

    plt.title(f"Pesos diarios – {tag.upper()}")
    plt.xlabel("Días")                           # ← eje X
    plt.ylabel("Peso en portafolio")             # ← eje Y
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"weights_{tag}.png", dpi=150)
    plt.close()
def plot_weight_means(stats: dict[str, np.ndarray],
                      asset_names=("S&P500", "Bitcoin", "Bono")):
    """
    stats  = {"PPO": mean_w, "LSTM": mean_w, ...}
    mean_w = np.array([w_sp500, w_btc, w_bono])
    Genera weight_means.png con barras agrupadas.
    """
    import matplotlib.pyplot as plt, numpy as np
    models = list(stats.keys())
    x = np.arange(len(asset_names))              # posiciones eje X
    width = 0.8 / len(models)                    # ancho de cada barra

    plt.figure(figsize=(6, 4))
    for idx, model in enumerate(models):
        plt.bar(x + idx * width, stats[model],
                width=width, label=model)

    plt.title("Peso medio por activo")
    plt.xlabel("Activo")                         # ← eje X
    plt.ylabel("Peso medio")                     # ← eje Y
    plt.xticks(x + width * (len(models)-1)/2, asset_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("weight_means.png", dpi=150)
    plt.close()


# ─────────────────────────────────  MAIN  ──────────────────────────────────
def main():
    print("🏁  MULTI-ASSET TRAINING & TEST")
    raw = load_multi_asset_data()
    train_df, _, test_df = split_data(raw)

    # ── Genera features de entrenamiento y guarda columnas referencia
    train_feats = build_feature_frame(train_df)       # (57 columnas)
    train_cols  = train_feats.columns

    # ── ENTRENAR PPO ─────────────────────────────────────────────────────
    print("Training PPO...")
    ppo_model = train_ppo(train_df, total_timesteps=1_000_000)
    ppo_env   = MultiAssetTradingEnv(
        test_df,
        build_feature_frame(test_df, ref_cols=train_cols)   # asegura mismas 57 columnas
    )
    ppo_met   = rollout_sb3(ppo_model, ppo_env)
    ppo_curve = ppo_env.net_worth_history

    # ── ENTRENAR LSTM ────────────────────────────────────────────────────
    print("Training LSTM...")
    lstm_model = train_lstm(train_df, seq_len=30, epochs=300)
    lstm_env   = MultiAssetTradingEnv(
        test_df,
        build_feature_frame(test_df, ref_cols=train_cols)
    )
    lstm_met   = rollout_lstm(lstm_model, lstm_env, seq_len=30)
    lstm_curve = lstm_env.net_worth_history
    summarize_weights(ppo_env.weights_history, tag="ppo")
    summarize_weights(lstm_env.weights_history, tag="lstm")
    
    # ── BUY & HOLD igual-ponderado ───────────────────────────────────────
    bh_met = buy_hold_equal_weight(test_df)
    bh_curve = (
        (1/3) * test_df["sp500_Close"].values +
        (1/3) * test_df["btc_Close"].values   +
        (1/3) * test_df["bond_Close"].values
    ).tolist()
    # ── Guardar curvas de pesos diarios ──
    plot_weight_share(ppo_env.weights_history, tag="ppo")
    plot_weight_share(lstm_env.weights_history, tag="lstm")

    # ── Barras de pesos medios ──
    weight_means_stats = {
        "PPO":  np.vstack(ppo_env.weights_history).mean(axis=0),
        "LSTM": np.vstack(lstm_env.weights_history).mean(axis=0),
    }
    plot_weight_means(weight_means_stats)

    # ── GRÁFICOS ─────────────────────────────────────────────────────────
    plot_equity_curves({
        "Buy&Hold": bh_curve,
        "PPO":      ppo_curve,
        "LSTM":     lstm_curve
    })

    allm = {"Buy&Hold": bh_met, "PPO": ppo_met, "LSTM": lstm_met}
    for k, (ttl, yl, fn) in {
        "annual_return": ("Retorno anual (%)", "%",     "annual_ret.png"),
        "sharpe_ratio":  ("Sharpe ratio",       "",     "sharpe.png"),
        "trades":        ("# transacciones",    "count","trades.png"),
    }.items():
        plot_metric_bars(allm, k, ttl, yl, fn)

    # ── RESUMEN en consola ───────────────────────────────────────────────
    for n, m in allm.items():
        print(f"{n:9s} | Ret {m['annual_return']:.2f}% "
              f"| Sharpe {m['sharpe_ratio']:.2f} "
              f"| MaxDD {m['max_drawdown']:.2f}% "
              f"| Trades {m['trades']}")



if __name__ == "__main__":
    main()
