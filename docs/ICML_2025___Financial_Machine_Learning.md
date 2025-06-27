# Benchmarking Machine Learning Methods for Portfolio Management: Challenges and Opportunities

**Anonymous Authors**

## Abstract

Machine Learning has become a powerful tool in portfolio management over the last decade. However, certain key practical challenges are often overlooked, including market diversity, realistic transaction costs for large trades, and robust testing constraints. This work evaluates the effectiveness and scalability of machine learning methods under more realistic conditions using the stocks that make up the S&P 500 and DJIA. We analyze Reinforcement Learning, Imitation Learning, DAgger and Model-Based techniques. To the best of our knowledge, this is the first study to systematically compare all these approaches. Our findings demonstrate that the best methods outperform the annualized return and Sharpe ratio of the standard benchmarks.

## 1. Introduction

According to standard financial economic theory, investors aim to maximize their utility by achieving an optimal balance between the expected returns and the associated volatility of their investment portfolios (Fabozzi & Markowitz, 2011). To achieve this objective, diversification is essential, as it leverages the variability of asset returns (variances) and their relationships (covariances) to construct portfolios that reduce risk while maintaining expected returns, following modern portfolio theory.

In passive investment, resources are allocated across a predefined set of assets, allowing returns to compound over time based on market performance, without frequent trading or active management. However, dynamic portfolio management involves actively adjusting allocations in response to evolving market conditions, leveraging new available data.

Processing available data has become increasingly relevant, as quantitative trading, where portfolio management is a primary focus, now accounts for over 70% of trading volume in developed markets such as the U.S. and Europe, and 40% in emerging markets like China and India (Sun et al., 2023). However, as emphasized by Liu et al. (2018) and Yu et al. (2019), effective strategies can be difficult to find due to the complexity and diversity of market dynamics. Therefore, the ability to process large data quickly and minimize human bias has driven the adoption of machine learning techniques.

Despite the growing interest in machine learning methods for portfolio management, various surveys underscore the challenges in comparing different approaches effectively. This is primarily attributed to the lack of consistency across research efforts. While many studies demonstrate improvements in specific areas, they often overlook critical practical constraints that are addressed in other works.

The most frequently overlooked limitations are as follows:
- Although transaction costs are often incorporated, they ignore the price impact of large trades, which would significantly increase transaction expenses.
- Constraints designed to limit large trades are rarely applied.
- Testing periods are often arbitrarily defined and frequently too short.
- The full asset universe is rarely considered, as market representation is often limited to ETFs.
- Most recent Reinforcement Learning (RL) and Imitation Learning (IL) studies simulate fewer than 30 stocks or ETFs, reducing real-world relevance.

This work aims to evaluate the effectiveness and scalability of various general machine learning approaches with realistic settings that could be implemented in practice.

Using the stocks that constitute the Standard & Poor’s 500 Index (S&P 500) and the Dow Jones Industrial Average (DJIA), rather than exchange-traded funds (ETFs), we analyze action and weights predictions based on RL and IL. We also employ Model-Based techniques. To the best of our knowledge, this is the first study to compare all of them systematically using constituent stocks, though individual methods have been evaluated against one another.

Our results suggest that:
- It is possible to outperform benchmark indices without using any input data at all.
- Price prediction app

## 2. Related Work

The application of machine learning (ML) techniques in portfolio management has become a central theme in recent financial research. Despite having similar objectives, all of these studies focus on quite different things.

Wang et al. (2019) developed a risk-adjusted reinforcement learning (RL) method that uses the Sharpe Ratio as the reward function. Herbert (2024) integrated various ML methods for stock price prediction with post-prediction portfolio optimization strategies. In contrast, Xu et al. (2020) focus on directly predicting portfolio weights while introducing a penalty for large trades.

Liu et al. (2018) applied RL with segmented discrete actions within a classic Markov Decision Process (MDP) framework on the Dow Jones index during a bull market. Kong & So (2023) extend this approach by evaluating multiple RL methods with continuous actions across diverse environments, achieving strong results on the KOSPI30 and JPX30. Nevertheless, identifying the most promising methods and output types for further development remains challenging due to the absence of a standardized comparison framework.

Almahdi & Yang (2017) managed a small portfolio of ETFs using an RL agent with the Calmar Ratio as an alternative reward function. Yu et al. (2019) trained a model-based agent using inputs from a price forecasting model and explored noise addition for state exploration. Benhamou et al. (2020) investigated the use of noise, macroeconomic features, and partially observable states in portfolio management.

Wang et al. (2021) explored alternative reward functions, employing the negative maximum drawdown for portfolios with long and short positions. Huang & Tanaka (2022) proposed a modular approach, separating stock prediction from strategy modules to improve portability. Dong & Zheng (2023) investigated imitation learning combined with RL, incorporating expert decisions into the loss function with decreasing influence over time. Caparrini et al. (2024) used ML methods to select an optimal subset of 15 stocks, constructing an equal-weight portfolio.

Huotari et al. (2020) employed RL to predict portfolio weights, selecting the top 20 stocks during training and limiting trading to these stocks during testing. Unfortunately, alternative reward functions and state exploration techniques are evaluated in isolation, limiting meaningful comparisons of their impact and potential benefits. Moreover, scalability to portfolios with several hundred stocks remains uncertain.

### 2.1. Other Relevant Research

In addition to the studies discussed above, we highlight relevant surveys and work in quantitative trading that broaden the understanding of industry approaches.

- Dakalbab et al. (2024) review AI techniques in financial trading, categorizing them by output: price prediction, pattern recognition, portfolio weights optimization, and direct action prediction. Their work underscores the diversity of approaches and AI’s adaptability across trading scenarios.

- Sun et al. (2023) provide a structured analysis of quantitative trading tasks, categorizing them into algorithmic trading, portfolio management, order execution, and market making. They emphasize the limitations of rule-based methods and the challenges inherent in financial markets, such as noise and unpredictability.

- Borkar & Jadhav (2024) identify key gaps, such as the absence of risk pattern identification in RL systems, the limited advancement of auto-generated strategies, and insufficient knowledge transfer across related domains.

Several algorithmic trading studies illustrate the application of various action segmentation techniques, threshold-based methods, and other custom strategies (e.g., Adegboye et al., 2023; Eilers et al., 2014; Nakano et al., 2018; Dragan et al., 2019; Sermpinis et al., 2019). Other studies focus on predicting these actions using reinforcement learning agents (Chen, 2019; Ye & Schuller, 2023; Bisi et al., 2020; Zhang et al., 2020). Unfortunately, none of this research offers a comprehensive comparison of different action mappings and post-prediction strategies.

Other relevant studies include Chalvatzis & Hristu-Varsakelis (2020), which show that better price forecasting does not always translate to higher profitability. Bao & Liu (2019) provide an in-depth analysis of price impact and transaction costs, examining multi-agent RL behavior in liquidation strategies. Frazzini & Pedersen (2018) conduct a comprehensive empirical study on real-world trading costs and their trends. Finally, Glosten & Milgrom (1985) introduce a theoretical framework for trading with asymmetric information, which remains relevant today.

## 3. Preliminaries

We model the trading environment as a Markov Decision Process (MDP). An MDP is a tuple \( M = \langle S, A, r, p, \gamma \rangle \) where:

- \( S \): a finite set of states  
- \( A \): a finite set of actions  
- \( r: S \times A \times S \rightarrow \mathbb{R} \): the reward function  
- \( p(s_{t+1} \mid s_t, a_t) \): the transition probability distribution  
- \( \gamma \in (0, 1] \): the discount factor

The goal is to find an optimal policy \( \pi(a|s) \), a probability distribution over actions given a state. At each time step \( t \), the environment is in a state \( s_t \). Then, an action \( a_t \sim \pi(\cdot|s_t) \) is selected. The state transitions to \( s_{t+1} \sim p(\cdot|s_t, a_t) \), and the agent receives a reward \( r_{t+1} = r(s_t, a_t, s_{t+1}) \).

The value function of a policy \( \pi \) at state \( s \) is the expected discounted return:

\[
v^\pi(s) = \mathbb{E}^\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t = s \right]
\]

An optimal policy \( \pi^* \) maximizes value for all states:  
\[
v^{\pi^*}(s) \geq v^\pi(s), \quad \forall s \in S, \pi \in \Pi
\]

In practical applications, the state space is often too large to find the optimal policy exactly. Instead, we approximate the policy using a function approximator \( \pi_\theta(a|s) \) (e.g., a neural network). We train this policy on a set of training states \( S_{\text{train}} \subset S \) and evaluate generalization on testing states \( S_{\text{test}} \subset S \), with \( S_{\text{train}} \cap S_{\text{test}} = \emptyset \).

To learn \( \pi_\theta \), we consider four approaches:

- Reinforcement Learning (RL)  
- Imitation Learning (IL)  
- Dataset Aggregation (DAgger)  
- Model-Based Methods

## 4. The Investor’s Problem

This section defines the dynamic portfolio optimization problem, also known as the investor’s problem (Cong & Oosterlee, 2017). The investor begins with an initial budget \( B_0 \) and aims to maximize expected wealth over a finite time horizon by buying and selling stocks.

Let:

- \( I \): the set of stocks
- \( P_{it} \): the adjusted close price of stock \( i \in I \) at time \( t \)
- \( h_{it} \): the number of shares of stock \( i \) held at time \( t \)
- \( b_t \): the cash balance at time \( t \)

The total portfolio value at time \( t \) is:

\[
V_t = b_t + \sum_{i \in I} P_{it} \cdot h_{it}
\]

Assume initially \( h_{i0} = 0 \) for all \( i \in I \).

### Objective

Maximize portfolio value at final time step \( T \):

\[
\max \mathbb{E} \left[ b_T + \sum_{i \in I} P_{iT} \cdot h_{iT} \right]
\]

Subject to:

1. **Holdings update**  
\[
h_{i,t+1} = h_{it} + a_{it} \quad \forall i \in I, \ t \in \{0, \dots, T-1\}
\]

2. **Balance update**  
\[
b_{t+1} = b_t (1 + R_{f,t})^{1/d} - \delta_t
\]

3. **Transaction cost**  
\[
\delta_t = \sum_i P_{it} a_{it} (1 - c)
\]

4. **Constraints**  
\[
h_{it}, b_t \geq 0, \quad h_{i0} = 0, \quad \delta_t \in \mathbb{R}
\]

5. **Action limits**  
\[
a_{it} \in \{-K, \dots, K\}, \quad b_0 = B_0
\]

Where:

- \( a_{it} \): number of shares of stock \( i \) bought or sold at time \( t \)  
- \( K \): maximum number of shares per trade  
- \( R_{f,t} \): annualized risk-free rate at time \( t \)  
- \( d \): number of trading days in a year  
- \( c \): fixed transaction cost

This formulation accounts for realistic trading constraints such as limited liquidity and transaction costs.

## 5. Simulation Environment

Modeling the investor’s problem as a Markov Decision Process (MDP) is straightforward:

- **State**: \( s_t = (h_{it}, b_t) \), i.e., current holdings and balance  
- **Action**: \( a_t = [a_{1t}, \dots, a_{|I|t}] \), i.e., buy/sell decisions per stock  
- **Reward**: Change in portfolio value:

\[
r(s_t, a_t, s_{t+1}) = V_{t+1} - V_t
\]

Assuming no discounting (\( \gamma = 1 \)), the MDP’s solution aligns with the investor’s optimization objective.

### 5.1 Generalization and Performance Metrics

We divide the data into 3 non-overlapping periods:

- **Training**: Jan 2009 – Dec 2014  
- **Validation**: Jan 2015 – Dec 2015  
- **Testing**: Jan 2016 – July 2020

This setup captures both bull and bear markets.

Two key performance metrics are:

**Annualized Return (AR):**

\[
AR = \left( \prod_{t=1}^{T} \frac{V_t}{V_{t-1}} \right)^{d/T} - 1
\]

where \( d = 252 \) trading days/year.

**Sharpe Ratio (SR):**

\[
SR_p = \frac{AR_p - R_f}{\sigma_p}
\]

where \( \sigma_p \): annualized volatility of returns,  
and \( R_f \): annualized risk-free rate.

### 5.2 Historical Information: DJIA and S&P 500

We created two environments:

- **DJIA**: 30 well-established firms  
- **S&P 500**: 426 stocks (those continuously listed during training)

These benchmarks represent U.S. market diversity.

- **Transaction cost**: 0.2%  
- **Max shares per trade**: \( K = 1000 \)  
- **Initial balance**: \$1,000,000

### 5.3 State, Actions, and Rewards

**State features include:**

- \( b_t \), \( h_{it} \)
- Stock-specific features from Fama-French-Carhart: momentum, size, value, profitability (Novy-Marx)
- General market features: daily returns of 13-week Treasury Bills

**Actions \( a_{it} \in [-K, K] \)** are mapped via 4 strategies:

1. **Direct**: \( a'_{it} \in \{-1, 0, 1\} \)  
2. **Segmented**: \( a'_{it} \in \{-5, \dots, 5\} \)  
3. **Continuous**: \( a'_{it} \in [-1, 1] \)  
4. **Weights**: \( a'_{it} \in [0, 1] \), with \( \sum_i a'_{it} = 1 \)

## 6. Machine Learning Methods

This section discusses the different techniques used to learn policies \( \pi_\theta \) for solving the investor’s problem. We explore four classes of methods: Reinforcement Learning, Imitation Learning, Dataset Aggregation (DAgger), and Model-Based methods.

---

### Reinforcement Learning (RL)

RL methods learn policies by interacting with the environment (Sutton & Barto, 2018). We use **Proximal Policy Optimization (PPO)** (Schulman et al., 2017), a state-of-the-art algorithm that:

- Models \( \pi_\theta \) with a neural network
- Runs multiple agents in parallel
- Updates the policy by increasing the likelihood of high-reward actions

We use validation to select the best model and report test performance. Unlike prior work, we:
- Use PPO (known to generalize well)
- Incorporate richer market features (Section 5.3)

---

### Imitation Learning (IL)

IL trains a policy by mimicking expert behavior (Hussein et al., 2017). We construct a dataset \( T \) of state-action pairs \( (s, a^*) \), where \( a^* \) is what the expert would do.

To define expert supervision, we:

- Look \( w \) days into the future
- Buy stocks where the predicted return exceeds a threshold \( \theta_b \)
- Sell stocks where the predicted drop exceeds \( \theta_s \)

Formally, define:

\[
d_{it} = \frac{P_{i,t+w}}{P_{it}} - 1
\]

Then:

\[
a^*_{it} =
\begin{cases}
\min(K, \left\lfloor \frac{b_t}{P_{it}} \cdot \frac{d_{it}}{\sum_{i: d_{it} > 0} d_{it}} \right\rfloor), & \text{if } d_{it} > \theta_b \\
\max(-K, -h_{it}), & \text{if } d_{it} < \theta_s \\
0, & \text{otherwise}
\end{cases}
\]

Although rarely used in this context, some studies combine IL with RL for pretraining (e.g., Dong & Zheng, 2023).

---

### Dataset Aggregation (DAgger)

DAgger improves upon IL by expanding the training data iteratively (Ross et al., 2011):

1. Train \( \pi_\theta \) using IL
2. Deploy it and collect new states \( s_{\text{new}} \)
3. Ask the expert for action \( a_{\text{new}} \) on those states
4. Add \( (s_{\text{new}}, a_{\text{new}}) \) to \( T \), and retrain

DAgger reduces error in unfamiliar states and results in more robust policies. It hasn’t previously been applied to portfolio management.

---

### Model-Based Methods

Model-based methods learn a predictive model \( \hat{p}(s_{t+1} \mid s_t, a_t; \mu) \approx p(s_{t+1} \mid s_t, a_t) \), and then use it to plan actions.

Inspired by prior work, we:

- Train a model to predict percentage price change:  
  \[
  d_{it} = \frac{P_{i,t+w}}{P_{it}} - 1
  \]

- Define actions using fixed or variable threshold strategies:
  - **Fixed Threshold**: Buy if \( d_{it} > \theta_b \), sell if \( d_{it} < \theta_s \)
  - **Variable Threshold**: Thresholds change based on remaining balance \( b_t / V_t \)

These methods are efficient to train and can be competitive when paired with good strategies.

## 7. Performance Evaluation

We aim to evaluate the effectiveness of various ML methods for portfolio management. The primary metric is **Annualized Return (AR)**; we also report the **Sharpe Ratio (SR)** for risk-adjusted performance.

To reduce randomness and ensure robustness, all results are averages over **five independent runs** of training, validation, and testing (Kong & So, 2023).

Due to computational costs, we first evaluate all methods on the **DJIA benchmark**. Then, we test the best methods on the **S&P 500** to assess scalability.

We compare:

- Reinforcement Learning (RL)
- Imitation Learning (IL)
- DAgger (DA)
- Model-based methods (Neural Network and Random Forest)
- Each method is evaluated under the four action types from Section 5.3

---

### 7.1 Results on the DJIA Benchmark

**Table 1**: Performance (Annualized Return and Sharpe Ratio)

| ML Technique           | AR      | SR   |
|------------------------|---------|------|
| RL Direct              | 13.76%  | 0.66 |
| RL Segmented           | 11.95%  | 0.53 |
| RL Continuous          | 10.85%  | 0.49 |
| RL Weights             | 11.69%  | 0.54 |
| IL Direct              | 13.73%  | 0.71 |
| IL Segmented           | **17.21%** | **0.75** |
| IL Continuous          | 11.59%  | 0.56 |
| IL Weights             | 15.21%  | 0.73 |
| DA Direct              | 12.31%  | 0.62 |
| DA Segmented           | 15.47%  | 0.72 |
| DA Continuous          | 11.50%  | 0.52 |
| DA Weights             | 15.22%  | 0.73 |
| RF Fixed Threshold     | 0.98%   | -0.02 |
| RF Variable Threshold  | 13.97%  | 0.68 |
| NN Fixed Threshold     | 11.10%  | 0.52 |
| NN Variable Threshold  | 13.98%  | 0.64 |
| **DJIA Benchmark**     | 9.957%  | 0.57 |

**Observations:**

- **Best performer**: IL with segmented actions  
- DAgger also strong, especially with segmented/weights  
- RL and Model-Based methods outperform DJIA but are slightly weaker  
- RF (fixed threshold) underperforms significantly

---

### 7.2 Results on the S&P 500 Benchmark

We tested top methods from DJIA on the S&P 500. DAgger was excluded due to high computational costs.

**Table 2**: Performance (Annualized Return and Sharpe Ratio)

| Approach               | AR      | SR   |
|------------------------|---------|------|
| RL Direct              | 14.57%  | 0.71 |
| IL Segmented           | 14.45%  | 0.71 |
| IL Weights             | 15.56%  | 0.75 |
| DA Segmented           | 16.55%  | 0.89 |
| DA Weights             | 0%      | 0    |
| NN Variable Threshold  | **19.22%** | **0.87** |
| RF Variable Threshold  | 16.25%  | 0.73 |
| **S&P 500 Benchmark**  | 10.28%  | 0.51 |

**Conclusion:**  
All top methods scale well to larger environments. NN with variable threshold performs best.

---

### 7.3 Ablation Study

We examined RL’s behavior by running it with **dummy observations** (constant state input of 1). The agent had no state awareness but still learned via rewards.

**Key Insight**:  
RL outperforms the benchmark **even without input data**, and performs similarly to when full state information is used.

**Table 3**: DJIA - RL vs. Dummy Agent

| Method        | AR      | SR   |
|---------------|---------|------|
| RL (AR-tuned) | 13.76%  | 0.66 |
| RL (last)     | 13.21%  | 0.56 |
| Dummy (AR)    | 8.20%   | 0.33 |
| Dummy (SR)    | 11.64%  | 0.51 |
| Dummy (last)  | 12.23%  | 0.58 |
| DJIA          | 9.957%  | 0.57 |

**Table 4**: S&P 500 - RL vs. Dummy Agent

| Method        | AR      | SR   |
|---------------|---------|------|
| RL (AR-tuned) | 14.57%  | 0.71 |
| RL (last)     | 12.32%  | 0.60 |
| Dummy (AR)    | 14.36%  | 0.71 |
| Dummy (SR)    | 14.68%  | 0.72 |
| Dummy (last)  | 13.36%  | 0.66 |
| S&P 500       | 10.28%  | 0.51 |

This suggests RL may be learning **static portfolio allocations** rather than responding dynamically to market features.

## 8. Discussion

**Action Space Mapping Matters**  
Table 1 shows that using continuous actions leads to lower performance than discrete actions. This may be because:

- Continuous mappings often result in **fewer trades**, as the agent is cautious to avoid transaction costs.
- The agent may avoid "neutral" trades due to difficulty predicting exact zeros.

**Strategy Matters More Than Prediction Accuracy**  
The final rows of Table 1 highlight a key insight:

- **The trading strategy used with predictions has a greater impact on performance than the raw accuracy of the predictions.**
- Given the difficulty of consistently accurate price forecasting, efforts may be better spent on improving how models *use* predictions, rather than the predictions themselves.

**Baselines for Future Research**  
Tables 1 and 2 establish clear baselines for:

- ML techniques (RL, IL, DA, Model-Based)
- Different action spaces
- Performance relative to DJIA and S&P 500

These baselines can serve as reference points for future work aiming to improve machine learning for portfolio management.

**Challenging the Efficient Market Hypothesis (EMH)**  
The fact that machine learning strategies systematically outperform benchmark indexes suggests they may identify **inefficiencies** in the market. This offers evidence **against the strict interpretation of the EMH**.

**Scalability**  
All techniques scaled to the S&P 500. The additional asset diversity allowed for **more optimization opportunities**, but also increased computational requirements.

- Neural network input/output layer sizes grow rapidly with asset count.
- Training time on S&P 500 was over 5× longer than DJIA.

**Outperforming Without Market Data**  
Tables 3 and 4 show that agents using **no state information** can still outperform DJIA/S&P. This implies:

- Agents may be optimizing toward **static portfolios**
- There is further opportunity to **improve responsiveness** to market signals

**Cost Efficiency of ML Systems**  
Machine learning-driven strategies are **far cheaper** than traditional active management:

- No salaries, bonuses, or incentives
- Once deployed, algorithms incur minimal ongoing costs

This makes ML particularly attractive compared to active fund managers, who often command high fixed and variable compensation.

**Implication:**  
ML-based portfolio strategies can democratize investing by reducing costs and enabling **transparent**, **reproducible**, and **automated** management.

## 9. Conclusions and Future Work

We have presented a **comprehensive evaluation** of machine learning methods for dynamic portfolio management under realistic trading constraints and over long historical periods.

**Key contributions:**

- **Comparative Analysis**:  
  We compared RL, IL, DAgger, and Model-Based methods under a unified framework — the first such study to do so with constituent stocks of DJIA and S&P 500.

- **Realistic Environment**:  
  The simulation included:
  - Hundreds of stocks
  - Transaction limits and realistic trading costs
  - Multi-year testing periods
  - Action mappings commonly used in real-world systems

- **Findings**:  
  - Less commonly used methods like **DAgger** and **Model-Based approaches** performed **best overall**
  - Strategies that **used predicted price movements** effectively outperformed those that simply predicted prices
  - It’s possible to **beat the market even without using state inputs**, which questions assumptions in RL literature and EMH theory

- **Best Results**:
  - Achieved an annualized return of **19.22%** and Sharpe Ratio of **0.87**
  - This is **nearly double** the performance of the DJIA and S&P 500 benchmarks over the same test periods

---

### Future Work

Our results suggest multiple promising directions:

1. **Hybrid Architectures**:  
   Combining imitation learning for initialization with reinforcement learning for fine-tuning.

2. **Adaptive Thresholds**:  
   Using model confidence or market volatility to dynamically adjust buy/sell thresholds.

3. **Multi-Agent Systems**:  
   Studying the emergent behavior of multiple agents managing different portfolios under joint market dynamics.

4. **Online Learning**:  
   Allowing strategies to evolve continuously as new market data arrives.

5. **Risk-sensitive Objectives**:  
   Beyond Sharpe ratio, incorporating downside risk, max drawdown, and other investor preferences.

---

By providing **open, reproducible baselines**, we hope this study helps guide the development of **more reliable**, **scalable**, and **transparent** machine learning systems for portfolio management.
