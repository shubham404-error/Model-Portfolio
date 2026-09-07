# streamlit run model_portfolio_india_app.py
# streamlit run myapp13.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(
    page_title="CapitalSense Portfolio Terminal",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&display=swap');

.stApp {
    background-color: #FAF9F6;
}

* {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] p {
    color: #0F172A;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Clash Display', sans-serif !important;
}

h1 { font-size: clamp(24px, 5vw, 40px) !important; }
h2 { font-size: clamp(20px, 4vw, 32px) !important; }
h3 { font-size: clamp(18px, 3vw, 24px) !important; }

.block-container {
    max-width: 1200px !important;
    padding-top: 2rem !important;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 0.5rem;
    padding: 1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

div[data-testid="stMetricValue"] {
    color: #4C7766 !important;
}

.css-pill {
    background-color: #e9ecef;
    color: #1a2332;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-right: 10px;
    margin-bottom: 10px;
    border: 1px solid #dee2e6;
}
.css-pill.status-bull { background-color: #d1e7dd; color: #0f5132; border-color: #badbcc; }
.css-pill.status-bear { background-color: #f8d7da; color: #842029; border-color: #f5c2c7; }
</style>
""", unsafe_allow_html=True)

st.title("📈 CapitalSense Portfolio Terminal — India")

# =========================
# INPUTS
# =========================
st.sidebar.header("Portfolio Builder")

tickers = st.sidebar.text_input(
    "Tickers (NSE, use .NS suffix)",
    "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS"
)

ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

start_date = st.sidebar.date_input(
    "Start Date",
    pd.Timestamp("2023-01-01")
)

# =========================
# WEIGHTS
# =========================
st.sidebar.subheader("Weights")

if 'weight_df' not in st.session_state or set(st.session_state.weight_df['Ticker']) != set(ticker_list):
    st.session_state.weight_df = pd.DataFrame({
        "Ticker": ticker_list,
        "Weight": [1.0/len(ticker_list)] * len(ticker_list)
    })

if st.sidebar.button("Auto-Normalize to 100%"):
    total = st.session_state.weight_df['Weight'].sum()
    if total > 0:
        st.session_state.weight_df['Weight'] = st.session_state.weight_df['Weight'] / total

edited_df = st.sidebar.data_editor(
    st.session_state.weight_df,
    hide_index=True,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", disabled=False),
        "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
    },
    key="weight_editor"
)

st.session_state.weight_df = edited_df
ticker_list = edited_df['Ticker'].tolist()

total_weight = edited_df['Weight'].sum()
st.sidebar.metric("Total Weight", f"{total_weight*100:.1f}%")

if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning("Weights do not sum to 100%. Please Auto-Normalize.")

weights = edited_df['Weight'].values
if weights.sum() == 0:
    weights = np.ones(len(weights)) / len(weights)
else:
    weights = weights / weights.sum()

allocation = pd.DataFrame({
    "Ticker": ticker_list,
    "Weight": weights
})

# =========================
# DATA
# =========================
with st.spinner("Fetching market data and computing metrics..."):
    data = yf.download(
        ticker_list,
        start=start_date,
        auto_adjust=True
    )["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = ticker_list

    data = data.dropna(how="all")

    spy = yf.download("^NSEI", start=start_date, auto_adjust=True)["Close"].squeeze()
    
    factor_prices = yf.download(
        ["^NSEI", "^NSEBANK", "^NSEMDCP50", "^INDIAVIX"],
        start=start_date,
        auto_adjust=True
    )["Close"]

# =========================
# PORTFOLIO ENGINE
# =========================
returns = data.pct_change().dropna()
portfolio_returns = returns.dot(weights)

cumulative = (1 + portfolio_returns).cumprod()

# =========================
# PORTFOLIO OPTIMIZATION (MAX SHARPE)
# =========================

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

num_assets = len(ticker_list)

# =========================
# INDIVIDUAL ASSET STATS
# =========================

asset_stats = pd.DataFrame({
    "Ticker": ticker_list,
    "Return": mean_returns.values,
    "Volatility": np.sqrt(np.diag(cov_matrix))
})

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.065):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (returns - risk_free_rate) / volatility if volatility > 0 else 0
    return returns, volatility, sharpe

def neg_sharpe(weights):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.065)[2]

# constraints
constraints = (
    {"type": "eq", "fun": lambda x: np.sum(x) - 1}
)

# bounds (long-only + optional cap)
max_weight = 0.5
bounds = tuple((0, max_weight) for _ in range(num_assets))

# initial guess
init_guess = np.array([1/num_assets] * num_assets)

opt_result = minimize(
    neg_sharpe,
    init_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

opt_weights = opt_result.x

opt_return, opt_vol, opt_sharpe = portfolio_performance(
    opt_weights, mean_returns, cov_matrix
)

optimized_returns = returns.dot(opt_weights)
optimized_cum = (1 + optimized_returns).cumprod()

# =========================
# MONTE CARLO PORTFOLIOS
# =========================

n_portfolios = 5000

mc_returns = []
mc_vols = []
mc_sharpes = []
mc_weights = []

for _ in range(n_portfolios):

    w = np.random.random(len(ticker_list))
    w /= w.sum()

    ret = np.sum(mean_returns * w)

    vol = np.sqrt(
        np.dot(w.T, np.dot(cov_matrix, w))
    )

    sharpe_mc = (ret - 0.065) / vol if vol > 0 else 0

    mc_returns.append(ret)
    mc_vols.append(vol)
    mc_sharpes.append(sharpe_mc)
    mc_weights.append(w)

mc_df = pd.DataFrame({
    "Return": mc_returns,
    "Volatility": mc_vols,
    "Sharpe": mc_sharpes
})

max_sharpe_idx = mc_df["Sharpe"].idxmax()

# =========================
# SPY BENCHMARK
# =========================
spy_returns = spy.pct_change().dropna()

# align to portfolio
aligned = pd.concat([portfolio_returns, spy_returns], axis=1, join="inner")
aligned.columns = ["portfolio", "spy"]

portfolio_aligned = aligned["portfolio"]
spy_aligned = aligned["spy"]

spy_cum = (1 + spy_aligned).cumprod()
cumulative = (1 + portfolio_aligned).cumprod()

# risk metrics for SPY
spy_vol = spy_returns.rolling(20).std() * np.sqrt(252)
spy_drawdown = spy_cum / spy_cum.cummax() - 1

# =========================
# FACTOR DATA
# =========================

factor_returns = factor_prices.pct_change().dropna()

# =========================
# FACTOR BETAS
# =========================

factor_betas = {}

for factor in ["^NSEI", "^NSEBANK", "^NSEMDCP50", "^INDIAVIX"]:

    aligned_factor = pd.concat(
        [portfolio_returns, factor_returns[factor]],
        axis=1
    ).dropna()

    cov = aligned_factor.iloc[:,0].cov(aligned_factor.iloc[:,1])
    var = aligned_factor.iloc[:,1].var()

    factor_betas[factor] = cov / var

factor_beta_df = pd.DataFrame(
    factor_betas.items(),
    columns=["Factor", "Beta"]
)

# =========================
# CAPM ALPHA
# =========================

market_beta = factor_betas["^NSEI"]
rf_rate = 0.065
annualized_portfolio_return = portfolio_returns.mean() * 252
annualized_market_return = factor_returns["^NSEI"].mean() * 252

capm_alpha = (
    annualized_portfolio_return
    - (rf_rate + market_beta * (annualized_market_return - rf_rate))
)

treynor = (
    (annualized_portfolio_return - rf_rate)
    / market_beta if market_beta != 0 else 0
)

# =========================
# REGIME SWITCHING ENGINE
# =========================

spy_price = spy

# trend signals
spy_ma50 = spy_price.rolling(50).mean()
spy_ma200 = spy_price.rolling(200).mean()

trend_signal = (spy_ma50 > spy_ma200).astype(int)

# momentum signal (12D)
spy_mom = spy_price.pct_change(12)
mom_signal = (spy_mom > 0).astype(int)

# volatility signal (risk-off when high vol)
spy_vol_regime = spy_returns.rolling(20).std() * np.sqrt(252)
vol_threshold = spy_vol_regime.quantile(0.7)

vol_signal = (spy_vol_regime < vol_threshold).astype(int)

# align
regime_df = pd.concat([trend_signal, mom_signal, vol_signal], axis=1).dropna()
regime_df.columns = ["trend", "momentum", "vol"]

regime_score = regime_df.mean(axis=1)

# final regime
regime_state = np.where(regime_score > 0.66, "BULL",
                 np.where(regime_score < 0.33, "BEAR", "NEUTRAL"))

spy_ma200 = factor_prices["^NSEI"].rolling(200).mean()
qqq_ma200 = factor_prices["^NSEBANK"].rolling(200).mean()

risk_on = (
    (factor_prices["^NSEI"] > spy_ma200)
    & (factor_prices["^NSEBANK"] > qqq_ma200)
    & (factor_prices["^INDIAVIX"] < 15)
)

current_regime = (
    "RISK ON"
    if risk_on.iloc[-1]
    else "RISK OFF"
)

# =========================
# ALIGN SPY WITH REGIME
# =========================

spy_df = pd.DataFrame({
    "NIFTY": spy_price.reindex(regime_df.index),
})

spy_df["Regime"] = regime_state[:len(spy_df)]
spy_df = spy_df.dropna()

# =========================
# REGIME ALLOCATION ENGINE
# =========================

def equal_weight(n):
    return np.array([1/n] * n)

eq_weights = equal_weight(len(ticker_list))

# define allocations
bull_weights = opt_weights          # from optimizer
bear_weights = np.zeros(len(ticker_list))
# fallback defensive = SPY proxy (equal-weight proxy if needed)
neutral_weights = eq_weights

strategy_returns = []

for i in range(len(portfolio_aligned)):
    if regime_state[i] == "BULL":
        w = bull_weights
    elif regime_state[i] == "BEAR":
        w = bear_weights
    else:
        w = neutral_weights

    r = np.dot(returns.iloc[i], w)
    strategy_returns.append(r)

strategy_returns = pd.Series(strategy_returns, index=portfolio_aligned.index)
strategy_cum = (1 + strategy_returns).cumprod()

# =========================
# METRICS
# =========================
portfolio_total_return = float((cumulative.iloc[-1] - 1) * 100)
spy_total_return = float((spy_cum.iloc[-1] - 1) * 100)

alpha = float(portfolio_total_return - spy_total_return)

# =========================
# RISK ENGINE (PORTFOLIO)
# =========================
corr = returns.corr()
rolling_vol = portfolio_returns.rolling(20).std() * np.sqrt(252)
drawdown = cumulative / cumulative.cummax() - 1

# =========================
# ROLLING BETA ENGINE
# =========================
rolling_window = 20

cov = portfolio_aligned.rolling(rolling_window).cov(spy_aligned)
var = spy_aligned.rolling(rolling_window).var()

rolling_beta = cov / var
rolling_beta = rolling_beta.dropna()

# rolling alpha (excess returns)
rolling_alpha = (
    portfolio_aligned.rolling(rolling_window).mean()
    - rolling_beta * spy_aligned.rolling(rolling_window).mean()
)

# =========================
# FACTOR ENGINE (MARKET + MOMENTUM)
# =========================

# --------
# MOMENTUM FACTOR (SPY PROXY)
# --------
momentum_window = 12

spy_momentum = spy_aligned.pct_change(momentum_window)

# align everything
factor_df = pd.concat([
    portfolio_aligned,
    spy_aligned,
    spy_momentum
], axis=1).dropna()

# =========================
# ATTRIBUTION vs SPY
# =========================

active_returns = portfolio_aligned - spy_aligned
attribution = pd.DataFrame({
    "Active Return": active_returns
})

attribution["Cumulative Active"] = (1 + attribution["Active Return"]).cumprod() - 1

factor_df.columns = ["portfolio", "market", "momentum"]

rolling_window = 60

beta_matrix = pd.DataFrame()

spy_factor = factor_returns["^NSEI"]

for ticker in ticker_list:

    aligned = pd.concat(
        [returns[ticker], spy_factor],
        axis=1
    ).dropna()

    beta = (
        aligned.iloc[:,0]
        .rolling(rolling_window)
        .cov(aligned.iloc[:,1])
        /
        aligned.iloc[:,1]
        .rolling(rolling_window)
        .var()
    )

    beta_matrix[ticker] = beta

# --------
# ROLLING MOMENTUM EXPOSURE
# --------
rolling_mom_corr = factor_df["portfolio"].rolling(20).corr(factor_df["momentum"])

# --------
# ROLLING MARKET BETA (already exists but recomputed cleanly here for consistency)
# --------
rolling_mkt_beta = (
    factor_df["portfolio"].rolling(20).cov(factor_df["market"])
    / factor_df["market"].rolling(20).var()
)

rolling_mkt_beta = rolling_mkt_beta.dropna()
rolling_mom_corr = rolling_mom_corr.dropna()

# =========================
# STRATEGY ENGINE
# =========================
signal_list = []

for t in ticker_list:
    px_data = data[t].dropna()

    ma20 = px_data.rolling(20).mean()
    ma50 = px_data.rolling(50).mean()

    signal_list.append((ma20 > ma50).astype(int))

signal_df = pd.concat(signal_list, axis=1)
signal_df.columns = ticker_list
signal_df = signal_df.dropna()

portfolio_signal_series = signal_df.mean(axis=1)
final_signal = "BUY" if portfolio_signal_series.iloc[-1] > 0.5 else "SELL"

# =========================
# METRICS
# =========================
volatility = float(portfolio_returns.std() * np.sqrt(252) * 100)

annualized_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe = 0.0 if annualized_volatility == 0 else float(
    ((portfolio_returns.mean() * 252) - 0.065) / annualized_volatility
)

beta_latest = float(rolling_beta.dropna().iloc[-1])

mkt_beta_latest = float(rolling_mkt_beta.dropna().iloc[-1])
mom_exp_latest = float(rolling_mom_corr.dropna().iloc[-1])

# =========================
# SUMMARY STRIP
# =========================
regime_css_class = "status-bull" if current_regime == "RISK ON" else "status-bear"
regime_icon = "🟢" if current_regime == "RISK ON" else "🔴"

st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <span class="css-pill">🇮🇳 India Equities</span>
        <span class="css-pill {regime_css_class}">{regime_icon} {current_regime}</span>
        <span class="css-pill">Avg Beta: {beta_latest:.2f}</span>
    </div>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio",
    "⚠️ Risk",
    "📡 Signals",
    "🔁 Backtest",
    "📉 Factors"
])

# =========================
# TAB 1
# =========================
with tab1:

    st.subheader("Portfolio Engine")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Portfolio Return", f"{portfolio_total_return:.2f}%")
    col2.metric("NIFTY Return", f"{spy_total_return:.2f}%")
    col3.metric("Alpha vs NIFTY", f"{alpha:.2f}%")
    col4.metric("Sharpe", f"{sharpe:.2f}")
    st.metric("Rolling Beta (NIFTY)", f"{beta_latest:.2f}")
    st.metric("Market Beta", f"{mkt_beta_latest:.2f}")
    st.metric("Momentum Exposure", f"{mom_exp_latest:.2f}")
    st.metric("Optimized Sharpe", f"{opt_sharpe:.2f}")
    st.metric("Optimized Return", f"{opt_return*100:.2f}%")
    st.metric("Optimized Volatility", f"{opt_vol*100:.2f}%")
    st.metric("CAPM Alpha",f"{capm_alpha*100:.2f}%")
    st.metric("Treynor Ratio",f"{treynor:.2f}")
    st.metric("Regime",current_regime)

    st.plotly_chart(
        px.pie(allocation, names="Ticker", values="Weight", title="Allocation"),
        use_container_width=True
    )

    st.subheader("Optimized Portfolio (Max Sharpe)")

    opt_allocation = pd.DataFrame({
    "Ticker": ticker_list,
    "Weight": opt_weights
    })

    st.plotly_chart(
    px.pie(opt_allocation, names="Ticker", values="Weight", title="Optimized Allocation"),
    use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative,
        name="Portfolio"
    ))

    fig.add_trace(go.Scatter(
        x=spy_cum.index,
        y=spy_cum,
        name="NIFTY 50 Benchmark"
    ))

    fig_opt = go.Figure()

    fig_opt.add_trace(go.Scatter(
    x=cumulative.index,
    y=cumulative,
    name="Current Portfolio"
    ))

    # -------------------------
    # Individual Assets
    # -------------------------
    fig_opt.add_trace(
        go.Scatter(
        x=asset_stats["Volatility"],
        y=asset_stats["Return"],
        mode="markers+text",
        text=asset_stats["Ticker"],
        textposition="top center",
        marker=dict(
            size=14,
            symbol="diamond"
        ),
        name="Assets"
        )
    )

    fig_opt.add_trace(go.Scatter(
    x=optimized_cum.index,
    y=optimized_cum,
    name="Max Sharpe Portfolio"
    ))

    fig_opt.add_trace(go.Scatter(
    x=spy_cum.index,
    y=spy_cum,
    name="NIFTY"
    ))

    fig_opt.update_layout(
    title="Optimized vs Current vs NIFTY",
    yaxis_title="Growth of ₹1"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # EFFICIENT FRONTIER
    # =========================

    fig_frontier = px.scatter(
    mc_df,
    x="Volatility",
    y="Return",
    color="Sharpe",
    title="Monte Carlo Efficient Frontier"
    )
    fig_frontier.update_traces(
    marker=dict(opacity=0.3)
    )

    # -------------------------
    # Individual Assets
    # -------------------------
    fig_frontier.add_trace(
    go.Scatter(
        x=asset_stats["Volatility"],
        y=asset_stats["Return"],
        mode="markers+text",
        text=asset_stats["Ticker"],
        textposition="top center",
        textfont=dict(
            color="black",size=14   # ~2x larger
        ),
        marker=dict(
            size=14,
            symbol="diamond",
            color="black",
            line=dict(width=1)
        ),
        name="Assets"
    )
    )

    # optimized portfolio
    fig_frontier.add_trace(
        go.Scatter(
        x=[opt_vol],
        y=[opt_return],
        mode="markers",
        marker=dict(
            size=32,color="red",
            symbol="star"
        ),
        name="Max Sharpe"
    )
    )

    fig_frontier.update_layout(
    coloraxis_colorbar=dict(
        title="Sharpe",
        x=1.15,      # move further right
        y=0.5,
        len=0.75
        )
    )

    st.plotly_chart(
    fig_frontier,
    use_container_width=True
    )

# =========================
# TAB 2 (RISK)
# =========================
with tab2:

    st.subheader("Risk Engine")

    st.plotly_chart(
        px.imshow(corr, text_auto=True, title="Correlation Matrix"),
        use_container_width=True
    )

    # =========================
    # DRAWDOWN
    # =========================
    fig_dd = go.Figure()

    fig_dd.add_trace(go.Scatter(
        x=cumulative.index,
        y=drawdown,
        name="Portfolio Drawdown"
    ))

    fig_dd.add_trace(go.Scatter(
        x=spy_drawdown.index,
        y=spy_drawdown,
        name="NIFTY Drawdown"
    ))

    fig_dd.update_layout(
        title="Drawdown Comparison",
        yaxis_title="Drawdown"
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    # =========================
    # VOLATILITY
    # =========================
    fig_vol = go.Figure()

    fig_vol.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        name="Portfolio Volatility"
    ))

    fig_vol.add_trace(go.Scatter(
        x=spy_vol.index,
        y=spy_vol,
        name="NIFTY Volatility"
    ))

    fig_vol.update_layout(
        title="Rolling Volatility (20D Annualized)",
        yaxis_title="Volatility"
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    # =========================
    # ROLLING BETA
    # =========================
    fig_beta = go.Figure()

    fig_beta.add_trace(go.Scatter(
        x=rolling_beta.index,
        y=rolling_beta,
        name="Rolling Beta (20D)"
    ))

    fig_beta.add_hline(y=1.0, line_dash="dash", line_color="gray")

    fig_beta.update_layout(
        title="Rolling Beta vs NIFTY (20D)",
        yaxis_title="Beta"
    )

    st.plotly_chart(fig_beta, use_container_width=True)

    # =========================
    # MARKET FACTOR BETA
    # =========================

    # =========================
    # MOMENTUM EXPOSURE
    # =========================
    fig_mom = go.Figure()

    fig_mom.add_trace(go.Scatter(
        x=rolling_mom_corr.index,
        y=rolling_mom_corr,
        name="Momentum Exposure (Corr)"
    ))

    fig_mom.add_hline(y=0.0, line_dash="dash", line_color="gray")

    fig_mom.update_layout(
        title="Momentum Factor Exposure (Rolling)",
        yaxis_title="Correlation"
    )

    st.plotly_chart(fig_mom, use_container_width=True)
# =========================
# TAB 3
# =========================
with tab3:

    st.subheader("Signal Engine")

    st.metric("Portfolio Signal", final_signal)

    st.line_chart(portfolio_signal_series)

    st.dataframe(signal_df.tail(20))

# =========================
# TAB 4
# =========================
with tab4:

    st.subheader("Backtest Engine")

    backtest_df = pd.DataFrame({
        "Portfolio": cumulative,
        "NIFTY": spy_cum
    }).dropna()

    st.dataframe(backtest_df.tail(10))

    st.plotly_chart(
        go.Figure([
            go.Scatter(x=backtest_df.index, y=backtest_df["Portfolio"], name="Portfolio"),
            go.Scatter(x=backtest_df.index, y=backtest_df["NIFTY"], name="NIFTY")
        ]),
        use_container_width=True
    )

    st.plotly_chart(
        px.histogram(portfolio_returns, nbins=50),
        use_container_width=True
    )

    st.subheader("Regime Switching Strategy")

    fig_reg = go.Figure()

    fig_reg.add_trace(go.Scatter(
    x=strategy_cum.index,
    y=strategy_cum,
    name="Regime Strategy"
    ))

    fig_reg.add_trace(go.Scatter(
    x=cumulative.index,
    y=cumulative,
    name="Static Portfolio"
    ))

    fig_reg.add_trace(go.Scatter(
    x=spy_cum.index,
    y=spy_cum,
    name="NIFTY"
    ))

    fig_reg.update_layout(
    title="Bull/Bear Regime Switching Strategy",
    yaxis_title="Growth of ₹1"
    )

    st.plotly_chart(fig_reg, use_container_width=True)

    regime_map = pd.DataFrame({
    "Date": regime_df.index,
    "Regime": regime_state[:len(regime_df)]
    })

    st.subheader("Market Regime")

    st.line_chart(regime_score)

    st.dataframe(regime_map.tail(20))

    # =========================
    # SPY PRICE COLORED BY REGIME
    # =========================

    fig_regime_scatter = px.scatter(
    spy_df,
    x=spy_df.index,
    y="NIFTY",
    color="Regime",
    title="NIFTY Price Regime Map (Bull / Neutral / Bear)",
    color_discrete_map={
        "BULL": "green",
        "NEUTRAL": "orange",
        "BEAR": "red"
    }
    )

    fig_regime_scatter.update_traces(mode="markers")

    fig_regime_scatter.update_layout(
    yaxis_title="NIFTY Price",
    xaxis_title="Date"
    )

    st.plotly_chart(fig_regime_scatter, use_container_width=True)
# =========================
# TAB 5 (FACTORS)
# =========================
with tab5:
    st.markdown("### Factor Analytics")
    
    with st.container(border=True):
        st.markdown("#### 📊 Core Exposures")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NIFTY Beta", f"{factor_betas['^NSEI']:.2f}")
        col2.metric("BANKNIFTY Beta", f"{factor_betas['^NSEBANK']:.2f}")
        col3.metric("MIDCAP Beta", f"{factor_betas['^NSEMDCP50']:.2f}")
        col4.metric("INDIA VIX Beta", f"{factor_betas['^INDIAVIX']:.2f}")

    with st.container(border=True):
        st.markdown("#### 🎯 Performance Attribution")
        col1, col2, col3 = st.columns(3)
        col1.metric("CAPM Alpha", f"{capm_alpha*100:.2f}%")
        col2.metric("Treynor Ratio", f"{treynor:.2f}")
        col3.metric("Market Regime", current_regime)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        with st.container(border=True):
            st.markdown("#### Factor Betas")
            fig_factor = px.bar(factor_beta_df, x="Factor", y="Beta", text="Beta")
            fig_factor.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_color="#1a2332")
            fig_factor.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
            st.plotly_chart(fig_factor, use_container_width=True)
            
    with col_c2:
        with st.container(border=True):
            st.markdown("#### Active Return vs NIFTY")
            fig_attr = go.Figure()
            fig_attr.add_trace(go.Scatter(x=attribution.index, y=attribution["Cumulative Active"], name="Active Return", fill='tozeroy', line_color="#4a5d4e"))
            fig_attr.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
            st.plotly_chart(fig_attr, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Rolling Beta Heatmap (60-Day)")
        fig_heat = px.imshow(beta_matrix.T, aspect="auto", color_continuous_scale="RdBu_r")
        fig_heat.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    latest_betas = (
        beta_matrix
        .ffill()
        .iloc[-1]
        .sort_values(ascending=False)
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        with st.container(border=True):
            st.markdown("#### 📈 Highest Beta Names")
            st.dataframe(latest_betas.head(5).to_frame("Beta"), use_container_width=True)
    with col_b2:
        with st.container(border=True):
            st.markdown("#### 📉 Lowest Beta Names")
            st.dataframe(latest_betas.tail(5).to_frame("Beta"), use_container_width=True)

