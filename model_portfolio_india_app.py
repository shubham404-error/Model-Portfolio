import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import sqlite3
import os

st.set_page_config(
    page_title="CapitalSense Portfolio Terminal",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif !important;
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
    background-color: #1e1e1e;
    border: 1px solid #333;
    border-radius: 0.5rem;
    padding: 1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

div[data-testid="stMetricValue"] {
    color: #4C7766 !important;
}

.css-pill {
    background-color: #2b2b2b;
    color: #e9ecef;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-right: 10px;
    margin-bottom: 10px;
    border: 1px solid #444;
}
.css-pill.status-bull { background-color: #0f5132; color: #d1e7dd; border-color: #badbcc; }
.css-pill.status-bear { background-color: #842029; color: #f8d7da; border-color: #f5c2c7; }
</style>
""", unsafe_allow_html=True)

st.title("📈 CapitalSense Portfolio Terminal — India")

# =========================
# STATE CLEARING (Risk Mitigation)
# =========================
def on_ticker_change():
    if "calc_run" in st.session_state:
        del st.session_state["calc_run"]
    st.cache_data.clear()

# =========================
# INPUTS
# =========================
st.sidebar.header("Portfolio Builder")

tickers = st.sidebar.text_input(
    "Tickers (NSE, use .NS suffix)",
    "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS",
    on_change=on_ticker_change
)

ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

start_date = st.sidebar.date_input(
    "Start Date",
    pd.Timestamp("2023-01-01"),
    on_change=on_ticker_change
)

st.sidebar.subheader("Weights")

if 'weight_df' not in st.session_state or set(st.session_state.weight_df['Ticker']) != set(ticker_list):
    st.session_state.weight_df = pd.DataFrame({
        "Ticker": ticker_list,
        "Weight": [1.0/len(ticker_list)] * len(ticker_list)
    })
    on_ticker_change()

if st.sidebar.button("Auto-Normalize to 100%"):
    total = st.session_state.weight_df['Weight'].sum()
    if total > 0:
        st.session_state.weight_df['Weight'] = st.session_state.weight_df['Weight'] / total
    on_ticker_change()

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

# Update list based on edits
st.session_state.weight_df = edited_df
ticker_list = edited_df['Ticker'].tolist()
weights = edited_df['Weight'].values
if weights.sum() == 0:
    weights = np.ones(len(weights)) / len(weights)
else:
    weights = weights / weights.sum()

total_weight = edited_df['Weight'].sum()
st.sidebar.metric("Total Weight", f"{total_weight*100:.1f}%")

if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning("Weights do not sum to 100%. Please Auto-Normalize.")

# =========================
# CACHED FUNCTIONS
# =========================
@st.cache_data(ttl=3600)
def fetch_data(tickers, start):
    valid_tickers = []
    failed_tickers = []
    ticker_data = {}
    
    # Try fetching each ticker gracefully (Data Fetch Failures Mitigation)
    for t in tickers:
        try:
            df = yf.download(t, start=start, auto_adjust=True)["Close"]
            if df.empty or df.dropna().empty:
                failed_tickers.append(t)
            else:
                ticker_data[t] = df
                valid_tickers.append(t)
        except Exception:
            failed_tickers.append(t)
            
    if ticker_data:
        data = pd.DataFrame(ticker_data).dropna(how="all")
    else:
        data = pd.DataFrame()
        
    try:
        spy = yf.download("^NSEI", start=start, auto_adjust=True)["Close"].squeeze()
    except:
        spy = pd.Series()
        
    try:
        factor_prices = yf.download(["^NSEI", "^NSEBANK", "^NSEMDCP50", "^INDIAVIX"], start=start, auto_adjust=True)["Close"]
    except:
        factor_prices = pd.DataFrame()

    return data, spy, factor_prices, valid_tickers, failed_tickers

@st.cache_data(ttl=3600)
def fetch_dvm_scores(tickers):
    db_path = "data/capitalsense_dvm.sqlite"
    scores = {}
    if os.path.exists(db_path):
        try:
            # Read-only mode mitigation
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            for t in tickers:
                try:
                    cursor.execute("SELECT Durability, Valuation, Momentum FROM scores WHERE Ticker=?", (t,))
                    row = cursor.fetchone()
                    if row:
                        scores[t] = {"Durability": row[0], "Valuation": row[1], "Momentum": row[2]}
                except Exception:
                    pass
            conn.close()
        except Exception:
            pass
    return scores

@st.cache_data(ttl=3600)
def optimize_portfolio(returns_df, risk_free_rate=0.065):
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    num_assets = len(returns_df.columns)
    
    def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.065):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def neg_sharpe(w):
        return -portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[2]
        
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.5) for _ in range(num_assets))
    init_guess = np.array([1/num_assets] * num_assets)
    
    opt_result = minimize(
        neg_sharpe,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return opt_result.x

# =========================
# ACTION GATE
# =========================
if st.sidebar.button("Calculate Optimal Portfolio"):
    st.session_state.calc_run = True

if not st.session_state.get("calc_run", False):
    st.info("Configure your portfolio and click 'Calculate Optimal Portfolio' to begin.")
    st.stop()

# =========================
# ENGINE EXECUTION
# =========================
with st.spinner("Fetching market data and computing metrics..."):
    data, spy, factor_prices, valid_tickers, failed_tickers = fetch_data(ticker_list, start_date)

if failed_tickers:
    st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}. They have been dropped from calculations.")

if not valid_tickers:
    st.error("No valid tickers to process.")
    st.stop()

# Align weights to valid tickers
valid_weights = []
for t in valid_tickers:
    idx = ticker_list.index(t)
    valid_weights.append(weights[idx])
valid_weights = np.array(valid_weights)
if valid_weights.sum() > 0:
    valid_weights /= valid_weights.sum()
else:
    valid_weights = np.ones(len(valid_tickers)) / len(valid_tickers)

allocation = pd.DataFrame({"Ticker": valid_tickers, "Weight": valid_weights})
returns = data.pct_change().dropna()
portfolio_returns = returns.dot(valid_weights)
cumulative = (1 + portfolio_returns).cumprod()

opt_weights = optimize_portfolio(returns, risk_free_rate=0.065)
opt_returns = returns.dot(opt_weights)
opt_cumulative = (1 + opt_returns).cumprod()

spy_returns = spy.pct_change().dropna()
aligned = pd.concat([portfolio_returns, spy_returns], axis=1, join="inner")
aligned.columns = ["portfolio", "spy"]
portfolio_aligned = aligned["portfolio"]
spy_aligned = aligned["spy"]
spy_cum = (1 + spy_aligned).cumprod()

# Core Metrics
annualized_portfolio_return = portfolio_returns.mean() * 252
annualized_volatility = portfolio_returns.std() * np.sqrt(252)
rf_rate = 0.065
sharpe = (annualized_portfolio_return - rf_rate) / annualized_volatility if annualized_volatility > 0 else 0

# CapitalSense DVM
dvm_scores = fetch_dvm_scores(valid_tickers)
dvm_text = ""
if dvm_scores:
    dur = sum(dvm_scores.get(t, {}).get("Durability", 0) * w for t, w in zip(valid_tickers, valid_weights))
    val = sum(dvm_scores.get(t, {}).get("Valuation", 0) * w for t, w in zip(valid_tickers, valid_weights))
    mom = sum(dvm_scores.get(t, {}).get("Momentum", 0) * w for t, w in zip(valid_tickers, valid_weights))
    overall = (dur + val + mom) / 3
    dvm_score_display = f"{overall:.1f}"
    dvm_text = f"Your portfolio has an average Quality score of {overall:.1f}/100, but a Valuation score of {val:.1f}/100."
else:
    dvm_score_display = "N/A"
    dvm_text = "CapitalSense scoring data not available."

# =========================
# PHASE 2: UI/UX REDESIGN (FinTech Grid)
# =========================
st.markdown("### Portfolio Vital Signs")

# Top KPI Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Annual Return", f"{annualized_portfolio_return*100:.1f}%")
col2.metric("Annualized Volatility", f"{annualized_volatility*100:.1f}%")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
col4.metric("Portfolio DVM Score", dvm_score_display)

# Phase 3: Portfolio Quality Widget
st.info(f"**Portfolio Quality Insight**: {dvm_text}")

st.markdown("<br>", unsafe_allow_html=True)

# Main Visual Grid
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("#### Allocation")
    fig_pie = px.pie(allocation, names="Ticker", values="Weight", hole=0.4)
    fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    with st.expander("View Raw Weights"):
        st.dataframe(allocation, use_container_width=True)

with col_right:
    st.markdown("#### Growth of ₹10,000")
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=cumulative.index, y=cumulative * 10000, name="Current Portfolio"))
    fig_growth.add_trace(go.Scatter(x=opt_cumulative.index, y=opt_cumulative * 10000, name="Optimized Portfolio (Max Sharpe)", line=dict(dash='dash')))
    fig_growth.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum * 10000, name="NIFTY 50 Benchmark"))
    fig_growth.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_growth, use_container_width=True)
