import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import sqlite3
import os
import config

st.set_page_config(
    page_title="CapitalSense Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

div[data-testid="stMetricValue"] {
    color: #4C7766 !important;
    font-family: 'Clash Display', sans-serif !important;
    font-weight: 600;
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
.status-bull { background-color: #0f5132; color: #d1e7dd; border-color: #badbcc; }
.status-bear { background-color: #842029; color: #f8d7da; border-color: #f5c2c7; }

/* Custom Container Styling */
div[data-testid="stVerticalBlock"] > div[style*="border-radius: 0.5rem"] {
    background-color: #1e1e1e !important;
    border-color: #333 !important;
}
</style>
""", unsafe_allow_html=True)


def main_page():
    # =========================
    # STATE CLEARING (Risk Mitigation)
    # =========================
    def on_input_change():
        if "calc_run" in st.session_state:
            del st.session_state["calc_run"]
        st.cache_data.clear()
        
    st.title("📈 CapitalSense Portfolio Terminal — India")
    
    st.markdown("Build, analyze, and optimize your actual holdings using quantitative engines.", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # =========================
    # INPUTS (Moved to Main Page)
    # =========================
    with st.container(border=True):
        st.subheader("Holdings Input")
        st.markdown("Enter your stock symbols (without `.NS`), quantities, and average buy price. The engine will automatically fetch the latest prices to compute your current portfolio weights.", unsafe_allow_html=True)
        
        col_start, _ = st.columns([1, 2])
        with col_start:
            start_date = st.date_input(
                "Historical Engine Start Date",
                pd.Timestamp("2023-01-01"),
                on_change=on_input_change,
                help="The date from which the engine calculates covariance and mean returns."
            )

        if 'holdings_df' not in st.session_state:
            st.session_state.holdings_df = pd.DataFrame({
                "Ticker": config.DEFAULT_TICKERS,
                "Quantity": config.DEFAULT_QUANTITIES,
                "Avg Buy Price": config.DEFAULT_BUY_PRICES,
                "Date": pd.to_datetime(config.DEFAULT_DATES)
            })
            on_input_change()

        edited_df = st.data_editor(
            st.session_state.holdings_df,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker (No .NS)", required=True),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.01, required=True),
                "Avg Buy Price": st.column_config.NumberColumn("Avg Buy Price (₹)", min_value=0.01, required=True),
                "Date": st.column_config.DateColumn("Date (Optional)")
            },
            key="holdings_editor",
            on_change=on_input_change
        )
        
        st.session_state.holdings_df = edited_df

    # Button below the container
    calculate = st.button("Calculate Optimal Portfolio", type="primary", use_container_width=True)
    
    if calculate:
        st.session_state.calc_run = True

    if not st.session_state.get("calc_run", False):
        st.info("Configure your portfolio and click 'Calculate Optimal Portfolio' to begin.")
        st.stop()

    # Extract cleaned tickers and append .NS
    clean_df = edited_df.dropna(subset=["Ticker", "Quantity"]).copy()
    clean_df["Ticker"] = clean_df["Ticker"].astype(str).str.strip().str.upper()
    grouped = clean_df.groupby("Ticker", as_index=False)["Quantity"].sum()
    
    raw_tickers = grouped["Ticker"].tolist()
    ticker_list = [t + ".NS" for t in raw_tickers if t]
    quantities = grouped["Quantity"].values
    
    if len(ticker_list) == 0:
        st.error("Please enter at least one valid ticker.")
        st.stop()

    # =========================
    # CACHED FUNCTIONS
    # =========================
    @st.cache_data(ttl=3600)
    def fetch_data(tickers, start):
        valid_tickers = []
        failed_tickers = []
        ticker_data = {}
        latest_prices = {}
        
        for t in tickers:
            try:
                df = yf.download(t, start=start, auto_adjust=True)["Close"]
                if df.empty or df.dropna().empty:
                    failed_tickers.append(t)
                else:
                    ticker_data[t] = df.squeeze()
                    valid_tickers.append(t)
                    # Extract latest available price
                    latest_prices[t] = ticker_data[t].iloc[-1]
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
    
        return data, spy, factor_prices, valid_tickers, failed_tickers, latest_prices
    
    @st.cache_data(ttl=3600)
    def fetch_dvm_scores(tickers):
        db_path = config.DVM_DB_PATH
        scores = {}
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                cursor = conn.cursor()
                for t in tickers:
                    try:
                        clean_ticker = t.replace('.NS', '')
                        cursor.execute("SELECT Durability, Valuation, Momentum FROM scores WHERE Ticker=?", (clean_ticker,))
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
    def optimize_portfolio(returns_df, risk_free_rate=config.RISK_FREE_RATE):
        mean_returns = returns_df.mean() * config.TRADING_DAYS_PER_YEAR
        cov_matrix = returns_df.cov() * config.TRADING_DAYS_PER_YEAR
        num_assets = len(returns_df.columns)
        
        def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=config.RISK_FREE_RATE):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            return ret, vol, sharpe
    
        def neg_sharpe(w):
            return -portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[2]
            
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
        bounds = tuple((config.MIN_ASSET_WEIGHT, config.MAX_ASSET_WEIGHT) for _ in range(num_assets))
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
    # ENGINE EXECUTION
    # =========================
    with st.spinner("Fetching live market data and computing metrics..."):
        data, spy, factor_prices, valid_tickers, failed_tickers, latest_prices = fetch_data(ticker_list, start_date)
    
    if failed_tickers:
        st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}. They have been dropped from calculations.")
    
    if not valid_tickers:
        st.error("No valid tickers to process.")
        st.stop()
    
    # Align weights to valid tickers based on dynamic value
    valid_values = []
    for t in valid_tickers:
        idx = ticker_list.index(t)
        qty = quantities[idx]
        current_price = latest_prices[t]
        value = qty * current_price
        valid_values.append(value)
        
    valid_values = np.array(valid_values)
    total_market_value = valid_values.sum()
    
    if total_market_value > 0:
        valid_weights = valid_values / total_market_value
    else:
        valid_weights = np.ones(len(valid_tickers)) / len(valid_tickers)
    
    allocation = pd.DataFrame({
        "Ticker": [t.replace('.NS', '') for t in valid_tickers], 
        "Weight": valid_weights,
        "Value (₹)": valid_values
    })
    
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(valid_weights)
    cumulative = (1 + portfolio_returns).cumprod()
    
    opt_weights = optimize_portfolio(returns, risk_free_rate=config.RISK_FREE_RATE)
    opt_returns = returns.dot(opt_weights)
    opt_cumulative = (1 + opt_returns).cumprod()
    
    spy_returns = spy.pct_change().dropna()
    aligned = pd.concat([portfolio_returns, spy_returns], axis=1, join="inner")
    aligned.columns = ["portfolio", "spy"]
    portfolio_aligned = aligned["portfolio"]
    spy_aligned = aligned["spy"]
    spy_cum = (1 + spy_aligned).cumprod()
    
    # Core Metrics
    annualized_portfolio_return = portfolio_returns.mean() * config.TRADING_DAYS_PER_YEAR
    annualized_volatility = portfolio_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    rf_rate = config.RISK_FREE_RATE
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
    st.markdown("---")
    st.markdown("### Portfolio Vital Signs")
    
    # Total Value Hero Metric
    st.markdown(f"<h2 style='color: #4C7766; text-align: center; margin-bottom: 2rem;'>Total Portfolio Value: ₹{total_market_value:,.2f}</h2>", unsafe_allow_html=True)
    
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
        with st.container(border=True):
            st.markdown("#### Allocation")
            fig_pie = px.pie(allocation, names="Ticker", values="Weight", hole=0.4)
            fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            with st.expander("View Allocation Details"):
                st.dataframe(allocation, use_container_width=True, hide_index=True)
    
    with col_right:
        with st.container(border=True):
            st.markdown("#### Growth of ₹10,000")
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(x=cumulative.index, y=cumulative * 10000, name="Current Portfolio"))
            fig_growth.add_trace(go.Scatter(x=opt_cumulative.index, y=opt_cumulative * 10000, name="Optimized Portfolio (Max Sharpe)", line=dict(dash='dash')))
            fig_growth.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum * 10000, name="NIFTY 50 Benchmark"))
            fig_growth.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_growth, use_container_width=True)
    

def guide_page():
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>CapitalSense User Guide</h1>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; color: #a0a0a0; margin-bottom: 3rem;'>Learn how to use the quantitative engine to analyze and optimize your holdings.</div>", unsafe_allow_html=True)

    # 1. Quick Start Section
    st.markdown("### 🚀 Quick Start")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center;'>1️⃣ Input Holdings</h3>", unsafe_allow_html=True)
            st.markdown("Go to the **Portfolio Builder** tab. Enter your actual stock symbols (e.g. `RELIANCE`), the quantity you own, and the average buy price. The engine automatically handles NSE formatting.")
        with col2:
            st.markdown("<h3 style='text-align: center;'>2️⃣ Set Timeline</h3>", unsafe_allow_html=True)
            st.markdown("Select a **Historical Start Date**. This date is critical: the engine uses historical data from this date to calculate asset covariance and volatility.")
        with col3:
            st.markdown("<h3 style='text-align: center;'>3️⃣ Optimize</h3>", unsafe_allow_html=True)
            st.markdown("Click **Calculate Optimal Portfolio**. The engine will fetch live market data, score your portfolio, and run Modern Portfolio Theory math to find maximum efficiency.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Outputs vs Inputs
    col_in, col_out = st.columns(2)
    
    with col_in:
        st.markdown("### 📥 What You Provide")
        with st.container(border=True):
            st.info("**Tickers (Raw NSE Symbols)**\n\nNo need to add `.NS`. Just type the symbol. The engine cleans the input automatically.")
            st.info("**Quantity & Avg Buy Price**\n\nThe app calculates your *True Current Weight* by multiplying your quantity by the live market price.")
            st.info("**Start Date**\n\nDetermines the lookback period for risk calculations.")

    with col_out:
        st.markdown("### 🎯 What You Get")
        with st.container(border=True):
            st.success("**Max Sharpe Allocation**\n\nThe exact mathematical weights required to maximize return for your given risk level.")
            st.success("**CapitalSense DVM Score**\n\nA proprietary blend of Durability, Valuation, and Momentum scores for your specific mix of assets.")
            st.success("**Performance & Risk Metrics**\n\nExpected Annual Return, Annualized Volatility, and a historical growth benchmark against the NIFTY 50.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Deep Dive Expanders
    st.markdown("### 🧠 How the Engine Works")
    with st.expander("The Mathematics of the Optimal Portfolio"):
        st.markdown("""
        The engine uses **Modern Portfolio Theory (MPT)** and specifically the `scipy.optimize` SLSQP solver to maximize the **Sharpe Ratio** of your portfolio. 
        
        It looks at the historical covariance matrix (how the stocks move together) and their mean returns, and attempts to find a combination of weights that maximizes returns while mathematically minimizing volatility.
        """)
        
    with st.expander("Understanding the DVM Score"):
        st.markdown("""
        The **CapitalSense DVM Score** reads from an offline, proprietary SQLite database of corporate metrics.
        
        - **Durability:** A measure of the company's financial health and moat.
        - **Valuation:** How expensive the stock is relative to historical averages and peers.
        - **Momentum:** The technical price trend of the asset.
        
        Your portfolio's overall score is the weighted average of these underlying asset scores.
        """)
        
    with st.expander("Why doesn't the app auto-calculate when I type?"):
        st.markdown("""
        **To protect your computational bandwidth and avoid API rate limits.** Fetching live data from Yahoo Finance for a dozen stocks across several years is a heavy operation. By gating the calculation behind a button, you can comfortably build your entire portfolio table before triggering the expensive math engine.
        """)

pages = {
    "Start": [
        st.Page(guide_page, title="User Guide", icon="📖", default=True)
    ],
    "Tools": [
        st.Page(main_page, title="Portfolio Builder", icon="📈")
    ]
}

pg = st.navigation(pages, position="sidebar")
pg.run()
