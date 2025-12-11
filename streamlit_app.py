import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import csv
import json


st.set_page_config(page_title="Stock Predictor — Monte Carlo", layout="wide")

HISTORY_FILE = "prediction_history.csv"
SAVED_TICKERS_FILE = "saved_tickers.json"


def init_saved_tickers():
    """Create saved tickers file if it doesn't exist."""
    if not os.path.exists(SAVED_TICKERS_FILE):
        with open(SAVED_TICKERS_FILE, 'w') as f:
            json.dump([], f)


def load_saved_tickers():
    """Load saved tickers from JSON."""
    init_saved_tickers()
    try:
        with open(SAVED_TICKERS_FILE, 'r') as f:
            tickers = json.load(f)
        return list(set(tickers))  # Remove duplicates and return
    except:
        return []


def save_ticker(ticker):
    """Add ticker to saved list."""
    init_saved_tickers()
    tickers = load_saved_tickers()
    ticker_upper = ticker.upper()
    if ticker_upper not in tickers:
        tickers.append(ticker_upper)
        with open(SAVED_TICKERS_FILE, 'w') as f:
            json.dump(tickers, f)


def delete_ticker(ticker):
    """Remove ticker from saved list."""
    init_saved_tickers()
    tickers = load_saved_tickers()
    ticker_upper = ticker.upper()
    if ticker_upper in tickers:
        tickers.remove(ticker_upper)
        with open(SAVED_TICKERS_FILE, 'w') as f:
            json.dump(tickers, f)


def init_history_file():
    """Create history file if it doesn't exist."""
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'prediction_date', 'ticker', 'current_price', 'horizon', 
                'n_simulations', 'random_seed', 'median_price', 'p10', 'p90',
                'future_date', 'actual_price', 'accuracy_pct'
            ])


def save_prediction(ticker, current_price, horizon, n_sims, seed, median, p10, p90, future_date):
    """Save prediction to CSV."""
    init_history_file()
    # Ensure all values are Python natives, not pandas/numpy types
    current_price = float(current_price)
    median = float(median)
    p10 = float(p10)
    p90 = float(p90)
    with open(HISTORY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker.upper(),
            f"{current_price:.2f}",
            horizon,
            n_sims,
            seed,
            f"{median:.2f}",
            f"{p10:.2f}",
            f"{p90:.2f}",
            future_date.strftime("%Y-%m-%d"),
            "",  # actual_price (empty until date passes)
            ""   # accuracy_pct (empty until date passes)
        ])


def load_prediction_history():
    """Load prediction history from CSV."""
    init_history_file()
    try:
        df = pd.read_csv(HISTORY_FILE)
        return df
    except:
        return pd.DataFrame()


def delete_prediction(index):
    """Delete a prediction from history."""
    init_history_file()
    try:
        df = pd.read_csv(HISTORY_FILE)
        df = df.drop(index).reset_index(drop=True)
        df.to_csv(HISTORY_FILE, index=False)
    except:
        pass


@st.cache_data(ttl=3600)
def get_current_price(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        info = tk.info
        return float(info.get('regularMarketPrice', np.nan))
    except Exception:
        return np.nan


@st.cache_data(ttl=3600)
def get_history(ticker: str, period='5y'):
    try:
        df = yf.download(ticker, period=period, progress=False)
        return df
    except Exception:
        return pd.DataFrame()


def monte_carlo_simulation(S0, returns, n_sims=1000, n_steps=252, seed=42):
    np.random.seed(seed)
    # Convert pandas Series to numpy array
    if isinstance(returns, pd.Series):
        log_ret_array = returns.values
    else:
        log_ret_array = np.asarray(returns)
    
    # Remove NaN values
    log_ret_array = log_ret_array[~np.isnan(log_ret_array)]
    
    if len(log_ret_array) == 0:
        raise ValueError("Not enough return data for simulation")
    
    # Calculate statistics using numpy
    mu = np.mean(log_ret_array)
    sigma = np.std(log_ret_array, ddof=1)
    drift = mu - 0.5 * (sigma ** 2)
    
    # Ensure S0 is a pure float
    S0 = float(S0)

    # Generate random numbers
    rand = np.random.normal(0, 1, (n_steps, n_sims))
    
    # Initialize price paths
    price_paths = np.zeros((n_steps + 1, n_sims), dtype=np.float64)
    price_paths[0, :] = S0
    
    # Simulate paths
    for t in range(1, n_steps + 1):
        price_paths[t, :] = price_paths[t - 1, :] * np.exp(drift + sigma * rand[t - 1, :])

    return price_paths


def trading_days_for_horizon(choice: str) -> int:
    mapping = {
        'Next week': 5,
        'Month': 21,
        '3 months': 63,
        '6 months': 126,
        '1 year': 252,
    }
    return mapping.get(choice, 21)


def main():
    st.title("Monte Carlo Stock Predictor — US Stocks")

    # Top row: datetime and major indices
    col1, col2 = st.columns([1, 2])
    with col1:
        now = datetime.now()
        st.markdown("**Current local date & time**")
        st.write(now.strftime("%Y-%m-%d %H:%M:%S"))

    with col2:
        st.markdown("**Major US indices (current price)**")
        idx_col1, idx_col2, idx_col3 = st.columns(3)
        dji = get_current_price("^DJI")
        ixic = get_current_price("^IXIC")
        gspc = get_current_price("^GSPC")
        idx_col1.metric("Dow Jones (^DJI)", f"{dji:,.2f}" if not np.isnan(dji) else "N/A")
        idx_col2.metric("Nasdaq (^IXIC)", f"{ixic:,.2f}" if not np.isnan(ixic) else "N/A")
        idx_col3.metric("S&P 500 (^GSPC)", f"{gspc:,.2f}" if not np.isnan(gspc) else "N/A")

    st.markdown("---")

    st.sidebar.header("Prediction settings")
    
    # Use session state for selected ticker
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "AAPL"
    
    ticker = st.sidebar.text_input("Ticker (US stock)", value=st.session_state.selected_ticker)
    
    # Save ticker button
    if st.sidebar.button("💾 Save this ticker"):
        if ticker.strip():
            save_ticker(ticker.strip())
            st.success(f"Saved {ticker.upper()}!")
            st.rerun()
    
    # Saved tickers section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Saved tickers")
    saved_tickers = load_saved_tickers()
    
    if saved_tickers:
        for saved_ticker in sorted(saved_tickers):
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(f"📌 {saved_ticker}", key=f"btn_{saved_ticker}"):
                    st.session_state.selected_ticker = saved_ticker
            with col2:
                if st.button("🗑️", key=f"del_{saved_ticker}"):
                    delete_ticker(saved_ticker)
                    st.rerun()
    else:
        st.sidebar.info("No saved tickers yet. Search and save one!")
    
    st.sidebar.markdown("---")
    horizon_choice = st.sidebar.selectbox("Predict horizon", ['Next week', 'Month', '3 months', '6 months', '1 year'])
    sims = st.sidebar.slider("Number of simulations", 100, 5000, 1000, step=100)
    seed = st.sidebar.number_input("Random seed", value=42)

    if not ticker:
        st.warning("Please enter a ticker symbol in the sidebar.")
        return

    st.subheader(f"Ticker: {ticker.upper()}")
    current_price = get_current_price(ticker)
    st.write("Current price:", f"{current_price:,.2f}" if not np.isnan(current_price) else "N/A")

    # Run simulation
    if st.button("Run Monte Carlo Prediction"):
        with st.spinner("Downloading data and running simulations..."):
            hist = get_history(ticker, period='5y')
            if hist.empty:
                st.error("Could not download historical data for that ticker.")
                return
            
            # Sort by date ascending (oldest to newest) to ensure correct order
            hist = hist.sort_index(ascending=True)
            
            st.write(f"**Data range: {hist.index[0].date()} to {hist.index[-1].date()}** ({len(hist)} trading days)")
            
            if 'Adj Close' in hist.columns:
                price_series = hist['Adj Close']
            elif 'Close' in hist.columns:
                price_series = hist['Close']
            else:
                st.error("Historical data does not contain close prices.")
                return

            # daily log returns
            log_returns = np.log(price_series / price_series.shift(1)).dropna()
            S0 = float(price_series.iloc[-1])  # Convert to pure float
            n_steps = trading_days_for_horizon(horizon_choice)

            try:
                paths = monte_carlo_simulation(S0, log_returns, n_sims=sims, n_steps=n_steps, seed=int(seed))
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                return

            final_prices = paths[-1]
            median = np.median(final_prices)
            p10 = np.percentile(final_prices, 10)
            p90 = np.percentile(final_prices, 90)

            # Calculate future date
            future_date = datetime.now() + timedelta(days=n_steps)
            
            # Save prediction to history
            save_prediction(ticker, S0, horizon_choice, sims, int(seed), median, p10, p90, future_date)

            st.success(f"Simulation complete ({sims} sims, horizon: {horizon_choice})")
            st.write(f"Predicted median price after {horizon_choice}: {median:,.2f}")
            st.write(f"10th percentile: {p10:,.2f}  —  90th percentile: {p90:,.2f}")
            
            # Calculate % differences
            pct_diff_median = ((median - S0) / S0) * 100
            pct_diff_p10 = ((p10 - S0) / S0) * 100
            pct_diff_p90 = ((p90 - S0) / S0) * 100
            
            direction = "increase" if pct_diff_median > 0 else "decrease"
            st.write(f"**Price {direction} {abs(pct_diff_median):.2f}% from today:**")
            st.write(f"  Median: {pct_diff_median:+.2f}% | 10th %ile: {pct_diff_p10:+.2f}% | 90th %ile: {pct_diff_p90:+.2f}%")

            # plot sample paths
            fig, ax = plt.subplots(figsize=(8, 4))
            sample = paths[:, :min(50, sims)]
            ax.plot(sample)
            ax.set_title(f"Sample simulated paths ({min(50, sims)} paths)")
            ax.set_xlabel("Trading days")
            ax.set_ylabel("Price")
            st.pyplot(fig)

            # histogram of final prices
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.hist(final_prices, bins=50)
            ax2.axvline(median, color='r', linestyle='--', label='Median')
            ax2.legend()
            ax2.set_title("Distribution of simulated final prices")
            st.pyplot(fig2)

    # Display prediction history
    st.markdown("---")
    st.subheader("Prediction History")
    hist_df = load_prediction_history()
    
    if not hist_df.empty:
        st.write("**All predictions made. Update actual prices when dates pass or delete rows:**")
        
        # Header row
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
        with col1:
            st.write("**Pred Date**")
        with col2:
            st.write("**Ticker**")
        with col3:
            st.write("**Curr Price**")
        with col4:
            st.write("**Horizon**")
        with col5:
            st.write("**Sims**")
        with col6:
            st.write("**Seed**")
        with col7:
            st.write("**Median**")
        with col8:
            st.write("**Future Date**")
        with col9:
            st.write("**Actual Price**")
        with col10:
            st.write("**Accuracy**")
        with col11:
            st.write("**Action**")
        
        # Data rows
        for idx, row in hist_df.iterrows():
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
            with col1:
                st.write(row['prediction_date'][:10])
            with col2:
                st.write(row['ticker'])
            with col3:
                st.write(row['current_price'])
            with col4:
                st.write(row['horizon'])
            with col5:
                st.write(row['n_simulations'])
            with col6:
                st.write(row['random_seed'])
            with col7:
                st.write(row['median_price'])
            with col8:
                st.write(row['future_date'])
            with col9:
                actual = st.text_input(f"Actual price {idx}", value=row['actual_price'] or "", key=f"actual_{idx}")
                # Update CSV with actual price if entered
                if actual and actual != row['actual_price']:
                    try:
                        actual_p = float(actual)
                        median_p = float(row['median_price'])
                        acc = 100 * (1 - abs(actual_p - median_p) / median_p)
                        # Update the CSV
                        hist_df.at[idx, 'actual_price'] = str(actual_p)
                        hist_df.at[idx, 'accuracy_pct'] = f"{acc:.1f}%"
                        hist_df.to_csv(HISTORY_FILE, index=False)
                    except:
                        pass
            with col10:
                if row['actual_price'] and row['median_price']:
                    try:
                        median_p = float(row['median_price'])
                        actual_p = float(row['actual_price'])
                        acc = 100 * (1 - abs(actual_p - median_p) / median_p)
                        st.write(f"{acc:.1f}%")
                    except:
                        st.write("—")
                else:
                    st.write("—")
            with col11:
                if st.button("🗑️ Delete", key=f"del_pred_{idx}"):
                    delete_prediction(idx)
                    st.success("Prediction deleted!")
                    st.rerun()
    else:
        st.info("No predictions yet. Run a Monte Carlo simulation to start tracking.")


if __name__ == '__main__':
    main()
