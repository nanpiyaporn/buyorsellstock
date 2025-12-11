# buyorsellstock

This workspace contains a simple Streamlit app that predicts future stock prices using a Monte Carlo simulation based on 5 years of historical daily data from Yahoo Finance.

Files added:
- `streamlit_app.py`: Streamlit frontend + Monte Carlo backend.
- `requirements.txt`: Python dependencies.

Quick start:

1. Create (or activate) a Python environment with Python 3.9+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_app.py
```

Usage notes:
- Enter a US ticker in the sidebar (e.g., AAPL).
- Pick a horizon: Next week, Month, 3 months, 6 months, or 1 year.
- Click "Run Monte Carlo Prediction" to simulate price paths and view percentiles.

Caveats: This is a simple Monte Carlo using historical log-returns (geometric Brownian motion approximation). It does not account for corporate events, jumps, or regime shifts and is for educational/demo purposes only.

Everyone want to get rich from stock market why not we analyze it buy ourselves without paying other.
