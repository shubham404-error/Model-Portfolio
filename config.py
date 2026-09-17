# Configuration for CapitalSense Model Portfolio App

# General Settings
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.065 # 6.5% for the Indian market

# Optimization Constraints
MAX_ASSET_WEIGHT = 0.5 # Maximum allowable weight for a single asset during optimization
MIN_ASSET_WEIGHT = 0.0 # Minimum allowable weight

# Data Settings
DVM_DB_PATH = "data/capitalsense_dvm.sqlite"

# Default UI Inputs
DEFAULT_TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
DEFAULT_QUANTITIES = [10, 5, 20, 15]
DEFAULT_BUY_PRICES = [2500.0, 3500.0, 1500.0, 1400.0]
DEFAULT_DATES = ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"]
