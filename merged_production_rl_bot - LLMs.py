# Standard library imports
import asyncio
import copy
import json
import logging
import os
import random
import sqlite3
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta

# Third-party imports
import aiohttp
# eod import moved to optional section
import exchange_calendars as xcals
import google.generativeai as genai
import gymnasium as gym
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import xgboost as xgb

# Scientific computing
from scipy.signal import argrelextrema
from scipy.special import expit

# Machine Learning
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
# optuna imports moved to optional section

# Deep Learning
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Reinforcement Learning
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

# Technical Analysis
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

# Sentiment Analysis
# NewsApiClient import moved to optional section
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure warnings
warnings.filterwarnings("ignore")

# Compatibility functions for scikit-learn version differences
def create_calibrated_classifier(estimator, method='isotonic', cv=3):
    """Create CalibratedClassifierCV with version compatibility"""
    try:
        # Try new parameter name (scikit-learn >= 0.24)
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        try:
            # Fallback to old parameter name (scikit-learn < 0.24)
            return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)
        except Exception as e:
            logging.error(f"CalibratedClassifierCV compatibility error: {e}")
            # Return uncalibrated estimator as last resort
            return estimator

class PurgedGroupTimeSeriesSplit:
    """
    Custom implementation of Purged Group Time Series Split for financial data
    Based on advances in financial machine learning practices
    """
    
    def __init__(self, n_splits=5, gap=0, max_train_size=None):
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate test size for each fold
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Test set boundaries
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)
            
            if test_start >= n_samples:
                break
                
            # Training set boundaries (before test set with gap)
            train_end = max(0, test_start - self.gap)
            train_start = 0
            
            # Apply max_train_size if specified
            if self.max_train_size and (train_end - train_start) > self.max_train_size:
                train_start = train_end - self.max_train_size
            
            # Ensure we have valid training data
            if train_start < train_end and test_start < test_end:
                train_indices = indices[train_start:train_end]
                test_indices = indices[test_start:test_end]
                
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits

def safe_cross_val_score(estimator, X, y, cv=5, scoring='f1', n_jobs=None):
    """Safe cross-validation with error handling"""
    try:
        return cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    except Exception as e:
        logging.warning(f"Cross-validation error: {e}")
        # Fallback to simple train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        from sklearn.metrics import f1_score
        return [f1_score(y_test, y_pred)]

# Optional GPU configuration for TensorFlow
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.set_visible_devices([], 'GPU')

GOOGLE_AI_API_KEY = "AIzaSyBCexoODvgrN2QRG8_iKv3p5VTJ5jaJ_B0"

# === CONSTANTS ===
# Market timing constants
FOREX_MARKET_CLOSE_HOUR_UTC = 21  # Friday close time
WEEKEND_FRIDAY = 4
WEEKEND_SATURDAY = 5
WEEKEND_SUNDAY = 6

# Cache timeout constants (seconds)
NEWS_CACHE_TIMEOUT = 3600  # 1 hour
ECONOMIC_CALENDAR_CACHE_TIMEOUT = 86400  # 24 hours

# Trading constants
DEFAULT_CANDLE_WAIT_MINUTES = 60  # H1 default
CRYPTO_PREFIXES = ["BTC", "ETH"]

# Feature engineering constants
DEFAULT_ATR_MULTIPLIER = 2.0
DEFAULT_RANGE_WINDOW = 50
DEFAULT_EMA_PERIOD = 200
DEFAULT_ADX_THRESHOLD = 25

# Advanced ML constants
STACKING_CV_FOLDS = 5
CALIBRATION_CV_FOLDS = 3
MIN_SAMPLES_LEAF = 50
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour

# Quality gates thresholds
MIN_SHARPE_RATIO = 1.2
MAX_DRAWDOWN_THRESHOLD = 0.15
MIN_CALMAR_RATIO = 0.8
MIN_INFORMATION_RATIO = 0.5

# Enhanced risk management configuration by asset class
RISK_CONFIG_BY_ASSET_CLASS = {
    "equity_index": {
        "max_position_size": 0.25,  # 25% max per equity index
        "correlation_threshold": 0.7,
        "var_multiplier": 1.0,
        "stop_loss_atr": 2.0,
        "take_profit_atr": 4.0,
        "max_daily_loss": 0.02,
        "session_risk_adjustment": True,
        "gap_risk_factor": 1.5
    },
    "commodity": {
        "max_position_size": 0.20,  # 20% max per commodity
        "correlation_threshold": 0.6,
        "var_multiplier": 1.2,
        "stop_loss_atr": 2.5,
        "take_profit_atr": 5.0,
        "max_daily_loss": 0.025,
        "session_risk_adjustment": False,
        "gap_risk_factor": 1.0
    },
    "forex": {
        "max_position_size": 0.15,  # 15% max per forex pair
        "correlation_threshold": 0.8,
        "var_multiplier": 0.8,
        "stop_loss_atr": 1.5,
        "take_profit_atr": 3.0,
        "max_daily_loss": 0.015,
        "session_risk_adjustment": True,
        "gap_risk_factor": 1.2
    },
    "cryptocurrency": {
        "max_position_size": 0.10,  # 10% max per crypto
        "correlation_threshold": 0.5,
        "var_multiplier": 1.5,
        "stop_loss_atr": 3.0,
        "take_profit_atr": 6.0,
        "max_daily_loss": 0.03,
        "session_risk_adjustment": False,
        "gap_risk_factor": 1.0
    }
}

# Portfolio-level risk limits
PORTFOLIO_RISK_LIMITS = {
    "max_total_exposure": 0.80,  # 80% max total portfolio exposure
    "max_asset_class_exposure": 0.50,  # 50% max per asset class
    "max_correlation_exposure": 0.40,  # 40% max for highly correlated positions
    "daily_var_limit": 0.02,  # 2% daily VaR limit
    "weekly_var_limit": 0.05,  # 5% weekly VaR limit
    "max_drawdown_limit": 0.15,  # 15% max drawdown
    "concentration_limit": 0.30  # 30% max concentration in single position
}

# Observability constants
DISCORD_RATE_LIMIT_SECONDS = 60
ALERT_COOLDOWN_MINUTES = 15
PERFORMANCE_LOG_INTERVAL = 24  # hours

# === DATA FRESHNESS LIMITS (minutes) ===
DATA_FRESHNESS_LIMIT_MIN = {
    "H1": 180,
    "H4": 600,
    "D1": 3000,     # 48 hours
    "W1": 20160     # 14 days
}

def is_weekend():
    """
    Check if it's weekend, including Friday evening.
    Forex market closes around 21:00 UTC on Friday.
    
    Returns:
        bool: True if it's weekend or Friday after 21:00 UTC, False otherwise.
    """
    now_utc = datetime.utcnow()
    weekday = now_utc.weekday()
    # Saturday or Sunday
    if weekday >= WEEKEND_SATURDAY:
        return True
    # Friday after market close time
    if weekday == WEEKEND_FRIDAY and now_utc.hour >= FOREX_MARKET_CLOSE_HOUR_UTC:
        return True
    return False

def pre_trade_weekend_guard(symbol: str) -> bool:
    """
    Check if trading is allowed for a symbol during weekends.
    
    Args:
        symbol: Trading symbol to check
        
    Returns:
        bool: True = ALLOW trading, False = BLOCK trading.
              Policy: block weekends for non-crypto; crypto allowed 24/7.
    """
    sym = (symbol or "").upper()
    if is_crypto_symbol(sym):
        return True  # crypto 24/7
    if TRADE_FILTERS.get("AVOID_WEEKEND", False) and is_weekend():
        logging.info(f"[Pre-Trade] Skip open: weekend policy for non-crypto ({sym}).")
        return False
    return True
# ---------------------------------------------------------------------------

def is_market_open(symbol):
    """
    Check if the market for a specific symbol is currently open.
    Special rule: Allow ETHUSD trading on weekends.
    
    Args:
        symbol (str): Trading symbol to check
        
    Returns:
        bool: True if market is open, False otherwise
    """
    # Special rule for ETHUSD
    if symbol == "ETHUSD":
        logging.info(f"[Market Status] Special rule applied: ETHUSD allowed on weekends.")
        return True  # Always allow trading for ETHUSD

    # Exchange mapping for different symbols
    SYMBOL_EXCHANGE_MAP = {
        "SPX500": "XNYS",
        "DE40":   "XFRA",
        "JP225":  "XTKS",
        "XAUUSD": "CMES",
        "AUDNZD": "FOREX",
    }
    
    exchange_code = SYMBOL_EXCHANGE_MAP.get(symbol)
    
    if not exchange_code or exchange_code in ["FOREX", "CMES"]:
        return not is_weekend()

    try:
        calendar = xcals.get_calendar(exchange_code)
        now_utc = pd.Timestamp.utcnow()
        return calendar.is_open_on_minute(now_utc)
    except (ValueError, KeyError) as e:
        logging.warning(f"Invalid exchange code '{exchange_code}' for {symbol}: {e}")
        return True  # Default to open if unable to check
    except Exception as e:
        logging.warning(f"Unexpected error checking trading calendar for {symbol}: {e}")
        return True  # Default to open if unable to check

def is_crypto_symbol(symbol):
    """
    Check if a symbol represents a cryptocurrency.
    
    Args:
        symbol (str): Trading symbol to check
        
    Returns:
        bool: True if symbol is a cryptocurrency, False otherwise
    """
    if not symbol:
        return False
    symbol_upper = symbol.upper()
    return any(symbol_upper.startswith(prefix) for prefix in CRYPTO_PREFIXES)

def is_past_weekend_close_time(symbol: str | None = None) -> bool:
    """
    Check if current time is past weekend close time.
    
    Args:
        symbol: Optional trading symbol to check. Crypto symbols are exempt.
        
    Returns:
        bool: True = past close time (LOCKED), False = allowed.
              Crypto: always False to not lock on weekends.
    """
    if not WEEKEND_CLOSE_CONFIG.get("ENABLED", False):
        return False
    
    # If symbol is crypto, always return False to not lock on weekends
    if symbol and is_crypto_symbol(symbol):
        return False

    now_utc = datetime.utcnow()
    close_day = WEEKEND_CLOSE_CONFIG["CLOSE_DAY_UTC"]   # 4 = Fri
    close_hour = WEEKEND_CLOSE_CONFIG["CLOSE_HOUR_UTC"] # 17:00 UTC
    
    # Fri after 17:00 UTC OR Sat/Sun
    if (now_utc.weekday() == close_day and now_utc.hour >= close_hour) or (now_utc.weekday() > close_day):
        return True
    return False

def wait_for_next_primary_candle(primary_tf, symbol="DUMMY"):
    """
    Wait until the next primary timeframe candle starts.
    Skip if it's a crypto symbol.
    
    Args:
        primary_tf (str): Primary timeframe (e.g., "H1", "H4", "D1")
        symbol (str): Trading symbol, defaults to "DUMMY"
    """
    # Input validation: ensure symbol is a valid string
    if symbol is None or not isinstance(symbol, str):
        symbol = "DUMMY"
    
    if is_crypto_symbol(symbol):
        logging.info(f"[Sync] Crypto {symbol} doesn't need candle synchronization.")
        return

    now_utc = datetime.utcnow()
    minutes_per_candle = DEFAULT_CANDLE_WAIT_MINUTES  # Default H1

    if "M" in primary_tf:
        minutes_per_candle = int(primary_tf.replace("M", ""))
    elif "H" in primary_tf:
        minutes_per_candle = int(primary_tf.replace("H", "")) * 60
    elif "D" in primary_tf:
        minutes_per_candle = 24 * 60
    elif "W" in primary_tf:
        minutes_per_candle = 7 * 24 * 60

    # Calculate current candle time in the cycle
    current_minute_in_cycle = (now_utc.minute + now_utc.hour * 60) % minutes_per_candle
    
    # Calculate wait time until next candle
    minutes_to_wait = minutes_per_candle - current_minute_in_cycle
    wait_seconds = minutes_to_wait * 60

    if wait_seconds > 0:
        next_candle_time = now_utc + timedelta(minutes=minutes_to_wait)
        print(f"✅ Bot is synchronized. Waiting for {int(wait_seconds / 60)} minutes until the next {primary_tf} candle starts at {next_candle_time.strftime('%H:%M')} UTC...")
        time.sleep(wait_seconds)

# Note: Old wait_for_next_h4_candle() function has been removed as it's no longer needed
def process_symbol_cycle(symbol: str):
    # 0) Determine primary_tf and timeframe set to use
    primary_tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, "H1")
    timeframes_to_use = resolve_timeframes_for_symbol(symbol, primary_tf)

    # 1) Get multi-timeframe data
    multi_tf_data = fetch_multi_timeframe_data(symbol, 5000, timeframes_to_use)

    # 2) Log sanity check
    logging.info(f"[Sanity] weekend={is_weekend()} sym={symbol} crypto={is_crypto_symbol(symbol)} primary_tf={primary_tf}")

    # 3) Candle synchronization: non-crypto must wait; crypto bypassed
    wait_for_next_primary_candle(primary_tf, symbol)

    # 4) Create features
    features = create_enhanced_features_from_multi_tf(symbol, multi_tf_data)
    if features is None:
        logging.info(f"[Features] Skip {symbol}: feature build failed.")
        return

    # 5) Open trades (based on signals)
    maybe_open_position(
        symbol,
        primary_tf,
        multi_tf_data,
        signal_provider=get_signal_provider(),
        risk_manager=get_risk_manager(),
        broker=get_broker_adapter()
    )

    # 6) Manage open positions if you have this flow
    manage_open_positions(symbol, multi_tf_data)


def should_wait_for_sync(symbol: str, primary_tf: str) -> bool:
    """
    Non-crypto: giữ chờ nến mới. Crypto: chạy ngay, không chờ.
    """
    return not is_crypto_symbol(symbol)

def choose_sync_primary_tf():
    """
    Chọn TF đồng bộ chung cho vòng lặp:
    - Nếu có bất kỳ symbol primary H1 → dùng H1
    - Nếu không có H1 mà có H4 → dùng H4
    - Ngược lại → D1
    """
    # If you already have PRIMARY_TIMEFRAME_BY_SYMBOL, use that map;
    # if NOT available, return default PRIMARY_TIMEFRAME (usually "H4")
    try:
        # Prioritize H1 if available in the list
        tfs = set()
        for s in SYMBOLS:
            tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(s, PRIMARY_TIMEFRAME)
            tfs.add(tf)
        if "H1" in tfs:
            return "H1"
        if "H4" in tfs:
            return "H4"
        return "D1"
    except NameError:
        # fallback when symbol map is not declared
        return PRIMARY_TIMEFRAME  # example "H4"

import pytz

# Optional imports with error handling
try:
    import tradingeconomics as te
    TRADING_ECONOMICS_AVAILABLE = True
except ImportError:
    logging.warning("tradingeconomics package not available. Economic calendar features will be limited.")
    TRADING_ECONOMICS_AVAILABLE = False
    te = None

# Check for other optional packages
OPTIONAL_PACKAGES = {}

# Check for additional ML packages
try:
    import optuna
    OPTIONAL_PACKAGES['optuna'] = True
except ImportError:
    OPTIONAL_PACKAGES['optuna'] = False
    logging.warning("optuna not available. Hyperparameter optimization will be limited.")

try:
    from optuna.integration import LightGBMPruningCallback
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTIONAL_PACKAGES['optuna_integrations'] = True
except ImportError:
    OPTIONAL_PACKAGES['optuna_integrations'] = False
    logging.warning("optuna integrations not available. Some optimization features disabled.")

# Check for financial data packages
try:
    import eod
    OPTIONAL_PACKAGES['eod'] = True
except ImportError:
    OPTIONAL_PACKAGES['eod'] = False
    logging.warning("eod package not available. EODHD news provider will be disabled.")

try:
    from newsapi import NewsApiClient
    OPTIONAL_PACKAGES['newsapi'] = True
except ImportError:
    OPTIONAL_PACKAGES['newsapi'] = False
    logging.warning("newsapi package not available. NewsAPI provider will be disabled.")

# Utilities
import time
import joblib
import os
import optuna
import json
import logging
# TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY

# TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY

def validate_dataframe_freshness(
    multi_tf_data,
    primary_tf: str,
    *,
    symbol: str | None = None,
    max_stale_minutes: int | float | None = None,
    per_tf_override: dict | None = None
):
    """
    Ưu tiên ngưỡng: per_tf_override > max_stale_minutes > DATA_FRESHNESS_LIMIT_MIN.
    Bỏ qua kiểm tra cho **mọi crypto** vào cuối tuần. Trả về True/False.
    """
    import logging
    from datetime import datetime, timezone

    if not multi_tf_data or primary_tf not in multi_tf_data:
        return False

    # BYPASS for crypto on weekends
    sym = (symbol or "").upper()
    if sym and is_crypto_symbol(sym) and is_weekend():
        logging.info(f"[Data Freshness] Skipping check for {sym} (crypto) on weekends.")
        return True

    limits = DATA_FRESHNESS_LIMIT_MIN.copy()
    if isinstance(per_tf_override, dict):
        limits.update({str(k).upper(): per_tf_override[k] for k in per_tf_override})

    now_utc = datetime.now(timezone.utc)

    for tf, df in multi_tf_data.items():
        if df is None or df.empty:
            logging.warning(f"[Data Freshness] Dữ liệu rỗng cho {tf}.")
            return False

        tf_key = str(tf).upper()
        tf_limit = (per_tf_override or {}).get(tf_key)
        if tf_limit is None:
            tf_limit = max_stale_minutes if max_stale_minutes is not None else limits.get(tf_key, 720)

        # Normalize type
        try:
            tf_limit = float(tf_limit)
        except Exception:
            logging.error(f"[Data Freshness] Invalid threshold for {tf_key}: {tf_limit}.")
            return False

        idx = df.index
        if getattr(idx, "tz", None) is None:
            last_ts = idx[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            last_ts = idx[-1].tz_convert("UTC").to_pydatetime()

        minutes_diff = (now_utc - last_ts).total_seconds() / 60.0
        if minutes_diff > tf_limit:
            logging.error(f"[Data Freshness] ❌ STALE DATA! {tf_key} delayed {int(minutes_diff)} minutes (threshold {tf_limit}).")
            return False

    logging.info("[Data Freshness] ✅ All timeframes valid.")
    return True


# Replace old drift configuration with these lines
# === DRIFT MANAGEMENT CONFIG ===
DRIFT_SAMPLE_SIZE = 8          # Number of random symbols to check in each cycle
DRIFT_WARNING_INCREMENT = 1    # Points added when drift is detected
DRIFT_SCORE_DECAY = 1          # Points subtracted (cooling down) after each check cycle
DRIFT_SCORE_THRESHOLD = 3      # Score threshold to trigger retraining
# ==============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
# @title Advanced system configuration
#OANDA_API_KEY = "abcb6aac77695382ee0f6b99c4074fb5-d7ff9754c4aed1d310d706b423078dfb"  # Replace with actual API key
#OANDA_URL = "https://api-fxpractice.oanda.com/v3"
OANDA_API_KEY = "814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d"  # Replace with actual API key
OANDA_URL = "https://api-fxtrade.oanda.com/v3"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1410915486551511062/pzCWm4gbe0w-xFyI0pKbsy417sbsYwjwjg-iWMLhccIGRJR2FqJ4kUwlzIZAyw3C2Fhq"
# Optimized symbol configuration with asset class metadata
SYMBOLS = [
    "SPX500",      # S&P 500 (US) - US Equity Index
    "DE40",        # DAX 40 (Germany) - European Equity Index
    "XAUUSD",      # Gold - Precious Metal Commodity
    "AUDNZD",      # Aussie vs Kiwi - Forex Cross Pair
    "BTCUSD",      # Bitcoin - Cryptocurrency
    "ETHUSD",      # Ethereum - Cryptocurrency
    "JP225",       # Nikkei 225 (Japan) - Asian Equity Index
]

# Enhanced symbol metadata for optimization
SYMBOL_METADATA = {
    "SPX500": {
        "asset_class": "equity_index",
        "region": "US",
        "session": "US",
        "volatility_profile": "medium",
        "correlation_group": "us_indices",
        "pip_value": 1.0,
        "min_lot_size": 0.01,
        "spread_typical": 0.5,
        "trading_hours": {"start": 9, "end": 16, "timezone": "US/Eastern"},
        "high_impact_events": ["NFP", "CPI", "FOMC", "GDP", "PCE"],
        "news_sources": ["financial_news", "economic_calendar"],
        "technical_focus": ["trend_following", "momentum", "support_resistance"]
    },
    "DE40": {
        "asset_class": "equity_index",
        "region": "Europe",
        "session": "European",
        "volatility_profile": "medium",
        "correlation_group": "european_indices",
        "pip_value": 1.0,
        "min_lot_size": 0.01,
        "spread_typical": 0.8,
        "trading_hours": {"start": 9, "end": 17, "timezone": "Europe/Berlin"},
        "high_impact_events": ["ECB", "German_IFO", "PMI", "GDP"],
        "news_sources": ["financial_news", "economic_calendar"],
        "technical_focus": ["trend_following", "breakout", "mean_reversion"]
    },
    "XAUUSD": {
        "asset_class": "commodity",
        "region": "global",
        "session": "24h",
        "volatility_profile": "high",
        "correlation_group": "precious_metals",
        "pip_value": 0.1,
        "min_lot_size": 0.01,
        "spread_typical": 2.0,
        "trading_hours": {"start": 0, "end": 23, "timezone": "UTC"},
        "high_impact_events": ["Fed_Meetings", "Inflation", "Geopolitical", "USD_Strength"],
        "news_sources": ["financial_news", "economic_calendar", "geopolitical"],
        "technical_focus": ["trend_following", "support_resistance", "fibonacci"]
    },
    "AUDNZD": {
        "asset_class": "forex",
        "region": "Pacific",
        "session": "Asian",
        "volatility_profile": "medium",
        "correlation_group": "commodity_currencies",
        "pip_value": 0.0001,
        "min_lot_size": 0.01,
        "spread_typical": 1.5,
        "trading_hours": {"start": 21, "end": 7, "timezone": "UTC"},
        "high_impact_events": ["RBA", "RBNZ", "Commodity_Prices", "Risk_Sentiment"],
        "news_sources": ["financial_news", "economic_calendar", "commodity_news"],
        "technical_focus": ["range_trading", "carry_trade", "correlation_analysis"]
    },
    "BTCUSD": {
        "asset_class": "cryptocurrency",
        "region": "global",
        "session": "24h",
        "volatility_profile": "very_high",
        "correlation_group": "crypto",
        "pip_value": 0.01,
        "min_lot_size": 0.01,
        "spread_typical": 3.0,
        "trading_hours": {"start": 0, "end": 23, "timezone": "UTC"},
        "high_impact_events": ["Crypto_Regulation", "Tech_News", "Market_Sentiment", "Institutional_Adoption"],
        "news_sources": ["crypto_news", "tech_news", "social_sentiment"],
        "technical_focus": ["momentum", "volatility_breakout", "support_resistance"]
    },
    "ETHUSD": {
        "asset_class": "cryptocurrency",
        "region": "global",
        "session": "24h",
        "volatility_profile": "very_high",
        "correlation_group": "crypto",
        "pip_value": 0.01,
        "min_lot_size": 0.01,
        "spread_typical": 5.0,
        "trading_hours": {"start": 0, "end": 23, "timezone": "UTC"},
        "high_impact_events": ["Crypto_Regulation", "Tech_News", "Market_Sentiment", "BTC_Correlation"],
        "news_sources": ["crypto_news", "tech_news", "social_sentiment"],
        "technical_focus": ["momentum", "volatility_breakout", "support_resistance"]
    },
    "JP225": {
        "asset_class": "equity_index",
        "region": "Asia",
        "session": "Asian",
        "volatility_profile": "medium",
        "correlation_group": "asian_indices",
        "pip_value": 1.0,
        "min_lot_size": 0.01,
        "spread_typical": 1.0,
        "trading_hours": {"start": 9, "end": 15, "timezone": "Asia/Tokyo"},
        "high_impact_events": ["BOJ", "Japanese_GDP", "PMI", "USD_JPY"],
        "news_sources": ["financial_news", "economic_calendar"],
        "technical_focus": ["trend_following", "mean_reversion", "session_trading"]
    }
}

# Asset class groupings for correlation analysis
ASSET_CLASS_GROUPS = {
    "equity_indices": ["SPX500", "DE40", "JP225"],
    "commodities": ["XAUUSD"],
    "forex": ["AUDNZD"],
    "cryptocurrency": ["BTCUSD", "ETHUSD"],
    "us_indices": ["SPX500"],
    "european_indices": ["DE40"],
    "asian_indices": ["JP225"],
    "precious_metals": ["XAUUSD"],
    "commodity_currencies": ["AUDNZD"],
    "crypto": ["BTCUSD", "ETHUSD"]
}
# (Add near other configuration variables like OANDA_API_KEY)
EODHD_API_KEY = " 68bafd7d44a7f025202650"
NEWSAPI_ORG_API_KEY = "abd8f43b808f42fdb8d28fb1c429af72"
FINNHUB_API_KEY = "d1b3ichr01qjhvtsbj70d1b3ichr01qjhvtsbj7g"
MARKETAUX_API_KEY = "CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql"

# You can change timeframes and main timeframe here
# --- CODE MỚI ---
#TIMEFRAMES = ["H4", "D1", "W1"]  # fallback if symbol doesn't have individual map
PRIMARY_TIMEFRAME = "H4"
# Add these 2 lines to configuration
# Add these thresholds to common configuration
MIN_F1_SCORE_GATE = 0.55       # Minimum F1-Score threshold
MAX_STD_F1_GATE = 0.12         # Maximum F1 standard deviation threshold
MIN_ACCURACY_GATE = 0.54       # Minimum win rate threshold
MIN_SAMPLES_GATE = 200         # Minimum sample count threshold for training
MAX_RETRAIN_ATTEMPTS = 3
ML_CONFIG = {
    "ENSEMBLE_MODELS": ["xgboost", "lightgbm", "lstm", "random_forest", "logistic_regression", "knn"],
    "CONFIDENCE_THRESHOLD": 0.60,              # Increased threshold for better quality trades
    "MIN_CONFIDENCE_TRADE": 0.55,              # Higher minimum confidence
    "CLOSE_ON_CONFIDENCE_DROP_THRESHOLD": 0.40,# Lower threshold for early exit
    "CONFIDENCE_CHECK_GRACE_PERIOD_CANDLES": 8, # Reduced grace period for faster response
    "LOOKBACK_PERIODS": [21,55,89,144,233],    # add 233 for long-term trend
    "LSTM_SEQUENCE_LENGTH": 60,                # 50–60 H4 helps look further
    "FEATURE_SELECTION_TOP_K": 60,             # Increased features for better performance
    "CV_N_SPLITS": 7,                           # More CV splits for better validation
    "EMA_SHORT_TERM_PERIOD_FOR_CLOSE": 20,    # Faster EMA for quicker response
    "MIN_SAMPLES_FOR_TRAINING": 300,           # Increased minimum samples for stability
    "MAX_CORRELATION_THRESHOLD": 0.90,        # Stricter correlation threshold
    # New optimization parameters
    "EARLY_STOPPING_PATIENCE": 15,            # Early stopping patience
    "VALIDATION_SPLIT": 0.2,                  # Validation split ratio
    "BATCH_SIZE": 64,                          # Batch size for neural networks
    "LEARNING_RATE_DECAY": 0.95,              # Learning rate decay factor
    "DROPOUT_RATE": 0.3,                       # Dropout rate for regularization
    "L2_REGULARIZATION": 0.001,               # L2 regularization strength
    "FEATURE_IMPORTANCE_THRESHOLD": 0.01,     # Minimum feature importance threshold
    "MODEL_STACKING_ENABLED": True,           # Enable model stacking
    "STACKING_CV_FOLDS": 5,                    # CV folds for stacking
    "ENSEMBLE_WEIGHT_OPTIMIZATION": True       # Enable ensemble weight optimization
}

RISK_MANAGEMENT = {
    "MAX_RISK_PER_TRADE": 0.0075,        # 0.75%/trade (safe, durable)
    "MAX_PORTFOLIO_RISK": 0.12,          # portfolio risk ceiling 12%
    "VOLATILITY_LOOKBACK": 21,           # synchronize ATR/MA 21
    "DYNAMIC_POSITION_SIZING": True,
    "CORRELATION_THRESHOLD": 0.65,       # limit correlated trades
    "MAX_OPEN_POSITIONS": 20,             # 3–5; choose 4 for balance
    "TRAILING_STOP_MULTIPLIER": 2.8,     # 2.5–3.0×ATR; choose 2.8 to follow trend
    "SL_ATR_MULTIPLIER": 2.0,            # Increase from 1.5 to 2.0
    "BASE_RR_RATIO": 4.0,                # Increase basic R:R ratio to 4.0
    # NEW – loss discipline
    "DAILY_LOSS_LIMIT": 0.03,            # stop day if -3%
    "WEEKLY_LOSS_LIMIT": 0.06            # stop week if -6%
}


# === TRADE FILTERS (new) ===
TRADE_FILTERS = {
    "SKIP_NEAR_HIGH_IMPACT_EVENTS": True,  # skip ±2h around major news
    "EVENT_BUFFER_HOURS": 2,
    "AVOID_WEEKEND": True,                  # don't open new trades near weekend
    "SEND_PRE_CHECK_STATUS_ALERT": True
}


# === OPTUNA (Enhanced Optimization) ===
OPTUNA_CONFIG = {
    "N_TRIALS": 100,     # Increased trials for better optimization
    "TIMEOUT_SEC": 1200, # Increased timeout for thorough search
    "PRUNING_ENABLED": True,  # Enable pruning for efficiency
    "SAMPLER": "TPE",    # Tree-structured Parzen Estimator
    "EARLY_STOPPING_ROUNDS": 10
}

# === TIMEFRAME MAPPING (mới) ===
# Giữ biến cũ làm default để không phá chỗ khác:
PRIMARY_TIMEFRAME_DEFAULT = PRIMARY_TIMEFRAME  # thường là "H4"

# Theo tài sản, chọn primary TF phù hợp:
# TÌM VÀ THAY THẾ BIẾN NÀY

# Optimized timeframe mapping based on asset characteristics
PRIMARY_TIMEFRAME_BY_SYMBOL = {
    # --- EQUITY INDICES (Fast reaction during session hours) ---
    "SPX500": "H1",  # US market - fast reaction to news
    "DE40":   "H1",  # European market - session-based trading
    "JP225":  "H1",  # Asian market - session-based trading

    # --- COMMODITIES (Swing trading following trends) ---
    "XAUUSD": "H4",  # Gold - trend following, less noise

    # --- FOREX (Swing trading with carry considerations) ---
    "AUDNZD": "H4",  # Cross pair - swing trading

    # --- CRYPTO (Fast reaction to volatility) ---
    "BTCUSD": "H1",  # Bitcoin - high volatility, fast moves
    "ETHUSD": "H1",  # Ethereum - high volatility, fast moves
}

# Secondary timeframes for multi-timeframe analysis
SECONDARY_TIMEFRAMES_BY_SYMBOL = {
    "SPX500": ["H4", "D1"],  # Trend context
    "DE40":   ["H4", "D1"],  # Trend context
    "JP225":  ["H4", "D1"],  # Trend context
    "XAUUSD": ["H1", "D1"],  # Entry timing + trend
    "AUDNZD": ["H1", "D1"],  # Entry timing + trend
    "BTCUSD": ["M15", "H4"], # Scalping + swing
    "ETHUSD": ["M15", "H4"], # Scalping + swing
}

# Optimal lookback periods by asset class
LOOKBACK_PERIODS_BY_ASSET_CLASS = {
    "equity_index": [21, 55, 89, 144],  # Standard Fibonacci
    "commodity": [21, 55, 89, 144, 233],  # Longer trends
    "forex": [21, 55, 89],  # Shorter for forex
    "cryptocurrency": [14, 21, 55, 89],  # Crypto-specific
}

# Enhanced timeframe mapping with optimized session awareness
ENHANCED_TIMEFRAME_MAPPING = {
    "SPX500": {
        "primary": "H1",
        "secondary": ["H4", "D1"],
        "session_hours": {"start": 9, "end": 16, "timezone": "US/Eastern"},
        "weekend_guard": True,
        "pre_market": {"start": 4, "end": 9},
        "after_hours": {"start": 16, "end": 20},
        "high_volatility_periods": [{"start": 9, "end": 11}, {"start": 14, "end": 16}],
        "news_sensitivity": "high",
        "gap_risk": True
    },
    "DE40": {
        "primary": "H1", 
        "secondary": ["H4", "D1"],
        "session_hours": {"start": 9, "end": 17, "timezone": "Europe/Berlin"},
        "weekend_guard": True,
        "lunch_break": {"start": 12, "end": 13},
        "high_volatility_periods": [{"start": 9, "end": 11}, {"start": 15, "end": 17}],
        "news_sensitivity": "high",
        "gap_risk": True
    },
    "XAUUSD": {
        "primary": "H4",
        "secondary": ["H1", "D1"],
        "session_hours": {"start": 0, "end": 23, "timezone": "UTC"},  # 24h market
        "weekend_guard": True,
        "high_activity_hours": [{"start": 8, "end": 10}, {"start": 13, "end": 15}, {"start": 20, "end": 22}],
        "news_sensitivity": "very_high",
        "gap_risk": False,
        "overnight_risk": True
    },
    "AUDNZD": {
        "primary": "H4",
        "secondary": ["H1", "D1"],
        "session_hours": {"start": 21, "end": 7, "timezone": "UTC"},  # Asian session
        "weekend_guard": True,
        "overlap_hours": [{"start": 21, "end": 1}, {"start": 7, "end": 9}],
        "news_sensitivity": "medium",
        "gap_risk": False,
        "carry_trade_focus": True
    },
    "BTCUSD": {
        "primary": "H1",
        "secondary": ["M15", "H4"],
        "session_hours": {"start": 0, "end": 23, "timezone": "UTC"},  # 24/7
        "weekend_guard": False,  # Crypto trades on weekends
        "high_volatility_hours": [{"start": 8, "end": 12}, {"start": 20, "end": 24}],
        "news_sensitivity": "very_high",
        "gap_risk": False,
        "weekend_trading": True
    },
    "ETHUSD": {
        "primary": "H1",
        "secondary": ["M15", "H4"],
        "session_hours": {"start": 0, "end": 23, "timezone": "UTC"},  # 24/7
        "weekend_guard": False,  # Crypto trades on weekends
        "high_volatility_hours": [{"start": 8, "end": 12}, {"start": 20, "end": 24}],
        "news_sensitivity": "very_high",
        "gap_risk": False,
        "weekend_trading": True
    },
    "JP225": {
        "primary": "H1",
        "secondary": ["H4", "D1"],
        "session_hours": {"start": 9, "end": 15, "timezone": "Asia/Tokyo"},
        "weekend_guard": True,
        "lunch_break": {"start": 11, "end": 12},
        "high_volatility_periods": [{"start": 9, "end": 11}, {"start": 13, "end": 15}],
        "news_sensitivity": "high",
        "gap_risk": True
    }
}

# Bộ TF đi kèm theo primary
TIMEFRAME_SET_BY_PRIMARY = {
    "H4": ["H4","D1","W1"],   # ra quyết định H4; D1/W1 lọc xu hướng
    "H1": ["H1","H4","D1"],   # ra quyết định H1; H4/D1 làm bối cảnh
    "D1": ["D1","W1"]         # nếu dùng D1 làm primary
}
WEEKEND_CLOSE_CONFIG = {
    "ENABLED": True,
    "CLOSE_DAY_UTC": 4,          # 4 = Thứ 6 (giữ nguyên)
    "CLOSE_HOUR_UTC": 17,        # Sửa thành 17:00 UTC
    "CLOSE_MINUTE_UTC": 0
}
MODEL_DIR = f"saved_models_{PRIMARY_TIMEFRAME.lower()}"
os.makedirs(MODEL_DIR, exist_ok=True)
# --- Bước 1: Định nghĩa "Giao diện" chung cho tất cả các nhà cung cấp ---
class NewsProvider(ABC):
    """
    Lớp trừu tượng (Interface) định nghĩa các hàm mà mọi nhà cung cấp tin tức phải có.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key or "DÁN_KEY" in self.api_key:
            self.enabled = False
        else:
            self.enabled = True

    @abstractmethod
    async def fetch_news(self, session, symbol: str, stock_map: dict):
        """
        Get news for a specific symbol. Must return a list of dictionaries
        in standardized format.
        """
        pass

    def _standardize_news(self, source: str, title: str, summary: str, url: str, published_at: datetime):
        """Helper function to create news dictionary in common format."""
        return {
            "source": source,
            "title": title,
            "summary": summary,
            "url": url,
            "published_at": published_at
        }

class FinnhubProvider(NewsProvider):
    async def fetch_news(self, session, symbol: str, stock_map: dict):
        if not self.enabled: return []
        stock_symbol = stock_map.get(symbol, symbol)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={stock_symbol}&from={start_date}&to={end_date}&token={self.api_key}"
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200: return []
                news_items = await response.json()
                return [self._standardize_news(
                    source="Finnhub", title=item.get('headline'), summary=item.get('summary'),
                    url=item.get('url'), published_at=datetime.fromtimestamp(item.get('datetime'))
                ) for item in news_items[:10] if item.get('headline')]
        except Exception as e:
            logging.error(f"❌ Finnhub error: {e}")
            return []

# TÌM VÀ THAY THẾ TOÀN BỘ LỚP NÀY

class EODHDProvider(NewsProvider):
    def __init__(self, api_key):
        super().__init__(api_key)
        if self.enabled:
            try:
                # <<< THAY ĐỔI Ở ĐÂY: Dùng eod thay vì eodhd >>>
                self.client = eod.EodHistoricalData(api_key)
            except Exception as e:
                logging.error(f"❌ EODHD client initialization error: {e}")
                self.enabled = False

    async def fetch_news(self, session, symbol: str, stock_map: dict):
        if not self.enabled: return []
        stock_symbol = stock_map.get(symbol, symbol)
        
        # EODHD không cung cấp tin tức qua API này, mà là dữ liệu cơ bản
        # Giữ nguyên logic cũ để lấy dữ liệu cơ bản
        try:
            loop = asyncio.get_running_loop()
            # Sử dụng một hàm khác phù hợp hơn nếu có, hoặc giữ nguyên get_fundamental_equity
            fundamental_data = await loop.run_in_executor(
                None, 
                self.client.get_fundamental_equity, 
                stock_symbol
            )

            market_cap = fundamental_data.get('Highlights', {}).get('MarketCapitalization', 'N/A')
            pe_ratio = fundamental_data.get('Highlights', {}).get('PERatio', 'N/A')
            beta = fundamental_data.get('Technicals', {}).get('Beta', 'N/A')
            summary = f"Vốn hóa: {market_cap}, P/E: {pe_ratio}, Beta: {beta}"

            return [self._standardize_news(
                source="EODHD", title=f"Dữ liệu cơ bản cho {stock_symbol}",
                summary=summary, url="", published_at=datetime.now()
            )]
        except Exception as e:
            # Lỗi này thường xảy ra khi gói Free không hỗ trợ symbol (ví dụ Forex)
            # print(f"ℹ️ Lưu ý EODHD: {e}") # Bỏ comment để debug nếu cần
            return []

class MarketauxProvider(NewsProvider):
    async def fetch_news(self, session, symbol: str, stock_map: dict):
        if not self.enabled: return []
        stock_symbol = stock_map.get(symbol, symbol)
        url = f"https://api.marketaux.com/v1/news/all?symbols={stock_symbol}&filter_entities=true&limit=5&api_token={self.api_key}"
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200: return []
                news_data = await response.json()
                news_items = news_data.get('data', [])
                return [self._standardize_news(
                    source="Marketaux", title=item.get('title'), summary=item.get('snippet'),
                    url=item.get('url'), published_at=pd.to_datetime(item.get('published_at')).tz_localize(None)
                ) for item in news_items if item.get('title')]
        except Exception as e:
            logging.error(f"❌ Marketaux error: {e}")
            return []
# DÁN TOÀN BỘ LỚP NÀY VÀO FILE CỦA BẠN

class NewsApiOrgProvider(NewsProvider):
    def __init__(self, api_key):
        super().__init__(api_key)
        if self.enabled:
            try:
                # Thư viện này không hỗ trợ async, nên ta sẽ khởi tạo client ở đây
                self.client = NewsApiClient(api_key=api_key)
            except Exception as e:
                logging.error(f"❌ NewsApiClient initialization error: {e}")
                self.enabled = False

    async def fetch_news(self, session, symbol: str, stock_map: dict):
        if not self.enabled: return []
        
        # Lấy query phù hợp, ví dụ SPX500 thì tìm "S&P 500"
        query = stock_map.get(symbol, symbol)
        
        try:
            loop = asyncio.get_running_loop()
            
            # Chạy hàm get_everything (vốn là hàm đồng bộ) trong một executor riêng
            # để không làm block vòng lặp async chính
            news_data = await loop.run_in_executor(
                None, 
                lambda: self.client.get_everything(q=query, language='en', sort_by='publishedAt', page_size=10)
            )
            
            news_items = news_data.get('articles', [])
            
            # Chuẩn hóa kết quả trả về
            return [self._standardize_news(
                source="NewsAPI.org", 
                title=item.get('title'), 
                summary=item.get('description'),
                url=item.get('url'), 
                published_at=pd.to_datetime(item.get('publishedAt')).tz_localize(None)
            ) for item in news_items if item.get('title')]
            
        except Exception as e:
            logging.error(f"❌ NewsAPI.org error: {e}")
            return []
# Lớp này không thay đổi
class NewsEconomicManager:
    def __init__(self):
        self.te_key = "a284ad0cdba547c:p5oyv77j6kovqhv:a284ad0cdba547c:p5oyv77j6kovqhv"
        self.llm_analyzer = LLMSentimentAnalyzer(api_key=GOOGLE_AI_API_KEY)
        self.news_cache = {}

        # Danh sách các nhà cung cấp tin tức
        self.news_providers = [
            FinnhubProvider(FINNHUB_API_KEY),
            EODHDProvider(EODHD_API_KEY),
            MarketauxProvider(MARKETAUX_API_KEY),
            NewsApiOrgProvider(NEWSAPI_ORG_API_KEY),
        ]
        
        # Mapping symbol của bạn sang symbol chuẩn cho các API chứng khoán
        self.stock_map = {
            "SPX500": "SPY", "DE40": "DAX", "JP225": "NIKKEI"
        }

        if TRADING_ECONOMICS_AVAILABLE and te:
            try:
                te.login(self.te_key)
                logging.info("✅ Trading Economics login successful!")
            except Exception as e:
                logging.error(f"❌ Trading Economics login error: {e}")
        else:
            logging.warning("Trading Economics not available - economic calendar features disabled")
        self.economic_calendar_cache = []
        self.last_calendar_fetch_date = None
    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM get_economic_calendar BẰNG PHIÊN BẢN NÀY

    def get_economic_calendar(self, init_date=None, end_date=None):
        """
        Lấy lịch kinh tế từ Trading Economics.
        NÂNG CẤP: Tích hợp caching để chỉ gọi API 1 lần mỗi ngày.
        """
        # <<< LOGIC CACHING BẮT ĐẦU TẠI ĐÂY >>>
        today = datetime.utcnow().date()
        # Nếu cache đã có và được lấy trong hôm nay, dùng lại cache
        if self.last_calendar_fetch_date == today and self.economic_calendar_cache:
            # print("   [Cache] Sử dụng lịch kinh tế từ cache.")
            return self.economic_calendar_cache
        # <<< KẾT THÚC LOGIC CACHING >>>

        try:
            # Nếu không có cache, tiến hành gọi API như bình thường
            logging.info("   [API Call] Fetching new economic calendar for today...")
            today_utc = datetime.utcnow()
            if init_date is None:
                init_date = (today_utc - timedelta(days=2)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = (today_utc + timedelta(days=2)).strftime('%Y-%m-%d')

            if not TRADING_ECONOMICS_AVAILABLE or not te:
                logging.warning("Trading Economics not available - returning empty calendar")
                return []
            
            calendar_raw = te.getCalendarData(initDate=init_date, endDate=end_date)
            if not isinstance(calendar_raw, list):
                return []
            
            high_impact_keywords = ["NFP", "CPI", "FOMC", "Interest Rate", "GDP"]
            filtered_events = []
            for event in calendar_raw:
                importance = str(event.get("Importance", "")).lower()
                event_name = str(event.get("Event", "")).lower()
                is_high_importance = (importance == "high")
                is_keyword_match = any(keyword.lower() in event_name for keyword in high_impact_keywords)

                if is_high_importance or is_keyword_match:
                    filtered_events.append(event)
            
            # <<< LƯU KẾT QUẢ VÀO CACHE SAU KHI GỌI API THÀNH CÔNG >>>
            self.economic_calendar_cache = filtered_events
            self.last_calendar_fetch_date = today
            # <<< KẾT THÚC LƯU CACHE >>>

            return filtered_events
        except Exception as e:
            logging.error(f"❌ Error fetching economic calendar from Trading Economics: {e}")
            # If error, return old cache (if available) instead of empty list
            return self.economic_calendar_cache if self.economic_calendar_cache else []
    async def get_aggregated_news(self, symbol: str):
        """
        Get news from ALL providers asynchronously,
        aggregate and remove duplicates.
        """
        # Check cache first
        cached_data = self.news_cache.get(symbol)
        if cached_data and (datetime.now() - cached_data['timestamp']).total_seconds() < NEWS_CACHE_TIMEOUT:
            return cached_data['news']

        logging.info(f"📰 Starting news aggregation for {symbol} from multiple sources...")
        async with aiohttp.ClientSession() as session:
            tasks = [provider.fetch_news(session, symbol, self.stock_map) for provider in self.news_providers if provider.enabled]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_news = []
        for res in results:
            if isinstance(res, list):
                all_news.extend(res)
        
        # Loại bỏ các bài báo trùng lặp dựa trên tiêu đề
        unique_news = list({item['title'].lower(): item for item in all_news}.values())
        
        # Sắp xếp tin tức theo thời gian mới nhất
        unique_news.sort(key=lambda x: x['published_at'], reverse=True)
        
        print(f"📰 Tổng hợp xong, có {len(unique_news)} tin tức độc nhất cho {symbol}.")

        # Lưu vào cache
        self.news_cache[symbol] = {'news': unique_news, 'timestamp': datetime.now()}
        
        return unique_news

    def analyze_sentiment(self, news_items):
        if not news_items:
            return {"score": 0.0, "reasoning": "Không có tin tức."}
        return self.llm_analyzer.analyze_sentiment_of_news(news_items)

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP NewsEconomicManager

    def add_economic_event_features(self, df, symbol):
        """
        NÂNG CẤP: Thêm các đặc trưng về sự kiện kinh tế vào DataFrame,
        đã sửa lỗi xác định tiền tệ cho chỉ số và hàng hóa.
        """
        logging.info(f"   [Features] Starting to add economic event features for {symbol}...")
        df["is_near_high_impact_event"] = 0

        # Lấy lịch kinh tế cho khoảng thời gian của DataFrame
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")

        try:
            if not TRADING_ECONOMICS_AVAILABLE or not te:
                logging.warning("Trading Economics not available - skipping economic event features")
                return df
            
            calendar_raw = te.getCalendarData(initDate=start_date, endDate=end_date)
            if not isinstance(calendar_raw, list) or not calendar_raw:
                return df
        except Exception as e:
            print(f"   [Features] ⚠️ Lỗi khi lấy lịch kinh tế lịch sử: {e}")
            return df

        # Chuẩn hóa các sự kiện kinh tế
        high_impact_events = []
        high_impact_keywords = ["NFP", "CPI", "FOMC", "Interest Rate", "GDP"]
        for event in calendar_raw:
            try:
                is_high_impact = event.get("Importance") == "High" or any(
                    keyword.lower() in event.get("Event", "").lower()
                    for keyword in high_impact_keywords
                )
                if is_high_impact:
                    event_time = pd.to_datetime(event["Date"]).tz_localize("UTC")
                    event_currency = event.get("Currency", "").upper()
                    high_impact_events.append({"time": event_time, "currency": event_currency})
            except:
                continue

        if not high_impact_events:
            return df

        # <<< LOGIC SỬA LỖI: Xác định đúng tiền tệ liên quan >>>
        SYMBOL_CURRENCY_MAP = {
            "SPX500": ["USD"], "DE40": ["EUR"], "JP225": ["JPY"],
            "XAUUSD": ["USD"], "ETHUSD": ["USD"]
        }
        relevant_currencies = []
        if symbol in SYMBOL_CURRENCY_MAP:
            relevant_currencies = SYMBOL_CURRENCY_MAP[symbol]
        elif len(symbol) >= 6:
            relevant_currencies.extend([symbol[:3].upper(), symbol[3:].upper()])
        # <<< KẾT THÚC SỬA LỖI >>>

        if not relevant_currencies:
            return df

        # Đánh dấu các hàng gần sự kiện
        df_utc_index = df.index.tz_convert("UTC")
        for event in high_impact_events:
            # Chỉ xử lý nếu sự kiện liên quan đến symbol
            if event["currency"] in relevant_currencies:
                event_start_window = event["time"] - timedelta(hours=2)
                event_end_window = event["time"] + timedelta(hours=2)
                mask = (df_utc_index >= event_start_window) & (df_utc_index <= event_end_window)
                df.loc[mask, "is_near_high_impact_event"] = 1

        logging.info(f"   [Features] ✅ Economic event features processing completed. {df['is_near_high_impact_event'].sum()} candles marked.")
        return df
class LLMSentimentAnalyzer:
    def __init__(self, api_key):
        if not api_key or "DÁN_API_KEY" in api_key:
            logging.warning("⚠️ Google AI API Key not configured. LLM features will be disabled.")
            self.model = None
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logging.info("✅ Successfully connected to Gemini API.")
        except Exception as e:
            logging.error(f"❌ Error connecting to Gemini API: {e}")
            self.model = None

    def analyze_sentiment_of_news(self, news_items: list):
        if not self.model:
            return {"score": 0.0, "reasoning": "LLM not operational."}

        # 1. Định dạng lại tin tức để đưa vào prompt
        formatted_news = ""
        for i, item in enumerate(news_items[:5]): # Chỉ lấy 5 tin tức đầu tiên
            formatted_news += f"Tin {i+1}: {item.get('title', '')}\nNội dung: {item.get('summary', '')}\n\n"

        if not formatted_news:
            return {"score": 0.0, "reasoning": "Không có tin tức để phân tích."}

        # 2. Thiết kế Prompt (câu lệnh) cho LLM
        prompt = f"""
        Bạn là một nhà phân tích tài chính chuyên nghiệp cho một quỹ phòng hộ.
        Hãy phân tích các tin tức dưới đây và đánh giá tác động tổng thể của chúng đến thị trường.

        Tin tức:
        ---
        {formatted_news}
        ---

        Dựa trên những tin tức này, hãy cung cấp một phân tích theo định dạng JSON sau.
        Chỉ trả về duy nhất một khối JSON, không giải thích gì thêm.

        {{
          "sentiment_score": <một số float từ -1.0 (rất tiêu cực) đến 1.0 (rất tích cực)>,
          "reasoning": "<một câu tóm tắt ngắn gọn lý do cho điểm số của bạn>"
        }}
        """

        # 3. Gửi yêu cầu tới Gemini và xử lý kết quả
        try:
            response = self.model.generate_content(prompt)
            # Trích xuất phần JSON từ text trả về
            json_text = response.text.strip().replace('```json', '').replace('```', '')
            result = json.loads(json_text)

            # Đảm bảo các khóa tồn tại
            result.setdefault('score', result.get('sentiment_score', 0.0))
            result.setdefault('reasoning', 'Không có lý do được cung cấp.')

            return result

        except Exception as e:
            logging.error(f"❌ Error analyzing sentiment with LLM: {e}")
            return {"score": 0.0, "reasoning": "LLM processing error."}


# Lớp này không thay đổi
class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.symbol_configs = self._initialize_symbol_configs()
        
    def _initialize_symbol_configs(self):
        """Initialize symbol-specific feature configurations"""
        return {
            "SPX500": {
                "rsi_periods": [14, 21],
                "ema_periods": [5, 10, 20, 50, 100, 200],
                "bb_periods": [10, 20, 50],
                "atr_period": 14,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "stoch_params": {"k": 14, "d": 3},
                "volume_weight": 0.8,
                "session_features": True,
                "gap_features": True,
                "volatility_adjustment": True
            },
            "DE40": {
                "rsi_periods": [14, 21],
                "ema_periods": [5, 10, 20, 50, 100, 200],
                "bb_periods": [10, 20, 50],
                "atr_period": 14,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "stoch_params": {"k": 14, "d": 3},
                "volume_weight": 0.8,
                "session_features": True,
                "gap_features": True,
                "volatility_adjustment": True
            },
            "XAUUSD": {
                "rsi_periods": [14, 21, 34],
                "ema_periods": [5, 10, 20, 50, 100, 200, 300],
                "bb_periods": [20, 50, 100],
                "atr_period": 21,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "stoch_params": {"k": 14, "d": 3},
                "volume_weight": 0.3,
                "session_features": False,
                "gap_features": False,
                "volatility_adjustment": True,
                "fibonacci_levels": True
            },
            "AUDNZD": {
                "rsi_periods": [14, 21],
                "ema_periods": [5, 10, 20, 50, 100],
                "bb_periods": [10, 20, 50],
                "atr_period": 14,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "stoch_params": {"k": 14, "d": 3},
                "volume_weight": 0.5,
                "session_features": True,
                "gap_features": False,
                "volatility_adjustment": False,
                "carry_trade_features": True
            },
            "BTCUSD": {
                "rsi_periods": [14, 21],
                "ema_periods": [5, 10, 20, 50],
                "bb_periods": [10, 20],
                "atr_period": 14,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "stoch_params": {"k": 14, "d": 3},
                "volume_weight": 1.0,
                "session_features": False,
                "gap_features": False,
                "volatility_adjustment": True,
                "crypto_specific": True
            },
            "ETHUSD": {
                "rsi_periods": [14, 21],
                "ema_periods": [5, 10, 20, 50],
                "bb_periods": [10, 20],
                "atr_period": 14,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "stoch_params": {"k": 14, "d": 3},
                "volume_weight": 1.0,
                "session_features": False,
                "gap_features": False,
                "volatility_adjustment": True,
                "crypto_specific": True
            },
            "JP225": {
                "rsi_periods": [14, 21],
                "ema_periods": [5, 10, 20, 50, 100, 200],
                "bb_periods": [10, 20, 50],
                "atr_period": 14,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "stoch_params": {"k": 14, "d": 3},
                "volume_weight": 0.8,
                "session_features": True,
                "gap_features": True,
                "volatility_adjustment": True
            }
        }
    
    def _get_symbol_config(self, symbol):
        """Get configuration for specific symbol"""
        return self.symbol_configs.get(symbol, self.symbol_configs["SPX500"])

    def add_technical_indicators(df):
        # Nếu bạn có logic thực sự thì đưa vào đây. Tạm thời cho qua lỗi:
        return df

    # <<< SỬA LỖI DỨT ĐIỂM 1/2 >>>
    # Hàm này đã được sửa để không đặt tên cột RSI cứng nhắc nữa.
    # Nó sẽ tạo ra cột 'rsi' chung, giúp code linh hoạt hơn.
    def create_technical_features(self, df, symbol=None):
        """Create advanced technical indicators optimized for specific symbol"""
        config = self._get_symbol_config(symbol) if symbol else self.symbol_configs["SPX500"]
        metadata = SYMBOL_METADATA.get(symbol, {}) if symbol else {}
        
        # Basic Price Features
        df["hl2"] = (df["high"] + df["low"]) / 2
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
        df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

        # Volatility Features with symbol-specific periods
        atr_period = config["atr_period"]
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=atr_period)
        df["atr"] = atr.average_true_range()
        df["atr_normalized"] = df["atr"] / df["close"]
        
        # Asset class specific volatility adjustments
        if config["volatility_adjustment"]:
            df["atr_percentile"] = df["atr_normalized"].rolling(100).rank(pct=True)
            df["volatility_regime"] = pd.cut(df["atr_percentile"], 
                                           bins=[0, 0.25, 0.75, 1.0], 
                                           labels=["low", "normal", "high"])

        # Bollinger Bands with symbol-specific periods
        for period in config["bb_periods"]:
            bb = BollingerBands(df["close"], window=period, window_dev=2)
            df[f"bb_upper_{period}"] = bb.bollinger_hband()
            df[f"bb_lower_{period}"] = bb.bollinger_lband()
            df[f"bb_width_{period}"] = bb.bollinger_wband()
            df[f"bb_position_{period}"] = (df["close"] - bb.bollinger_lband()) / (
                bb.bollinger_hband() - bb.bollinger_lband()
            )
            # Squeeze detection
            df[f"bb_squeeze_{period}"] = (df[f"bb_width_{period}"] < df[f"bb_width_{period}"].rolling(20).mean() * 0.8).astype(int)

        # Moving Averages with symbol-specific periods
        for period in config["ema_periods"]:
            ema = EMAIndicator(df["close"], window=period)
            df[f"ema_{period}"] = ema.ema_indicator()
            if period > 5:
                df[f"ema_ratio_{period}"] = df["close"] / df[f"ema_{period}"]
                df[f"ema_distance_{period}"] = (df["close"] - df[f"ema_{period}"]) / df["atr"]
        
        # EMA crossovers
        if len(config["ema_periods"]) >= 2:
            fast_period = min(config["ema_periods"])
            slow_period = max([p for p in config["ema_periods"] if p > fast_period])
            df[f"ema_cross_{fast_period}_{slow_period}"] = (
                (df[f"ema_{fast_period}"] > df[f"ema_{slow_period}"]) & 
                (df[f"ema_{fast_period}"].shift(1) <= df[f"ema_{slow_period}"].shift(1))
            ).astype(int)

        # RSI with multiple periods
        for period in config["rsi_periods"]:
            rsi = RSIIndicator(df["close"], window=period)
            df[f"rsi_{period}"] = rsi.rsi()
            df[f"rsi_oversold_{period}"] = (df[f"rsi_{period}"] < 30).astype(int)
            df[f"rsi_overbought_{period}"] = (df[f"rsi_{period}"] > 70).astype(int)
            
            # RSI divergence detection
            df[f"rsi_divergence_{period}"] = self._detect_rsi_divergence(df, period)
        
        # Primary RSI (for compatibility)
        primary_rsi_period = config["rsi_periods"][0]
        df["rsi"] = df[f"rsi_{primary_rsi_period}"]
        df["rsi_oversold"] = df[f"rsi_oversold_{primary_rsi_period}"]
        df["rsi_overbought"] = df[f"rsi_overbought_{primary_rsi_period}"]

        # MACD with symbol-specific parameters
        macd_params = config["macd_params"]
        macd = MACD(df["close"], 
                   window_fast=macd_params["fast"], 
                   window_slow=macd_params["slow"], 
                   window_sign=macd_params["signal"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        df["macd_cross"] = ((df["macd"] > df["macd_signal"]) & 
                          (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)

        # ADX Trend Strength
        adx = ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["adx_trending"] = (df["adx"] > 25).astype(int)
        df["adx_strong_trend"] = (df["adx"] > 50).astype(int)

        # Stochastic Oscillator with symbol-specific parameters
        stoch_params = config["stoch_params"]
        stoch = StochasticOscillator(df["high"], df["low"], df["close"], 
                                   window=stoch_params["k"], k_period=stoch_params["k"], d_period=stoch_params["d"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
        df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
        
        # Asset class specific features
        if config.get("fibonacci_levels") and symbol == "XAUUSD":
            df = self._add_fibonacci_features(df)
        
        if config.get("crypto_specific") and symbol in ["BTCUSD", "ETHUSD"]:
            df = self._add_crypto_features(df)
        
        if config.get("carry_trade_features") and symbol == "AUDNZD":
            df = self._add_carry_trade_features(df)
        
        # Ensure required indicators exist
        required_indicators = ["rsi", "atr", "macd", "adx"]
        for indicator in required_indicators:
            if indicator not in df.columns:
                df[indicator] = 0

        return df
    
    def _detect_rsi_divergence(self, df, period):
        """Detect RSI divergence patterns"""
        rsi_col = f"rsi_{period}"
        if rsi_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Look for divergence over 20 periods
        lookback = 20
        divergence = pd.Series(0, index=df.index)
        
        for i in range(lookback, len(df)):
            price_slice = df["close"].iloc[i-lookback:i]
            rsi_slice = df[rsi_col].iloc[i-lookback:i]
            
            # Bullish divergence: lower lows in price, higher lows in RSI
            price_low_idx = price_slice.idxmin()
            rsi_low_idx = rsi_slice.idxmin()
            
            if price_low_idx != rsi_low_idx:
                recent_price_low = price_slice.iloc[-5:].min()
                recent_rsi_low = rsi_slice.iloc[-5:].min()
                
                if recent_price_low < price_slice.min() and recent_rsi_low > rsi_slice.min():
                    divergence.iloc[i] = 1
        
        return divergence
    
    def _add_fibonacci_features(self, df):
        """Add Fibonacci retracement features for Gold"""
        # Calculate swing highs and lows
        swing_high = df["high"].rolling(20).max()
        swing_low = df["low"].rolling(20).min()
        
        # Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            df[f"fib_{level}"] = swing_low + (swing_high - swing_low) * level
            df[f"near_fib_{level}"] = (abs(df["close"] - df[f"fib_{level}"]) < df["atr"]).astype(int)
        
        return df
    
    def _add_crypto_features(self, df):
        """Add cryptocurrency-specific features"""
        # Crypto volatility features
        df["crypto_volatility"] = df["close"].rolling(24).std() / df["close"].rolling(24).mean()
        df["crypto_momentum"] = df["close"].pct_change(24)
        
        # Weekend effect (crypto trades 24/7)
        if isinstance(df.index, pd.DatetimeIndex):
            df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
            df["weekend_volatility"] = df["crypto_volatility"] * df["is_weekend"]
        
        return df
    
    def _add_carry_trade_features(self, df):
        """Add carry trade specific features for forex pairs"""
        # Interest rate differential proxy (simplified)
        df["carry_signal"] = (df["close"] > df["close"].rolling(50).mean()).astype(int)
        
        # Risk-on/Risk-off sentiment
        df["risk_sentiment"] = df["close"].rolling(20).mean() / df["close"].rolling(100).mean()
        
        return df
    # Đặt hàm này bên trong class AdvancedFeatureEngineer của bạn
    def create_wyckoff_features(self, df, atr_multiplier=DEFAULT_ATR_MULTIPLIER, range_window=DEFAULT_RANGE_WINDOW):
        """
        Create numerical features based on Wyckoff method principles.
        """
        logging.info(f"   [Features] Starting Wyckoff features creation...")
        if "atr" not in df.columns:
            df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

        # 1. Identify Trading Range
        df['rolling_max'] = df['high'].rolling(window=range_window).max()
        df['rolling_min'] = df['low'].rolling(window=range_window).min()
        df['is_in_range'] = (df['close'] < df['rolling_max']) & (df['close'] > df['rolling_min'])

        # 2. Quantify Key Events (Spring & Upthrust)
        # Spring: Price breaks below support zone then quickly returns
        df['spring_signal'] = (
            (df['is_in_range'].shift(1) == True) & # Previous candle was in range
            (df['low'] < df['rolling_min'].shift(1)) &      # Current candle breaks below
            (df['close'] > df['rolling_min'].shift(1))     # and closes back inside range
        ).astype(int)

        # Upthrust: Price breaks above resistance zone then quickly returns
        df['upthrust_signal'] = (
            (df['is_in_range'].shift(1) == True) & # Previous candle was in range
            (df['high'] > df['rolling_max'].shift(1)) &     # Current candle breaks above
            (df['close'] < df['rolling_max'].shift(1))     # and closes back inside range
        ).astype(int)

        # 3. Volume Spread Analysis (VSA Features)
        df['price_spread'] = df['high'] - df['low']
        df['volume_ema_50'] = df['volume'].ewm(span=50, adjust=False).mean()
        df['is_high_volume'] = (df['volume'] > df['volume_ema_50'] * 2).astype(int)

        # No Demand signal: Rising candle, narrow spread, low volume -> weak buying
        df['no_demand_signal'] = (
            (df['close'] > df['open']) &
            (df['price_spread'] < df['price_spread'].rolling(20).mean() * 0.7) &
            (df['volume'] < df['volume_ema_50'] * 0.8)
        ).astype(int)

        # No Supply signal: Falling candle, narrow spread, low volume -> weak selling
        df['no_supply_signal'] = (
            (df['close'] < df['open']) &
            (df['price_spread'] < df['price_spread'].rolling(20).mean() * 0.7) &
            (df['volume'] < df['volume_ema_50'] * 0.8)
        ).astype(int)

        # 4. Identify Phases (simplified)
        # Phase 1/3 (Accumulation/Distribution): Low volatility
        df['atr_normalized'] = df['atr'] / df['close']
        df['low_volatility_phase'] = (df['atr_normalized'] < df['atr_normalized'].rolling(range_window).quantile(0.25)).astype(int)

        # Phase 2/4 (Mark-up/Mark-down): High volatility, trending
        df['high_volatility_phase'] = (df['atr_normalized'] > df['atr_normalized'].rolling(range_window).quantile(0.75)).astype(int)

        logging.info(f"   [Features] ✅ Wyckoff features creation completed.")
        return df
    def create_statistical_features(self, df):
        """Create statistical features"""
        # Returns với multiple periods
        for period in [1, 3, 5, 10, 20]:
            df[f"returns_{period}"] = df["close"].pct_change(period)
            df[f"log_returns_{period}"] = np.log(
                df["close"] / df["close"].shift(period)
            )

        # Rolling Statistics
        for window in [10, 20, 50]:
            df[f"rolling_mean_{window}"] = df["close"].rolling(window).mean()
            df[f"rolling_std_{window}"] = df["close"].rolling(window).std()
            df[f"rolling_skew_{window}"] = df["close"].rolling(window).skew()
            df[f"rolling_kurt_{window}"] = df["close"].rolling(window).kurt()

            # Z-Score
            df[f"zscore_{window}"] = (df["close"] - df[f"rolling_mean_{window}"]) / df[
                f"rolling_std_{window}"
            ]

        # High-Low Ratios
        for period in [5, 10, 20]:
            df[f"high_low_ratio_{period}"] = (
                df["high"].rolling(period).max() / df["low"].rolling(period).min()
            )
            df[f"close_position_{period}"] = (
                df["close"] - df["low"].rolling(period).min()
            ) / (df["high"].rolling(period).max() - df["low"].rolling(period).min())

        return df

    def create_pattern_features(self, df):
        """Create pattern recognition features"""
        # --- GỢI Ý CẢI THIỆN 4: NÂNG CẤP NHẬN DIỆN MẪU HÌNH ---
        # Giữ lại các mẫu hình cũ
        df["doji"] = (
            abs(df["open"] - df["close"]) <= (df["high"] - df["low"]) * 0.1
        ).astype(int)
        df["hammer"] = (
            (df["close"] > df["open"])
            & ((df["close"] - df["open"]) <= (df["high"] - df["low"]) * 0.3)
            & ((df["open"] - df["low"]) >= (df["high"] - df["low"]) * 0.6)
        ).astype(int)

        # Thêm mẫu hình nhấn chìm (Engulfing)
        prev_body = abs(df["open"].shift(1) - df["close"].shift(1))
        current_body = abs(df["open"] - df["close"])
        df["bullish_engulfing"] = (
            (df["close"] > df["open"])
            & (df["close"].shift(1) < df["open"].shift(1))
            & (df["close"] > df["open"].shift(1))
            & (df["open"] < df["close"].shift(1))
            & (current_body > prev_body)
        ).astype(int)
        df["bearish_engulfing"] = (
            (df["close"] < df["open"])
            & (df["close"].shift(1) > df["open"].shift(1))
            & (df["close"] < df["open"].shift(1))
            & (df["open"] > df["close"].shift(1))
            & (current_body > prev_body)
        ).astype(int)

        # Support/Resistance Levels
        df["support_level"] = df["low"].rolling(20).min()
        df["resistance_level"] = df["high"].rolling(20).max()
        df["near_support"] = (
            abs(df["close"] - df["support_level"]) <= df["atr"]
        ).astype(int)
        df["near_resistance"] = (
            abs(df["close"] - df["resistance_level"]) <= df["atr"]
        ).astype(int)

        # Trend Features
        df["higher_highs"] = (df["high"] > df["high"].shift(1)).rolling(5).sum()
        df["lower_lows"] = (df["low"] < df["low"].shift(1)).rolling(5).sum()
        df["trend_strength"] = df["higher_highs"] - df["lower_lows"]

        return df

    def create_volume_features(self, df):
        """Create volume-based features with EMA"""
        if "volume" in df.columns:
            # Volume Exponential Moving Average
            df["volume_ema"] = df["volume"].ewm(span=20, adjust=False).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ema"]

            # Price-Volume Relationship
            df["pv_trend"] = np.where(
                df["close"] > df["close"].shift(1), df["volume"], -df["volume"]
            )
            df["pv_cumulative"] = df["pv_trend"].rolling(20).sum()

        return df

    def create_market_microstructure_features(self, df):
        """Create market microstructure features"""
        # Bid-Ask Spread Proxy
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        df["spread_volatility"] = df["spread_proxy"].rolling(20).std()

        # Intraday Patterns
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            # Session Features (Asian, European, American)
            df["asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
            df["european_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
            df["american_session"] = ((df["hour"] >= 16) & (df["hour"] < 24)).astype(
                int
            )
        else:
            df["hour"] = 0
            df["day_of_week"] = 0
            df["asian_session"] = 0
            df["european_session"] = 0
            df["american_session"] = 0

        return df

    # Bên trong class AdvancedFeatureEngineer

    # <<< SỬA LẠI HÀM NÀY TRONG LỚP AdvancedFeatureEngineer >>>
    def create_all_features(self, df):
        """Create all features"""
        df = self.create_technical_features(df)
        df = self.create_statistical_features(df)
        df = self.create_pattern_features(df)
        df = self.create_volume_features(df)
        df = self.create_market_microstructure_features(df)
        df = self.create_wyckoff_features(df)

        # <<< CẢI TIẾN FEATURE ENGINEERING: Gọi hàm tạo feature trạng thái thị trường >>>
        df = self.create_market_regime_feature(df)

        zones = self._find_sd_zones(df)
        print(
            f"  Đã tìm thấy tổng cộng {len(zones)} vùng cho {df.name if hasattr(df, 'name') else 'current symbol'}."
        )
        df = self.create_supply_demand_features(df, zones)
        df = self.create_market_structure_signals(df)

        for horizon in [1, 3, 5]:
            df[f"label_{horizon}"] = (df["close"].shift(-horizon) > df["close"]).astype(
                int
            )

        return df
    # Đặt bên trong class AdvancedFeatureEngineer
    def _find_sd_zones(self, df, atr_multiplier=1.2):
        """
        Hàm nội bộ để xác định các vùng Cung (Supply) và Cầu (Demand).
        Bao gồm cả các vùng Đảo chiều và Tiếp diễn.
        """
        if "atr" not in df.columns or df["atr"].isnull().all():
            # print("   [Zones] Cột 'atr' bị thiếu hoặc toàn giá trị null. Bỏ qua tìm vùng.")
            return []

        zones = []
        # Bắt đầu từ nến thứ 2 và kết thúc trước nến cuối cùng để có thể truy cập i-1 và i+1
        for i in range(1, len(df) - 1):
            # Xác định nến "base" (thân nhỏ)
            is_base_candle = (
                abs(df["close"].iloc[i] - df["open"].iloc[i])
                < (df["high"].iloc[i] - df["low"].iloc[i]) * 0.6
            )
            if not is_base_candle:
                continue

            # Xác định nến mạnh trước và sau nến base
            move_before = abs(df["close"].iloc[i - 1] - df["open"].iloc[i - 1])
            move_after = abs(df["close"].iloc[i + 1] - df["open"].iloc[i + 1])

            # Chỉ lấy ATR hợp lệ
            atr_value = df["atr"].iloc[i]
            if pd.isna(atr_value) or atr_value == 0:
                continue
            atr_threshold = atr_value * atr_multiplier

            is_strong_move_before = move_before > atr_threshold
            is_strong_move_after = move_after > atr_threshold

            if is_strong_move_before and is_strong_move_after:
                # Xác định hướng của các nến mạnh
                is_bullish_before = df["close"].iloc[i - 1] > df["open"].iloc[i - 1]
                is_bullish_after = df["close"].iloc[i + 1] > df["open"].iloc[i + 1]

                # --- MẪU HÌNH ĐẢO CHIỀU ---
                # Drop-Base-Rally -> Vùng Cầu Đảo chiều (Demand Reversal)
                if not is_bullish_before and is_bullish_after:
                    zones.append(
                        {
                            "type": "demand_reversal",
                            "high": df["high"].iloc[i],
                            "low": df["low"].iloc[i],
                            "index": df.index[i],
                        }
                    )
                # Rally-Base-Drop -> Vùng Cung Đảo chiều (Supply Reversal)
                elif is_bullish_before and not is_bullish_after:
                    zones.append(
                        {
                            "type": "supply_reversal",
                            "high": df["high"].iloc[i],
                            "low": df["low"].iloc[i],
                            "index": df.index[i],
                        }
                    )

                # --- MẪU HÌNH TIẾP DIỄN ---
                # Rally-Base-Rally -> Vùng Cầu Tiếp diễn (Demand Continuation)
                elif is_bullish_before and is_bullish_after:
                    zones.append(
                        {
                            "type": "demand_continuation",
                            "high": df["high"].iloc[i],
                            "low": df["low"].iloc[i],
                            "index": df.index[i],
                        }
                    )
                # Drop-Base-Drop -> Vùng Cung Tiếp diễn (Supply Continuation)
                elif not is_bullish_before and not is_bullish_after:
                    zones.append(
                        {
                            "type": "supply_continuation",
                            "high": df["high"].iloc[i],
                            "low": df["low"].iloc[i],
                            "index": df.index[i],
                        }
                    )
        return zones

    def create_supply_demand_features(self, df, zones):
        """
        Tạo các feature Cung/Cầu.
        PHIÊN BẢN TỐI ƯU HÓA HIỆU NĂNG bằng cách vector hóa.
        """
        # --- Phần 1: Tính toán "in_zone" (giữ nguyên logic vòng lặp vì phức tạp) ---
        in_sr = np.zeros(len(df))
        in_dr = np.zeros(len(df))
        in_sc = np.zeros(len(df))
        in_dc = np.zeros(len(df))

        close_prices_np = df["close"].to_numpy()
        df_index_np = df.index

        for i in range(len(df)):
            current_price = close_prices_np[i]
            current_time = df_index_np[i]
            for zone in [z for z in zones if z["index"] < current_time]:
                if zone["low"] <= current_price <= zone["high"]:
                    if zone["type"] == "supply_reversal":
                        in_sr[i] = 1
                    elif zone["type"] == "demand_reversal":
                        in_dr[i] = 1
                    elif zone["type"] == "supply_continuation":
                        in_sc[i] = 1
                    elif zone["type"] == "demand_continuation":
                        in_dc[i] = 1
                    break

        df["in_supply_reversal"] = in_sr
        df["in_demand_reversal"] = in_dr
        df["in_supply_continuation"] = in_sc
        df["in_demand_continuation"] = in_dc

        # --- Phần 2: Tối ưu hóa tính toán "distance_to..." bằng vector hóa ---

        # Tách các vùng ra theo từng loại
        supply_rev_zones = [z for z in zones if z["type"] == "supply_reversal"]
        demand_rev_zones = [z for z in zones if z["type"] == "demand_reversal"]
        supply_cont_zones = [z for z in zones if z["type"] == "supply_continuation"]
        demand_cont_zones = [z for z in zones if z["type"] == "demand_continuation"]

        # Hàm trợ giúp để tạo "bản đồ" giá của các vùng
        def create_zone_map(df_index, zone_list, price_key):
            zone_map = pd.Series(np.nan, index=df_index)
            for zone in zone_list:
                zone_map.loc[zone["index"]] = zone[price_key]
            return (
                zone_map.ffill()
            )  # Forward fill để mỗi cây nến biết được vùng gần nhất trong quá khứ

        # Tạo bản đồ cho từng loại vùng
        sr_map = create_zone_map(df.index, supply_rev_zones, "low")
        dr_map = create_zone_map(df.index, demand_rev_zones, "high")
        sc_map = create_zone_map(df.index, supply_cont_zones, "low")
        dc_map = create_zone_map(df.index, demand_cont_zones, "high")

        # Tính toán khoảng cách đồng loạt (vectorized)
        dist_sr = (sr_map - df["close"]) / df["close"]
        dist_dr = (df["close"] - dr_map) / df["close"]
        dist_sc = (sc_map - df["close"]) / df["close"]
        dist_dc = (df["close"] - dc_map) / df["close"]

        # Chỉ giữ lại các khoảng cách hợp lệ (vùng cung phải ở trên, vùng cầu phải ở dưới)
        df["distance_to_supply_reversal"] = np.where(dist_sr > 0, dist_sr, np.nan)
        df["distance_to_demand_reversal"] = np.where(dist_dr > 0, dist_dr, np.nan)
        df["distance_to_supply_continuation"] = np.where(dist_sc > 0, dist_sc, np.nan)
        df["distance_to_demand_continuation"] = np.where(dist_dc > 0, dist_dc, np.nan)

        # --- Phần 3: Dọn dẹp cuối cùng ---
        # Dùng bfill để điền các giá trị NaN ở đầu (trước khi vùng đầu tiên xuất hiện)
        cols_to_fill = [
            "distance_to_supply_reversal",
            "distance_to_demand_reversal",
            "distance_to_supply_continuation",
            "distance_to_demand_continuation",
        ]
        df[cols_to_fill] = df[cols_to_fill].bfill()

        return df

    def create_market_structure_signals(self, df):
        """
        Tạo các tín hiệu dựa trên cấu trúc thị trường.
        PHIÊN BẢN MỚI: Sử dụng tính toán phân kỳ RSI được tối ưu hóa.
        """
        # 1. Tín hiệu kiệt sức từ Bollinger Bands (giữ nguyên)
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_exhaustion_sell"] = ((df["close"].shift(1) > df["bb_upper"].shift(1)) & (df["close"] < df["bb_upper"])).astype(int)
        df["bb_exhaustion_buy"] = ((df["close"].shift(1) < df["bb_lower"].shift(1)) & (df["close"] > df["bb_lower"])).astype(int)

        # 2. Tín hiệu Phân kỳ RSI (RSI Divergence) - GỌI HÀM MỚI
        if 'rsi_14' not in df.columns:
            df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()

        # <<< THAY THẾ TOÀN BỘ VÒNG LẶP lồng nhau bằng 1 dòng này >>>
        df = calculate_rsi_divergence_vectorized(df)

        # Dọn dẹp các cột tạm nếu có
        cols_to_drop = ['price_peak', 'price_trough', 'rsi_peak', 'rsi_trough', 'rsi_14']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

        return df
    # <<< THÊM HÀM NÀY VÀO TRONG LỚP AdvancedFeatureEngineer >>>
    def create_market_regime_feature(self, df, adx_threshold=DEFAULT_ADX_THRESHOLD, ema_period=DEFAULT_EMA_PERIOD):
        """
        Tạo feature số để xác định trạng thái thị trường.
        1: Uptrend, -1: Downtrend, 0: Sideways
        """
        logging.info("   [Features] Starting market state feature creation...")
        # Tính toán các chỉ báo cần thiết
        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx_regime'] = adx_indicator.adx()
        df['ema_regime'] = EMAIndicator(df['close'], window=ema_period).ema_indicator()

        # Xác định trạng thái
        conditions = [
            (df['adx_regime'] > adx_threshold) & (df['close'] > df['ema_regime']), # Uptrend
            (df['adx_regime'] > adx_threshold) & (df['close'] < df['ema_regime'])  # Downtrend
        ]
        choices = [1, -1]
        df['market_regime'] = np.select(conditions, choices, default=0) # Sideways là mặc định

        # Dọn dẹp các cột tạm
        df.drop(columns=['adx_regime', 'ema_regime'], inplace=True)
        logging.info("   [Features] ✅ Market state feature creation completed.")
        return df

class EnhancedEnsembleModel:
    """Enhanced ensemble with advanced CV, CPCV, and explainability"""
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.cv_results = {}
        self.base_model_feature_importance = {}
        self.quality_validator = QualityGateValidator()
        self.feature_store = AdvancedFeatureStore()
        self.explainability_data = {}
        
    def enhanced_time_series_split(self, X, y, n_splits=5, gap=0, max_train_size=None):
        """Enhanced time series split with gap and purging"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        test_size = n_samples // (n_splits + 1)
        splits = []
        
        for i in range(n_splits):
            # Calculate test indices
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Calculate train indices with gap
            train_end = test_start - gap
            train_start = 0
            
            if max_train_size and train_end - train_start > max_train_size:
                train_start = train_end - max_train_size
            
            if train_start >= 0 and train_end > train_start:
                train_indices = indices[train_start:train_end]
                splits.append((train_indices, test_indices))
        
        return splits
    
    def combinatorial_purged_cv(self, X, y, n_splits=5, n_test_splits=2, gap=24):
        """Combinatorial Purged Cross-Validation for financial data"""
        n_samples = len(X)
        test_size = n_samples // n_splits
        
        # Generate all possible test set combinations
        test_starts = [i * test_size for i in range(n_splits)]
        test_combinations = []
        
        from itertools import combinations
        for combo in combinations(range(n_splits), n_test_splits):
            test_indices = []
            for i in combo:
                start_idx = test_starts[i]
                end_idx = min(start_idx + test_size, n_samples)
                test_indices.extend(range(start_idx, end_idx))
            
            # Remove overlapping periods and apply gap
            test_indices = sorted(set(test_indices))
            
            # Create train indices with purging
            train_indices = []
            for i in range(n_samples):
                # Check if index is far enough from any test index
                min_distance = min([abs(i - t) for t in test_indices]) if test_indices else gap + 1
                if min_distance > gap:
                    train_indices.append(i)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                test_combinations.append((train_indices, test_indices))
        
        return test_combinations
    
    def get_model_explanation(self, X_sample, feature_names=None, top_k=5):
        """Get model explanation for a prediction"""
        explanations = {}
        
        # Get base model predictions and feature importance
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(X_sample.reshape(1, -1))[0, 1]
            else:
                prediction = model.predict(X_sample.reshape(1, -1))[0]
            
            # Get feature importance for this prediction
            if hasattr(model, 'feature_importances_'):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(X_sample))]
                
                feature_importance = dict(zip(
                    feature_names, 
                    model.feature_importances_
                ))
                
                # Get top contributing features
                top_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:top_k]
                
                explanations[name] = {
                    'prediction': prediction,
                    'top_features': top_features,
                    'confidence': prediction if prediction > 0.5 else 1 - prediction
                }
        
        return explanations

class OrderSafetyManager:
    """Advanced order safety with multiple validation layers"""
    
    def __init__(self):
        self.risk_manager = AdvancedRiskManager()
        self.observability = AdvancedObservability()
        self.order_history = deque(maxlen=1000)
        self.safety_violations = deque(maxlen=100)
        
    def validate_order_size(self, symbol, proposed_size, account_balance, max_risk_per_trade=0.01):
        """Validate order size against risk limits"""
        max_position_value = account_balance * max_risk_per_trade
        
        # Get current price (simplified - in production, fetch real price)
        current_price = 1.0  # Placeholder
        proposed_value = abs(proposed_size) * current_price
        
        if proposed_value > max_position_value:
            return False, f"Position value {proposed_value} exceeds max risk {max_position_value}"
        
        return True, "Order size validated"
    
    def validate_order_frequency(self, symbol, cooldown_minutes=5):
        """Validate order frequency to prevent overtrading"""
        current_time = datetime.now()
        recent_orders = [
            order for order in self.order_history 
            if order['symbol'] == symbol and 
               (current_time - order['timestamp']).total_seconds() < cooldown_minutes * 60
        ]
        
        if len(recent_orders) > 0:
            return False, f"Recent order for {symbol} within {cooldown_minutes} minutes"
        
        return True, "Order frequency validated"
    
    def validate_market_conditions(self, symbol):
        """Validate market conditions for safe trading"""
        # Check if market is open
        if not is_market_open(symbol):
            return False, f"Market closed for {symbol}"
        
        # Check weekend conditions
        if is_weekend() and not is_crypto_symbol(symbol):
            return False, f"Weekend trading not allowed for {symbol}"
        
        return True, "Market conditions validated"
    
    async def comprehensive_order_validation(self, symbol, size, direction, account_balance, existing_positions=None, returns_history=None):
        """Comprehensive order validation with all safety checks"""
        validation_result = {
            'approved': True,
            'violations': [],
            'safety_score': 1.0,
            'recommendations': []
        }
        
        # 1. Size validation
        size_ok, size_msg = self.validate_order_size(symbol, size, account_balance)
        if not size_ok:
            validation_result['approved'] = False
            validation_result['violations'].append(f"Size: {size_msg}")
        
        # 2. Frequency validation
        freq_ok, freq_msg = self.validate_order_frequency(symbol)
        if not freq_ok:
            validation_result['approved'] = False
            validation_result['violations'].append(f"Frequency: {freq_msg}")
        
        # 3. Market conditions
        market_ok, market_msg = self.validate_market_conditions(symbol)
        if not market_ok:
            validation_result['approved'] = False
            validation_result['violations'].append(f"Market: {market_msg}")
        
        # 4. Risk management validation
        if existing_positions and returns_history:
            risk_validation = await self.risk_manager.validate_trade(
                symbol, size, existing_positions, returns_history
            )
            if not risk_validation['approved']:
                validation_result['approved'] = False
                validation_result['violations'].extend([
                    f"Risk: {v}" for v in risk_validation['violations']
                ])
        
        # 5. Calculate safety score
        total_checks = 4
        passed_checks = sum([
            size_ok, freq_ok, market_ok, 
            risk_validation.get('approved', True) if 'risk_validation' in locals() else True
        ])
        validation_result['safety_score'] = passed_checks / total_checks
        
        # 6. Log validation result
        if not validation_result['approved']:
            violation_msg = f"Order validation failed for {symbol} {direction}:\n"
            for violation in validation_result['violations']:
                violation_msg += f"- {violation}\n"
            
            await self.observability.send_discord_alert(violation_msg, "ERROR")
            
            # Store violation
            self.safety_violations.append({
                'symbol': symbol,
                'violations': validation_result['violations'],
                'timestamp': datetime.now()
            })
        
        return validation_result
    
    def record_order(self, symbol, size, direction, price=None):
        """Record order for tracking and analysis"""
        order_record = {
            'symbol': symbol,
            'size': size,
            'direction': direction,
            'price': price,
            'timestamp': datetime.now()
        }
        
        self.order_history.append(order_record)
        logging.info(f"Order recorded: {symbol} {direction} {size}")

# Original class preserved for compatibility
class LSTMModel:
    def __init__(self, sequence_length=60, features_dim=40):
        self.sequence_length = sequence_length
        self.features_dim = features_dim
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self):
        """Build LSTM model with integrated ATTENTION mechanism"""
        input_layer = Input(shape=(self.sequence_length, self.features_dim))

        # Phiên bản này sẽ tự động dùng GPU nếu có và tương thích với mọi môi trường
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, activation='tanh', recurrent_activation='sigmoid')(input_layer)
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2, activation='tanh', recurrent_activation='sigmoid')(lstm1)

        attention_out = Attention()([lstm2, lstm2])

        lstm3 = LSTM(32, return_sequences=False, dropout=0.2, activation='tanh', recurrent_activation='sigmoid')(attention_out)

        dense1 = Dense(16, activation="relu")(lstm3)
        dropout1 = Dropout(0.3)(dense1)
        output = Dense(1, activation="sigmoid")(dropout1)

        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001, decay=1e-6),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )
        return self.model

    def prepare_sequences(self, X, y=None):
        """Prepare sequences for LSTM"""
        if X.empty or (y is not None and y.empty):
            logging.warning("   [LSTM PreSeq] Input X or y is empty.")
            return None, None if y is not None else None

        try:
            X_scaled = self.scaler.fit_transform(X)
        except ValueError as e:  # Có thể xảy ra nếu X chỉ có 1 sample khi fit_transform
            print(
                f"   [LSTM PreSeq] Error scaling X (len {len(X)}): {e}. Using X as is for now."
            )
            if len(X.shape) == 1:  # Nếu X là 1D array
                X_scaled = (
                    X.to_numpy().reshape(-1, 1)
                    if isinstance(X, pd.Series)
                    else np.array(X).reshape(-1, 1)
                )
            else:
                X_scaled = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
            # Hoặc thử transform nếu scaler đã được fit trước đó
            # try:
            #    X_scaled = self.scaler.transform(X)
            # except NotFittedError:
            #    print("    Scaler not fitted, and cannot fit with current X. Returning None.")
            #    return None, None

        sequences = []
        targets = [] if y is not None else None

        max_length = len(X_scaled)
        if y is not None:
            max_length = min(len(X_scaled), len(y))

        # Điều kiện để vòng lặp chạy: max_length > self.sequence_length
        if max_length <= self.sequence_length:
            # print(f"   [LSTM PreSeq] Not enough data (max_length {max_length}) for sequence_length {self.sequence_length}. Need max_length > sequence_length.")
            return np.array([]), np.array([]) if y is not None else np.array([])

        for i in range(self.sequence_length, max_length):
            sequences.append(X_scaled[i - self.sequence_length : i])
            if y is not None:
                # Quan trọng: y là y_train_ensemble, nó là pd.Series
                # Cần đảm bảo index của y tương ứng với cách lấy X_scaled
                # Nếu X là DataFrame và y là Series, X.iloc[i] và y.iloc[i] là ok
                targets.append(y.iloc[i])

        sequences = np.array(sequences)
        if targets is not None:
            targets = np.array(targets)
            if len(sequences) == 0:  # Double check
                return np.array([]), np.array([])
            return sequences, targets

        if len(sequences) == 0:  # Double check
            return np.array([])
        return sequences

    # TÌM VÀ THAY THẾ HÀM NÀY TRONG LỚP LSTMModel

    def train(self, X, y):
        X_seq, y_seq = self.prepare_sequences(X, y)
        if X_seq is None or len(X_seq) < 10: # Cần ít nhất 10 chuỗi để huấn luyện
            print(f"   [LSTM] Không đủ dữ liệu chuỗi để huấn luyện. Bỏ qua.")
            self.model = None
            return None

        if self.model is None:
            self.build_model()

        # <<< CẢI TIẾN: LẤY SPLIT CUỐI CÙNG TỪ TSCV THAY VÌ SPLIT ĐẦU TIÊN >>>
        n_lstm_splits = 5
        if len(X_seq) < n_lstm_splits:
            n_lstm_splits = max(2, len(X_seq) -1) # Đảm bảo có ít nhất 1 split

        tscv_lstm = TimeSeriesSplit(n_splits=n_lstm_splits)
        
        # Lấy ra split cuối cùng
        all_splits = list(tscv_lstm.split(X_seq))
        train_idx_lstm, val_idx_lstm = all_splits[-1]
        
        X_train_lstm, X_val_lstm = X_seq[train_idx_lstm], X_seq[val_idx_lstm]
        y_train_lstm, y_val_lstm = y_seq[train_idx_lstm], y_seq[val_idx_lstm]
        # <<< KẾT THÚC CẢI TIẾN >>>

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy", 
                patience=ML_CONFIG.get("EARLY_STOPPING_PATIENCE", 15), 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_accuracy", 
                patience=5, 
                factor=ML_CONFIG.get("LEARNING_RATE_DECAY", 0.95),
                min_lr=1e-7,
                verbose=1
            ),
        ]

        logging.info(f"   [LSTM] Starting training with {len(X_train_lstm)} train samples and {len(X_val_lstm)} validation samples...")
        history = self.model.fit(
            X_train_lstm, y_train_lstm,
            validation_data=(X_val_lstm, y_val_lstm),
            epochs=200, 
            batch_size=ML_CONFIG.get("BATCH_SIZE", 64), 
            callbacks=callbacks, 
            verbose=1,
            shuffle=False  # Important for time series data
        )
        return history

    # <<< THAY THẾ TOÀN BỘ HÀM predict_proba TRONG LỚP EnsembleModel >>>

# ### ĐẶT HÀM NÀY VÀO TRONG CLASS LSTMModel ###
    # ### THAY THẾ HOÀN TOÀN HÀM predict_proba CŨ CỦA NÓ ###
    def predict_proba(self, X):
        """
        Chuẩn bị chuỗi dữ liệu (sequence) từ đầu vào và dự đoán xác suất.
        """
        # DÙNG self.model (đúng với class LSTMModel)
        if self.model is None:
            return None # Model chưa được huấn luyện

        # Chuẩn bị chuỗi dữ liệu từ DataFrame đầu vào
        X_seq = self.prepare_sequences(X, y=None)

        # Kiểm tra nếu không thể tạo được chuỗi (do không đủ dữ liệu)
        if X_seq is None or len(X_seq) == 0:
            return None

        # Dự đoán bằng model LSTM đã được huấn luyện
        # Trả về một mảng chứa xác suất cho tất cả các chuỗi
        return self.model.predict(X_seq, verbose=0)

# Lớp này không thay đổi
class EnsembleModel:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.model_weights = {}
        self.cv_results = {}
        self.meta_model = LogisticRegression()
        self.base_model_feature_importance = {}

    # <<< THAY THẾ TOÀN BỘ HÀM _objective TRONG LỚP EnsembleModel >>>

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnsembleModel
    def _objective(self, trial, X, y, model_name):
        """
        Hàm mục tiêu để Optuna tối ưu hóa.
        PHIÊN BẢN NÂNG CẤP: Bổ sung Regularization (reg_alpha, reg_lambda) để chống overfitting.
        """
        if model_name == "xgboost":
            params = {
                "objective": "binary:logistic", "eval_metric": "logloss",
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 3.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                # === ENHANCED REGULARIZATION ===
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0), # L1 regularization
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0), # L2 regularization
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
                # ===================================================
                "random_state": 42, "n_jobs": -1
            }
            model = xgb.XGBClassifier(**params)

        elif model_name == "lightgbm":
            params = {
                "objective": "binary", "metric": "binary_logloss",
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0),
                # === ENHANCED REGULARIZATION ===
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0), # L1 regularization
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0), # L2 regularization
                "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                # ===================================================
                "random_state": 42, "verbose": -1, "n_jobs": -1
            }
            model = lgb.LGBMClassifier(**params)

        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 0.9]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "max_samples": trial.suggest_float("max_samples", 0.7, 1.0),
                "random_state": 42, "n_jobs": -1
            }
            model = RandomForestClassifier(**params)

        elif model_name == "logistic_regression":
            params = {
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
                "C": trial.suggest_float("C", 0.001, 100.0, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                "max_iter": trial.suggest_int("max_iter", 1000, 5000)
            }
            if params.get("penalty") == "l1":
                params["solver"] = "liblinear"
            elif params.get("penalty") == "elasticnet":
                params["solver"] = "saga"
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
            else:
                params["solver"] = "lbfgs"
            model = LogisticRegression(**params, random_state=42)

        elif model_name == "knn":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan", "chebyshev"]),
                "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
                "leaf_size": trial.suggest_int("leaf_size", 10, 50),
                "p": trial.suggest_int("p", 1, 3),  # Power parameter for Minkowski metric
                "n_jobs": -1
            }
            model = KNeighborsClassifier(**params)
        else:
            return 0.0

        tscv = TimeSeriesSplit(n_splits=3)
        scores = safe_cross_val_score(model, X, y, cv=tscv, scoring="f1", n_jobs=-1)
        return np.mean(scores)

    def evaluate_model_with_purged_cv(self, model, X, y, n_splits=5, embargo=5):
        X_idx = np.arange(len(X))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_pred = np.full(len(X), np.nan)
        fold_metrics = []

        for fold_id, (tr, te) in enumerate(tscv.split(X_idx), 1):
            # --- LOGIC CỐT LÕI NẰM Ở ĐÂY ---
            te_start = te[0] # Thời điểm bắt đầu của tập test

            # Embargoing: Loại bỏ các mẫu training ngay SÁT tập test
            # Điều này ngăn các mẫu test ảnh hưởng đến các mẫu train cuối cùng
            tr = tr[tr < max(0, te_start - embargo)]
            # --------------------------------

            if len(tr) == 0: continue
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            X_te, y_te = X.iloc[te], y.iloc[te]

            # Purging được thực hiện một cách gián tiếp bởi cách bạn tạo label.
            # Khi label của một mẫu train (ví dụ: tại thời điểm t) được tạo từ giá ở t+3,
            # và mẫu test bắt đầu ở t+5, thì mẫu train đó không "nhìn" vào dữ liệu của mẫu test.
            # Việc embargo làm cho quá trình này an toàn hơn nữa.

            mdl = clone(model)
            mdl.fit(X_tr, y_tr)
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X_te)[:, 1]
                y_hat = (p >= 0.5).astype(int)
            else:
                y_hat = mdl.predict(X_te)
                p = y_hat.astype(float)
            f1 = f1_score(y_te, y_hat)
            acc = accuracy_score(y_te, y_hat)
            fold_metrics.append((f1, acc))
            oof_pred[te] = p

        f1s = [m[0] for m in fold_metrics]
        accs = [m[1] for m in fold_metrics]
        results = {
            "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
            "std_f1":  float(np.std(f1s))  if f1s else 0.0,
            "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
            "std_accuracy":  float(np.std(accs))  if accs else 0.0,
            "oof_proba": oof_pred,
            "X_train_index_for_oof": X.index
        }
        print(f"   [PurgedCV] F1={results['mean_f1']:.3f}±{results['std_f1']:.3f} | "
              f"ACC={results['mean_accuracy']:.3f}±{results['std_accuracy']:.3f} | embargo={embargo}")
        return results


    # <<< THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnsembleModel >>>

    def train_ensemble(self, X, y):
        base_model_names = [m for m in ML_CONFIG["ENSEMBLE_MODELS"] if m != "lstm"]
        self.models = {}
        self.cv_results = {}
        self.base_model_feature_importance = {}

        logging.info("   [Stacking] Starting training of base models (Level 0)...")
        tscv = TimeSeriesSplit(n_splits=ML_CONFIG["CV_N_SPLITS"])
        oof_predictions = {}

        for name in base_model_names:
            logging.info(f"      -> Training and getting OOF predictions for {name}...")
            model = self._get_optimized_model(name, X, y)
            oof_preds_for_model = np.full(len(X), np.nan)
            for train_idx, val_idx in tscv.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                oof_preds_for_model[val_idx] = model_clone.predict_proba(X_val)[:, 1]

            oof_predictions[name] = oof_preds_for_model
            model.fit(X, y)
            self.models[name] = model
            self.cv_results[name] = self.evaluate_model_with_purged_cv(model, X, y)

            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=X.columns)
                self.base_model_feature_importance[name] = importances.nlargest(5).to_dict()

        # === Level 1: Meta-Model từ OOF ===
        meta_features_df = pd.DataFrame(oof_predictions, index=X.index).dropna()
        y_for_meta = y.reindex(meta_features_df.index)

        logging.info("\n   [Stacking] Starting Meta-Model training and evaluation (Level 1)...")
        meta_model_for_scoring = LogisticRegression()
        meta_score = np.mean(
            safe_cross_val_score(meta_model_for_scoring, meta_features_df, y_for_meta, cv=tscv, scoring="f1")
        )
        logging.info(f"   [Stacking] ✅ Meta-Model F1-Score (cross-validated): {meta_score:.4f}")

        # === Level 2: Calibration (Isotonic/Platt) giữ trật tự thời gian ===
        # Chia theo thời gian: 80% train meta, 20% calibrate
        split_idx = int(len(meta_features_df) * 0.8)
        if split_idx < 1 or split_idx >= len(meta_features_df):
            # Fallback an toàn nếu dữ liệu quá ít
            split_idx = max(1, len(meta_features_df) - 1)

        X_meta_train = meta_features_df.iloc[:split_idx]
        y_meta_train = y_for_meta.iloc[:split_idx]
        X_meta_cal   = meta_features_df.iloc[split_idx:]
        y_meta_cal   = y_for_meta.iloc[split_idx:]

        # Fit meta-estimator trên phần "train" thời gian trước
        base_meta = LogisticRegression(max_iter=1000)
        base_meta.fit(X_meta_train, y_meta_train)

        # Hiệu chỉnh xác suất trên phần "calibration" thời gian sau
        logging.info("   [Stacking] Starting probability calibration (Level 2: Calibration)...")
        try:
            calibrated = create_calibrated_classifier(base_meta, method='isotonic', cv='prefit')
            calibrated.fit(X_meta_cal, y_meta_cal)  # <== BƯỚC QUAN TRỌNG: thực sự huấn luyện Level 2
            self.meta_model = calibrated
            logging.info("   [Stacking] ✅ Calibration (isotonic, prefit) completed.")
        except Exception as e:
            # Fallback: dùng k-fold calibration nếu prefit thất bại (dữ liệu ít/khó)
            logging.warning(f"   [Stacking] ⚠️ Calibration prefit error: {e}. Using cv=3 for auto fit + calibrate.")
            calibrated = create_calibrated_classifier(LogisticRegression(max_iter=1000),
                                        method='isotonic', cv=3)
            calibrated.fit(meta_features_df, y_for_meta)
            self.meta_model = calibrated
            logging.info("   [Stacking] ✅ Calibration (isotonic, cv=3) completed.")

        logging.info("✅ Stacking ensemble training (Level 0 + Level 1 + Level 2) completed!")

    def _train_stacking_model(self, X, y, oof_predictions):
        """Enhanced stacking with multiple meta-learners"""
        if not ML_CONFIG.get("MODEL_STACKING_ENABLED", True):
            return
            
        logging.info("   [Enhanced Stacking] Training advanced meta-learners...")
        
        # Prepare meta-features
        meta_features = np.column_stack(list(oof_predictions.values()))
        meta_features_df = pd.DataFrame(meta_features, columns=list(oof_predictions.keys()))
        
        # Train multiple meta-learners
        meta_learners = {}
        
        # 1. XGBoost meta-learner
        try:
            xgb_meta = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_meta.fit(meta_features_df, y)
            meta_learners['xgb_meta'] = xgb_meta
        except Exception as e:
            logging.warning(f"XGBoost meta-learner failed: {e}")
        
        # 2. LightGBM meta-learner
        try:
            lgb_meta = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            lgb_meta.fit(meta_features_df, y)
            meta_learners['lgb_meta'] = lgb_meta
        except Exception as e:
            logging.warning(f"LightGBM meta-learner failed: {e}")
        
        # 3. Logistic Regression meta-learner
        try:
            lr_meta = LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000
            )
            lr_meta.fit(meta_features_df, y)
            meta_learners['lr_meta'] = lr_meta
        except Exception as e:
            logging.warning(f"Logistic Regression meta-learner failed: {e}")
        
        # Store meta-learners
        self.meta_learners = meta_learners
        logging.info(f"   [Enhanced Stacking] ✅ Trained {len(meta_learners)} meta-learners")

    # <<< THÊM HÀM MỚI NÀY VÀO TRONG LỚP EnsembleModel >>>
    def get_base_model_feature_influence(self):
        """
        Tổng hợp ảnh hưởng của các feature từ TẤT CẢ các model cơ sở.
        """
        if not self.base_model_feature_importance:
            return "Không có dữ liệu feature influence."

        # Tổng hợp importance từ tất cả các model cơ sở
        combined_influence = {}
        for model_name, features in self.base_model_feature_importance.items():
            for feature, importance in features.items():
                if feature not in combined_influence:
                    combined_influence[feature] = 0.0
                combined_influence[feature] += importance # Cộng dồn điểm importance

        # Sắp xếp và lấy top 5 feature có ảnh hưởng nhất trên toàn bộ các model cơ sở
        sorted_influence = sorted(combined_influence.items(), key=lambda item: item[1], reverse=True)
        top_5_text = ", ".join([f"{feat.replace('_', ' ')} ({val:.2f})" for feat, val in sorted_influence[:5]])
        return top_5_text

    # <<< THÊM HÀM HELPER NÀY VÀO LỚP EnsembleModel >>>
    def _get_optimized_model(self, name, X, y):
        """Helper function to run Optuna and return model with best parameters."""
        logging.info(f"      -> Optimizing for {name}...")
        # Enhanced study configuration
        sampler = TPESampler(seed=42) if OPTUNA_CONFIG.get("SAMPLER") == "TPE" else None
        pruner = MedianPruner() if OPTUNA_CONFIG.get("PRUNING_ENABLED", True) else None
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        try: # <<< THÊM TRY Ở ĐÂY
            study.optimize(
                lambda trial: self._objective(trial, X, y, name),
                n_trials=OPTUNA_CONFIG["N_TRIALS"],
                timeout=OPTUNA_CONFIG["TIMEOUT_SEC"],
            )
            best_params = study.best_params
        except Exception as e: # <<< THÊM EXCEPT
            print(f"   [Optuna Warning] Lỗi trong quá trình tối ưu hóa cho {name}: {e}. Sử dụng tham số mặc định.")
            best_params = {} # Sử dụng dict rỗng để model dùng tham số mặc định

        if name == "xgboost":
            return xgb.XGBClassifier(**best_params, random_state=42)
        elif name == "lightgbm":
            return lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
        elif name == "random_forest":
            return RandomForestClassifier(**best_params, random_state=42)
        elif name == "logistic_regression":
            if best_params.get("penalty") == "l1": best_params["solver"] = "liblinear"
            else: best_params["solver"] = "lbfgs"
            return LogisticRegression(**best_params, random_state=42, max_iter=1000)
        elif name == "knn":
            return KNeighborsClassifier(**best_params, n_jobs=-1)
        return None

# ### ĐẶT HÀM NÀY VÀO TRONG CLASS EnsembleModel ###
    # ### THAY THẾ HOÀN TOÀN HÀM predict_proba CŨ CỦA NÓ ###
    def predict_proba(self, X):
        """
        Dự đoán xác suất cuối cùng bằng cách lấy trung bình có trọng số.
        PHIÊN BẢN SỬA LỖI: Đảm bảo TẤT CẢ model con đều dùng dữ liệu đã làm sạch.
        """
        # --- BƯỚC 1: Xử lý dữ liệu đầu vào ---
        # Kiểm tra nếu X rỗng hoặc không phải là DataFrame
        if not isinstance(X, pd.DataFrame) or X.empty:
            return 0.5

        # Làm sạch dữ liệu một lần duy nhất tại đây
        X_clean = X.copy()
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Sử dụng ffill/bfill để xử lý các NaN còn sót lại, sau đó điền bằng 0
        X_clean.fillna(method='ffill', inplace=True)
        X_clean.fillna(method='bfill', inplace=True)
        X_clean.fillna(0, inplace=True)

            # --- BƯỚC 1: LẤY DỰ ĐOÁN TỪ CÁC MODEL CƠ SỞ ---
        base_predictions = {}
        for name, model in self.models.items():
            if model is None: continue
            try:
                # Chỉ cần dòng dữ liệu cuối cùng cho dự đoán live
                X_pred = X_clean.tail(1)
                prob = model.predict_proba(X_pred)
                base_predictions[name] = prob[0][1] # Lấy xác suất của lớp 1
            except Exception as e:
                # print(f"⚠️ Lỗi khi dự đoán với model cơ sở {name}: {e}")
                base_predictions[name] = 0.5 # Fallback

        # --- BƯỚC 2: TẠO FEATURE CHO META-MODEL ---
        # Tạo DataFrame với đúng thứ tự cột như lúc huấn luyện
        meta_features = pd.DataFrame([base_predictions])

        # --- BƯỚC 3: DÙNG META-MODEL ĐỂ RA DỰ ĐOÁN CUỐI CÙNG ---
        try:
            final_proba = self.meta_model.predict_proba(meta_features)[:, 1]
            return final_proba[0]
        except Exception as e:
            # print(f"⚠️ Lỗi khi dự đoán với meta-model: {e}")
            return 0.5 # Fallback

        # Enhanced stacking prediction if available
        if hasattr(self, 'meta_learners') and self.meta_learners:
            try:
                # Create meta-features from base predictions
                meta_features = np.array(list(predictions.values())).reshape(1, -1)
                meta_features_df = pd.DataFrame(meta_features, columns=list(predictions.keys()))
                
                # Get predictions from all meta-learners
                meta_predictions = []
                for meta_name, meta_model in self.meta_learners.items():
                    meta_pred = meta_model.predict_proba(meta_features_df)[:, 1]
                    meta_predictions.append(meta_pred[0])
                
                # Average meta-learner predictions
                if meta_predictions:
                    stacked_prediction = np.mean(meta_predictions)
                    logging.info(f"   [Enhanced Stacking] Using {len(self.meta_learners)} meta-learners")
                    return stacked_prediction
            except Exception as e:
                logging.warning(f"Enhanced stacking failed, falling back to weighted average: {e}")

        # Fallback to weighted average
        final_prediction = 0.0
        total_weight = 0.0
        for name, pred_proba in predictions.items():
            weight = self.model_weights.get(name, 0)
            final_prediction += pred_proba * weight
            total_weight += weight

        return final_prediction / total_weight if total_weight > 0 else 0.5
# ### THÊM HÀM MỚI NÀY VÀO TRONG CLASS EnsembleModel ###
    def predict_proba_on_df(self, X):
        """
        Dự đoán xác suất cho TOÀN BỘ DataFrame đầu vào (dùng cho huấn luyện).
        Trả về một mảng numpy có cùng độ dài với X.
        """
        if not isinstance(X, pd.DataFrame) or X.empty:
            return np.array([0.5] * len(X))

        # 1. Làm sạch dữ liệu
        X_clean = X.copy()
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(method='ffill', inplace=True)
        X_clean.fillna(method='bfill', inplace=True)
        X_clean.fillna(0, inplace=True)

        all_probas = []
        all_weights = []

        # 2. Lấy dự đoán hàng loạt từ các model (trừ LSTM vì xử lý sequence phức tạp hơn)
        for name, model in self.models.items():
            weight = self.model_weights.get(name, 0)
            if weight > 0 and model is not None and hasattr(model, 'predict_proba') and name != "lstm":
                try:
                    # Các model scikit-learn có thể dự đoán trên toàn bộ DataFrame
                    probas = model.predict_proba(X_clean)[:, 1] # Lấy xác suất của lớp 1 (BUY)
                    all_probas.append(probas)
                    all_weights.append(weight)
                except Exception as e:
                    print(f"Lỗi khi dự đoán hàng loạt với model {name}: {e}")

        if not all_probas:
            return np.full(len(X), 0.5)

        # 3. Tính trung bình có trọng số bằng numpy
        # Chuyển all_probas thành mảng 2D (mỗi hàng là một model, mỗi cột là một sample)
        probas_array = np.array(all_probas)
        # np.average sẽ tính trung bình có trọng số dọc theo cột (axis=0)
        final_probas = np.average(probas_array, axis=0, weights=all_weights)

        return final_probas

# Lớp này không thay đổi
class EnhancedDataManager:
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.data_cache = {}
        self.asset_class_configs = self._initialize_asset_class_configs()
        
    def _initialize_asset_class_configs(self):
        """Initialize asset class specific configurations"""
        return {
            "equity_index": {
                "min_candles": 100,
                "max_candles": 5000,
                "session_filter": True,
                "gap_handling": "forward_fill",
                "volatility_adjustment": True,
                "news_sensitivity": "high"
            },
            "commodity": {
                "min_candles": 200,
                "max_candles": 10000,
                "session_filter": False,
                "gap_handling": "interpolate",
                "volatility_adjustment": True,
                "news_sensitivity": "very_high"
            },
            "forex": {
                "min_candles": 150,
                "max_candles": 3000,
                "session_filter": True,
                "gap_handling": "forward_fill",
                "volatility_adjustment": False,
                "news_sensitivity": "medium"
            },
            "cryptocurrency": {
                "min_candles": 50,
                "max_candles": 2000,
                "session_filter": False,
                "gap_handling": "none",
                "volatility_adjustment": True,
                "news_sensitivity": "very_high"
            }
        }
    
    def _get_symbol_config(self, symbol):
        """Get configuration for specific symbol"""
        metadata = SYMBOL_METADATA.get(symbol, {})
        asset_class = metadata.get("asset_class", "equity_index")
        return self.asset_class_configs.get(asset_class, self.asset_class_configs["equity_index"])
    
    def _get_optimal_candle_count(self, symbol):
        """Get optimal candle count based on symbol characteristics"""
        config = self._get_symbol_config(symbol)
        metadata = SYMBOL_METADATA.get(symbol, {})
        
        # Base count from config
        base_count = config["max_candles"]
        
        # Adjust based on volatility profile
        volatility_profile = metadata.get("volatility_profile", "medium")
        if volatility_profile == "very_high":
            base_count = min(base_count, 1500)  # Less history for very volatile assets
        elif volatility_profile == "high":
            base_count = min(base_count, 2500)
        elif volatility_profile == "low":
            base_count = min(base_count, 5000)  # More history for stable assets
            
        return base_count

    def fetch_multi_timeframe_data(self, symbol, count=None, timeframes_to_use=None):
        """Enhanced data fetching with asset class optimization"""
        all_data = {}
        
        # Get optimal candle count for symbol
        if count is None:
            count = self._get_optimal_candle_count(symbol)
        
        # Get symbol-specific configuration
        config = self._get_symbol_config(symbol)
        metadata = SYMBOL_METADATA.get(symbol, {})
        
        # Chọn primary theo symbol; nếu không có thì dùng default
        if timeframes_to_use is None:
            primary_tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, PRIMARY_TIMEFRAME_DEFAULT)
            timeframes_to_use = TIMEFRAME_SET_BY_PRIMARY.get(primary_tf, ["H4", "D1", "W1"])

        # Map granularity chuẩn OANDA (Monthly dùng "M", không phải "M1")
        granularity_map = {
            "M15": "M15", "M30": "M30",
            "H1": "H1",   "H4": "H4",
            "D1": "D",    "W1": "W",
            "MN1": "M"    # nếu sau này bạn muốn dùng monthly, đặt TF là "MN1"
        }

        for timeframe in timeframes_to_use:
            if timeframe not in granularity_map:
                continue
            try:
                # Get instrument mapping with enhanced symbol support
                instrument = self._get_oanda_instrument(symbol)
                
                granularity = granularity_map[timeframe]
                url = f"{OANDA_URL}/instruments/{instrument}/candles"
                headers = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}
                params = {"count": count, "granularity": granularity, "price": "M"}

                print(f"   --> Fetching: {instrument} | TF: {timeframe} | Count: {count}")
                response = requests.get(url, headers=headers, params=params, timeout=30)
                if response.status_code != 200:
                    print(f"❌ API {response.status_code} {instrument} {timeframe}: {response.text}")
                    continue

                candles_raw = response.json().get("candles", [])
                print(f"   --> Received {len(candles_raw)} candles ({instrument} {timeframe}).")
                if not candles_raw:
                    continue

                candles = [
                    {
                        "time": pd.to_datetime(c["time"]),
                        "open": float(c["mid"]["o"]),
                        "high": float(c["mid"]["h"]),
                        "low":  float(c["mid"]["l"]),
                        "close":float(c["mid"]["c"]),
                        "volume": c.get("volume", 1000),
                    }
                    for c in candles_raw
                    if c.get("complete", True)
                ]
                
                # Asset class specific validation
                min_required = config["min_candles"]
                if len(candles) < min_required:
                    print(f"⚠️ Only {len(candles)} complete candles for {instrument} {timeframe}. Required: {min_required}. Skip.")
                    continue

                df = pd.DataFrame(candles).set_index("time")
                df.index = df.index.tz_convert(pytz.timezone("Asia/Bangkok"))
                
                # Apply asset class specific preprocessing
                df = self._apply_asset_class_preprocessing(df, symbol, config)
                
                all_data[timeframe] = df
            except Exception as e:
                print(f"❌ Unexpected error {symbol} {timeframe}: {e}")
        return all_data
    
    def _get_oanda_instrument(self, symbol):
        """Enhanced instrument mapping with all supported symbols"""
        instrument_map = {
            # Commodities
            "XAUUSD": "XAU_USD",
            "XAGUSD": "XAG_USD",
            
            # Cryptocurrencies
            "BTCUSD": "BTC_USD",
            "ETHUSD": "ETH_USD",
            
            # US Indices
            "SPX500": "SPX500_USD",
            "NAS100": "NAS100_USD",
            "US30": "US30_USD",
            
            # European Indices
            "DE40": "DE30_EUR",
            "UK100": "UK100_GBP",
            "FR40": "FR40_EUR",
            
            # Asian Indices
            "JP225": "JP225_USD",
            "HK50": "HK33_HKD",
            "AU200": "AU200_AUD",
            
            # Energy
            "WTICO_USD": "WTICO_USD",
            "BCO_USD": "BCO_USD",
        }
        
        if symbol in instrument_map:
            return instrument_map[symbol]
        else:
            # Default logic for forex pairs
            return f"{symbol[:3]}_{symbol[3:]}"
    
    def _apply_asset_class_preprocessing(self, df, symbol, config):
        """Apply asset class specific preprocessing"""
        metadata = SYMBOL_METADATA.get(symbol, {})
        
        # Handle gaps based on asset class
        gap_handling = config["gap_handling"]
        if gap_handling == "forward_fill":
            df = df.fillna(method="ffill")
        elif gap_handling == "interpolate":
            df = df.interpolate(method="linear")
        # For "none", no gap handling (crypto)
        
        # Add asset class specific features
        df["asset_class"] = metadata.get("asset_class", "equity_index")
        df["volatility_profile"] = metadata.get("volatility_profile", "medium")
        df["pip_value"] = metadata.get("pip_value", 1.0)
        
        # Session-based filtering for applicable assets
        if config["session_filter"] and metadata.get("trading_hours"):
            df = self._apply_session_filter(df, metadata["trading_hours"])
        
        return df
    
    def _apply_session_filter(self, df, trading_hours):
        """Filter data to trading hours only"""
        try:
            # Convert to trading timezone
            timezone = trading_hours.get("timezone", "UTC")
            df_tz = df.index.tz_convert(timezone)
            
            start_hour = trading_hours.get("start", 0)
            end_hour = trading_hours.get("end", 23)
            
            # Create boolean mask for trading hours
            if start_hour <= end_hour:
                mask = (df_tz.hour >= start_hour) & (df_tz.hour < end_hour)
            else:
                # Handle overnight sessions (e.g., forex)
                mask = (df_tz.hour >= start_hour) | (df_tz.hour < end_hour)
            
            return df[mask]
        except Exception as e:
            logging.warning(f"Session filtering failed for {symbol}: {e}")
            return df


    # <<< THÊM HÀM MỚI NÀY VÀO CLASS >>>

    def get_current_price(self, symbol):
        try:
            granularity_map = {
            "M15": "M15",
            "M30": "M30",
            "H1": "H1",
            "H4": "H4",
            "D1": "D",
            "W1": "W",
            "M1": "M",
            }

            if symbol == "XAUUSD":
                instrument = "XAU_USD"
            elif symbol == "XAGUSD":
                instrument = "XAG_USD"
            elif symbol == "BTCUSD":
                instrument = "BTC_USD"
            elif symbol == "ETHUSD":
                instrument = "ETH_USD"
            elif symbol == "SPX500":
                instrument = "SPX500_USD"
            elif symbol == "NAS100":
                instrument = "NAS100_USD"
            elif symbol == "US30":
                instrument = "US30_USD"
            elif symbol == "UK100":
                instrument = "UK100_GBP"
            elif symbol == "DE40":
                instrument = "DE30_EUR"
            elif symbol == "JP225":
                instrument = "JP225_USD"
            elif symbol == "HK50":
                instrument = "HK33_HKD"
            elif symbol == "AU200":
                instrument = "AU200_AUD"
            elif symbol == "WTICO_USD":
                instrument = "WTICO_USD"
            elif symbol == "BCO_USD":
                instrument = "BCO_USD"
            else: # Logic mặc định cho các cặp tiền tệ Forex
                instrument = f"{symbol[:3]}_{symbol[3:]}"

            granularity = granularity_map.get(PRIMARY_TIMEFRAME)
            if not granularity:
                print(f"❌ Khung thời gian chính {PRIMARY_TIMEFRAME} không hợp lệ.")
                return None

            url = f"{OANDA_URL}/instruments/{instrument}/candles"
            headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
            params = {"count": 1, "price": "M", "granularity": granularity}
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code != 200:
                print(
                    f"❌ Lỗi API {response.status_code} khi lấy giá nhanh cho {symbol}: {response.text}"
                )
                return None

            candles = response.json().get("candles", [])
            if candles:
                return float(candles[0]["mid"]["c"])
            else:
                print(f"⚠️ Không nhận được nến nào khi lấy giá nhanh cho {symbol}.")
                return None
        except Exception as e:
            print(f"❌ Lỗi không xác định khi lấy giá nhanh cho {symbol}: {e}")
            return None

    # <<< SỬA LỖI DỨT ĐIỂM 2/2 >>>
    # Hàm này đã được sửa để quản lý việc đổi tên cột 'rsi' và ghép dữ liệu
    # một cách chính xác, đảm bảo không còn lỗi trùng lặp.

    # <<< SỬA LỖI DỨT ĐIỂM 2/2 >>>
    # Hàm này đã được sửa để quản lý việc đổi tên cột 'rsi' và ghép dữ liệu
    # một cách chính xác, đảm bảo không còn lỗi trùng lặp.
    
    def create_enhanced_features(self, symbol):
            """
            Tạo features nâng cao từ multi-timeframe data.
            """
            primary_tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, PRIMARY_TIMEFRAME_DEFAULT)
            timeframes_to_use = TIMEFRAME_SET_BY_PRIMARY.get(primary_tf)
            
            multi_tf_data = self.fetch_multi_timeframe_data(symbol, 5000, timeframes_to_use)
            logging.info(f"[Sanity] weekend={is_weekend()} sym={symbol} crypto={is_crypto_symbol(symbol)} primary_tf={primary_tf}")
            # This call is now correct.
            if not validate_dataframe_freshness(multi_tf_data, primary_tf, symbol=symbol):
                print(f"❌ Dữ liệu cho {symbol} đã quá cũ. Bỏ qua việc tạo feature.")
                return None

            # <<< KẾT THÚC THAY ĐỔI >>>
            # ========================
            if primary_tf not in multi_tf_data:
                print(f"❌ Không tìm thấy dữ liệu cho khung thời gian chính {primary_tf} của {symbol}")
                return None

            # 2. Bắt đầu với khung thời gian chính VÀ TẠO TẤT CẢ FEATURE CHO NÓ
            df_enhanced = multi_tf_data[primary_tf].copy()
            df_enhanced.name = symbol

            df_enhanced = self.feature_engineer.create_all_features(df_enhanced)

            if 'rsi' in df_enhanced.columns:
                df_enhanced.rename(columns={'rsi': f'rsi_{primary_tf}'}, inplace=True)

            for htf in timeframes_to_use:
                if htf != primary_tf and htf in multi_tf_data:
                    df_htf = multi_tf_data[htf].copy()

                    ema20_htf = EMAIndicator(df_htf["close"], 20).ema_indicator()
                    ema50_htf = EMAIndicator(df_htf["close"], 50).ema_indicator()
                    rsi_htf = RSIIndicator(df_htf["close"]).rsi()

                    htf_features = pd.DataFrame({
                        f"ema20_{htf}": ema20_htf,
                        f"ema50_{htf}": ema50_htf,
                        f"rsi_{htf}": rsi_htf,
                        f"trend_{htf}": (ema20_htf > ema50_htf).astype(int),
                    }, index=df_htf.index)

                    htf_resampled = htf_features.reindex(df_enhanced.index, method="ffill").bfill()
                    df_enhanced = df_enhanced.join(htf_resampled)

            df_enhanced.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_enhanced.fillna(method='ffill', inplace=True)
            df_enhanced.fillna(method='bfill', inplace=True)
            df_enhanced.fillna(0, inplace=True)
            return df_enhanced

    # THÊM HÀM MỚI NÀY VÀO TRONG LỚP EnhancedDataManager

    async def fetch_multi_timeframe_data_async(self, session, symbol, count=2500, timeframes_to_use=None):
        all_data = {}
        primary_tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, PRIMARY_TIMEFRAME_DEFAULT)
        timeframes_to_use = TIMEFRAME_SET_BY_PRIMARY.get(primary_tf, ["H4", "D1", "W1"])
        granularity_map = {"H1": "H1", "H4": "H4", "D1": "D", "W1": "W"}

        for timeframe in timeframes_to_use:
            if timeframe not in granularity_map: continue
            try:
                # (Logic lấy instrument giữ nguyên như hàm cũ của bạn)
                instrument = f"{symbol[:3]}_{symbol[3:]}" # Ví dụ đơn giản hóa
                
                granularity = granularity_map[timeframe]
                url = f"{OANDA_URL}/instruments/{instrument}/candles"
                headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
                params = {"count": count, "granularity": granularity, "price": "M"}

                async with session.get(url, headers=headers, params=params, timeout=30) as response:
                    if response.status != 200:
                        print(f"❌ API Async {response.status} {instrument} {timeframe}")
                        continue
                    
                    data = await response.json()
                    candles_raw = data.get("candles", [])
                    
                    candles = [
                        {"time": pd.to_datetime(c["time"]), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": c.get("volume", 1000)}
                        for c in candles_raw if c.get("complete", True)
                    ]

                    if candles:
                        df = pd.DataFrame(candles).set_index("time")
                        df.index = df.index.tz_convert(pytz.timezone("Asia/Bangkok"))
                        all_data[timeframe] = df
            except Exception as e:
                print(f"❌ Lỗi Async Fetch {symbol} {timeframe}: {e}")
                
        return symbol, all_data
    # Các hàm lưu/tải mô hình không thay đổi
    # <<< THAY THẾ TOÀN BỘ 2 HÀM NÀY >>>
    # THÊM HÀM MỚI NÀY VÀO TRONG LỚP EnhancedTradingBot

    async def run_async_data_gathering(self):
        """
        Tạo và chạy các tác vụ lấy dữ liệu và tạo features một cách bất đồng bộ.
        """
        print("🚀 Bắt đầu chu kỳ lấy dữ liệu bất đồng bộ...")
        full_data_cache = {}
        async with aiohttp.ClientSession() as session:
            # Tạo danh sách các tác vụ cần thực hiện
            tasks = [self.data_manager.fetch_multi_timeframe_data_async(session, symbol) for symbol in SYMBOLS]
            
            # Chạy tất cả các tác vụ song song
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Xử lý kết quả
            for result in results:
                if isinstance(result, Exception):
                    print(f"⚠️ Lỗi trong một tác vụ: {result}")
                    continue
                
                symbol, multi_tf_data = result
                if multi_tf_data:
                    primary_tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, PRIMARY_TIMEFRAME_DEFAULT)
                    if primary_tf in multi_tf_data:
                        # (Logic tạo feature từ multi_tf_data của bạn sẽ được gọi ở đây)
                        # Ví dụ:
                        # df_enhanced = self.data_manager.create_enhanced_features_from_data(multi_tf_data)
                        # full_data_cache[symbol] = df_enhanced
                        print(f"✅ Đã xử lý xong dữ liệu cho {symbol}")
                    else:
                        print(f"❌ Dữ liệu cho {symbol} không có khung thời gian chính.")
                else:
                    print(f"❌ Không lấy được dữ liệu cho {symbol}")

        print("✅ Chu kỳ lấy dữ liệu bất đồng bộ hoàn tất.")
        return full_data_cache
def save_model_with_metadata(symbol, model_data, model_type="ensemble"):
    """
    Lưu model và metadata.
    PHIÊN BẢN MỚI: Bổ sung cv_mean_accuracy vào metadata.
    """
    if model_data is None or "ensemble" not in model_data:
        print(f"⚠️ Không có model_data hoặc ensemble để lưu cho {symbol} ({model_type}). Bỏ qua.")
        return

    ensemble_model = model_data["ensemble"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = os.path.join(MODEL_DIR, f"{model_type}_model_{symbol}_{timestamp}.pkl")
    metadata_file = filename.replace(".pkl", ".json")

    try:
        joblib.dump(model_data, filename)

        first_model_name = list(ensemble_model.cv_results.keys())[0] if ensemble_model.cv_results else "unknown"
        cv_info = ensemble_model.cv_results.get(first_model_name, {})

        metadata = {
            "symbol": symbol,
            "timestamp": timestamp,
            "feature_columns": model_data.get("feature_columns", []),
            "cv_mean_f1": cv_info.get("mean_f1"),
            "cv_std_f1": cv_info.get("std_f1"),
            # === THÊM DÒNG NÀY ===
            "cv_mean_accuracy": cv_info.get("mean_accuracy"),
            # ======================
            "model_weights": ensemble_model.model_weights,
            "model_type": model_type,
            "model_file": filename,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model for {symbol} ({model_type}) saved at {filename}")
        print(f"   - CV F1-Score: {metadata['cv_mean_f1']:.3f} +/- {metadata['cv_std_f1']:.3f}")
        print(f"   - CV Accuracy: {metadata['cv_mean_accuracy']:.3f}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu model cho {symbol} ({model_type}): {e}")

# <<< THÊM HÀM MỚI NÀY VÀO FILE BOT >>>




# TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY

# TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

def load_latest_model(symbol, model_type="ensemble"):
    if not os.path.exists(MODEL_DIR): return None

    pattern = f"{model_type}_model_{symbol}"
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(pattern) and f.endswith(".pkl")]
    if not files: return None

    files.sort(reverse=True)
    latest_pkl_file = os.path.join(MODEL_DIR, files[0])
    metadata_file = latest_pkl_file.replace(".pkl", ".json")

    try:
        model_data = joblib.load(latest_pkl_file)
        
        # <<< CẢI TIẾN: LOGIC KIỂM TRA TÍNH TƯƠNG THÍCH MỚI >>>
        is_valid = False
        ensemble_model = model_data.get("ensemble")
        # Dấu hiệu của model Stacking (Bot LLMs) là có thuộc tính 'meta_model'
        if ensemble_model and hasattr(ensemble_model, 'meta_model'):
            is_valid = True
        
        if is_valid:
            print(f"✅ Đã tải model ({model_type}) tương thích cho {symbol} từ: {latest_pkl_file}")
            return model_data
        else:
            print(f"⚠️ Phát hiện model cũ không tương thích (thiếu 'meta_model') cho {symbol}.")
            print(f"   -> Tự động xóa file: {latest_pkl_file}")
            try:
                os.remove(latest_pkl_file)
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
            except OSError as e:
                print(f"   ❌ Lỗi khi xóa file cũ: {e}")
            return None # Trả về None để kích hoạt huấn luyện lại

    except Exception as e:
        print(f"❌ Lỗi khi tải hoặc kiểm tra model từ {latest_pkl_file}: {e}")
        return None

def save_open_positions(
    open_positions, filename=f"open_positions_{PRIMARY_TIMEFRAME.lower()}.json"
):
    try:
        positions_to_save = {
            symbol: {
                **pos,
                "opened_at": (
                    pos["opened_at"].isoformat()
                    if isinstance(pos.get("opened_at"), datetime)
                    else pos.get("opened_at")
                ),
            }
            for symbol, pos in open_positions.items()
        }
        with open(filename, "w") as f:
            json.dump(positions_to_save, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving positions: {e}")


def load_open_positions(filename=f"open_positions_{PRIMARY_TIMEFRAME.lower()}.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        loaded_positions = {}
        for sym, pos in data.items():
            # Tự động gán symbol vào bên trong dictionary con
            pos["symbol"] = sym

            # Xử lý thời gian
            pos["opened_at"] = (
                datetime.fromisoformat(pos["opened_at"]).astimezone(
                    pytz.timezone("Asia/Bangkok")
                )
                if "opened_at" in pos and isinstance(pos["opened_at"], str)
                else None
            )

            loaded_positions[sym] = pos

        print(f"✅ Loaded {len(loaded_positions)} positions")
        return loaded_positions
    except (FileNotFoundError, json.JSONDecodeError):
        print("📝 No existing positions file, starting fresh")
        return {}
    except Exception as e:
        print(f"❌ Error loading positions: {e}")
        return {}


# <<< CẢI TIẾN 1: THAY THẾ TOÀN BỘ LỚP TRADINGENVIRONMENT BẰNG LỚP NÀY >>>
# TÌM VÀ THAY THẾ TOÀN BỘ LỚP NÀY

class PortfolioEnvironment(gym.Env):
    """
    Môi trường RL quản lý danh mục.
    NÂNG CẤP: Hàm thưởng (reward) dựa trên Sharpe Ratio để hướng tới lợi nhuận ổn định.
    """
    metadata = {'render_modes': ['human']}

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP PortfolioEnvironment

    def __init__(self, dict_df_features, dict_feature_columns, symbols, initial_balance=10000):
        super(PortfolioEnvironment, self).__init__()

        self.initial_balance = initial_balance
        self.feature_columns_map = dict_feature_columns
        self.dfs = {}
        self.features = {}
        valid_symbols_for_env = []

        # --- Bước 1: Chuẩn bị dữ liệu thực tế (giữ nguyên) ---
        for symbol in symbols:
            df = dict_df_features.get(symbol)
            feature_cols = self.feature_columns_map.get(symbol)
            if df is None or df.empty or not feature_cols: continue

            # Chỉ lấy các cột có sẵn trong dataframe
            available_cols = [col for col in feature_cols if col in df.columns]
            features_df = df[available_cols].dropna()

            if not features_df.empty:
                self.features[symbol] = features_df.to_numpy()
                self.dfs[symbol] = df.loc[features_df.index]
                valid_symbols_for_env.append(symbol)
        
        self.symbols = valid_symbols_for_env
        self.n_symbols = len(self.symbols)

        if self.n_symbols == 0:
            raise ValueError("Không thể tạo Env: Không có symbol hợp lệ.")

        self.max_steps = min(len(self.features[s]) for s in self.symbols)

        # --- Bước 2: <<< CẢI TIẾN: Tính toán kích thước Observation Space DỰA TRÊN DỮ LIỆU THỰC TẾ >>> ---
        # Lấy kích thước của số feature thị trường từ chính array numpy đã được tạo
        single_symbol_market_features_dim = self.features[self.symbols[0]].shape[1]
        
        # Cộng thêm 3 state của vị thế (vị thế, pnl, thời gian)
        single_symbol_obs_size = single_symbol_market_features_dim + 3
        
        # Cộng thêm 4 state toàn cục
        total_obs_size = self.n_symbols * single_symbol_obs_size + 4
        
        print(f"   [Env Init] Kích thước Observation Space được tính toán: {total_obs_size}")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3] * self.n_symbols)
        # --- KẾT THÚC CẢI TIẾN ---

        self.returns_history = deque(maxlen=30)
        self.reset()

    def _get_observation(self):
        # Hàm này sử dụng logic nâng cao từ Bot LLMs
        all_symbol_obs, num_open_positions = [], 0
        all_atr_normalized, trending_symbols_count = [], 0

        for i, symbol in enumerate(self.symbols):
            current_candle_data = self.dfs[symbol].iloc[self.current_step]
            market_obs = self.features[symbol][self.current_step]
            all_atr_normalized.append(current_candle_data.get('atr_normalized', 0))
            if current_candle_data.get('market_regime', 0) != 0: trending_symbols_count += 1

            pos_state = self.positions[i]
            pnl_state = self.unrealized_pnls[i] / self.balance if self.balance != 0 else 0.0
            time_state = self.time_in_trades[i] / 100.0
            symbol_obs = np.append(market_obs, [pos_state, pnl_state, time_state])
            all_symbol_obs.append(symbol_obs)
            if self.positions[i] != 0: num_open_positions += 1

        flat_obs = np.concatenate(all_symbol_obs).astype(np.float32)
        global_states = np.array([
            (self.balance / self.initial_balance) - 1.0,
            num_open_positions / self.n_symbols,
            np.mean(all_atr_normalized) if all_atr_normalized else 0,
            trending_symbols_count / self.n_symbols
        ]).astype(np.float32)
        return np.append(flat_obs, global_states)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        self.positions = [0] * self.n_symbols
        self.entry_prices = [0] * self.n_symbols
        self.unrealized_pnls = [0] * self.n_symbols
        self.time_in_trades = [0] * self.n_symbols
        self.returns_history.clear() # <<< THÊM MỚI
        return self._get_observation(), {}

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP PortfolioEnvironment

    def step(self, action):
        previous_balance = self.balance

        # --- PHẦN LOGIC CẬP NHẬT TRẠNG THÁI BỊ THIẾU ---
        # (Phần này được lấy lại từ logic chuẩn của môi trường)
        total_unrealized_pnl = 0
        for i in range(self.n_symbols):
            if self.positions[i] != 0:
                self.time_in_trades[i] += 1
                price_now = self.dfs[self.symbols[i]]['close'].iloc[self.current_step]
                pnl_for_symbol = (price_now - self.entry_prices[i]) * self.positions[i]
                self.unrealized_pnls[i] = pnl_for_symbol
                total_unrealized_pnl += pnl_for_symbol

        realized_pnl_this_step = 0
        action_vector = np.asarray(action).flatten()
        for i, action_code in enumerate(action_vector):
            symbol = self.symbols[i]
            current_price = self.dfs[symbol]['close'].iloc[self.current_step]
            pos = self.positions[i]

            if action_code == 1 and pos == 0: # Mở lệnh MUA
                self.positions[i] = 1
                self.entry_prices[i] = current_price
                self.time_in_trades[i] = 0
            elif action_code == 2 and pos == 0: # Mở lệnh BÁN
                self.positions[i] = -1
                self.entry_prices[i] = current_price
                self.time_in_trades[i] = 0
            elif action_code == 0 and pos != 0: # Đóng lệnh
                pnl = (current_price - self.entry_prices[i]) * self.positions[i]
                realized_pnl_this_step += pnl
                self.positions[i] = 0
                self.entry_prices[i] = 0
                self.unrealized_pnls[i] = 0
                self.time_in_trades[i] = 0
        
        self.balance += realized_pnl_this_step
        # --- KẾT THÚC PHẦN LOGIC CẬP NHẬT ---

        # Phần tính toán hàm thưởng Sharpe Ratio đã đúng
        step_return = (self.balance / previous_balance) - 1 if previous_balance > 0 else 0.0
        self.returns_history.append(step_return)
        if len(self.returns_history) > 1:
            risk = np.std(self.returns_history) + 1e-9
            reward = np.mean(self.returns_history) / risk
        else:
            reward = step_return
        if np.all(action_vector == 0): reward -= 0.001
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps - 1
        obs = self._get_observation()
        info = {'balance': self.balance}
        return obs, reward, terminated, False, info
# --- NEW: RL AGENT CLASS ---
class RLAgent:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path)
            print(f"✅ RL Agent loaded from {model_path}")
        else:
            print("⚠️ RL Agent not loaded. Needs to be trained.")

    def train(self, env, total_timesteps=20000, save_path=None):
        """
        Huấn luyện PPO agent với Callback giám sát Drawdown.
        """
        if self.model:
            self.model.set_env(env)
        else:
            # Bạn có thể truyền các tham số PPO đã được Optuna tối ưu hóa ở đây
            self.model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=f"./ppo_tensorboard_{PRIMARY_TIMEFRAME.lower()}/")

        # <<< CẢI TIẾN: Khởi tạo và sử dụng Callback >>>
        # Dừng huấn luyện nếu trong một episode, tài khoản sụt giảm quá 25%
        drawdown_callback = StopTrainingOnMaxDrawdown(max_drawdown_threshold=0.25, verbose=1)
        # <<< KẾT THÚC CẢI TIẾN >>>

        print("🤖 Bắt đầu huấn luyện RL Agent (với giám sát Drawdown)...")
        # Truyền callback vào hàm learn()
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=drawdown_callback) 
        logging.info("✅ RL Agent training completed.")

        if save_path:
            self.model.save(save_path)
            print(f"💾 RL Agent đã được lưu tại {save_path}")

    def predict(self, observation):
        """Predicts the next action."""
        if not self.model:
            return 0 # Default to Hold if not trained

        action, _states = self.model.predict(observation, deterministic=True)
        return action.item()


class EnhancedTradingBot:
    # TÌM VÀ THAY THẾ HÀM NÀY TRONG LỚP EnhancedTradingBot

    def __init__(self):
        self.logger = logging.getLogger('EnhancedTradingBot')
        self.models = {}
        self.trending_models = {}
        self.ranging_models = {}
        self.data_manager = EnhancedDataManager()
        self.market_regime_cache = {}
        self.drift_monitor = None
        self.risk_manager = PortfolioRiskManager(SYMBOLS, self.data_manager)
        raw_positions = load_open_positions()
        self.open_positions = {}
        for sym, pos_data in raw_positions.items():
            pos_data.setdefault("initial_confidence", pos_data.get("confidence", 0.0))
            pos_data.setdefault(
                "last_notified_confidence", pos_data.get("confidence", 0.0)
            )
            pos_data.setdefault("last_sl_notified", pos_data.get("sl"))
            self.open_positions[sym] = pos_data

        self.db_file = f"trading_log_portfolio_rl1.db"
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            print(f"✅ Kết nối thành công tới database SQLite: {self.db_file}")
            self._create_trades_table()
        except Exception as e:
            print(f"❌ Lỗi kết nối database SQLite: {e}")

        self.performance_metrics = {}
        self.update_performance_metrics()

        self.news_manager = NewsEconomicManager()
        self.CONFIDENCE_CHANGE_THRESHOLD = 0.05
        self.SL_CHANGE_NOTIFICATION_THRESHOLD_PIPS = 5
        self.high_impact_events = ["NFP", "CPI", "FOMC", "Interest Rate", "GDP", "PCE"]

        self.portfolio_rl_agent = None
        self.use_rl = True
        self.weekend_close_executed = False
        self.active_symbols = set()

        # <<< THÊM 2 DÒNG NÀY VÀO CUỐI HÀM __init__ >>>
        self.drift_scores = {} # Lưu điểm cảnh báo drift cho mỗi symbol
        self.symbols_flagged_for_retrain = set() # Lưu các symbol chờ huấn luyện vào cuối tuần
        self.consecutive_data_failures = 0
    def _filter_data_for_curriculum(self, df, difficulty='hard'):
        """
        Hàm trợ giúp để lọc dữ liệu cho Curriculum Learning.
        'easy': Xu hướng rõ, biến động thấp.
        'medium': Thêm cả sideways.
        'hard': Toàn bộ dữ liệu.
        """
        if difficulty == 'hard':
            return df

        # Tính ADX và ATR để lọc
        adx = ADXIndicator(df['high'], df['low'], df['close']).adx()
        atr_norm = (AverageTrueRange(df['high'], df['low'], df['close']).average_true_range() / df['close'])

        if difficulty == 'easy':
            # Lấy 30% dữ liệu có xu hướng mạnh nhất (ADX cao) và 30% có biến động thấp nhất (ATR thấp)
            strong_trend_threshold = adx.quantile(0.7)
            low_volatility_threshold = atr_norm.quantile(0.3)

            filtered_df = df[(adx > strong_trend_threshold) & (atr_norm < low_volatility_threshold)]
            print(f"   [Curriculum] Easy mode: Lọc được {len(filtered_df)} / {len(df)} mẫu.")
            return filtered_df if not filtered_df.empty else df # Fallback

        if difficulty == 'medium':
            # Lấy 70% dữ liệu có ADX thấp (bao gồm cả sideways và trend nhẹ)
            medium_trend_threshold = adx.quantile(0.7)
            filtered_df = df[adx < medium_trend_threshold]
            print(f"   [Curriculum] Medium mode: Lọc được {len(filtered_df)} / {len(df)} mẫu.")
            return filtered_df if not filtered_df.empty else df # Fallback

        return df
    def check_api_connection(self):
            """Check connection to OANDA API."""
            try:
                response = requests.get(
                    f"{OANDA_URL}/accounts",
                    headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
                    timeout=10,
                )
                if response.status_code == 200:
                    print("✅ Kết nối OANDA API thành công.")
                    return True
                else:
                    print(f"❌ Kết nối OANDA API thất bại. Status: {response.status_code}, Response: {response.text}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"❌ Lỗi kết nối OANDA API: {e}")
                return False
    # VỊ TRÍ: Thay thế hàm _create_trades_table trong lớp EnhancedTradingBot

    # TÌM VÀ THAY THẾ HÀM NÀY trong lớp EnhancedTradingBot

    def _create_trades_table(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    closed_at TIMESTAMP,
                    reason TEXT,
                    pips REAL,
                    confidence REAL,
                    -- CÁC CỘT MỚI CHO TCA --
                    signal_price REAL,    -- Giá mid tại thời điểm tín hiệu
                    execution_slippage_pips REAL, -- Độ trượt giá khi thực thi
                    spread_cost_pips REAL -- Chi phí spread
                )
            """
            )
            self.conn.commit()
            print("[DB] Bảng 'trades' đã sẵn sàng với các cột TCA.")
        except Exception as e:
            print(f"❌ Lỗi tạo bảng 'trades': {e}")

    def update_performance_metrics(self):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            total_trades_result = cursor.execute(
                "SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL"
            ).fetchone()
            total_trades = total_trades_result[0] if total_trades_result else 0

            winning_trades_result = cursor.execute(
                "SELECT COUNT(*) FROM trades WHERE pips > 0 AND exit_price IS NOT NULL"
            ).fetchone()
            winning_trades = winning_trades_result[0] if winning_trades_result else 0

            total_pips_result = cursor.execute(
                "SELECT SUM(pips) FROM trades WHERE exit_price IS NOT NULL"
            ).fetchone()
            total_pips = total_pips_result[0] if total_pips_result[0] is not None else 0

            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            self.performance_metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "total_pips": total_pips,
                "win_rate": win_rate,
            }
        except Exception as e:
            print(f"❌ Lỗi cập nhật chỉ số hiệu suất: {e}")
    # <<< THÊM HÀM MỚI NÀY VÀO TRONG LỚP EnhancedTradingBot >>>
    def _objective_rl_portfolio(self, trial, train_env, val_env):
        """
        Hàm mục tiêu Optuna cho PortfolioEnvironment.
        """
        params = {
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048]),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.1, log=True),
            'clip_range': trial.suggest_categorical('clip_range', [0.2, 0.3]),
        }

        try:
            # <<< CẢI TIẾN 2: SỬ DỤNG MLP-LSTM POLICY >>>
            # (bên trong _objective_rl_portfolio)
            # <<< SỬA LỖI: Sử dụng "MlpPolicy" với policy_kwargs để kích hoạt LSTM >>>
            # (bên trong _objective_rl_portfolio)
            # <<< SỬA LỖI: Sử dụng "MlpPolicy" tiêu chuẩn, bỏ tham số LSTM >>>
            policy_kwargs = dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64])
            )
            model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=0, **params)
            model.learn(total_timesteps=15000, progress_bar=False)
        except Exception as e:
            print(f"   [Optuna Portfolio] Trial failed. Error: {e}")
            return -1e9

        obs, _ = val_env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = val_env.step(action)

        final_balance = val_env.balance
        return final_balance
    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    def load_or_train_models(self, force_retrain_symbols=None):
        if force_retrain_symbols is None:
            force_retrain_symbols = []

        if force_retrain_symbols:
            print(f"🔥 Bắt đầu chu kỳ huấn luyện lại bắt buộc cho: {force_retrain_symbols}")
            self.data_manager.data_cache.clear()

        self.active_symbols.clear()
        print("Bắt đầu xây dựng/cập nhật danh sách symbol hoạt động...")
        full_data_cache = {}

        def _check_quality_gates(model_data, symbol, model_type):
            if not model_data: return False
            f1, std_f1, accuracy = 0, 1.0, 0
            metadata_f1 = model_data.get("cv_mean_f1")
            if metadata_f1 is not None:
                f1, std_f1, accuracy = metadata_f1, model_data.get("cv_std_f1", 1.0), model_data.get("cv_mean_accuracy", 0)
            elif "ensemble" in model_data and model_data["ensemble"].cv_results:
                try:
                    key = next(iter(model_data['ensemble'].cv_results))
                    res = model_data['ensemble'].cv_results[key]
                    f1, std_f1, accuracy = res['mean_f1'], res['std_f1'], res['mean_accuracy']
                except (KeyError, IndexError, StopIteration): return False
            else: return False
            if f1 >= MIN_F1_SCORE_GATE and std_f1 <= MAX_STD_F1_GATE and accuracy >= MIN_ACCURACY_GATE: return True
            else:
                print(f"  ❌ Model {model_type} cho {symbol} không đạt chuẩn (F1:{f1:.3f}, STD:{std_f1:.3f}, Acc:{accuracy:.3f}).")
                return False

        def _train_and_evaluate(symbol_to_train, df_filtered, model_type):
            best_model_data, best_f1 = None, -1
            if df_filtered is None or len(df_filtered) < MIN_SAMPLES_GATE:
                print(f"  ❌ Không đủ dữ liệu ({len(df_filtered) if df_filtered is not None else 0}/{MIN_SAMPLES_GATE} mẫu) cho model {model_type}.")
                return None
            for attempt in range(MAX_RETRAIN_ATTEMPTS):
                print(f"   -> Đang huấn luyện model {model_type} cho {symbol_to_train} lần {attempt + 1}/{MAX_RETRAIN_ATTEMPTS}...")
                model_data_new = self.train_enhanced_model(symbol_to_train, df_filtered)
                if model_data_new:
                    try:
                        key = next(iter(model_data_new['ensemble'].cv_results))
                        f1 = model_data_new['ensemble'].cv_results[key]['mean_f1']
                        print(f"   -> Kết quả lần {attempt + 1}: F1-Score = {f1:.3f}")
                        if f1 > best_f1: best_f1, best_model_data = f1, model_data_new
                        if best_f1 >= MIN_F1_SCORE_GATE:
                            print("   🎉 Đạt F1-Score mục tiêu! Dừng sớm.")
                            break
                    except (KeyError, IndexError, StopIteration): continue
            return best_model_data

        # --- Vòng lặp xử lý model Ensemble (Trending/Ranging) ---
            # --- Vòng lặp xử lý model Ensemble (Trending/Ranging) ---
        for symbol in SYMBOLS:
            print("-" * 40)
            print(f"Processing models for symbol: {symbol}")

            # <<< BƯỚC 3: TÍCH HỢP HÀM KIỂM TRA GIỜ GIAO DỊCH >>>
            if not is_market_open(symbol):
                print(f"   [Market Status] ℹ️ Thị trường cho {symbol} đang đóng cửa. Bỏ qua.")
                continue # Chuyển sang symbol tiếp theo
            # <<< KẾT THÚC TÍCH HỢP >>>
            
            # (Toàn bộ logic còn lại của vòng lặp để tải hoặc huấn luyện model giữ nguyên)
            is_forced = symbol in force_retrain_symbols
            is_symbol_active = False
            df_full_for_symbol = self.data_manager.create_enhanced_features(symbol)
            
            if df_full_for_symbol is None:
                # create_enhanced_features đã log lỗi (ví dụ: do dữ liệu cũ)
                continue
                
            full_data_cache[symbol] = df_full_for_symbol
            
            # Xử lý model Trending
            model_trending = load_latest_model(symbol, "ensemble_trending")
            if is_forced or not model_trending:
                df_trending = df_full_for_symbol[df_full_for_symbol['market_regime'] != 0].copy()
                best_trending_model = _train_and_evaluate(symbol, df_trending, "TRENDING")
                if _check_quality_gates(best_trending_model, symbol, "TRENDING"):
                    self.trending_models[symbol] = best_trending_model
                    save_model_with_metadata(symbol, best_trending_model, "ensemble_trending")
                    is_symbol_active = True
            elif _check_quality_gates(model_trending, symbol, "TRENDING"):
                self.trending_models[symbol] = model_trending
                is_symbol_active = True
                
            # Xử lý model Ranging
            model_ranging = load_latest_model(symbol, "ensemble_ranging")
            if is_forced or not model_ranging:
                df_ranging = df_full_for_symbol[df_full_for_symbol['market_regime'] == 0].copy()
                best_ranging_model = _train_and_evaluate(symbol, df_ranging, "RANGING")
                if _check_quality_gates(best_ranging_model, symbol, "RANGING"):
                    self.ranging_models[symbol] = best_ranging_model
                    save_model_with_metadata(symbol, best_ranging_model, "ensemble_ranging")
                    is_symbol_active = True
            elif _check_quality_gates(model_ranging, symbol, "RANGING"):
                self.ranging_models[symbol] = model_ranging
                is_symbol_active = True

            if is_symbol_active:
                self.active_symbols.add(symbol)
                print(f"  ✅ Symbol {symbol} đã được KÍCH HOẠT.")
            else:
                self.trending_models.pop(symbol, None)
                self.ranging_models.pop(symbol, None)
                print(f"  ❌ Symbol {symbol} BỊ VÔ HIỆU HÓA do không có model nào đạt chuẩn.")



        # --- Vòng lặp xử lý RL Agent ---
        if self.use_rl:
            rl_model_path = os.path.join(MODEL_DIR, "rl_portfolio_agent.zip")
            force_retrain_rl = any(s in self.active_symbols for s in force_retrain_symbols)

            if os.path.exists(rl_model_path) and not force_retrain_rl:
                print(f"✅ Đã tìm thấy file Portfolio RL Agent, đang tải và gắn môi trường...")
                self.portfolio_rl_agent = RLAgent(model_path=rl_model_path)
                # (Logic kiểm tra tương thích môi trường giữ nguyên)
            else:
                # <<< BẮT ĐẦU KHỐI LOGIC HUẤN LUYỆN RL ĐÃ NÂNG CẤP >>>
                print("-" * 40)
                print("🤖 Bắt đầu quá trình huấn luyện Portfolio RL Agent...")

                dict_dfs, dict_feature_cols = {}, {}
                min_data_length = float('inf')
                active_symbols_for_rl = list(self.active_symbols)

                if len(active_symbols_for_rl) < 2:
                    print("❌ Không đủ symbol hoạt động (>1) để huấn luyện RL Agent. Bỏ qua.")
                    self.portfolio_rl_agent = None
                    return

                # Chuẩn bị dữ liệu cho RL
                for symbol in active_symbols_for_rl:
                    model_ref_data = self.trending_models.get(symbol) or self.ranging_models.get(symbol)
                    if not model_ref_data: continue
                    df_full = full_data_cache.get(symbol)
                    if df_full is None or df_full.empty: continue
                    dict_dfs[symbol] = df_full
                    dict_feature_cols[symbol] = model_ref_data['feature_columns'] + ['confidence'] # Thêm confidence vào features
                    min_data_length = min(min_data_length, len(df_full))

                if min_data_length < 500: # Cần đủ dữ liệu để chia
                     print(f"❌ Dữ liệu chung quá ngắn ({min_data_length} nến) để huấn luyện RL. Bỏ qua.")
                     self.portfolio_rl_agent = None
                     return
                
                # Cắt ngắn tất cả dataframe về cùng độ dài
                for symbol in active_symbols_for_rl:
                    dict_dfs[symbol] = dict_dfs[symbol].tail(min_data_length)

                # --- LOGIC SPLIT DỮ LIỆU AN TOÀN 60-20-20 ---
                print(f"   [RL Training] Phân chia {min_data_length} nến: 60% Train, 20% Validation (cho Optuna), 20% Final Test...")
                train_end_idx = int(min_data_length * 0.60)
                val_end_idx = int(min_data_length * 0.80)

                dict_dfs_train = {s: df.iloc[:train_end_idx] for s, df in dict_dfs.items()}
                dict_dfs_val = {s: df.iloc[train_end_idx:val_end_idx] for s, df in dict_dfs.items()}
                dict_dfs_test = {s: df.iloc[val_end_idx:] for s, df in dict_dfs.items()} # Tập này Optuna không thấy
                # --- KẾT THÚC LOGIC SPLIT ---

                # Tạo môi trường cho Optuna và test cuối cùng
                train_env = PortfolioEnvironment(dict_dfs_train, dict_feature_cols, active_symbols_for_rl)
                val_env = PortfolioEnvironment(dict_dfs_val, dict_feature_cols, active_symbols_for_rl)
                test_env = PortfolioEnvironment(dict_dfs_test, dict_feature_cols, active_symbols_for_rl)

                print("\n--- 🤖 Bắt đầu tối ưu hóa siêu tham số RL với Optuna ---")
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: self._objective_rl_portfolio(trial, train_env, val_env), n_trials=30)
                best_params = study.best_params
                print(f"   [Optuna] ✅ Tham số tốt nhất được tìm thấy: {best_params}")

                # Huấn luyện agent cuối cùng trên 80% dữ liệu
                print("\n--- 🤖 Bắt đầu huấn luyện RL Agent cuối cùng với tham số tốt nhất ---")
                full_train_data = {s: df.iloc[:val_end_idx] for s, df in dict_dfs.items()}
                full_train_env = PortfolioEnvironment(full_train_data, dict_feature_cols, active_symbols_for_rl)

                final_agent = RLAgent()
                final_model = PPO("MlpPolicy", full_train_env, verbose=0, **best_params, tensorboard_log="./portfolio_ppo_tensorboard/")
                final_agent.model = final_model
                final_agent.train(full_train_env, total_timesteps=30000, save_path=rl_model_path)
                self.portfolio_rl_agent = final_agent
                print("✅✅✅ Portfolio RL Agent đã được huấn luyện!")

                # Đánh giá cuối cùng trên tập test chưa từng thấy
                print("\n--- 📊 Đánh giá cuối cùng trên tập Test Set (unseen data) ---")
                obs, _ = test_env.reset()
                terminated, truncated = False, False
                while not (terminated or truncated):
                    action, _ = final_agent.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = test_env.step(action)
                
                final_balance_test = test_env.balance
                performance_test = (final_balance_test / test_env.initial_balance - 1) * 100
                print(f"   [Final Test] Lợi nhuận trên tập test: {performance_test:.2f}%")
                self.send_discord_alert(f"📈 **Kết Quả Huấn Luyện RL Mới** 📈\n- Lợi nhuận trên tập test (dữ liệu chưa từng thấy): **{performance_test:.2f}%**")
                # <<< KẾT THÚC KHỐI LOGIC HUẤN LUYỆN RL >>>
    def train_enhanced_model(self, symbol, df_to_train): # <--- Thêm df_to_train
        """This function now takes a pre-filtered DataFrame."""
        print(f"🎯 Bắt đầu huấn luyện model cho {symbol}...")
        df = df_to_train # <--- Sử dụng DataFrame được truyền vào

        if df is None or df.empty:
            print(f"❌ Không có dữ liệu để huấn luyện cho {symbol}.")
            return None
        # --- BƯỚC CẢI TIẾN: THÊM FEATURE KINH TẾ ---
        # Lưu ý: Việc thêm sentiment lịch sử yêu cầu nguồn dữ liệu trả phí.
        # Ở đây chúng ta tập trung vào dữ liệu lịch kinh tế có sẵn.
        df = self.news_manager.add_economic_event_features(df, symbol)
        # --- KẾT THÚC BƯỚC CẢI TIẾN ---

        # Tạo label ở đây, sau khi đã có đủ features
        for horizon in [1, 3, 5]:
            df[f"label_{horizon}"] = (df["close"].shift(-horizon) > df["close"]).astype(
                int
            )

        # Pipeline làm sạch dữ liệu giữ nguyên như cũ
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["label_3"], inplace=True)

        feature_cols = [col for col in df.columns if not col.startswith("label")]
        X = df[feature_cols].copy()
        y = df["label_3"].copy()

        X.fillna(method="ffill", inplace=True)
        X.fillna(method="bfill", inplace=True)

        cols_to_drop = X.columns[X.isna().all()].tolist()
        if cols_to_drop:
            X.drop(columns=cols_to_drop, inplace=True)
            feature_cols = [col for col in feature_cols if col not in cols_to_drop]

        y = y.reindex(X.index)
        all_data = pd.concat([X, y], axis=1)
        all_data.dropna(inplace=True)
        X = all_data[feature_cols]
        y = all_data["label_3"]

        # Enhanced minimum samples check
        min_samples = ML_CONFIG.get("MIN_SAMPLES_FOR_TRAINING", 300)
        if len(X) < min_samples:
            print(
                f"❌ Không đủ dữ liệu sạch cho {symbol}: chỉ có {len(X)} records (yêu cầu {min_samples})."
            )
            return None

        # ... (code loại bỏ tương quan, chọn feature, huấn luyện ensemble)
        print(
            f"✅ Đã làm sạch dữ liệu, còn lại {len(X)} records để huấn luyện cho {symbol}."
        )

        # Enhanced correlation removal with stricter threshold
        correlation_threshold = ML_CONFIG.get("MAX_CORRELATION_THRESHOLD", 0.90)
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > correlation_threshold)
        ]
        X = X.drop(columns=high_corr_features)

        # Enhanced feature selection with multiple algorithms
        rf_temp = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42, 
            n_jobs=-1
        )
        rf_temp.fit(X, y)
        
        # Get feature importance from Random Forest
        rf_importance = pd.DataFrame({
            "feature": X.columns, 
            "rf_importance": rf_temp.feature_importances_
        })
        
        # Also get feature importance from XGBoost for comparison
        try:
            xgb_temp = xgb.XGBClassifier(
                n_estimators=50, 
                max_depth=6, 
                random_state=42, 
                n_jobs=-1
            )
            xgb_temp.fit(X, y)
            xgb_importance = pd.DataFrame({
                "feature": X.columns,
                "xgb_importance": xgb_temp.feature_importances_
            })
            
            # Combine importances
            feature_importance = rf_importance.merge(xgb_importance, on="feature")
            feature_importance["combined_importance"] = (
                feature_importance["rf_importance"] * 0.6 + 
                feature_importance["xgb_importance"] * 0.4
            )
        except:
            feature_importance = rf_importance
            feature_importance["combined_importance"] = feature_importance["rf_importance"]
        
        # Filter by minimum importance threshold
        min_importance = ML_CONFIG.get("FEATURE_IMPORTANCE_THRESHOLD", 0.01)
        feature_importance = feature_importance[
            feature_importance["combined_importance"] >= min_importance
        ]
        
        # Select top features
        top_features = feature_importance.nlargest(
            ML_CONFIG["FEATURE_SELECTION_TOP_K"], 
            "combined_importance"
        )["feature"].tolist()
        X_selected = X[top_features]

        logging.info(f"📊 Training with {len(top_features)} best features for {symbol}")
        ensemble = EnsembleModel()
        ensemble.train_ensemble(X_selected, y)  # Hàm này sẽ được nâng cấp ở bước 2
        print("   [Drift] Khởi tạo Drift Monitor với dữ liệu tham chiếu mới...")
        self.drift_monitor = DriftMonitor(X_selected)
        model_data = {"ensemble": ensemble, "feature_columns": top_features}
        return model_data

    # This is a helper function, no changes needed
    def get_enhanced_signal(self, symbol, for_open_position_check=False):
        """
        Lấy tín hiệu và độ tin cậy từ model Ensemble phù hợp với trạng thái thị trường.
        PHIÊN BẢN SỬA LỖI: Tìm kiếm model trong self.trending_models hoặc self.ranging_models.
        """
        # --- BƯỚC 1: LẤY DỮ LIỆU VÀ XÁC ĐỊNH TRẠNG THÁI THỊ TRƯỜNG ---
        df_features = self.data_manager.create_enhanced_features(symbol)
        if df_features is None or len(df_features) < 100:
            print(f"[{symbol}] ⚠️ Không đủ dữ liệu để phân tích trong get_enhanced_signal.")
            return None, 0.0, None

        current_regime = df_features['market_regime'].iloc[-1]

        # --- BƯỚC 2: CHỌN ĐÚNG MODEL DỰA TRÊN TRẠNG THÁI ---
        if current_regime != 0:  # Thị trường có xu hướng
            model_data = self.trending_models.get(symbol)
        else:  # Thị trường đi ngang
            model_data = self.ranging_models.get(symbol)

        # --- BƯỚC 3: KIỂM TRA MODEL VÀ TIẾP TỤC LOGIC ---
        if model_data is None:
            print(f"[{symbol}] ❌ Không tìm thấy model phù hợp với trạng thái thị trường hiện tại (Regime: {current_regime}).")
            return None, 0.0, None

        model = model_data.get("ensemble")
        feature_columns = model_data.get("feature_columns")

        if not model or not feature_columns:
            print(f"[{symbol}] ❌ Model hoặc feature_columns không hợp lệ trong model_data.")
            return None, 0.0, None

        # Pipeline làm sạch dữ liệu mạnh mẽ (giữ nguyên từ logic cũ của bạn)
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features.fillna(method="ffill", inplace=True)
        df_features.fillna(method="bfill", inplace=True)
        cols_to_drop = df_features.columns[df_features.isna().all()].tolist()
        if cols_to_drop:
            df_features.drop(columns=cols_to_drop, inplace=True)
        df_features.dropna(inplace=True)

        if len(df_features) < 10:
            print(f"[{symbol}] ⚠️ Dữ liệu đặc trưng quá ít sau khi làm sạch ({len(df_features)} dòng).")
            return None, 0.0, None

        try:
            # Đảm bảo các cột features của model phải tồn tại trong df_features
            available_columns = [col for col in feature_columns if col in df_features.columns]
            X = df_features[available_columns]

            # Đảm bảo X luôn có đủ các cột như lúc huấn luyện, đúng thứ tự.
            # Nếu thiếu cột, nó sẽ được tạo ra và điền bằng 0.
            X = X.reindex(columns=feature_columns, fill_value=0)

            prob_buy = model.predict_proba(X)

            if prob_buy is None or not isinstance(prob_buy, (int, float)) or not (0 <= prob_buy <= 1):
                print(f"[{symbol}] ⚠️ Model trả về xác suất không hợp lệ: {prob_buy}. Bỏ qua.")
                return None, 0.0, None

            if prob_buy >= 0.5:
                signal = "BUY"
                confidence = prob_buy
            else:
                signal = "SELL"
                confidence = 1.0 - prob_buy

            if for_open_position_check:
                return None, float(confidence), None

            proba_array = np.array([1.0 - prob_buy, prob_buy])

            return signal, float(confidence), proba_array

        except Exception as e:
            import traceback
            print(f"[{symbol}] ❌ Lỗi khi dự đoán tín hiệu: {e}\n{traceback.format_exc()}")
            return None, 0.0, None


  # <<< THAY THẾ TOÀN BỘ HÀM run_portfolio_rl_strategy TRONG LỚP EnhancedTradingBot >>>

    def _decode_actions_to_vector(self, action, symbols_in_env, model):
        """
        Giải mã hành động từ model RL thành một vector.
        0=HOLD, 1=BUY, 2=SELL.
        """
        import numpy as np
        action_space = model.action_space

        # Chuyển đổi action sang mảng numpy để xử lý
        action_array = np.asarray(action).flatten()

        # Đảm bảo vector trả về luôn có độ dài bằng số lượng symbol
        # và các giá trị nằm trong khoảng [0, 1, 2]
        valid_actions = [int(a) if a in [0, 1, 2] else 0 for a in action_array]

        # Nếu vector trả về ngắn hơn số symbol (trường hợp hiếm), điền phần còn lại bằng 0 (HOLD)
        if len(valid_actions) < len(symbols_in_env):
            valid_actions.extend([0] * (len(symbols_in_env) - len(valid_actions)))

        return valid_actions[:len(symbols_in_env)]

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM run_portfolio_rl_strategy NÀY BẰNG ĐOẠN CODE DƯỚI ĐÂY

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM run_portfolio_rl_strategy NÀY BẰNG ĐOẠN CODE DƯỚI ĐÂY

    async def run_portfolio_rl_strategy(self):
        """
        Chạy chiến lược RL trên toàn bộ portfolio.
        NÂNG CẤP: Đã chuyển sang async và thực thi các lệnh song song.
        PHIÊN BẢN SỬA LỖI: Tự động pad observation vector để khớp với shape model mong đợi.
        """
        if not self.use_rl or self.portfolio_rl_agent is None or self.portfolio_rl_agent.model is None:
            logging.warning("RL Agent chưa sẵn sàng, bỏ qua chu kỳ.")
            return

        logging.info("--- [Portfolio RL Strategy] 🤖 Agent bắt đầu phân tích toàn bộ danh mục ---")

        try:
            # Lấy danh sách symbol mà Agent đã được huấn luyện trên đó
            vec_env = self.portfolio_rl_agent.model.get_env()
            symbols_agent_knows = []
            if vec_env is not None:
                try:
                    symbols_agent_knows = vec_env.get_attr("symbols")[0]
                except Exception:
                    pass # Fallback if get_attr fails
            
            # Fallback nếu không lấy được env (trường hợp phổ biến khi load model)
            if not symbols_agent_knows:
                # Giả định agent được huấn luyện trên tất cả các symbol đang hoạt động
                # Đây là nguồn gốc của lỗi, nhưng logic padding bên dưới sẽ sửa nó
                symbols_agent_knows = list(self.active_symbols)


            if not symbols_agent_knows:
                logging.warning("[RL Strategy] Không có symbol nào để phân tích.")
                return

            # --- BƯỚC 1: XÂY DỰNG OBSERVATION (Trạng thái thị trường) ---
            all_symbol_obs, live_prices, live_confidences = [], {}, {}
            
            live_data_cache = {}
            active_symbols_to_process = self.active_symbols.intersection(symbols_agent_knows)
            for symbol in active_symbols_to_process:
                live_data_cache[symbol] = self.data_manager.create_enhanced_features(symbol)

            for symbol in symbols_agent_knows:
                market_obs = None
                model_ref_data = self.trending_models.get(symbol) or self.ranging_models.get(symbol)

                if symbol in live_data_cache and live_data_cache[symbol] is not None and model_ref_data:
                    df_features = live_data_cache[symbol]
                    ensemble_model, feature_columns = model_ref_data['ensemble'], model_ref_data['feature_columns']
                    
                    prob_buy = ensemble_model.predict_proba(df_features[feature_columns])
                    confidence = prob_buy if prob_buy >= 0.5 else 1.0 - prob_buy
                    
                    live_confidences[symbol] = confidence
                    # Quan trọng: Dùng feature columns gốc (không thêm confidence) để khớp với lúc train
                    market_obs = df_features[feature_columns].tail(1).to_numpy()[0]
                    live_prices[symbol] = df_features['close'].iloc[-1]
                
                if market_obs is None:
                    if model_ref_data:
                        num_features = len(model_ref_data['feature_columns'])
                        market_obs = np.zeros(num_features)
                    else:
                        # Fallback an toàn nếu không có model nào cho symbol này
                        num_features = ML_CONFIG["FEATURE_SELECTION_TOP_K"]
                        market_obs = np.zeros(num_features)
                
                pos_state, pnl_state, time_state = 0.0, 0.0, 0.0
                if symbol in self.open_positions:
                    pos = self.open_positions[symbol]
                    price = live_prices.get(symbol, pos['entry_price'])
                    pos_state = 1.0 if pos['signal'] == 'BUY' else -1.0
                    pnl = (price - pos['entry_price']) * pos_state
                    pnl_state = pnl / pos.get('initial_balance_at_open', 10000)
                    time_open = (datetime.now(pytz.utc) - pos['opened_at'].astimezone(pytz.utc)).total_seconds() / 3600
                    time_state = time_open / 24.0

                all_symbol_obs.append(np.append(market_obs, [pos_state, pnl_state, time_state]))

            flat_obs = np.concatenate(all_symbol_obs).astype(np.float32)
            global_states = np.array([
                (10000 / 10000) - 1.0,
                len(self.open_positions) / len(symbols_agent_knows) if symbols_agent_knows else 0,
                0.0, 0.0
            ]).astype(np.float32)
            final_live_observation = np.append(flat_obs, global_states)
            
            # --- BƯỚC 2: AGENT DỰ ĐOÁN HÀNH ĐỘNG (VỚI LOGIC SỬA LỖI PADDING) ---
            
            expected_shape = self.portfolio_rl_agent.model.observation_space.shape
            
            # So sánh và sửa lỗi shape nếu cần
            if final_live_observation.shape != expected_shape:
                logging.warning(f"Observation shape mismatch! Got {final_live_observation.shape}, expected {expected_shape}. Padding...")
                
                current_size = final_live_observation.shape[0]
                expected_size = expected_shape[0]
                
                if current_size < expected_size:
                    # Tách phần global states (luôn là 4 phần tử cuối)
                    obs_without_global = final_live_observation[:-4]
                    global_s = final_live_observation[-4:]
                    
                    # Tạo padding cho các symbol bị thiếu
                    padding_size = (expected_size - 4) - len(obs_without_global)
                    padding = np.zeros(padding_size)
                    
                    # Ghép lại observation đúng thứ tự
                    padded_obs = np.concatenate([obs_without_global, padding, global_s])
                    final_live_observation = padded_obs.astype(np.float32)
                    logging.info(f"Successfully padded observation to {final_live_observation.shape}")
                else:
                    logging.error("Cannot fix observation shape mismatch: current shape is larger than expected. Aborting prediction.")
                    return

            final_live_observation = np.nan_to_num(final_live_observation)

            action, _ = self.portfolio_rl_agent.model.predict(final_live_observation, deterministic=True)
            action_vector = self._decode_actions_to_vector(action, symbols_agent_knows, self.portfolio_rl_agent.model)
            logging.info(f"[RL Decision] Vector hành động: {list(zip(symbols_agent_knows, action_vector))}")

            # --- BƯỚC 3: THỰC THI HÀNH ĐỘNG BẤT ĐỒNG BỘ ---
            tasks = []
            for i, action_code in enumerate(action_vector):
                if i < len(symbols_agent_knows) and action_code in [1, 2]:
                    symbol_to_act = symbols_agent_knows[i]
                    
                    if symbol_to_act in self.active_symbols and symbol_to_act not in self.open_positions:
                        action_name = {1: "BUY", 2: "SELL"}[action_code]
                        confidence = live_confidences.get(symbol_to_act, 0.5)
                        tasks.append(self.handle_position_logic(symbol_to_act, action_name, confidence))

            if tasks:
                print(f"   [Async Execution] Đang thực thi đồng thời {len(tasks)} tín hiệu giao dịch...")
                await asyncio.gather(*tasks)

        except Exception as e:
            import traceback
            logging.error(f"LỖI NGHIÊM TRỌNG trong run_portfolio_rl_strategy: {e}\n{traceback.format_exc()}")
    def calculate_dynamic_position_size(self, symbol, confidence, llm_sentiment_score=0.0):
        """
        Tính toán kích thước vị thế linh hoạt, kết hợp confidence, volatility và LLM sentiment.
        """
        base_risk = RISK_MANAGEMENT["MAX_RISK_PER_TRADE"]

        # Yếu tố 1: Độ tin cậy của model Ensemble/RL
        confidence_multiplier = 1.0 + (confidence - ML_CONFIG["CONFIDENCE_THRESHOLD"])

        # Yếu tố 2: Cảm tính tin tức từ LLM
        # Điểm từ -1.0 đến 1.0 -> Chuyển thành hệ số từ 0.75 đến 1.25
        # Nếu LLM tiêu cực, giảm rủi ro. Nếu tích cực, tăng rủi ro.
        llm_multiplier = 1.0 + (llm_sentiment_score * 0.25)

        print(f"   [Risk] Confidence Multiplier: {confidence_multiplier:.2f}, LLM Multiplier: {llm_multiplier:.2f}")

        # Yếu tố 3: Độ biến động thị trường (giữ nguyên)
        try:
            primary_tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, PRIMARY_TIMEFRAME_DEFAULT)
            df_primary_tf = self.data_manager.fetch_multi_timeframe_data(
                symbol, count=RISK_MANAGEMENT["VOLATILITY_LOOKBACK"] + 5
            ).get(primary_tf)
            if df_primary_tf is not None and not df_primary_tf.empty:
                atr_indicator = AverageTrueRange(df_primary_tf["high"], df_primary_tf["low"], df_primary_tf["close"], window=RISK_MANAGEMENT["VOLATILITY_LOOKBACK"])
                atr_normalized = (atr_indicator.average_true_range() / df_primary_tf["close"]).ffill().bfill()
                recent_volatility = atr_normalized.dropna().iloc[-1] if not atr_normalized.dropna().empty else 0.01
                volatility_adjustment = 1 / (1 + recent_volatility * 5)
            else:
                volatility_adjustment = 0.8
        except Exception:
            volatility_adjustment = 0.8

        # Tính toán rủi ro cuối cùng
        adjusted_risk = base_risk * confidence_multiplier * llm_multiplier * volatility_adjustment

        # Giới hạn rủi ro trong một khoảng an toàn
        final_risk = min(adjusted_risk, RISK_MANAGEMENT["MAX_RISK_PER_TRADE"] * 1.5)
        final_risk = max(final_risk, RISK_MANAGEMENT["MAX_RISK_PER_TRADE"] * 0.25) # Giảm mức tối thiểu để linh hoạt hơn

        print(f"   [Risk] Final calculated risk for {symbol}: {final_risk*100:.2f}%")
        return final_risk

    def enhanced_risk_management(self, symbol, signal, current_price, confidence):
        try:
            df_primary_tf = self.data_manager.fetch_multi_timeframe_data(
                symbol, count=RISK_MANAGEMENT["VOLATILITY_LOOKBACK"] + 5
            ).get(PRIMARY_TIMEFRAME)
            if df_primary_tf is not None and not df_primary_tf.empty:
                atr_indicator = AverageTrueRange(
                    df_primary_tf["high"],
                    df_primary_tf["low"],
                    df_primary_tf["close"],
                    window=RISK_MANAGEMENT["VOLATILITY_LOOKBACK"],
                )
                atr_value = (
                    atr_indicator.average_true_range().dropna().iloc[-1]
                    if not atr_indicator.average_true_range().dropna().empty
                    else current_price * 0.005
                )
            else:
                atr_value = current_price * 0.005
        except Exception:
            atr_value = current_price * 0.005

        sl_atr_multiplier = RISK_MANAGEMENT.get("SL_ATR_MULTIPLIER", 1.5)

        # --- LOGIC AN TOÀN MỚI ---
        # 1. Tính khoảng cách SL theo ATR
        sl_distance_atr = atr_value * sl_atr_multiplier

        # 2. Tính khoảng cách SL tối thiểu an toàn (ví dụ: 0.3% giá hiện tại)
        min_safe_sl_distance = current_price * 0.003

        # 3. Chọn khoảng cách lớn hơn để làm SL cuối cùng
        sl_distance = max(sl_distance_atr, min_safe_sl_distance)
        # --- KẾT THÚC LOGIC AN TOÀN ---

        base_rr_ratio = RISK_MANAGEMENT.get("BASE_RR_RATIO", 1.5)
        rr_ratio = base_rr_ratio + (confidence - 0.5)
        tp_distance = sl_distance * rr_ratio

        if signal == "BUY":
            tp = current_price + tp_distance
            sl = current_price - sl_distance
        else:  # SELL
            tp = current_price - tp_distance
            sl = current_price + sl_distance

        print(
            f"    [Risk] SL distance for {symbol}: {sl_distance:.5f} (ATR-based: {sl_distance_atr:.5f}, Safe Min: {min_safe_sl_distance:.5f})"
        )

        return tp, sl

    def portfolio_risk_check(self):
        total_risk_percentage = sum(
            pos.get("risk_amount_percent", 0) for pos in self.open_positions.values()
        )
        # print(f"ℹ️ Tổng rủi ro danh mục hiện tại: {total_risk_percentage*100:.2f}%")
        return total_risk_percentage < RISK_MANAGEMENT["MAX_PORTFOLIO_RISK"]

    def _get_implied_usd_direction(self, symbol, signal):
        if not isinstance(symbol, str) or len(symbol) < 6:
            return None
        base, quote = symbol[:3].upper(), symbol[3:].upper()
        if quote == "USD":
            return "WEAK_USD" if signal == "BUY" else "STRONG_USD"
        if base == "USD":
            return "STRONG_USD" if signal == "BUY" else "WEAK_USD"
        if symbol in ["XAUUSD", "BTCUSD"]:
            return "WEAK_USD" if signal == "BUY" else "STRONG_USD"
        return None

    def correlation_check(self, new_symbol, new_signal):
        """
        Kiểm tra xung đột danh mục bằng cách phân tích mức độ tiếp xúc của từng đồng tiền.
        PHIÊN BẢN NÂNG CẤP.
        """
        # 1. Tính toán mức độ tiếp xúc của toàn bộ danh mục hiện tại
        portfolio_exposure = {}
        for position in self.open_positions.values():
            exposures = self._get_currency_exposures(
                position["symbol"], position["signal"]
            )
            for currency, direction in exposures.items():
                if currency not in portfolio_exposure:
                    portfolio_exposure[currency] = []
                portfolio_exposure[currency].append(direction)

        # 2. Lấy mức độ tiếp xúc của lệnh mới
        new_trade_exposure = self._get_currency_exposures(new_symbol, new_signal)

        # 3. Kiểm tra xung đột
        for currency, new_direction in new_trade_exposure.items():
            if currency in portfolio_exposure:
                # Xác định hướng đối nghịch
                opposite_direction = "SHORT" if new_direction == "LONG" else "LONG"

                # Nếu trong danh mục đang có một lệnh ngược chiều với lệnh mới cho cùng 1 đồng tiền -> Chặn
                if opposite_direction in portfolio_exposure[currency]:
                    print(
                        f"❌ Chặn do xung đột danh mục: Lệnh mới {new_signal} {new_symbol} ({new_direction} {currency}) "
                        f"mâu thuẫn với vị thế đang mở ({opposite_direction} {currency})."
                    )
                    return False

        # Nếu không có xung đột nào được tìm thấy
        return True


    # THAY THẾ TOÀN BỘ HÀM CŨ BẰNG HÀM MỚI NÀY
    def has_high_impact_event_soon(self, symbol):
        """
        Kiểm tra sự kiện kinh tế quan trọng sắp diễn ra.
        PHIÊN BẢN NÂNG CẤP: Hỗ trợ Chỉ số, Hàng hóa và Crypto.
        """
        try:
            # 1. Tạo một bản đồ để liên kết các tài sản với quốc gia/tiền tệ cốt lõi
            SYMBOL_CURRENCY_MAP = {
                # Indices
                "SPX500": ["USD"], "NAS100": ["USD"], "US30": ["USD"],
                "DE40":   ["EUR"], "JP225":  ["JPY"], "AU200":  ["AUD"], "HK50": ["HKD", "CNY"],
                # Commodities & Crypto
                "XAUUSD": ["USD"], "BTCUSD": ["USD"],
            }

            # 2. Xác định các đồng tiền/quốc gia cần theo dõi cho symbol hiện tại
            relevant_currencies = []
            if symbol in SYMBOL_CURRENCY_MAP:
                relevant_currencies = SYMBOL_CURRENCY_MAP[symbol]
            elif len(symbol) >= 6: # Xử lý cho các cặp Forex
                relevant_currencies.extend([symbol[:3].upper(), symbol[3:].upper()])

            if not relevant_currencies:
                return None # Không xác định được tiền tệ liên quan

            # 3. Lấy và lọc lịch kinh tế
            # <<< ĐÃ SỬA: GỌI HÀM get_economic_calendar MỚI >>>
            calendar = self.news_manager.get_economic_calendar()
            if not calendar:
                return None

            now_utc = datetime.now(pytz.utc)
            buffer_hours = TRADE_FILTERS.get("EVENT_BUFFER_HOURS", 2) # Lấy từ config

            for event in calendar:
                event_time_str = event.get("Date") # SỬA: Trading Economics dùng "Date"
                if not event_time_str: continue

                try:
                    # Chuyển đổi thời gian sự kiện sang múi giờ UTC
                    event_time = pd.to_datetime(event_time_str).tz_convert('UTC')
                except Exception:
                    try: # Thử phương pháp khác nếu có lỗi
                        event_time = pd.to_datetime(event_time_str).tz_localize('UTC')
                    except Exception:
                        continue

                # 4. Kiểm tra thời gian và tác động
                time_diff_hours = (event_time - now_utc).total_seconds() / 3600
                is_within_window = -buffer_hours < time_diff_hours < buffer_hours
                is_high_impact = event.get("Importance") == "High" or any(
                    keyword.lower() in event.get("Event", "").lower()
                    for keyword in self.high_impact_events
                )

                if is_within_window and is_high_impact:
                    # 5. Kiểm tra sự liên quan của sự kiện với symbol
                    event_currency = event.get("Currency", "").upper() # SỬA: Trading Economics dùng "Currency"
                    if event_currency in relevant_currencies:
                        print(f"⚠️  Sự kiện KTQT cho {symbol}: {event.get('Event')} lúc {event_time.strftime('%Y-%m-%d %H:%M')} UTC.")
                        return {"title": event.get("Event"), "time": event_time}

            return None # Không có sự kiện nào phù hợp

        except Exception as e:
            print(f"❌ Lỗi trong hàm has_high_impact_event_soon: {e}")
            return None

    def determine_market_regime(self, symbol, use_cache=True):
        """
        Xác định xu hướng thị trường chính (Uptrend, Downtrend, Sideways)
        dựa trên dữ liệu khung D1.
        """
        # Kiểm tra cache trước để tiết kiệm API call
        if use_cache and symbol in self.market_regime_cache:
            cached_regime, cache_time = self.market_regime_cache[symbol]
            # Nếu cache chưa quá 4 giờ thì dùng lại
            if (datetime.now() - cache_time).total_seconds() < 4 * 3600:
                return cached_regime

        try:
            # Lấy dữ liệu D1
            df_d1_data = self.data_manager.fetch_multi_timeframe_data(symbol, count=250)
            if "D1" not in df_d1_data:
                print(
                    f"[{symbol}] ⚠️ Không có dữ liệu D1 để xác định xu hướng. Mặc định là Sideways."
                )
                return "Sideways"

            df_d1 = df_d1_data["D1"]

            # Tính toán các chỉ báo cần thiết trên D1
            adx_d1 = ADXIndicator(
                df_d1["high"], df_d1["low"], df_d1["close"], window=14
            ).adx()
            ema200_d1 = EMAIndicator(df_d1["close"], window=200).ema_indicator()

            latest_close = df_d1["close"].iloc[-1]
            latest_adx = adx_d1.iloc[-1]
            latest_ema200 = ema200_d1.iloc[-1]

            regime = "Sideways"
            # Nếu ADX > 25, thị trường có xu hướng rõ ràng
            if latest_adx > 25:
                if latest_close > latest_ema200:
                    regime = "Uptrend"
                else:
                    regime = "Downtrend"

            # Lưu vào cache
            self.market_regime_cache[symbol] = (regime, datetime.now())
            return regime

        except Exception as e:
            print(
                f"[{symbol}] ❌ Lỗi khi xác định xu hướng: {e}. Mặc định là Sideways."
            )
            return "Sideways"
    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot
    def check_and_close_for_major_events(self):
        """
        Quét các sự kiện vĩ mô. Gửi báo cáo trạng thái và đóng lệnh nếu cần.
        """
        # --- BƯỚC 1: XÂY DỰNG BÁO CÁO TRẠNG THÁI ---
        status_report_parts = ["**[Pre-Check] 🔍 Bắt đầu quét các sự kiện vĩ mô sắp tới...**"]
        print(status_report_parts[0]) # In ra log console

        # Chỉ quét nếu có vị thế đang mở (để quyết định đóng lệnh) hoặc nếu bật thông báo
        should_check = self.open_positions or TRADE_FILTERS.get("SEND_PRE_CHECK_STATUS_ALERT", False)
        if not should_check:
            return

        try:
            # <<< ĐÃ SỬA: GỌI HÀM get_economic_calendar MỚI >>>
            calendar = self.news_manager.get_economic_calendar()
            if calendar:
                status_report_parts.append(f"✅ Lấy được {len(calendar)} sự kiện thô từ Trading Economics.")
                print(status_report_parts[-1])
            else:
                status_report_parts.append(f"❌ Không lấy được lịch kinh tế.")
                return

            now_utc = datetime.now(pytz.utc)
            lookahead_hours = 6
            events_to_act_on = []
            SYMBOL_CURRENCY_MAP = {
                "SPX500": ["USD"], "NAS100": ["USD"], "US30": ["USD"],
                "DE40": ["EUR"], "JP225": ["JPY"], "AU200": ["AUD"], "HK50": ["HKD", "CNY"],
                "XAUUSD": ["USD"], "BTCUSD": ["USD"], "ETHUSD": ["USD"]
            }

            for event in calendar:
                event_time_str = event.get("Date")
                if not event_time_str: continue
                try:
                    event_time = pd.to_datetime(event_time_str).tz_convert('UTC')
                except Exception: continue

                time_diff_hours = (event_time - now_utc).total_seconds() / 3600
                
                # <<< ĐÃ SỬA LỖI: ÉP KIỂU SANG STRING TRƯỚC KHI SO SÁNH >>>
                is_super_high_impact = str(event.get("Importance", "")).lower() == "high" or any(
                    keyword.lower() in str(event.get("Event", "")).lower() for keyword in self.high_impact_events
                )
                # <<< KẾT THÚC SỬA LỖI >>>

                if 0 < time_diff_hours < lookahead_hours and is_super_high_impact:
                    events_to_act_on.append({
                        "title": event.get("Event"),
                        "currency": event.get("Currency", "").upper(),
                        "time_diff_hours": time_diff_hours
                    })

            # --- BƯỚC 2: HÀNH ĐỘNG VÀ GỬI THÔNG BÁO ---

            # Kịch bản 1: Có sự kiện -> Đóng lệnh và gửi cảnh báo khẩn
            if events_to_act_on and self.open_positions:
                positions_to_close = []
                for symbol, position in self.open_positions.items():
                    for event in events_to_act_on:
                        pos_currencies = SYMBOL_CURRENCY_MAP.get(symbol, [])
                        if len(symbol) >= 6 and not pos_currencies:
                             pos_currencies = [symbol[:3].upper(), symbol[3:].upper()]
                        
                        if event["currency"] in pos_currencies:
                            positions_to_close.append({
                                "symbol": symbol,
                                "reason": f"Đóng chủ động trước sự kiện '{event['title']}' ({event['time_diff_hours']:.1f}h nữa)"
                            })
                            break

                if positions_to_close:
                    alert_message = (
                        f"🚨 **CẢNH BÁO: ĐÓNG LỆNH HÀNG LOẠT TRƯỚC TIN TỨC** 🚨\n"
                        f"Phát hiện sự kiện vĩ mô quan trọng sắp diễn ra. Tiến hành đóng các vị thế sau để bảo toàn vốn:\n"
                    )
                    for pos_to_close in positions_to_close:
                        symbol_to_close = pos_to_close["symbol"]
                        reason_to_close = pos_to_close["reason"]
                        if symbol_to_close in self.open_positions:
                            current_price = self.data_manager.get_current_price(symbol_to_close)
                            if current_price:
                                alert_message += f"- **{symbol_to_close}**: {reason_to_close}\n"
                                self.close_position_enhanced(symbol_to_close, reason_to_close, current_price)
                    self.send_discord_alert(alert_message)
                return

            # Kịch bản 2: Không có sự kiện -> Gửi báo cáo trạng thái "an toàn"
            status_report_parts.append("✅ Không có sự kiện vĩ mô nào sắp diễn ra. An toàn để giao dịch.")
            print(status_report_parts[-1])
            if TRADE_FILTERS.get("SEND_PRE_CHECK_STATUS_ALERT", False):
                self.send_discord_alert("\n".join(status_report_parts))

        except Exception as e:
            print(f"❌ Lỗi trong hàm check_and_close_for_major_events: {e}")
    def _execute_trending_strategy(self, symbol):
        """
        Thực thi chiến lược đi theo xu hướng (logic cũ của bạn).
        Bao gồm tất cả các bộ lọc đã được cải tiến.
        """
        print(
            f"[{symbol}] [Trending Strategy] 🌊 Đang phân tích theo chiến lược xu hướng..."
        )

        if symbol not in self.models:
            print(f"[{symbol}] ⚠️ Model cho {symbol} chưa sẵn sàng.")
            return

        # 1. Lấy tín hiệu từ mô hình ML
        tech_signal_raw, tech_confidence, _ = self.get_enhanced_signal(symbol)
        print(
            f"[{symbol}] [ML] 🔍 Dự đoán: {tech_signal_raw}, Xác suất: {tech_confidence:.2%}"
        )

        if (
            tech_signal_raw not in ["BUY", "SELL"]
            or tech_confidence < ML_CONFIG["MIN_CONFIDENCE_TRADE"]
        ):
            return

        df_features = self.data_manager.create_enhanced_features(symbol)
        if df_features is None or df_features.empty:
            return

        latest = df_features.iloc[-1]

        # 2. Bộ lọc xu hướng H1
        ema50_h1 = latest.get("ema50_H1")
        close_price = latest.get("close")
        if ema50_h1 and close_price:
            if tech_signal_raw == "BUY" and close_price < ema50_h1:
                print(f"[{symbol}] ❌ Lệnh BUY bị chặn! Giá M15 dưới H1 EMA50.")
                return
            if tech_signal_raw == "SELL" and close_price > ema50_h1:
                print(f"[{symbol}] ❌ Lệnh SELL bị chặn! Giá M15 trên H1 EMA50.")
                return

        # 3. Bộ lọc Price Action Score
        pa_score = self.calculate_price_action_score(df_features)
        PA_CONFIRMATION_THRESHOLD = 0.1
        if not (
            (tech_signal_raw == "BUY" and pa_score > PA_CONFIRMATION_THRESHOLD)
            or (tech_signal_raw == "SELL" and pa_score < -PA_CONFIRMATION_THRESHOLD)
        ):
            print(
                f"[{symbol}] ⚠️ Tín hiệu ML không được PA xác nhận (Score: {pa_score:.2f})."
            )
            return
        print(f"[{symbol}] ✅ Tín hiệu ML được PA xác nhận (Score: {pa_score:.2f}).")

        # 4. Bộ lọc Phân kỳ & Kiệt sức
        if tech_signal_raw == "BUY":
            # Chỉ chặn lệnh MUA nếu có tín hiệu đảo chiều GIẢM
            if latest.get("bearish_divergence") == 1 or latest.get("bb_exhaustion_sell") == 1:
                print(f"[{symbol}] ❌ Lệnh MUA bị chặn! Phát hiện tín hiệu đảo chiều/kiệt sức GIẢM.")
                return
        elif tech_signal_raw == "SELL":
            # Chỉ chặn lệnh BÁN nếu có tín hiệu đảo chiều TĂNG
            if latest.get("bullish_divergence") == 1 or latest.get("bb_exhaustion_buy") == 1:
                print(f"[{symbol}] ❌ Lệnh BÁN bị chặn! Phát hiện tín hiệu đảo chiều/kiệt sức TĂNG.")
                return

        # Nếu vượt qua tất cả, xử lý vào lệnh
        print(
            f"[{symbol}] [Trending Strategy] ✅ Tín hiệu hợp lệ: {tech_signal_raw} với Confidence {tech_confidence:.2%}"
        )
        self.handle_position_logic(symbol, tech_signal_raw, confidence=tech_confidence)

    # (Trong lớp EnhancedTradingBot, thay thế hàm này)
    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    # TÌM VÀ THAY THẾ HÀM NÀY TRONG LỚP EnhancedTradingBot

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    async def handle_position_logic(self, symbol, signal, confidence):
        """
        Hàm orchestra chính để xử lý logic mở một vị thế mới.
        Quy trình: Lấy luận điểm -> Lọc cơ bản -> Hỏi Master Agent -> Tính toán -> Mở lệnh.
        Đã được chuyển đổi sang async để tương thích với các hàm mới.
        """
        try:
            # --- BƯỚC 1: LẤY CÁC LUẬN ĐIỂM KỸ THUẬT ---
            # (Không gọi LLM ở bước này)
            tech_reasoning = await self.gather_trade_reasoning(symbol, signal, confidence)
            if "Lỗi" in tech_reasoning:
                print(f"[{symbol}] ⛔ Bỏ qua do lỗi khi thu thập luận điểm: {tech_reasoning['Lỗi']}")
                return

            # --- BƯỚC 2: CÁC BỘ LỌC CƠ BẢN (KHÔNG TỐN API) ---
            if confidence < ML_CONFIG["MIN_CONFIDENCE_TRADE"]:
                print(f"[{symbol}] ⛔ Bỏ qua: Confidence ({confidence:.2%}) quá thấp.")
                return
            if TRADE_FILTERS.get("SKIP_NEAR_HIGH_IMPACT_EVENTS", True) and self.has_high_impact_event_soon(symbol):
                # Hàm con đã in lý do
                return
            if not self.portfolio_risk_check():
                print(f"[{symbol}] ⛔ Bỏ qua: Rủi ro danh mục đã đạt ngưỡng tối đa.")
                return
            if not self.correlation_check(symbol, signal):
                # Hàm con đã in lý do
                return
            if len(self.open_positions) >= RISK_MANAGEMENT["MAX_OPEN_POSITIONS"]:
                print(f"[{symbol}] ⛔ Bỏ qua: Đã đạt số lượng vị thế mở tối đa.")
                return
            if TRADE_FILTERS.get("AVOID_WEEKEND", True) and is_weekend() and not is_crypto_symbol(symbol):
                print(f"[{symbol}] ⛔ Bỏ qua: Giao dịch cuối tuần.")
                return

            # --- BƯỚC 3: LẤY TIN TỨC VÀ HỎI Ý KIẾN MASTER AGENT (1 API CALL DUY NHẤT) ---
            news_items = await self.news_manager.get_aggregated_news(symbol)
            master_decision_data = await self.consult_master_agent(symbol, signal, tech_reasoning, news_items)

            # Nếu Master Agent từ chối, gửi thông báo và dừng lại
            if master_decision_data.get("decision", "").upper() == "REJECT":
                self.send_discord_alert(f"❌ **LỆNH BỊ TỪ CHỐI BỞI MASTER AGENT** ❌\n- **Lệnh:** `{signal} {symbol}`\n- **Lý do:** *{master_decision_data.get('justification')}*")
                return

            # --- BƯỚC 4: HOÀN THIỆN LUẬN ĐIỂM VÀ TÍNH TOÁN THAM SỐ LỆNH ---
            final_reasoning = tech_reasoning
            final_reasoning["✅ Phê duyệt Master Agent"] = master_decision_data.get('justification')
            final_reasoning["🤖 Phân tích LLM"] = f"**Điểm: {master_decision_data.get('sentiment_score', 0.0):.2f}**"

            current_price = self.data_manager.get_current_price(symbol)
            if current_price is None:
                return

            # Lấy điểm sentiment từ kết quả của Master Agent để tính toán kích thước lệnh
            llm_score = master_decision_data.get('sentiment_score', 0.0)
            position_size = self.calculate_dynamic_position_size(symbol, confidence, llm_score)
            tp, sl = self.enhanced_risk_management(symbol, signal, current_price, confidence)

            # --- BƯỚC 5: MỞ VỊ THẾ ---
            self.open_position_enhanced(symbol, signal, current_price, tp, sl, position_size, confidence, final_reasoning)

        except Exception as e:
            import traceback
            error_message = f"❌ Lỗi nghiêm trọng khi xử lý vị thế {symbol}: {e}\n{traceback.format_exc()}"
            print(error_message)
            self.send_discord_alert(error_message)

    def open_position_enhanced(
            self, symbol, signal, entry_price, tp, sl, position_size_percent, confidence,
            reasoning
        ):
            # === SỬA LỖI: Ép kiểu các giá trị sang float để đảm bảo tương thích JSON ===
            position_details = {
                "symbol": symbol,
                "signal": signal,
                "entry_price": float(entry_price),
                "tp": float(tp),
                "sl": float(sl),
                "position_size_raw_percent": float(position_size_percent),
                "confidence": float(confidence),
                "initial_confidence": float(confidence),
                "last_notified_confidence": float(confidence),
                "risk_amount_percent": float(position_size_percent),
                "opened_at": datetime.now(pytz.timezone("Asia/Bangkok")),
                "initial_sl": float(sl),
                "last_sl_notified": float(sl),
                "reasoning": reasoning
            }
            # ========================================================================

            self.open_positions[symbol] = position_details
            save_open_positions(self.open_positions)

            self.send_enhanced_alert(
                symbol, signal, entry_price, tp, sl, confidence, position_size_percent,
                reasoning
            )
            print(
                f"✅✅✅ Mở {signal} {symbol} @{entry_price:.5f} (Conf:{confidence:.2%}, Risk:{position_size_percent*100:.2f}%)"
            )
            self.update_performance_metrics()

    # TÌM VÀ THAY THẾ HÀM NÀY trong lớp EnhancedTradingBot

    def close_position_enhanced(self, symbol, reason, exit_price):
        if symbol not in self.open_positions:
            return
        position = self.open_positions.pop(symbol)
        save_open_positions(self.open_positions)
        closed_at = datetime.now(pytz.timezone("Asia/Bangkok"))
        
        # --- LOGIC TCA MỚI ---
        pip_value = self.calculate_pip_value(symbol)
        signal_price = position.get("signal_price", position["entry_price"]) # Lấy giá lúc có tín hiệu
        
        # 1. Tính Slippage: Chênh lệch giữa giá tín hiệu và giá thực thi
        slippage_pips = 0
        if pip_value != 0:
            if position["signal"] == "BUY":
                slippage_pips = (position["entry_price"] - signal_price) / pip_value
            else: # SELL
                slippage_pips = (signal_price - position["entry_price"]) / pip_value
                
        # 2. Tính chi phí Spread (ước tính)
        spread_pips = position.get("spread_at_open", 0.0) # Sẽ thêm ở bước sau
        
        # 3. Tính Pips cuối cùng (đã trừ spread)
        pips = 0
        if pip_value != 0:
            if position["signal"] == "BUY":
                pips = (exit_price - position["entry_price"]) / pip_value
            else: # SELL
                pips = (position["entry_price"] - exit_price) / pip_value
        # pips -= spread_pips # Trừ đi chi phí spread để có P/L thực
        # --- KẾT THÚC LOGIC TCA ---

        if self.conn:
            try:
                cursor = self.conn.cursor()
                # Thêm các giá trị mới vào tuple
                trade_data = (
                    symbol, position["signal"], position["entry_price"],
                    exit_price, closed_at.strftime("%Y-%m-%d %H:%M:%S"),
                    reason, pips, position.get("initial_confidence", 0.0),
                    signal_price, slippage_pips, spread_pips # Giá trị mới
                )
                # Cập nhật câu lệnh INSERT
                cursor.execute(
                    """INSERT INTO trades (symbol, signal, entry_price, exit_price, 
                                          closed_at, reason, pips, confidence,
                                          signal_price, execution_slippage_pips, spread_cost_pips) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    trade_data,
                )
                self.conn.commit()
            except Exception as e:
                print(f"❌ Lỗi ghi DB cho {symbol}: {e}")
                
        self.update_performance_metrics()
        self.send_close_alert_enhanced(symbol, position, reason, exit_price, pips)
        print(f"🔴🔴🔴 Đóng {position['signal']} {symbol} @{exit_price:.5f}. Lý do: {reason}. Pips: {pips:.1f}")

    def calculate_pip_value(self, symbol):
        symbol = symbol.upper()

        # Nhóm Chỉ số
        indices = ["SPX500", "NAS100", "US30", "JP225", "HK50", "AU200", "DE40", "HK33", "UK100"]
        for index in indices:
            if index in symbol:
                return 1.0  # Đối với chỉ số, 1 điểm = 1.0

        # Nhóm Hàng hóa
        if "XAU" in symbol or "XAG" in symbol: # Vàng, Bạc
            return 0.01
        if "WTICO_USD" in symbol or "BCO_USD" in symbol: # Dầu
            return 0.01

        # Nhóm Crypto
        if "BTC" in symbol or "ETH" in symbol:
            return 0.01

        # Nhóm Forex
        if "JPY" in symbol:
            return 0.01

        # Mặc định cho các cặp Forex khác
        return 0.0001


    # (Trong lớp EnhancedTradingBot, thay thế hàm này)
    def send_enhanced_alert(self, symbol, signal, entry_price, tp, sl, confidence, position_size_percent, reasoning):
        strategy_type = "[RL]" if self.use_rl else "[Ensemble]"

        reasoning_text = "🧠 **Luận điểm vào lệnh:**\n"
        for key, value in reasoning.items():
            if isinstance(value, list) and not value: value = "Không có"
            if value:
                reasoning_text += f"- {key}: {value}\n" # Bỏ ** để LLM tự định dạng

        message = (
            f"🚨 **VỊ THẾ MỚI {strategy_type}: {signal} {symbol}** 🚨\n"
            f"-------------------------------------\n"
            f"{reasoning_text}"
            f"-------------------------------------\n"
            f"- Giá vào lệnh: `{entry_price:.5f}`\n"
            f"- Take Profit: `{tp:.5f}`\n"
            f"- Stop Loss: `{sl:.5f}`\n"
            f"- Rủi ro lệnh: `{position_size_percent*100:.2f}%` tài khoản\n"
            f"-------------------------------------\n"
            f"Win Rate: {self.performance_metrics.get('win_rate', 0):.2f}% | Tổng P&L: {self.performance_metrics.get('total_pips', 0):.1f} pips"
        )
        self.send_discord_alert(message)

    def send_close_alert_enhanced(self, symbol, position, reason, exit_price, pips):
        profit_emoji = "🟢" if pips >= 0 else "🔴"
        duration = datetime.now(pytz.timezone("Asia/Bangkok")) - position["opened_at"]
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m"
        message = (
            f"{profit_emoji} **ĐÓNG VỊ THẾ: {position.get('signal','')} {symbol}** {profit_emoji}\n"
            f"- Lý do đóng: {reason}\n"
            f"- Giá đóng: {exit_price:.5f}\n"
            f"- Giá mở: {position.get('entry_price',0):.5f}\n"
            f"- Kết quả: {pips:.1f} pips\n"
            f"- Thời gian giữ lệnh: {duration_str}\n"
            f"- ML Confidence ban đầu: {position.get('initial_confidence', 0.0):.2%}\n"
            f"-------------------------------------\n"
            f"Win Rate: {self.performance_metrics.get('win_rate', 0):.2f}% | Tổng P&L: {self.performance_metrics.get('total_pips', 0):.1f} pips"
        )
        self.send_discord_alert(message)

    def send_discord_alert(self, message):
        try:
            max_len = 1900
            if len(message) > max_len:
                parts = [
                    message[i : i + max_len] for i in range(0, len(message), max_len)
                ]
                for part in parts:
                    requests.post(
                        DISCORD_WEBHOOK, json={"content": part}, timeout=10
                    ).raise_for_status()
            else:
                requests.post(
                    DISCORD_WEBHOOK, json={"content": message}, timeout=10
                ).raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Lỗi Discord: {str(e)}")

    def check_existing_positions(self):
        symbols_to_process = list(self.open_positions.keys())
        for symbol in symbols_to_process:
            if symbol not in self.open_positions:
                continue
            try:
                position = self.open_positions[symbol]

                # <<< CẢI TIẾN 1: LẤY DỮ LIỆU GẦN ĐÂY ĐỂ TÍNH EMA >>>
                # Lấy dữ liệu gần đây đủ để tính EMA ngắn hạn
                primary_tf = PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, PRIMARY_TIMEFRAME_DEFAULT)
                recent_data = self.data_manager.fetch_multi_timeframe_data(symbol, count=50).get(primary_tf)

                if recent_data is None or recent_data.empty:
                    print(f"[{symbol}] ⚠️ Không có dữ liệu gần đây để kiểm tra vị thế, bỏ qua.")
                    continue

                current_price = recent_data['close'].iloc[-1]

                # 1. Kiểm tra chạm TP/SL trước tiên (Logic cũ giữ nguyên)
                closed = False
                if position["signal"] == "BUY":
                    if current_price >= position["tp"]:
                        self.close_position_enhanced(symbol, "Chạm Take Profit", current_price)
                        closed = True
                    elif current_price <= position["sl"]:
                        self.close_position_enhanced(symbol, "Chạm Stop Loss", current_price)
                        closed = True
                else:  # SELL
                    if current_price <= position["tp"]:
                        self.close_position_enhanced(symbol, "Chạm Take Profit", current_price)
                        closed = True
                    elif current_price >= position["sl"]:
                        self.close_position_enhanced(symbol, "Chạm Stop Loss", current_price)
                        closed = True
                if closed:
                    continue

                # 2. KIỂM TRA "THỜI GIAN ÂN HẠN" (Grace Period) (Logic cũ giữ nguyên)
                grace_period_candles = ML_CONFIG.get("CONFIDENCE_CHECK_GRACE_PERIOD_CANDLES", 0)
                time_since_open = (datetime.now(pytz.timezone("Asia/Bangkok")) - position["opened_at"])

                minutes_per_candle = 60 # Default H1
                tf_string = primary_tf.upper()
                if "M" in tf_string: minutes_per_candle = int(tf_string.replace("M", ""))
                elif "H" in tf_string: minutes_per_candle = int(tf_string.replace("H", "")) * 60
                elif "D" in tf_string: minutes_per_candle = 1440
                grace_period_minutes = grace_period_candles * minutes_per_candle
                is_in_grace_period = (time_since_open.total_seconds() / 60) < grace_period_minutes

                # 3. KIỂM TRA ĐÓNG LỆNH (Chỉ khi đã qua thời gian ân hạn)
                if not is_in_grace_period:
                    _, current_confidence, _ = self.get_enhanced_signal(symbol, for_open_position_check=True)
                    close_threshold = ML_CONFIG.get("CLOSE_ON_CONFIDENCE_DROP_THRESHOLD", 0)

                    if current_confidence > 0 and close_threshold > 0 and current_confidence < close_threshold:
                        # <<< CẢI TIẾN 2: THÊM ĐIỀU KIỆN XÁC NHẬN KỸ THUẬT >>>
                        ema_period = ML_CONFIG.get("EMA_SHORT_TERM_PERIOD_FOR_CLOSE", 20)
                        latest_ema_short = EMAIndicator(recent_data["close"], window=ema_period).ema_indicator().iloc[-1]

                        technical_confirmation = False
                        if position["signal"] == "BUY" and current_price < latest_ema_short:
                            technical_confirmation = True
                            print(f"[{symbol}] [CloseConfirm] ✅ Giá hiện tại ({current_price:.5f}) đã nằm dưới EMA{ema_period} ({latest_ema_short:.5f}).")
                        elif position["signal"] == "SELL" and current_price > latest_ema_short:
                            technical_confirmation = True
                            print(f"[{symbol}] [CloseConfirm] ✅ Giá hiện tại ({current_price:.5f}) đã nằm trên EMA{ema_period} ({latest_ema_short:.5f}).")

                        # <<< CẢI TIẾN 3: LOGIC ĐÓNG LỆNH KẾT HỢP >>>
                        if technical_confirmation:
                            is_in_profit = (position["signal"] == "BUY" and current_price > position["entry_price"]) or \
                                           (position["signal"] == "SELL" and current_price < position["entry_price"])

                            reason = f"ML Conf giảm ({current_confidence:.2%}) VÀ giá phá vỡ EMA{ema_period}"
                            reason += " -> Chốt lời sớm" if is_in_profit else " -> Cắt lệnh sớm"

                            print(f"[{symbol}] 🔴 Tự động đóng lệnh do: {reason}")
                            self.close_position_enhanced(symbol, reason, current_price)
                            continue
                        else:
                            print(f"[{symbol}] ℹ️ Confidence giảm ({current_confidence:.2%}) nhưng giá vẫn giữ cấu trúc. Tạm thời giữ lệnh.")

                # 4. CẬP NHẬT TRAILING STOP (Logic cũ giữ nguyên)
                self.update_trailing_stop(symbol, current_price)

            except Exception as e:
                import traceback
                print(f"❌ Lỗi kiểm tra vị thế {symbol}: {e}\n{traceback.format_exc()}")
                pass
    def close_all_non_crypto_positions_for_weekend(self):
            """
            Lặp qua tất cả các vị thế đang mở và đóng những vị thế không phải là crypto.
            """
            positions_to_close = []
            for symbol, position in self.open_positions.items():
                if not is_crypto_symbol(symbol):
                    positions_to_close.append(symbol)

            if not positions_to_close:
                print("[Weekend Close] Không có lệnh nào cần đóng vào cuối tuần.")
                return

            alert_message = "🔴 **[THÔNG BÁO] ĐÓNG LỆNH CUỐI TUẦN** 🔴\n"
            alert_message += "Đang đóng các vị thế sau để tránh rủi ro cuối tuần:\n"

            print(f"[Weekend Close] Đang đóng {len(positions_to_close)} vị thế...")

            closed_count = 0
            for symbol in positions_to_close:
                current_price = self.data_manager.get_current_price(symbol)
                if current_price:
                    reason = "Đóng lệnh tự động cuối tuần"
                    signal = self.open_positions[symbol]['signal']
                    alert_message += f"- Đã đóng {signal} **{symbol}** @ `{current_price:.5f}`\n"
                    self.close_position_enhanced(symbol, reason, current_price)
                    closed_count += 1

            if closed_count > 0:
                self.send_discord_alert(alert_message)

    def check_and_execute_weekend_close(self):
            """
            Kiểm tra xem có phải thời điểm đóng lệnh cuối tuần không và thực thi nếu cần.
            """
            if not WEEKEND_CLOSE_CONFIG["ENABLED"]:
                return

            now_utc = datetime.utcnow()

            # Reset cờ vào ngày Thứ 2
            if now_utc.weekday() == 0 and self.weekend_close_executed:
                print("[Weekend Close] Reset cờ đóng lệnh cho tuần mới.")
                self.weekend_close_executed = False

            # Kiểm tra điều kiện đóng lệnh
            is_close_day = (now_utc.weekday() == WEEKEND_CLOSE_CONFIG["CLOSE_DAY_UTC"])
            is_past_close_time = (now_utc.hour >= WEEKEND_CLOSE_CONFIG["CLOSE_HOUR_UTC"] and
                                  now_utc.minute >= WEEKEND_CLOSE_CONFIG["CLOSE_MINUTE_UTC"])

            if is_close_day and is_past_close_time and not self.weekend_close_executed:
                print("[Weekend Close] Đã đến thời điểm đóng lệnh cuối tuần!")
                self.close_all_non_crypto_positions_for_weekend()
                self.weekend_close_executed = True # Đánh dấu đã thực hiện để không lặp lại
    def update_trailing_stop(self, symbol, current_price):
        if RISK_MANAGEMENT.get("TRAILING_STOP_MULTIPLIER", 0) <= 0:
            return
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        initial_sl_value = position.get("initial_sl", position["sl"])
        trail_distance_price = 0

        # --- LOGIC MỚI: TÍNH TRAILING STOP DỰA TRÊN ATR ---
        try:
            # Lấy dữ liệu gần đây để tính ATR (50 nến là đủ)
            df_recent = self.data_manager.fetch_multi_timeframe_data(
                symbol, count=50
            ).get(PRIMARY_TIMEFRAME)
            if df_recent is None or df_recent.empty:
                raise ValueError("Không có dữ liệu để tính ATR")

            # Tính ATR trên 14 nến gần nhất (thông số tiêu chuẩn)
            atr_indicator = AverageTrueRange(
                df_recent["high"], df_recent["low"], df_recent["close"], window=14
            )
            current_atr = atr_indicator.average_true_range().dropna().iloc[-1]

            # Lấy hệ số nhân từ cấu hình RISK_MANAGEMENT
            atr_multiplier = RISK_MANAGEMENT.get(
                "TRAILING_STOP_MULTIPLIER", 3.0
            )  # Mặc định là 3.0 nếu không có

            trail_distance_price = current_atr * atr_multiplier
            # print(f"ℹ️ ATR-based trail distance for {symbol}: {trail_distance_price:.5f}") # Bỏ comment để debug

        except Exception as e:
            # Fallback: Nếu không tính được ATR, dùng lại logic cũ với pip cố định
            print(
                f"⚠️ Không thể tính ATR cho {symbol}, dùng giá trị pip cố định. Lỗi: {e}"
            )
            base_trail_pips = 35  # Giá trị dự phòng
            pip_value = self.calculate_pip_value(symbol)
            trail_distance_price = base_trail_pips * pip_value
        # --- KẾT THÚC LOGIC MỚI ---

        if trail_distance_price == 0:
            return  # Không làm gì nếu khoảng cách bằng 0

        new_sl = position["sl"]
        old_sl_for_notification = position.get("last_sl_notified", initial_sl_value)

        if position["signal"] == "BUY":
            potential_new_sl = current_price - trail_distance_price
            if (
                potential_new_sl > initial_sl_value
                and potential_new_sl > position["sl"]
            ):
                new_sl = potential_new_sl
        else:  # SELL
            potential_new_sl = current_price + trail_distance_price
            if (
                potential_new_sl < initial_sl_value
                and potential_new_sl < position["sl"]
            ):
                new_sl = potential_new_sl

        if new_sl != position["sl"]:
            pip_value = self.calculate_pip_value(
                symbol
            )  # Tính lại pip_value để thông báo
            sl_change_pips = (
                abs(new_sl - old_sl_for_notification) / pip_value if pip_value else 0
            )

            if sl_change_pips >= self.SL_CHANGE_NOTIFICATION_THRESHOLD_PIPS:
                alert_message = (
                    f"⚙️ **TRAILING STOP UPDATED cho {position['signal']} {symbol}** ⚙️\n"
                    f"- SL cũ: {old_sl_for_notification:.5f}\n"
                    f"- SL mới: {new_sl:.5f}\n"
                    f"- Giá hiện tại: {current_price:.5f}"
                )
                self.send_discord_alert(alert_message)
                position["last_sl_notified"] = new_sl

            position["sl"] = new_sl
            save_open_positions(self.open_positions)
            # print(f"ℹ️ Trailing Stop cho {symbol}: SL {old_sl_for_notification:.5f} -> {new_sl:.5f}")

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM run_enhanced_bot BẰNG ĐOẠN CODE NÀY

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM run_enhanced_bot BẰNG ĐOẠN CODE DƯỚI ĐÂY

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM run_enhanced_bot BẰNG ĐOẠN CODE NÀY

    async def run_enhanced_bot(self):
        print("🚀 Bot Giao Dịch Nâng Cao Đã Khởi Động!")
        if not self.check_api_connection():
            print("❌ Không thể kết nối OANDA API. Vui lòng kiểm tra lại.")
            return

        is_first_run = True
        self.send_discord_alert("✅ Bot Nâng Cao (LLMs + RL) đã khởi động và sẵn sàng hoạt động!")

        while True:
            try:
                # Đồng bộ chu kỳ chính xác vào đầu mỗi giờ
                if not is_first_run:
                    await self.wait_until_top_of_the_hour() # <<< THAY ĐỔI CỐT LÕI NẰM Ở ĐÂY
                else:
                    print("🚀 Performing initial analysis run immediately...")
                    is_first_run = False

                current_time_vn = datetime.now(pytz.timezone("Asia/Bangkok"))
                print(f"\n----- ⏰ Chu kỳ Phân Tích: {current_time_vn.strftime('%Y-%m-%d %H:%M:%S')} (VN) -----")

                # 1. Quản lý trạng thái bot và model
                self.load_or_train_models()

                # (Phần xử lý Circuit Breaker giữ nguyên)
                if not self.active_symbols:
                    self.consecutive_data_failures += 1
                    if self.consecutive_data_failures >= 3:
                        # ... (logic ngắt mạch) ...
                        await asyncio.sleep(3600)
                        continue
                else:
                    self.consecutive_data_failures = 0

                # 2. Quản lý các vị thế đang mở
                self.check_existing_positions()
                self.check_and_execute_weekend_close()
                self.check_and_close_for_major_events()

                # 3. Thực thi chiến lược giao dịch
                if self.active_symbols:
                    print(f"   [Strategy] Có {len(self.active_symbols)} symbol đang hoạt động, bắt đầu phân tích...")
                    await self.run_portfolio_rl_strategy()
                else:
                    print("   [Safe Mode] ⚠️ Không có symbol nào hoạt động. Chỉ quản lý các lệnh đang mở.")

                print("\n----- 📊 Trạng Thái Danh Mục -----")
                for symbol, pos in self.open_positions.items():
                    print(f"   - {symbol}: {pos['signal']} @ {pos['entry_price']:.5f}")
                print(f"   - Tổng vị thế mở: {len(self.open_positions)}/{RISK_MANAGEMENT['MAX_OPEN_POSITIONS']}")

            except KeyboardInterrupt:
                print("\n🛑 Bot dừng thủ công.")
                self.send_discord_alert("⚠️ Bot đã bị dừng thủ công!")
                if self.conn: self.conn.close()
                break
            except Exception as e:
                import traceback
                error_message = f"❌ LỖI NGHIÊM TRỌNG TRONG VÒNG LẶP CHÍNH: {e}\n{traceback.format_exc()}"
                print(error_message)
                self.send_discord_alert(error_message[:1900])
                print("--- Bot sẽ thử lại sau 60 giây ---")
                await asyncio.sleep(60)

    def calculate_price_action_score(self, df_features):
        """
        Tính toán một điểm số tổng hợp DỰA TRÊN NHIỀU YẾU TỐ Price Action.
        Trả về điểm từ -1.0 (cực kỳ giảm) đến +1.0 (cực kỳ tăng).
        """
        if df_features.empty:
            return 0.0

        latest = df_features.iloc[-1]
        score = 0.0
        weights = {
            "sd_zone": 0.35,
            "engulfing_pattern": 0.25,
            "trend_structure": 0.20,
            "volume_confirmation": 0.10,
            "rsi_convergence": 0.10,
        }

        # 1. Điểm từ Vùng Cung/Cầu (quan trọng nhất)
        sd_score = 0.0
        if latest.get("in_demand_reversal"):
            sd_score += 1.0
        if latest.get("in_demand_continuation"):
            sd_score += 0.5
        if latest.get("in_supply_reversal"):
            sd_score -= 1.0
        if latest.get("in_supply_continuation"):
            sd_score -= 0.5
        score += sd_score * weights["sd_zone"]

        # 2. Điểm từ Mẫu hình Nến Nhấn Chìm (Engulfing)
        engulfing_score = 0.0
        if latest.get("bullish_engulfing"):
            engulfing_score += 1.0
        if latest.get("bearish_engulfing"):
            engulfing_score -= 1.0
        score += engulfing_score * weights["engulfing_pattern"]

        # 3. Điểm từ Cấu trúc thị trường (Higher Highs / Lower Lows)
        trend_strength = latest.get("trend_strength", 0)  # Từ -5 đến 5
        score += (trend_strength / 5.0) * weights["trend_structure"]

        # 4. Xác nhận bởi Volume
        volume_ratio = latest.get("volume_ratio", 1.0)
        if (engulfing_score > 0 or sd_score > 0) and volume_ratio > 1.5:
            score += 1.0 * weights["volume_confirmation"]
        if (engulfing_score < 0 or sd_score < 0) and volume_ratio > 1.5:
            score -= 1.0 * weights["volume_confirmation"]

        # 5. Hội tụ với RSI
        rsi = latest.get(f"rsi_{PRIMARY_TIMEFRAME}", 50)
        # Nếu có tín hiệu mua và RSI vừa thoát khỏi vùng quá bán
        if (sd_score > 0 or engulfing_score > 0) and (20 < rsi < 40):
            score += 1.0 * weights["rsi_convergence"]
        # Nếu có tín hiệu bán và RSI vừa thoát khỏi vùng quá mua
        if (sd_score < 0 or engulfing_score < 0) and (60 < rsi < 80):
            score -= 1.0 * weights["rsi_convergence"]

        return max(-1.0, min(1.0, score))

    def _get_currency_exposures(self, symbol, signal):
        """
        Phân tích một giao dịch thành mức độ tiếp xúc với từng đồng tiền.
        Ví dụ: SELL EURUSD -> {'EUR': 'SHORT', 'USD': 'LONG'}
        """
        if not isinstance(symbol, str) or len(symbol) < 6:
            return {}

        base, quote = symbol[:3].upper(), symbol[3:].upper()

        if signal.upper() == "BUY":
            return {base: "LONG", quote: "SHORT"}
        elif signal.upper() == "SELL":
            return {base: "SHORT", quote: "LONG"}
        return {}

    def _execute_sideways_strategy(self, symbol):
        """
        Thực thi chiến lược giao dịch trong biên độ (sideway).
        Logic: Mua gần hỗ trợ, bán gần kháng cự.
        """
        print(
            f"[{symbol}] [Sideway Strategy] 🕵️ Đang phân tích theo chiến lược đi ngang..."
        )

        df_features = self.data_manager.create_enhanced_features(symbol)
        if df_features is None or df_features.empty:
            return

        latest = df_features.iloc[-1]
        signal = None
        confidence = 0.0

        # Điều kiện MUA: Gần hỗ trợ và RSI vừa đi lên từ vùng quá bán
        is_near_support = latest.get("near_support", False)
        rsi_m15 = latest.get(f"rsi_{PRIMARY_TIMEFRAME}", 50)

        if is_near_support and 30 < rsi_m15 < 45:
            signal = "BUY"
            # Confidence có thể dựa vào việc giá gần hỗ trợ đến mức nào
            confidence = 0.75 + (0.10 * (1 - latest.get("close_position_20", 1)))

        # Điều kiện BÁN: Gần kháng cự và RSI vừa đi xuống từ vùng quá mua
        is_near_resistance = latest.get("near_resistance", False)
        if is_near_resistance and 55 < rsi_m15 < 70:
            signal = "SELL"
            confidence = 0.75 + (0.10 * latest.get("close_position_20", 0))

        if signal and confidence >= ML_CONFIG["MIN_CONFIDENCE_TRADE"]:
            print(
                f"[{symbol}] [Sideway Strategy] ✅ Tín hiệu hợp lệ: {signal} với Confidence {confidence:.2%}"
            )
            self.handle_position_logic(symbol, signal, confidence)
        else:
            print(f"[{symbol}] [Sideway Strategy] ℹ️ Không có tín hiệu hợp lệ.")
    # TÌM VÀ THAY THẾ HÀM NÀY TRONG LỚP EnhancedTradingBot

    # TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY TRONG LỚP EnhancedTradingBot

    async def gather_trade_reasoning(self, symbol, signal, confidence):
        """
        NÂNG CẤP: Lấy luận điểm giao dịch bất đồng bộ và sử dụng đúng hàm lấy tin tức.
        """
        try:
            # Lấy dữ liệu features (hàm này không cần async)
            df_features = self.data_manager.create_enhanced_features(symbol)
            if df_features is None or df_features.empty:
                return {"Lỗi": "Không thể lấy dữ liệu features."}

            latest = df_features.iloc[-1]
            current_regime = latest.get('market_regime', 0)
            
            # Lấy ảnh hưởng của feature từ model phù hợp
            feature_influence_text = "Không xác định"
            model_data = self.trending_models.get(symbol) if current_regime != 0 else self.ranging_models.get(symbol)
            if model_data:
                ensemble_model = model_data.get('ensemble')
                if ensemble_model and hasattr(ensemble_model, 'get_base_model_feature_influence'):
                    feature_influence_text = ensemble_model.get_base_model_feature_influence()

            # Lấy tóm tắt xu hướng từ các TF cao hơn
            trend_summary_parts = []
            for htf in ["H4", "D1"]:
                if f"trend_{htf}" in latest:
                    trend = "Tăng" if latest[f'trend_{htf}'] == 1 else "Giảm"
                    trend_summary_parts.append(f"{htf}: {trend}")
            trend_summary = ", ".join(trend_summary_parts) if trend_summary_parts else "Không xác định"

            # *** PHẦN SỬA LỖI CHÍNH ***
            # Gọi hàm get_aggregated_news bất đồng bộ để lấy tin tức
            news_items = await self.news_manager.get_aggregated_news(symbol)
            
            reasoning = {
                "Tín hiệu Chính": f"{signal} với Confidence {confidence:.2%}",
                "📊 Yếu tố Kỹ thuật Chính": feature_influence_text,
                "📈 Phân tích Xu hướng": trend_summary,
                "📰 Tin tức liên quan": f"Tìm thấy {len(news_items)} tin mới nhất."
            }
            return reasoning
            
        except Exception as e:
            import traceback
            print(f"Lỗi trong gather_trade_reasoning: {e}\n{traceback.format_exc()}")
            return {"Lỗi": str(e)}
  # (Thêm hàm này vào trong lớp EnhancedTradingBot)
    # TÌM VÀ THAY THẾ HÀM NÀY TRONG LỚP EnhancedTradingBot

    async def consult_master_agent(self, symbol, signal, reasoning_data, news_items):
        """
        NÂNG CẤP: Gộp 2 nhiệm vụ (phân tích sentiment + phê duyệt) vào 1 API call.
        Đã chuyển sang async.
        """
        if not self.news_manager.llm_analyzer or not self.news_manager.llm_analyzer.model:
            return {"decision": "APPROVE", "justification": "LLM not operational, auto-approve.", "sentiment_score": 0.0}

        # (Phần prompt giữ nguyên)
        formatted_news = "\n".join([f"- {item.get('title', '')}" for item in news_items[:5]])
        if not formatted_news: formatted_news = "Không có tin tức đáng chú ý."
        reasoning_text = "\n".join([f"- {key}: {value}" for key, value in reasoning_data.items()])
        prompt = f"""
        Bạn là Giám đốc Quản lý Rủi ro (CRO) của một quỹ đầu tư.
        Một hệ thống AI cấp dưới đề xuất một lệnh: {signal} {symbol}.
        NHIỆM VỤ 1: PHÂN TÍCH TIN TỨC. Tin tức liên quan:
        ---
        {formatted_news}
        ---
        NHIỆM VỤ 2: XEM XÉT LUẬN ĐIỂM KỸ THUẬT
        {reasoning_text}

        NHIỆM VỤ 3: RA QUYẾT ĐỊNH CUỐI CÙNG.
        Dựa trên cả hai phân tích, hãy đưa ra quyết định cuối cùng.
        Chỉ trả về duy nhất một khối JSON với định dạng sau:
        {{
          "sentiment_score": <số float từ -1.0 đến 1.0 dựa trên tin tức>,
          "decision": "APPROVE" hoặc "REJECT",
          "justification": "<Lý do ngắn gọn cho quyết định của bạn>"
        }}
        """

        print("   [MasterAgent] 🧠 Đang gửi yêu cầu gộp tới Master Agent...")
        for attempt in range(3):
            try:
                # Sử dụng generate_content_async cho hoạt động bất đồng bộ
                response = await self.news_manager.llm_analyzer.model.generate_content_async(prompt)
                json_text = response.text.strip().replace('```json', '').replace('```', '')
                decision_data = json.loads(json_text)
                print(f"   [MasterAgent] ✅ Quyết định: {decision_data.get('decision')}. Score: {decision_data.get('sentiment_score', 0):.2f}")
                return decision_data
            except Exception as e:
                print(f"   [MasterAgent] ⚠️ Lỗi (Lần {attempt+1}/3): {e}. Thử lại sau 2 giây...")
                await asyncio.sleep(2)

        print("   [MasterAgent] ❌ Không thể nhận được phản hồi hợp lệ từ LLM. Tự động phê duyệt.")
        return {"decision": "APPROVE", "justification": "LLM processing error, auto-approve.", "sentiment_score": 0.0}
    # THÊM HÀM MỚI NÀY VÀO BÊN TRONG LỚP EnhancedTradingBot

    async def wait_until_top_of_the_hour(self):
        """
        Tính toán và chờ cho đến đúng XX:00:00 của giờ tiếp theo.
        """
        # Sử dụng múi giờ Việt Nam để log cho dễ hiểu
        now = datetime.now(pytz.timezone("Asia/Bangkok"))
        
        # Tính toán thời điểm đầu giờ tiếp theo
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        
        wait_seconds = (next_hour - now).total_seconds()

        # Đảm bảo luôn chờ một khoảng thời gian dương
        if wait_seconds <= 0:
            wait_seconds += 3600

        print(f"✅ Đồng bộ chu kỳ. Chờ {int(wait_seconds // 60)} phút {int(wait_seconds % 60)} giây cho đến {next_hour.strftime('%H:%M:%S')}...")
        await asyncio.sleep(wait_seconds)
class DriftMonitor:
    """
    Theo dõi sự thay đổi (drift) trong phân phối dữ liệu giữa tập huấn luyện
    và dữ liệu live bằng cách sử dụng Population Stability Index (PSI).
    """
    def __init__(self, X_train_reference):
        """
        Khởi tạo với DataFrame dùng làm tham chiếu (dữ liệu đã huấn luyện).
        """
        self.reference_data = X_train_reference
        self.reference_bins = {}
        self.top_features = X_train_reference.columns.tolist()

        # Tạo các "bin" (ngưỡng phân chia) cho từng feature dựa trên dữ liệu tham chiếu
        for col in self.top_features:
            # Sử dụng quantiles để chia dữ liệu thành 10 phần bằng nhau
            self.reference_bins[col] = pd.qcut(self.reference_data[col], 10, labels=False, duplicates='drop')

    def calculate_psi(self, feature_name, live_data_column):
        """
        Tính toán chỉ số PSI cho một feature cụ thể.
        """
        # Lấy các bin đã được tính toán trước từ dữ liệu tham chiếu
        reference_dist = self.reference_bins[feature_name].value_counts(normalize=True).sort_index()

        # Áp dụng các bin này lên dữ liệu live
        live_bins = pd.cut(live_data_column, bins=pd.qcut(self.reference_data[feature_name], 10, retbins=True, duplicates='drop')[1], labels=False)
        live_dist = live_bins.value_counts(normalize=True).sort_index()

        # Kết hợp hai phân phối để đảm bảo chúng có cùng các bin
        psi_df = pd.DataFrame({'reference': reference_dist, 'live': live_dist}).fillna(0.0001) # Điền 0.0001 để tránh lỗi chia cho 0

        # Tính toán PSI
        psi_df['psi'] = (psi_df['live'] - psi_df['reference']) * np.log(psi_df['live'] / psi_df['reference'])

        return psi_df['psi'].sum()

    def check_drift(self, X_live, psi_threshold=0.25):
        """
        Kiểm tra drift trên toàn bộ các feature quan trọng.
        Trả về danh sách các feature đã bị "drift".
        """
        drifted_features = {}
        if len(X_live) < 20: # Cần đủ dữ liệu live để so sánh
            return None

        for feature in self.top_features:
            if feature in X_live.columns:
                psi_score = self.calculate_psi(feature, X_live[feature])

                # Quy tắc chung:
                # PSI < 0.1: Không có thay đổi đáng kể
                # 0.1 <= PSI < 0.25: Thay đổi nhỏ
                # PSI >= 0.25: Thay đổi lớn, cần xem xét lại model
                if psi_score >= psi_threshold:
                    drifted_features[feature] = psi_score

        return drifted_features
# <<< THÊM LỚP MỚI NÀY VÀO FILE BOT >>>

class AdvancedRiskManager:
    """Enhanced risk management with asset class specific configurations"""
    
    def __init__(self):
        self.position_sizes = {}
        self.correlations = {}
        self.risk_budgets = {}
        self.var_history = deque(maxlen=252)  # 1 year of daily VaR
        self.correlation_matrix = {}
        self.sector_exposure = {}
        self.observability = AdvancedObservability()
        self.asset_class_exposure = {}
        self.symbol_risk_configs = self._initialize_symbol_risk_configs()
        
    def _initialize_symbol_risk_configs(self):
        """Initialize risk configurations for each symbol"""
        configs = {}
        for symbol in SYMBOLS:
            metadata = SYMBOL_METADATA.get(symbol, {})
            asset_class = metadata.get("asset_class", "equity_index")
            base_config = RISK_CONFIG_BY_ASSET_CLASS.get(asset_class, RISK_CONFIG_BY_ASSET_CLASS["equity_index"])
            
            # Symbol-specific adjustments
            symbol_config = base_config.copy()
            
            # Adjust for volatility profile
            volatility_profile = metadata.get("volatility_profile", "medium")
            if volatility_profile == "very_high":
                symbol_config["max_position_size"] *= 0.7
                symbol_config["var_multiplier"] *= 1.3
                symbol_config["stop_loss_atr"] *= 1.2
            elif volatility_profile == "high":
                symbol_config["max_position_size"] *= 0.8
                symbol_config["var_multiplier"] *= 1.1
            elif volatility_profile == "low":
                symbol_config["max_position_size"] *= 1.2
                symbol_config["var_multiplier"] *= 0.9
                symbol_config["stop_loss_atr"] *= 0.8
            
            # Adjust for region/session
            region = metadata.get("region", "US")
            if region in ["Asia", "Europe"]:
                symbol_config["session_risk_adjustment"] = True
            
            configs[symbol] = symbol_config
        
        return configs
    
    def _get_symbol_risk_config(self, symbol):
        """Get risk configuration for specific symbol"""
        return self.symbol_risk_configs.get(symbol, RISK_CONFIG_BY_ASSET_CLASS["equity_index"])
    
    def calculate_asset_class_exposure(self, positions):
        """Calculate exposure by asset class"""
        exposure = {}
        for position in positions:
            symbol = position['symbol']
            metadata = SYMBOL_METADATA.get(symbol, {})
            asset_class = metadata.get("asset_class", "equity_index")
            
            if asset_class not in exposure:
                exposure[asset_class] = 0
            exposure[asset_class] += position.get('weight', 0)
        
        self.asset_class_exposure = exposure
        return exposure
    
    def check_asset_class_limits(self, symbol, existing_positions):
        """Check if adding position would violate asset class limits"""
        metadata = SYMBOL_METADATA.get(symbol, {})
        asset_class = metadata.get("asset_class", "equity_index")
        
        # Calculate current exposure
        current_exposure = self.calculate_asset_class_exposure(existing_positions)
        current_asset_exposure = current_exposure.get(asset_class, 0)
        
        # Get limits
        max_asset_class_exposure = PORTFOLIO_RISK_LIMITS["max_asset_class_exposure"]
        
        violations = []
        if current_asset_exposure >= max_asset_class_exposure:
            violations.append({
                'type': 'asset_class_limit',
                'asset_class': asset_class,
                'current_exposure': current_asset_exposure,
                'limit': max_asset_class_exposure
            })
        
        return len(violations) == 0, violations
    
    def calculate_symbol_specific_var(self, symbol, position_size, returns_history):
        """Calculate VaR specific to symbol characteristics"""
        config = self._get_symbol_risk_config(symbol)
        
        if symbol not in returns_history or len(returns_history[symbol]) < 30:
            return 0.0
        
        symbol_returns = np.array(returns_history[symbol])
        
        # Apply asset class specific VaR multiplier
        var_multiplier = config["var_multiplier"]
        
        # Calculate VaR
        var = np.percentile(symbol_returns, 5) * var_multiplier * position_size
        
        return abs(var)
    
    def calculate_portfolio_var(self, positions, returns_history, confidence_level=0.05):
        """Calculate Portfolio Value at Risk using historical simulation with asset class adjustments"""
        if len(returns_history) < 30:
            return 0.0
        
        portfolio_returns = []
        for position in positions:
            symbol = position['symbol']
            symbol_returns = returns_history.get(symbol, [])
            if len(symbol_returns) > 0:
                # Apply symbol-specific risk adjustments
                config = self._get_symbol_risk_config(symbol)
                adjusted_returns = np.array(symbol_returns) * position['weight'] * config["var_multiplier"]
                portfolio_returns.append(adjusted_returns)
        
        if not portfolio_returns:
            return 0.0
        
        # Sum portfolio returns
        total_portfolio_returns = np.sum(portfolio_returns, axis=0)
        
        # Calculate VaR at specified confidence level
        var = np.percentile(total_portfolio_returns, confidence_level * 100)
        return abs(var)
    
    def calculate_correlation_matrix(self, returns_data):
        """Calculate correlation matrix with asset class grouping"""
        if len(returns_data) < 2:
            return {}
        
        symbols = list(returns_data.keys())
        n_symbols = len(symbols)
        correlation_matrix = np.zeros((n_symbols, n_symbols))
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    returns1 = np.array(returns_data[symbol1])
                    returns2 = np.array(returns_data[symbol2])
                    
                    if len(returns1) > 10 and len(returns2) > 10:
                        # Align lengths
                        min_len = min(len(returns1), len(returns2))
                        corr = np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1]
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        # Store as dictionary for easy access
        correlation_dict = {}
        for i, symbol1 in enumerate(symbols):
            correlation_dict[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                correlation_dict[symbol1][symbol2] = correlation_matrix[i, j]
        
        self.correlation_matrix = correlation_dict
        return correlation_dict
    
    def check_correlation_limits(self, new_symbol, existing_positions):
        """Check if adding new position would violate correlation limits"""
        if new_symbol not in self.correlation_matrix:
            return True, []  # Allow if no correlation data
        
        config = self._get_symbol_risk_config(new_symbol)
        correlation_threshold = config["correlation_threshold"]
        
        violations = []
        for position in existing_positions:
            existing_symbol = position['symbol']
            if existing_symbol in self.correlation_matrix[new_symbol]:
                correlation = abs(self.correlation_matrix[new_symbol][existing_symbol])
                if correlation > correlation_threshold:
                    violations.append({
                        'symbol_pair': (new_symbol, existing_symbol),
                        'correlation': correlation,
                        'limit': correlation_threshold
                    })
        
        return len(violations) == 0, violations
    
    def calculate_optimal_position_size(self, symbol, account_balance, atr_value, confidence):
        """Calculate optimal position size based on symbol characteristics"""
        config = self._get_symbol_risk_config(symbol)
        metadata = SYMBOL_METADATA.get(symbol, {})
        
        # Base position size from risk config
        max_position_size = config["max_position_size"]
        
        # Adjust for confidence
        confidence_multiplier = min(confidence / 0.7, 1.0)  # Scale confidence to 0-1
        
        # Adjust for volatility profile
        volatility_profile = metadata.get("volatility_profile", "medium")
        if volatility_profile == "very_high":
            volatility_multiplier = 0.6
        elif volatility_profile == "high":
            volatility_multiplier = 0.8
        elif volatility_profile == "low":
            volatility_multiplier = 1.2
        else:
            volatility_multiplier = 1.0
        
        # Calculate position size
        base_size = account_balance * max_position_size * confidence_multiplier * volatility_multiplier
        
        # Apply ATR-based sizing
        pip_value = metadata.get("pip_value", 1.0)
        stop_loss_atr = config["stop_loss_atr"]
        stop_loss_pips = (atr_value * stop_loss_atr) / pip_value
        
        # Risk per trade (2% of account)
        risk_per_trade = account_balance * 0.02
        atr_based_size = risk_per_trade / stop_loss_pips
        
        # Use the smaller of the two
        optimal_size = min(base_size, atr_based_size)
        
        return optimal_size
    
    async def validate_trade(self, symbol, size, existing_positions, returns_history, account_balance):
        """Comprehensive trade validation with asset class specific checks"""
        validation_result = {
            'approved': True,
            'violations': [],
            'risk_metrics': {},
            'recommended_size': size
        }
        
        # Create hypothetical new positions list
        new_positions = existing_positions.copy()
        new_positions.append({'symbol': symbol, 'size': size, 'weight': size / account_balance})
        
        # 1. Asset class exposure check
        asset_class_ok, asset_class_violations = self.check_asset_class_limits(symbol, existing_positions)
        if not asset_class_ok:
            validation_result['approved'] = False
            validation_result['violations'].extend(asset_class_violations)
        
        # 2. Correlation check
        corr_ok, corr_violations = self.check_correlation_limits(symbol, existing_positions)
        if not corr_ok:
            validation_result['approved'] = False
            validation_result['violations'].extend(corr_violations)
        
        # 3. VaR check
        portfolio_var = self.calculate_portfolio_var(new_positions, returns_history)
        validation_result['risk_metrics']['portfolio_var'] = portfolio_var
        
        daily_var_limit = PORTFOLIO_RISK_LIMITS["daily_var_limit"]
        if portfolio_var > daily_var_limit:
            validation_result['approved'] = False
            validation_result['violations'].append({
                'type': 'var_limit',
                'var': portfolio_var,
                'limit': daily_var_limit
            })
        
        # 4. Position size check
        config = self._get_symbol_risk_config(symbol)
        max_position_size = config["max_position_size"]
        if size / account_balance > max_position_size:
            validation_result['approved'] = False
            validation_result['violations'].append({
                'type': 'position_size_limit',
                'size': size / account_balance,
                'limit': max_position_size
            })
        
        # 5. Send alerts for violations
        if not validation_result['approved']:
            violation_msg = f"Trade validation failed for {symbol}:\n"
            for violation in validation_result['violations']:
                violation_msg += f"- {violation}\n"
            await self.observability.send_discord_alert(violation_msg, "WARNING")
        
        return validation_result

class PortfolioRiskManager:
    """
    Portfolio risk management based on asset correlation.
    """
    def __init__(self, symbols, data_manager):
        self.symbols = symbols
        self.data_manager = data_manager
        self.correlation_matrix = None
        self.last_update_time = None

    def update_correlation_matrix(self, force_update=False):
        """
        Tính toán và lưu lại ma trận tương quan dựa trên lợi nhuận hàng ngày.
        """
        now = datetime.now()

    # === Hybrid correlation: long (D1) + short (H1/H4) ===
    def _compute_corr(self, timeframe='D1', lookback=250):
        all_returns = {}
        for symbol in self.symbols:
            df_tf = self.data_manager.fetch_multi_timeframe_data(symbol, count=int(lookback), timeframes_to_use=[timeframe]).get(timeframe)
            if df_tf is not None and not df_tf.empty:
                all_returns[symbol] = df_tf['close'].pct_change().dropna()
        if not all_returns:
            return pd.DataFrame()
        # Align to common index
        ret_df = pd.DataFrame(all_returns).dropna(how='all').fillna(0)
        return ret_df.corr().clip(-1.0, 1.0)

    def update_correlation_hybrid(self, tf_long='D1', tf_short='H1', lb_long=250, lb_short=500, method='max', w_long=0.6):
        print(f"[Risk Manager] Hybrid Corr → long={tf_long}/{lb_long}, short={tf_short}/{lb_short}, method={method}")
        corr_long = self._compute_corr(tf_long, lb_long)
        corr_short = self._compute_corr(tf_short, lb_short)
        # Ensure same axes
        symbols = sorted(set(corr_long.index).union(set(corr_short.index)))
        corr_long = corr_long.reindex(index=symbols, columns=symbols).fillna(0)
        corr_short = corr_short.reindex(index=symbols, columns=symbols).fillna(0)
        if method == 'max':
            self.corr_matrix = np.maximum(corr_long.values, corr_short.values)
            self.corr_symbols = symbols
        else:
            wl = float(w_long)
            ws = 1.0 - wl
            self.corr_matrix = (wl * corr_long.values) + (ws * corr_short.values)
            self.corr_symbols = symbols
        print("   [Hybrid] Corr matrix updated.")

    def get_corr_pair(self, a, b):
        try:
            i = self.corr_symbols.index(a)
            j = self.corr_symbols.index(b)
            return float(self.corr_matrix[i, j])
        except Exception:
            return 0.0

        # Chỉ cập nhật 1 lần mỗi 24 giờ để tiết kiệm tài nguyên
        if not force_update and self.last_update_time and (now - self.last_update_time).total_seconds() < 24 * 3600:
            return

        print("[Risk Manager] Cập nhật ma trận tương quan...")
        all_returns = {}
        for symbol in self.symbols:
            # Lấy dữ liệu D1 để tính tương quan dài hạn
            df_d1 = self.data_manager.fetch_multi_timeframe_data(symbol, count=100, timeframes_to_use=["D1"]).get("D1")
            if df_d1 is not None and not df_d1.empty:
                all_returns[symbol] = df_d1['close'].pct_change().dropna()

        if not all_returns:
            print("[Risk Manager] ⚠️ Không có dữ liệu để tính tương quan.")
            return

        returns_df = pd.DataFrame(all_returns).dropna()
        self.correlation_matrix = returns_df.corr()
        self.last_update_time = now
        print("[Risk Manager] ✅ Ma trận tương quan đã được cập nhật.")
        # print(self.correlation_matrix) # Bỏ comment để xem ma trận

    def get_adjusted_risk_factor(self, new_symbol, new_signal, open_positions, threshold=0.6):
        """
        Tính toán hệ số điều chỉnh rủi ro cho một lệnh mới.
        Hệ số < 1.0 nếu có tương quan cao.
        """
        if self.correlation_matrix is None:
            return 1.0 # Chưa có ma trận, không điều chỉnh

        risk_factor = 1.0
        max_correlation_impact = 0.0

        for open_pos in open_positions.values():
            open_symbol = open_pos['symbol']
            open_signal = open_pos['signal']

            if new_symbol == open_symbol: continue

            try:
                correlation = self.correlation_matrix.loc[new_symbol, open_symbol]
            except KeyError:
                continue

            # Kiểm tra xem 2 lệnh có cùng hướng hay không (ví dụ: cùng BUY hoặc cùng SELL)
            # Hoặc ngược hướng nếu tương quan là âm
            is_same_direction_bet = (new_signal == open_signal and correlation > 0) or \
                                    (new_signal != open_signal and correlation < 0)

            if abs(correlation) > threshold and is_same_direction_bet:
                # Ghi nhận mức độ ảnh hưởng của tương quan mạnh nhất
                max_correlation_impact = max(max_correlation_impact, abs(correlation))

        if max_correlation_impact > 0:
            # Giảm rủi ro dựa trên mức độ tương quan
            # Ví dụ: corr=0.8 -> giảm 40% rủi ro (1 - 0.8*0.5)
            # Ví dụ: corr=0.9 -> giảm 45% rủi ro
            risk_factor = 1.0 - (max_correlation_impact * 0.5)
            print(f"[Risk Manager] Tương quan cao được phát hiện ({max_correlation_impact:.2f}). Giảm rủi ro lệnh mới xuống {risk_factor:.2f} lần.")

        return risk_factor
# <<< THAY THẾ TOÀN BỘ HÀM NÀY BẰNG PHIÊN BẢN ĐÃ SỬA LỖI >>>

# TÌM VÀ THAY THẾ TOÀN BỘ HÀM NÀY BẰNG PHIÊN BẢN ĐÃ SỬA LỖI >>>

def calculate_rsi_divergence_vectorized(df, window=14, lookback=60):
    """
    Xác định phân kỳ RSI bằng phương pháp vector hóa, nhanh hơn vòng lặp.
    """
    df['bearish_divergence'] = 0
    df['bullish_divergence'] = 0

    # Chỉ xử lý trên `lookback` nến gần nhất để tiết kiệm thời gian
    df_slice = df.iloc[-lookback:].copy()

    # Tìm tất cả các đỉnh và đáy cục bộ
    price_peaks_idx = argrelextrema(df_slice['high'].to_numpy(), np.greater_equal, order=5)[0]
    price_troughs_idx = argrelextrema(df_slice['low'].to_numpy(), np.less_equal, order=5)[0]
    rsi_peaks_idx = argrelextrema(df_slice['rsi_14'].to_numpy(), np.greater_equal, order=5)[0]

    # <<< ĐÃ SỬA LỖI: Xóa một chữ 'l' bị thừa >>>
    rsi_troughs_idx = argrelextrema(df_slice['rsi_14'].to_numpy(), np.less_equal, order=5)[0]

    if len(price_peaks_idx) < 2 or len(rsi_peaks_idx) < 2 or \
       len(price_troughs_idx) < 2 or len(rsi_troughs_idx) < 2:
        return df # Không đủ đỉnh/đáy để so sánh

    # -- Kiểm tra phân kỳ giảm (Bearish Divergence) --
    # Giá tạo đỉnh cao hơn (Higher High)
    if df_slice['high'].iloc[price_peaks_idx[-1]] > df_slice['high'].iloc[price_peaks_idx[-2]]:
        # RSI tạo đỉnh thấp hơn (Lower High)
        if df_slice['rsi_14'].iloc[rsi_peaks_idx[-1]] < df_slice['rsi_14'].iloc[rsi_peaks_idx[-2]]:
            # Đánh dấu tại vị trí của đỉnh giá cuối cùng
            df.loc[df_slice.index[price_peaks_idx[-1]], 'bearish_divergence'] = 1

    # -- Kiểm tra phân kỳ tăng (Bullish Divergence) --
    # Giá tạo đáy thấp hơn (Lower Low)
    if df_slice['low'].iloc[price_troughs_idx[-1]] < df_slice['low'].iloc[price_troughs_idx[-2]]:
        # RSI tạo đáy cao hơn (Higher Low)
        if df_slice['rsi_14'].iloc[rsi_troughs_idx[-1]] > df_slice['rsi_14'].iloc[rsi_troughs_idx[-2]]:
            # Đánh dấu tại vị trí của đáy giá cuối cùng
            df.loc[df_slice.index[price_troughs_idx[-1]], 'bullish_divergence'] = 1

    return df
# 1. Sửa hàm fetch dữ liệu thành hàm `async`
async def fetch_symbol_data_async(session, symbol, data_manager):
    """Hàm bất đồng bộ để fetch dữ liệu cho 1 symbol."""
    print(f"   -> Bắt đầu fetch {symbol}...")
    # Logic fetch của bạn sẽ được chuyển đổi để dùng `await session.get(...)`
    # Đây chỉ là ví dụ, không phải code chạy được ngay
    # multi_tf_data = await data_manager.fetch_multi_timeframe_data_async(session, symbol)
    # df_features = data_manager.feature_engineer.create_all_features(multi_tf_data)
    # return symbol, df_features
    await asyncio.sleep(2) # Giả lập thời gian chờ
    print(f"   <- Hoàn thành fetch {symbol}.")
    return symbol, pd.DataFrame() # Trả về DataFrame rỗng

# 2. Viết lại vòng lặp chính để chạy bất đồng bộ
async def main_async_loop(bot):
    async with aiohttp.ClientSession() as session:
        while True:
            print("\n----- Bắt đầu chu kỳ phân tích bất đồng bộ -----")

            # Tạo các tác vụ fetch dữ liệu cho tất cả symbol
            tasks = [fetch_symbol_data_async(session, symbol, bot.data_manager) for symbol in SYMBOLS]

            # Chạy tất cả các tác vụ song song và đợi chúng hoàn thành
            results = await asyncio.gather(*tasks)

            # Xử lý kết quả sau khi đã có dữ liệu của tất cả symbols
            live_data_all_symbols = {symbol: df for symbol, df in results}

            # ... Gọi hàm run_portfolio_rl_strategy với dữ liệu đã được fetch song song ...
            # bot.run_portfolio_rl_strategy(live_data_all_symbols)

            print("----- Kết thúc chu kỳ phân tích -----")
            await asyncio.sleep(60) # Chờ 60 giây



# ==== RL PRODUCTION EXECUTION ADAPTER (Appended by Assistant) ====
import json, math, requests

def _decode_actions_to_vector(action, n_symbols, action_space):
    """Return vector length N with values in {0,1,2} (0=HOLD,1=BUY,2=SELL)."""
    import numpy as _np
    arr = _np.asarray(action).reshape(-1)

    # MultiDiscrete?
    try:
        if hasattr(action_space, "nvec") and int(getattr(action_space.nvec, "size", 0)) == n_symbols:
            vec = arr[:n_symbols].astype(int).tolist()
            return [v if v in (0,1,2) else 0 for v in vec]
    except Exception:
        pass

    # Discrete fallback
    try:
        n_total = int(getattr(action_space, "n", n_symbols * 3))
    except Exception:
        n_total = n_symbols * 3
    A = 3 if n_total % max(1, n_symbols) in (0,3) else max(1, n_total // max(1, n_symbols))
    raw = int(arr[0]) if arr.size else 0
    sym_idx = int(_np.clip(raw // A, 0, max(0, n_symbols-1)))
    act = int(raw % A)
    vec = [0]*n_symbols
    vec[sym_idx] = act if act in (0,1,2) else 0
    return vec

def _to_oanda_instrument(sym: str) -> str:
    m = {"XAUUSD":"XAU_USD","XAGUSD":"XAG_USD","BTCUSD":"BTC_USD","ETHUSD":"ETH_USD"}
    if sym in m: return m[sym]
    if len(sym)==6 and sym.isalpha(): return f"{sym[:3]}_{sym[3:]}"
    return sym

def _place_market_order(symbol: str, side: str, units: int,
                        sl_price=None, tp_price=None):
    """
    Send market order to OANDA v20. If OANDA_ACCOUNT_ID not defined, DRY-RUN logs only.
    """
    import logging
    account_id = globals().get("OANDA_ACCOUNT_ID", None)
    api_key = globals().get("OANDA_API_KEY", None)
    base_url = globals().get("OANDA_URL", "https://api-fxtrade.oanda.com/v3")
    if not account_id or not api_key:
        logging.warning("[DRY-RUN] Missing OANDA_ACCOUNT_ID/API_KEY; skip live order.")
        return {"dry_run": True, "symbol": symbol, "side": side, "units": units}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    instrument = _to_oanda_instrument(symbol)
    u = int(units) if side.lower()=="buy" else -int(units)
    u = 1 if u==0 else u

    payload = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(u),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }
    if sl_price:
        payload["order"]["stopLossOnFill"] = {"timeInForce": "GTC", "price": f"{sl_price:.5f}"}
    if tp_price:
        payload["order"]["takeProfitOnFill"] = {"timeInForce": "GTC", "price": f"{tp_price:.5f}"}

    url = f"{base_url}/accounts/{account_id}/orders"
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
    if r.status_code >= 300:
        raise RuntimeError(f"OANDA error {r.status_code}: {r.text}")
    return r.json()

def _calc_position_size(balance: float, atr: float, pip_value: float,
                        risk_per_trade: float, sl_atr_mult: float, conf: float):
    import numpy as _np
    if atr is None or atr <= 0 or pip_value <= 0:
        return max(1, int(balance*0.0001)), None
    risk_amount = max(1.0, balance * risk_per_trade * max(0.2, float(conf)))
    sl_dist = atr * sl_atr_mult
    units = int(max(1, risk_amount / (sl_dist * pip_value)))
    return units, sl_dist

def rl_execute_production(model, symbols, observation, features_map, get_balance_func,
                          rl_deterministic=True,
                          min_conf_threshold=None,
                          risk_per_trade=0.0075,
                          sl_atr_mult=2.0,
                          logger_name="ProdBot"):
    """
    Plug-and-play executor:
      - model: SB3 model (PPO/A2C/...) đã load
      - symbols: list[str] thứ tự đúng như env
      - observation: obs theo policy của model
      - features_map: dict[symbol] -> {'confidence': float, 'atr': float, ...}
      - get_balance_func: callable -> float
    """
    import logging, numpy as _np
    logger = logging.getLogger(logger_name)

    action, _ = model.predict(observation, deterministic=bool(rl_deterministic))

    try:
        env = model.get_env()
        env_syms = env.get_attr("symbols")[0]
        if env_syms and len(env_syms)==len(symbols):
            symbols = env_syms
    except Exception:
        pass

    vec = _decode_actions_to_vector(action, len(symbols), getattr(model, "action_space", None))
    logger.info(f"[RL Decision] vector={vec} (0=HOLD,1=BUY,2=SELL)")
    logger.info(f"[Trace] Mapping: {list(enumerate(symbols))}")

    balance = float(get_balance_func())
    any_action = False

    # thresholds
    conf_gate = 0.0 if min_conf_threshold is None else float(min_conf_threshold)

    for idx, a in enumerate(vec):
        if a not in (1,2):
            continue
        sym = symbols[idx]
        feats = features_map.get(sym, {})
        conf = float(feats.get("confidence", 1.0))
        if conf < conf_gate:
            logger.info(f"[Gate] Skip {sym} due to confidence {conf:.2f} < {conf_gate:.2f}")
            continue

        atr = feats.get("atr", None)
        pip_val = feats.get("pip_value", 0.0001)

        units, sl_dist = _calc_position_size(balance, atr, pip_val, risk_per_trade, sl_atr_mult, conf)
        side = "buy" if a==1 else "sell"

        # Optional SL/TP derivation hook (user to implement via feats)
        sl_price = feats.get("sl_price")
        tp_price = feats.get("tp_price")

        try:
            resp = _place_market_order(sym, side, units, sl_price, tp_price)
            txid = resp.get("lastTransactionID") or resp.get("orderCreateTransaction", {}).get("id") or "dry-run"
            logger.info(f"[OPENED] {sym} {side.upper()} units={units} tx={txid}")
            any_action = True
        except Exception as e:
            logger.error(f"[ORDER ERROR] {sym}: {e}")

    if not any_action:
        logger.info("[RL Decision] HOLD (portfolio)")
    return any_action

class QualityGateValidator:
    """Advanced quality gates for model validation"""
    
    def __init__(self, config=None):
        self.config = config or {
            'min_sharpe_ratio': MIN_SHARPE_RATIO,
            'max_drawdown': MAX_DRAWDOWN_THRESHOLD,
            'min_calmar_ratio': MIN_CALMAR_RATIO,
            'min_information_ratio': MIN_INFORMATION_RATIO,
            'min_samples': MIN_SAMPLES_GATE,
            'min_f1_score': MIN_F1_SCORE_GATE,
            'max_std_f1': MAX_STD_F1_GATE
        }
        
    def calculate_deflated_sharpe(self, returns, benchmark_returns=None):
        """Calculate Deflated Sharpe Ratio to account for multiple testing bias"""
        if len(returns) < 30:
            return 0.0
            
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Deflation factor based on number of trials and skewness/kurtosis
        n_trials = 100  # Assume 100 different strategies tested
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        
        # Deflation adjustment
        variance_adjustment = (1 + (skewness**2)/6 + (kurtosis**2)/24) / len(returns)
        deflated_sharpe = sharpe / np.sqrt(1 + variance_adjustment * np.log(n_trials))
        
        return deflated_sharpe
    
    def _calculate_skewness(self, returns):
        """Calculate skewness of returns"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 3)
    
    def _calculate_kurtosis(self, returns):
        """Calculate excess kurtosis of returns"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    def calculate_maximum_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def calculate_calmar_ratio(self, returns):
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = np.mean(returns) * 252
        max_dd = self.calculate_maximum_drawdown(returns)
        return annual_return / max_dd if max_dd > 0 else 0
    
    def validate_model_quality(self, model_results, returns=None):
        """Comprehensive quality gate validation"""
        validation_results = {
            'passed': True,
            'failures': [],
            'metrics': {}
        }
        
        # Statistical validation
        if 'cv_scores' in model_results:
            cv_scores = model_results['cv_scores']
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            validation_results['metrics']['mean_cv_score'] = mean_score
            validation_results['metrics']['std_cv_score'] = std_score
            
            if mean_score < self.config['min_f1_score']:
                validation_results['passed'] = False
                validation_results['failures'].append(f"Mean CV score {mean_score:.3f} < {self.config['min_f1_score']}")
                
            if std_score > self.config['max_std_f1']:
                validation_results['passed'] = False
                validation_results['failures'].append(f"CV score std {std_score:.3f} > {self.config['max_std_f1']}")
        
        # Financial metrics validation
        if returns is not None and len(returns) > 0:
            deflated_sharpe = self.calculate_deflated_sharpe(returns)
            max_dd = self.calculate_maximum_drawdown(returns)
            calmar = self.calculate_calmar_ratio(returns)
            
            validation_results['metrics']['deflated_sharpe'] = deflated_sharpe
            validation_results['metrics']['max_drawdown'] = max_dd
            validation_results['metrics']['calmar_ratio'] = calmar
            
            if deflated_sharpe < self.config['min_sharpe_ratio']:
                validation_results['passed'] = False
                validation_results['failures'].append(f"Deflated Sharpe {deflated_sharpe:.3f} < {self.config['min_sharpe_ratio']}")
                
            if max_dd > self.config['max_drawdown']:
                validation_results['passed'] = False
                validation_results['failures'].append(f"Max drawdown {max_dd:.3f} > {self.config['max_drawdown']}")
                
            if calmar < self.config['min_calmar_ratio']:
                validation_results['passed'] = False
                validation_results['failures'].append(f"Calmar ratio {calmar:.3f} < {self.config['min_calmar_ratio']}")
        
        return validation_results

class AdvancedObservability:
    """Enhanced observability with Discord alerts and comprehensive logging"""
    
    def __init__(self, discord_webhook=None):
        self.discord_webhook = discord_webhook or DISCORD_WEBHOOK
        self.last_alert_time = {}
        self.performance_metrics = {}
        self.alert_queue = deque(maxlen=100)
        
    async def send_discord_alert(self, message, alert_type="INFO", force=False):
        """Send Discord alert with rate limiting"""
        current_time = datetime.now()
        alert_key = f"{alert_type}_{hash(message) % 1000}"
        
        # Rate limiting
        if not force and alert_key in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[alert_key]).total_seconds()
            if time_diff < DISCORD_RATE_LIMIT_SECONDS:
                return False
        
        self.last_alert_time[alert_key] = current_time
        
        # Format message
        color_map = {
            "ERROR": 15158332,  # Red
            "WARNING": 15105570,  # Orange
            "SUCCESS": 3066993,  # Green
            "INFO": 3447003  # Blue
        }
        
        embed = {
            "title": f"Trading Bot Alert - {alert_type}",
            "description": message,
            "color": color_map.get(alert_type, 3447003),
            "timestamp": current_time.isoformat(),
            "footer": {"text": "Trading Bot Monitoring"}
        }
        
        payload = {"embeds": [embed]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=payload) as response:
                    if response.status == 200:
                        logging.info(f"Discord alert sent: {alert_type}")
                        return True
                    else:
                        logging.error(f"Discord alert failed: {response.status}")
                        return False
        except Exception as e:
            logging.error(f"Discord alert error: {e}")
            return False
    
    def calculate_performance_metrics(self, returns, positions=None):
        """Calculate comprehensive performance metrics"""
        if len(returns) < 2:
            return {}
        
        returns_array = np.array(returns)
        
        # Basic metrics
        total_return = np.prod(1 + returns_array) - 1
        annual_return = np.mean(returns_array) * 252
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Advanced metrics
        quality_validator = QualityGateValidator()
        max_drawdown = quality_validator.calculate_maximum_drawdown(returns_array)
        calmar_ratio = quality_validator.calculate_calmar_ratio(returns_array)
        deflated_sharpe = quality_validator.calculate_deflated_sharpe(returns_array)
        
        # Win rate and other trading metrics
        positive_returns = returns_array[returns_array > 0]
        negative_returns = returns_array[returns_array < 0]
        
        win_rate = len(positive_returns) / len(returns_array) if len(returns_array) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'deflated_sharpe': deflated_sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(returns_array)
        }
        
        self.performance_metrics = metrics
        return metrics
    
    async def send_performance_report(self, metrics, symbol=None):
        """Send comprehensive performance report"""
        symbol_text = f" for {symbol}" if symbol else ""
        
        report = f"""
**Performance Report{symbol_text}**
📈 Total Return: {metrics.get('total_return', 0):.2%}
📊 Annual Return: {metrics.get('annual_return', 0):.2%}
📉 Volatility: {metrics.get('volatility', 0):.2%}
⚡ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
🎯 Deflated Sharpe: {metrics.get('deflated_sharpe', 0):.3f}
📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
🔥 Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}
🎲 Win Rate: {metrics.get('win_rate', 0):.2%}
💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}
📋 Total Trades: {metrics.get('total_trades', 0)}
        """.strip()
        
        await self.send_discord_alert(report, "INFO", force=True)

class AdvancedFeatureStore:
    """Lightweight feature store with schema validation"""
    
    def __init__(self, db_path="feature_store.db"):
        self.db_path = db_path
        self.schema = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize feature store database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    symbol TEXT,
                    timestamp TEXT,
                    timeframe TEXT,
                    feature_name TEXT,
                    feature_value REAL,
                    feature_type TEXT,
                    created_at TEXT,
                    PRIMARY KEY (symbol, timestamp, timeframe, feature_name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_schema (
                    feature_name TEXT PRIMARY KEY,
                    feature_type TEXT,
                    min_value REAL,
                    max_value REAL,
                    description TEXT,
                    created_at TEXT
                )
            """)
    
    def register_feature_schema(self, feature_name, feature_type, min_value=None, max_value=None, description=""):
        """Register feature schema for validation"""
        schema_entry = {
            'type': feature_type,
            'min_value': min_value,
            'max_value': max_value,
            'description': description
        }
        
        self.schema[feature_name] = schema_entry
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_schema 
                (feature_name, feature_type, min_value, max_value, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (feature_name, feature_type, min_value, max_value, description, datetime.now().isoformat()))
    
    def validate_feature(self, feature_name, feature_value):
        """Validate feature against schema"""
        if feature_name not in self.schema:
            return True  # Allow unknown features
        
        schema = self.schema[feature_name]
        
        # Type validation
        if schema['type'] == 'numeric' and not isinstance(feature_value, (int, float)):
            return False
        
        # Range validation
        if schema['min_value'] is not None and feature_value < schema['min_value']:
            return False
        if schema['max_value'] is not None and feature_value > schema['max_value']:
            return False
        
        return True
    
    def store_features(self, symbol, timestamp, timeframe, features_dict):
        """Store features with validation"""
        valid_features = {}
        
        for feature_name, feature_value in features_dict.items():
            if self.validate_feature(feature_name, feature_value):
                valid_features[feature_name] = feature_value
            else:
                logging.warning(f"Feature validation failed: {feature_name}={feature_value}")
        
        with sqlite3.connect(self.db_path) as conn:
            for feature_name, feature_value in valid_features.items():
                feature_type = self.schema.get(feature_name, {}).get('type', 'numeric')
                conn.execute("""
                    INSERT OR REPLACE INTO features 
                    (symbol, timestamp, timeframe, feature_name, feature_value, feature_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, timestamp, timeframe, feature_name, feature_value, feature_type, datetime.now().isoformat()))
    
    def get_features(self, symbol, timeframe, start_time=None, end_time=None, feature_names=None):
        """Retrieve features with filtering"""
        query = """
            SELECT timestamp, feature_name, feature_value 
            FROM features 
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if feature_names:
            placeholders = ','.join(['?' for _ in feature_names])
            query += f" AND feature_name IN ({placeholders})"
            params.extend(feature_names)
        
        query += " ORDER BY timestamp, feature_name"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results, columns=['timestamp', 'feature_name', 'feature_value'])
            return df.pivot(index='timestamp', columns='feature_name', values='feature_value')
        else:
            return pd.DataFrame()

class AdvancedScheduler:
    """Enhanced scheduler with session guards and timeframe-aware synchronization"""
    
    def __init__(self):
        self.symbol_sessions = {}
        self.last_sync_times = {}
        self.observability = AdvancedObservability()
        
    def is_symbol_session_active(self, symbol):
        """Check if symbol's trading session is currently active"""
        if symbol not in ENHANCED_TIMEFRAME_MAPPING:
            return True  # Default to active if no mapping
        
        mapping = ENHANCED_TIMEFRAME_MAPPING[symbol]
        session_hours = mapping.get('session_hours', {})
        
        if not session_hours:
            return True
        
        try:
            import pytz
            timezone = pytz.timezone(session_hours.get('timezone', 'UTC'))
            current_time = datetime.now(timezone)
            current_hour = current_time.hour
            
            start_hour = session_hours.get('start', 0)
            end_hour = session_hours.get('end', 23)
            
            # Handle overnight sessions (e.g., AUDNZD: 21-7)
            if start_hour > end_hour:
                return current_hour >= start_hour or current_hour <= end_hour
            else:
                return start_hour <= current_hour <= end_hour
                
        except Exception as e:
            logging.warning(f"Session check error for {symbol}: {e}")
            return True
    
    def should_skip_weekend(self, symbol):
        """Check if symbol should skip weekend trading"""
        if symbol not in ENHANCED_TIMEFRAME_MAPPING:
            return True  # Default to skip weekends
        
        mapping = ENHANCED_TIMEFRAME_MAPPING[symbol]
        weekend_guard = mapping.get('weekend_guard', True)
        
        if not weekend_guard:
            return False  # Crypto symbols can trade on weekends
        
        return is_weekend()
    
    def get_optimal_timeframes(self, symbol):
        """Get optimal timeframes for a symbol"""
        if symbol not in ENHANCED_TIMEFRAME_MAPPING:
            return PRIMARY_TIMEFRAME_BY_SYMBOL.get(symbol, "H1"), ["H4", "D1"]
        
        mapping = ENHANCED_TIMEFRAME_MAPPING[symbol]
        primary = mapping.get('primary', 'H1')
        secondary = mapping.get('secondary', ['H4', 'D1'])
        
        return primary, secondary
    
    def should_wait_for_candle_sync(self, symbol, primary_tf):
        """Enhanced candle synchronization with session awareness"""
        # Skip sync for crypto
        if is_crypto_symbol(symbol):
            return False
        
        # Skip if outside trading session
        if not self.is_symbol_session_active(symbol):
            return False
        
        # Skip on weekends for non-crypto
        if self.should_skip_weekend(symbol):
            return False
        
        return True
    
    async def wait_for_optimal_entry_time(self, symbol):
        """Wait for optimal entry time based on session and volatility patterns"""
        if symbol not in ENHANCED_TIMEFRAME_MAPPING:
            return True
        
        mapping = ENHANCED_TIMEFRAME_MAPPING[symbol]
        
        # Check for lunch breaks
        if 'lunch_break' in mapping:
            lunch = mapping['lunch_break']
            try:
                import pytz
                timezone = pytz.timezone(mapping['session_hours'].get('timezone', 'UTC'))
                current_time = datetime.now(timezone)
                
                if lunch['start'] <= current_time.hour <= lunch['end']:
                    wait_minutes = (lunch['end'] - current_time.hour) * 60
                    logging.info(f"Waiting {wait_minutes} minutes for {symbol} lunch break to end")
                    await asyncio.sleep(wait_minutes * 60)
            except Exception as e:
                logging.warning(f"Lunch break check error for {symbol}: {e}")
        
        # Check for high activity periods
        if 'high_activity_hours' in mapping:
            high_activity = mapping['high_activity_hours']
            try:
                import pytz
                timezone = pytz.timezone(mapping['session_hours'].get('timezone', 'UTC'))
                current_time = datetime.now(timezone)
                current_hour = current_time.hour
                
                # Check if we're in a high activity period
                in_high_activity = any(
                    period['start'] <= current_hour <= period['end'] 
                    for period in high_activity
                )
                
                if not in_high_activity:
                    # Find next high activity period
                    next_period = None
                    for period in high_activity:
                        if period['start'] > current_hour:
                            next_period = period
                            break
                    
                    if next_period:
                        wait_hours = next_period['start'] - current_hour
                        logging.info(f"Waiting {wait_hours} hours for {symbol} high activity period")
                        await asyncio.sleep(wait_hours * 3600)
                        
            except Exception as e:
                logging.warning(f"High activity check error for {symbol}: {e}")
        
        return True
    
    async def schedule_symbol_processing(self, symbol):
        """Schedule symbol processing with all guards and optimizations"""
        try:
            # 1. Weekend guard
            if self.should_skip_weekend(symbol):
                await self.observability.send_discord_alert(
                    f"Skipping {symbol} - weekend guard active", "INFO"
                )
                return False
            
            # 2. Session guard
            if not self.is_symbol_session_active(symbol):
                logging.info(f"Skipping {symbol} - outside trading session")
                return False
            
            # 3. Wait for optimal entry time
            await self.wait_for_optimal_entry_time(symbol)
            
            # 4. Candle synchronization
            primary_tf, _ = self.get_optimal_timeframes(symbol)
            if self.should_wait_for_candle_sync(symbol, primary_tf):
                wait_for_next_primary_candle(primary_tf, symbol)
            
            # 5. Record sync time
            self.last_sync_times[symbol] = datetime.now()
            
            return True
            
        except Exception as e:
            logging.error(f"Scheduler error for {symbol}: {e}")
            await self.observability.send_discord_alert(
                f"Scheduler error for {symbol}: {e}", "ERROR"
            )
            return False

class EventSentimentGate:
    """Advanced event and sentiment gating system"""
    
    def __init__(self):
        self.news_manager = NewsEconomicManager()
        self.sentiment_cache = {}
        self.event_cache = {}
        self.observability = AdvancedObservability()
        
    async def check_high_impact_events(self, symbol, buffer_hours=2):
        """Check for high impact economic events"""
        try:
            events = self.news_manager.get_economic_calendar()
            current_time = datetime.utcnow()
            
            # Filter events within buffer window
            upcoming_events = []
            for event in events:
                if event.get('importance') == 'high':
                    event_time = event.get('time')
                    if event_time:
                        time_diff = abs((event_time - current_time).total_seconds() / 3600)
                        if time_diff <= buffer_hours:
                            upcoming_events.append(event)
            
            if upcoming_events:
                event_msg = f"High impact events near {symbol}:\n"
                for event in upcoming_events[:3]:  # Show max 3 events
                    event_msg += f"- {event.get('event', 'Unknown')} at {event.get('time')}\n"
                
                await self.observability.send_discord_alert(event_msg, "WARNING")
                return False, upcoming_events
            
            return True, []
            
        except Exception as e:
            logging.error(f"Event check error for {symbol}: {e}")
            return True, []  # Default to allow trading
    
    async def analyze_market_sentiment(self, symbol):
        """Analyze market sentiment from multiple sources"""
        try:
            # Get recent news
            news_items = await self.news_manager.get_aggregated_news(symbol)
            
            if not news_items:
                return {'score': 0.0, 'confidence': 0.0, 'reasoning': 'No news available'}
            
            # Use LLM for sentiment analysis
            sentiment_result = self.news_manager.llm_analyzer.analyze_sentiment_of_news(news_items)
            
            # Cache result
            self.sentiment_cache[symbol] = {
                'result': sentiment_result,
                'timestamp': datetime.now(),
                'news_count': len(news_items)
            }
            
            return sentiment_result
            
        except Exception as e:
            logging.error(f"Sentiment analysis error for {symbol}: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'reasoning': f'Analysis error: {e}'}
    
    async def validate_trade_sentiment(self, symbol, trade_direction):
        """Validate trade against sentiment analysis"""
        try:
            sentiment = await self.analyze_market_sentiment(symbol)
            sentiment_score = sentiment.get('score', 0.0)
            confidence = sentiment.get('confidence', 0.0)
            
            # Define sentiment thresholds
            STRONG_POSITIVE = 0.3
            STRONG_NEGATIVE = -0.3
            MIN_CONFIDENCE = 0.6
            
            validation_result = {
                'approved': True,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'reasoning': sentiment.get('reasoning', ''),
                'warnings': []
            }
            
            # Check for conflicting sentiment
            if confidence > MIN_CONFIDENCE:
                if trade_direction == 'buy' and sentiment_score < STRONG_NEGATIVE:
                    validation_result['approved'] = False
                    validation_result['warnings'].append(
                        f"Strong negative sentiment ({sentiment_score:.2f}) conflicts with BUY signal"
                    )
                elif trade_direction == 'sell' and sentiment_score > STRONG_POSITIVE:
                    validation_result['approved'] = False
                    validation_result['warnings'].append(
                        f"Strong positive sentiment ({sentiment_score:.2f}) conflicts with SELL signal"
                    )
            
            # Send alert for sentiment conflicts
            if not validation_result['approved']:
                warning_msg = f"Sentiment conflict for {symbol} {trade_direction.upper()}:\n"
                warning_msg += f"Sentiment: {sentiment_score:.2f} (confidence: {confidence:.2f})\n"
                warning_msg += validation_result['reasoning']
                
                await self.observability.send_discord_alert(warning_msg, "WARNING")
            
            return validation_result
            
        except Exception as e:
            logging.error(f"Sentiment validation error for {symbol}: {e}")
            return {
                'approved': True,  # Default to approve on error
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Validation error: {e}',
                'warnings': []
            }

class StopTrainingOnMaxDrawdown(BaseCallback):
    """
    Một Callback tùy chỉnh để dừng huấn luyện nếu Max Drawdown vượt ngưỡng.
    """
    def __init__(self, max_drawdown_threshold=0.30, check_freq=1, verbose=1):
        """
        Khởi tạo Callback.
        :param max_drawdown_threshold: Ngưỡng sụt giảm tối đa cho phép (ví dụ: 0.30 cho 30%).
        :param check_freq: Số episode cần chạy trước khi kiểm tra 1 lần.
        :param verbose: Bật/tắt log.
        """
        super(StopTrainingOnMaxDrawdown, self).__init__(verbose)
        self.max_drawdown_threshold = max_drawdown_threshold
        self.check_freq = check_freq
        self.episode_count = 0
        self.episode_balances = []

    def _on_step(self) -> bool:
        """
        Hàm này được gọi sau mỗi bước trong môi trường.
        """
        # Ghi nhận lại balance của từng bước trong episode
        current_balance = self.training_env.get_attr("balance")[0]
        self.episode_balances.append(current_balance)
        
        # Kiểm tra xem episode đã kết thúc chưa
        if self.locals["dones"][0]:
            self.episode_count += 1
            
            # Chỉ kiểm tra drawdown sau mỗi `check_freq` episode
            if self.episode_count % self.check_freq == 0:
                balances_series = pd.Series(self.episode_balances)
                
                # Tính toán Max Drawdown
                peak = balances_series.cummax()
                drawdown = (balances_series - peak) / peak
                max_dd = -drawdown.min() # Lấy giá trị dương
                
                if self.verbose > 0:
                    print(f"   [Callback] Episode {self.episode_count} End. Max Drawdown: {max_dd:.2%}")
                
                # Nếu Max Drawdown vượt ngưỡng, dừng huấn luyện
                if max_dd > self.max_drawdown_threshold:
                    if self.verbose > 0:
                        print(f"   [Callback] 🚨 DỪNG HUẤN LUYỆN SỚM! Max Drawdown ({max_dd:.2%}) đã vượt ngưỡng an toàn ({self.max_drawdown_threshold:.2%}).")
                    return False # Trả về False để dừng quá trình .learn()
            
            # Reset danh sách balance cho episode tiếp theo
            self.episode_balances = [] 
            
        return True # Trả về True để tiếp tục huấn luyện
def run_bot_h4_with_rl():
    bot = EnhancedTradingBot()
    bot.run_enhanced_bot()


# ==============================================================================
# 4. KHỐI THỰC THI CHÍNH (MAIN EXECUTION BLOCK)
# ==============================================================================
# THAY THẾ KHỐI if __name__ == "__main__": CŨ BẰNG KHỐI NÀY

# TÌM VÀ THAY THẾ TOÀN BỘ KHỐI THỰC THI CHÍNH (MAIN) Ở CUỐI FILE BẰNG KHỐI NÀY

# TÌM VÀ THAY THẾ TOÀN BỘ KHỐI THỰC THI CHÍNH Ở CUỐI FILE

if __name__ == "__main__":
    # 1. Khởi tạo đối tượng bot
    bot = EnhancedTradingBot()
    
    # 2. Sử dụng asyncio.run() để khởi chạy hàm async run_enhanced_bot
    try:
        # asyncio.run sẽ tự động tạo, chạy và đóng vòng lặp sự kiện
        asyncio.run(bot.run_enhanced_bot())
    except KeyboardInterrupt:
        print("\n🛑 Bot đã được dừng thủ công.")
    except Exception as e:
        import traceback
        # Bắt các lỗi nghiêm trọng không được xử lý bên trong vòng lặp
        print(f"❌ LỖI KHÔNG XÁC ĐỊNH Ở CẤP CAO NHẤT: {e}\n{traceback.format_exc()}")