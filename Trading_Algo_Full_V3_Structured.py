# =========================== #
#        IMPORTS & ENV       #
# =========================== #
from dotenv import load_dotenv
import os
import sys
import json
import logging
import csv
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import time
from textblob import TextBlob
from pytrends.request import TrendReq
import requests
from fredapi import Fred
import wbdata
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump, load

# =========================== #
#        CONFIG & ENV        #
# =========================== #
load_dotenv()
SYMBOL             = os.getenv("SYMBOL")
LOGIN              = int(os.getenv("MT5_ACCOUNT_ID"))
PASSWORD           = os.getenv("MT5_PASSWORD")
SERVER             = os.getenv("MT5_SERVER")
MODE               = os.getenv("MODE", "train")  # train or live
LOOKBACK_DAYS      = int(os.getenv("LOOKBACK_DAYS", 90))
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", 0.7))
RISK_PCT           = float(os.getenv("RISK_PCT", 0.01))
DAILY_DD_PCT       = float(os.getenv("DAILY_DD_PCT", 0.05))
OVERALL_DD_PCT     = float(os.getenv("OVERALL_DD_PCT", 0.10))
MODEL_FILE         = os.getenv("MODEL_FILE", "fist_v3_model_updated.joblib")
FEATURE_FILE       = os.getenv("FEATURE_FILE", "features.json")
NEWS_API_KEY       = os.getenv("NEWS_API_KEY")
FRED_API_KEY       = os.getenv("FRED_API_KEY")

logging.basicConfig(
    filename='trading_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =========================== #
#      MT5 INITIALIZATION    #
# =========================== #

def initialize_mt5():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        raise ValueError(f"Symbol {SYMBOL} not available or not visible in Market Watch.")
    logging.info("MT5 initialized and symbol selected")

# =========================== #
#     DATA FETCHING UTIL     #
# =========================== #

def fetch_historical_data(symbol, timeframe, days):
    utc_from = datetime.utcnow() - timedelta(days=days)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, days * 24)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data fetched for {symbol} timeframe {timeframe}.")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# =========================== #
#    FEATURE ENGINEERING     #
# =========================== #

def add_intermarket_correlation(df, target_symbol="USDJPY"):
    df_other = fetch_historical_data(target_symbol, mt5.TIMEFRAME_H1, LOOKBACK_DAYS)
    df_other = df_other.reindex(df.index, method='nearest')
    df['inter_corr'] = df['close'].rolling(20).corr(df_other['close'])
    return df


def enrich_features(df):
    df = df.copy()
    df['tick_volume']    = df.get('tick_volume', df.get('real_volume', 0))
    df['return_1h']      = df['close'].pct_change()
    df['return_4h']      = df['close'].pct_change(4)
    df['EMA_50']         = ta.ema(df['close'], length=50)
    df['EMA_200']        = ta.ema(df['close'], length=200)
    df['ATR']            = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['RSI']            = ta.rsi(df['close'], length=14)
    df['macd']           = ta.macd(df['close'])['MACD_12_26_9']
    df['ema_diff']       = df['EMA_50'] - df['EMA_200']
    df['hour']           = df.index.hour
    df['dayofweek']      = df.index.dayofweek
    df['hour_sin']       = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']       = np.cos(2 * np.pi * df['hour'] / 24)
    df['range_high']     = df['high'].rolling(14).max()
    df['breakout_up']    = (df['close'] > df['range_high']).astype(int)
    df['volatility_regime'] = (df['ATR'] > df['ATR'].rolling(20).median()).astype(int)
    df = add_intermarket_correlation(df)
    df.dropna(inplace=True)
    return df

# =========================== #
#  SENTIMENT & FUNDAMENTALS   #
# =========================== #

def get_sentiment():
    news_score = trend_score = 0
    try:
        articles = requests.get(
            f"https://newsapi.org/v2/everything?q=forex&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        ).json().get('articles', [])
        scores = [TextBlob(a['title']).sentiment.polarity for a in articles]
        news_score = np.mean(scores) if scores else 0
    except Exception as e:
        logging.error(f"News sentiment error: {e}")
    # Google Trends sentiment with retry/backoff on 429
    trend_score = 0
    max_retries = 3
    for attempt in range(1, max_retries+1):
        try:
            py = TrendReq(hl='en-US', tz=360)
            py.build_payload(['forex','trading'], timeframe='now 1-d')
            trends = py.interest_over_time()
            trend_score = trends.mean().values[0] if not trends.empty else 0
            break
        except Exception as e:
            err_msg = str(e)
            logging.error(f"Trends error (attempt {attempt}): {err_msg}")
            # If rate limited, wait and retry
            if '429' in err_msg or 'Too Many Requests' in err_msg:
                sleep_time = 60 * attempt
                logging.warning(f"Rate limit hit, sleeping for {sleep_time}s before retry")
                time.sleep(sleep_time)
                continue
            else:
                break
    return (news_score + trend_score) / 2

def get_fundamentals():
    data = {'interest_rate':0.25,'gdp_growth':0,'inflation':0}
    # Fetch latest Fed funds rate
    try:
        fred = Fred(api_key=FRED_API_KEY)
        val = fred.get_series_latest_release('FEDFUNDS')
        # support Series or scalar
        if hasattr(val, 'iloc'):
            val = val.iloc[-1]
        data['interest_rate'] = float(val)
    except Exception as e:
        logging.error(f"FRED error: {e}")
    # Fetch latest from World Bank
    try:
        inds = {'NY.GDP.MKTP.KD.ZG':'gdp_growth','FP.CPI.TOTL.ZG':'inflation'}
        wb = wbdata.get_dataframe(inds, country='US')
        if not wb.empty:
            latest = wb.iloc[-1]
            for k, v in inds.items():
                if k in latest:
                    data[v] = float(latest[k])
    except Exception as e:
        logging.error(f"World Bank error: {e}")
    return data

# =========================== #
#       LABELING LOGIC       #
# =========================== #

def label_data(df):
    ret = df['close'].shift(-1) - df['close']
    high_q, low_q = ret.quantile(0.75), ret.quantile(0.25)
    df['label'] = ret.apply(lambda x: 1 if x>=high_q else (-1 if x<=low_q else 0))
    return df.dropna()

# =========================== #
#     DRAWDOWN TRACKERS      #
# =========================== #

class EquityTracker:
    def __init__(self, initial_capital=10000, max_dd_pct=0.1):
        self.capital = initial_capital
        self.peak    = initial_capital
        self.max_dd_pct = max_dd_pct
        self.paused_until = None
    def record(self, pnl, timestamp):
        self.capital += pnl
        self.peak = max(self.peak, self.capital)
        dd = (self.peak - self.capital) / self.peak
        if dd > self.max_dd_pct:
            self.paused_until = timestamp + timedelta(minutes=60)
            logging.warning(f"Drawdown {dd:.2%} exceeded. Pausing until {self.paused_until}")
    def can_trade(self, timestamp):
        return self.paused_until is None or timestamp >= self.paused_until

class FundingPipsTracker(EquityTracker):
    def __init__(self, initial_capital=10000, daily_dd_pct=0.05, overall_dd_pct=0.10):
        super().__init__(initial_capital, overall_dd_pct)
        self.daily_start = initial_capital
        self.daily_dd_pct = daily_dd_pct
        self.current_day = None
    def record(self, pnl, timestamp):
        day = timestamp.date()
        if self.current_day != day:
            self.current_day = day
            self.daily_start = self.capital
        super().record(pnl, timestamp)
        daily_dd = (self.daily_start - self.capital) / self.daily_start
        if daily_dd > self.daily_dd_pct:
            self.paused_until = datetime.combine(day + timedelta(days=1), datetime.min.time())
            logging.warning(f"Daily DD {daily_dd:.2%} exceeded. Pausing until next day.")

# =========================== #
#   POSITION SIZING & LOG    #
# =========================== #

def calculate_lot_size(balance, risk_pct, sl_pips, pip_value=0.0001):
    risk_amount = balance * risk_pct
    lot = risk_amount / (sl_pips * pip_value)
    return round(lot, 2)

def log_trade(timestamp, symbol, side, price, sl, tp, pnl, balance):
    with open('fp_trades.csv','a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp.isoformat(), symbol, side, price, sl, tp, pnl, balance])

# =========================== #
#   TRAINING & BACKTESTING   #
# =========================== #

def train_and_tune():
    initialize_mt5()
    df_h1 = fetch_historical_data(SYMBOL, mt5.TIMEFRAME_H1, LOOKBACK_DAYS)
    df = enrich_features(df_h1)
    df_h4 = enrich_features(fetch_historical_data(SYMBOL, mt5.TIMEFRAME_H4, LOOKBACK_DAYS))
    df_h4 = df_h4.add_suffix('_4h').reindex(df.index).dropna()
    df = df.join(df_h4, how='inner')
    df = label_data(df)
    df['sentiment']   = get_sentiment()
    funds = get_fundamentals()
    df['interest_rate'] = funds['interest_rate']
    df['gdp_growth']    = funds['gdp_growth']
    df['inflation']     = funds['inflation']

    # time-based split
    idx = int(len(df)*0.8)
    train, test = df.iloc[:idx], df.iloc[idx:]
    X_train, y_train = train.drop(columns=['label']), train['label']
    X_test, y_test   = test.drop(columns=['label']), test['label']

    # save feature list
    features = X_train.columns.tolist()
    with open(FEATURE_FILE,'w') as f: json.dump(features, f)

    # oversample with fixed seed
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # hyperparameter search with seed
    base = LGBMClassifier(class_weight={-1:5,0:1,1:5}, random_state=42)
    param_dist = {
        'num_leaves': [31,50,70],
        'max_depth': [3,5,7],
        'learning_rate': [0.01,0.05,0.1],
        'n_estimators': [100,200,300],
        'min_data_in_leaf': [10,20,50]
    }
    rs = RandomizedSearchCV(
        base, param_dist, n_iter=50,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='f1_macro', random_state=42
    )
    rs.fit(X_res, y_res)
    best = rs.best_estimator_

    # cross-validated f1_macro
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(best, X_res, y_res, cv=cv, scoring='f1_macro')
    print("CV f1_macro scores:", scores, "mean:", scores.mean())

    # raw predictions vs thresholds
    print("=== Raw Predictions ===")
    print(classification_report(y_test, best.predict(X_test)))
    for thresh in [0.6,0.7,0.8,0.9]:
        preds = []
        probs = best.predict_proba(X_test)
        for p in probs:
            cls = best.classes_[np.argmax(p)]
            preds.append(cls if p.max()>=thresh and cls!=0 else 0)
        print(f"-- Threshold {thresh} --")
        print(classification_report(y_test, preds))

    # drawdown backtest
    tracker = EquityTracker(initial_capital=10000, max_dd_pct=OVERALL_DD_PCT)
    for i, p in enumerate(best.predict_proba(X_test)):
        ts = X_test.index[i]
        cls = best.classes_[np.argmax(p)]
        if p.max()>=CONF_THRESHOLD and cls!=0 and tracker.can_trade(ts):
            pnl = 1 if cls==y_test.iloc[i] else -1
            tracker.record(pnl, ts)
    print(f"Final capital: {tracker.capital}")

    dump(best, MODEL_FILE)
    return best

# =========================== #
#       LIVE TRADING LOOP    #
# =========================== #

def live_loop():
    initialize_mt5()
    print(f"=== LIVE MODE for {SYMBOL} @ thr={CONF_THRESHOLD} ===")
    model = load(MODEL_FILE)
    # load the exact feature list you trained on
    try:
        features = load_feature_list()
    except FileNotFoundError:
        features = model.booster_.feature_name()

    tracker = FundingPipsTracker(
        initial_capital=10000,
        daily_dd_pct=DAILY_DD_PCT,
        overall_dd_pct=OVERALL_DD_PCT
    )

    while True:
        ts = datetime.utcnow()

        # 1) H1 features
        df_h1 = fetch_historical_data(SYMBOL, mt5.TIMEFRAME_H1, LOOKBACK_DAYS)
        df1   = enrich_features(df_h1)

        # 2) H4 features (suffix _4h)
        df_h4 = fetch_historical_data(SYMBOL, mt5.TIMEFRAME_H4, LOOKBACK_DAYS)
        df4   = enrich_features(df_h4).add_suffix('_4h')

        # 3) Merge on timestamp
        feat = df1.join(df4.reindex(df1.index), how='inner')

        # 4) Add sentiment & fundamentals
        feat['sentiment']     = get_sentiment()
        funds                 = get_fundamentals()
        feat['interest_rate'] = funds['interest_rate']
        feat['gdp_growth']    = funds['gdp_growth']
        feat['inflation']     = funds['inflation']

        # 5) Select exactly the model’s features
        feat = feat[features].dropna()
        latest = feat.iloc[[-1]]

        # Predict & threshold
        probs = model.predict_proba(latest)[0]
        cls   = model.classes_[np.argmax(probs)]
        p     = probs.max()
        print(f"[{ts:%Y-%m-%d %H:%M:%S}] Pred {cls} @ {p:.2f}")

        # Execute trade if signal passes
        if cls != 0 and p >= CONF_THRESHOLD and tracker.can_trade(ts):
            sl_pips = df1['ATR'].iloc[-1]
            balance = mt5.account_info().balance
            lot     = calculate_lot_size(balance, RISK_PCT, sl_pips)

            tick     = mt5.symbol_info_tick(SYMBOL)
            price    = tick.ask if cls == 1 else tick.bid
            sl_price = price - sl_pips if cls == 1 else price + sl_pips
            tp_price = price + 2*sl_pips if cls == 1 else price - 2*sl_pips
            side     = 'BUY' if cls == 1 else 'SELL'

            print(f"[{ts:%Y-%m-%d %H:%M:%S}] Placing {side} {lot}@{price:.5f} SL={sl_price:.5f} TP={tp_price:.5f}")
            execute_trade(cls)

            # record & log (you may replace dummy PnL with actual execution result)
            pnl = 1 if cls == label_data(df1).iloc[-1]['label'] else -1
            tracker.record(pnl, ts)
            log_trade(ts, SYMBOL, side, price, sl_price, tp_price, pnl, tracker.capital)

        time.sleep(60)


def load_feature_list():
    try:
        with open(FEATURE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        mdl = load(MODEL_FILE)
        return mdl.booster_.feature_name()


def predict_signal(model):
    """
    Fetch H1 and H4 data, engineer features, and predict a trade signal.
    Returns 1 (buy), -1 (sell), or 0 (hold).
    """
    # Fetch and engineer H1 features
    df_h1 = fetch_historical_data(SYMBOL, mt5.TIMEFRAME_H1, LOOKBACK_DAYS)
    df1 = enrich_features(df_h1)
    # Fetch and engineer H4 features
    df_h4 = fetch_historical_data(SYMBOL, mt5.TIMEFRAME_H4, LOOKBACK_DAYS)
    df4 = enrich_features(df_h4).add_suffix('_4h')
    # Align and merge
    df_merged = df1.join(df4.reindex(df1.index), how='inner')
    # Add sentiment and fundamentals
    df_merged['sentiment'] = get_sentiment()
    funds = get_fundamentals()
    df_merged['interest_rate'] = funds['interest_rate']
    df_merged['gdp_growth']    = funds['gdp_growth']
    df_merged['inflation']     = funds['inflation']
    # Load feature list
    feats = load_feature_list()
    # Prepare latest row
    latest = df_merged[feats].dropna().iloc[-1:]
    # Predict
    probs = model.predict_proba(latest)[0]
    cls = model.classes_[np.argmax(probs)]
    # Apply threshold
    return cls if probs.max() >= CONF_THRESHOLD else 0

def execute_trade(signal):
    """
    Place a market order with ATR‐based SL/TP sizing.
    signal:  1 = buy, -1 = sell, 0 = no trade.
    """
    # Fetch account balance
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to fetch account info in execute_trade()")
        return
    balance = account_info.balance

    # Calculate ATR using raw price data (ensure enough history)
    lookback = max(LOOKBACK_DAYS, 15)
    df_hist = fetch_historical_data(SYMBOL, mt5.TIMEFRAME_H1, lookback)
    atr_series = ta.atr(df_hist['high'], df_hist['low'], df_hist['close'], length=14)
    atr = atr_series.dropna().iloc[-1]

    # Determine pip value for the symbol
    info = mt5.symbol_info(SYMBOL)
    contract_size = info.trade_contract_size
    point = info.point
    pip_value = contract_size * point

    # Compute lot size based on risk percentage
    sl_points = atr
    risk_amount = balance * RISK_PCT
    lot = max(info.volume_min, round(risk_amount / (sl_points * pip_value), 2))

    # Determine price, SL, TP and order type
    ticks = mt5.symbol_info_tick(SYMBOL)
    if signal == 1:
        price = ticks.ask
        sl    = price - sl_points
        tp    = price + (2 * sl_points)
        order_type = mt5.ORDER_TYPE_BUY
    elif signal == -1:
        price = ticks.bid
        sl    = price + sl_points
        tp    = price - (2 * sl_points)
        order_type = mt5.ORDER_TYPE_SELL
    else:
        return

    # Build and send the trade request
    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      SYMBOL,
        "volume":      lot,
        "type":        order_type,
        "price":       price,
        "sl":          sl,
        "tp":          tp,
        "deviation":   20,
        "magic":       123456,
        "comment":     "FIST live trade",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling":mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade failed, retcode={result.retcode}")
    else:
        logging.info(f"Trade executed: signal={signal}, lot={lot}, sl={sl}, tp={tp}")



# =========================== #
#      ENTRY POINT / MAIN    #
# =========================== #

if __name__=='__main__':
    print(f"=== FIST Algo starting in {MODE.upper()} mode ===")
    if MODE=='train':
        initialize_mt5(); train_and_tune()
    elif MODE=='live':
        live_loop()
    else:
        print("Unknown MODE: must be 'train' or 'live'", file=sys.stderr)
    mt5.shutdown()
    print("MT5 shutdown complete.")