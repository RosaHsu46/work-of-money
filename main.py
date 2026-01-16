
# -*- coding: utf-8 -*-
# ==========================================
# MVP 2.3：2330.TW + VIX/DXY/US10Y/SOX/SP500/TWII
# - 不用 pandas_ta
# - 特徵全部 lag=1（避免資訊洩漏）
# - Target：Return_3d > 0.3% (3日累積報酬)
# - TimeSeriesSplit + 每一折詳細報告
# - 用 predict_proba 做交易門檻 + 簡易回測(含成本)
# ==========================================

# !pip -q install yfinance scikit-learn pandas numpy
from datetime import datetime, timedelta

# --------------------------
# CONFIG
# --------------------------
TARGET_TICKER = "2330.TW"  # 可以在這裡修改股票代號 (例如 "NVDA", "AAPL", "0050.TW")

# 自動設定時間：今天往回推 3 年
today = datetime.now()
three_years_ago = today - timedelta(days=365*3)

START_DATE = three_years_ago.strftime("%Y-%m-%d")
END_DATE = today.strftime("%Y-%m-%d")

print(f"分析區間：{START_DATE} ~ {END_DATE}")

import yfinance as yf
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Force UTF-8 output for Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------
# 0) 小工具：技術指標
# --------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --------------------------
# 1) 下載資料
# --------------------------
print("📥 下載資料中...")

df_tw = yf.download(TARGET_TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
if isinstance(df_tw.columns, pd.MultiIndex):
    df_tw.columns = df_tw.columns.get_level_values(0)

df_tw = df_tw[['Close', 'Volume']].copy()
df_tw.index = pd.to_datetime(df_tw.index).tz_localize(None)

tickers_macro = ['^VIX', 'DX-Y.NYB', '^TNX', '^SOX', '^GSPC', '^TWII']
df_macro = yf.download(tickers_macro, start=START_DATE, end=END_DATE, auto_adjust=True)

# yfinance 多標的會變成多層欄位
if isinstance(df_macro.columns, pd.MultiIndex):
    df_macro_close = df_macro['Close'].copy()
else:
    # 極少見情況：若沒有 MultiIndex，退一步處理
    df_macro_close = df_macro.copy()

df_macro_close.index = pd.to_datetime(df_macro_close.index).tz_localize(None)

# --------------------------
# 2) 對齊合併（以台股交易日為主）
# --------------------------
print("🛠️ 正在對齊台美股資料...")

df = df_tw.join(df_macro_close, how='left')
df.ffill(inplace=True)
df.dropna(inplace=True)

df.rename(columns={
    '^VIX': 'VIX', 
    'DX-Y.NYB': 'DXY', 
    '^TNX': 'US_10Y',
    '^SOX': 'SOX',
    '^GSPC': 'SP500',
    '^TWII': 'TWII'
}, inplace=True)


# --------------------------
# 3) 特徵工程
# --------------------------
# 技術面
df['SMA_5'] = sma(df['Close'], 5)
df['RSI_14'] = rsi(df['Close'], 14)
df['Mom_3d'] = df['Close'].pct_change(3)  # [NEW] 3日動能

# 宏觀變化率（情緒「變動」往往比絕對值更有用）
df['VIX_Chg'] = df['VIX'].pct_change()
df['VIX_Chg_3d'] = df['VIX'].pct_change(3)  # [NEW] 3日 VIX 變化
df['DXY_Chg'] = df['DXY'].pct_change()
df['US10Y_Chg'] = df['US_10Y'].pct_change()
df['SOX_Chg'] = df['SOX'].pct_change()
df['SP500_Chg'] = df['SP500'].pct_change()
df['TWII_Chg'] = df['TWII'].pct_change()


# --------------------------
# 4) Target：未來 3 日累積報酬 > 0.3%
# --------------------------
# 用於【Backtest】：計算每日報酬 (Hold 1 day)
df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1

# 用於【Target】：預測未來 3 日漲幅
df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1

threshold = 0.003
df['Target'] = (df['Return_3d'] > threshold).astype(int)

# --------------------------
# 5) 🔥避免資訊洩漏：所有特徵 lag 1 天
# --------------------------
features = [
    'SMA_5', 'RSI_14', 'Mom_3d', 'Volume', 
    'VIX', 'VIX_Chg', 'VIX_Chg_3d', 
    'DXY', 'DXY_Chg', 
    'US_10Y', 'US10Y_Chg',
    'SOX_Chg', 'SP500_Chg', 'TWII_Chg'
]
for c in features:
    df[c] = df[c].shift(1)

df.dropna(inplace=True)

# --------------------------
# A1) 市場狀態標記（Regime Labels）- 在分割前計算以避免 Look-ahead bias
# --------------------------
# 先算好 Regime（用全歷史當尺）
df['VIX_med60'] = df['VIX'].rolling(60).median().shift(1)
df['VIX_high']  = (df['VIX'] > df['VIX_med60']).astype(int)
df['Rate_up']   = (df['US10Y_Chg'] > 0).astype(int)
df['Risk_on']   = ((df['VIX'] < df['VIX_med60']) & (df['DXY_Chg'] <= 0)).astype(int)

X = df[features]
y = df['Target']

# --------------------------
# 6) 模型訓練與策略掃描 (Dynamic Quantile & Daily Backtest)
# --------------------------
print("\n🤖 開始模型訓練與策略掃描 (Dynamic Quantile)...")

tscv = TimeSeriesSplit(n_splits=5)
scan_results = []

# 固定使用 Market Mode 0 (VIX < Med)
MARKET_MODE = 0
# 掃描動態門檻的分位數 (Quantile)
quantiles = [0.50, 0.55, 0.60, 0.65, 0.70]

COST = 0.001

for q in quantiles:
    # Initialize model
    model = HistGradientBoostingClassifier(random_state=42)
    fold_stats = []
    
    # Debug vars for Fold 5
    f5_trades = 0
    f5_net = 1.0
    f5_mdd = 0.0
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        
        # 1. 計算動態門檻 (Rolling Quantile)
        # 先預測 Train 的一部分 (用最近 252 天做為 Context)
        lookback = 252
        if len(X_train) > lookback:
            X_ctx = X_train.iloc[-lookback:]
            proba_ctx = model.predict_proba(X_ctx)[:, 1]
        else:
            proba_ctx = model.predict_proba(X_train)[:, 1]
            
        proba_test = model.predict_proba(X_test)[:, 1]
        
        # 串接 Context + Test
        proba_full = np.concatenate([proba_ctx, proba_test])
        proba_series = pd.Series(proba_full)
        
        # 計算 Rolling Quantile (min_periods=1 確保初期有值)
        rolling_thresh = proba_series.rolling(window=lookback, min_periods=1).quantile(q)
        
        # 取出對應 Test 段的 Threshold
        thresh_test = rolling_thresh.iloc[-len(proba_test):].values
        
        # --------------------------
        # 2. Backtest Logic (Daily Accumulation)
        # --------------------------
        bt = df.iloc[test_index].copy()
        bt['proba_up'] = proba_test
        bt['dyn_thresh'] = thresh_test
        
        # (A) 市場 Gate (Mode 0: VIX Only)
        bt['market_ok'] = (bt['VIX'] < bt['VIX_med60']).astype(int)
        
        # (B) 模型 Gate (Dynamic)
        bt['model_ok'] = (bt['proba_up'] >= bt['dyn_thresh']).astype(int)
        
        # (C) Intersection
        bt['trade_allowed'] = (bt['market_ok'] & bt['model_ok']).astype(int)
        
        # State Machine (Daily Equity)
        HOLD_DAYS = 3
        hold_count = 0
        
        strategy_rets = np.zeros(len(bt))
        signals = np.zeros(len(bt)) # 1=Entry
        
        next_ret_values = bt['Next_Return'].values
        trade_allowed_values = bt['trade_allowed'].values
        
        for i in range(len(bt)):
            if hold_count > 0:
                # 持有期，吃 Next_Return
                strategy_rets[i] = next_ret_values[i]
                hold_count -= 1
            else:
                if trade_allowed_values[i] == 1:
                    # 進場，扣成本，開始持有
                    strategy_rets[i] = next_ret_values[i] - COST
                    signals[i] = 1
                    hold_count = HOLD_DAYS - 1
                else:
                    strategy_rets[i] = 0.0
        
        bt['signal'] = signals
        bt['strat_daily_ret'] = strategy_rets
        
        # 計算淨值
        bt['equity'] = (1 + bt['strat_daily_ret']).cumprod()
        
        # Metrics
        final_equity = bt['equity'].iloc[-1]
        trades = int(signals.sum())
        
        # Max Drawdown
        roll_max = bt['equity'].cummax()
        drawdown = bt['equity'] / roll_max - 1.0
        max_dd = drawdown.min()
        
        # Win Rate (based on Entry 3-day hold roughly, for stats)
        # 這裡從簡：只看 trade entry 那個當下的 Return_3d 是否 > 0 (作為本次交易勝率參考)
        if trades > 0:
            entries = bt[bt['signal']==1]
            win_rate = (entries['Return_3d'] > 0).mean()
        else:
            win_rate = 0.0
            
        fold_stats.append({
            'net_equity': final_equity,
            'trades': trades,
            'max_dd': max_dd,
            'win_rate': win_rate
        })
        
        if fold == 5:
            f5_trades = trades
            f5_net = final_equity
            f5_mdd = max_dd
        
        fold += 1
        
    # Aggregate Stats per Quantile
    df_stats = pd.DataFrame(fold_stats)
    
    avg_net = df_stats['net_equity'].mean()
    avg_trades = df_stats['trades'].mean()
    avg_mdd = df_stats['max_dd'].mean()
    avg_win = df_stats[df_stats['trades']>0]['win_rate'].mean() if len(df_stats[df_stats['trades']>0]) > 0 else 0.0
    
    scan_results.append({
        'Quantile': q,
        'AvgNet': avg_net,
        'AvgTrades': avg_trades,
        'MaxDD': avg_mdd,
        'WinRate': avg_win,
        'Fold5Trades': f5_trades,
        'Fold5Net': f5_net, 
        'Fold5DD': f5_mdd
    })
    
    print(f"   Done: Q={q} -> AvgNet={avg_net:.3f}x, AvgTrades={avg_trades:.1f}")

# --------------------------
# 7) 產出選參數報表 (Scoring & Ranking)
# --------------------------
res_final = pd.DataFrame(scan_results)

print("\n📊 --- Strategy Scan Summary (Market Mode=0) ---")

cols = ['Quantile', 'AvgNet', 'AvgTrades', 'MaxDD', 'WinRate', 'Fold5Trades', 'Fold5Net']
print(res_final[cols].to_string(index=False, formatters={
    'AvgNet': '{:.3f}'.format,
    'AvgTrades': '{:.1f}'.format,
    'MaxDD': '{:.1%}'.format,
    'WinRate': '{:.1%}'.format,
    'Fold5Trades': '{:}'.format,
    'Fold5Net': '{:.3f}'.format
}))

# Scoring Logic
# 1. 篩選：AvgTrades >= 2 且 Fold5Trades > 0
candidates = res_final[
    (res_final['AvgTrades'] >= 2.0) & 
    (res_final['Fold5Trades'] > 0)
].copy()

print("\n🏆 --- Best Parameter Selection ---")

if len(candidates) > 0:
    # 2. 排序：按 AvgNet 降冪
    best = candidates.sort_values('AvgNet', ascending=False).iloc[0]
    
    print(f"🥇 Winner: Quantile = {best['Quantile']:.2f}")
    print(f"   - Avg Net Equity: {best['AvgNet']:.3f}x")
    print(f"   - Avg Trades:     {best['AvgTrades']:.1f}")
    print(f"   - Avg Win Rate:   {best['WinRate']:.1%}")
    print(f"   - Avg MaxDD:      {best['MaxDD']:.1%}")
    print(f"   - Fold 5 Stats:   {int(best['Fold5Trades'])} trades, {best['Fold5Net']:.3f}x")
    print("\nReason: Validated by volume (AvgTrades>=2) and recent activity (Fold5>0), then sorted by profitability.")
else:
    print("⚠️ No parameter met the strict criteria (AvgTrades>=2 & Fold5>0).")
    print("Top by Profitability:")
    best = res_final.sort_values('AvgNet', ascending=False).iloc[0]
    print(f"Q={best['Quantile']:.2f}, Net={best['AvgNet']:.3f}x")

# Reset
pd.reset_option('display.float_format')
