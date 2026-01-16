
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier

# 設定頁面資訊
st.set_page_config(page_title="AI 量化交易策略實驗室", layout="wide", page_icon="📈")

st.title("🤖 AI 量化交易策略實驗室")
st.markdown("### 2-Stage Gate: VIX Regime + Dynamic Quantile Model")

# --------------------------
# Sidebar: User Inputs
# --------------------------
st.sidebar.header("⚙️ 參數設定")

TARGET_TICKER = st.sidebar.text_input("股票代號 (Yahoo Finance)", value="2330.TW")
YEARS_BACK = st.sidebar.slider("回測年數", min_value=1, max_value=5, value=3)

# 進階參數區
with st.sidebar.expander("進階參數 (Advanced)", expanded=False):
    COST = st.number_input("單邊交易成本 (Cost)", value=0.001, step=0.0005, format="%.4f")
    HOLD_DAYS = st.number_input("持有天數 (Hold Days)", value=3, min_value=1)
    MARKET_MODE = 0  # 固定 Mode 0
    st.info("Market Gate: Mode 0 (VIX < Median)")

run_btn = st.sidebar.button("🚀 開始分析 (Run Analysis)", type="primary")

# --------------------------
# Logic Functions
# --------------------------
@st.cache_data(ttl=3600)
def download_data(ticker, years):
    today = datetime.now()
    start_date = (today - timedelta(days=365*years)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    with st.spinner(f"📥 下載 {ticker} 資料中 ({start_date} ~ {end_date})..."):
        df_target = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df_target.columns, pd.MultiIndex):
            df_target.columns = df_target.columns.get_level_values(0)
            
        tickers_macro = ['^VIX', 'DX-Y.NYB', '^TNX']
        df_macro = yf.download(tickers_macro, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df_macro.columns, pd.MultiIndex):
            df_macro_close = df_macro['Close'].copy()
        else:
            df_macro_close = df_macro.copy()
            
        # Merge
        df = df_target[['Close', 'Volume']].join(df_macro_close, how='left')
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'^VIX': 'VIX', 'DX-Y.NYB': 'DXY', '^TNX': 'US_10Y'}, inplace=True)
        
    return df

def feature_engineering(df):
    df = df.copy()
    # Tiny helper
    def sma(s, n): return s.rolling(n).mean()
    def rsi(close, n=14):
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(n).mean() / loss.rolling(n).mean()
        return 100 - (100 / (1 + rs))

    # Tech
    df['SMA_5'] = sma(df['Close'], 5)
    df['RSI_14'] = rsi(df['Close'], 14)
    df['Mom_3d'] = df['Close'].pct_change(3)

    # Macro
    df['VIX_Chg'] = df['VIX'].pct_change()
    df['VIX_Chg_3d'] = df['VIX'].pct_change(3)
    df['DXY_Chg'] = df['DXY'].pct_change()
    df['US10Y_Chg'] = df['US_10Y'].pct_change()

    # Target Logic
    # Backtest uses this
    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    # Training uses this
    df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = (df['Return_3d'] > 0.003).astype(int)

    # Shift Features
    features = ['SMA_5', 'RSI_14', 'Mom_3d', 'Volume', 'VIX', 'VIX_Chg', 'VIX_Chg_3d', 'DXY', 'DXY_Chg', 'US_10Y', 'US10Y_Chg']
    for c in features:
        df[c] = df[c].shift(1)
        
    df.dropna(inplace=True)
    
    # Regime
    df['VIX_med60'] = df['VIX'].rolling(60).median().shift(1)
    
    return df, features

def run_backtest(df, features, quantiles=[0.50, 0.55, 0.60, 0.65, 0.70]):
    X = df[features]
    y = df['Target']
    
    tscv = TimeSeriesSplit(n_splits=5)
    scan_results = []
    
    # Placeholders for best equity curve
    best_curve = None
    best_q = None
    max_net_equity = -1.0

    progress_bar = st.progress(0)
    total_steps = len(quantiles) * 5
    step_count = 0

    for q in quantiles:
        # Loop Quantiles
        fold_stats = []
        
        # Merge all folds equity for visualization? No, let's just track the last fold or concat?
        # Ideally we want a full out-of-sample curve. 
        # For simplicity in this UI, let's concatenate the OOS parts of each fold to form a continuous backtest.
        
        oos_equity_segments = []
        
        # 參數內的 Cross Val
        model = HistGradientBoostingClassifier(random_state=42)
        
        full_signals = []
        full_dates = []
        
        for train_idx, test_idx in tscv.split(X):
            step_count += 1
            progress_bar.progress(step_count / total_steps)
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            
            # Dynamic Threshold Calculation
            lookback = 252
            if len(X_train) > lookback:
                proba_ctx = model.predict_proba(X_train.iloc[-lookback:])[:, 1]
            else:
                proba_ctx = model.predict_proba(X_train)[:, 1]
            
            proba_test = model.predict_proba(X_test)[:, 1]
            
            # Concat for quantile
            proba_full = np.concatenate([proba_ctx, proba_test])
            rolling_thresh = pd.Series(proba_full).rolling(lookback, min_periods=1).quantile(q)
            thresh_test = rolling_thresh.iloc[-len(proba_test):].values
            
            # Backtest
            bt_fold = df.iloc[test_idx].copy()
            bt_fold['proba_up'] = proba_test
            bt_fold['dyn_thresh'] = thresh_test
            
            # Gates
            bt_fold['market_ok'] = (bt_fold['VIX'] < bt_fold['VIX_med60']).astype(int)
            bt_fold['model_ok'] = (bt_fold['proba_up'] >= bt_fold['dyn_thresh']).astype(int)
            bt_fold['trade_allowed'] = (bt_fold['market_ok'] & bt_fold['model_ok']).astype(int)
            
            # State Machine
            hold_count = 0
            strat_rets = np.zeros(len(bt_fold))
            signals = np.zeros(len(bt_fold))
            
            next_rets = bt_fold['Next_Return'].values
            allowed = bt_fold['trade_allowed'].values
            
            for i in range(len(bt_fold)):
                if hold_count > 0:
                    strat_rets[i] = next_rets[i]
                    hold_count -= 1
                else:
                    if allowed[i] == 1:
                        strat_rets[i] = next_rets[i] - COST
                        signals[i] = 1
                        hold_count = HOLD_DAYS - 1
            
            # Collect metrics
            bt_fold['strat_ret'] = strat_rets
            
            # Append OOS results for this fold
            oos_equity_segments.append(bt_fold[['strat_ret']])
            if trades := signals.sum():
                 pass # simplified metrics for UI
                 
        # Stitch folds together to make a "Walk-Forward" Equity Curve
        oos_df = pd.concat(oos_equity_segments)
        oos_df.sort_index(inplace=True)
        # Handle overlaps if any (TSC doesn't overlap test sets usually)
        oos_df = oos_df[~oos_df.index.duplicated(keep='first')]
        
        oos_df['equity'] = (1 + oos_df['strat_ret']).cumprod()
        oos_df['benchmark'] = (1 + df.loc[oos_df.index, 'Next_Return']).cumprod()
        
        final_eq = oos_df['equity'].iloc[-1]
        
        # Metrics
        total_trades = 0 # Need to recalc
        # Re-run logic on full stitched? No, just sum
        # Ideally calculate metrics on the stitched curve
        
        dd = oos_df['equity'] / oos_df['equity'].cummax() - 1
        mdd = dd.min()
        
        scan_results.append({
            "Quantile": q,
            "Net Equity": final_eq,
            "MaxDD": mdd
        })
        
        if final_eq > max_net_equity:
            max_net_equity = final_eq
            best_curve = oos_df
            best_q = q

    progress_bar.empty()
    return pd.DataFrame(scan_results), best_curve, best_q

# --------------------------
# Main App
# --------------------------
if run_btn:
    # 1. Download
    raw_df = download_data(TARGET_TICKER, YEARS_BACK)
    
    if raw_df is None or raw_df.empty:
        st.error(f"❌ 下載失敗：無法取得 {TARGET_TICKER} 資料。請檢查代號或網路。")
        st.stop()
        
    # Check Macro columns
    if 'VIX' not in raw_df.columns or raw_df['VIX'].isnull().all():
        st.warning("⚠️ 警告：VIX 數據遺失 (全為 NaN)。這將導致 DropNA 後資料全空。")
        # Optional: display raw head for debug
        st.write("Raw Data Head:", raw_df.head())
    
    st.success(f"資料下載完成！共 {len(raw_df)} 筆交易日。")
    
    # 2. Features
    df_feat, feature_names = feature_engineering(raw_df)
    
    if df_feat.empty:
        st.error("❌ 錯誤：特徵工程後資料為空。可能原因：\n1. 宏觀數據(VIX)對齊失敗導致全被 Drop \n2. 資料長度不足以計算 60 日均線。")
        st.stop()

    if len(df_feat) < 50:
        st.error(f"❌ 樣本數不足 ({len(df_feat)})，無法進行 TimeSeriesSplit。請嘗試拉長回測年數。")
        st.stop()
    
    # Show Data Preview
    with st.expander("數據預覽 (Data Preview)"):
        st.dataframe(df_feat.tail(10))
        st.caption("最近 10 筆數據 (包含特徵)")
    
    # 3. Model & Backtest
    st.write("🏃‍♂️ 正在執行 Walk-Forward Validation 與參數掃描...")
    res_df, best_curve_df, best_q = run_backtest(df_feat, feature_names)
    
    # 4. Results
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 參數掃描結果")
        st.dataframe(res_df.style.format({
            "Net Equity": "{:.3f}x",
            "MaxDD": "{:.2%}"
        }).highlight_max(subset=["Net Equity"], color="lightgreen"))
        
        st.info(f"🏆 最佳 Quantile: {best_q}")
    
    with col2:
        st.subheader("📈 最佳策略資金曲線 (Walk-Forward)")
        if best_curve_df is not None:
            # Plot
            fig = px.line(best_curve_df, y=['equity', 'benchmark'], 
                          title=f"Strategy (Q={best_q}) vs Benchmark (Buy&Hold)",
                          color_discrete_map={"equity": "green", "benchmark": "gray"})
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Strategy Final Equity", f"{best_curve_df['equity'].iloc[-1]:.3f}x")
            st.metric("Benchmark Final Equity", f"{best_curve_df['benchmark'].iloc[-1]:.3f}x")
            
    # 5. Signal for Today (Actionable)
    st.markdown("---")
    st.subheader("🔮 今日訊號 (最新預測)")
    
    # Retrain on FULL Data to get today's signal
    last_row = df_feat.iloc[[-1]] 
    # Use full data to train
    X_full = df_feat[feature_names]
    y_full = df_feat['Target']
    
    final_model = HistGradientBoostingClassifier(random_state=42)
    final_model.fit(X_full, y_full)
    
    # Predict on the *latest available feature set* (which is derived from Yesterday's Close to predict Today/Tomorrow)
    # Actually, main.py uses lag=1. So today's input (Close_t) predicts Return_t+1~t+3? 
    # Logic: Features are shift(1). So row T contains features from T-1.
    # We want to predict for T. We need features from T-1. 
    # df_feat already has shifted features. So the last row of df_feat contains features known at T-1 (yesterday close).
    # This prediction is valid for 'Today'.
    
    # Wait, we need to know if today is a Trading Day or if we are post-close.
    # Assuming standard usage: User runs this AFTER market close to get signal for TOMORROW? 
    # Or DURING market? 
    # Let's just output the "Latest Prediction" based on "Latest Data".
    
    latest_proba = final_model.predict_proba(last_row[feature_names])[:, 1][0]
    
    # Calculate current Dynamic Threshold (using last 252 days of full data)
    proba_history = final_model.predict_proba(X_full)[:, 1]
    current_thresh = pd.Series(proba_history).rolling(252).quantile(best_q).iloc[-1]
    
    # Market Gate
    latest_vix = df_feat['VIX'].iloc[-1]
    vix_med = df_feat['VIX_med60'].iloc[-1]
    market_ok = latest_vix < vix_med
    
    model_ok = latest_proba >= current_thresh
    
    c1, c2, c3 = st.columns(3)
    c1.metric("模型信心 (Proba)", f"{latest_proba:.1%}")
    c1.caption(f"門檻值: {current_thresh:.1%}")
    
    c2.metric("市場狀態 (VIX)", f"{latest_vix:.2f}", delta=f"{latest_vix - vix_med:.2f} vs Med", delta_color="inverse")
    c2.caption(f"VIX Med60: {vix_med:.2f}")
    
    c3.metric("最終決策", 
              "✅ ALLOWED" if (market_ok and model_ok) else "🛑 REJECTED",
              delta="Buy Signal" if (market_ok and model_ok) else "Wait",
              delta_color="normal" if (market_ok and model_ok) else "off")
    
else:
    st.info("👈 請在左側設定參數並點擊 '開始分析' 來執行策略。")
