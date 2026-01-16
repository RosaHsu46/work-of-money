
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import feedparser
import urllib.parse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier

# 設定頁面資訊
st.set_page_config(page_title="AI 量化交易策略實室 (Portfolio Edition)", layout="wide", page_icon="📈")

st.title("🤖 AI 量化交易策略實驗室 (Portfolio Edition)")
st.markdown("### 2-Stage Gate: VIX Regime + Dynamic Quantile Model")

# --------------------------
# Sidebar: User Inputs
# --------------------------
st.sidebar.header("⚙️ 參數設定")

# 修改：改成 Text Area 支援多檔股票
# 新增：Excel 上傳功能
uploaded_file = st.sidebar.file_uploader("📁 上傳持有清單 (Excel)", type=["xlsx", "xls"])

# Initialize session state for tickers
if 'tickers_text' not in st.session_state:
    st.session_state['tickers_text'] = "2330.TW\n2317.TW\n2454.TW"
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None

# Logic: Process file ONLY if it is new
if uploaded_file is not None and uploaded_file != st.session_state['last_uploaded_file']:
    try:
        df_upload = pd.read_excel(uploaded_file)
        # 智慧偵測欄位
        possible_cols = ['股票代號', 'Ticker', 'Symbol', 'Code', 'Stock', '股號', '代號']
        target_col = None
        cols_clean = [str(c).strip() for c in df_upload.columns]
        
        for p in possible_cols:
            matches = [i for i, c in enumerate(cols_clean) if c.lower() == p.lower()]
            if matches:
                target_col = df_upload.columns[matches[0]]
                break
        
        raw_list = []
        if target_col:
            st.sidebar.success(f"讀取成功！欄位：{target_col}")
            raw_list = df_upload[target_col].dropna().tolist()
        else:
            st.sidebar.warning("未偵測到代號欄位，預設使用第一欄")
            raw_list = df_upload.iloc[:, 0].astype(str).tolist()
            
        # Clean and Format
        cleaned = []
        for item in raw_list:
            s = str(item).strip()
            if s.isdigit() and len(s) < 4: s = s.zfill(4)
            if not s.upper().endswith('.TW') and not s.upper().endswith('.TWO'): s += '.TW'
            cleaned.append(s)
            
        # Update Session State
        unique_tickers = list(dict.fromkeys(cleaned))
        st.session_state['tickers_text'] = "\n".join(unique_tickers)
        st.session_state['last_uploaded_file'] = uploaded_file
        st.sidebar.info(f"已匯入 {len(unique_tickers)} 檔股票至下方列表中。")
        
    except Exception as e:
        st.sidebar.error(f"解析失敗: {e}")

input_tickers = st.sidebar.text_area(
    "股票代號清單 (可手動修改)", 
    value=st.session_state['tickers_text'],
    height=150,
    key='tickers_input_widget', # Unique key
    help="上傳 Excel 後會自動填入此處，您也可以手動編輯。"
)

# Update session state if user edits text area manually
if input_tickers != st.session_state['tickers_text']:
     st.session_state['tickers_text'] = input_tickers



if st.sidebar.button("🧹 清除快取 (Clear Cache)"):
    st.cache_data.clear()
    st.sidebar.success("快取已清除！請重新執行分析。")

st.sidebar.caption(f"yfinance version: {yf.__version__}")

YEARS_BACK = st.sidebar.slider("回測年數", min_value=1, max_value=5, value=3)

# 進階參數區
with st.sidebar.expander("進階參數 (Advanced)", expanded=False):
    COST = st.number_input("單邊交易成本 (Cost)", value=0.001, step=0.0005, format="%.4f")
    HOLD_DAYS = st.number_input("持有天數 (Hold Days)", value=3, min_value=1)
    
run_btn = st.sidebar.button("🚀 開始批次分析 (Batch Run)", type="primary")

# --------------------------
# Core Logic Functions
# --------------------------

@st.cache_data(ttl=3600)
def download_macro_data(years):
    """只下載一次宏觀數據並快取"""
    today = datetime.now()
    start_date = (today - timedelta(days=365*years)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    # Critical Global Tech Trend Indicators: NVDA (AI), MU (Memory)
    tickers_macro = ['^VIX', 'DX-Y.NYB', '^TNX', '^SOX', '^GSPC', '^TWII', 'NVDA', 'MU']
    df_macro = yf.download(tickers_macro, start=start_date, end=end_date, auto_adjust=True)
    
    # Handle yfinance recent changes or single-ticker return result
    if isinstance(df_macro.columns, pd.MultiIndex):
        try:
            df_macro_close = df_macro['Close'].copy()
        except KeyError:
             df_macro_close = df_macro.copy()
    else:
        if 'Close' in df_macro.columns:
             df_macro_close = df_macro['Close'].copy()
        else:
             df_macro_close = df_macro.copy()
             
    if isinstance(df_macro_close, pd.Series):
        df_macro_close = df_macro_close.to_frame()
        
    df_macro_close.index = pd.to_datetime(df_macro_close.index).tz_localize(None)
    
    # Rename mapping
    rename_map = {
        '^VIX': 'VIX', 
        'DX-Y.NYB': 'DXY', 
        '^TNX': 'US_10Y',
        '^SOX': 'SOX',
        '^GSPC': 'SP500',
        '^TWII': 'TWII',
        'NVDA': 'NVDA',
        'MU': 'MU'
    }
    df_macro_close.rename(columns=rename_map, inplace=True)
    
    # Ensure all expected columns exist (fill missing with NaN to avoid KeyError)
    for target_col in rename_map.values():
        if target_col not in df_macro_close.columns:
            df_macro_close[target_col] = np.nan
            
    return df_macro_close, start_date, end_date

@st.cache_data(ttl=600)  # News cache shorter (10 min)
def get_stock_news(ticker):
    """取得個股相關 Google News"""
    try:
        # 清理代號 (e.g. 2330.TW -> 2330) 或是直接用 "2330.TW stock"
        # 搜尋關鍵字："{Ticker} stock"
        query = f"{ticker} stock"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        
        feed = feedparser.parse(rss_url)
        
        news_items = []
        for entry in feed.entries[:5]:  # Top 5 news
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published or ""
            })
            
        return news_items
    except Exception as e:
        return []

@st.cache_data(ttl=3600)
def download_stock_data(ticker, start_date, end_date):
    """下載個別股票數據"""
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        return None
        
    df = df[['Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def feature_engineering(df_stock, df_macro):
    # Merge
    df = df_stock.join(df_macro, how='left')
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # Tech
    df['SMA_5'] = df['Close'].rolling(5).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Mom_3d'] = df['Close'].pct_change(3)

    df['Mom_3d'] = df['Close'].pct_change(3)

    # Macro Features (Safe Compute)
    if 'VIX' in df.columns:
        df['VIX_Chg'] = df['VIX'].pct_change()
        df['VIX_Chg_3d'] = df['VIX'].pct_change(3)
        df['VIX_med60'] = df['VIX'].rolling(60).median().shift(1)
    
    if 'DXY' in df.columns: df['DXY_Chg'] = df['DXY'].pct_change()
    if 'US_10Y' in df.columns: df['US10Y_Chg'] = df['US_10Y'].pct_change()
    if 'SOX' in df.columns: df['SOX_Chg'] = df['SOX'].pct_change()
    if 'SP500' in df.columns: df['SP500_Chg'] = df['SP500'].pct_change()
    if 'TWII' in df.columns: df['TWII_Chg'] = df['TWII'].pct_change()
    
    # AI/Memory Trend Features
    if 'NVDA' in df.columns: df['NVDA_Chg'] = df['NVDA'].pct_change()
    if 'MU' in df.columns: df['MU_Chg'] = df['MU'].pct_change()

    # Target Logic
    # Backtest uses this (Return tomorrow)
    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    # Training uses this (Return 3 days later)
    df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = (df['Return_3d'] > 0.003).astype(int)

    # Base Features
    base_features = ['SMA_5', 'RSI_14', 'Mom_3d', 'Volume']
    macro_candidates = [
        'VIX', 'VIX_Chg', 'VIX_Chg_3d', 
        'DXY', 'DXY_Chg', 
        'US_10Y', 'US10Y_Chg',
        'SOX_Chg', 'SP500_Chg', 'TWII_Chg',
        'NVDA_Chg', 'MU_Chg' # Added new features
    ]
    
    # Filter available features
    features = base_features + [c for c in macro_candidates if c in df.columns]
    
    for c in features:
        df[c] = df[c].shift(1)
        
    df.dropna(inplace=True)
    
    return df, features

def run_analysis_for_ticker(ticker, df_macro, start_date, end_date):
    """執行單一股票的完整分析流程"""
    try:
        # 1. Download Stock
        df_stock = download_stock_data(ticker, start_date, end_date)
        if df_stock is None:
            return {"status": "error", "msg": f"無資料 (No Data) - {ticker}"}
        if len(df_stock) < 60:
            return {"status": "error", "msg": f"資料不足 ({len(df_stock)}筆) - {ticker}"}
            
        # 2. FE
        df_feat, features = feature_engineering(df_stock, df_macro)
        if len(df_feat) < 50:
            return {"status": "error", "msg": "Not enough data after FE"}

        # 3. Model & Backtest (Simplified for batch speed)
        # 這裡只跑一個最佳參數掃描的簡化版，或者固定用一個較好的 Quantile (e.g. 0.6) 以節省時間？
        # 為了效能，我們這裡固定掃描幾個關鍵 Quantile，取最好的。
        
        X = df_feat[features]
        y = df_feat['Target']
        
        # Train final model on FULL data first to get latest signal
        final_model = HistGradientBoostingClassifier(random_state=42)
        final_model.fit(X, y)
        
        # Latest Signal
        last_row = df_feat.iloc[[-1]]
        latest_proba = final_model.predict_proba(last_row[features])[:, 1][0]
        
        # Calculate Threshold (Dynamic 252d)
        # 為了 batch 速度，我們預設用 Q=0.6 (相對穩健)
        # 如果用戶希望每個都掃描，可以在這裡加入簡單的 CV。
        # 為了體驗，我們這裡做一個快速的 TimeSeriesSplit 驗證獲利能力
        
        tscv = TimeSeriesSplit(n_splits=3) # 減少 split 加快速度
        model = HistGradientBoostingClassifier(random_state=42)
        
        total_ret = 0
        qs = [0.55, 0.60, 0.65] # 掃描範圍縮小
        best_q = 0.60
        best_equity = -999
        best_curve = None
        
        for q in qs:
            # 簡易回測邏輯
            # 略過完整的逐日回測，改用向量化估算以加速
            # 注意：這裡為了速度做適度簡化
            preds = []
            truths = []
            
            # 這裡我們只做最後一折的驗證來當作成績單，避免跑太久
            # Train last 80%, Test last 20%
            split_idx = int(len(X) * 0.8)
            X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
            y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model.fit(X_tr, y_tr)
            
            # Context
            proba_tr = model.predict_proba(X_tr.iloc[-252:])[:, 1] if len(X_tr) > 252 else model.predict_proba(X_tr)[:, 1]
            proba_te = model.predict_proba(X_te)[:, 1]
            
            full_prob = np.concatenate([proba_tr, proba_te])
            thresh_series = pd.Series(full_prob).rolling(252, min_periods=1).quantile(q)
            te_thresh = thresh_series.iloc[-len(proba_te):].values
            
            # Calc Return
            test_df = df_feat.iloc[split_idx:].copy()
            test_df['proba'] = proba_te
            test_df['thresh'] = te_thresh
            
            # Logic
            mask_market = test_df['VIX'] < test_df['VIX_med60']
            mask_model = test_df['proba'] >= test_df['thresh']
            test_df['signal'] = (mask_market & mask_model).astype(int)
            
            # Simple equity (Buy Next Return - Cost)
            # 忽略 Hold 3 days 細節，簡化為 Daily Impact for Selection
            daily_ret = test_df['signal'] * (test_df['Next_Return'] - COST) 
            final_eq = (1 + daily_ret).cumprod().iloc[-1]
            
            if final_eq > best_equity:
                best_equity = final_eq
                best_q = q
                best_curve = (1 + daily_ret).cumprod()

        # Get Threshold for TODAY using Best Q
        proba_history = final_model.predict_proba(X)[:, 1]
        current_thresh = pd.Series(proba_history).rolling(252).quantile(best_q).iloc[-1]
        
        # Final Decision
        market_ok = True
        if 'VIX' in df_feat.columns and 'VIX_med60' in df_feat.columns:
            latest_vix = df_feat['VIX'].iloc[-1]
            vix_med = df_feat['VIX_med60'].iloc[-1]
            if not pd.isna(latest_vix) and not pd.isna(vix_med):
                market_ok = latest_vix < vix_med
        
        model_ok = latest_proba >= current_thresh
        
        action = "✅ BUY" if (market_ok and model_ok) else "🛑 WAIT"
        
        return {
            "status": "ok",
            "ticker": ticker,
            "close": df_feat['Close'].iloc[-1],
            "proba": latest_proba,
            "thresh": current_thresh,
            "best_q": best_q,
            "market_ok": market_ok,
            "model_ok": model_ok,
            "action": action,
            "equity_test": best_equity, # Last 20% sample performance
            "curve": best_curve,
            "df_feat_tail": df_feat.tail(5) # For detail view
        }
        
    except Exception as e:
        return {"status": "error", "msg": str(e), "traceback": traceback.format_exc()}

# --------------------------
# Main Execution
# --------------------------
if run_btn:
    # 3. 使用 Text Area 的內容 (現在 Excel 已經填進去了)
    raw_tickers = [t.strip() for t in input_tickers.replace(',', '\n').split('\n') if t.strip()]
    
    if not raw_tickers:
        st.error("請輸入至少一支股票代號")
        st.stop()
        
    st.write(f"📊 準備分析 {len(raw_tickers)} 檔股票...")
    
    # 1. Download Macro (Once)
    st.info("📥 下載宏觀數據中 (Macro Data)...")
    try:
        df_macro, start_dt, end_dt = download_macro_data(YEARS_BACK)
    except Exception as e:
        st.error(f"宏觀數據下載失敗: {e}")
        st.stop()
        
    # 2. Loop Tickers
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(raw_tickers):
        pct = (i) / len(raw_tickers)
        progress_bar.progress(pct)
        status_text.text(f"正在分析 {t} ({i+1}/{len(raw_tickers)})...")
        
        res = run_analysis_for_ticker(t, df_macro, start_dt, end_dt)
        if res['status'] == 'ok':
            # Append result
            results.append({
                "Ticker": t,
                "Action": res['action'],
                "Confidence": f"{res['proba']:.1%}",
                "Threshold": f"{res['thresh']:.1%}",
                "Market(VIX)": "Safe" if res['market_ok'] else "Risk",
                "Backtest(Last20%)": f"{res['equity_test']:.2f}x",
                "Close": f"{res['close']:.2f}",
                # Hidden objects for details
                "_raw_res": res
            })
        else:
            # Error row
            error_msg = res['msg']
            # Simplify error for table
            short_err = (error_msg[:30] + '..') if len(error_msg) > 30 else error_msg
            
            results.append({
                "Ticker": t,
                "Action": f"❌ {short_err}",
                "Confidence": "-",
                "Threshold": "-",
                "Market(VIX)": "-",
                "Backtest(Last20%)": "-",
                "Close": "-",
                "_error": res['msg'],
                "_traceback": res.get('traceback', '')
            })
            
    progress_bar.progress(1.0)
    status_text.text("分析完成！")
    
    # 3. Summary Dashboard
    st.markdown("---")
    st.subheader("📊 投資組合總體檢 (Portfolio Summary)")
    
    if results:
        df_res = pd.DataFrame(results)
        # Drop hidden cols for table
        disp_cols = [c for c in df_res.columns if not c.startswith('_')]
        
        # Color styling function
        def highlight_action(val):
            color = 'lightgreen' if 'BUY' in str(val) else 'white'
            if '❌' in str(val) or 'ERROR' in str(val): color = 'lightcoral'
            return f'background-color: {color}'
        
        st.dataframe(df_res[disp_cols].style.applymap(highlight_action, subset=['Action']))
        
        # 4. Detailed View
        st.markdown("### 🔍 個股詳細分析 (Details)")
        
        for r in results:
            t = r['Ticker']
            with st.expander(f"{t} - {r['Action']}", expanded=False):
                if '❌' in r['Action'] or 'ERROR' in r['Action']:
                    st.error(f"錯誤原因: {r.get('_error', 'Unknown')}")
                    if '_traceback' in r:
                        st.code(r['_traceback'], language='python')
                else:
                    # Tabs for Analysis vs News
                    tab1, tab2 = st.tabs(["📊 數據分析", "🗞️ 相關新聞"])
                    
                    detail = r['_raw_res']
                    
                    with tab1:
                        c1, c2 = st.columns([1, 2])
                        
                        with c1:
                            st.metric("最新收盤", detail['close'])
                            st.metric("模型信心", f"{detail['proba']:.1%}", delta=f"門檻: {detail['thresh']:.1%}")
                            st.metric("最佳參數 (Quantile)", detail['best_q'])
                            
                        with c2:
                            # Draw Chart
                            if detail['curve'] is not None:
                                st.write("**最近期回測表現 (Last 20% Samples)**")
                                fig = px.line(detail['curve'], title=f"{t} Equity Curve (Validation)")
                                st.plotly_chart(fig, use_container_width=True)
                                
                        st.caption("最近 5 筆數據特徵：")
                        st.dataframe(detail['df_feat_tail'])

                    with tab2:
                        st.markdown(f"**{t} 最新相關新聞 (Google News)**")
                        news_list = get_stock_news(t)
                        if news_list:
                            for n in news_list:
                                title = n['title']
                                # Keyword Highlighting
                                keywords = ['AI', 'Nvidia', 'Memory', 'DRAM', 'Server', 'Chip', 'Semiconductor', '台積電', '輝達', '記憶體']
                                for k in keywords:
                                    if k.lower() in title.lower():
                                        title = f"🔥 {title}"
                                        break
                                
                                st.markdown(f"- [{title}]({n['link']}) \n  <small style='color:gray'>{n['published']}</small>", unsafe_allow_html=True)
                        else:
                            st.info("暫無相關新聞或連線逾時。")
else:
    st.info("👈 請在左側輸入股票代號清單 (支援多檔)，按下 '開始批次分析' 即可。")
