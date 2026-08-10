import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# 1. 資產池定義
TECH_POOL = ["FNGS", "QQQ", "IYW", "SMH", "TSM", "STX", "NVDA", "SOXX", "XLK", "VUG", "IWF", "PLTR", "WDC"]
DEF_POOL  = ["BIL", "SHY", "IEF", "GLD", "TLT", "XLV", "VDE"]
ALL_TICKERS = list(set(TECH_POOL + DEF_POOL))

def fetch_data():
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    
    if api_key and secret_key:
        try:
            print("正在透過 Alpaca 官方 API (IEX 數據源) 抓取價格...")
            client = StockHistoricalDataClient(api_key, secret_key)
            start_date = datetime.now() - timedelta(days=500)
            request_params = StockBarsRequest(
                symbol_or_symbols=ALL_TICKERS,
                timeframe=TimeFrame.Day,
                start=start_date,
                feed=DataFeed.IEX  # 專門支援 Alpaca 免費/模擬帳號數據源
            )
            bars = client.get_stock_bars(request_params)
            close_df = bars.df["close"].unstack(level=0).dropna(how="all")
            if not close_df.empty:
                return close_df
        except Exception as e:
            print(f"⚠️ Alpaca API 抓取失敗: {e}\n正在自動切換至備用數據源...")
            
    print("正在透過備用數據源抓取價格...")
    import yfinance as yf
    df = yf.download(ALL_TICKERS, period="2y", progress=False)["Close"].dropna(how="all")
    return df

def get_market_regime(df):
    qqq = df["QQQ"].dropna()
    fngs = df["FNGS"].dropna()
    
    q_now = qqq.iloc[-1]
    f_now = fngs.iloc[-1]
    
    sma_qqq_200 = qqq.iloc[-200:].mean()
    sma_qqq_100 = qqq.iloc[-100:].mean()
    sma_qqq_50  = qqq.iloc[-50:].mean()
    sma_qqq_150 = qqq.iloc[-150:].mean()
    sma_fngs_100 = fngs.iloc[-100:].mean()
    q_ret_21 = (q_now / qqq.iloc[-22]) - 1.0
    
    if q_now < sma_qqq_200:
        return "bear", "QQQ 低於 200 日均線"
    if (q_now < sma_qqq_100) and (f_now < sma_fngs_100):
        return "bear", "QQQ 與 FNGS 皆低於 100 日均線"
    if (q_now < sma_qqq_50) and (q_ret_21 < -0.08):
        return "bear", "QQQ 低於 50 日均線且近 21 日跌幅 > 8%"
    if q_now > (sma_qqq_150 * 1.03):
        return "bull", "QQQ 高於 150 日均線 3%"
    return "neutral", "市場處於中性整理"

def score_candidates(df, tickers):
    scores, sigmas_63d = {}, {}
    for t in tickers:
        if t not in df.columns: continue
        series = df[t].dropna()
        if len(series) < 126: continue
        
        p_now, p_21, p_63, p_126 = series.iloc[-1], series.iloc[-22], series.iloc[-64], series.iloc[-127]
        m1, m3, m6 = (p_now/p_21)-1.0, (p_now/p_63)-1.0, (p_now/p_126)-1.0
        momentum = 0.3 * m1 + 0.4 * m3 + 0.3 * m6
        
        daily_ret = series.pct_change().dropna()
        volatility = daily_ret.iloc[-126:].std() * np.sqrt(252)
        sigmas_63d[t] = daily_ret.iloc[-63:].std()
        
        trend = (p_now / series.iloc[-100:].mean()) - 1.0
        drawdown = (p_now / series.iloc[-63:].max()) - 1.0
        
        scores[t] = (momentum / volatility) + trend + drawdown if volatility > 0 else -999.0
    return scores, sigmas_63d

def get_vol_scale(df):
    qqq = df["QQQ"].dropna().pct_change().dropna()
    if len(qqq) < 252: return 1.0
    curr_v = qqq.iloc[-21:].std() * np.sqrt(252)
    avg_v  = qqq.iloc[-252:].std() * np.sqrt(252)
    return float(np.clip(avg_v / curr_v, 0.5, 1.0)) if curr_v > 0 else 1.0

def calculate_target_weights(df, regime):
    scores, sigmas_63d = score_candidates(df, ALL_TICKERS)
    vol_scale = get_vol_scale(df)
    strong_tech = {t: scores[t] for t in TECH_POOL if t in scores and scores[t] > 0.10}
    strong_def  = {t: scores[t] for t in DEF_POOL if t in scores and scores[t] > 0.10}
    target_weights = {}

    if regime == "bull":
        sorted_tech = [t for t, s in sorted(strong_tech.items(), key=lambda x: x[1], reverse=True)[:4]]
        while len(sorted_tech) < 4: sorted_tech.append("QQQ")
        tot = 1.0 * vol_scale * 1.03
        raw_s = [max(scores.get(t, 0.10), 0.01) for t in sorted_tech]
        for t, sc in zip(sorted_tech, raw_s):
            target_weights[t] = target_weights.get(t, 0.0) + tot * (sc / sum(raw_s))

    elif regime == "neutral":
        sorted_tech = [t for t, s in sorted(strong_tech.items(), key=lambda x: x[1], reverse=True)[:2]]
        while len(sorted_tech) < 2: sorted_tech.append("QQQ")
        t_sc = [max(scores.get(t, 0.10), 0.01) for t in sorted_tech]
        for t, sc in zip(sorted_tech, t_sc):
            target_weights[t] = target_weights.get(t, 0.0) + (0.5 * vol_scale) * (sc / sum(t_sc))
            
        sorted_def = [t for t, s in sorted(strong_def.items(), key=lambda x: x[1], reverse=True)[:3]]
        while len(sorted_def) < 3: sorted_def.append("BIL")
        d_sc = [max(scores.get(t, 0.10), 0.01) for t in sorted_def]
        for t, sc in zip(sorted_def, d_sc):
            target_weights[t] = target_weights.get(t, 0.0) + 0.5 * (sc / sum(d_sc))

    elif regime == "bear":
        all_def = {t: scores.get(t, -999) for t in DEF_POOL if t in scores}
        sorted_def = [t for t, s in sorted(all_def.items(), key=lambda x: x[1], reverse=True)[:3]]
        while len(sorted_def) < 3: sorted_def.append("BIL")
        inv_v = [1.0 / max(sigmas_63d.get(t, 0.01), 0.0001) for t in sorted_def]
        for t, iv in zip(sorted_def, inv_v):
            target_weights[t] = target_weights.get(t, 0.0) + 1.0 * (iv / sum(inv_v))

    return {k: round(v, 4) for k, v in target_weights.items() if round(v, 4) > 0}, vol_scale

def run_strategy():
    try:
        total_capital = float(os.environ.get("TOTAL_CAPITAL", 10000))
    except:
        total_capital = 10000.0

    df = fetch_data()
    regime, reason = get_market_regime(df)
    target_weights, vol_scale = calculate_target_weights(df, regime)
    
    state_file = "last_target.json"
    last_target = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f: last_target = json.load(f)
        except: pass
            
    all_t = set(list(target_weights.keys()) + list(last_target.keys()))
    max_change = max([abs(target_weights.get(t,0.0) - last_target.get(t,0.0)) for t in all_t]) if all_t else 1.0
    is_triggered = max_change >= 0.20 or len(last_target) == 0

    msg = [
        "📊 **【月度動態策略訊號通知】**",
        f"🗓 日期: {datetime.now().strftime('%Y-%m-%d')}",
        f"🌤 當前市況: **{regime.upper()}** ({reason})",
        f"📉 波動調節 (vol_scale): `{vol_scale:.2f}`",
        f"💰 設定本金: `${total_capital:,.0f} USD`",
        f"🔄 最大權重變動: `{max_change*100:.1f}%` (門檻 20.0%)",
        "----------------------------------"
    ]
    
    if is_triggered:
        msg.append("🚨 **【觸發換倉指令】** 請按以下目標權重與金額執行：\n")
        total_allocated = 0.0
        for t, w in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
            amount_usd = total_capital * w
            total_allocated += amount_usd
            
            latest_price = df[t].dropna().iloc[-1] if t in df.columns else 0
            shares_str = f" (約 `{amount_usd/latest_price:.1f} 股` @ ${latest_price:.2f})" if latest_price > 0 else ""
            msg.append(f"• **{t}**: `{w*100:.1f}%` ➔ **${amount_usd:,.0f} USD**{shares_str}")
            
        unallocated_cash = total_capital - total_allocated
        if unallocated_cash > 1:
            msg.append(f"\n💵 **剩餘保留現金/國債(BIL)**: `${unallocated_cash:,.0f} USD` (`{(unallocated_cash/total_capital)*100:.1f}%`)")
            
        with open(state_file, "w") as f: json.dump(target_weights, f, indent=2)
    else:
        msg.append("⏸ **【本月維持原持股（不換倉）】** 變動未達 20% 門檻。")

    send_telegram("\n".join(msg))

def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        res = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
        if res.status_code != 200:
            print(f"❌ Telegram 發送失敗: {res.text}")
        else:
            print("✅ Telegram 訊息已成功發送！")

if __name__ == "__main__":
    run_strategy()
