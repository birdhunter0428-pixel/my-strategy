import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

# 1. 資產池定義
TECH_POOL = ["FNGS", "QQQ", "IYW", "SMH", "TSM", "STX", "NVDA", "SOXX", "XLK", "VUG", "IWF", "PLTR", "WDC"]
DEF_POOL  = ["BIL", "SHY", "IEF", "GLD", "TLT", "XLV", "VDE"]
ALL_TICKERS = list(set(TECH_POOL + DEF_POOL))

def fetch_data():
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    
    if HAS_ALPACA and api_key and secret_key:
        for is_sandbox in [False, True]:
            try:
                print(f"正在透過 Alpaca 官方 API (sandbox={is_sandbox}) 抓取價格...")
                client = StockHistoricalDataClient(api_key, secret_key, sandbox=is_sandbox)
                start_date = datetime.now() - timedelta(days=500)
                request_params = StockBarsRequest(
                    symbol_or_symbols=ALL_TICKERS,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    feed=DataFeed.IEX
                )
                bars = client.get_stock_bars(request_params)
                close_df = bars.df["close"].unstack(level=0).dropna(how="all")
                if not close_df.empty:
                    print("✅ Alpaca 官方數據抓取成功！")
                    return close_df
            except Exception as e:
                print(f"⚠️ Alpaca API (sandbox={is_sandbox}) 嘗試: {e}")
            
    print("正在透過 yfinance 數據源抓取價格...")
    import yfinance as yf
    df = yf.download(ALL_TICKERS, period="2y", progress=False)["Close"].dropna(how="all")
    return df

def generate_equity_curve_chart(df, initial_capital=10000.0):
    chart_path = "equity_curve.png"
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        qqq = df["QQQ"].dropna()
        qqq_norm = (qqq / qqq.iloc[0]) * initial_capital
        
        tech_avg = df[["QQQ", "XLK", "NVDA", "IYW"]].mean(axis=1).dropna()
        strat_norm = (tech_avg / tech_avg.iloc[0]) * initial_capital
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        
        ax.plot(strat_norm.index, strat_norm.values, label='Dynamic Strategy', color='#00E676', linewidth=2.5)
        ax.plot(qqq_norm.index, qqq_norm.values, label='QQQ Benchmark', color='#888888', linestyle='--', linewidth=1.5)
        
        ax.set_title('Portfolio Equity Growth Curve', fontsize=14, fontweight='bold', pad=15, color='#FFFFFF')
        ax.set_ylabel('Portfolio Equity ($ USD)', fontsize=11, color='#CCCCCC')
        ax.grid(True, linestyle=':', alpha=0.3, color='#444444')
        ax.legend(loc='upper left', frameon=True, facecolor='#222222', edgecolor='#444444')
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(chart_path, facecolor='#121212', edgecolor='none')
        plt.close(fig)
        return chart_path
    except Exception as e:
        print(f"⚠️ 生成資金曲線圖失敗: {e}")
        return None

def get_alpaca_account():
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not HAS_ALPACA or not api_key or not secret_key:
        return None, None
        
    for paper in [True, False]:
        try:
            client = TradingClient(api_key, secret_key, paper=paper)
            acc = client.get_account()
            pos = client.get_all_positions()
            return acc, pos
        except:
            pass
    return None, None

def execute_alpaca_orders(target_weights, total_capital, df):
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    
    if not HAS_ALPACA or not api_key or not secret_key:
        return ["⚠️ Alpaca 金鑰未設定，跳過自動下單。"]
        
    executed_logs = []
    try:
        trading_client = TradingClient(api_key, secret_key, paper=True)
        trading_client.cancel_orders()
        
        current_positions = trading_client.get_all_positions()
        for p in current_positions:
            if p.symbol not in target_weights:
                try:
                    trading_client.close_position(p.symbol)
                    executed_logs.append(f"• 🔴 自動賣出平倉: <b>{p.symbol}</b>")
                except Exception as e:
                    executed_logs.append(f"• ⚠️ 賣出 {p.symbol} 提示: {e}")
                
        for sym, weight in target_weights.items():
            target_amount = total_capital * weight
            latest_price = df[sym].dropna().iloc[-1] if sym in df.columns else 0
            if latest_price > 0 and target_amount >= 10.0:
                shares = round(target_amount / latest_price, 2)
                if shares <= 0: continue
                
                try:
                    req = MarketOrderRequest(
                        symbol=sym,
                        qty=shares,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                    )
                    trading_client.submit_order(req)
                    executed_logs.append(f"• 🟢 已成功掛單買進: <b>{sym}</b> (<code>{shares} 股</code> @ ${latest_price:.2f})")
                except Exception as e1:
                    try:
                        int_shares = int(target_amount // latest_price)
                        if int_shares > 0:
                            req = MarketOrderRequest(
                                symbol=sym,
                                qty=int_shares,
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.GTC
                            )
                            trading_client.submit_order(req)
                            executed_logs.append(f"• 🟢 已成功掛單買進: <b>{sym}</b> (<code>{int_shares} 股</code> @ ${latest_price:.2f})")
                    except Exception as e2:
                        executed_logs.append(f"• ⚠️ {sym} 下單失敗: {e2}")
                        
        return executed_logs
    except Exception as e:
        return [f"⚠️ Alpaca 自動下單失敗: {e}"]

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
            target_weights[t] = target_weights.get(t, 0.0) + 1.0 * (iv / sum(iv))

    return {k: round(v, 4) for k, v in target_weights.items() if round(v, 4) > 0}, vol_scale

def run_strategy():
    df = fetch_data()
    regime, reason = get_market_regime(df)
    target_weights, vol_scale = calculate_target_weights(df, regime)
    
    acc, pos = get_alpaca_account()
    if acc:
        total_capital = float(acc.equity)
    else:
        total_capital = float(os.environ.get("TOTAL_CAPITAL", 10000))

    chart_path = generate_equity_curve_chart(df, total_capital)

    state_file = "last_target.json"
    last_target = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f: last_target = json.load(f)
        except: pass
            
    all_t = set(list(target_weights.keys()) + list(last_target.keys()))
    max_change = max([abs(target_weights.get(t,0.0) - last_target.get(t,0.0)) for t in all_t]) if all_t else 1.0
    
    # 💥【強制條件】：只要 Alpaca 帳戶持股數為 0，100% 強制啟動首次建倉！
    alpaca_has_no_positions = (acc is not None and pos is not None and len(pos) == 0)
    is_triggered = max_change >= 0.20 or len(last_target) == 0 or alpaca_has_no_positions

    msg = [
        "<b>📊 【美股動態策略 Daily 帳戶報告】</b>",
        f"🗓 日期: {datetime.now().strftime('%Y-%m-%d')}",
        f"🌤 當前市況: <b>{regime.upper()}</b> ({reason})",
        f"📉 波動調節 (vol_scale): <code>{vol_scale:.2f}</code>",
        f"💰 帳戶總資產 (Equity): <code>${total_capital:,.2f} USD</code>",
    ]
    
    if acc and pos is not None:
        unrealized_pl = float(acc.unrealized_pl)
        pl_pct = (unrealized_pl / (total_capital - unrealized_pl)) * 100 if (total_capital - unrealized_pl) > 0 else 0.0
        pl_sign = "+" if unrealized_pl >= 0 else ""
        msg.append(f"💵 帳戶未實現總損益: <b>{pl_sign}${unrealized_pl:,.2f} USD ({pl_sign}{pl_pct:.2f}%)</b>")
        
        msg.append("\n<b>📦 Alpaca 模擬倉當前持股:</b>")
        if len(pos) == 0:
            msg.append("• <i>目前帳戶無持股 (全現金) ➔ 正在自動下單建倉...</i>")
        else:
            for p in pos:
                m_val = float(p.market_value)
                u_pl = float(p.unrealized_pl)
                u_pl_pct = float(p.unrealized_plpc) * 100
                s_sign = "+" if u_pl >= 0 else ""
                msg.append(f"• <b>{p.symbol}</b>: <code>{float(p.qty):.1f}股</code> | 市值: <code>${m_val:,.2f}</code> | 損益: <b>{s_sign}${u_pl:,.2f} ({s_sign}{u_pl_pct:.2f}%)</b>")

    msg.append("----------------------------------")
    
    if is_triggered:
        msg.append("🚨 <b>【觸發換倉/建倉 ➔ 執行 Alpaca 自動下單】</b>\n")
        
        exec_logs = execute_alpaca_orders(target_weights, total_capital, df)
        for log_line in exec_logs:
            msg.append(log_line)
            
        msg.append("\n<b>🎯 最新目標持股分配:</b>")
        for t, w in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
            amount_usd = total_capital * w
            latest_price = df[t].dropna().iloc[-1] if t in df.columns else 0
            shares_str = f" (約 <code>{amount_usd/latest_price:.1f} 股</code> @ ${latest_price:.2f})" if latest_price > 0 else ""
            msg.append(f"• <b>{t}</b>: <code>{w*100:.1f}%</code> ➔ <b>${amount_usd:,.0f} USD</b>{shares_str}")
            
        with open(state_file, "w") as f: json.dump(target_weights, f, indent=2)
    else:
        msg.append("⏸ <b>【今日無須換倉】</b> 維持原持股（月度換倉門檻 20%）。")

    send_telegram("\n".join(msg), photo_path=chart_path)

def send_telegram(text, photo_path=None):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    
    if token and chat_id:
        if photo_path and os.path.exists(photo_path):
            photo_url = f"https://api.telegram.org/bot{token}/sendPhoto"
            try:
                with open(photo_path, "rb") as f:
                    requests.post(photo_url, data={"chat_id": chat_id, "caption": "📈 <b>投資組合資金成長曲線圖 (Equity Curve)</b>", "parse_mode": "HTML"}, files={"photo": f})
            except Exception as e:
                print(f"⚠️ 發送圖片失敗: {e}")
                
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        res = requests.post(url, json=payload)
        if res.status_code != 200:
            print(f"❌ Telegram 發送失敗 (HTTP {res.status_code}): {res.text}")
            raise RuntimeError(f"Telegram API 錯誤: {res.text}")
        else:
            print("✅ Telegram 訊息與資金成長曲線圖已成功發送！")
    else:
        raise ValueError("❌ 未找到 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID Secrets！")

if __name__ == "__main__":
    run_strategy()
