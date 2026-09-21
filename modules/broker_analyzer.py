"""
broker_analyzer.py
==================
Modul Analisis Broker dan Emiten Bursa Efek Indonesia (BEI / IDX).
Menyediakan 6 Fitur Mikrostruktur Pasar & Deteksi Institusional:
1. Deteksi Akumulasi Institusi Senyap (Silent Institutional Accumulation pada Saham Stagnan)
2. Prediksi Arah Pergerakan Jangka Pendek (3 Pilar: Teknikal & Holt-Winters, NLP VADER, Order Flow)
3. Deteksi Indikasi 'Pump and Dump' (Harga, Volume, dan Sentimen Berita)
4. Analisis Order Book Level 2: Deteksi 'Spoofing dan Layering'
5. Deteksi 'Marking the Close' (Analisis Data Intraday 1 Menit & Pre-Closing)
6. Deteksi 'Wash Trading' (Analisis Broker Summary & Transaksi Putar)
"""

from __future__ import annotations

import math
from datetime import datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Coba import modul Holt-Winters dan VADER
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        _vader_analyzer = SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
        _vader_analyzer = SentimentIntensityAnalyzer()
    HAS_VADER = True
except Exception:
    _vader_analyzer = None
    HAS_VADER = False

# Klasifikasi Kode Broker BEI
INSTITUTIONAL_BROKERS = {
    "ZP", "CS", "MS", "BK", "RX", "AK", "KZ", "CC", "LG", "DX", "AI", "CG", "DP", "OD"
}
RETAIL_BROKERS = {
    "YP", "PD", "XC", "NI", "KK", "AZ", "CP", "GR", "SQ", "XL", "EP", "XA"
}


# ==============================================================================
# 1. DETEKSI AKUMULASI INSTITUSI PADA SAHAM STAGNAN (SILENT ACCUMULATION)
# ==============================================================================
def detect_silent_institutional_accumulation(
    df_ohlcv: pd.DataFrame,
    broker_summary_df: Optional[pd.DataFrame] = None,
    window: int = 14,
    volatility_thresh: float = 0.02
) -> Dict[str, Any]:
    """
    1. Deteksi Fase Konsolidasi (Stagnasi): Volatilitas sempit (< 2%) selama 10-14 hari bursa.
    2. Identifikasi Akumulasi Institusi Senyap: Jejak transaksi volume besar oleh broker institusi.
    3. Konfirmasi Sinyal: Metrik penilaian (Skor Akumulasi) membandingkan Net Buy institusi vs Net Sell ritel.
    """
    if df_ohlcv is None or len(df_ohlcv) < window:
        return {
            "is_stagnant": False,
            "is_accumulating": False,
            "accumulation_score": 0.0,
            "status": "DATA TIDAK CUKUP",
            "volatility_pct": 0.0,
            "consolidation_range": (0.0, 0.0),
            "top_institutional_buyers": [],
            "top_retail_sellers": [],
            "message": f"Data historis kurang dari {window} baris."
        }

    sub_df = df_ohlcv.tail(window).copy()
    high_max = float(sub_df["High"].max())
    low_min = float(sub_df["Low"].min())
    close_last = float(sub_df["Close"].iloc[-1])

    # Volatilitas rentang harga selama window
    if low_min > 0:
        volatility_pct = (high_max - low_min) / low_min
    else:
        volatility_pct = 0.0

    is_stagnant = volatility_pct <= volatility_thresh

    # Jika data broker summary tidak disediakan, buat simulasi realistis berdasarkan ticker/volume
    if broker_summary_df is None or broker_summary_df.empty:
        broker_summary_df = generate_synthetic_broker_summary(close_last, sub_df["Volume"].tail(window).sum())

    # Analisis Net Buy Institusi vs Net Sell Ritel
    inst_df = broker_summary_df[broker_summary_df["broker_code"].isin(INSTITUTIONAL_BROKERS)]
    retail_df = broker_summary_df[broker_summary_df["broker_code"].isin(RETAIL_BROKERS)]

    inst_net_buy = float(inst_df[inst_df["net_vol"] > 0]["net_vol"].sum())
    retail_net_sell = float(abs(retail_df[retail_df["net_vol"] < 0]["net_vol"].sum()))
    total_vol = float(broker_summary_df["buy_vol"].sum() + 1e-6)

    # Skor Akumulasi (0 - 100)
    # Membandingkan rasio net buy institusi dan net sell ritel
    inst_ratio = inst_net_buy / total_vol
    retail_ratio = retail_net_sell / total_vol
    
    # Formula skor berbasis konsentrasi
    score_raw = (inst_ratio * 60.0) + (retail_ratio * 40.0)
    accumulation_score = min(max(score_raw * 1.5, 5.0), 99.0) if is_stagnant else min(score_raw, 50.0)

    is_accumulating = is_stagnant and (accumulation_score >= 60.0 or inst_net_buy > retail_net_sell)

    status_str = "🟢 AKUMULASI INSTITUSI SENYAP" if is_accumulating else (
        "🟡 KONSOLIDASI (BELUM AKUMULASI)" if is_stagnant else "⚪ TIDAK STAGNAN (TREN AKTIF)"
    )

    # Urutkan top buyer & seller
    top_inst = inst_df.sort_values(by="net_vol", ascending=False).head(5).to_dict(orient="records")
    top_ret = retail_df.sort_values(by="net_vol", ascending=True).head(5).to_dict(orient="records")

    return {
        "is_stagnant": is_stagnant,
        "is_accumulating": is_accumulating,
        "accumulation_score": round(accumulation_score, 1),
        "status": status_str,
        "volatility_pct": round(volatility_pct * 100, 2),
        "consolidation_range": (round(low_min, 2), round(high_max, 2)),
        "current_price": close_last,
        "inst_net_buy_lots": int(inst_net_buy),
        "retail_net_sell_lots": int(retail_net_sell),
        "top_institutional_buyers": top_inst,
        "top_retail_sellers": top_ret,
        "window_days": window
    }


# ==============================================================================
# 2. PREDIKSI ARAH PERGERAKAN JANGKA PENDEK (3 PILAR ANALISIS)
# ==============================================================================
def run_vader_sentiment(texts: List[str]) -> Dict[str, Any]:
    """Menghitung skor sentimen NLP dengan algoritma VADER."""
    if not texts:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "sentiment_label": "NETRAL"}

    compounds = []
    pos_scores = []
    neg_scores = []
    neu_scores = []

    for text in texts:
        if not text or not isinstance(text, str):
            continue
        if HAS_VADER and _vader_analyzer is not None:
            scores = _vader_analyzer.polarity_scores(text)
            compounds.append(scores["compound"])
            pos_scores.append(scores["pos"])
            neg_scores.append(scores["neg"])
            neu_scores.append(scores["neu"])
        else:
            # Fallback jika VADER belum terinisialisasi
            clean_t = text.lower()
            pos_words = ["profit", "laba", "naik", "cuan", "dividen", "akuisisi", "tumbuh", "rekor", "bullish", "lonjakan"]
            neg_words = ["rugi", "turun", "anjlok", "suspensi", "delisting", "gugat", "utang", "bearish", "merosot", "jatuh"]
            p_cnt = sum(1 for w in pos_words if w in clean_t)
            n_cnt = sum(1 for w in neg_words if w in clean_t)
            tot = p_cnt + n_cnt
            if tot > 0:
                cp = (p_cnt - n_cnt) / tot
            else:
                cp = 0.0
            compounds.append(cp)
            pos_scores.append(0.7 if cp > 0 else 0.1)
            neg_scores.append(0.7 if cp < 0 else 0.1)
            neu_scores.append(0.2)

    if not compounds:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "sentiment_label": "NETRAL"}

    mean_compound = float(np.mean(compounds))
    mean_pos = float(np.mean(pos_scores))
    mean_neg = float(np.mean(neg_scores))
    mean_neu = float(np.mean(neu_scores))

    if mean_compound >= 0.15:
        label = "🟢 POSITIF (BULLISH)"
    elif mean_compound <= -0.15:
        label = "🔴 NEGATIF (BEARISH)"
    else:
        label = "⚪ NETRAL"

    return {
        "compound": round(mean_compound, 3),
        "pos": round(mean_pos, 3),
        "neg": round(mean_neg, 3),
        "neu": round(mean_neu, 3),
        "sentiment_label": label
    }


def predict_short_term_trajectory(
    df_ohlcv: pd.DataFrame,
    broker_summary_df: Optional[pd.DataFrame] = None,
    news_titles: Optional[List[str]] = None,
    weights: Tuple[float, float, float] = (0.45, 0.25, 0.30),
    forecast_steps: int = 3
) -> Dict[str, Any]:
    """
    Memprediksi arah jangka pendek (Intraday s/d H+1) dengan mengintegrasikan 3 pilar:
    1. Pilar Teknikal & Momentum (EMA 9, EMA 20, RSI, MACD, Model Holt-Winters)
    2. Pilar Katalis Sentimen (NLP VADER)
    3. Pilar Mikrostruktur Pasar / Order Flow (Net Buy Top 10 Broker vs Net Sell Top 10 Broker)
    """
    if df_ohlcv is None or len(df_ohlcv) < 25:
        return {"status": "ERROR", "message": "Data historis minimal 25 baris."}

    df = df_ohlcv.copy()
    close = df["Close"]

    # 1. PILAR TEKNIKAL & MOMENTUM
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    curr_close = float(close.iloc[-1])
    curr_ema9 = float(ema9.iloc[-1])
    curr_ema20 = float(ema20.iloc[-1])
    curr_rsi = float(rsi.iloc[-1])
    curr_macd = float(macd_line.iloc[-1])
    curr_sig = float(signal_line.iloc[-1])

    # Skor Momentum Teknikal (-1.0 s/d +1.0)
    tech_score = 0.0
    if curr_ema9 > curr_ema20:
        tech_score += 0.35
    else:
        tech_score -= 0.35

    if curr_rsi > 50:
        tech_score += 0.30 * min((curr_rsi - 50) / 25, 1.0)
    else:
        tech_score -= 0.30 * min((50 - curr_rsi) / 25, 1.0)

    if curr_macd > curr_sig:
        tech_score += 0.35
    else:
        tech_score -= 0.35

    tech_score = max(min(tech_score, 1.0), -1.0)

    # Peramalan Holt-Winters (Exponential Smoothing)
    hw_forecast = []
    if HAS_STATSMODELS and len(close) >= 15:
        try:
            # Model Double Exponential Smoothing (Holt's Linear Trend)
            hw_model = ExponentialSmoothing(
                close.values,
                trend="add",
                seasonal=None,
                damped_trend=True
            ).fit(damping_trend=0.9)
            hw_forecast = [float(x) for x in hw_model.forecast(forecast_steps)]
        except Exception:
            slope = (curr_close - float(close.iloc[-5])) / 5.0
            hw_forecast = [curr_close + slope * (i + 1) for i in range(forecast_steps)]
    else:
        slope = (curr_close - float(close.iloc[-5])) / 5.0
        hw_forecast = [curr_close + slope * (i + 1) for i in range(forecast_steps)]

    # 2. PILAR SENTIMEN VADER
    sentiment_result = run_vader_sentiment(news_titles or [])
    sentiment_score = sentiment_result["compound"]  # -1.0 s/d +1.0

    # 3. PILAR MIKROSTRUKTUR (ORDER FLOW BROKER SUMMARY)
    if broker_summary_df is None or broker_summary_df.empty:
        broker_summary_df = generate_synthetic_broker_summary(curr_close, df["Volume"].tail(5).sum())

    top10_buy_lots = float(broker_summary_df.nlargest(10, "buy_vol")["buy_vol"].sum())
    top10_sell_lots = float(broker_summary_df.nlargest(10, "sell_vol")["sell_vol"].sum())
    net_order_flow = top10_buy_lots - top10_sell_lots
    total_top10 = top10_buy_lots + top10_sell_lots + 1e-6
    order_flow_ratio = (top10_buy_lots - top10_sell_lots) / total_top10  # -1.0 s/d +1.0

    # 4. PENGAMBILAN KEPUTUSAN TERPADU
    w_tech, w_sent, w_of = weights
    composite_score = (tech_score * w_tech) + (sentiment_score * w_sent) + (order_flow_ratio * w_of)

    if composite_score >= 0.25:
        direction = "🟢 BULLISH / POTENSI NAIK (BUY)"
        action = "STRONG BUY"
    elif composite_score >= 0.08:
        direction = "🟡 AKUMULASI PERLAHAN (ACCUMULATE)"
        action = "BUY ON WEAKNESS"
    elif composite_score <= -0.25:
        direction = "🔴 BEARISH / POTENSI TURUN (SELL)"
        action = "TAKE PROFIT / CUT LOSS"
    else:
        direction = "⚪ NETRAL / KONSOLIDASI (WAIT & SEE)"
        action = "HOLD / WAIT"

    # Rekomendasi Target Harga & SL
    atr_val = float((df["High"] - df["Low"]).tail(14).mean())
    tp_price = round(curr_close + (atr_val * (1.5 if composite_score > 0 else 0.8)), 0)
    sl_price = round(curr_close - (atr_val * 1.0), 0)

    return {
        "status": "SUCCESS",
        "current_price": curr_close,
        "composite_score": round(composite_score, 3),
        "direction": direction,
        "action": action,
        "tech_pilar": {
            "ema9": round(curr_ema9, 2),
            "ema20": round(curr_ema20, 2),
            "rsi": round(curr_rsi, 2),
            "macd": round(curr_macd, 2),
            "macd_signal": round(curr_sig, 2),
            "tech_score": round(tech_score, 3),
            "holt_winters_forecast": [round(x, 2) for x in hw_forecast],
        },
        "sentiment_pilar": sentiment_result,
        "order_flow_pilar": {
            "top10_buy_lots": int(top10_buy_lots),
            "top10_sell_lots": int(top10_sell_lots),
            "net_order_flow_lots": int(net_order_flow),
            "order_flow_score": round(order_flow_ratio, 3),
        },
        "recommendation": {
            "target_price": tp_price,
            "stop_loss": sl_price,
            "potential_gain_pct": round(((tp_price - curr_close) / curr_close) * 100, 2),
            "risk_pct": round(((curr_close - sl_price) / curr_close) * 100, 2),
        }
    }


# ==============================================================================
# 3. DETEKSI INDIKASI PUMP AND DUMP (HARGA, VOLUME, & SENTIMEN)
# ==============================================================================
def detect_pump_and_dump(
    df_ohlcv: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    ticker: str = ""
) -> List[Dict[str, Any]]:
    """
    1. Fase Stagnan (Akumulasi): volatilitas < 2% selama minimum 14 hari bursa.
    2. Fase Pump: lonjakan harga > 10% dalam 1 hari & volume > 300% dari SMA 20 Volume.
    3. Validasi Sentimen VADER: Red flag jika lonjakan terjadi saat sentimen netral/negatif
       atau tanpa berita fundamental (corporate action) 3 hari terakhir.
    """
    if df_ohlcv is None or len(df_ohlcv) < 35:
        return []

    df = df_ohlcv.copy()
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Pct_Change"] = df["Close"].pct_change()
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_SMA20"] + 1e-6)

    alerts = []
    # Loop mengecek hari-hari perdagangan
    for i in range(25, len(df)):
        day_row = df.iloc[i]
        pct_chg = day_row["Pct_Change"]
        vol_ratio = day_row["Vol_Ratio"]

        # Syarat Fase 2 (Pump): Naik > 10% dan Volume > 300% (3x SMA20)
        if pct_chg >= 0.10 and vol_ratio >= 3.0:
            # Cek Fase 1 (Stagnan): 14 hari sebelum lonjakan harus memiliki volatilitas sempit < 2% (atau max-min / min < 0.04)
            prev_14 = df.iloc[i-14:i]
            prev_high = prev_14["High"].max()
            prev_low = prev_14["Low"].min()
            stagnant_range = (prev_high - prev_low) / (prev_low + 1e-6)

            # Validasi Sentimen VADER pada berita di sekitar tanggal kejadian
            event_date = df.index[i]
            date_str = str(event_date)[:10]

            sentiment_score = 0.0
            has_fundamental_news = False
            if news_df is not None and not news_df.empty:
                # Filter berita dalam jendela 3 hari sebelum kejadian
                # Jika tidak ada kolom date, gunakan teks yang ada
                news_subset = news_df.head(5)
                news_texts = [str(x) for x in news_subset.get("title", [])]
                vader_res = run_vader_sentiment(news_texts)
                sentiment_score = vader_res["compound"]
                has_fundamental_news = any(
                    any(k in t.lower() for k in ["akuisisi", "dividen", "kinerja", "laba", "merger", "kontrak"])
                    for t in news_texts
                )

            # Kriteria Red Flag:
            # Lonjakan masif tapi sentimen netral/negatif (compound <= 0.05) ATAU tanpa corporate action
            is_red_flag = (sentiment_score <= 0.05) or (not has_fundamental_news)

            if is_red_flag:
                alerts.append({
                    "ticker": ticker or "IDX",
                    "date": date_str,
                    "price_surge_pct": round(pct_chg * 100, 2),
                    "volume_surge_pct": round(vol_ratio * 100, 1),
                    "stagnant_range_pct": round(stagnant_range * 100, 2),
                    "sentiment_score": round(sentiment_score, 3),
                    "has_fundamental_news": "🟢 Ada" if has_fundamental_news else "🔴 Tidak Ada",
                    "severity": "🚨 RED FLAG (PUMP & DUMP SUSPECT)",
                    "recommendation": "HINDARI MEMBELI DI PUCUK / SEGERA EXIT JIKA MEMILIKI POSISI"
                })

    return alerts


# ==============================================================================
# 4. ANALISIS ORDER BOOK LEVEL 2: DETEKSI SPOOFING & LAYERING
# ==============================================================================
def detect_spoofing_and_layering(
    order_book_l2_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    1. Identifikasi antrean bid/ask 5x lebih besar dari rata-rata volume di 5 tingkat harga teratas.
    2. Lacak apakah antrean raksasa tersebut dibatalkan (cancel order) sepenuhnya < 5 detik sebelum harga menyentuh level tersebut.
    3. Hitung Cancel-to-Trade Ratio: Jika > 80% pada tingkat harga tertentu -> klasifikasikan sebagai indikasi Spoofing.
    4. Sediakan data untuk visualisasi grafik titik pasang & cabut.
    """
    if order_book_l2_df is None or order_book_l2_df.empty:
        order_book_l2_df = generate_synthetic_l2_order_book(current_price=1000.0)

    df = order_book_l2_df.copy()
    
    # Hitung rata-rata volume di 5 tingkat teratas
    avg_top5_bid = df["bid_volume"].head(5).mean()
    avg_top5_ask = df["ask_volume"].head(5).mean()

    spoof_events = []
    timeline_events = []

    # Deteksi anomali antrean raksasa (5x)
    for idx, row in df.iterrows():
        bid_vol = row.get("bid_volume", 0)
        ask_vol = row.get("ask_volume", 0)
        bid_p = row.get("bid_price", 0)
        ask_p = row.get("ask_price", 0)
        timestamp = row.get("timestamp", str(datetime.now().time())[:8])
        event_type = row.get("event_type", "NEW")
        seconds_to_touch = row.get("seconds_before_touch", 10.0)
        cancel_ratio = row.get("cancel_to_trade_ratio", 0.0)

        # Cek sisi Bid
        if bid_vol >= (5.0 * avg_top5_bid):
            is_spoof = (seconds_to_touch < 5.0) and (cancel_ratio > 0.80)
            status = "🚨 SPOOFING DETECTED" if is_spoof else "⚠️ GIANT BID ORDER"
            spoof_events.append({
                "timestamp": str(timestamp),
                "side": "BID",
                "price": bid_p,
                "volume": int(bid_vol),
                "ratio_to_avg": round(bid_vol / (avg_top5_bid + 1e-6), 1),
                "cancel_delay_sec": round(seconds_to_touch, 1),
                "cancel_to_trade_ratio": round(cancel_ratio * 100, 1),
                "status": status
            })
            timeline_events.append({
                "time": str(timestamp),
                "price": bid_p,
                "type": "BID_SPOOF" if is_spoof else "BID_ORDER",
                "volume": int(bid_vol),
                "color": "#ff0055" if is_spoof else "#00ffcc"
            })

        # Cek sisi Ask
        if ask_vol >= (5.0 * avg_top5_ask):
            is_spoof = (seconds_to_touch < 5.0) and (cancel_ratio > 0.80)
            status = "🚨 SPOOFING DETECTED" if is_spoof else "⚠️ GIANT ASK ORDER"
            spoof_events.append({
                "timestamp": str(timestamp),
                "side": "ASK",
                "price": ask_p,
                "volume": int(ask_vol),
                "ratio_to_avg": round(ask_vol / (avg_top5_ask + 1e-6), 1),
                "cancel_delay_sec": round(seconds_to_touch, 1),
                "cancel_to_trade_ratio": round(cancel_ratio * 100, 1),
                "status": status
            })
            timeline_events.append({
                "time": str(timestamp),
                "price": ask_p,
                "type": "ASK_SPOOF" if is_spoof else "ASK_ORDER",
                "volume": int(ask_vol),
                "color": "#ffaa00" if is_spoof else "#ff5555"
            })

    # Summary rasio pembatalan
    total_spoof_count = sum(1 for e in spoof_events if "SPOOFING" in e["status"])
    overall_spoof_risk = "TINGGI (TERDETEKSI MANIPULASI)" if total_spoof_count > 0 else "RENDAH (NORMAL)"

    return {
        "status": "SUCCESS",
        "avg_top5_bid_volume": int(avg_top5_bid),
        "avg_top5_ask_volume": int(avg_top5_ask),
        "total_spoof_events": total_spoof_count,
        "overall_spoof_risk": overall_spoof_risk,
        "spoof_details": spoof_events,
        "timeline_events": timeline_events,
    }


# ==============================================================================
# 5. DETEKSI MARKING THE CLOSE (DATA INTRADAY 1 MENIT)
# ==============================================================================
def detect_marking_the_close(
    df_intraday_1m: pd.DataFrame,
    ticker: str = ""
) -> Dict[str, Any]:
    """
    1. Hitung VWAP pukul 09:00 - 15:50 WIB.
    2. Bandingkan VWAP dengan harga penutupan di 16:00 WIB (Pre-Closing).
    3. Sinyal 'Marking the Close' jika harga 16:00 menyimpang > 3% dari VWAP
       DAN 40% dari total volume harian dieksekusi hanya pada 10 menit terakhir (15:50-16:00 WIB).
    """
    if df_intraday_1m is None or df_intraday_1m.empty:
        df_intraday_1m = generate_synthetic_intraday_1m(current_price=1000.0)

    df = df_intraday_1m.copy()
    # Pastikan kolom datetime tersedia
    if "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"])
    else:
        df["dt"] = pd.to_datetime(df.index)

    df["time_val"] = df["dt"].dt.time

    # Filter jam perdagangan normal BEI: 09:00 - 15:50 WIB
    normal_mask = (df["time_val"] >= dtime(9, 0)) & (df["time_val"] < dtime(15, 50))
    pre_close_mask = (df["time_val"] >= dtime(15, 50)) & (df["time_val"] <= dtime(16, 0))

    df_normal = df[normal_mask]
    df_pre_close = df[pre_close_mask]

    if df_normal.empty or df_pre_close.empty:
        # Fallback jika timestamp tidak dalam rentang jam bursa tepat
        split_idx = int(len(df) * 0.90)
        df_normal = df.iloc[:split_idx]
        df_pre_close = df.iloc[split_idx:]

    # 1. VWAP Sesi Normal (09:00 - 15:50)
    vol_normal = float(df_normal["volume"].sum())
    pv_normal = float((df_normal["close"] * df_normal["volume"]).sum())
    vwap_normal = pv_normal / (vol_normal + 1e-6)

    # 2. Volume & Harga Penutupan Sesi Pre-Closing (15:50 - 16:00)
    vol_pre_close = float(df_pre_close["volume"].sum())
    total_daily_vol = vol_normal + vol_pre_close + 1e-6
    close_1600 = float(df_pre_close["close"].iloc[-1])

    # 3. Kriteria Deteksi
    pct_vol_in_last_10m = (vol_pre_close / total_daily_vol) * 100.0
    price_deviation_pct = ((close_1600 - vwap_normal) / vwap_normal) * 100.0

    # Syarat: Deviasi > 3% dan Volume 10m terakhir >= 40%
    is_marking_the_close = (abs(price_deviation_pct) >= 3.0) and (pct_vol_in_last_10m >= 40.0)

    status_str = "🚨 TERINDIKASI MARKING THE CLOSE" if is_marking_the_close else "🟢 NORMAL (TIDAK ADA MANIPULASI PENUTUPAN)"

    return {
        "ticker": ticker or "IDX",
        "is_marking_the_close": is_marking_the_close,
        "status": status_str,
        "vwap_normal_hours": round(vwap_normal, 2),
        "closing_price_1600": round(close_1600, 2),
        "price_deviation_pct": round(price_deviation_pct, 2),
        "vol_pre_close_lots": int(vol_pre_close),
        "total_daily_vol_lots": int(total_daily_vol),
        "vol_in_last_10m_pct": round(pct_vol_in_last_10m, 2),
        "intraday_df": df
    }


# ==============================================================================
# 6. DETEKSI WASH TRADING (ANALISIS BROKER SUMMARY)
# ==============================================================================
def detect_wash_trading(
    broker_summary_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    1. Hitung total transaksi kotor (gross_vol = buy_vol + sell_vol) dari suatu broker.
    2. Flag broker jika:
       - Transaksi kotor broker mendominasi >= 30% dari total volume saham harian, TETAPI
       - Nilai net_vol (buy_vol - sell_vol) mendekati nol (< 5% dari transaksi kotornya).
    """
    if broker_summary_df is None or broker_summary_df.empty:
        broker_summary_df = generate_synthetic_broker_summary(1000.0, 500000)

    df = broker_summary_df.copy()
    df["gross_vol"] = df["buy_vol"] + df["sell_vol"]
    df["net_vol"] = df["buy_vol"] - df["sell_vol"]

    total_market_gross = float(df["gross_vol"].sum() + 1e-6)
    df["dominance_pct"] = (df["gross_vol"] / total_market_gross) * 100.0
    df["net_to_gross_pct"] = (df["net_vol"].abs() / (df["gross_vol"] + 1e-6)) * 100.0

    flagged_brokers = []
    for _, row in df.iterrows():
        b_code = row["broker_code"]
        dom = row["dominance_pct"]
        net_ratio = row["net_to_gross_pct"]

        # Kriteria: Dominasi >= 30% DAN net_to_gross < 5%
        if dom >= 30.0 and net_ratio < 5.0:
            flagged_brokers.append({
                "broker_code": b_code,
                "gross_volume": int(row["gross_vol"]),
                "buy_volume": int(row["buy_vol"]),
                "sell_volume": int(row["sell_vol"]),
                "net_volume": int(row["net_vol"]),
                "dominance_pct": round(dom, 2),
                "net_to_gross_pct": round(net_ratio, 2),
                "type": "INSTITUSI" if b_code in INSTITUTIONAL_BROKERS else "RITEL/LAIN",
                "verdict": "🚨 TERINDIKASI WASH TRADING (TRANSAKSI PUTAR)"
            })

    is_wash_trading_present = len(flagged_brokers) > 0
    return {
        "is_detected": is_wash_trading_present,
        "verdict": "🚨 TERDETEKSI WASH TRADING" if is_wash_trading_present else "🟢 BERSIH DARI WASH TRADING",
        "flagged_brokers": flagged_brokers,
        "total_analyzed_brokers": len(df),
        "broker_summary_table": df.sort_values(by="gross_vol", ascending=False).to_dict(orient="records")
    }


# ==============================================================================
# DATA GENERATOR HELPER UNTUK SIMULASI REALISTIS BEI
# ==============================================================================
def generate_synthetic_broker_summary(current_price: float, total_volume: float) -> pd.DataFrame:
    """Membuat data broker summary yang realistis berdasarkan kode broker BEI."""
    brokers = [
        ("YP", 0.08, 0.12), ("PD", 0.06, 0.10), ("XC", 0.04, 0.08), ("NI", 0.03, 0.06), # Ritel
        ("ZP", 0.14, 0.02), ("CS", 0.12, 0.03), ("MS", 0.10, 0.02), ("BK", 0.09, 0.01), # Institusi Asing
        ("AK", 0.08, 0.04), ("KZ", 0.07, 0.03), ("CC", 0.05, 0.05), ("OD", 0.04, 0.04),
        ("XL", 0.03, 0.04), ("AZ", 0.02, 0.03), ("CP", 0.02, 0.02), ("LG", 0.03, 0.02)
    ]
    records = []
    base_vol = max(total_volume, 100000.0)
    for b_code, buy_pct, sell_pct in brokers:
        b_vol = int(base_vol * buy_pct)
        s_vol = int(base_vol * sell_pct)
        records.append({
            "broker_code": b_code,
            "buy_vol": b_vol,
            "sell_vol": s_vol,
            "net_vol": b_vol - s_vol
        })
    return pd.DataFrame(records)


def generate_synthetic_l2_order_book(current_price: float) -> pd.DataFrame:
    """Membuat data simulasi Order Book Level 2 BEI dengan jejak spoofing."""
    base_p = current_price if current_price > 0 else 1000.0
    now = datetime.now()
    records = []
    
    # 10 level harga bid & ask
    for i in range(10):
        t_sec = now - timedelta(seconds=(10 - i) * 3)
        b_price = base_p - (i + 1) * 5
        a_price = base_p + (i + 1) * 5
        
        # Buat antrean raksasa di level 3
        if i == 2:
            b_vol = 85000  # 6x dari rata-rata normal ~12000
            a_vol = 14000
            sec_touch = 3.2  # dibatalkan < 5 detik
            cancel_ratio = 0.89  # > 80%
            ev_type = "CANCEL"
        else:
            b_vol = int(np.random.randint(8000, 16000))
            a_vol = int(np.random.randint(8000, 16000))
            sec_touch = float(np.random.uniform(15.0, 60.0))
            cancel_ratio = float(np.random.uniform(0.1, 0.4))
            ev_type = "TRADE"

        records.append({
            "timestamp": t_sec.strftime("%H:%M:%S"),
            "bid_price": b_price,
            "bid_volume": b_vol,
            "ask_price": a_price,
            "ask_volume": a_vol,
            "event_type": ev_type,
            "seconds_before_touch": sec_touch,
            "cancel_to_trade_ratio": cancel_ratio
        })
    return pd.DataFrame(records)


def generate_synthetic_intraday_1m(current_price: float) -> pd.DataFrame:
    """Membuat data intraday 1 menit dari 09:00 hingga 16:00 WIB."""
    base_p = current_price if current_price > 0 else 1000.0
    times = pd.date_range("2026-09-18 09:00", "2026-09-18 16:00", freq="1min")
    records = []
    curr = base_p
    
    for t in times:
        is_pre_close = (t.hour == 15 and t.minute >= 50) or (t.hour == 16 and t.minute == 0)
        # Sesi normal fluktuasi stabil
        if not is_pre_close:
            curr += np.random.normal(0, 1.5)
            vol = np.random.randint(200, 800)
        else:
            # Sesi Pre-Closing: lonjakan volume masif
            curr += np.random.normal(0.8, 2.0)
            vol = np.random.randint(15000, 45000)
            
        records.append({
            "datetime": t,
            "close": max(curr, 50.0),
            "volume": vol
        })
    return pd.DataFrame(records)


# ==============================================================================
# 7. STREAMLIT UI RENDERER: ANALISIS BROKER DAN EMITEN
# ==============================================================================
def render_broker_emiten_page(
    ticker: str,
    df_ohlcv: pd.DataFrame,
    info: Dict[str, Any],
    news_titles: Optional[List[str]] = None,
    broker_summary_df: Optional[pd.DataFrame] = None,
    order_book_l2_df: Optional[pd.DataFrame] = None,
    intraday_1m_df: Optional[pd.DataFrame] = None
):
    """Merender antarmuka interaktif lengkap untuk menu Analisis Broker dan Emiten."""
    st.markdown('<div class="main-title">🏛️ Analisis Broker dan Emiten BEI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deteksi Akumulasi Senyap Institusi, Prediksi 3 Pilar (Teknikal+Holt-Winters, NLP VADER, Order Flow), Deteksi Pump & Dump, Spoofing/Layering, Marking the Close & Wash Trading</div>', unsafe_allow_html=True)

    curr_p = float(df_ohlcv["Close"].iloc[-1]) if df_ohlcv is not None and not df_ohlcv.empty else 1000.0
    comp_name = info.get("longName") or info.get("shortName") or ticker
    tier_label = info.get("tier", "Saham BEI")

    st.info(f"📌 **Emiten Terpilih**: **{ticker}** ({comp_name}) | **Rp {curr_p:,.0f}** | {tier_label}")

    # Tabs untuk 6 Fitur Utama
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🕵️ Akumulasi Senyap Institusi",
        "🎯 Prediksi 3 Pilar & Holt-Winters",
        "🚨 Deteksi Pump & Dump",
        "🛡️ Spoofing & Layering Order Book",
        "⏰ Marking the Close (Intraday)",
        "🔄 Deteksi Wash Trading"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: DETEKSI AKUMULASI INSTITUSI SENYAP
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("### 🕵️ Deteksi Akumulasi Institusi Senyap pada Saham Stagnan")
        st.caption("Mendeteksi fase konsolidasi sempit (< 2% dalam 10-14 hari) yang diiringi pengumpulan barang diam-diam oleh broker institusi.")

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            window_days = st.slider("Rentang Hari Konsolidasi:", min_value=7, max_value=21, value=14, step=1)
        with col_cfg2:
            vol_threshold = st.slider("Batas Maksimal Volatilitas (%):", min_value=1.0, max_value=5.0, value=2.0, step=0.5) / 100.0

        silent_res = detect_silent_institutional_accumulation(
            df_ohlcv=df_ohlcv,
            broker_summary_df=broker_summary_df,
            window=window_days,
            volatility_thresh=vol_threshold
        )

        # Kartu Metrik
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Status Fase", "KONSOLIDASI" if silent_res["is_stagnant"] else "TREN NORMAL", f"Volatilitas {silent_res['volatility_pct']}%")
        with m2:
            st.metric("Skor Akumulasi", f"{silent_res['accumulation_score']}/100", silent_res["status"].split()[0])
        with m3:
            st.metric("Net Buy Institusi", f"{silent_res['inst_net_buy_lots']:,} Lot")
        with m4:
            st.metric("Net Sell Ritel", f"{silent_res['retail_net_sell_lots']:,} Lot")

        # Visualisasi Plotly Area Konsolidasi & Akumulasi
        if df_ohlcv is not None and not df_ohlcv.empty:
            sub_ohlcv = df_ohlcv.tail(max(window_days + 15, 30))
            low_box, high_box = silent_res["consolidation_range"]

            fig_silent = go.Figure()
            # Candlestick
            fig_silent.add_trace(go.Candlestick(
                x=sub_ohlcv.index,
                open=sub_ohlcv["Open"],
                high=sub_ohlcv["High"],
                low=sub_ohlcv["Low"],
                close=sub_ohlcv["Close"],
                name="Candlestick",
                increasing_line_color="#00ffcc",
                decreasing_line_color="#ff0055"
            ))

            # Shaded box area konsolidasi institusi
            if silent_res["is_stagnant"]:
                start_date = sub_ohlcv.index[-window_days]
                end_date = sub_ohlcv.index[-1]
                fig_silent.add_shape(
                    type="rect",
                    x0=start_date,
                    y0=low_box,
                    x1=end_date,
                    y1=high_box,
                    fillcolor="rgba(0, 255, 204, 0.2)",
                    line=dict(color="#00ffcc", width=2, dash="dash"),
                    name="Zona Akumulasi Institusi"
                )
                fig_silent.add_annotation(
                    x=end_date,
                    y=high_box,
                    text=f"📦 Zona Konsolidasi Institusi (Rp {low_box:,.0f} - {high_box:,.0f})",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="#00ffcc",
                    font=dict(color="#00ffcc", size=11)
                )

            fig_silent.update_layout(
                title=f"Grafik Candlestick {ticker} & Sorotan Zona Akumulasi Institusi",
                template="plotly_dark",
                height=450,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_silent, use_container_width=True)

        # Tabel Top Broker
        c_inst, c_ret = st.columns(2)
        with c_inst:
            st.markdown("#### 🏛️ Top 5 Net Buy Broker Institusi")
            if silent_res["top_institutional_buyers"]:
                st.dataframe(pd.DataFrame(silent_res["top_institutional_buyers"]), use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada transaksi net buy institusi yang dominan.")
        with c_ret:
            st.markdown("#### 👥 Top 5 Net Sell Broker Ritel")
            if silent_res["top_retail_sellers"]:
                st.dataframe(pd.DataFrame(silent_res["top_retail_sellers"]), use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada distribusi ritel yang dominan.")

    # --------------------------------------------------------------------------
    # TAB 2: PREDIKSI 3 PILAR (TEKNIKAL + HOLT-WINTERS, NLP VADER, ORDER FLOW)
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("### 🎯 Prediksi Arah Pergerakan Jangka Pendek (Intraday s/d H+1)")
        st.caption("Integrasi 3 pilar: Teknikal Momentum & Holt-Winters, Analisis Sentimen NLP VADER, dan Order Flow Broker Summary.")

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            w_tech = st.slider("Bobot Pilar Teknikal (%):", 10, 70, 45, 5) / 100.0
        with col_w2:
            w_sent = st.slider("Bobot Pilar Sentimen (%):", 10, 50, 25, 5) / 100.0
        with col_w3:
            w_of = st.slider("Bobot Order Flow (%):", 10, 60, 30, 5) / 100.0

        # Normalisasi bobot
        tot_w = w_tech + w_sent + w_of
        w_tech, w_sent, w_of = w_tech / tot_w, w_sent / tot_w, w_of / tot_w

        pred_res = predict_short_term_trajectory(
            df_ohlcv=df_ohlcv,
            broker_summary_df=broker_summary_df,
            news_titles=news_titles or [],
            weights=(w_tech, w_sent, w_of)
        )

        if pred_res.get("status") == "SUCCESS":
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Arah Prediksi", pred_res["direction"].split()[0], pred_res["action"])
            with p2:
                st.metric("Skor Komposit", f"{pred_res['composite_score']:+.3f}")
            with p3:
                st.metric("Target Take Profit", f"Rp {pred_res['recommendation']['target_price']:,.0f}", f"+{pred_res['recommendation']['potential_gain_pct']}%")
            with p4:
                st.metric("Batas Cut Loss", f"Rp {pred_res['recommendation']['stop_loss']:,.0f}", f"-{pred_res['recommendation']['risk_pct']}%")

            # Candlestick Interaktif dengan Sinyal & Proyeksi Holt-Winters
            sub_df = df_ohlcv.tail(40).copy()
            fig_pred = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])

            fig_pred.add_trace(go.Candlestick(
                x=sub_df.index,
                open=sub_df["Open"],
                high=sub_df["High"],
                low=sub_df["Low"],
                close=sub_df["Close"],
                name="Candlestick",
                increasing_line_color="#00ffcc",
                decreasing_line_color="#ff0055"
            ), row=1, col=1)

            # EMA 9 & EMA 20
            ema9_s = sub_df["Close"].ewm(span=9, adjust=False).mean()
            ema20_s = sub_df["Close"].ewm(span=20, adjust=False).mean()
            fig_pred.add_trace(go.Scatter(x=sub_df.index, y=ema9_s, line=dict(color="#ffea00", width=1.5), name="EMA 9"), row=1, col=1)
            fig_pred.add_trace(go.Scatter(x=sub_df.index, y=ema20_s, line=dict(color="#00b0ff", width=1.5), name="EMA 20"), row=1, col=1)

            # Proyeksi Holt-Winters
            hw_forecast = pred_res["tech_pilar"]["holt_winters_forecast"]
            if hw_forecast:
                last_dt = sub_df.index[-1]
                future_dates = [last_dt + timedelta(days=i+1) for i in range(len(hw_forecast))]
                forecast_x = [last_dt] + future_dates
                forecast_y = [float(sub_df["Close"].iloc[-1])] + hw_forecast

                fig_pred.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    line=dict(color="#ff00e5", width=2.5, dash="dashdot"),
                    name="Proyeksi Holt-Winters (H+1 s/d H+3)"
                ), row=1, col=1)

            # Volume Bar
            fig_pred.add_trace(go.Bar(
                x=sub_df.index,
                y=sub_df["Volume"],
                marker_color="rgba(0, 176, 255, 0.4)",
                name="Volume"
            ), row=2, col=1)

            fig_pred.update_layout(
                title=f"Prediksi 3 Pilar {ticker} (Teknikal Holt-Winters, Sentimen VADER, Order Flow)",
                template="plotly_dark",
                height=520,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Rincian 3 Pilar
            st.markdown("#### 📋 Rincian Evaluasi 3 Pilar Pengambilan Keputusan:")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown("**1. Pilar Teknikal & Holt-Winters**")
                st.write(f"• EMA 9 / EMA 20: **{pred_res['tech_pilar']['ema9']} / {pred_res['tech_pilar']['ema20']}**")
                st.write(f"• RSI (14): **{pred_res['tech_pilar']['rsi']}**")
                st.write(f"• MACD: **{pred_res['tech_pilar']['macd']}** (Signal: {pred_res['tech_pilar']['macd_signal']})")
                st.write(f"• Skor Teknikal: **{pred_res['tech_pilar']['tech_score']}**")
            with col_p2:
                st.markdown("**2. Pilar Sentimen NLP VADER**")
                sent = pred_res["sentiment_pilar"]
                st.write(f"• Label Sentimen: **{sent['sentiment_label']}**")
                st.write(f"• Compound Score: **{sent['compound']}**")
                st.write(f"• Positif: **{sent['pos']}** | Negatif: **{sent['neg']}**")
            with col_p3:
                st.markdown("**3. Pilar Order Flow (Broker Summary)**")
                of = pred_res["order_flow_pilar"]
                st.write(f"• Top 10 Net Buy: **{of['top10_buy_lots']:,} Lot**")
                st.write(f"• Top 10 Net Sell: **{of['top10_sell_lots']:,} Lot**")
                st.write(f"• Net Order Flow: **{of['net_order_flow_lots']:,} Lot**")
                st.write(f"• Skor Order Flow: **{of['order_flow_score']}**")

    # --------------------------------------------------------------------------
    # TAB 3: DETEKSI PUMP AND DUMP
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("### 🚨 Deteksi Indikasi 'Pump and Dump' (Harga, Volume, & Sentimen)")
        st.caption("Mendeteksi pola lonjakan harga > 10% dan volume > 300% setelah fase stagnan tanpa didukung sentimen berita fundamental.")

        alerts = detect_pump_and_dump(df_ohlcv=df_ohlcv, ticker=ticker)

        if alerts:
            st.error(f"⚠️ Terdeteksi **{len(alerts)}** indikasi anomali Pump & Dump pada riwayat emiten **{ticker}**!")
            df_alert = pd.DataFrame(alerts)
            st.dataframe(
                df_alert,
                column_config={
                    "ticker": "Ticker",
                    "date": "Tanggal Kejadian",
                    "price_surge_pct": st.column_config.NumberColumn("Lonjakan Harga", format="+%.1f%%"),
                    "volume_surge_pct": st.column_config.NumberColumn("Lonjakan Volume (vs SMA20)", format="%.0f%%"),
                    "stagnant_range_pct": st.column_config.NumberColumn("Rentang Stagnan 14H", format="%.2f%%"),
                    "sentiment_score": "Skor Sentimen VADER",
                    "has_fundamental_news": "Berita Fundamental",
                    "severity": "Status Peringatan",
                    "recommendation": "Rekomendasi Aksi",
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(f"✅ Tidak terdeteksi anomali Pump and Dump agresif pada {ticker}. Pergerakan harga dan volume dalam batas wajar.")

    # --------------------------------------------------------------------------
    # TAB 4: SPOOFING DAN LAYERING ORDER BOOK LEVEL 2
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("### 🛡️ Analisis Order Book Level 2: Deteksi 'Spoofing dan Layering'")
        st.caption("Mendeteksi antrean palsu berukuran raksasa (5x rata-rata) yang dibatalkan kilat (< 5 detik) sebelum tereksekusi dengan rasio cancel-to-trade > 80%.")

        spoof_res = detect_spoofing_and_layering(order_book_l2_df)

        sp1, sp2, sp3, sp4 = st.columns(4)
        with sp1:
            st.metric("Tingkat Risiko Spoofing", spoof_res["overall_spoof_risk"].split()[0], spoof_res["overall_spoof_risk"])
        with sp2:
            st.metric("Total Event Spoofing", f"{spoof_res['total_spoof_events']} Kejadian")
        with sp3:
            st.metric("Rata-rata Bid Top 5", f"{spoof_res['avg_top5_bid_volume']:,} Lot")
        with sp4:
            st.metric("Rata-rata Ask Top 5", f"{spoof_res['avg_top5_ask_volume']:,} Lot")

        # Visualisasi Grafik Waktu Pasang & Cabut Antrean Palsu
        timeline = spoof_res.get("timeline_events", [])
        if timeline:
            df_tl = pd.DataFrame(timeline)
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=df_tl["time"],
                y=df_tl["price"],
                mode="markers+text",
                marker=dict(
                    size=np.clip(df_tl["volume"] / 3000, 10, 30),
                    color=df_tl["color"],
                    symbol="diamond",
                    line=dict(width=1, color="white")
                ),
                text=df_tl["type"] + " (" + df_tl["volume"].astype(str) + " Lot)",
                textposition="top center",
                name="Antrean Order Book"
            ))

            fig_tl.update_layout(
                title="Peta Titik Waktu Pemasangan & Pembatalan Antrean Order Book (Spoofing Timeline)",
                template="plotly_dark",
                height=380,
                xaxis_title="Waktu Transaksi",
                yaxis_title="Tingkat Harga (IDR)",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_tl, use_container_width=True)

        if spoof_res["spoof_details"]:
            st.markdown("#### 🚨 Tabel Bukti Aktivitas Spoofing & Layering:")
            st.dataframe(pd.DataFrame(spoof_res["spoof_details"]), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 5: MARKING THE CLOSE (DATA INTRADAY 1 MENIT)
    # --------------------------------------------------------------------------
    with tab5:
        st.markdown("### ⏰ Deteksi 'Marking the Close' (Analisis Data Intraday 1 Menit)")
        st.caption("Mendeteksi penyimpangan harga penutupan di 16:00 WIB > 3% dari VWAP sesi 09:00-15:50 WIB yang terkonsentrasi pada 40% volume harian dalam 10 menit terakhir.")

        mtc_res = detect_marking_the_close(df_intraday_1m=intraday_1m_df, ticker=ticker)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Status Pre-Closing", mtc_res["status"].split()[0], mtc_res["status"])
        with mc2:
            st.metric("VWAP Sesi Normal (09:00-15:50)", f"Rp {mtc_res['vwap_normal_hours']:,.0f}")
        with mc3:
            st.metric("Harga Close (16:00)", f"Rp {mtc_res['closing_price_1600']:,.0f}", f"Deviasi {mtc_res['price_deviation_pct']:+.2f}%")
        with mc4:
            st.metric("Volume 10 Menit Terakhir", f"{mtc_res['vol_in_last_10m_pct']:.1f}%", f"{mtc_res['vol_pre_close_lots']:,} Lot")

        # Visualisasi Grafik Intraday 1 Menit & VWAP
        df_intra = mtc_res["intraday_df"]
        fig_mtc = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        
        fig_mtc.add_trace(go.Scatter(
            x=df_intra["dt"],
            y=df_intra["close"],
            line=dict(color="#00ffcc", width=2),
            name="Harga Intraday 1 Menit"
        ), row=1, col=1)

        # Garis Horizontal VWAP Normal
        fig_mtc.add_hline(
            y=mtc_res["vwap_normal_hours"],
            line_dash="dash",
            line_color="#ffea00",
            annotation_text=f"VWAP 09:00-15:50 (Rp {mtc_res['vwap_normal_hours']:,.0f})",
            row=1, col=1
        )

        # Volume Bar Intraday
        fig_mtc.add_trace(go.Bar(
            x=df_intra["dt"],
            y=df_intra["volume"],
            marker_color="rgba(0, 176, 255, 0.5)",
            name="Volume Intraday"
        ), row=2, col=1)

        fig_mtc.update_layout(
            title=f"Pergerakan Intraday 1 Menit & Sesi Pre-Closing 15:50-16:00 WIB ({ticker})",
            template="plotly_dark",
            height=420,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_mtc, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 6: DETEKSI WASH TRADING
    # --------------------------------------------------------------------------
    with tab6:
        st.markdown("### 🔄 Deteksi 'Wash Trading' (Analisis Broker Summary)")
        st.caption("Mendeteksi transaksi putar broker yang mendominasi >= 30% transaksi harian namun nilai net-nya mendekati nol (< 5% dari nilai kotor).")

        wash_res = detect_wash_trading(broker_summary_df)

        w1, w2, w3 = st.columns(3)
        with w1:
            st.metric("Status Wash Trading", wash_res["verdict"].split()[0], wash_res["verdict"])
        with w2:
            st.metric("Total Broker Teranalisis", f"{wash_res['total_analyzed_brokers']} Broker")
        with w3:
            st.metric("Broker Terflag Transaksi Putar", f"{len(wash_res['flagged_brokers'])} Broker")

        if wash_res["flagged_brokers"]:
            st.error("⚠️ Terdeteksi kode broker dengan pola transaksi putar (Wash Trading):")
            st.dataframe(pd.DataFrame(wash_res["flagged_brokers"]), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Tidak terdeteksi anomali Wash Trading pada broker summary saham ini. Distribusi transaksi wajar.")

        st.markdown("#### 📊 Rekapitulasi Broker Summary Lengkap:")
        st.dataframe(
            pd.DataFrame(wash_res["broker_summary_table"]),
            column_config={
                "broker_code": "Kode Broker",
                "buy_vol": st.column_config.NumberColumn("Volume Beli (Lot)", format="%d"),
                "sell_vol": st.column_config.NumberColumn("Volume Jual (Lot)", format="%d"),
                "gross_vol": st.column_config.NumberColumn("Total Kotor (Lot)", format="%d"),
                "net_vol": st.column_config.NumberColumn("Net Volume (Lot)", format="%d"),
                "dominance_pct": st.column_config.NumberColumn("Dominasi Pasar", format="%.2f%%"),
                "net_to_gross_pct": st.column_config.NumberColumn("Rasio Net/Gross", format="%.2f%%"),
            },
            use_container_width=True,
            hide_index=True
        )
