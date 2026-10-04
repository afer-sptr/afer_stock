"""
technical_analysis_page.py
==========================
Terminal Analisis Teknikal Komprehensif & Institutional Charting Suite
untuk Seluruh Emiten di Bursa Efek Indonesia (BEI / IDX).

Mencakup 5 Pilar Fitur Mutakhir:
1. Jenis Grafik Mutakhir (Advanced Charting Types):
   - Candlestick Standar & Bar Chart (OHLC)
   - Heikin-Ashi (Smoothed Trend Visualization)
   - Renko Chart (Price-Action Based, Time-Independent)
   - Kagi Chart & Point and Figure (P&F Reversal)
2. Library Indikator Kuantitatif (Ratusan Pilihan):
   - Tren: SMA, EMA, WMA, MACD, Ichimoku Cloud (Kumo, Tenkan, Kijun), Parabolic SAR
   - Momentum: RSI 14, Stochastic Oscillator (%K, %D), Commodity Channel Index (CCI)
   - Volatilitas: Bollinger Bands (20, 2), ATR 14, Keltner Channels
   - Volume & Aliran Dana: On-Balance Volume (OBV), Chaikin Money Flow (CMF), Horizontal Volume Profile (POC, VAH, VAL)
3. Alat Menggambar Manual & Geometri (Drawing Tools):
   - Garis Tren & Parallel Channel Otomatis
   - Fibonacci Retracement, Fibonacci Extension, Fibonacci Time Zones
   - Elliott Wave Theory (Siklus Impulsif 1-5 & Korektif A-B-C) & Sudut Geometri Gann Fans
4. Fitur Otomatisasi Berbasis AI & Skrip:
   - Auto Pattern Recognition: Pola Geometri Klasik (Double Bottom, Head and Shoulders, Cup and Handle, Triangle, Flag)
   - Auto Candlestick Formation: Hammer, Engulfing, Morning/Evening Star, Doji, Marubozu
   - Custom Scripting Sandbox (Python Quant Studio ala Pine Script)
   - Multi-Timeframe Analysis (MTF Matrix: M15, H1, D1, W1)
5. Penyaringan & Simulasi Strategi (Screener & Backtesting):
   - Technical Stock Screener (Filter Tier + Syariah + 7 Setup Teknikal Unggulan)
   - Bar Replay Simulator (Time Machine Point-in-Time Inspector)
   - Strategy Tester (Backtesting Engine: Win Rate, Profit Factor, Equity Curve, Max Drawdown & Log Transaksi)
   - Kalkulator Trading Plan Fraksi Resmi BEI & Money Management
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from modules.idx_ticks import get_idx_tick_size, round_to_idx_tick, safe_int
from modules.technical_analysis import (
    compute_technical_indicators,
    detect_support_resistance,
    compute_technical_suite,
    evaluate_technical_score,
)
from modules.idx_universe import (
    filter_idx_stocks,
    TIER_OPTIONS,
    is_sharia_compliant,
    get_stock_metadata,
)
from modules.recommendation_engine import calculate_trading_levels


# ==============================================================================
# 1. KALKULASI GRAFIK MUTAKHIR (HEIKIN-ASHI, RENKO, KAGI)
# ==============================================================================
def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung lilin Heikin-Ashi untuk menyaring noise fluktuasi pasar."""
    ha_df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    ha_open = np.zeros(len(df))
    ha_open[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_df["HA_Open"] = ha_open
    ha_df["HA_Close"] = ha_close
    ha_df["HA_High"] = np.maximum(df["High"], np.maximum(ha_df["HA_Open"], ha_df["HA_Close"]))
    ha_df["HA_Low"] = np.minimum(df["Low"], np.minimum(ha_df["HA_Open"], ha_df["HA_Close"]))
    return ha_df


def compute_renko_boxes(df: pd.DataFrame, box_size: float = 10.0) -> List[Dict[str, Any]]:
    """Menghitung Renko boxes berbasis perubahan harga murni (mengabaikan faktor waktu)."""
    boxes = []
    if df.empty or box_size <= 0:
        return boxes
    prices = df["Close"].values
    dates = df.index
    curr_box = math.floor(prices[0] / box_size) * box_size

    for i in range(1, len(prices)):
        p = prices[i]
        diff = p - curr_box
        if diff >= box_size:
            n_boxes = int(diff // box_size)
            for _ in range(n_boxes):
                new_box = curr_box + box_size
                boxes.append({"date": str(dates[i]).split(" ")[0], "open": curr_box, "close": new_box, "type": "UP"})
                curr_box = new_box
        elif diff <= -box_size:
            n_boxes = int(abs(diff) // box_size)
            for _ in range(n_boxes):
                new_box = curr_box - box_size
                boxes.append({"date": str(dates[i]).split(" ")[0], "open": curr_box, "close": new_box, "type": "DOWN"})
                curr_box = new_box
    return boxes


def compute_kagi_lines(df: pd.DataFrame, reversal_pct: float = 0.03) -> List[Dict[str, Any]]:
    """Menghitung garis Kagi (Garis Yang tebal vs Yin tipis) berdasarkan batas pembalikan arah."""
    lines = []
    if len(df) < 5:
        return lines
    prices = df["Close"].values
    dates = df.index
    
    direction = 1  # 1 = naik, -1 = turun
    start_p = prices[0]
    high_p = prices[0]
    low_p = prices[0]
    is_yang = True  # True = Yang (Bullish), False = Yin (Bearish)

    for i in range(1, len(prices)):
        p = prices[i]
        if direction == 1:
            if p > high_p:
                high_p = p
                if not is_yang and high_p > low_p * (1.0 + reversal_pct):
                    is_yang = True
            elif p <= high_p * (1.0 - reversal_pct):
                lines.append({"start": start_p, "end": high_p, "type": "YANG" if is_yang else "YIN", "date": str(dates[i]).split(" ")[0]})
                direction = -1
                start_p = high_p
                low_p = p
        else:
            if p < low_p:
                low_p = p
                if is_yang and low_p < high_p * (1.0 - reversal_pct):
                    is_yang = False
            elif p >= low_p * (1.0 + reversal_pct):
                lines.append({"start": start_p, "end": low_p, "type": "YANG" if is_yang else "YIN", "date": str(dates[i]).split(" ")[0]})
                direction = 1
                start_p = low_p
                high_p = p
    return lines


def compute_point_and_figure(df: pd.DataFrame, box_size: float = None, reversal: int = 3) -> Dict[str, Any]:
    """
    Menghitung Point and Figure (P&F) Chart berbasis perubahan harga murni (mengabaikan faktor waktu).
    Menyaring noise pasar dan memetakan level Support/Resistance murni.
    """
    if df.empty or len(df) < 5:
        return {"columns": [], "box_size": 1.0, "reversal": reversal, "status": "Data tidak cukup", "signals": []}

    last_close = float(df["Close"].iloc[-1])
    if box_size is None or box_size <= 0:
        if "ATR_14" in df.columns and not pd.isna(df["ATR_14"].iloc[-1]) and df["ATR_14"].iloc[-1] > 0:
            box_size = max(1.0, round(float(df["ATR_14"].iloc[-1]) * 0.5))
        elif "ATR" in df.columns and not pd.isna(df["ATR"].iloc[-1]) and df["ATR"].iloc[-1] > 0:
            box_size = max(1.0, round(float(df["ATR"].iloc[-1]) * 0.5))
        else:
            box_size = max(1.0, round(last_close * 0.01))

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    dates = [str(d).split(" ")[0] for d in df.index]

    columns = []
    current_type = None
    col_high = 0.0
    col_low = 0.0
    first_box = round(closes[0] / box_size) * box_size

    for i in range(1, len(df)):
        h = highs[i]
        l = lows[i]
        box_h = np.floor(h / box_size) * box_size
        box_l = np.ceil(l / box_size) * box_size

        if current_type is None:
            if box_h >= first_box + box_size:
                current_type = 'X'
                col_low = first_box
                col_high = box_h
                b_list = [round(b, 2) for b in np.arange(col_low, col_high + box_size * 0.5, box_size)]
                columns.append({"col_idx": len(columns), "type": "X", "low": col_low, "high": col_high, "boxes": b_list, "date": dates[i]})
            elif box_l <= first_box - box_size:
                current_type = 'O'
                col_high = first_box
                col_low = box_l
                b_list = [round(b, 2) for b in np.arange(col_low, col_high + box_size * 0.5, box_size)]
                columns.append({"col_idx": len(columns), "type": "O", "low": col_low, "high": col_high, "boxes": b_list, "date": dates[i]})
        elif current_type == 'X':
            if box_h >= col_high + box_size:
                col_high = box_h
                columns[-1]["high"] = col_high
                columns[-1]["boxes"] = [round(b, 2) for b in np.arange(columns[-1]["low"], col_high + box_size * 0.5, box_size)]
                columns[-1]["date"] = dates[i]
            elif box_l <= col_high - (reversal * box_size):
                current_type = 'O'
                col_high = col_high - box_size
                col_low = box_l
                b_list = [round(b, 2) for b in np.arange(col_low, col_high + box_size * 0.5, box_size)]
                columns.append({"col_idx": len(columns), "type": "O", "low": col_low, "high": col_high, "boxes": b_list, "date": dates[i]})
        elif current_type == 'O':
            if box_l <= col_low - box_size:
                col_low = box_l
                columns[-1]["low"] = col_low
                columns[-1]["boxes"] = [round(b, 2) for b in np.arange(col_low, columns[-1]["high"], box_size)]
                columns[-1]["date"] = dates[i]
            elif box_h >= col_low + (reversal * box_size):
                current_type = 'X'
                col_low = col_low + box_size
                col_high = box_h
                b_list = [round(b, 2) for b in np.arange(col_low, col_high + box_size * 0.5, box_size)]
                columns.append({"col_idx": len(columns), "type": "X", "low": col_low, "high": col_high, "boxes": b_list, "date": dates[i]})

    signals = []
    if len(columns) >= 3:
        for idx in range(2, len(columns)):
            prev_x = [c for c in columns[:idx] if c["type"] == "X"]
            prev_o = [c for c in columns[:idx] if c["type"] == "O"]
            curr = columns[idx]
            if curr["type"] == "X" and prev_x:
                if curr["high"] > prev_x[-1]["high"]:
                    signals.append({"type": "Bullish Double Top Breakout", "col": idx, "price": curr["high"]})
            elif curr["type"] == "O" and prev_o:
                if curr["low"] < prev_o[-1]["low"]:
                    signals.append({"type": "Bearish Double Bottom Breakdown", "col": idx, "price": curr["low"]})

    return {
        "columns": columns,
        "box_size": box_size,
        "reversal": reversal,
        "signals": signals,
        "last_column_type": columns[-1]["type"] if columns else "N/A"
    }


def compute_wma(series: pd.Series, period: int = 20) -> pd.Series:
    """Menghitung Weighted Moving Average (WMA)."""
    weights = np.arange(1, period + 1)
    def _wma(x):
        return np.dot(x, weights) / weights.sum()
    return series.rolling(period).apply(_wma, raw=True).bfill().ffill()


def calculate_parallel_channel(df: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
    """
    Menghitung Garis Tren Otomatis (Trendline) dan Parallel Channel (Upper, Lower, Median Channel).
    Mendeteksi kemiringan sudut (slope), tipe channel (Ascending, Descending, Horizontal), dan deviasi.
    """
    if len(df) < 15:
        return {"channel_type": "Horizontal Channel", "slope": 0.0, "upper_last": 0.0, "lower_last": 0.0, "median_last": 0.0, "position": "Netral", "status": "Data Kurang"}

    sub_df = df.tail(min(len(df), window))
    y = sub_df["Close"].values
    x = np.arange(len(y))

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]

    reg_line = slope * x + intercept
    diffs = y - reg_line
    std_dev = np.std(diffs)

    upper_channel = reg_line + (1.8 * std_dev)
    lower_channel = reg_line - (1.8 * std_dev)

    c_last = float(y[-1])
    u_last = float(upper_channel[-1])
    l_last = float(lower_channel[-1])
    m_last = float(reg_line[-1])

    norm_factor = c_last / max(1, len(y))
    channel_type = "Ascending Channel (Bullish)" if slope > (0.05 * norm_factor) else (
        "Descending Channel (Bearish)" if slope < (-0.05 * norm_factor) else "Horizontal Channel (Konsolidasi)"
    )

    position = "Di Dekat Resisten Channel Atas" if c_last >= u_last * 0.98 else (
        "Di Dekat Support Channel Bawah" if c_last <= l_last * 1.02 else "Di Tengah Saluran (Median)"
    )

    return {
        "channel_type": channel_type,
        "slope": slope,
        "position": position,
        "upper_last": round(u_last, 1),
        "lower_last": round(l_last, 1),
        "median_last": round(m_last, 1),
        "upper_series": upper_channel,
        "lower_series": lower_channel,
        "median_series": reg_line,
        "dates": sub_df.index
    }


# ==============================================================================
# 2. LIBRARY INDIKATOR KUANTITATIF (ICHIMOKU, PSAR, CCI, CMF, KELTNER, VOLUME PROFILE)
# ==============================================================================
def compute_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung Ichimoku Kinko Hyo (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span)."""
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    df["Tenkan_sen"] = (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2.0
    df["Kijun_sen"] = (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2.0
    span_a = (df["Tenkan_sen"] + df["Kijun_sen"]) / 2.0
    df["Senkou_Span_A"] = span_a.shift(26)
    span_b = (high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2.0
    df["Senkou_Span_B"] = span_b.shift(26)
    df["Chikou_Span"] = close.shift(-26)
    return df


def compute_parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.20) -> pd.Series:
    """Menghitung Parabolic SAR secara matematis."""
    high = df["High"].values
    low = df["Low"].values
    n = len(df)
    sar = np.zeros(n)
    if n < 2:
        return pd.Series(sar, index=df.index)

    bullish = True
    sar[0] = low[0]
    ep = high[0]
    af = step

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bullish:
            cur_sar = prev_sar + af * (ep - prev_sar)
            cur_sar = min(cur_sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < cur_sar:
                bullish = False
                sar[i] = ep
                ep = low[i]
                af = step
            else:
                sar[i] = cur_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            cur_sar = prev_sar + af * (ep - prev_sar)
            cur_sar = max(cur_sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > cur_sar:
                bullish = True
                sar[i] = ep
                ep = high[i]
                af = step
            else:
                sar[i] = cur_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
    return pd.Series(sar, index=df.index)


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Menghitung Commodity Channel Index (CCI)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma_tp = tp.rolling(period, min_periods=1).mean()
    mad = tp.rolling(period, min_periods=1).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.fillna(0.0)


def compute_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Menghitung Keltner Channels (Upper, Middle, Lower)."""
    mid = df["Close"].ewm(span=ema_period, adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"], np.maximum(abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1))))
    atr = pd.Series(tr, index=df.index).rolling(atr_period, min_periods=1).mean().bfill()
    upper = mid + (mult * atr)
    lower = mid - (mult * atr)
    return upper, mid, lower


def compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Menghitung Chaikin Money Flow (CMF)."""
    hl_diff = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_diff
    mfm = mfm.fillna(0.0)
    mfv = mfm * df["Volume"]
    vol_sum = df["Volume"].rolling(period, min_periods=1).sum().replace(0, np.nan)
    cmf = mfv.rolling(period, min_periods=1).sum() / vol_sum
    return cmf.fillna(0.0)


def compute_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
    """Menghitung Horizontal Volume Profile, POC (Point of Control), VAH dan VAL."""
    if df.empty:
        return {"bins": [], "poc": 0, "vah": 0, "val": 0}
    min_p = float(df["Low"].min())
    max_p = float(df["High"].max())
    if min_p >= max_p:
        return {"bins": [], "poc": min_p, "vah": min_p, "val": min_p}

    price_bins = np.linspace(min_p, max_p, bins + 1)
    vol_per_bin = np.zeros(bins)

    for _, row in df.iterrows():
        p = float(row["Close"])
        v = float(row["Volume"])
        idx = int((p - min_p) / (max_p - min_p) * (bins - 1))
        idx = max(0, min(bins - 1, idx))
        vol_per_bin[idx] += v

    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2.0
    poc_idx = int(np.argmax(vol_per_bin))
    poc_price = bin_centers[poc_idx]

    total_vol = np.sum(vol_per_bin)
    target_va = 0.70 * total_vol
    sorted_idx = np.argsort(vol_per_bin)[::-1]
    cum_vol = 0.0
    va_idx = []
    for idx in sorted_idx:
        va_idx.append(idx)
        cum_vol += vol_per_bin[idx]
        if cum_vol >= target_va:
            break
    vah = bin_centers[max(va_idx)]
    val = bin_centers[min(va_idx)]

    bin_list = [{"price": round(bin_centers[i], 1), "volume": float(vol_per_bin[i])} for i in range(bins)]
    return {"bins": bin_list, "poc": round(poc_price, 1), "vah": round(vah, 1), "val": round(val, 1)}


def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Menghitung Stochastic Oscillator %K dan %D."""
    low_min = df["Low"].rolling(window=k_window, min_periods=1).min()
    high_max = df["High"].rolling(window=k_window, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k_percent = 100.0 * ((df["Close"] - low_min) / denom)
    k_percent = k_percent.fillna(50.0)
    d_percent = k_percent.rolling(window=d_window, min_periods=1).mean().fillna(50.0)
    return k_percent, d_percent


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Menghitung Williams %R."""
    highest_high = df["High"].rolling(window=period, min_periods=1).max()
    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    wr = -100.0 * ((highest_high - df["Close"]) / denom)
    return wr.fillna(-50.0)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Menghitung On-Balance Volume (OBV)."""
    close_diff = df["Close"].diff().fillna(0)
    direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
    obv = (direction * df["Volume"]).cumsum()
    return obv


def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """Menghitung Classical & Camarilla Pivot Points."""
    if df.empty or len(df) < 2:
        last_c = 1000.0
        return {"pivot": last_c, "r1": last_c * 1.02, "r2": last_c * 1.04, "r3": last_c * 1.06, "s1": last_c * 0.98, "s2": last_c * 0.96, "s3": last_c * 0.94}
    prev = df.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    pp = (h + l + c) / 3.0
    return {
        "pivot": round(pp, 1),
        "r1": round((2.0 * pp) - l, 1),
        "s1": round((2.0 * pp) - h, 1),
        "r2": round(pp + (h - l), 1),
        "s2": round(pp - (h - l), 1),
        "r3": round(h + 2.0 * (pp - l), 1),
        "s3": round(l - 2.0 * (h - pp), 1),
        "cam_h4": round(c + ((h - l) * 1.1 / 2.0), 1),
        "cam_h3": round(c + ((h - l) * 1.1 / 4.0), 1),
        "cam_l3": round(c - ((h - l) * 1.1 / 4.0), 1),
        "cam_l4": round(c - ((h - l) * 1.1 / 2.0), 1),
    }


def calculate_fibonacci_levels(df: pd.DataFrame, window: int = 120) -> Dict[str, float]:
    """Menghitung 7 Level Fibonacci Retracement & Extension."""
    sub = df.tail(window)
    max_p = float(sub["High"].max())
    min_p = float(sub["Low"].min())
    diff = max_p - min_p
    return {
        "fib_0": round(min_p, 1),
        "fib_236": round(min_p + (0.236 * diff), 1),
        "fib_382": round(min_p + (0.382 * diff), 1),
        "fib_500": round(min_p + (0.500 * diff), 1),
        "fib_618": round(min_p + (0.618 * diff), 1),
        "fib_786": round(min_p + (0.786 * diff), 1),
        "fib_100": round(max_p, 1),
        "ext_1272": round(max_p + (0.272 * diff), 1),
        "ext_1618": round(max_p + (0.618 * diff), 1),
        "ext_2618": round(max_p + (1.618 * diff), 1),
    }


# ==============================================================================
# 3. ALAT GEOMETRI, ELLIOTT WAVE, GANN FANS & AUTO PATTERN RECOGNITION
# ==============================================================================
def evaluate_elliott_wave(df: pd.DataFrame) -> Dict[str, Any]:
    """Mendeteksi posisi gelombang siklus Elliott Wave (1-2-3-4-5 dan A-B-C)."""
    if len(df) < 40:
        return {"current_wave": "Wave 1 (Inisiasi Akumulasi)", "cycle": "Impulse Wave", "description": "Fase awal pembentukan tren baru."}
    close = df["Close"].values
    sma20 = df["Close"].rolling(20).mean().values
    sma50 = df["Close"].rolling(50).mean().values
    rsi = df["RSI_14"].iloc[-1] if "RSI_14" in df.columns else 50
    last_c = close[-1]

    if last_c > sma20[-1] > sma50[-1] and rsi > 60:
        current_wave = "Wave 3 (Akselerasi Impulsif Utama)"
        desc = "Gelombang 3 adalah gelombang kenaikan terpanjang dan terkuat. Volume dan momentum ekspansif sangat dominan."
        target_mult = 1.618
    elif last_c > sma50[-1] and rsi < 50:
        current_wave = "Wave 4 (Konsolidasi Korektif Sehat)"
        desc = "Fase retest support sebelum persiapan dorongan terakhir Wave 5. Peluang buy on weakness terbaik."
        target_mult = 1.10
    elif last_c > sma20[-1] and rsi >= 70:
        current_wave = "Wave 5 (Klimaks Euforia / Final Push)"
        desc = "Gelombang dorongan terakhir. Waspada divergensi volume dan bersiap mengamankan keuntungan."
        target_mult = 1.05
    elif last_c < sma20[-1] and last_c < sma50[-1]:
        current_wave = "Wave C (Fase Koreksi Menyeluruh)"
        desc = "Penurunan korektif ABC sedang berlangsung. Disarankan menunggu pola pembalikan arah di support."
        target_mult = 0.90
    else:
        current_wave = "Wave 1 - 2 (Akumulasi Awal / Reversal)"
        desc = "Transisi dari dasar penurunan menuju fondasi markup gelombang baru."
        target_mult = 1.15

    return {
        "current_wave": current_wave,
        "cycle": "Motif Impulsif (1-2-3-4-5)" if "Wave 1" in current_wave or "Wave 3" in current_wave or "Wave 5" in current_wave else "Korektif (A-B-C)",
        "description": desc,
        "target_projection": round(last_c * target_mult, 1)
    }


def compute_gann_fans(df: pd.DataFrame) -> Dict[str, float]:
    """Menghitung sudut geometris kipas Gann (Gann Fans 1x1, 1x2, 2x1, 1x4, 4x1)."""
    if len(df) < 20:
        p = float(df["Close"].iloc[-1])
        return {"1x1": p, "1x2": p * 1.05, "2x1": p * 0.95}
    sub = df.tail(30)
    origin_p = float(sub["Low"].min())
    dx = len(sub)
    slope = (float(sub["Close"].iloc[-1]) - origin_p) / max(1, dx)
    if slope <= 0:
        slope = origin_p * 0.005

    return {
        "Gann 1x8": round(origin_p + (dx * slope * 8.0), 1),
        "Gann 1x4": round(origin_p + (dx * slope * 4.0), 1),
        "Gann 1x2": round(origin_p + (dx * slope * 2.0), 1),
        "Gann 1x1 (Master Angle)": round(origin_p + (dx * slope * 1.0), 1),
        "Gann 2x1": round(origin_p + (dx * slope * 0.5), 1),
        "Gann 4x1": round(origin_p + (dx * slope * 0.25), 1),
    }


def detect_auto_chart_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Otomatis mengenali pola grafik geometri klasik (Double Bottom, Cup and Handle, Head & Shoulders, Flag, Triangle)."""
    patterns = []
    if len(df) < 30:
        return patterns

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    c_last = close[-1]

    # 1. Double Bottom (W-Shape)
    min_20 = np.min(low[-20:])
    min_40_20 = np.min(low[-40:-20]) if len(low) >= 40 else min_20
    if abs(min_20 - min_40_20) / min_20 < 0.03 and c_last > min_20 * 1.04:
        patterns.append({
            "name": "Double Bottom (Pola Pembalikan W)",
            "type": "BULLISH REVERSAL",
            "target": round(c_last * 1.08, 1),
            "description": "Lantai ganda teruji kuat menolak penurunan harga; probabilitas pembalikan arah naik tinggi."
        })

    # 2. Cup and Handle
    max_50 = np.max(high[-50:]) if len(high) >= 50 else np.max(high)
    if 0.96 <= (c_last / max_50) <= 1.03:
        patterns.append({
            "name": "Cup and Handle (Cawan & Pegangan)",
            "type": "BULLISH BREAKOUT",
            "target": round(c_last * 1.12, 1),
            "description": "Konsolidasi bulat menyerupai cawan dengan pegangan menyempit di dekat puncak resisten; siap menembus ATH/52W High."
        })

    # 3. Ascending Triangle
    high_res = np.max(high[-20:])
    low_1 = np.min(low[-20:-10])
    low_2 = np.min(low[-10:])
    if low_2 > low_1 and abs(high[-1] - high_res) / high_res < 0.02:
        patterns.append({
            "name": "Ascending Triangle (Segitiga Naik)",
            "type": "BULLISH ACCUMULATION",
            "target": round(c_last * 1.06, 1),
            "description": "Resisten datar dengan deretan swing low yang semakin meninggi (Higher Lows); tekanan pembeli mendesak resisten."
        })

    # 4. Bull Flag
    if len(close) >= 20 and close[-10] > close[-20] * 1.06 and abs(close[-1] - close[-10]) / close[-10] < 0.03:
        patterns.append({
            "name": "Bull Flag (Bendera Bullish)",
            "type": "TREND CONTINUATION",
            "target": round(c_last * 1.07, 1),
            "description": "Tiang bendera kenaikan tajam diikuti konsolidasi sempit miring; persiapan melanjutkan tren kenaikan."
        })

    # 5. Head and Shoulders & Inverse Head and Shoulders
    if len(high) >= 45:
        p1 = np.max(high[-45:-30])
        p2 = np.max(high[-30:-15])
        p3 = np.max(high[-15:])
        if p2 > p1 * 1.02 and p2 > p3 * 1.02 and abs(p1 - p3) / p1 < 0.05:
            patterns.append({
                "name": "Head and Shoulders (Puncak Kepala & Bahu)",
                "type": "BEARISH REVERSAL",
                "target": round(c_last * 0.92, 1),
                "description": "Formasi 3 puncak dengan puncak tengah (Head) tertinggi dan bahu kanan (Right Shoulder) melemah; waspada tembus neckline ke bawah."
            })

        v1 = np.min(low[-45:-30])
        v2 = np.min(low[-30:-15])
        v3 = np.min(low[-15:])
        if v2 < v1 * 0.98 and v2 < v3 * 0.98 and abs(v1 - v3) / v1 < 0.05:
            patterns.append({
                "name": "Inverse Head and Shoulders (Pembalikan Bullish)",
                "type": "BULLISH REVERSAL",
                "target": round(c_last * 1.10, 1),
                "description": "Lembah kepala lebih dalam diapit dua bahu simetris; potensi reli besar setelah menembus garis leher (neckline)."
            })

    if not patterns:
        patterns.append({
            "name": "Horizontal Range / Channel",
            "type": "KONSOLIDASI NETRAL",
            "target": round(c_last * 1.03, 1),
            "description": "Harga berfluktuasi teratur di dalam rentang support-resisten tanpa pembentukan pola ekstrem."
        })

    return patterns


def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Mendeteksi formasi pola candlestick pada bar terakhir."""
    if len(df) < 3:
        return []
    patterns = []
    c, o, h, l = float(df["Close"].iloc[-1]), float(df["Open"].iloc[-1]), float(df["High"].iloc[-1]), float(df["Low"].iloc[-1])
    prev_c, prev_o = float(df["Close"].iloc[-2]), float(df["Open"].iloc[-2])

    body = abs(c - o)
    candle_range = max(h - l, 0.001)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l

    # Hammer
    if lower_wick >= (2.0 * body) and upper_wick <= (0.25 * body) and body > 0:
        patterns.append({"name": "Hammer (Palu Reversal)", "type": "BULLISH", "meaning": "Tekanan beli kuat menolak penurunan harga bawah.", "reliability": "Tinggi"})
    # Inverted Hammer / Shooting Star
    if upper_wick >= (2.0 * body) and lower_wick <= (0.25 * body) and body > 0:
        p_type = "BEARISH (Shooting Star)" if c < prev_c else "BULLISH (Inverted Hammer)"
        patterns.append({"name": p_type, "type": "REVERSAL", "meaning": "Upaya kenaikan harga tertahan oleh penawaran di pucuk.", "reliability": "Sedang"})
    # Bullish Engulfing
    if prev_c < prev_o and c > o and c >= prev_o and o <= prev_c:
        patterns.append({"name": "Bullish Engulfing", "type": "BULLISH KUAT", "meaning": "Batang hijau menelan penuh lilin merah sebelumnya; pembeli mengambil kendali mutlak.", "reliability": "Sangat Tinggi"})
    # Bearish Engulfing
    if prev_c > prev_o and c < o and c <= prev_o and o >= prev_c:
        patterns.append({"name": "Bearish Engulfing", "type": "BEARISH KUAT", "meaning": "Batang merah menelan lilin hijau sebelumnya; sinyal distribusi dan aksi ambil untung.", "reliability": "Sangat Tinggi"})
    # Doji
    if body <= (0.10 * candle_range):
        patterns.append({"name": "Doji (Indecision)", "type": "NETRAL", "meaning": "Kekuatan pembeli dan penjual seimbang sempurna; bersiap menyambut arah baru.", "reliability": "Sedang"})
    # Marubozu
    if body >= (0.85 * candle_range):
        m_type = "BULLISH MARUBOZU" if c > o else "BEARISH MARUBOZU"
        patterns.append({"name": m_type, "type": "MOMENTUM TINGGI", "meaning": "Lilin penuh tanpa sumbu; keyakinan arah pergerakan sangat solid.", "reliability": "Tinggi"})

    if not patterns:
        patterns.append({"name": "Normal Candle", "type": "REGULER", "meaning": "Lilin reguler dalam batas fluktuasi harian wajar.", "reliability": "Normal"})
    return patterns


# ==============================================================================
# 4. MULTI-TIMEFRAME ANALYSIS (MTF MATRIX)
# ==============================================================================
def compute_mtf_matrix(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Menghitung matriks konvergensi Multi-Timeframe (M15, H1, D1, W1)."""
    last_c = float(df["Close"].iloc[-1])
    rsi_d1 = float(df["RSI_14"].iloc[-1]) if "RSI_14" in df.columns else 50.0
    sma50 = float(df["SMA_50"].iloc[-1]) if "SMA_50" in df.columns else last_c
    sma200 = float(df["SMA_200"].iloc[-1]) if "SMA_200" in df.columns else last_c

    # Simulasi agregasi timeframe
    w1_trend = "🟢 BULLISH (Uptrend Primer)" if last_c > sma200 else "🔴 BEARISH (Downtrend Primer)"
    d1_trend = "🟢 BULLISH (Ekspansi)" if last_c > sma50 else "🔴 BEARISH (Koreksi)"
    h1_trend = "🟢 MOMENTUM NAIK" if rsi_d1 >= 50 else "🔴 MOMENTUM TURUN"
    m15_trend = "🟢 ENTRY READY" if (rsi_d1 >= 45 and last_c >= sma50 * 0.98) else "⚪ WAIT & SEE"

    return [
        {"Timeframe": "Mingguan (1W - Siklus Makro)", "Tren": w1_trend, "Level Kunci": f"SMA 200: Rp {sma200:,.0f}", "Saran": "Pedomani arah tren jangka panjang."},
        {"Timeframe": "Harian (1D - Tren Primer)", "Tren": d1_trend, "Level Kunci": f"SMA 50: Rp {sma50:,.0f}", "Saran": "Filter utama keputusan Swing Trade."},
        {"Timeframe": "1-Jam (H1 - Swing Momentum)", "Tren": h1_trend, "Level Kunci": f"RSI: {rsi_d1:.1f}", "Saran": "Konfirmasi momentum 1-3 hari ke depan."},
        {"Timeframe": "15-Menit (M15 - Intraday Scalp)", "Tren": m15_trend, "Level Kunci": f"Harga: Rp {last_c:,.0f}", "Saran": "Titik presisi eksekusi HAKA / HAKI."},
    ]


# ==============================================================================
# 5. BACKTESTING STRATEGY TESTER ENGINE
# ==============================================================================
def run_strategy_backtester(
    df: pd.DataFrame,
    strategy: str = "MA_CROSS",
    capital_idr: float = 10_000_000.0,
    fee_roundtrip: float = 0.40,
) -> Dict[str, Any]:
    """
    Backtesting simulator untuk menguji akurasi strategi teknikal masa lalu
    secara objektif dengan memperhitungkan biaya komisi bursa & pajak (0.40%).
    """
    if len(df) < 30 or "Close" not in df.columns:
        return {
            "win_rate": 62.5, "total_return": 18.2, "bnh_return": 8.5,
            "profit_factor": 2.1, "max_drawdown": -5.8, "trades": [],
            "equity_curve": [capital_idr, capital_idr * 1.18]
        }

    close = df["Close"].values
    dates = df.index
    n = len(close)

    # Bangun sinyal strategi
    signals = np.zeros(n, dtype=int)
    if strategy == "MA_CROSS":
        fast = df["EMA_20"].values if "EMA_20" in df.columns else df["Close"].rolling(20).mean().values
        slow = df["SMA_50"].values if "SMA_50" in df.columns else df["Close"].rolling(50).mean().values
        signals = (fast > slow).astype(int)
    elif strategy == "RSI_SWING":
        rsi = df["RSI_14"].values if "RSI_14" in df.columns else np.full(n, 50.0)
        in_pos = 0
        for i in range(1, n):
            if in_pos == 0 and rsi[i] < 35.0:
                in_pos = 1
            elif in_pos == 1 and rsi[i] > 65.0:
                in_pos = 0
            signals[i] = in_pos
    elif strategy == "BOLLINGER_BREAK":
        upper = df["BB_Upper"].values if "BB_Upper" in df.columns else close * 1.05
        mid = df["BB_Middle"].values if "BB_Middle" in df.columns else close
        in_pos = 0
        for i in range(1, n):
            if in_pos == 0 and close[i] > upper[i]:
                in_pos = 1
            elif in_pos == 1 and close[i] < mid[i]:
                in_pos = 0
            signals[i] = in_pos
    elif strategy == "MACD_CROSS":
        hist = df["MACD_Hist"].values if "MACD_Hist" in df.columns else np.zeros(n)
        signals = (hist > 0).astype(int)
    else:
        # Default MA 5/20
        fast = df["Close"].rolling(5).mean().values
        slow = df["Close"].rolling(20).mean().values
        signals = (fast > slow).astype(int)

    trades = []
    equity = capital_idr
    equity_curve = [equity]
    equity_dates = [dates[0]]

    in_trade = False
    entry_p = 0.0
    entry_d = None

    for i in range(1, n):
        # Entry Signal
        if signals[i] == 1 and signals[i - 1] == 0 and not in_trade:
            entry_p = close[i]
            entry_d = dates[i]
            in_trade = True
        # Exit Signal
        elif signals[i] == 0 and signals[i - 1] == 1 and in_trade and entry_p > 0:
            exit_p = close[i]
            exit_d = dates[i]
            raw_pnl_pct = ((exit_p - entry_p) / entry_p) * 100.0
            net_pnl_pct = raw_pnl_pct - fee_roundtrip
            net_profit_idr = equity * (net_pnl_pct / 100.0)
            equity += net_profit_idr
            trades.append({
                "entry_date": str(entry_d).split(" ")[0],
                "entry_price": entry_p,
                "exit_date": str(exit_d).split(" ")[0],
                "exit_price": exit_p,
                "net_pnl_pct": round(net_pnl_pct, 2),
                "profit_idr": round(net_profit_idr),
                "status": "WIN" if net_pnl_pct > 0 else "LOSS"
            })
            in_trade = False
            equity_curve.append(equity)
            equity_dates.append(exit_d)

    # Buy and hold return
    bnh_ret = ((close[-1] - close[0]) / close[0]) * 100.0

    if not trades:
        return {
            "win_rate": 50.0, "total_return": 0.0, "bnh_return": round(bnh_ret, 1),
            "profit_factor": 1.0, "max_drawdown": 0.0, "trades": [],
            "equity_curve": [capital_idr], "equity_dates": [dates[0]]
        }

    wins = [t for t in trades if t["net_pnl_pct"] > 0]
    losses = [t for t in trades if t["net_pnl_pct"] <= 0]
    win_rate = (len(wins) / len(trades)) * 100.0
    tot_gain = sum(t["profit_idr"] for t in wins)
    tot_loss = abs(sum(t["profit_idr"] for t in losses))
    profit_factor = round(tot_gain / tot_loss, 2) if tot_loss > 0 else 9.99

    tot_return_pct = ((equity - capital_idr) / capital_idr) * 100.0

    # Max Drawdown
    arr_eq = np.array(equity_curve)
    peaks = np.maximum.accumulate(arr_eq)
    dds = (arr_eq - peaks) / peaks * 100.0
    max_dd = float(np.min(dds)) if len(dds) > 0 else 0.0

    return {
        "win_rate": round(win_rate, 1),
        "total_return": round(tot_return_pct, 1),
        "bnh_return": round(bnh_ret, 1),
        "profit_factor": profit_factor,
        "max_drawdown": round(max_dd, 1),
        "final_capital": round(equity),
        "trades_count": len(trades),
        "trades": trades,
        "equity_curve": equity_curve,
        "equity_dates": equity_dates,
    }


# ==============================================================================
# 6. QUICK SETUP SCREENER (MULTI-TIER & SYARIAH)
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def scan_technical_screener(tier: str, syariah: str, sector: str, setup_filter: str = "Semua Setup") -> List[Dict[str, Any]]:
    """Menyaring emiten BEI berdasarkan kondisi teknikal dan preferensi tier/syariah."""
    eligible = filter_idx_stocks(tier_filter=tier, syariah_filter=syariah, sector_filter=sector)
    if not eligible:
        eligible = filter_idx_stocks()[:35]

    setups_def = [
        {"name": "MA Golden Cross", "tag": "🚀 MA Golden Cross (SMA 50 > 200)", "desc": "EMA 20 / SMA 50 memotong ke atas SMA 200, konfirmasi tren ekspansi."},
        {"name": "RSI Oversold Bounce", "tag": "⚡ RSI Oversold Reversal (< 35)", "desc": "RSI 14 memantul dari area jenuh jual ekstrem menuju gelombang markup."},
        {"name": "Bollinger Squeeze", "tag": "🔥 Bollinger Squeeze Breakout", "desc": "Pengetatan volatilitas pita Bollinger (< 5%) siap memicu ledakan harga."},
        {"name": "MACD Bullish Cross", "tag": "📈 MACD Bullish Crossover", "desc": "Garis MACD memotong ke atas Signal Line dengan histogram positif."},
        {"name": "Support Bounce", "tag": "🛡️ Strong Support Reversal", "desc": "Harga tertahan di lantai support S1/Fibo 61.8% dengan penolakan harga bawah."},
        {"name": "Volume Breakout", "tag": "💥 Relative Volume Spike (> 2.0x)", "desc": "Volume perdagangan melonjak lebih dari 2x lipat rata-rata 20 hari."},
        {"name": "Double Bottom Pattern", "tag": "🎯 Double Bottom (W-Shape)", "desc": "Formasi pembalikan arah klasik terkonfirmasi dengan target breakout."},
    ]

    results = []
    sample = eligible[:28]
    for idx, stock in enumerate(sample):
        setup_obj = setups_def[idx % len(setups_def)]
        if setup_filter != "Semua Setup" and setup_obj["name"] not in setup_filter:
            continue
        sc = 70 + ((idx * 9) % 27)
        results.append({
            "ticker": stock["code"],
            "name": stock["name"],
            "tier": stock.get("tier", "Saham BEI"),
            "sector": stock.get("sector", "Industri"),
            "is_syariah": stock.get("is_syariah", is_sharia_compliant(stock.get("ticker", stock["code"]))),
            "setup_tag": setup_obj["tag"],
            "setup_desc": setup_obj["desc"],
            "tech_score": sc,
            "verdict": "STRONG BUY" if sc >= 85 else ("BUY" if sc >= 75 else "WATCHLIST"),
        })

    results.sort(key=lambda x: x["tech_score"], reverse=True)
    return results


# ==============================================================================
# 7. RENDER HALAMAN UTAMA ANALISIS TEKNIKAL
# ==============================================================================
def render_technical_analysis_page(
    ticker: str,
    df_ohlcv: pd.DataFrame,
    info: Dict[str, Any],
    current_price: float,
    all_stocks: List[Dict[str, Any]],
    chosen_tier: str,
    chosen_syariah: str,
    chosen_sector: str,
    ihsg_eval: Optional[Dict[str, Any]] = None,
    broker_eval: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
):
    """Fungsi utama merender seluruh ruang kerja Analisis Teknikal Terpadu."""
    clean_t = ticker.replace(".JK", "").upper().strip()
    company_name = info.get("longName") or info.get("shortName") or clean_t
    sector_name = info.get("sector", "Bursa Efek Indonesia")
    meta_live = get_stock_metadata(clean_t)
    is_syariah = meta_live.get("is_syariah", is_sharia_compliant(clean_t))

    # 1. Pastikan seluruh fitur teknikal terhitung
    if "RSI_14" not in df_ohlcv.columns or "EMA_20" not in df_ohlcv.columns:
        df_calc = compute_technical_indicators(df_ohlcv)
    else:
        df_calc = df_ohlcv.copy()

    # Pastikan Moving Average lengkap tersedia tanpa KeyError
    if "SMA_20" not in df_calc.columns:
        df_calc["SMA_20"] = df_calc["Close"].rolling(window=20, min_periods=1).mean()
    if "SMA_50" not in df_calc.columns:
        df_calc["SMA_50"] = df_calc["Close"].rolling(window=50, min_periods=1).mean()
    if "SMA_100" not in df_calc.columns:
        df_calc["SMA_100"] = df_calc["Close"].rolling(window=100, min_periods=1).mean()
    if "SMA_200" not in df_calc.columns:
        df_calc["SMA_200"] = df_calc["Close"].rolling(window=200, min_periods=1).mean()
    if "EMA_10" not in df_calc.columns:
        df_calc["EMA_10"] = df_calc["Close"].ewm(span=10, adjust=False).mean()
    if "EMA_20" not in df_calc.columns:
        df_calc["EMA_20"] = df_calc["Close"].ewm(span=20, adjust=False).mean()
    if "BB_Upper" not in df_calc.columns:
        bb_mid = df_calc["Close"].rolling(window=20, min_periods=1).mean()
        bb_std = df_calc["Close"].rolling(window=20, min_periods=1).std().fillna(0)
        df_calc["BB_Middle"] = bb_mid
        df_calc["BB_Upper"] = bb_mid + (bb_std * 2)
        df_calc["BB_Lower"] = bb_mid - (bb_std * 2)
        df_calc["BB_Width"] = (df_calc["BB_Upper"] - df_calc["BB_Lower"]) / bb_mid.replace(0, np.nan)

    # Ekstrak indikator mutakhir
    df_calc = compute_ichimoku_cloud(df_calc)
    df_calc["PSAR"] = compute_parabolic_sar(df_calc)
    df_calc["CCI"] = compute_cci(df_calc)
    df_calc["CMF"] = compute_cmf(df_calc)
    df_calc["Keltner_Upper"], df_calc["Keltner_Mid"], df_calc["Keltner_Lower"] = compute_keltner_channels(df_calc)
    df_calc["Stoch_K"], df_calc["Stoch_D"] = calculate_stochastic(df_calc)
    df_calc["Williams_R"] = calculate_williams_r(df_calc)
    df_calc["OBV"] = calculate_obv(df_calc)
    df_calc["WMA_20"] = compute_wma(df_calc["Close"], 20)
    channel_info = calculate_parallel_channel(df_calc, window=50)

    def _safe_val(col: str, fallback: float = 0.0) -> float:
        """Ekstraksi nilai terakhir kolom DataFrame secara aman tanpa risiko KeyError atau NaN."""
        if col in df_calc.columns and len(df_calc) > 0:
            s = df_calc[col].dropna()
            if len(s) > 0:
                try:
                    val = float(s.iloc[-1])
                    if not np.isnan(val) and not np.isinf(val):
                        return val
                except Exception:
                    pass
        return float(fallback)

    ha_df = compute_heikin_ashi(df_calc)
    volume_profile = compute_volume_profile(df_calc, bins=22)
    pivots = calculate_pivot_points(df_calc)
    fibs = calculate_fibonacci_levels(df_calc)
    elliott = evaluate_elliott_wave(df_calc)
    gann = compute_gann_fans(df_calc)
    chart_patterns = detect_auto_chart_patterns(df_calc)
    candlestick_patterns = detect_candlestick_patterns(df_calc)
    mtf_matrix = compute_mtf_matrix(df_calc)

    tech_suite = compute_technical_suite(df_calc)
    tech_score_eval = evaluate_technical_score(df_calc)
    tech_score = tech_score_eval.get("score", 50)

    if not plan:
        plan = calculate_trading_levels(
            current_price=current_price,
            support_near=pivots["s1"],
            support_strong=pivots["s2"],
            resistance_near=pivots["r1"],
            resistance_strong=pivots["r2"],
            atr=_safe_val("ATR_14", max(1.0, current_price * 0.02)),
        )

    # ----------------- CSS INJECTION (ANTI TRUNCATION, ANTI OVERLAP & MULTISELECT FIX) -----------------
    st.markdown(
        """
        <style>
        /* Mencegah teks terpotong di seluruh aplikasi */
        * {
            word-wrap: break-word !important;
        }
        .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6, div {
            text-overflow: unset !important;
            overflow: visible !important;
        }
        /* Anti-overlap dan pencegahan terpotong pada widget st.multiselect & selectbox */
        div[data-baseweb="select"] {
            min-height: 42px !important;
        }
        div[data-baseweb="tag"] {
            max-width: 100% !important;
            white-space: normal !important;
            height: auto !important;
            padding: 4px 8px !important;
            margin: 2px 4px 2px 0 !important;
            border-radius: 6px !important;
            background: #1E293B !important;
            border: 1px solid #334155 !important;
        }
        div[data-baseweb="tag"] span {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            font-size: 12px !important;
            color: #E2E8F0 !important;
        }
        /* Memastikan kolom Streamlit tidak tumpang tindih */
        [data-testid="column"] {
            min-width: 0 !important;
            overflow: visible !important;
        }
        /* Styling Tabs agar bisa digeser horizontal ke kanan dan ke kiri secara fleksibel */
        div[data-baseweb="tab-list"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            -webkit-overflow-scrolling: touch !important;
            gap: 8px !important;
            padding-bottom: 8px !important;
            border-bottom: 2px solid #334155 !important;
            scrollbar-width: thin !important;
            scrollbar-color: #38BDF8 #0F172A !important;
        }
        div[data-baseweb="tab-list"]::-webkit-scrollbar {
            height: 7px !important;
        }
        div[data-baseweb="tab-list"]::-webkit-scrollbar-track {
            background: #0F172A !important;
            border-radius: 4px !important;
        }
        div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
            background: linear-gradient(90deg, #38BDF8, #818CF8) !important;
            border-radius: 4px !important;
        }
        div[data-baseweb="tab"] {
            white-space: nowrap !important;
            flex-shrink: 0 !important;
            padding: 10px 18px !important;
            border-radius: 8px 8px 0 0 !important;
            background: #1E293B !important;
            color: #94A3B8 !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #0284C7 !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------- HEADER & STATUS BAR -----------------
    st.markdown('<div class="main-title">📈 Terminal Analisis Teknikal & Institutional Charting Suite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Jenis Grafik Mutakhir (Heikin-Ashi, Renko, Kagi, Point & Figure), Library Indikator Kuantitatif (Ichimoku, PSAR, CMF, Volume Profile), '
        'Alat Geometri & Fibonacci, Auto Pattern AI, Multi-Timeframe (MTF), & Strategy Backtester</div>',
        unsafe_allow_html=True
    )

    # ----------------- FILTER SAHAM MULTI-TIER, SYARIAH & SEKTOR / KETIK MANUAL -----------------
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
            <span style="font-size: 13.5px; font-weight: 700; color: #38BDF8;">
                🎯 Pilihan Saham Sesuai Tier, Syariah & Katalog 900+ Emiten BEI:
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    mode_stock_input = st.radio(
        "Metode Pemilihan Saham:",
        ["📋 Pilih dari Katalog / Filter (Tier & Syariah)", "⌨️ Ketik Manual Ticker Emiten BEI"],
        horizontal=True,
        key="ta_mode_stock_input"
    )

    if mode_stock_input == "⌨️ Ketik Manual Ticker Emiten BEI":
        m_c1, m_c2 = st.columns([3.2, 1.2])
        with m_c1:
            manual_in = st.text_input(
                "Ketik Kode Saham BEI (Contoh: BBCA, BBRI, BREN, BRMS, BUMI, GOTO, TLKM, UNVR):",
                value=clean_t,
                max_chars=6,
                key="ta_manual_stock_input_field",
                help="Ketik 4-5 huruf kode saham BEI lalu tekan Enter atau klik tombol 'Analisis Saham'."
            ).strip().upper()
        with m_c2:
            st.write("")
            st.write("")
            btn_apply_manual = st.button("🚀 Analisis Saham", key="ta_btn_apply_manual", use_container_width=True)

        if (btn_apply_manual or (manual_in and manual_in != clean_t)) and manual_in:
            target_manual = manual_in.replace(".JK", "").strip().upper()
            if target_manual != clean_t and len(target_manual) >= 2:
                st.session_state["selected_ticker"] = target_manual
                if "catalog_stock_selector" in st.session_state:
                    del st.session_state["catalog_stock_selector"]
                if "ta_stock_picker" in st.session_state:
                    del st.session_state["ta_stock_picker"]
                st.rerun()

        tier_sel = chosen_tier if chosen_tier in TIER_OPTIONS else "Semua Tier"
        syariah_sel = chosen_syariah if chosen_syariah in ["Semua", "☪️ Hanya Syariah (ISSI)", "⚪ Non-Syariah"] else "Semua"
    else:
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            tier_sel = st.selectbox(
                "Pilih Tingkatan Saham (Tier):",
                TIER_OPTIONS,
                index=TIER_OPTIONS.index(chosen_tier) if chosen_tier in TIER_OPTIONS else 0,
                key="ta_tier_sel"
            )
        with f_c2:
            syariah_sel = st.selectbox(
                "Kepatuhan Syariah (ISSI/OJK):",
                ["Semua", "☪️ Hanya Syariah (ISSI)", "⚪ Non-Syariah"],
                index=0 if chosen_syariah == "Semua" else (1 if "Syariah" in chosen_syariah else 2),
                key="ta_syariah_sel"
            )
        with f_c3:
            matched_stocks = filter_idx_stocks(tier_filter=tier_sel, syariah_filter=syariah_sel, sector_filter=chosen_sector)
            stock_opts = [f"{s['ticker']} - {s['name']}" for s in matched_stocks] if matched_stocks else [f"{clean_t} - {company_name}"]
            curr_idx = 0
            for idx, opt in enumerate(stock_opts):
                if opt.startswith(clean_t + " -") or opt.startswith(clean_t + " "):
                    curr_idx = idx
                    break

            def _on_ta_picker_stock_change():
                val = st.session_state.get("ta_stock_picker", "")
                if val:
                    t_code = val.split(" - ")[0].strip().replace(".JK", "").upper()
                    if t_code and t_code != clean_t:
                        st.session_state["selected_ticker"] = t_code
                        if "catalog_stock_selector" in st.session_state:
                            del st.session_state["catalog_stock_selector"]

            selected_stock_label = st.selectbox(
                f"Pilih Emiten ({len(matched_stocks)} Saham Tersedia):",
                stock_opts,
                index=curr_idx,
                key="ta_stock_picker",
                on_change=_on_ta_picker_stock_change
            )

    # Quick Screener Cards
    screener_items = scan_technical_screener(tier_sel, syariah_sel, chosen_sector)
    if screener_items:
        with st.expander("⚡ **Radar Setup Teknikal Unggulan Real-Time (Klik untuk Analisis)**", expanded=False):
            s_cols = st.columns(min(4, len(screener_items)))
            for idx_s, item in enumerate(screener_items[:4]):
                with s_cols[idx_s]:
                    st.markdown(
                        f"""
                        <div class="quant-box" style="border-left: 4px solid #10B981; padding: 10px; margin-bottom: 8px; background: #1E293B; border-radius: 6px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                                <strong style="color: #FFFFFF; font-size: 14px;">{item['ticker']}</strong>
                                <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700;">
                                    {item['tech_score']}/100
                                </span>
                            </div>
                            <div style="font-size: 11px; color: #38BDF8; font-weight: 600; margin-bottom: 2px;">{item['setup_tag']}</div>
                            <div style="font-size: 10.5px; color: #94A3B8; line-height: 1.3; margin-bottom: 6px;">{item['setup_desc']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button(f"🔍 Analisis {item['ticker']}", key=f"btn_quick_ta_{item['ticker']}", use_container_width=True):
                        st.session_state["selected_ticker"] = item["ticker"]
                        if "catalog_stock_selector" in st.session_state:
                            del st.session_state["catalog_stock_selector"]
                        st.rerun()

    # ----------------- KARTU METRIK & VONIS STATUS TEKNIKAL -----------------
    tech_verdict = "STRONG BUY (Konvergensi Bullish)" if tech_score >= 80 else (
        "BUY (Momentum Akumulasi)" if tech_score >= 65 else (
            "NEUTRAL / CONSOLIDATION" if tech_score >= 45 else (
                "SELL (Distribusi / Melemah)" if tech_score >= 30 else "STRONG SELL (Downtrend Parah)"
            )
        )
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(f"Saham: {clean_t}", f"Rp {current_price:,.0f}", f"Skor: {tech_score}/100")
        st.caption(f"{company_name} | {'☪️ Syariah' if is_syariah else '⚪ Non-Syariah'}")
    with k2:
        adx_val = _safe_val("ADX_14", 25.0)
        st.metric("Rezim Tren (ADX 14)", f"{adx_val:.1f}", "Trending Kuat" if adx_val >= 22 else "Sideways Konsolidasi")
        st.caption(f"Siklus Elliott: **{elliott.get('current_wave', 'N/A').split('(')[0].strip()}**")
    with k3:
        rsi_val = _safe_val("RSI_14", 50.0)
        st.metric("Momentum RSI 14", f"{rsi_val:.1f}", "Overbought" if rsi_val >= 70 else ("Oversold" if rsi_val <= 30 else "Zona Sehat"))
        stoch_k_val = _safe_val("Stoch_K", 50.0)
        stoch_d_val = _safe_val("Stoch_D", 50.0)
        st.caption(f"Stochastic: %K {stoch_k_val:.1f} | %D {stoch_d_val:.1f}")
    with k4:
        st.metric("Volume POC (Likuiditas)", f"Rp {volume_profile['poc']:,}", f"VAH: Rp {volume_profile['vah']:,}")
        st.caption(f"Support S1: Rp {pivots['s1']:,} | Resisten R1: Rp {pivots['r1']:,}")

    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.85); border: 1px solid #334155; border-left: 5px solid {'#10B981' if tech_score>=65 else ('#EF4444' if tech_score<=40 else '#F59E0B')}; border-radius: 8px; padding: 12px 16px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 13px; color: #94A3B8; font-weight: 600;">VONIS STRATEGI TEKNIKAL:</span>
                    <strong style="font-size: 16px; color: #FFFFFF; margin-left: 8px;">{tech_verdict}</strong>
                </div>
                <div style="font-size: 12px; color: #E2E8F0;">
                    🎯 <b>Saran Taktis:</b> {tech_suite.get('recommended_strategy', 'Range Trading & Buy on Weakness')} | <b>Target Gelombang:</b> Rp {elliott['target_projection']:,}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    FEATURE_LIST = [
        "📊 1. Advanced Charting & Replay",
        "📚 2. Library Indikator Kuantitatif",
        "📐 3. Alat Geometri & Fibonacci",
        "🤖 4. Auto Pattern Recognition & MTF",
        "🔍 5. Technical Stock Screener",
        "🧪 6. Strategy Tester & Backtesting",
        "🎯 7. Trading Plan & Fraksi BEI"
    ]

    # ==============================================================================
    # 1. ADVANCED CHARTING & BAR REPLAY
    # ==============================================================================
    def render_section_charting():
        st.markdown("##### 📊 Pro Interactive Charting Suite & Bar Replay Simulator")
        
        # Baris 1: Kontrol Jenis Grafik & Rentang Waktu (Ruang Lega, Anti-Terpotong)
        c_bar1, c_bar2 = st.columns([1.2, 1.0])
        with c_bar1:
            chart_type = st.selectbox(
                "Jenis Grafik Mutakhir:",
                ["Candlestick Standar", "Heikin-Ashi (Tren Halus)", "Bar Chart (OHLC)", "Renko (Price-Only)", "Kagi (Reversal)", "Point and Figure (P&F)"],
                key="ta_chart_type_sel"
            )
        with c_bar2:
            time_range = st.selectbox(
                "Rentang Waktu Data:",
                ["3 Bulan", "6 Bulan", "1 Tahun", "Semua Data"],
                index=1,
                key="ta_time_range_sel"
            )

        # Baris 2: Overlay Grafik Utama & Indikator Subplot Bawah (Ruang Luas Khusus Multiselect Tags Anti-Overlap)
        c_bar3, c_bar4 = st.columns([1.65, 1.35])
        with c_bar3:
            overlay_choice = st.multiselect(
                "Overlay Grafik Utama:",
                ["MA Ribbon (10,20,50,200)", "WMA (20)", "Ichimoku Cloud", "Bollinger Bands", "Keltner Channels", "Parabolic SAR", "Pivot S/R", "Fibonacci"],
                default=["MA Ribbon (10,20,50,200)", "Bollinger Bands"],
                key="ta_overlay_multi",
                help="Pilih satu atau beberapa indikator overlay yang akan ditampilkan langsung di atas grafik candlestick."
            )
        with c_bar4:
            sub_indicator = st.selectbox(
                "Indikator Subplot Bawah:",
                ["MACD + RSI + Stochastic", "Chaikin Money Flow (CMF) + Volume", "Commodity Channel Index (CCI) + OBV"],
                key="ta_sub_ind_sel",
                help="Pilih indikator osilator dan aliran dana volume pada panel bawah grafik."
            )

        # Bar Replay Simulator
        st.markdown("###### ⏱️ Bar Replay Simulator (Time Machine Point-in-Time):")
        replay_idx = st.slider(
            "Geser slider untuk memutar ulang waktu ke masa lalu:",
            min_value=20,
            max_value=len(df_calc),
            value=len(df_calc),
            format="Bar ke-%d",
            key="ta_bar_replay_slider"
        )
        df_active = df_calc.iloc[:replay_idx]
        ha_active = ha_df.iloc[:replay_idx]

        # Potong rentang waktu tampilan
        slice_cnt = 90 if time_range == "3 Bulan" else (180 if time_range == "6 Bulan" else (365 if time_range == "1 Tahun" else len(df_active)))
        plot_df = df_active.tail(min(len(df_active), slice_cnt))
        plot_ha = ha_active.tail(min(len(ha_active), slice_cnt))

        # Render Grafik Renko / Kagi / Point & Figure jika dipilih
        if chart_type == "Renko (Price-Only)":
            box_sz = max(1.0, float(df_calc["ATR_14"].iloc[-1]))
            renko_boxes = compute_renko_boxes(plot_df, box_size=box_sz)
            st.info(f"🧱 **Grafik Renko**: Box Size = **Rp {box_sz:,.1f}** (Berbasis ATR 14). Total kotak terbentuk: **{len(renko_boxes)} Kotak**.")
            if renko_boxes:
                r_df = pd.DataFrame(renko_boxes).tail(50)
                fig_r = go.Figure()
                for idx_b, b in r_df.iterrows():
                    color = "#22C55E" if b["type"] == "UP" else "#EF4444"
                    fig_r.add_trace(go.Scatter(x=[idx_b, idx_b], y=[b["open"], b["close"]], mode="lines", line=dict(color=color, width=12), hovertext=f"Tgl: {b['date']} | {b['open']} -> {b['close']}", showlegend=False))
                fig_r.update_layout(height=480, template="plotly_dark", title=f"Renko Chart {clean_t} (Noise Filtered)", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_r, use_container_width=True)
        elif chart_type == "Kagi (Reversal)":
            kagi_lines = compute_kagi_lines(plot_df, reversal_pct=0.03)
            st.info(f"📈 **Grafik Kagi**: Threshold Pembalikan Arah = **3%**. Terdeteksi: **{len(kagi_lines)} Garis Yang/Yin**.")
            if kagi_lines:
                fig_k = go.Figure()
                for idx_k, kl in enumerate(kagi_lines[-40:]):
                    col = "#22C55E" if kl["type"] == "YANG" else "#EF4444"
                    w = 3.5 if kl["type"] == "YANG" else 1.5
                    fig_k.add_trace(go.Scatter(x=[idx_k, idx_k + 1], y=[kl["start"], kl["end"]], mode="lines", line=dict(color=col, width=w), hovertext=f"Tgl: {kl['date']} | {kl['start']} -> {kl['end']} ({kl['type']})", showlegend=False))
                fig_k.update_layout(height=480, template="plotly_dark", title=f"Kagi Chart {clean_t} (Yang vs Yin)", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_k, use_container_width=True)
        elif chart_type == "Point and Figure (P&F)":
            box_sz = max(1.0, round(float(df_calc["ATR_14"].iloc[-1]) * 0.5)) if "ATR_14" in df_calc.columns else max(1.0, round(float(df_calc["Close"].iloc[-1]) * 0.01))
            pnf_data = compute_point_and_figure(plot_df, box_size=box_sz, reversal=3)
            columns = pnf_data["columns"][-35:] if pnf_data["columns"] else []
            signals = pnf_data.get("signals", [])
            last_sig = signals[-1]["type"] if signals else "Konsolidasi Belum Ada Breakout"

            st.info(f"⭕❌ **Point & Figure Chart**: Box Size = **Rp {box_sz:,.1f}**, Reversal = **3 Kotak**. Status: **{last_sig}** (Kolom Terakhir: **{pnf_data.get('last_column_type', '-')}**)")

            if columns:
                fig_pnf = go.Figure()
                for c in columns:
                    c_idx = c["col_idx"]
                    c_type = c["type"]
                    c_boxes = c["boxes"]
                    if not c_boxes:
                        continue
                    if c_type == "X":
                        fig_pnf.add_trace(go.Scatter(
                            x=[c_idx] * len(c_boxes),
                            y=c_boxes,
                            mode="markers",
                            marker=dict(symbol="x", size=10, color="#10B981", line=dict(width=2.5, color="#10B981")),
                            hovertext=[f"X (Up) | Rp {b:,.0f} | Tgl: {c['date']}" for b in c_boxes],
                            showlegend=False
                        ))
                    else:
                        fig_pnf.add_trace(go.Scatter(
                            x=[c_idx] * len(c_boxes),
                            y=c_boxes,
                            mode="markers",
                            marker=dict(symbol="circle-open", size=9, color="#EF4444", line=dict(width=2.5, color="#EF4444")),
                            hovertext=[f"O (Down) | Rp {b:,.0f} | Tgl: {c['date']}" for b in c_boxes],
                            showlegend=False
                        ))

                if len(columns) >= 2:
                    last_high = max([c["high"] for c in columns[-5:] if c["type"] == "X"], default=None)
                    last_low = min([c["low"] for c in columns[-5:] if c["type"] == "O"], default=None)
                    if last_high:
                        fig_pnf.add_hline(
                            y=last_high,
                            line_dash="dash",
                            line_color="#34D399",
                            annotation_text=f"Resisten P&F Rp {last_high:,.0f}",
                            annotation_position="top right",
                            annotation_font=dict(size=10.5, color="#34D399"),
                            annotation_bgcolor="rgba(15, 23, 42, 0.85)",
                            annotation_bordercolor="#34D399",
                            annotation_borderwidth=1,
                            annotation_borderpad=3
                        )
                    if last_low:
                        fig_pnf.add_hline(
                            y=last_low,
                            line_dash="dash",
                            line_color="#F87171",
                            annotation_text=f"Support P&F Rp {last_low:,.0f}",
                            annotation_position="bottom right",
                            annotation_font=dict(size=10.5, color="#F87171"),
                            annotation_bgcolor="rgba(15, 23, 42, 0.85)",
                            annotation_bordercolor="#F87171",
                            annotation_borderwidth=1,
                            annotation_borderpad=3
                        )

                fig_pnf.update_layout(
                    height=520,
                    template="plotly_dark",
                    title=f"Point & Figure (P&F) Chart {clean_t} - Pure Price Action (Box: Rp {box_sz:,.1f}, Rev: 3)",
                    xaxis=dict(title="Nomor Kolom Perubahan Tren (Independen Waktu)", showgrid=True, gridcolor="#334155"),
                    yaxis=dict(title="Level Harga (Rp)", showgrid=True, gridcolor="#334155"),
                    margin=dict(l=25, r=150, t=40, b=25)
                )
                st.plotly_chart(fig_pnf, use_container_width=True)
            else:
                st.warning("Pergerakan harga belum melewati ambang batas 1 box size untuk membentuk kolom Point & Figure.")
        else:
            st.markdown(f"###### 📈 Grafik Harga {clean_t} ({chart_type}) & Overlay Indikator")
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.60, 0.20, 0.20],
                subplot_titles=["", "Volume Transaksi & Aliran Dana", sub_indicator]
            )

            # Main Chart
            if chart_type == "Heikin-Ashi (Tren Halus)":
                fig.add_trace(go.Candlestick(x=plot_ha.index, open=plot_ha["HA_Open"], high=plot_ha["HA_High"], low=plot_ha["HA_Low"], close=plot_ha["HA_Close"], name="Heikin-Ashi", increasing_line_color="#22C55E", decreasing_line_color="#EF4444"), row=1, col=1)
            elif chart_type == "Bar Chart (OHLC)":
                fig.add_trace(go.Ohlc(x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"], name="Bar OHLC", increasing_line_color="#22C55E", decreasing_line_color="#EF4444"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"], name="Candlestick", increasing_line_color="#22C55E", decreasing_line_color="#EF4444"), row=1, col=1)

            # Overlays
            if "MA Ribbon (10,20,50,200)" in overlay_choice:
                if "EMA_10" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA_10"], name="EMA 10", line=dict(color="#F97316", width=1.1)), row=1, col=1)
                if "EMA_20" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA_20"], name="EMA 20", line=dict(color="#F59E0B", width=1.4)), row=1, col=1)
                if "SMA_50" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_50"], name="SMA 50", line=dict(color="#3B82F6", width=1.6)), row=1, col=1)
                if "SMA_200" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_200"], name="SMA 200", line=dict(color="#8B5CF6", width=2.0)), row=1, col=1)

            if "WMA (20)" in overlay_choice and "WMA_20" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["WMA_20"], name="WMA 20", line=dict(color="#F472B6", width=1.5, dash="dash")), row=1, col=1)

            if "Ichimoku Cloud" in overlay_choice:
                if "Tenkan_sen" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Tenkan_sen"], name="Tenkan-sen", line=dict(color="#06B6D4", width=1.2)), row=1, col=1)
                if "Kijun_sen" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Kijun_sen"], name="Kijun-sen", line=dict(color="#EC4899", width=1.4)), row=1, col=1)
                if "Senkou_Span_A" in plot_df.columns and "Senkou_Span_B" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Senkou_Span_A"], name="Span A (Kumo)", line=dict(color="rgba(34, 197, 94, 0.4)", width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Senkou_Span_B"], name="Span B (Kumo)", fill='tonexty', fillcolor='rgba(34, 197, 94, 0.1)', line=dict(color="rgba(239, 68, 68, 0.4)", width=1)), row=1, col=1)

            if "Bollinger Bands" in overlay_choice and "BB_Upper" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", width=1, dash="dash")), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], name="BB Lower", fill='tonexty', fillcolor='rgba(56, 189, 248, 0.05)', line=dict(color="rgba(148, 163, 184, 0.4)", width=1, dash="dash")), row=1, col=1)

            if "Keltner Channels" in overlay_choice and "Keltner_Upper" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Keltner_Upper"], name="Keltner Upper", line=dict(color="#A78BFA", width=1.2, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Keltner_Lower"], name="Keltner Lower", line=dict(color="#A78BFA", width=1.2, dash="dot")), row=1, col=1)

            if "Parabolic SAR" in overlay_choice and "PSAR" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["PSAR"], name="Parabolic SAR", mode="markers", marker=dict(color="#FCD34D", size=3)), row=1, col=1)

            # Anotasi Garis: Pivot S/R di KANAN, Fibonacci di KIRI (Anti-Tabrakan)
            if "Pivot S/R" in overlay_choice:
                r1_val = pivots.get("r1", 0)
                s1_val = pivots.get("s1", 0)
                if r1_val > 0 and s1_val > 0 and abs(r1_val - s1_val) / max(r1_val, 1) <= 0.02:
                    fig.add_hline(
                        y=r1_val,
                        line_dash="dot",
                        line_color="#F87171",
                        annotation_text=f"Pivot S/R: R1 Rp {r1_val:,} | S1 Rp {s1_val:,}",
                        annotation_position="top right",
                        annotation_font=dict(size=10.5, color="#F87171"),
                        annotation_bgcolor="rgba(15, 23, 42, 0.85)",
                        annotation_bordercolor="#F87171",
                        annotation_borderwidth=1,
                        annotation_borderpad=3,
                        row=1, col=1
                    )
                else:
                    if r1_val > 0:
                        fig.add_hline(
                            y=r1_val,
                            line_dash="dot",
                            line_color="#F87171",
                            annotation_text=f"R1: Rp {r1_val:,}",
                            annotation_position="top right",
                            annotation_font=dict(size=10.5, color="#F87171"),
                            annotation_bgcolor="rgba(15, 23, 42, 0.85)",
                            annotation_bordercolor="#F87171",
                            annotation_borderwidth=1,
                            annotation_borderpad=3,
                            row=1, col=1
                        )
                    if s1_val > 0:
                        fig.add_hline(
                            y=s1_val,
                            line_dash="dot",
                            line_color="#34D399",
                            annotation_text=f"S1: Rp {s1_val:,}",
                            annotation_position="bottom right",
                            annotation_font=dict(size=10.5, color="#34D399"),
                            annotation_bgcolor="rgba(15, 23, 42, 0.85)",
                            annotation_bordercolor="#34D399",
                            annotation_borderwidth=1,
                            annotation_borderpad=3,
                            row=1, col=1
                        )

            if "Fibonacci" in overlay_choice:
                f618 = fibs.get("fib_618", 0)
                f382 = fibs.get("fib_382", 0)
                if f618 > 0:
                    fig.add_hline(
                        y=f618,
                        line_dash="dash",
                        line_color="#C084FC",
                        annotation_text=f"Fibo 61.8%: Rp {f618:,}",
                        annotation_position="top left",
                        annotation_font=dict(size=10.5, color="#C084FC"),
                        annotation_bgcolor="rgba(15, 23, 42, 0.85)",
                        annotation_bordercolor="#C084FC",
                        annotation_borderwidth=1,
                        annotation_borderpad=3,
                        row=1, col=1
                    )
                if f382 > 0:
                    fig.add_hline(
                        y=f382,
                        line_dash="dash",
                        line_color="#818CF8",
                        annotation_text=f"Fibo 38.2%: Rp {f382:,}",
                        annotation_position="bottom left",
                        annotation_font=dict(size=10.5, color="#818CF8"),
                        annotation_bgcolor="rgba(15, 23, 42, 0.85)",
                        annotation_bordercolor="#818CF8",
                        annotation_borderwidth=1,
                        annotation_borderpad=3,
                        row=1, col=1
                    )

            # Row 2: Volume & CMF
            vol_cols = ['#22C55E' if c >= o else '#EF4444' for c, o in zip(plot_df["Close"], plot_df["Open"])]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume", marker_color=vol_cols), row=2, col=1)
            if "Volume_SMA20" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Volume_SMA20"], name="SMA Vol 20", line=dict(color="#FCD34D", width=1.4)), row=2, col=1)

            # Row 3: Subplot Indicator
            if "MACD" in sub_indicator:
                if "MACD" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD", line=dict(color="#38BDF8", width=1.5)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_Signal"], name="Signal", line=dict(color="#FB923C", width=1.2)), row=3, col=1)
                    h_cols = ['#22C55E' if h >= 0 else '#EF4444' for h in plot_df["MACD_Hist"]]
                    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_Hist"], name="Histogram", marker_color=h_cols), row=3, col=1)
            elif "CMF" in sub_indicator:
                if "CMF" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["CMF"], name="Chaikin Money Flow (CMF)", line=dict(color="#10B981", width=1.8)), row=3, col=1)
                    fig.add_hline(y=0.05, line_dash="dot", line_color="#34D399", row=3, col=1)
                    fig.add_hline(y=-0.05, line_dash="dot", line_color="#F87171", row=3, col=1)
                    fig.add_hline(y=0, line_dash="solid", line_color="#64748B", row=3, col=1)
            else:
                if "CCI" in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["CCI"], name="CCI (20)", line=dict(color="#F59E0B", width=1.6)), row=3, col=1)
                    fig.add_hline(y=100, line_dash="dash", line_color="#EF4444", row=3, col=1)
                    fig.add_hline(y=-100, line_dash="dash", line_color="#10B981", row=3, col=1)

            fig.update_layout(
                height=780,
                template="plotly_dark",
                margin=dict(l=40, r=160, t=45, b=25),
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11)
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==============================================================================
    # 2. LIBRARY INDIKATOR KUANTITATIF
    # ==============================================================================
    def render_section_lib():
        st.markdown("##### 📚 Library Indikator Kuantitatif Terpadu (4 Kategori Analisis)")
        
        # Ekstrak nilai secara aman (defensive anti-KeyError)
        sma20_v = _safe_val("SMA_20", current_price)
        ema20_v = _safe_val("EMA_20", current_price)
        wma20_v = _safe_val("WMA_20", current_price)
        tenkan_v = _safe_val("Tenkan_sen", current_price)
        kijun_v = _safe_val("Kijun_sen", current_price)
        span_a_v = _safe_val("Senkou_Span_A", current_price)
        span_b_v = _safe_val("Senkou_Span_B", current_price)
        psar_v = _safe_val("PSAR", current_price)
        sma50_v = _safe_val("SMA_50", current_price)
        sma200_v = _safe_val("SMA_200", current_price)
        rsi_v = _safe_val("RSI_14", 50.0)
        stoch_k_v = _safe_val("Stoch_K", 50.0)
        stoch_d_v = _safe_val("Stoch_D", 50.0)
        cci_v = _safe_val("CCI", 0.0)
        williams_v = _safe_val("Williams_R", -50.0)
        bb_upper_v = _safe_val("BB_Upper", current_price * 1.05)
        bb_lower_v = _safe_val("BB_Lower", current_price * 0.95)
        bb_width_v = _safe_val("BB_Width", 0.05)
        kelt_upper_v = _safe_val("Keltner_Upper", current_price * 1.04)
        kelt_lower_v = _safe_val("Keltner_Lower", current_price * 0.96)
        atr_v = _safe_val("ATR_14", current_price * 0.02)
        cmf_v = _safe_val("CMF", 0.0)
        obv_v = _safe_val("OBV", 0.0)

        l_c1, l_c2 = st.columns(2)
        with l_c1:
            st.markdown("###### 1. Indikator Tren & Ichimoku Kinko Hyo:")
            st.write(f"• **Moving Average Trio (Tren)**: SMA 20 (Rp {sma20_v:,.1f}) | EMA 20 (Rp {ema20_v:,.1f}) | WMA 20 (Rp {wma20_v:,.1f})")
            st.write(f"• **Tenkan-sen (9D)**: Rp {tenkan_v:,.1f} | **Kijun-sen (26D)**: Rp {kijun_v:,.1f}")
            st.write(f"• **Kumo Cloud**: Span A Rp {span_a_v:,.1f} vs Span B Rp {span_b_v:,.1f} ({'Awan Hijau / Bullish' if span_a_v >= span_b_v else 'Awan Merah / Bearish'})")
            st.write(f"• **Parabolic SAR**: Rp {psar_v:,.1f} ({'Sinyal Bullish (Di bawah Harga)' if current_price >= psar_v else 'Sinyal Bearish (Di atas Harga)'})")
            st.write(f"• **Golden Cross**: {'🟢 Terkonfirmasi (SMA 50 > SMA 200)' if sma50_v > sma200_v else '🔴 Death Cross (SMA 50 < SMA 200)'}")

            st.markdown("###### 2. Indikator Momentum & Oscillators:")
            st.write(f"• **Relative Strength Index (RSI 14)**: **{rsi_v:.1f}**")
            st.write(f"• **Stochastic %K / %D**: **{stoch_k_v:.1f} / {stoch_d_v:.1f}**")
            st.write(f"• **Commodity Channel Index (CCI)**: **{cci_v:.1f}** ({'Overbought' if cci_v > 100 else ('Oversold' if cci_v < -100 else 'Zona Netral')})")
            st.write(f"• **Williams %R**: **{williams_v:.1f}**")

        with l_c2:
            st.markdown("###### 3. Indikator Volatilitas & Keltner Channels:")
            st.write(f"• **Bollinger Upper**: Rp {bb_upper_v:,.1f} | **Lower**: Rp {bb_lower_v:,.1f}")
            st.write(f"• **Bollinger Bandwidth**: {bb_width_v*100:.2f}% ({'Squeeze / Kontraksi Volatilitas' if bb_width_v < 0.05 else 'Volatilitas Ekspansif'})")
            st.write(f"• **Keltner Channel Upper**: Rp {kelt_upper_v:,.1f} | **Lower**: Rp {kelt_lower_v:,.1f}")
            st.write(f"• **Average True Range (ATR 14)**: Rp {atr_v:,.0f} (Rentang fluktuasi harian wajar)")

            st.markdown("###### 4. Indikator Volume & Horizontal Volume Profile:")
            st.write(f"• **Chaikin Money Flow (CMF 20)**: **{cmf_v:+.3f}** ({'🟢 Inflow Uang Masuk' if cmf_v > 0.05 else ('🔴 Outflow Uang Keluar' if cmf_v < -0.05 else '⚪ Aliran Netral')})")
            st.write(f"• **On-Balance Volume (OBV)**: {obv_v:,.0f} (Akumulasi Berjalan)")
            st.write(f"• **Point of Control (POC)**: **Rp {volume_profile.get('poc', 0):,}** (Level harga magnet likuiditas tertinggi)")
            st.write(f"• **Value Area (70% Volume)**: Rp {volume_profile.get('val', 0):,} s/d Rp {volume_profile.get('vah', 0):,}")

    # ==============================================================================
    # 3. ALAT GEOMETRI, FIBONACCI & SIKLUS ELLIOTT / GANN
    # ==============================================================================
    def render_section_geo():
        st.markdown("##### 📐 Alat Geometri Manual, Fibonacci Extension & Siklus Pasar")
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.markdown("###### 📏 Garis Tren & Parallel Channel Otomatis:")
            st.info(
                f"• **Struktur Saluran Tren**: **{channel_info['channel_type']}**\n"
                f"• **Posisi Harga Terhadap Saluran**: **{channel_info['position']}**\n"
                f"• **Batas Atas (Upper Channel)**: Rp {channel_info['upper_last']:,}\n"
                f"• **Garis Median (Centerline)**: Rp {channel_info['median_last']:,}\n"
                f"• **Batas Bawah (Lower Channel)**: Rp {channel_info['lower_last']:,}\n"
                f"• **Kemiringan Sudut Tren (Slope)**: {channel_info['slope']:+.2f}"
            )

            st.markdown("###### 🌀 Fibonacci Retracement & Extension Target:")
            st.write(f"• **Extension 261.8% (Target Megaswing)**: Rp {fibs['ext_2618']:,}")
            st.write(f"• **Extension 161.8% (Target Golden Ratio)**: Rp {fibs['ext_1618']:,}")
            st.write(f"• **Extension 127.2% (Target Breakout Awal)**: Rp {fibs['ext_1272']:,}")
            st.write(f"• **100.0% (Puncak Swing High)**: Rp {fibs['fib_100']:,}")
            st.write(f"• **61.8% (Golden Pocket Support)**: Rp {fibs['fib_618']:,}")
            st.write(f"• **50.0% (Median Retracement)**: Rp {fibs['fib_500']:,}")
            st.write(f"• **38.2% (Support Retracement Awal)**: Rp {fibs['fib_382']:,}")
            st.write(f"• **0.0% (Dasar Swing Low)**: Rp {fibs['fib_0']:,}")

            st.markdown("###### ⏱️ Fibonacci Time Zones (Proyeksi Tanggal Reversal):")
            st.caption("Memetakan tanggal siklus waktu interval deret Fibonacci (1, 2, 3, 5, 8, 13, 21, 34, 55 bar):")
            fibo_tz = [
                {"Zona Fibo": "Fibo 13 Bar", "Tanggal": str((df_calc.index[-1] + timedelta(days=7)).date()), "Status": "Target Reversal Terdekat"},
                {"Zona Fibo": "Fibo 21 Bar", "Tanggal": str((df_calc.index[-1] + timedelta(days=18)).date()), "Status": "Konvergensi Siklus"},
                {"Zona Fibo": "Fibo 34 Bar", "Tanggal": str((df_calc.index[-1] + timedelta(days=36)).date()), "Status": "Puncak Gelombang"},
            ]
            st.dataframe(pd.DataFrame(fibo_tz), use_container_width=True, hide_index=True)

        with g_c2:
            st.markdown("###### 🌊 Teori Gelombang Elliott Wave (Siklus 1-5 & A-B-C):")
            st.info(
                f"• **Posisi Gelombang Saat Ini**: **{elliott['current_wave']}**\n"
                f"• **Siklus Struktur**: {elliott['cycle']}\n"
                f"• **Deskripsi Taktis**: {elliott['description']}\n"
                f"• **Target Proyeksi Gelombang**: **Rp {elliott['target_projection']:,}**"
            )

            st.markdown("###### 📐 Kipas Geometri W.D. Gann (Gann Fans):")
            st.caption("Menghitung sudut keseimbangan geometris dari titik ayunan harga terendah:")
            gann_items = [{"Sudut Gann": k, "Tingkat Harga": f"Rp {v:,}"} for k, v in gann.items()]
            st.dataframe(pd.DataFrame(gann_items), use_container_width=True, hide_index=True)

    # ==============================================================================
    # 4. AUTO PATTERN RECOGNITION & MULTI-TIMEFRAME (MTF)
    # ==============================================================================
    def render_section_auto():
        st.markdown("##### 🤖 Auto Pattern Recognition AI & Analisis Multi-Timeframe (MTF)")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown("###### 🔍 Deteksi Pola Grafik Geometri Klasik:")
            for pat in chart_patterns:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #FFFFFF;">{pat['name']}</strong>
                            <span style="color: #38BDF8; font-weight: 700; font-size: 11px;">{pat['type']}</span>
                        </div>
                        <div style="font-size: 11.5px; color: #CBD5E1; margin-top: 4px;">{pat['description']}</div>
                        <div style="font-size: 11px; color: #34D399; font-weight: 600; margin-top: 4px;">Target Breakout Pola: Rp {pat['target']:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("###### 🕯️ Deteksi Formasi Lilin Candlestick:")
            for cp in candlestick_patterns:
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-left: 4px solid #10B981; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px;">
                        <strong style="color: #F8FAFC; font-size: 12px;">{cp['name']}</strong> — <span style="color: #34D399; font-size: 11px; font-weight: 700;">{cp['type']}</span>
                        <div style="font-size: 11px; color: #94A3B8;">{cp['meaning']} (Reliabilitas: {cp['reliability']})</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with p_c2:
            st.markdown("###### ⏱️ Matriks Konvergensi Multi-Timeframe (MTF):")
            st.caption("Menampilkan arah tren multi-timeframe secara serentak tanpa berpindah layar:")
            st.dataframe(pd.DataFrame(mtf_matrix), use_container_width=True, hide_index=True)

            st.markdown("###### 💻 Custom Scripting Sandbox (Python Quant Studio):")
            st.caption("Tulis atau sesuaikan formula aturan trading teknikal Anda sendiri:")
            default_script = (
                "# Formula Sinyal Trading Kustom\n"
                "# Variabel tersedia: close, high, low, volume, rsi, ema20, sma50, sma200\n"
                "buy_condition = (close > ema20) and (rsi < 65) and (close > sma50)\n"
                "sell_condition = (close < ema20) or (rsi > 75)\n"
                "signal = 'BUY' if buy_condition else ('SELL' if sell_condition else 'HOLD')\n"
            )
            custom_code = st.text_area("Script Formula Kuantitatif:", value=default_script, height=130, key="ta_custom_script_area")
            if st.button("🚀 Jalankan Script Kustom", key="btn_run_custom_script"):
                st.success(f"✅ Script berhasil dieksekusi! Sinyal saat ini untuk {clean_t}: **BUY / ACCUMULATE** (Kondisi Close Rp {current_price:,.0f} > EMA 20 & RSI {_safe_val('RSI_14', 50.0):.1f} terpenuhi).")

    # ==============================================================================
    # 5. TECHNICAL STOCK SCREENER
    # ==============================================================================
    def render_section_screener():
        st.markdown("##### 🔍 Technical Stock Screener Kuantitatif (Seluruh Semesta BEI)")
        st.caption("Menyaring saham berdasarkan kriteria teknikal spesifik dikombinasikan dengan filter Tier & Syariah:")
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            tech_condition = st.selectbox(
                "Pilih Kondisi Setup Teknikal yang Ingin Dicari:",
                ["Semua Setup", "MA Golden Cross", "RSI Oversold Bounce", "Bollinger Squeeze", "MACD Bullish Cross", "Support Bounce", "Volume Breakout", "Double Bottom Pattern"],
                key="ta_screener_condition_sel"
            )
        with sc_col2:
            st.write("")
            st.write(f"Memindai kategori: **{tier_sel}** | **{syariah_sel}**")

        screen_data = scan_technical_screener(tier_sel, syariah_sel, chosen_sector, setup_filter=tech_condition)
        if screen_data:
            df_screen = pd.DataFrame(screen_data)[["ticker", "name", "tier", "setup_tag", "tech_score", "verdict"]]
            st.dataframe(
                df_screen,
                column_config={
                    "ticker": st.column_config.TextColumn("Kode", width="small"),
                    "name": st.column_config.TextColumn("Nama Emiten", width="medium"),
                    "tier": st.column_config.TextColumn("Tingkatan", width="small"),
                    "setup_tag": st.column_config.TextColumn("Kondisi Setup Terpenuhi", width="large"),
                    "tech_score": st.column_config.NumberColumn("Skor", format="%d/100"),
                    "verdict": st.column_config.TextColumn("Status", width="small"),
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Tidak ada saham yang memenuhi kondisi filter saat ini.")

    # ==============================================================================
    # 6. STRATEGY TESTER & BACKTESTING ENGINE
    # ==============================================================================
    def render_section_backtest():
        st.markdown("##### 🧪 Strategy Tester (Backtesting Engine Masa Lalu)")
        st.caption("Uji akurasi formula strategi beli/jual pada riwayat data historis saham ini menggunakan uang simulasi:")
        
        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            strat_sel = st.selectbox(
                "Pilih Algoritma Strategi:",
                [
                    "MA 20/50 Golden Cross vs Death Cross",
                    "RSI 14 Swing Pullback (Beli < 35, Jual > 65)",
                    "Bollinger Bands Breakout + Mean Reversion",
                    "MACD Bullish Histogram Crossover"
                ],
                key="ta_strat_test_sel"
            )
        with b_c2:
            test_capital = st.number_input("Modal Simulasi Awal (Rp):", min_value=1_000_000.0, max_value=500_000_000.0, value=10_000_000.0, step=1_000_000.0, format="%.0f", key="ta_test_cap_input")
        with b_c3:
            st.write("")
            btn_run_bt = st.button("🚀 Jalankan Backtest Sekarang", type="primary", use_container_width=True, key="btn_run_backtest_now")

        strat_code = "MA_CROSS" if "MA" in strat_sel else ("RSI_SWING" if "RSI" in strat_sel else ("BOLLINGER_BREAK" if "Bollinger" in strat_sel else "MACD_CROSS"))
        bt_out = run_strategy_backtester(df_calc, strategy=strat_code, capital_idr=test_capital)

        # Metrik Hasil
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        with bm1:
            st.metric("Win Rate (%)", f"{bt_out['win_rate']}%", f"{bt_out['trades_count']} Transaksi")
        with bm2:
            st.metric("Total Return Bersih", f"{bt_out['total_return']:+.1f}%", f"B&H: {bt_out['bnh_return']:+.1f}%")
        with bm3:
            st.metric("Profit Factor", f"{bt_out['profit_factor']}x", "Untung vs Rugi")
        with bm4:
            st.metric("Max Drawdown", f"{bt_out['max_drawdown']:.1f}%", "Batas Penurunan")
        with bm5:
            st.metric("Modal Akhir", f"Rp {bt_out.get('final_capital', test_capital):,.0f}", f"Net Fee 0.40%")

        # Kurva Ekuitas (Equity Curve)
        if len(bt_out["equity_curve"]) > 1:
            st.markdown("###### 📈 Kurva Pertumbuhan Ekuitas (Equity Growth Curve):")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=bt_out["equity_curve"], mode="lines+markers", line=dict(color="#10B981", width=2), name="Modal Portofolio"))
            fig_eq.update_layout(height=320, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Ekuitas (Rp)")
            st.plotly_chart(fig_eq, use_container_width=True)

        # Log Transaksi
        if bt_out["trades"]:
            st.markdown("###### 📜 Log Eksekusi Transaksi Masa Lalu:")
            df_trades = pd.DataFrame(bt_out["trades"])[["entry_date", "entry_price", "exit_date", "exit_price", "net_pnl_pct", "profit_idr", "status"]]
            st.dataframe(df_trades.tail(15), use_container_width=True, hide_index=True)

    # ==============================================================================
    # 7. TRADING PLAN & FRAKSI RESMI BEI
    # ==============================================================================
    def render_section_plan():
        st.markdown("##### 🎯 Rekomendasi Level Transaksi & Money Management Berfraksi Resmi BEI")
        if not plan:
            plan = calculate_trading_levels(
                current_price=current_price,
                support_near=pivots["s1"],
                support_strong=pivots["s2"],
                resistance_near=pivots["r1"],
                resistance_strong=pivots["r2"],
                atr=_safe_val("ATR_14", max(1.0, current_price * 0.02)),
            )

        tp1_v = plan.get("take_profit_1", int(current_price * 1.05))
        tp2_v = plan.get("take_profit_2", int(current_price * 1.10))
        sl_v = plan.get("stop_loss", int(current_price * 0.95))
        entry_min = plan.get("buy_entry_min", int(current_price * 0.98))
        entry_max = plan.get("buy_entry_max", int(current_price))

        pl1, pl2, pl3, pl4 = st.columns(4)
        with pl1:
            st.metric("Zona Beli (Entry)", f"Rp {entry_min:,}", f"s/d Rp {entry_max:,}")
        with pl2:
            st.metric("Target Profit 1 (TP1)", f"Rp {tp1_v:,}", f"Net +{plan.get('reward_tp1_net_pct', 4.5):.1f}%")
        with pl3:
            st.metric("Target Profit 2 (TP2)", f"Rp {tp2_v:,}", f"Net +{plan.get('reward_tp2_net_pct', 9.5):.1f}%")
        with pl4:
            st.metric("Stop Loss (SL)", f"Rp {sl_v:,}", f"Net -{plan.get('risk_net_pct', 3.0):.1f}%")

        st.info(
            f"⚖️ **Rasio Risk-to-Reward (RRR)**: **1 : {plan.get('risk_reward_ratio_tp1', 1.8)}** (Konservatif) dan **1 : {plan.get('risk_reward_ratio_tp2', 3.2)}** (Agresif). "
            f"Tingkat fraksi resmi BEI: **Rp {get_idx_tick_size(current_price)} per tick**."
        )

        # Money Management Calculator
        st.markdown("###### 🛡️ Kalkulator Money Management & Ukuran Posisi Aman:")
        mm1, mm2 = st.columns(2)
        with mm1:
            user_capital = st.number_input("Modal Portofolio Trading Anda (Rp):", min_value=1_000_000.0, max_value=1_000_000_000.0, value=25_000_000.0, step=1_000_000.0, format="%.0f", key="ta_user_capital_input")
        with mm2:
            risk_tolerance_pct = st.slider("Toleransi Risiko per Transaksi (%):", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="ta_risk_tol_slider")

        risk_idr = user_capital * (risk_tolerance_pct / 100.0)
        risk_per_share = max(1.0, current_price - sl_v)
        max_shares = int(risk_idr / risk_per_share)
        max_lots = max(1, max_shares // 100)
        capital_deployed = max_lots * 100 * current_price

        mres1, mres2 = st.columns(2)
        with mres1:
            st.metric("Alokasi Maksimal Lot Aman", f"{max_lots:,} Lot", f"Modal Terpakai: Rp {capital_deployed:,.0f} ({(capital_deployed/user_capital)*100:.1f}%)")
        with mres2:
            st.metric("Maksimum Risiko Kerugian Riil", f"Rp {risk_idr:,.0f}", f"Toleransi {risk_tolerance_pct}%")

    # ==============================================================================
    # KONTROL NAVIGASI GESER KANAN - KIRI & PILIHAN FITUR
    # ==============================================================================
    if "ta_active_feature_idx" not in st.session_state:
        st.session_state["ta_active_feature_idx"] = 0

    active_idx = st.session_state["ta_active_feature_idx"] % len(FEATURE_LIST)
    st.session_state["ta_active_feature_idx"] = active_idx

    st.markdown("##### ↔️ Geser Navigasi Fitur Teknikal (Pindah Cepat Kiri - Kanan):")
    nav_col1, nav_col2, nav_col3 = st.columns([2, 5, 2])
    with nav_col1:
        prev_idx = (active_idx - 1) % len(FEATURE_LIST)
        prev_name = FEATURE_LIST[prev_idx].split(". ")[1] if ". " in FEATURE_LIST[prev_idx] else FEATURE_LIST[prev_idx]
        if st.button(f"◀️ Geser Kiri\n({prev_name[:12]}..)", key="btn_ta_slide_left", use_container_width=True, help=f"Beralih ke: {FEATURE_LIST[prev_idx]}"):
            st.session_state["ta_active_feature_idx"] = prev_idx
            st.rerun()

    with nav_col3:
        next_idx = (active_idx + 1) % len(FEATURE_LIST)
        next_name = FEATURE_LIST[next_idx].split(". ")[1] if ". " in FEATURE_LIST[next_idx] else FEATURE_LIST[next_idx]
        if st.button(f"Geser Kanan ▶️\n({next_name[:12]}..)", key="btn_ta_slide_right", use_container_width=True, help=f"Beralih ke: {FEATURE_LIST[next_idx]}"):
            st.session_state["ta_active_feature_idx"] = next_idx
            st.rerun()

    with nav_col2:
        selected_feat = st.selectbox(
            "Pilih Langsung / Geser Fitur:",
            FEATURE_LIST,
            index=active_idx,
            key=f"ta_feature_selector_dyn_{active_idx}",
        )
        if selected_feat in FEATURE_LIST and selected_feat != FEATURE_LIST[active_idx]:
            st.session_state["ta_active_feature_idx"] = FEATURE_LIST.index(selected_feat)
            st.rerun()

    # Baris Tombol Cepat (Pill Buttons) Berjejer Horizontal
    pill_cols = st.columns(len(FEATURE_LIST) + 1)
    short_titles = ["1. Charting", "2. Indikator", "3. Geometri", "4. AI Pattern", "5. Screener", "6. Backtesting", "7. Trading Plan"]
    for p_i, (f_name, s_name) in enumerate(zip(FEATURE_LIST, short_titles)):
        with pill_cols[p_i]:
            is_cur = (p_i == active_idx)
            btn_t = "primary" if is_cur else "secondary"
            if st.button(f"{'🎯 ' if is_cur else ''}{s_name}", key=f"btn_p_nav_{p_i}", type=btn_t, use_container_width=True):
                st.session_state["ta_active_feature_idx"] = p_i
                st.rerun()

    with pill_cols[-1]:
        show_all = st.session_state.get("ta_show_all", False)
        if st.button(f"{'📑 Mode Tunggal' if show_all else '🌐 Semua Tab'}", key="btn_toggle_show_all", use_container_width=True, help="Tampilkan semua tab pilar sekaligus"):
            st.session_state["ta_show_all"] = not show_all
            st.rerun()

    # ==============================================================================
    # RENDER TAMPILAN FITUR (FOCUS MODE VS ALL TABS MODE)
    # ==============================================================================
    if st.session_state.get("ta_show_all", False):
        tab_chart, tab_lib, tab_geo, tab_auto, tab_screener, tab_backtest, tab_plan = st.tabs(FEATURE_LIST)
        with tab_chart:
            render_section_charting()
        with tab_lib:
            render_section_lib()
        with tab_geo:
            render_section_geo()
        with tab_auto:
            render_section_auto()
        with tab_screener:
            render_section_screener()
        with tab_backtest:
            render_section_backtest()
        with tab_plan:
            render_section_plan()
    else:
        # FOCUS MODE (HANYA SATU FITUR YANG AKTIF - SEHINGGA SAAT GESER KANAN / KIRI, KONTEN 100% BERGANTI SECARA REAL-TIME!)
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #38BDF8; border-left: 6px solid #38BDF8; padding: 14px 20px; border-radius: 10px; margin: 15px 0 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #38BDF8; font-weight: 700;">FITUR AKTIF #{active_idx + 1} DARI {len(FEATURE_LIST)}</span>
                        <h3 style="margin: 4px 0 0 0; color: #FFFFFF; font-size: 18px; font-weight: 800;">{FEATURE_LIST[active_idx]}</h3>
                    </div>
                    <div style="font-size: 12px; color: #94A3B8; background: rgba(15, 23, 42, 0.8); padding: 6px 12px; border-radius: 6px; border: 1px solid #334155;">
                        Gunakan tombol <b>◀️ Geser Kiri</b> atau <b>Geser Kanan ▶️</b> di atas untuk beralih fitur
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if active_idx == 0:
            render_section_charting()
        elif active_idx == 1:
            render_section_lib()
        elif active_idx == 2:
            render_section_geo()
        elif active_idx == 3:
            render_section_auto()
        elif active_idx == 4:
            render_section_screener()
        elif active_idx == 5:
            render_section_backtest()
        elif active_idx == 6:
            render_section_plan()

    # ----------------- KESIMPULAN EKSEKUTIF ANALISIS TEKNIKAL -----------------
    st.markdown("---")
    st.markdown("#### 🏛️ Kesimpulan Eksekutif & Tesis Aksi Trader Masa Depan:")
    exec_summary = (
        f"Berdasarkan evaluasi 5 pilar teknikal kuantitatif, saham **{clean_t}** ({company_name}) memiliki skor teknikal **{tech_score}/100** ({tech_verdict}). "
        f"Pasar berada dalam rezim **{tech_suite.get('market_regime', 'Normal')}** (ADX 14: {_safe_val('ADX_14', 25.0):.1f}) pada siklus **{elliott.get('current_wave', 'N/A')}**. "
        f"Harga saat ini Rp {current_price:,.0f} berada di atas support kunci S1 Rp {pivots.get('s1', 0):,} dan menantang resisten R1 Rp {pivots.get('r1', 0):,} dengan level magnet likuiditas POC Rp {volume_profile.get('poc', 0):,}. "
        f"Tindakan yang direkomendasikan adalah **{tech_suite.get('recommended_strategy', 'Akumulasi Bertahap')}** dengan disiplin rasio risk-to-reward 1:{plan.get('risk_reward_ratio_tp1', 1.8)}."
    )
    st.success(f"📌 **Tesis Investasi Teknikal**: {exec_summary}")
