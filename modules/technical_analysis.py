"""
Modul Analisis Teknikal Saham Indonesia (IDX) - Terpadu Tingkat Institusional.
Menghitung indikator teknikal komprehensif:
  1. Moving Averages & MA Ribbon: EMA 10, EMA 20, EMA 200, SMA 50, SMA 100, SMA 200, MA 5, MA 10
  2. Deteksi Golden Cross & Death Cross
  3. Average Directional Index (ADX 14) & Filter Pasar Sideways vs Trending (Anti-False Signal)
  4. Relative Strength Index (RSI 14), MACD (12, 26, 9), Bollinger Bands (20, 2), ATR 14
  5. Support & Resisten Dinamis (Pivot Point Classic, Swing High/Low, & Fibonacci Retracement)
  6. Evaluasi Skor Teknikal Terkalibrasi (0 - 100)
  7. Interactive Point-in-Time Inspector untuk diagnosis multi-faktor per bar lilin
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def compute_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Menghitung Average Directional Index (ADX), +DI, dan -DI secara matematis."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window=period).mean().fillna(15.0)

    return adx, plus_di.fillna(20.0), minus_di.fillna(20.0)


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan seluruh kolom indikator teknikal komprehensif ke DataFrame OHLCV sekali jalan.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1. Moving Averages & MA Ribbon
    df["MA_5"] = close.rolling(window=5, min_periods=1).mean()
    df["MA_10"] = close.rolling(window=10, min_periods=1).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=10).mean()
    df["SMA_100"] = close.rolling(window=100, min_periods=10).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=20).mean()
    df["EMA_200"] = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else close.ewm(span=len(df), adjust=False).mean()

    # 2. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # 3. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 4. Bollinger Bands (20, 2)
    bb_mid = close.rolling(window=20, min_periods=5).mean()
    bb_std = close.rolling(window=20, min_periods=5).std()
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_mid + (bb_std * 2)
    df["BB_Lower"] = bb_mid - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (bb_mid.replace(0, np.nan))

    # 5. ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14, min_periods=5).mean().bfill()

    # 6. ADX (14)
    adx, plus_di, minus_di = compute_adx(df, period=14)
    df["ADX_14"] = adx
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    # 7. Volume Analysis
    df["Volume_SMA20"] = volume.rolling(window=20, min_periods=1).mean()
    df["Volume_Ratio"] = volume / (df["Volume_SMA20"].replace(0, np.nan))

    return df


def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float, float, float]:
    """
    Mendeteksi level Support dan Resistance terdekat berdasarkan swing low dan swing high historis.
    Returns: (Support_Kuat, Support_Terdekat, Resistance_Terdekat, Resistance_Kuat)
    """
    current_price = float(df["Close"].iloc[-1])
    recent_lows = df["Low"].tail(window * 3)
    recent_highs = df["High"].tail(window * 3)

    supports = recent_lows[recent_lows < current_price]
    resistances = recent_highs[recent_highs > current_price]

    atr = float(df["ATR_14"].iloc[-1]) if "ATR_14" in df.columns else (current_price * 0.02)

    if not supports.empty:
        s_near = float(supports.max())
        s_strong = float(supports.quantile(0.25))
    else:
        s_near = current_price - (1.5 * atr)
        s_strong = current_price - (3.0 * atr)

    if not resistances.empty:
        r_near = float(resistances.min())
        r_strong = float(resistances.quantile(0.75))
    else:
        r_near = current_price + (2.0 * atr)
        r_strong = current_price + (4.0 * atr)

    if r_near <= current_price:
        r_near = current_price + (1.5 * atr)
    if s_near >= current_price:
        s_near = current_price - (1.5 * atr)

    return round(s_strong), round(s_near), round(r_near), round(r_strong)


def compute_technical_suite(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ekstraksi suite analisis teknikal kuantitatif lengkap dengan MA Ribbon,
    filter pasar sideways vs trending, pivot points, dan level Fibonacci.
    """
    if df is None or len(df) < 15:
        return {}

    if "EMA_20" not in df.columns or "ADX_14" not in df.columns:
        df_calc = compute_technical_indicators(df)
    else:
        df_calc = df

    close = df_calc["Close"]
    high = df_calc["High"]
    low = df_calc["Low"]

    last_close = float(close.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])

    sma50 = df_calc["SMA_50"]
    sma200 = df_calc["SMA_200"]
    golden_cross = bool(sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]) if len(sma50) >= 2 else False
    death_cross = bool(sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]) if len(sma50) >= 2 else False
    is_golden_regime = bool(sma50.iloc[-1] > sma200.iloc[-1]) if len(sma50) > 0 else False

    last_adx = float(df_calc["ADX_14"].iloc[-1])
    last_bb_width = float(df_calc["BB_Width"].iloc[-1]) if "BB_Width" in df_calc.columns else 0.05
    is_sideways = bool(last_adx < 22.0 or last_bb_width < 0.05)

    if is_sideways:
        market_regime = "Konsolidasi / Sideways (Range-Bound)"
        ma_effectiveness = "KURANG EFEKTIF (Banyak False Signal) -> Beralih ke Support/Resistance & Volume"
        recommended_strategy = "Range Trading (Beli di Support S1/Fibo 61.8%, Jual di Resisten R1)"
    else:
        market_regime = "Trending Kuat (Directional Market)"
        ma_effectiveness = "SANGAT EFEKTIF (Sinyal MA Ribbon Valid & Terarah)"
        recommended_strategy = "Trend Following / Breakout (Ikuti susunan MA Ribbon)"

    # Pivot Point Classic
    pivot = (last_high + last_low + last_close) / 3.0
    r1 = (2.0 * pivot) - last_low
    s1 = (2.0 * pivot) - last_high
    r2 = pivot + (last_high - last_low)
    s2 = pivot - (last_high - last_low)

    # Fibonacci Retracement dari rentang 60 hari
    h_60 = float(high.tail(60).max())
    l_60 = float(low.tail(60).min())
    diff_60 = h_60 - l_60
    fibo_levels = {
        "0.0% (High)": h_60,
        "23.6%": h_60 - (0.236 * diff_60),
        "38.2%": h_60 - (0.382 * diff_60),
        "50.0% (Mid)": h_60 - (0.500 * diff_60),
        "61.8% (Golden)": h_60 - (0.618 * diff_60),
        "100.0% (Low)": l_60,
    }

    # Ribbon Stack Status
    ma5 = df_calc["MA_5"]
    ma10 = df_calc["MA_10"]
    ma20 = df_calc["EMA_20"]
    is_ribbon_bullish = bool(ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > sma50.iloc[-1])
    is_ribbon_bearish = bool(ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1] < sma50.iloc[-1])

    return {
        "last_price": last_close,
        "ema10": float(df_calc["EMA_10"].iloc[-1]),
        "ema20": float(df_calc["EMA_20"].iloc[-1]),
        "ema200": float(df_calc["EMA_200"].iloc[-1]),
        "sma50": float(sma50.iloc[-1]),
        "sma100": float(df_calc["SMA_100"].iloc[-1]),
        "sma200": float(sma200.iloc[-1]),
        "ma5": float(ma5.iloc[-1]),
        "ma10": float(ma10.iloc[-1]),
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "is_golden_regime": is_golden_regime,
        "adx": last_adx,
        "is_sideways": is_sideways,
        "market_regime": market_regime,
        "ma_effectiveness": ma_effectiveness,
        "recommended_strategy": recommended_strategy,
        "is_ribbon_bullish": is_ribbon_bullish,
        "is_ribbon_bearish": is_ribbon_bearish,
        "rsi": float(df_calc["RSI_14"].iloc[-1]),
        "macd_hist": float(df_calc["MACD_Hist"].iloc[-1]),
        "atr": float(df_calc["ATR_14"].iloc[-1]),
        "pivot": pivot,
        "r1": r1,
        "s1": s1,
        "r2": r2,
        "s2": s2,
        "fibo_levels": fibo_levels,
        "df_calc": df_calc,
    }


def inspect_point_in_time(
    df: pd.DataFrame,
    tech_data: Dict[str, Any],
    bar_index: int = -1,
) -> Dict[str, Any]:
    """
    Menghasilkan diagnosis diagnostik multi-faktor mendalam pada titik waktu tertentu (bar grafik).
    """
    if df is None or df.empty:
        return {}

    idx = bar_index if 0 <= bar_index < len(df) else len(df) - 1
    date_val = df.index[idx]
    c_price = float(df["Close"].iloc[idx])
    o_price = float(df["Open"].iloc[idx])
    h_price = float(df["High"].iloc[idx])
    l_price = float(df["Low"].iloc[idx])
    v_vol = float(df["Volume"].iloc[idx])

    df_calc = tech_data.get("df_calc", df)
    m5 = float(df_calc["MA_5"].iloc[idx]) if "MA_5" in df_calc.columns else c_price
    m20 = float(df_calc["EMA_20"].iloc[idx]) if "EMA_20" in df_calc.columns else c_price
    m50 = float(df_calc["SMA_50"].iloc[idx]) if "SMA_50" in df_calc.columns else c_price
    rsi_val = float(df_calc["RSI_14"].iloc[idx]) if "RSI_14" in df_calc.columns else 50.0
    adx_val = float(df_calc["ADX_14"].iloc[idx]) if "ADX_14" in df_calc.columns else 20.0

    candle_type = "Bullish" if c_price >= o_price else "Bearish"
    pos_vs_ma20 = "Di Atas EMA 20 (Dukungan Bullish)" if c_price >= m20 else "Di Bawah EMA 20 (Tekanan Jual)"
    regime = "Trending Kuat" if adx_val >= 22 else "Konsolidasi / Sideways"

    if rsi_val > 70:
        rsi_status = "Jenuh Beli (Overbought > 70) - Rawan Koreksi"
    elif rsi_val < 30:
        rsi_status = "Jenuh Jual (Oversold < 30) - Potensi Pantulan"
    else:
        rsi_status = f"Netral Seimbang ({rsi_val:.1f})"

    narrative = (
        f"Pada bar tanggal {str(date_val)[:10]}, harga saham ditutup di Rp {c_price:,.0f} ({candle_type}). "
        f"Posisi harga berada {pos_vs_ma20}. Indikator ADX berada di level {adx_val:.1f} ({regime}), "
        f"sementara RSI menunjukkan {rsi_status}. MA 5 berada di Rp {m5:,.0f} dan SMA 50 di Rp {m50:,.0f}."
    )

    return {
        "date": str(date_val)[:10],
        "close": c_price,
        "open": o_price,
        "high": h_price,
        "low": l_price,
        "volume": v_vol,
        "rsi": rsi_val,
        "adx": adx_val,
        "ma5": m5,
        "ma20": m20,
        "ma50": m50,
        "narrative": narrative,
    }


def evaluate_technical_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Menilai kondisi teknikal secara kuantitatif (Skor 0 - 100) dan mengidentifikasi sinyal spesifik.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    price = float(last["Close"])

    signals = []
    score = 0

    # 1. Evaluasi Trend Moving Average (Bobot: 35 poin)
    sma200 = last["SMA_200"] if pd.notna(last.get("SMA_200")) else None
    sma50 = last["SMA_50"] if pd.notna(last.get("SMA_50")) else None
    ema20 = last["EMA_20"] if pd.notna(last.get("EMA_20")) else None

    if sma200 and price > sma200:
        score += 15
        signals.append("Harga di atas SMA 200 (Tren Jangka Panjang Bullish)")
    elif sma200:
        signals.append("Harga di bawah SMA 200 (Tren Jangka Panjang Bearish)")

    if sma50 and price > sma50:
        score += 10
        signals.append("Harga di atas SMA 50 (Tren Menengah Positif)")

    if ema20 and sma50 and ema20 > sma50:
        score += 10
        signals.append("EMA 20 > SMA 50 (Struktur Golden Alignment)")

    # 2. Evaluasi RSI (Bobot: 20 poin)
    rsi = float(last["RSI_14"]) if "RSI_14" in last else 50.0
    if 45 <= rsi <= 65:
        score += 20
        signals.append(f"RSI {rsi:.1f}: Momentum sehat (Bullish Zone tanpa Overbought)")
    elif 30 <= rsi < 45:
        score += 12
        signals.append(f"RSI {rsi:.1f}: Zona Netral bawah, potensi akumulasi")
    elif rsi < 30:
        score += 15
        signals.append(f"RSI {rsi:.1f}: OVERSOLD (Potensi Rebound Kuat)")
    elif rsi > 70:
        score += 5
        signals.append(f"RSI {rsi:.1f}: OVERBOUGHT (Hati-hati risiko koreksi/profit taking)")

    # 3. Evaluasi MACD (Bobot: 25 poin)
    macd = float(last["MACD"]) if "MACD" in last else 0.0
    signal = float(last["MACD_Signal"]) if "MACD_Signal" in last else 0.0
    hist = float(last["MACD_Hist"]) if "MACD_Hist" in last else 0.0
    prev_hist = float(prev["MACD_Hist"]) if "MACD_Hist" in prev else hist

    if macd > signal:
        score += 15
        signals.append("MACD di atas Signal Line (Momentum Positif)")
        if hist > prev_hist:
            score += 10
            signals.append("Histogram MACD semakin melebar naik (Akselerasi Bullish)")
    else:
        if hist > prev_hist:
            score += 8
            signals.append("MACD Histogram mulai membaik (Pelemahan Bearish)")
        else:
            signals.append("MACD di bawah Signal Line (Tekanan Jual Masih Terjadi)")

    # 4. Evaluasi Volume & Bollinger Bands (Bobot: 20 poin)
    vol_ratio = float(last["Volume_Ratio"]) if pd.notna(last.get("Volume_Ratio")) else 1.0
    if vol_ratio > 1.3:
        score += 10
        signals.append(f"Volume tinggi ({vol_ratio:.2f}x rata-rata 20 hari) - Minat pasar besar")

    if pd.notna(last.get("BB_Lower")) and price <= (last["BB_Lower"] * 1.02):
        score += 10
        signals.append("Harga menguji Bollinger Band Bawah (Area pantulan teknikal)")
    elif pd.notna(last.get("BB_Middle")) and price > last["BB_Middle"]:
        score += 10
        signals.append("Harga bertahan di atas Mid-Bollinger Band")

    score = min(max(score, 0), 100)

    status = "NEUTRAL"
    if score >= 75:
        status = "VERY BULLISH"
    elif score >= 60:
        status = "BULLISH"
    elif score <= 30:
        status = "VERY BEARISH"
    elif score <= 45:
        status = "BEARISH"

    s_strong, s_near, r_near, r_strong = detect_support_resistance(df)

    return {
        "score": score,
        "status": status,
        "signals": signals,
        "rsi": round(rsi, 2),
        "macd": round(macd, 2),
        "macd_signal": round(signal, 2),
        "atr": round(float(last["ATR_14"]), 2) if "ATR_14" in df.columns else 0.0,
        "support_near": s_near,
        "support_strong": s_strong,
        "resistance_near": r_near,
        "resistance_strong": r_strong,
        "volume_ratio": round(vol_ratio, 2),
    }
