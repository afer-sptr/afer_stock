"""
Modul Analisis High and Low Saham Indonesia (IDX).
Mengevaluasi posisi harga terhadap 52-Week High/Low, Intraday Day High/Low,
Breakout 20-Day Donchian Channel, dan Fibonacci Retracement Levels.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def evaluate_high_low_aspects(df: pd.DataFrame, fast_info: Any = None) -> Dict[str, Any]:
    """
    Menganalisis seluruh aspek High dan Low saham:
    1. 52-Week High & Low Position
    2. Day High & Low Intraday Closing Power
    3. 20-Day High/Low Breakout (Turtle Trading Strategy)
    4. Level Retracement Fibonacci (Swing High/Low)
    5. Skor Keputusan High/Low (0 - 100)
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    current_price = float(close.iloc[-1])

    # 1. Metrik 52-Week High & Low
    yh = getattr(fast_info, "year_high", None) if fast_info else (fast_info.get("year_high") if isinstance(fast_info, dict) else None)
    yl = getattr(fast_info, "year_low", None) if fast_info else (fast_info.get("year_low") if isinstance(fast_info, dict) else None)
    if yh is not None and yl is not None:
        year_high = float(yh)
        year_low = float(yl)
    else:
        # Hitung dari 250 hari bursa terakhir (setara 1 tahun)
        last_250 = df.tail(250)
        year_high = float(last_250["High"].max())
        year_low = float(last_250["Low"].min())

    dist_to_52w_high_pct = ((current_price - year_high) / year_high) * 100
    dist_from_52w_low_pct = ((current_price - year_low) / year_low) * 100
    range_52w = max(year_high - year_low, 1.0)
    pos_52w_pct = round(((current_price - year_low) / range_52w) * 100, 1)

    # 2. Intraday Day High & Day Low
    dh = getattr(fast_info, "day_high", None) if fast_info else (fast_info.get("day_high") if isinstance(fast_info, dict) else None)
    dl = getattr(fast_info, "day_low", None) if fast_info else (fast_info.get("day_low") if isinstance(fast_info, dict) else None)
    if dh is not None and dl is not None:
        day_high = float(dh)
        day_low = float(dl)
    else:
        day_high = float(high.iloc[-1])
        day_low = float(low.iloc[-1])

    day_range = max(day_high - day_low, 1.0)
    # Close Location Value (CLV): 0.0 (tutup di low) s/d 1.0 (tutup di high)
    clv = (current_price - day_low) / day_range

    if clv >= 0.75:
        clv_status = "BULLISH DOMINANT (Penutupan dekat High Harian)"
    elif clv <= 0.25:
        clv_status = "BEARISH PRESSURE (Penutupan dekat Low Harian)"
    else:
        clv_status = "NEUTRAL / KONSOLIDASI (Penutupan di tengah range)"

    # 3. 20-Day High & Low Breakout (Donchian Channel)
    last_20 = df.tail(20)
    high_20d = float(last_20["High"].iloc[:-1].max()) if len(last_20) > 1 else year_high
    low_20d = float(last_20["Low"].iloc[:-1].min()) if len(last_20) > 1 else year_low

    is_breakout_20d = current_price >= high_20d
    is_breakdown_20d = current_price <= low_20d

    if is_breakout_20d:
        breakout_status = "BREAKOUT 20-DAY HIGH (Sinyal Beli Tren Baru)"
    elif is_breakdown_20d:
        breakout_status = "BREAKDOWN 20-DAY LOW (Sinyal Bahaya/Koreksi Lanjutan)"
    else:
        breakout_status = "INSIDE RANGE (Pergerakan Wajar di Dalam Kanal 20 Hari)"

    # 4. Fibonacci Retracement Levels (Berdasarkan Swing High & Low 6-12 Bulan)
    swing_period = min(180, len(df))
    swing_df = df.tail(swing_period)
    swing_high = float(swing_df["High"].max())
    swing_low = float(swing_df["Low"].min())
    diff = swing_high - swing_low

    fib_levels = {
        "fib_0": round(swing_low),                          # 0.0% (Swing Low)
        "fib_236": round(swing_low + (0.236 * diff)),       # 23.6%
        "fib_382": round(swing_low + (0.382 * diff)),       # 38.2%
        "fib_500": round(swing_low + (0.500 * diff)),       # 50.0% (Midpoint)
        "fib_618": round(swing_low + (0.618 * diff)),       # 61.8% (Golden Ratio)
        "fib_786": round(swing_low + (0.786 * diff)),       # 78.6%
        "fib_1000": round(swing_high)                       # 100.0% (Swing High)
    }

    # Cari zona Fibonacci terdekat
    if current_price >= fib_levels["fib_786"]:
        fib_zone = "Di atas Fib 78.6% (Menguji Puncak Swing High)"
    elif current_price >= fib_levels["fib_618"]:
        fib_zone = "Di area Golden Pocket Fib 61.8% - 78.6% (Support Kuat)"
    elif current_price >= fib_levels["fib_500"]:
        fib_zone = "Di area Keseimbangan Fib 50.0% - 61.8%"
    elif current_price >= fib_levels["fib_382"]:
        fib_zone = "Di area Fib 38.2% - 50.0%"
    else:
        fib_zone = "Di area Bawah Fib 0.0% - 23.6% (Dekat Titik Terendah)"

    # 5. Kalkulasi Skor High/Low Keputusan (Skor 0 - 100)
    hl_score = 50 # Baseline netral
    insights = []

    # Evaluasi 52-Week High Proximity
    if dist_to_52w_high_pct >= -5.0:
        hl_score += 15
        insights.append(f"Harga hanya {abs(dist_to_52w_high_pct):.1f}% di bawah 52-Week High (Momentum Kuat Menembus Puncak)")
    elif dist_to_52w_high_pct <= -40.0:
        if pos_52w_pct <= 15.0:
            hl_score += 10 # Potensi value bottoming
            insights.append(f"Harga berada di dekat 52-Week Low ({pos_52w_pct}% dari dasar) - Potensi Deep Value / Bottom Reversal")
        else:
            hl_score -= 10
            insights.append(f"Harga tertinggal jauh ({dist_to_52w_high_pct:.1f}%) dari puncak tahunan (Tren Lemah)")

    # Evaluasi Breakout 20-Day
    if is_breakout_20d:
        hl_score += 20
        insights.append("Berhasil BREAKOUT di atas High 20 Hari Terakhir (Konfirmasi Awal Penguatan Tren)")
    elif is_breakdown_20d:
        hl_score -= 20
        insights.append("BREAKDOWN di bawah Low 20 Hari Terakhir (Tekanan Jual Masih Kuat)")

    # Evaluasi Intraday Closing Power
    if clv >= 0.75:
        hl_score += 15
        insights.append("Kekuatan penutupan harian sangat tinggi (Pembeli menguasai penutupan bursa)")
    elif clv <= 0.25:
        hl_score -= 15
        insights.append("Penutupan harian di dekat titik terendah (Tekanan aksi jual hingga akhir sesi)")

    # Batasi skor 0 - 100
    hl_score = min(max(hl_score, 0), 100)

    return {
        "score": hl_score,
        "year_high": round(year_high),
        "year_low": round(year_low),
        "dist_to_52w_high_pct": round(dist_to_52w_high_pct, 2),
        "dist_from_52w_low_pct": round(dist_from_52w_low_pct, 2),
        "pos_52w_pct": pos_52w_pct,
        "day_high": round(day_high),
        "day_low": round(day_low),
        "clv": round(clv, 2),
        "clv_status": clv_status,
        "high_20d": round(high_20d),
        "low_20d": round(low_20d),
        "breakout_status": breakout_status,
        "fib_levels": fib_levels,
        "fib_zone": fib_zone,
        "insights": insights
    }
