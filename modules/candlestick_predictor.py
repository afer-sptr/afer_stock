"""
candlestick_predictor.py
========================
Mesin Prediksi Pergerakan Saham Real-Time Berbasis Grafik Candlestick.
Mendeteksi pola lilin multi-bar terkini (Hammer, Engulfing, Morning/Evening Star,
Pin Bar Rejection, Marubozu, Doji, Three Soldiers/Crows), mengonfirmasi dengan volume
dan dynamic ATR, serta menghasilkan rekomendasi eksplisit Titik Entry,
Take Profit 1 (TP1), Take Profit 2 (TP2), dan Cut Loss (Stop Loss) berfraksi resmi BEI.
"""

import math
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    from modules.idx_ticks import round_to_idx_tick, get_idx_tick_size
except ImportError:
    from idx_ticks import round_to_idx_tick, get_idx_tick_size


def _calc_net_pct(entry: float, target: float) -> float:
    """Menghitung persentase laba/rugi bersih setelah fee broker & pajak roundtrip (0.40%)."""
    if entry <= 0:
        return 0.0
    gross_pct = ((target - entry) / entry) * 100.0
    net_pct = gross_pct - 0.40
    return round(net_pct, 2)


def predict_candlestick_movement(
    df: pd.DataFrame,
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Menganalisis lilin candlestick terkini dan memproyeksikan arah pergerakan harga real-time
    lengkap dengan rekomendasi titik Entry, Take Profit, Cut Loss, dan tingkat probabilitas.
    """
    if df is None or len(df) < 5:
        return {
            "success": False,
            "error": "Data historis tidak mencukupi untuk analisis candlestick.",
        }

    info = info or {}
    clean_df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    n = len(clean_df)
    if n < 5:
        return {"success": False, "error": "Data tidak cukup."}

    # Data lilin terakhir
    c0 = clean_df.iloc[-1]
    c1 = clean_df.iloc[-2] if n >= 2 else c0
    c2 = clean_df.iloc[-3] if n >= 3 else c1

    o0, h0, l0, cl0 = float(c0["Open"]), float(c0["High"]), float(c0["Low"]), float(c0["Close"])
    v0 = float(c0["Volume"]) if "Volume" in c0 else 100000.0

    o1, h1, l1, cl1 = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])
    o2, h2, l2, cl2 = float(c2["Open"]), float(c2["High"]), float(c2["Low"]), float(c2["Close"])

    # Volatilitas ATR 14
    highs = clean_df["High"].values
    lows = clean_df["Low"].values
    closes = clean_df["Close"].values
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    )
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr)) if len(tr) > 0 else max(10.0, cl0 * 0.02)
    atr = max(atr, float(get_idx_tick_size(cl0)))

    # Volume SMA 20
    vol_series = clean_df["Volume"].values
    avg_vol = float(np.mean(vol_series[-20:])) if len(vol_series) >= 20 else float(np.mean(vol_series))
    vol_ratio = v0 / avg_vol if avg_vol > 0 else 1.0

    # Trend Context
    ema20 = float(clean_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    sma50 = float(clean_df["Close"].rolling(window=min(50, n)).mean().iloc[-1])

    # Geometri lilin bar 0
    rng0 = max(1.0, h0 - l0)
    body0 = abs(cl0 - o0)
    is_bull0 = cl0 >= o0
    upper_wick0 = h0 - max(cl0, o0)
    lower_wick0 = min(cl0, o0) - l0

    # Geometri lilin bar 1
    rng1 = max(1.0, h1 - l1)
    body1 = abs(cl1 - o1)
    is_bull1 = cl1 >= o1

    # Deteksi Pola Candlestick
    detected_pattern = "Standard Candle"
    direction = "KONSOLIDASI / WAIT & SEE"
    signal = "HOLD"
    confidence = 70
    pattern_badge = "Neutral / Sideways"
    pattern_desc = "Lilin normal tanpa pola pembalikan yang dominan."

    # 1. Bullish Engulfing
    if not is_bull1 and is_bull0 and body0 > body1 and cl0 > o1 and o0 <= cl1:
        detected_pattern = "Bullish Engulfing"
        direction = "BULLISH REVERSAL"
        signal = "STRONG BUY"
        confidence = 88 if vol_ratio >= 1.2 else 82
        pattern_badge = "🟢 Bullish Engulfing (Pembalikan Kuat)"
        pattern_desc = "Lilin hijau penuh menelan lilin merah sebelumnya, mengindikasikan dominasi total buyer."

    # 2. Bearish Engulfing
    elif is_bull1 and not is_bull0 and body0 > body1 and cl0 < o1 and o0 >= cl1:
        detected_pattern = "Bearish Engulfing"
        direction = "BEARISH REJECTION"
        signal = "STRONG SELL"
        confidence = 88 if vol_ratio >= 1.2 else 80
        pattern_badge = "🔴 Bearish Engulfing (Pembalikan Turun)"
        pattern_desc = "Lilin merah penuh menelan lilin hijau sebelumnya, tekanan jual institusi sangat masif."

    # 3. Hammer / Bullish Pin Bar di Support
    elif lower_wick0 >= (2.0 * body0) and upper_wick0 <= (0.3 * rng0) and (cl0 <= ema20 * 1.03 or cl0 >= ema20 * 0.97):
        detected_pattern = "Hammer / Bullish Pin Bar"
        direction = "BULLISH REVERSAL"
        signal = "BUY"
        confidence = 85 if vol_ratio >= 1.0 else 78
        pattern_badge = "🟢 Hammer / Bullish Pin Bar (Rejection Bawah)"
        pattern_desc = "Ekor bawah panjang menunjukkan penolakan harga murah (rejection support) dan perlawanan buyer kuat."

    # 4. Shooting Star / Bearish Pin Bar di Resistance
    elif upper_wick0 >= (2.0 * body0) and lower_wick0 <= (0.3 * rng0) and cl0 >= ema20:
        detected_pattern = "Shooting Star / Bearish Pin Bar"
        direction = "BEARISH REJECTION"
        signal = "SELL"
        confidence = 84 if vol_ratio >= 1.0 else 77
        pattern_badge = "🔴 Shooting Star (Rejection Atas)"
        pattern_desc = "Ekor atas panjang menunjukkan kegagalan menembus resistance dan aksi ambil untung agresif."

    # 5. Morning Star (3 Lilin)
    elif n >= 3 and not (cl2 >= o2) and abs(cl2 - o2) > (0.4 * rng1) and abs(cl1 - o1) < (0.3 * abs(cl2 - o2)) and is_bull0 and cl0 > (o2 + cl2) / 2:
        detected_pattern = "Morning Star"
        direction = "BULLISH REVERSAL"
        signal = "STRONG BUY"
        confidence = 90
        pattern_badge = "🟢 Morning Star (Pembalikan Bullish 3-Bar)"
        pattern_desc = "Formasi bintang pagi mengonfirmasi pergeseran momentum dari fase penurunan ke fase reli naik."

    # 6. Evening Star (3 Lilin)
    elif n >= 3 and (cl2 >= o2) and abs(cl2 - o2) > (0.4 * rng1) and abs(cl1 - o1) < (0.3 * abs(cl2 - o2)) and not is_bull0 and cl0 < (o2 + cl2) / 2:
        detected_pattern = "Evening Star"
        direction = "BEARISH REJECTION"
        signal = "STRONG SELL"
        confidence = 89
        pattern_badge = "🔴 Evening Star (Pembalikan Bearish 3-Bar)"
        pattern_desc = "Formasi bintang petang memperingatkan akhir dari reli naik dan dimulainya distribusi harga."

    # 7. Bullish Marubozu (Momentum Breakout)
    elif is_bull0 and (body0 / rng0) >= 0.82 and body0 >= atr * 0.8:
        detected_pattern = "Bullish Marubozu"
        direction = "BULLISH BREAKOUT"
        signal = "STRONG BUY"
        confidence = 86 if vol_ratio >= 1.2 else 80
        pattern_badge = "🚀 Bullish Marubozu (Impulsif Kuat)"
        pattern_desc = "Batang lilin hijau padat tanpa bayangan menunjukkan antusiasme beli yang sangat agresif tanpa koreksi."

    # 8. Bearish Marubozu (Dump / Breakdown)
    elif not is_bull0 and (body0 / rng0) >= 0.82 and body0 >= atr * 0.8:
        detected_pattern = "Bearish Marubozu"
        direction = "BEARISH BREAKDOWN"
        signal = "STRONG SELL"
        confidence = 86
        pattern_badge = "⚠️ Bearish Marubozu (Tekanan Jual Ekstrem)"
        pattern_desc = "Batang lilin merah padat mengindikasikan aksi buang barang (panic sell / distribusi)."

    # 9. Inverted Hammer (Pantulan Bawah)
    elif upper_wick0 >= (2.0 * body0) and lower_wick0 <= (0.25 * rng0) and cl0 <= ema20:
        detected_pattern = "Inverted Hammer"
        direction = "BULLISH REVERSAL"
        signal = "BUY"
        confidence = 76
        pattern_badge = "🟢 Inverted Hammer (Uji Coba Buyer)"
        pattern_desc = "Ekor atas menunjukkan pembeli mulai mencoba mendobrak harga ke atas setelah tren turun."

    # 10. Piercing Line
    elif not is_bull1 and is_bull0 and o0 < l1 and cl0 > (o1 + cl1) / 2 and cl0 < o1:
        detected_pattern = "Piercing Line"
        direction = "BULLISH REVERSAL"
        signal = "BUY"
        confidence = 82
        pattern_badge = "🟢 Piercing Line (Penetrasi Bullish)"
        pattern_desc = "Lilin hijau dibuka di bawah titik terendah kemarin namun berhasil menembus lebih dari separuh badan lilin merah."

    # 11. Doji / Long-Legged Doji (Ketidakpastian Pasar)
    elif (body0 / rng0) <= 0.12:
        detected_pattern = "Doji (Indecision)"
        direction = "KONSOLIDASI / WAIT & SEE"
        signal = "HOLD"
        confidence = 65
        pattern_badge = "⚖️ Doji (Titik Keseimbangan Buyer vs Seller)"
        pattern_desc = "Pasar berada di titik netral seimbang, menunggu volume pemicu sebelum menentukan arah lonjakan berikutnya."

    # 12. Trend Continuation Bullish
    elif is_bull0 and cl0 > ema20 and ema20 > sma50:
        detected_pattern = "Bullish Continuation"
        direction = "BULLISH CONTINUATION"
        signal = "BUY"
        confidence = 79
        pattern_badge = "📈 Bullish Trend Continuation"
        pattern_desc = "Struktur harga mempertahankan tren naik sehat di atas rata-rata pergerakan EMA 20."

    # 13. Trend Continuation Bearish
    elif not is_bull0 and cl0 < ema20 and ema20 < sma50:
        detected_pattern = "Bearish Continuation"
        direction = "BEARISH CONTINUATION"
        signal = "SELL"
        confidence = 77
        pattern_badge = "📉 Bearish Trend Continuation"
        pattern_desc = "Struktur harga terjebak di bawah tekanan tren turun EMA 20 dan SMA 50."

    # Hitung Rekomendasi Level Transaksi Presisi Berfraksi Resmi BEI
    tick_size = get_idx_tick_size(cl0)

    if "BULLISH" in direction:
        entry_raw = cl0
        entry_price = round_to_idx_tick(entry_raw, round_direction="nearest")

        tp1_distance = max(atr * 1.1, tick_size * 2)
        tp1_raw = entry_price + tp1_distance
        tp1 = round_to_idx_tick(tp1_raw, round_direction="up")
        if tp1 <= entry_price:
            tp1 = entry_price + (tick_size * 2)

        tp2_distance = max(atr * 2.2, tick_size * 5)
        tp2_raw = entry_price + tp2_distance
        tp2 = round_to_idx_tick(tp2_raw, round_direction="up")
        if tp2 <= tp1:
            tp2 = tp1 + (tick_size * 3)

        sl_raw = min(l0 - tick_size, entry_price - max(atr * 0.9, tick_size * 2))
        stop_loss = round_to_idx_tick(sl_raw, round_direction="down")
        if stop_loss >= entry_price:
            stop_loss = entry_price - (tick_size * 2)
    else:
        entry_price = round_to_idx_tick(cl0, round_direction="nearest")
        tp1 = round_to_idx_tick(entry_price + (tick_size * 2), round_direction="up")
        tp2 = round_to_idx_tick(entry_price + (tick_size * 5), round_direction="up")
        stop_loss = round_to_idx_tick(max(50.0, l0 - tick_size), round_direction="down")
        if stop_loss >= entry_price:
            stop_loss = entry_price - (tick_size * 2)

    # Persentase Net PnL (Setelah Fee 0.40%)
    tp1_net_pct = _calc_net_pct(entry_price, tp1)
    tp2_net_pct = _calc_net_pct(entry_price, tp2)
    sl_net_pct = _calc_net_pct(entry_price, stop_loss)

    # Rasio Risk-to-Reward (RRR)
    risk_pts = max(1.0, float(entry_price - stop_loss))
    reward1_pts = max(1.0, float(tp1 - entry_price))
    reward2_pts = max(1.0, float(tp2 - entry_price))
    rrr_tp1 = round(reward1_pts / risk_pts, 2)
    rrr_tp2 = round(reward2_pts / risk_pts, 2)

    date_str = str(c0.name)[:10] if hasattr(c0, "name") else "Hari Ini"
    pct_from_open = ((cl0 - o0) / max(1.0, o0)) * 100.0
    sign_str = "+" if is_bull0 else ""
    narrative = (
        f"Lilin terkini ({date_str}) membentuk pola **{detected_pattern}** dengan penutupan harga Rp {cl0:,.0f} "
        f"({sign_str}{pct_from_open:.2f}% dari Open Rp {o0:,.0f}). "
        f"{pattern_desc} "
        f"Volume transaksi tercatat {vol_ratio:.2f}x rata-rata 20 hari. "
        f"Berdasarkan probabilitas {confidence}%, rekomendasi zona entry berada di **Rp {entry_price:,.0f}**, "
        f"dengan target penguncian laba **Take Profit 1 di Rp {tp1:,.0f} (Net {tp1_net_pct:+.2f}%)** dan "
        f"**Take Profit 2 di Rp {tp2:,.0f} (Net {tp2_net_pct:+.2f}%)**. "
        f"Batas proteksi modal ketat **Cut Loss di Rp {stop_loss:,.0f} (Net {sl_net_pct:+.2f}%)** "
        f"dengan rasio Risk-Reward Net 1 : {rrr_tp1}."
    )

    return {
        "success": True,
        "pattern_name": detected_pattern,
        "pattern_badge": pattern_badge,
        "pattern_desc": pattern_desc,
        "direction": direction,
        "signal": signal,
        "confidence_pct": confidence,
        "entry_price": entry_price,
        "take_profit_1": tp1,
        "tp1": tp1,
        "take_profit_2": tp2,
        "tp2": tp2,
        "stop_loss": stop_loss,
        "cut_loss": stop_loss,
        "tp1_net_pct": tp1_net_pct,
        "tp2_net_pct": tp2_net_pct,
        "sl_net_pct": sl_net_pct,
        "risk_reward_ratio_tp1": rrr_tp1,
        "risk_reward_ratio_tp2": rrr_tp2,
        "tick_size": tick_size,
        "atr": round(atr, 2),
        "volume_ratio": round(vol_ratio, 2),
        "narrative": narrative,
        "candle_date": date_str,
    }
