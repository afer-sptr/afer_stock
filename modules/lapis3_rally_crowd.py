"""
lapis3_rally_crowd.py
=====================
Mesin Pemindai Saham Lapis 3 (Small-Cap Rally Hunter) dan
Analisis Sentimen Komunitas Ritel & Detektor Sinyal Kontrarian (Crowd Lab).
"""

from typing import Any, Dict
import pandas as pd


def screen_lapis_3(df: pd.DataFrame, snap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deteksi potensi rally saham lapis 3 berbasis Relative Volume (RVol > 2.0x),
    Bollinger Bandwidth Squeeze, Turnover > Rp 1 Miliar, dan dominasi Bid >= 65%.
    """
    if df is None or len(df) < 25:
        return {
            "is_rally": False,
            "score": 0.0,
            "rvol": 1.0,
            "is_squeeze": False,
            "turnover_idr": 0.0,
            "reason": "Data historis tidak mencukupi"
        }

    vol = df["Volume"]
    close = df["Close"]
    avg_vol_20 = vol.rolling(20).mean().iloc[-1]
    rvol = float(vol.iloc[-1] / max(1.0, avg_vol_20))

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + (2.0 * std20)
    bb_lower = sma20 - (2.0 * std20)
    bbw = ((bb_upper - bb_lower) / sma20).dropna()
    is_squeeze = bool(bbw.iloc[-2] <= bbw.quantile(0.25) and bbw.iloc[-1] > bbw.iloc[-2]) if len(bbw) >= 2 else False

    turnover_idr = float(close.iloc[-1] * vol.iloc[-1])
    turnover_ok = turnover_idr >= 1_000_000_000.0
    der_val = float(snap.get("der", 1.0))
    der_ok = der_val < 3.5
    pct_bid = float(snap.get("pct_bid", 50.0))
    order_book_ok = pct_bid >= 65.0

    is_rally = bool((rvol >= 2.0) and is_squeeze and turnover_ok and der_ok and order_book_ok)
    score = (
        (min(rvol, 5.0) / 5.0) * 35.0
        + (30.0 if is_squeeze else 0.0)
        + ((pct_bid / 100.0) * 35.0)
    )

    return {
        "is_rally": is_rally,
        "rvol": round(rvol, 2),
        "is_squeeze": is_squeeze,
        "turnover_idr": turnover_idr,
        "score": round(score, 1),
    }


def evaluate_crowd_contrarian(snap: Dict[str, Any], news_score: float) -> Dict[str, Any]:
    """
    Analisis sentimen obrolan komunitas & sinyal kontrarian:
    - Jebakan Distribusi / Pucuk FOMO
    - Sinyal Pembalikan / Contrarian Reversal / Buy the Panic
    """
    pct_bid = float(snap.get("pct_bid", 50.0))
    pct_offer = float(snap.get("pct_offer", 50.0))
    obi = float(snap.get("obi", 0.0))
    volume = float(snap.get("volume", 500000.0))

    crowd_sentiment = max(-1.0, min(1.0, news_score + (obi * 0.5)))
    buzz_velocity = float(volume / 250000.0)

    if crowd_sentiment > 0.65 and pct_offer >= 58.0:
        contrarian_signal = "⚠️ DISTRIBUTION TRAP / PUCUK FOMO"
        contrarian_desc = "Komunitas sangat bullish namun antrean offer membengkak. Waspada aksi ambil untung & guyuran bandar!"
    elif crowd_sentiment < -0.40 and pct_bid >= 65.0:
        contrarian_signal = "💎 CONTRARIAN REVERSAL / BUY THE PANIC"
        contrarian_desc = "Ketakutan ritel memuncak (panic sell) namun antrean bid tebal menyerap barang. Peluang emas akumulasi di dasar!"
    else:
        contrarian_signal = "🟢 NORMAL FLOW"
        contrarian_desc = "Aliran sentimen komunitas bergerak wajar dan selaras dengan pergerakan buku pesanan."

    return {
        "crowd_sentiment": round(crowd_sentiment, 2),
        "buzz_velocity": round(buzz_velocity, 2),
        "contrarian_signal": contrarian_signal,
        "contrarian_desc": contrarian_desc,
    }
