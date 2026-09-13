"""
order_book_microstructure.py
============================
Analisis Mikrostruktur Buku Pesanan (Order Book), Validasi Eksekusi Taktis (HAKA/HAKI),
Kalkulator Safe Exit Lot, dan Proyeksi Net PnL (Potongan Fee Broker & Pajak BEI).
"""

from typing import Any, Dict, Optional
from modules.idx_ticks import round_to_idx_tick, get_idx_tick_size, safe_int


def evaluate_order_book_execution(
    snap: Dict[str, Any],
    candle_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validasi eksekusi taktis HAKA/HAKI, kecepatan likuiditas keluar,
    dan rekomendasi antrean bid vs hajar kanan (offer).
    """
    pct_bid = float(snap.get("pct_bid", 50.0))
    pct_offer = float(snap.get("pct_offer", 50.0))
    rel_spread = float(snap.get("rel_spread", 0.5))
    bid = float(snap.get("bid", 1000.0))
    ask = float(snap.get("ask", 1005.0))
    bid_size = float(snap.get("bid_size", 10000.0))

    # 1. HAKA Approval Logic
    if pct_bid >= 65.0 and rel_spread < 1.5:
        haka_status = "APPROVED"
        haka_desc = "🟢 HAKA APPROVED: Tekanan beli masif (% Bid >= 65%), antrean offer tipis."
    elif 45.0 <= pct_bid < 65.0:
        haka_status = "BLOCKED"
        haka_desc = "⛔ HAKA BLOCKED: Antre pasif di Best Bid, hindari mengejar offer."
    else:
        haka_status = "VETO"
        haka_desc = "🚫 HAKA VETO: Offer tebal menahan kenaikan, potensi guyuran harga."

    # 2. Emergency HAKI Logic
    if pct_offer >= 65.0 or pct_bid < 35.0:
        emergency_haki = True
        haki_desc = f"🚨 EMERGENCY HAKI: Tekanan offer masif ({pct_offer:.1f}% Offer)! Segera buang barang ke Best Bid."
    else:
        emergency_haki = False
        haki_desc = "🟡 NORMAL EXIT: Order book seimbang, pasang antrean pasif TP."

    # 3. Safe Exit Lot Size (20% dari volume Best Bid agar tidak merusak harga saat exit)
    safe_exit_lot = max(1, safe_int(bid_size * 0.20, 100))
    if safe_exit_lot >= 1000 and pct_bid >= 60.0:
        exit_speed = "🟢 INSTAN (< 1 Menit)"
        exit_tier = "Sangat Likuid"
    elif safe_exit_lot >= 200:
        exit_speed = "🟡 SEDANG (5–15 Menit)"
        exit_tier = "Likuiditas Cukup"
    else:
        exit_speed = "🔴 LAMBAT / RISIKO SLIPPAGE (> 30 Menit)"
        exit_tier = "Likuiditas Tipis"

    haka_entry = ask
    bid_entry = bid

    # Target & Stop Loss berfraksi BEI
    tp1 = round_to_idx_tick(haka_entry * 1.03, "up")
    tp2 = round_to_idx_tick(haka_entry * 1.06, "up")
    sl = round_to_idx_tick(haka_entry * 0.975, "down")

    fee_roundtrip = 0.0040  # 0.15% beli + 0.25% jual (termasuk PPh 0.1%)
    gain_tp1_net = ((tp1 - haka_entry) / haka_entry - fee_roundtrip) * 100.0
    gain_tp2_net = ((tp2 - haka_entry) / haka_entry - fee_roundtrip) * 100.0
    risk_sl = abs((sl - haka_entry) / haka_entry + fee_roundtrip) * 100.0
    rrr = gain_tp1_net / max(0.01, risk_sl)

    return {
        "haka_status": haka_status,
        "haka_desc": haka_desc,
        "emergency_haki": emergency_haki,
        "haki_desc": haki_desc,
        "safe_exit_lot": safe_exit_lot,
        "exit_speed": exit_speed,
        "exit_tier": exit_tier,
        "haka_entry": haka_entry,
        "bid_entry": bid_entry,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "gain_tp1_net": gain_tp1_net,
        "gain_tp2_net": gain_tp2_net,
        "risk_sl": risk_sl,
        "rrr": rrr,
        "pct_bid": pct_bid,
        "pct_offer": pct_offer,
        "rel_spread": rel_spread,
    }


def calculate_net_pnl(
    entry_price: float,
    exit_price: float,
    shares: int,
    buy_fee_pct: float = 0.15,
    sell_fee_pct: float = 0.25,
) -> Dict[str, Any]:
    """
    Menghitung laba/rugi bersih (Net PnL) setelah dipotong komisi broker beli,
    komisi broker jual, levy BEI/KPEI/KSEI, dan PPh Final 0.1%.
    """
    gross_buy = entry_price * shares
    buy_fee = gross_buy * (buy_fee_pct / 100.0)
    total_capital_out = gross_buy + buy_fee

    gross_sell = exit_price * shares
    sell_fee = gross_sell * (sell_fee_pct / 100.0)
    total_proceeds = gross_sell - sell_fee

    net_pnl_idr = total_proceeds - total_capital_out
    net_pnl_pct = (net_pnl_idr / total_capital_out) * 100.0 if total_capital_out > 0 else 0.0
    total_fees = buy_fee + sell_fee

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "lots": shares // 100,
        "gross_buy": gross_buy,
        "buy_fee": buy_fee,
        "total_capital_out": total_capital_out,
        "gross_sell": gross_sell,
        "sell_fee": sell_fee,
        "total_fees": total_fees,
        "total_proceeds": total_proceeds,
        "net_pnl_idr": net_pnl_idr,
        "net_pnl_pct": net_pnl_pct,
    }
