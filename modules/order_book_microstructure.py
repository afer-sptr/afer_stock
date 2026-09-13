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
    candle_info: Optional[Dict[str, Any]] = None,
    tier_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validasi eksekusi taktis HAKA/HAKI, kecepatan likuiditas keluar,
    dan rekomendasi antrean bid vs hajar kanan (offer) terkalibrasi Tier saham.
    """
    pct_bid = float(snap.get("pct_bid", 50.0))
    pct_offer = float(snap.get("pct_offer", 50.0))
    rel_spread = float(snap.get("rel_spread", 0.5))
    bid = float(snap.get("bid", 1000.0))
    ask = float(snap.get("ask", 1005.0))
    bid_size = float(snap.get("bid_size", 1000.0))
    price = float(snap.get("price", bid))
    turnover_idr = float(snap.get("turnover_idr", 0.0))

    # Identifikasi Tier Saham
    tier = (tier_code or snap.get("tier_code", "RECEH")).upper()

    # 1. HAKA Approval Logic
    if pct_bid >= 62.0 and rel_spread < 2.0:
        haka_status = "APPROVED"
        haka_badge = "🟢 HAKA APPROVED"
        haka_desc = f"HAKA Disetujui: Tekanan beli dominan ({pct_bid:.1f}% Bid vs {pct_offer:.1f}% Offer) dengan spread rapat {rel_spread:.2f}%. Eksekusi Hajar Kanan aman."
    elif 45.0 <= pct_bid < 62.0:
        haka_status = "BLOCKED"
        haka_badge = "⛔ HAKA BLOCKED"
        haka_desc = f"HAKA Ditahan: Order book berimbang ({pct_bid:.1f}% Bid vs {pct_offer:.1f}% Offer). Disarankan pasang antrean pasif di Best Bid Rp {bid:,.0f}."
    else:
        haka_status = "VETO"
        haka_badge = "🚫 HAKA VETO"
        haka_desc = f"HAKA Ditolak: Antrean offer tebal ({pct_offer:.1f}% Offer) menahan kenaikan harga. Waspada tekanan jual & risiko guyuran."

    # 2. Emergency HAKI Logic
    if pct_offer >= 62.0 or pct_bid < 38.0:
        emergency_haki = True
        haki_badge = "🚨 EMERGENCY HAKI"
        haki_desc = f"Peringatan Jual Darurat: Tekanan offer masif ({pct_offer:.1f}% Offer). Disarankan segera buang barang ke Best Bid Rp {bid:,.0f}."
    else:
        emergency_haki = False
        haki_badge = "🟡 NORMAL EXIT"
        haki_desc = f"Keluar Teratur: Buku pesanan kondusif ({pct_bid:.1f}% Bid). Pasang antrean pasif bertahap pada target Take Profit."

    # 3. Safe Exit Lot Size & Kecepatan Keluar Terkalibrasi Berdasarkan Tier
    safe_exit_lot = max(1, safe_int(bid_size * 0.20, 10))
    safe_exit_idr = float(safe_exit_lot * 100 * price)

    if tier in {"PREMIUM", "L1"}:
        # Saham Premium / Blue Chip (> Rp 5.000)
        if safe_exit_lot >= 100 and pct_bid >= 55.0:
            exit_speed = "🟢 INSTAN (< 1 Menit)"
            exit_tier = "Sangat Likuid (Blue-Chip)"
        elif safe_exit_lot >= 20:
            exit_speed = "🟡 CEPAT (1–5 Menit)"
            exit_tier = "Likuiditas Cukup"
        else:
            exit_speed = "🔴 HATI-HATI (> 15 Menit)"
            exit_tier = "Antrean Tipis"
    elif tier in {"GOCAP", "L3"}:
        # Saham Gocap / Tidur (Rp 50 – Rp 100)
        if turnover_idr < 250_000_000 or bid_size <= 50:
            exit_speed = "🔴 RISIKO TERTIDUR (> 1 Jam)"
            exit_tier = "Saham Tidur (Likuiditas Rendah)"
            safe_exit_lot = min(safe_exit_lot, 50)
            safe_exit_idr = float(safe_exit_lot * 100 * price)
        elif safe_exit_lot >= 2000 and pct_bid >= 60.0:
            exit_speed = "🟢 INSTAN (< 2 Menit)"
            exit_tier = "Rally Aktif (Likuid)"
        else:
            exit_speed = "🟡 SEDANG (5–15 Menit)"
            exit_tier = "Likuiditas Terbatas"
    else:
        # Saham Receh / Murah (Rp 100 – Rp 1.000)
        if safe_exit_lot >= 300 and pct_bid >= 55.0:
            exit_speed = "🟢 INSTAN (< 2 Menit)"
            exit_tier = "Likuiditas Tinggi"
        elif safe_exit_lot >= 50:
            exit_speed = "🟡 SEDANG (5–10 Menit)"
            exit_tier = "Likuiditas Cukup"
        else:
            exit_speed = "🔴 RISIKO SLIPPAGE (> 20 Menit)"
            exit_tier = "Antrean Tipis"

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
        "haka_badge": haka_badge,
        "haka_desc": haka_desc,
        "emergency_haki": emergency_haki,
        "haki_badge": haki_badge,
        "haki_desc": haki_desc,
        "safe_exit_lot": safe_exit_lot,
        "safe_exit_idr": safe_exit_idr,
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
        "tier_code": tier,
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
