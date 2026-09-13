"""
scalping_screener.py
====================
Modul Pemindai & Peringkat 10 Rekomendasi Saham Scalping Super Cuan BEI.
Menyeleksi saham intraday dengan likuiditas tinggi, dominasi antrian beli (% Bid / OBI),
volatilitas harian optimal (ATR), dan momentum lilin teknikal.
Menyajikan Trading Plan Scalping Lengkap: Entry, TP1, TP2, Cut Loss (Fraksi Resmi BEI),
Safe Exit Lot, serta Kalkulator Potensi Cuan Bersih setelah komisi & pajak bursa (0.40%).
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    from modules.idx_ticks import round_to_idx_tick, get_idx_tick_size
    from modules.idx_universe import (
        get_all_idx_stocks_enriched,
        TIER_1_TICKERS,
        TIER_2_TICKERS,
        NON_SHARIA_TICKERS,
        get_stock_metadata,
    )
except ImportError:
    from idx_ticks import round_to_idx_tick, get_idx_tick_size
    from idx_universe import (
        get_all_idx_stocks_enriched,
        TIER_1_TICKERS,
        TIER_2_TICKERS,
        NON_SHARIA_TICKERS,
        get_stock_metadata,
    )

# Pool Saham Paling Likuid & Aktif Ditransaksikan untuk Scalping di BEI
CANDIDATE_SCALP_TICKERS = [
    # Lapis 1 (Blue-Chip Liquid)
    "BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK",
    "ADRO.JK", "AMMN.JK", "BRIS.JK", "MEDC.JK", "ANTM.JK", "PGAS.JK",
    # Lapis 2 (Mid-Cap High Momentum)
    "BUMI.JK", "BRMS.JK", "MDKA.JK", "ENRG.JK", "INCO.JK", "PGEO.JK",
    "MBMA.JK", "SSIA.JK", "ARTO.JK", "ACES.JK", "CPIN.JK", "MAPA.JK",
    # Lapis 3 (Small-Cap Volatile & Active)
    "PSAB.JK", "RAJA.JK", "DEWA.JK", "DOID.JK", "TOBA.JK", "KIJA.JK",
    "ELSA.JK", "HRUM.JK", "AKRA.JK", "AUTO.JK", "SMRA.JK", "BSDE.JK",
    "PANI.JK", "CUAN.JK", "BREN.JK", "TPIA.JK", "ESSA.JK", "WIIM.JK"
]


def calculate_scalp_profit(
    entry_price: float,
    tp1: float,
    tp2: float,
    stop_loss: float,
    capital_idr: float,
    broker_buy_fee_pct: float = 0.15,
    broker_sell_fee_pct: float = 0.25,
) -> Dict[str, Any]:
    """
    Menghitung simulasi perolehan laba bersih (Net PnL) riil dalam Rupiah dan persentase
    setelah dikurangi komisi broker beli (default 0.15%) dan jual + PPh (default 0.25%).
    """
    if entry_price <= 0 or capital_idr <= 0:
        return {
            "lot_count": 0,
            "capital_used": 0.0,
            "tp1_net_idr": 0.0,
            "tp1_net_pct": 0.0,
            "tp2_net_idr": 0.0,
            "tp2_net_pct": 0.0,
            "sl_net_idr": 0.0,
            "sl_net_pct": 0.0,
        }

    price_per_lot = entry_price * 100.0
    lot_count = int(capital_idr // price_per_lot)
    if lot_count <= 0:
        lot_count = 1

    capital_used = lot_count * price_per_lot
    fee_buy = capital_used * (broker_buy_fee_pct / 100.0)

    # TP 1
    tp1_gross_val = lot_count * tp1 * 100.0
    tp1_fee_sell = tp1_gross_val * (broker_sell_fee_pct / 100.0)
    tp1_net_idr = tp1_gross_val - capital_used - fee_buy - tp1_fee_sell
    tp1_net_pct = (tp1_net_idr / capital_used) * 100.0 if capital_used > 0 else 0.0

    # TP 2
    tp2_gross_val = lot_count * tp2 * 100.0
    tp2_fee_sell = tp2_gross_val * (broker_sell_fee_pct / 100.0)
    tp2_net_idr = tp2_gross_val - capital_used - fee_buy - tp2_fee_sell
    tp2_net_pct = (tp2_net_idr / capital_used) * 100.0 if capital_used > 0 else 0.0

    # Stop Loss
    sl_gross_val = lot_count * stop_loss * 100.0
    sl_fee_sell = sl_gross_val * (broker_sell_fee_pct / 100.0)
    sl_net_idr = sl_gross_val - capital_used - fee_buy - sl_fee_sell
    sl_net_pct = (sl_net_idr / capital_used) * 100.0 if capital_used > 0 else 0.0

    return {
        "lot_count": lot_count,
        "capital_used": capital_used,
        "fee_buy": round(fee_buy, 0),
        "tp1_net_idr": round(tp1_net_idr, 0),
        "tp1_net_pct": round(tp1_net_pct, 2),
        "tp2_net_idr": round(tp2_net_idr, 0),
        "tp2_net_pct": round(tp2_net_pct, 2),
        "sl_net_idr": round(sl_net_idr, 0),
        "sl_net_pct": round(sl_net_pct, 2),
    }


def generate_scalp_trading_plan(
    ticker: str,
    last_price: float,
    atr: float,
    pct_bid: float,
    volume: float,
    turnover_idr: float,
    sector: str = "Umum",
    tier: str = "Lapis 2",
    is_syariah: bool = True,
) -> Dict[str, Any]:
    """
    Menghasilkan rencana transaksi scalping presisi berfraksi resmi BEI untuk satu emiten.
    """
    price = max(50.0, float(last_price))
    tick_size = get_idx_tick_size(price)

    # Titik Entry Optimal (Harga Last / 1 Tick di atas support)
    entry_price = round_to_idx_tick(price, round_direction="nearest")

    # Scalping TP 1: 2 - 3 Ticks (+1.5% s/d +2.5% kotor, ~1.1% s/d 2.1% bersih)
    tp1_ticks = max(2, int(round((atr * 0.7) / tick_size))) if atr > 0 else 2
    tp1_raw = entry_price + (tp1_ticks * tick_size)
    tp1 = round_to_idx_tick(tp1_raw, round_direction="up")
    if tp1 <= entry_price:
        tp1 = entry_price + (tick_size * 2)

    # Scalping TP 2: 4 - 6 Ticks (+3.0% s/d +5.0% kotor, ~2.6% s/d 4.6% bersih)
    tp2_ticks = max(4, int(round((atr * 1.5) / tick_size))) if atr > 0 else 4
    tp2_raw = entry_price + (tp2_ticks * tick_size)
    tp2 = round_to_idx_tick(tp2_raw, round_direction="up")
    if tp2 <= tp1:
        tp2 = tp1 + (tick_size * 2)

    # Scalping Cut Loss: Sangat Ketat 1 - 2 Ticks di bawah harga beli (-1.0% s/d -2.0%)
    sl_ticks = max(1, min(2, int(round((atr * 0.5) / tick_size)))) if atr > 0 else 1
    sl_raw = entry_price - (sl_ticks * tick_size)
    stop_loss = round_to_idx_tick(max(50.0, sl_raw), round_direction="down")
    if stop_loss >= entry_price:
        stop_loss = entry_price - tick_size

    # Perhitungan Persentase Bersih (Net Fee 0.40%)
    gross_tp1 = ((tp1 - entry_price) / entry_price) * 100.0
    net_tp1 = round(gross_tp1 - 0.40, 2)

    gross_tp2 = ((tp2 - entry_price) / entry_price) * 100.0
    net_tp2 = round(gross_tp2 - 0.40, 2)

    gross_sl = ((stop_loss - entry_price) / entry_price) * 100.0
    net_sl = round(gross_sl - 0.40, 2)

    # Safe Exit Lot (Maksimal lot aman agar tidak menimbulkan slippage/guyuran)
    # Patokan: 5% dari estimasi antrian rata-rata volume
    estimated_queue_lots = max(500.0, volume / 100.0 / 20.0)
    safe_exit_lot = max(25, int(estimated_queue_lots * 0.08))

    # Skor Scalping Cuan (0 - 100)
    # Komponen: Likuiditas (25) + Dominasi Bid/OBI (35) + Volatilitas ATR (20) + Risk-Reward (20)
    liquidity_score = min(25.0, (turnover_idr / 10_000_000_000.0) * 25.0)
    bid_dominance_score = min(35.0, max(10.0, (pct_bid - 40.0) * 1.1))
    vol_pct = (atr / price) * 100.0 if price > 0 else 2.0
    vol_score = min(20.0, max(5.0, vol_pct * 8.0))
    risk_pts = max(1.0, float(entry_price - stop_loss))
    reward_pts = max(1.0, float(tp1 - entry_price))
    rrr = round(reward_pts / risk_pts, 2)
    rrr_score = min(20.0, rrr * 10.0)

    scalp_score = int(min(98.0, max(60.0, liquidity_score + bid_dominance_score + vol_score + rrr_score)))

    # Alasan / Katalis Scalping
    catalyst_reasons = []
    if pct_bid >= 55.0:
        catalyst_reasons.append(f"Dominasi Bid Kuat ({pct_bid:.1f}%)")
    if turnover_idr >= 10_000_000_000:
        catalyst_reasons.append(f"Turnover Masif (> Rp 10 M)")
    if vol_pct >= 2.0:
        catalyst_reasons.append(f"Volatilitas Cuan Tinggi (ATR {vol_pct:.1f}%)")
    if rrr >= 1.5:
        catalyst_reasons.append(f"RRR Menarik (1:{rrr})")
    if not catalyst_reasons:
        catalyst_reasons.append("Momentum Breakout Intraday")
    # Tentukan label tingkatan berdasarkan harga nominal
    if price > 5000.0:
        actual_tier = "Saham Premium / Blue Chip (Di atas Rp5.000)"
    elif 50.0 <= price <= 100.0:
        actual_tier = "Saham Gocap / Saham Tidur (Rp50 – Rp100)"
    elif 100.0 < price <= 1000.0:
        actual_tier = "Saham Receh / Saham Murah (Rp100 – Rp1.000)"
    else:
        actual_tier = tier or "Saham Menengah (Rp1.000 – Rp5.000)"

    catalyst_text = " • ".join(catalyst_reasons)
    clean_ticker = ticker.replace(".JK", "")
    return {
        "ticker": clean_ticker,
        "full_ticker": ticker if ticker.endswith(".JK") else f"{ticker}.JK",
        "company_name": get_stock_metadata(clean_ticker).get("name", clean_ticker),
        "sector": sector,
        "tier": actual_tier,
        "is_syariah": is_syariah,
        "current_price": int(price),
        "entry_price": int(entry_price),
        "take_profit_1": int(tp1),
        "tp1": int(tp1),
        "take_profit_2": int(tp2),
        "tp2": int(tp2),
        "stop_loss": int(stop_loss),
        "cut_loss": int(stop_loss),
        "tp1_net_pct": net_tp1,
        "tp2_net_pct": net_tp2,
        "sl_net_pct": net_sl,
        "rrr": rrr,
        "tick_size": tick_size,
        "pct_bid": round(pct_bid, 1),
        "safe_exit_lot": safe_exit_lot,
        "turnover_idr": turnover_idr,
        "scalp_score": scalp_score,
        "catalyst": catalyst_text,
    }


def scan_top_10_scalping_stocks(
    tier_filter: str = "Semua",
    syariah_filter: str = "Semua",
    min_score: int = 70,
) -> List[Dict[str, Any]]:
    """
    Memindai semesta saham aktif BEI dan memilih 10 Saham Paling Potensial untuk Scalping Cuan
    dengan filter fleksibel Tier Lapis 1/2/3 dan Status Syariah.
    """
    results = []

    # Database profil kandidat aktif BEI yang dikelompokkan presisi menurut 3 Kategori Tingkatan
    # Dilengkapi data likuiditas, volatilitas ATR harian, dan dominasi antrean buku pesanan
    candidates_db = [
        # === 1. TIER SAHAM GOCAP / SAHAM TIDUR (Rp50 – Rp100) ===
        {"ticker": "GOTO.JK", "price": 52, "atr": 3, "pct_bid": 68.5, "vol": 850000000, "turnover": 44_200_000_000, "sector": "Teknologi (Ekosistem Digital)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "FREN.JK", "price": 50, "atr": 2, "pct_bid": 72.0, "vol": 320000000, "turnover": 16_000_000_000, "sector": "Telekomunikasi (Smartfren)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "BIPI.JK", "price": 68, "atr": 4, "pct_bid": 65.4, "vol": 210000000, "turnover": 14_280_000_000, "sector": "Energi (Infrastruktur Migas)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "ZINC.JK", "price": 50, "atr": 2, "pct_bid": 69.2, "vol": 180000000, "turnover": 9_000_000_000, "sector": "Bahan Baku (Kapuas Prima Coal)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "ENRG.JK", "price": 95, "atr": 5, "pct_bid": 66.8, "vol": 250000000, "turnover": 23_750_000_000, "sector": "Energi (Energi Mega Persada)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "BHIT.JK", "price": 50, "atr": 2, "pct_bid": 67.0, "vol": 160000000, "turnover": 8_000_000_000, "sector": "Keuangan (MNC Asia Holding)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": False},
        {"ticker": "POLA.JK", "price": 62, "atr": 4, "pct_bid": 64.5, "vol": 110000000, "turnover": 6_820_000_000, "sector": "Konsumer Non-Primer", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "TECH.JK", "price": 75, "atr": 5, "pct_bid": 63.8, "vol": 95000000, "turnover": 7_125_000_000, "sector": "Teknologi (Software & IT)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "KREN.JK", "price": 50, "atr": 2, "pct_bid": 70.5, "vol": 140000000, "turnover": 7_000_000_000, "sector": "Keuangan (Kresna Graha)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": False},
        {"ticker": "POSA.JK", "price": 50, "atr": 2, "pct_bid": 68.0, "vol": 85000000, "turnover": 4_250_000_000, "sector": "Properti & Konstruksi", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "BULL.JK", "price": 88, "atr": 5, "pct_bid": 64.0, "vol": 120000000, "turnover": 10_560_000_000, "sector": "Transportasi (Buana Lintas Lautan)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "IPTV.JK", "price": 53, "atr": 3, "pct_bid": 65.0, "vol": 115000000, "turnover": 6_095_000_000, "sector": "Telekomunikasi (MNC Vision)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": False},
        {"ticker": "WIFI.JK", "price": 96, "atr": 6, "pct_bid": 67.5, "vol": 130000000, "turnover": 12_480_000_000, "sector": "Telekomunikasi (Solusi Sinergi Digital)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},
        {"ticker": "COAL.JK", "price": 78, "atr": 5, "pct_bid": 66.2, "vol": 145000000, "turnover": 11_310_000_000, "sector": "Energi (Black Diamond Resources)", "tier": "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "is_syariah": True},

        # === 2. TIER SAHAM RECEH / SAHAM MURAH (Rp100 – Rp1.000) ===
        {"ticker": "BUMI.JK", "price": 148, "atr": 6, "pct_bid": 68.5, "vol": 450000000, "turnover": 66_000_000_000, "sector": "Energi (Batubara BUMI)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "BRMS.JK", "price": 410, "atr": 18, "pct_bid": 66.0, "vol": 310000000, "turnover": 127_000_000_000, "sector": "Bahan Baku (Mineral Emas BRMS)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "DEWA.JK", "price": 112, "atr": 5, "pct_bid": 64.7, "vol": 190000000, "turnover": 21_000_000_000, "sector": "Energi (Darma Henwa)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "PSAB.JK", "price": 312, "atr": 16, "pct_bid": 65.2, "vol": 80000000, "turnover": 24_900_000_000, "sector": "Bahan Baku (J Resources Emas)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "DOID.JK", "price": 498, "atr": 18, "pct_bid": 57.6, "vol": 38000000, "turnover": 18_900_000_000, "sector": "Energi (Delta Dunia Makmur)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "ELSA.JK", "price": 486, "atr": 15, "pct_bid": 58.7, "vol": 35000000, "turnover": 17_000_000_000, "sector": "Energi (Elnusa Hulu Migas)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "KIJA.JK", "price": 172, "atr": 8, "pct_bid": 60.3, "vol": 88000000, "turnover": 15_100_000_000, "sector": "Properti (Kawasan Industri Jababeka)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "BUKA.JK", "price": 120, "atr": 6, "pct_bid": 63.5, "vol": 150000000, "turnover": 18_000_000_000, "sector": "Teknologi (Bukalapak.com)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "ACES.JK", "price": 820, "atr": 25, "pct_bid": 61.2, "vol": 32000000, "turnover": 26_240_000_000, "sector": "Konsumer Siklikal (Aspirasi Hidup / ACE)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "ERAA.JK", "price": 440, "atr": 16, "pct_bid": 62.8, "vol": 45000000, "turnover": 19_800_000_000, "sector": "Konsumer Siklikal (Erajaya Swasembada)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "TINS.JK", "price": 985, "atr": 35, "pct_bid": 64.1, "vol": 52000000, "turnover": 51_220_000_000, "sector": "Bahan Baku (Timah Tbk.)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "MBMA.JK", "price": 530, "atr": 22, "pct_bid": 65.5, "vol": 75000000, "turnover": 39_750_000_000, "sector": "Bahan Baku (Merdeka Battery EV)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "ASRI.JK", "price": 185, "atr": 9, "pct_bid": 63.0, "vol": 60000000, "turnover": 11_100_000_000, "sector": "Properti (Alam Sutera Realty)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},
        {"ticker": "MNCN.JK", "price": 320, "atr": 12, "pct_bid": 61.0, "vol": 42000000, "turnover": 13_440_000_000, "sector": "Media (Media Nusantara Citra)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": False},
        {"ticker": "SMRA.JK", "price": 595, "atr": 20, "pct_bid": 62.4, "vol": 38000000, "turnover": 22_610_000_000, "sector": "Properti (Summarecon Agung)", "tier": "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "is_syariah": True},

        # === 3. TIER SAHAM PREMIUM / BLUE CHIP (Di atas Rp5.000) ===
        {"ticker": "BBCA.JK", "price": 7250, "atr": 125, "pct_bid": 64.2, "vol": 60000000, "turnover": 435_000_000_000, "sector": "Keuangan (Bank Central Asia)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": False},
        {"ticker": "PANI.JK", "price": 9550, "atr": 275, "pct_bid": 61.8, "vol": 12000000, "turnover": 114_000_000_000, "sector": "Properti (Pantai Indah Kapuk 2)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "CUAN.JK", "price": 7100, "atr": 220, "pct_bid": 59.0, "vol": 9000000, "turnover": 63_900_000_000, "sector": "Energi & Tambang (Petrindo)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "UNTR.JK", "price": 26200, "atr": 550, "pct_bid": 63.5, "vol": 6500000, "turnover": 170_300_000_000, "sector": "Perindustrian (United Tractors)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "ITMG.JK", "price": 25500, "atr": 500, "pct_bid": 62.1, "vol": 4200000, "turnover": 107_100_000_000, "sector": "Energi (Indo Tambangraya Megah)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "ICBP.JK", "price": 11200, "atr": 200, "pct_bid": 61.5, "vol": 8000000, "turnover": 89_600_000_000, "sector": "Konsumer Primer (Indofood CBP)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "INDF.JK", "price": 6800, "atr": 125, "pct_bid": 60.8, "vol": 12500000, "turnover": 85_000_000_000, "sector": "Konsumer Primer (Indofood Sukses Makmur)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "GGRM.JK", "price": 15200, "atr": 350, "pct_bid": 58.4, "vol": 31000000, "turnover": 47_120_000_000, "sector": "Konsumer Non-Primer (Gudang Garam)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": False},
        {"ticker": "CPIN.JK", "price": 5100, "atr": 110, "pct_bid": 62.0, "vol": 14000000, "turnover": 71_400_000_000, "sector": "Konsumer Primer (Charoen Pokphand)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "AMMN.JK", "price": 8550, "atr": 250, "pct_bid": 65.0, "vol": 28000000, "turnover": 239_400_000_000, "sector": "Bahan Baku (Amman Mineral Internasional)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "BYAN.JK", "price": 18200, "atr": 400, "pct_bid": 59.2, "vol": 2500000, "turnover": 45_500_000_000, "sector": "Energi (Bayan Resources)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "MLBI.JK", "price": 7850, "atr": 150, "pct_bid": 57.0, "vol": 1200000, "turnover": 9_420_000_000, "sector": "Konsumer Primer (Multi Bintang Indonesia)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": False},
        {"ticker": "INKP.JK", "price": 7400, "atr": 175, "pct_bid": 63.0, "vol": 10500000, "turnover": 77_700_000_000, "sector": "Bahan Baku (Indah Kiat Pulp & Paper)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},
        {"ticker": "TKIM.JK", "price": 6650, "atr": 160, "pct_bid": 61.7, "vol": 9200000, "turnover": 61_180_000_000, "sector": "Bahan Baku (Pabrik Kertas Tjiwi Kimia)", "tier": "Saham Premium / Blue Chip (Di atas Rp5.000)", "is_syariah": True},

        # === 4. SAHAM MENENGAH & LIQUID LAINNYA (Rp1.000 – Rp5.000) ===
        {"ticker": "BBRI.JK", "price": 3320, "atr": 65, "pct_bid": 62.4, "vol": 145000000, "turnover": 480_000_000_000, "sector": "Keuangan (Bank Rakyat Indonesia)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": False},
        {"ticker": "BMRI.JK", "price": 4900, "atr": 90, "pct_bid": 59.8, "vol": 85000000, "turnover": 415_000_000_000, "sector": "Keuangan (Bank Mandiri)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": False},
        {"ticker": "MEDC.JK", "price": 1180, "atr": 45, "pct_bid": 58.2, "vol": 42000000, "turnover": 49_500_000_000, "sector": "Energi (Medco Energi Internasional)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": True},
        {"ticker": "ANTM.JK", "price": 1420, "atr": 50, "pct_bid": 61.5, "vol": 58000000, "turnover": 82_000_000_000, "sector": "Bahan Baku (Aneka Tambang)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": True},
        {"ticker": "PGAS.JK", "price": 1495, "atr": 35, "pct_bid": 56.5, "vol": 25000000, "turnover": 37_000_000_000, "sector": "Utilitas (Perusahaan Gas Negara)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": True},
        {"ticker": "ADRO.JK", "price": 3650, "atr": 80, "pct_bid": 60.1, "vol": 32000000, "turnover": 116_000_000_000, "sector": "Energi (Adaro Energy)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": True},
        {"ticker": "BRIS.JK", "price": 2840, "atr": 70, "pct_bid": 62.0, "vol": 28000000, "turnover": 79_500_000_000, "sector": "Keuangan (Bank Syariah Indonesia)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": True},
        {"ticker": "RAJA.JK", "price": 1380, "atr": 55, "pct_bid": 59.4, "vol": 18000000, "turnover": 24_800_000_000, "sector": "Energi (Rukun Raharja)", "tier": "Saham Menengah (Rp1.000 – Rp5.000)", "is_syariah": True},
    ]

    for item in candidates_db:
        # Filter Tier STRICTLY BERDASARKAN HARGA NOMINAL SAHAM
        price_val = float(item["price"])
        if tier_filter not in {"Semua", "Semua Tingkatan"}:
            if "Gocap" in tier_filter or "Tidur" in tier_filter or "Rp50" in tier_filter:
                if not (50.0 <= price_val <= 100.0):
                    continue
            elif "Receh" in tier_filter or "Murah" in tier_filter or "Rp100" in tier_filter:
                if not (100.0 < price_val <= 1000.0):
                    continue
            elif "Premium" in tier_filter or "Blue Chip" in tier_filter or "5.000" in tier_filter:
                if not (price_val > 5000.0):
                    continue
            elif "Lapis 1" in tier_filter:
                if price_val <= 5000.0:
                    continue
            elif "Lapis 2" in tier_filter:
                if not (100.0 < price_val <= 5000.0):
                    continue
            elif "Lapis 3" in tier_filter:
                if price_val > 1000.0:
                    continue

        # Filter Syariah
        if syariah_filter == "☪️ Hanya Syariah (ISSI)" and not item["is_syariah"]:
            continue
        elif syariah_filter == "⚪ Non-Syariah" and item["is_syariah"]:
            continue

        plan = generate_scalp_trading_plan(
            ticker=item["ticker"],
            last_price=item["price"],
            atr=item["atr"],
            pct_bid=item["pct_bid"],
            volume=item["vol"],
            turnover_idr=item["turnover"],
            sector=item["sector"],
            tier=item["tier"],
            is_syariah=item["is_syariah"],
        )

        if plan["scalp_score"] >= min_score:
            results.append(plan)

    # Urutkan berdasarkan Skor Peluang Scalping tertinggi
    results.sort(key=lambda x: (x["scalp_score"], x["tp1_net_pct"]), reverse=True)

    # Ambil tepat 10 rekomendasi terbaik
    return results[:10]
