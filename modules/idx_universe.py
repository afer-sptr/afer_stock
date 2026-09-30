"""
idx_universe.py
===============
Modul Pengelola Semesta Seluruh Emiten Terdaftar di Bursa Efek Indonesia (BEI / IDX)
Menyediakan:
  1. Katalog lengkap 900+ emiten terdaftar (kode, nama, sektor, papan pencatatan)
  2. Klasifikasi Tingkatan Emiten Presisi Tinggi berbasis harga pasar nominal riil:
     - Saham Gocap / Saham Tidur (Rp50 – Rp100) [GOCAP]
     - Saham Receh / Saham Murah (Rp100 – Rp1.000) [RECEH]
     - Saham Premium / Blue Chip (Di atas Rp5.000) [PREMIUM]
  3. Klasifikasi Syariah Resmi Kriteria OJK / DSN-MUI (Daftar Efek Syariah / ISSI) vs Non-Syariah
  4. Fungsi filter & pencarian multi-kriteria untuk screener, bot dispatcher, dan UI
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

# Pilihan Opsi Filter Tingkatan (Tier)
TIER_OPTIONS: List[str] = [
    "Semua Tingkatan",
    "Saham Gocap / Saham Tidur (Rp50 – Rp100)",
    "Saham Receh / Saham Murah (Rp100 – Rp1.000)",
    "Saham Premium / Blue Chip (Di atas Rp5.000)",
]

# Saham Induk Blue Chip & LQ45 Institusional BEI (Lapis 1 / Core Index Anchors)
BLUE_CHIP_TICKERS: Set[str] = {
    "BBCA", "BMRI", "BBNI", "BBRI", "ASII", "TLKM", "UNTR", "ITMG", "ICBP", "INDF",
    "BYAN", "CPIN", "AMMN", "BREN", "TPIA", "DSSA", "ADRO", "PTBA", "BRIS", "BRPT",
    "PGAS", "EXCL", "ISAT", "MDKA", "INKP", "TKIM", "JSMR", "SMGR", "MEDC", "ANTM",
    "GEMS", "MBAP", "STTP", "AALI", "ABMM", "TCPI", "BDMN", "BNGA", "NISP", "MEGA"
}

# Sektor & Emiten Non-Syariah Berdasarkan Kriteria Resmi OJK / DSN-MUI:
# 1. Bank Konvensional
CONVENTIONAL_BANKS: Set[str] = {
    "BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "BDMN", "BNGA", "BNII", "BBKP", "BTPN",
    "NOBU", "BABP", "MAYA", "AGRO", "BCIC", "BVIC", "INPC", "NISP", "MEGA", "PNBN",
    "BJBR", "BJTM", "BSIM", "BACA", "BGTG", "BINA", "DNAR", "MASB", "AMAR", "BKSW",
    "MCOR", "BEKS", "BAPO", "BBYB", "ARTO", "AMOR", "BNLI", "BCAP"
}

# 2. Asuransi Konvensional
CONVENTIONAL_INSURANCE: Set[str] = {
    "ASRM", "AMAG", "ASDM", "LPGI", "MREI", "PNIN", "ASJT", "AHAP", "ASBI",
    "LIFE", "MTWI", "VINS", "ABDA"
}

# 3. Multifinance & Leasing Konvensional
CONVENTIONAL_FINANCE: Set[str] = {
    "BFIN", "ADMF", "CFIN", "MFIN", "WOMF", "TRUS", "VRNA", "BBLD", "TIFA",
    "HDFA", "BPFI", "IMFI", "FINN", "DEFI", "VTNY", "POLA", "LPPS", "SMMA"
}

# 4. Sekuritas & Holding Investasi Konvensional
CONVENTIONAL_SECURITIES: Set[str] = {
    "PANS", "TRIM", "YULE", "APIC", "KREN", "RELI", "KBLM", "BHIT", "SRTG"
}

# 5. Produsen Rokok & Tembakau
TOBACCO_TICKERS: Set[str] = {
    "HMSP", "GGRM", "WIIM", "ITIC", "RMBA"
}

# 6. Minuman Beralkohol
ALCOHOL_TICKERS: Set[str] = {
    "MLBI", "DLTA", "WINE", "BEER", "STRK"
}

# Bank & Institusi Keuangan Syariah Resmi OJK (Daftar Efek Syariah)
ISLAMIC_FINANCIAL_TICKERS: Set[str] = {
    "BRIS", "BTPS", "BANK", "PNBS", "JMAS"
}

# Seluruh Ticker Non-Syariah Tergabung
NON_SHARIA_TICKERS: Set[str] = (
    CONVENTIONAL_BANKS |
    CONVENTIONAL_INSURANCE |
    CONVENTIONAL_FINANCE |
    CONVENTIONAL_SECURITIES |
    TOBACCO_TICKERS |
    ALCOHOL_TICKERS
)

_DATA_PATH = os.path.join(os.path.dirname(__file__), "idx_stocks.json")
_PRICES_PATH = os.path.join(os.path.dirname(__file__), "idx_prices.json")


def is_sharia_compliant(ticker: str, sector: str = "", name: str = "") -> bool:
    """
    Validasi Kepatuhan Syariah OJK / DSN-MUI secara ketat & permanen.
    - Semua bank/keuangan konvensional, asuransi konvensional, multifinance, sekuritas: Non-Syariah
    - Semua produsen rokok/tembakau (HMSP, GGRM, WIIM, ITIC, RMBA): Non-Syariah
    - Semua produsen minuman beralkohol (MLBI, DLTA, WINE, BEER, STRK): Non-Syariah
    - Emiten keuangan yang berprinsip Syariah (BRIS, BTPS, BANK, PNBS, JMAS): Syariah
    """
    clean = ticker.replace(".JK", "").upper().strip()
    if clean in ISLAMIC_FINANCIAL_TICKERS:
        return True
    if clean in NON_SHARIA_TICKERS:
        return False
    if sector.strip().lower() in {"financials", "keuangan"}:
        return False

    name_lower = name.lower()
    for kw in ["bank", "asuransi", "insurance", "finance", "multifinance", "securities", "sekuritas", "brewery", "beer", "wine", "tobacco", "rokok", "tembakau"]:
        if kw in name_lower:
            if any(sharia_kw in name_lower for sharia_kw in ["syariah", "sharia", "aladin"]):
                return True
            return False

    return True


@lru_cache(maxsize=1)
def load_idx_prices() -> Dict[str, float]:
    """Memuat database harga pasar penutupan riil seluruh saham BEI."""
    if os.path.exists(_PRICES_PATH):
        try:
            with open(_PRICES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def classify_tier_by_price(price: float) -> Tuple[str, str, str]:
    """Mengklasifikasikan tier secara presisi berdasarkan nilai nominal riil."""
    if price > 5000.0:
        return "Saham Premium / Blue Chip (Di atas Rp5.000)", "PREMIUM", "Premium >5k"
    elif price <= 100.0:
        return "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "GOCAP", "Gocap 50-100"
    elif 100.0 < price <= 1000.0:
        return "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "RECEH", "Receh 100-1k"
    else:
        return "Saham Menengah (Rp1.000 – Rp5.000)", "MENENGAH", "Menengah 1k-5k"


def classify_stock_tier(
    ticker: str,
    price: Optional[float] = None,
    board: str = "Utama"
) -> Tuple[str, str, str]:
    """
    Mengklasifikasikan saham secara presisi ke dalam Tingkatan (Tier) resmi:
    1. Saham Gocap / Saham Tidur (Rp50 – Rp100) [GOCAP]
    2. Saham Receh / Saham Murah (Rp100 – Rp1.000) [RECEH]
    3. Saham Menengah (Rp1.000 – Rp5.000) [MENENGAH]
    4. Saham Premium / Blue Chip (Di atas Rp5.000) [PREMIUM]
    """
    clean = ticker.upper().strip()
    
    # Ambil harga dari parameter atau lookup dari database harga riil
    current_p = price
    if current_p is None:
        prices = load_idx_prices()
        current_p = prices.get(clean)

    if current_p is not None and current_p > 0:
        return classify_tier_by_price(current_p)

    # Fallback jika harga belum tercatat di database harga
    if board in {"Pemantauan Khusus", "Akselerasi"}:
        return "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "GOCAP", "Gocap 50-100"
    elif clean in BLUE_CHIP_TICKERS:
        return "Saham Premium / Blue Chip (Di atas Rp5.000)", "PREMIUM", "Blue Chip"
    else:
        return "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "RECEH", "Receh 100-1k"


@lru_cache(maxsize=1)
def load_raw_idx_stocks() -> List[Dict[str, Any]]:
    """Memuat data JSON mentah seluruh saham BEI."""
    if not os.path.exists(_DATA_PATH):
        return [
            {"code": "BBCA.JK", "ticker": "BBCA", "name": "Bank Central Asia Tbk.", "sector": "Financials", "board": "Utama"},
            {"code": "BBRI.JK", "ticker": "BBRI", "name": "Bank Rakyat Indonesia Tbk.", "sector": "Financials", "board": "Utama"},
            {"code": "BMRI.JK", "ticker": "BMRI", "name": "Bank Mandiri Tbk.", "sector": "Financials", "board": "Utama"},
            {"code": "TLKM.JK", "ticker": "TLKM", "name": "Telkom Indonesia Tbk.", "sector": "Infrastructure", "board": "Utama"},
            {"code": "ASII.JK", "ticker": "ASII", "name": "Astra International Tbk.", "sector": "Industrials", "board": "Utama"},
            {"code": "ICBP.JK", "ticker": "ICBP", "name": "Indofood CBP Sukses Makmur Tbk.", "sector": "Consumer Non-Cyclicals", "board": "Utama"},
            {"code": "BRMS.JK", "ticker": "BRMS", "name": "Bumi Resources Minerals Tbk.", "sector": "Basic Materials", "board": "Pengembangan"},
            {"code": "BUMI.JK", "ticker": "BUMI", "name": "Bumi Resources Tbk.", "sector": "Energy", "board": "Pengembangan"},
        ]
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_all_idx_stocks_enriched() -> List[Dict[str, Any]]:
    """
    Mengembalikan seluruh data saham BEI yang telah diperkaya dengan:
    - Tier Riil (Saham Gocap / Tidur, Saham Receh / Murah, Saham Premium / Blue Chip)
    - Status Syariah Akurat OJK (☪️ Syariah ISSI vs ⚪ Non-Syariah)
    - Harga Terakhir Tercatat
    """
    raw = load_raw_idx_stocks()
    prices = load_idx_prices()
    enriched = []

    for item in raw:
        ticker = item.get("ticker", "").upper().strip()
        code = item.get("code") or f"{ticker}.JK"
        name = item.get("name", "")
        sector = item.get("sector", "Lainnya")
        board = item.get("board", "Utama")
        price = prices.get(ticker, 0.0)

        # 1. Klasifikasi Tingkatan (Tier) berbasis harga pasar riil
        tier, tier_code, tier_short = classify_stock_tier(ticker, price=price, board=board)

        # 2. Klasifikasi Syariah Presisi OJK / DSN-MUI
        is_syariah = is_sharia_compliant(ticker, sector=sector, name=name)
        syariah_label = "☪️ Syariah (ISSI)" if is_syariah else "⚪ Non-Syariah"

        enriched.append({
            "code": code,
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "board": board,
            "price": price,
            "tier": tier,
            "tier_code": tier_code,
            "tier_short": tier_short,
            "is_syariah": is_syariah,
            "syariah_label": syariah_label,
            "display_label": f"{ticker} - {name[:22]} [{tier_short} | {'Syariah' if is_syariah else 'Non-Syariah'}]",
        })

    return enriched


def filter_idx_stocks(
    tier_filter: str = "Semua Tingkatan",
    syariah_filter: str = "Semua",
    sector_filter: str = "Semua",
    search_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Filter data seluruh saham Indonesia berdasarkan Tier, Syariah, Sektor, dan Pencarian.
    """
    stocks = get_all_idx_stocks_enriched()
    filtered = []

    for s in stocks:
        # Filter Tier
        if tier_filter not in {"Semua", "Semua Tingkatan"}:
            if "Gocap" in tier_filter or "Tidur" in tier_filter or "Rp50" in tier_filter:
                if s["tier_code"] != "GOCAP":
                    continue
            elif "Receh" in tier_filter or "Murah" in tier_filter or "Rp100 – Rp1.000" in tier_filter:
                if s["tier_code"] != "RECEH":
                    continue
            elif "Menengah" in tier_filter or "Rp1.000 – Rp5.000" in tier_filter:
                if s["tier_code"] != "MENENGAH":
                    continue
            elif "Premium" in tier_filter or "Blue Chip" in tier_filter or "5.000" in tier_filter:
                if s["tier_code"] != "PREMIUM":
                    continue
            elif "Lapis 1" in tier_filter:
                if s["tier_code"] != "PREMIUM":
                    continue
            elif "Lapis 2" in tier_filter:
                if s["tier_code"] not in {"MENENGAH", "RECEH"}:
                    continue
            elif "Lapis 3" in tier_filter:
                if s["tier_code"] != "GOCAP":
                    continue

        # Filter Syariah
        if syariah_filter != "Semua":
            if "Non-Syariah" in syariah_filter:
                if s["is_syariah"]:
                    continue
            elif "Syariah" in syariah_filter:
                if not s["is_syariah"]:
                    continue

        # Filter Sektor
        if sector_filter != "Semua" and s["sector"].lower() != sector_filter.lower():
            continue

        # Filter Search Query
        if search_query:
            q = search_query.upper().strip()
            if q not in s["ticker"] and q not in s["name"].upper():
                continue

        filtered.append(s)

    return filtered


def get_stock_metadata(ticker_or_code: str) -> Dict[str, Any]:
    """Mengambil metadata tingkatan emiten dan status syariah untuk satu saham."""
    clean_ticker = ticker_or_code.replace(".JK", "").upper().strip()
    stocks = get_all_idx_stocks_enriched()
    for s in stocks:
        if s["ticker"] == clean_ticker:
            return s

    # Fallback jika emiten baru / belum ada di katalog
    prices = load_idx_prices()
    p = prices.get(clean_ticker, 0.0)
    tier, tier_code, tier_short = classify_stock_tier(clean_ticker, price=p)
    is_syariah = is_sharia_compliant(clean_ticker)

    return {
        "code": f"{clean_ticker}.JK",
        "ticker": clean_ticker,
        "name": clean_ticker,
        "sector": "Umum",
        "board": "Utama",
        "price": p,
        "tier": tier,
        "tier_code": tier_code,
        "tier_short": tier_short,
        "is_syariah": is_syariah,
        "syariah_label": "☪️ Syariah (ISSI)" if is_syariah else "⚪ Non-Syariah",
        "display_label": f"{clean_ticker} [{tier_short} | {'Syariah' if is_syariah else 'Non-Syariah'}]",
    }


def get_all_sectors() -> List[str]:
    """Mengambil daftar seluruh sektor unik emiten di BEI."""
    stocks = get_all_idx_stocks_enriched()
    sectors = sorted(list(set(s["sector"] for s in stocks if s.get("sector"))))
    return ["Semua"] + sectors


get_all_idx_stocks = get_all_idx_stocks_enriched
TIER_1_TICKERS = BLUE_CHIP_TICKERS
PREMIUM_BLUE_CHIP_TICKERS = {s["ticker"] for s in get_all_idx_stocks_enriched() if s["tier_code"] == "PREMIUM"}
MENENGAH_TICKERS = {s["ticker"] for s in get_all_idx_stocks_enriched() if s["tier_code"] == "MENENGAH"}
TIER_2_TICKERS = {s["ticker"] for s in get_all_idx_stocks_enriched() if s["tier_code"] in {"RECEH", "MENENGAH"}}
RECEH_MURAH_TICKERS = {s["ticker"] for s in get_all_idx_stocks_enriched() if s["tier_code"] == "RECEH"}
TIER_3_TICKERS = {s["ticker"] for s in get_all_idx_stocks_enriched() if s["tier_code"] == "GOCAP"}
GOCAP_TIDUR_TICKERS = TIER_3_TICKERS
