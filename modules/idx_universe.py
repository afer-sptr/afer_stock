"""
idx_universe.py
===============
Modul Pengelola Semesta Seluruh Emiten Terdaftar di Bursa Efek Indonesia (BEI / IDX)
Menyediakan:
  1. Katalog lengkap 900+ emiten terdaftar (kode, nama, sektor, papan pencatatan)
  2. Klasifikasi Tingkatan Emiten (Lapis 1 / Blue-Chip, Lapis 2 / Mid-Cap, Lapis 3 / Small-Cap)
  3. Klasifikasi Syariah (Daftar Efek Syariah / ISSI / JII) vs Non-Syariah
  4. Fungsi filter & pencarian multi-kriteria untuk screener dan bot dispatcher
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

# Ticker Acuan Saham Premium / Blue Chip (Harga pasar nominal > Rp 5.000)
PREMIUM_BLUE_CHIP_TICKERS: Set[str] = {
    "BBCA", "BMRI", "UNTR", "ITMG", "ICBP", "INDF", "GGRM", "SMMA", "BYAN", "CPIN",
    "AMMN", "BREN", "TPIA", "DSSA", "BDMN", "MLBI", "TCPI", "INKP", "TKIM", "ASII",
    "KLBF", "ADRO", "EXCL", "ISAT", "MIKA", "HEAL", "CMRY", "STTP", "AALI", "ABMM",
    "GEMS", "MBAP", "PTBA", "BBNI", "BBRI", "BRIS", "MEGA", "NISP", "BNGA"
}

# Ticker Acuan Saham Gocap / Saham Tidur (Rp 50 – Rp 100)
GOCAP_TIDUR_TICKERS: Set[str] = {
    "GOTO", "POLA", "GIAA", "BKSL", "LPKR", "ZATA", "MLPL", "SLIS", "JAST", "IKAN",
    "GEMA", "VRNA", "MCOR", "BVIC", "LPPS", "CPRO", "HDFA", "BABP", "BIPI", "ZINC",
    "ENRG", "BUMI", "DOID", "DEWA", "TRIM", "MDIA", "BAPA", "NASA", "KBAG", "CARE",
    "COAL", "ELTY", "LUCK", "MABA", "PPRO", "PURE", "REAL", "WAPO", "WIFI", "ZBRA",
    "BBYB", "BBKP", "KPIG", "BHIT", "BCAP", "OASA", "TOOL", "SBMA", "BEBS"
}

# Ticker Acuan Saham Receh / Saham Murah (Rp 100 – Rp 1.000)
RECEH_MURAH_TICKERS: Set[str] = {
    "BRMS", "KIJA", "ELSA", "RAJA", "PSAB", "MBMA", "SSIA", "ARTO", "ACES", "MAPA",
    "BUKA", "EMTK", "WIKA", "ADHI", "PTPP", "TOTL", "NRCA", "SIDO", "CLEO", "AUTO",
    "DRMA", "SMSM", "PWON", "CTRA", "BSDE", "SMRA", "ERAA", "NCKL", "VKTR", "CUAN",
    "PGEO", "SILO", "SSMS", "DSNG", "BTPS", "JPFA", "MAIN", "WOOD", "MARK", "CITA",
    "HRUM", "INDY", "TAPG", "ULTJ", "AVIA", "ARNA", "ESSA", "SCMA", "MNCN", "BMTR",
    "PGAS", "ANTM", "MEDC", "MAPI", "BBTN", "MDKA", "SMGR", "AMRT", "MYOR", "BRPT"
}

# Backward compatibility alias
TIER_1_TICKERS = PREMIUM_BLUE_CHIP_TICKERS
TIER_2_TICKERS = RECEH_MURAH_TICKERS
TIER_3_TICKERS = GOCAP_TIDUR_TICKERS

# Sektor & Emiten Non-Syariah Berdasarkan Kriteria OJK / DSN-MUI:
NON_SHARIA_TICKERS: Set[str] = {
    "BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "BDMN", "BNGA", "BNII", "BBKP", "BTPN",
    "NOBU", "BABP", "MAYA", "AGRO", "BCIC", "BVIC", "INPC", "NISP", "MEGA", "PNBN",
    "BJBR", "BJTM", "BSIM", "BACA", "BGTG", "BINA", "DNAR", "MASB", "AMAR", "BKSW",
    "MCOR", "BEKS", "BAPO", "BBYB", "ARTO", "GGRM", "HMSP", "WIIM", "ITIC", "MLBI",
    "DLTA", "BFIN", "ADMF", "CFIN", "MFIN", "WOMF", "TRUS", "ASRM", "AMAG", "ASDM",
    "LPGI", "MREI", "PNIN", "PANS", "TRIM", "YULE", "HDFA", "VRNA", "BBLD", "TIFA"
}

_DATA_PATH = os.path.join(os.path.dirname(__file__), "idx_stocks.json")


def classify_stock_tier(ticker: str, board: str = "Utama") -> Tuple[str, str, str]:
    """
    Mengklasifikasikan saham ke dalam salah satu dari 3 Tingkatan (Tier) resmi:
    1. Saham Gocap / Saham Tidur (Rp50 – Rp100) [GOCAP]
    2. Saham Receh / Saham Murah (Rp100 – Rp1.000) [RECEH]
    3. Saham Premium / Blue Chip (Di atas Rp5.000) [PREMIUM]
    """
    clean = ticker.upper().strip()
    if clean in PREMIUM_BLUE_CHIP_TICKERS:
        return "Saham Premium / Blue Chip (Di atas Rp5.000)", "PREMIUM", "Premium >5k"
    elif clean in GOCAP_TIDUR_TICKERS or board in {"Pemantauan Khusus", "Akselerasi"}:
        return "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "GOCAP", "Gocap 50-100"
    else:
        return "Saham Receh / Saham Murah (Rp100 – Rp1.000)", "RECEH", "Receh 100-1k"


def classify_tier_by_price(price: float) -> Tuple[str, str, str]:
    """Mengklasifikasikan tier berdasarkan harga nominal riil saat ini."""
    if price > 5000.0:
        return "Saham Premium / Blue Chip (Di atas Rp5.000)", "PREMIUM", "Premium >5k"
    elif 50.0 <= price <= 100.0:
        return "Saham Gocap / Saham Tidur (Rp50 – Rp100)", "GOCAP", "Gocap 50-100"
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
    - Tier (Saham Gocap / Tidur, Saham Receh / Murah, Saham Premium / Blue Chip)
    - Status Syariah (☪️ Syariah vs ⚪ Non-Syariah)
    """
    raw = load_raw_idx_stocks()
    enriched = []

    for item in raw:
        ticker = item.get("ticker", "").upper()
        code = item.get("code") or f"{ticker}.JK"
        name = item.get("name", "")
        sector = item.get("sector", "Lainnya")
        board = item.get("board", "Utama")

        # 1. Klasifikasi Tingkatan (Tier)
        tier, tier_code, tier_short = classify_stock_tier(ticker, board=board)

        # 2. Klasifikasi Syariah vs Non-Syariah
        is_syariah = ticker not in NON_SHARIA_TICKERS
        syariah_label = "☪️ Syariah (ISSI)" if is_syariah else "⚪ Non-Syariah"

        enriched.append({
            "code": code,
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "board": board,
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
            if "Gocap" in tier_filter or "Tidur" in tier_filter:
                if s["tier_code"] != "GOCAP":
                    continue
            elif "Receh" in tier_filter or "Murah" in tier_filter:
                if s["tier_code"] != "RECEH":
                    continue
            elif "Premium" in tier_filter or "Blue Chip" in tier_filter:
                if s["tier_code"] != "PREMIUM":
                    continue
            elif "Lapis 1" in tier_filter and s["tier_code"] != "PREMIUM":
                continue
            elif "Lapis 2" in tier_filter and s["tier_code"] != "RECEH":
                continue
            elif "Lapis 3" in tier_filter and s["tier_code"] != "GOCAP":
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
    is_syariah = clean_ticker not in NON_SHARIA_TICKERS
    tier, tier_code, tier_short = classify_stock_tier(clean_ticker)
    return {
        "code": f"{clean_ticker}.JK",
        "ticker": clean_ticker,
        "name": clean_ticker,
        "sector": "Umum",
        "board": "Utama",
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
