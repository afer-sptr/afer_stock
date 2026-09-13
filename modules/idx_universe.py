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
from typing import Any, Dict, List, Optional, Set

# Daftar Ticker Acuan Lapis 1 (Blue-Chip / LQ45 / IDX30 / Kapitalisasi Besar > Rp 50T)
TIER_1_TICKERS: Set[str] = {
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "ICBP", "INDF", "UNVR", "KLBF",
    "AMRT", "PGAS", "ADRO", "CPIN", "INCO", "ANTM", "MDKA", "PTBA", "BRIS", "SMGR",
    "GOTO", "BREN", "TPIA", "AMMN", "BFIN", "BBTN", "INKP", "TKIM", "UNTR", "GGRM",
    "HMSP", "TBIG", "TOWR", "MEDC", "AKRA", "MYOR", "BRPT", "ACES", "MAPI", "MAPA"
}

# Daftar Ticker Acuan Lapis 2 (Mid-Cap Growth / IDX80 / Kapitalisasi Rp 5T - 50T)
TIER_2_TICKERS: Set[str] = {
    "CTRA", "BSDE", "PWON", "SMRA", "MTEL", "EXCL", "ISAT", "HEAL", "MIKA", "ERAA",
    "ENRG", "RAJA", "PSAB", "MBMA", "NCKL", "VKTR", "CUAN", "PGEO", "SILO", "SSMS",
    "DSNG", "ARTO", "BBYB", "BTPS", "BDMN", "BNGA", "BNII", "SIDO", "JPFA", "MAIN",
    "WOOD", "MARK", "CITA", "HRUM", "INDY", "DOID", "ABMM", "AALI", "LSIP", "TAPG",
    "CLEO", "CMRY", "ULTJ", "STTP", "AUTO", "DRMA", "SMSM", "PTPP", "WIKA", "ADHI",
    "TOTL", "NRCA", "AVIA", "ARNA", "ESSA", "BUKA", "EMTK", "SCMA", "MNCN", "BMTR"
}

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
    - Tier (Lapis 1, Lapis 2, Lapis 3)
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

        # 1. Klasifikasi Tingkatan (Tier / Lapis)
        if ticker in TIER_1_TICKERS:
            tier = "Lapis 1 (Blue-Chip)"
            tier_code = "L1"
        elif ticker in TIER_2_TICKERS:
            tier = "Lapis 2 (Mid-Cap)"
            tier_code = "L2"
        else:
            tier = "Lapis 3 (Small-Cap)"
            tier_code = "L3"

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
            "is_syariah": is_syariah,
            "syariah_label": syariah_label,
            "display_label": f"{ticker} - {name[:24]} [{tier_code} | {'Syariah' if is_syariah else 'Non-Syariah'}]",
        })

    return enriched


def filter_idx_stocks(
    tier_filter: str = "Semua",
    syariah_filter: str = "Semua",
    sector_filter: str = "Semua",
    search_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Filter data seluruh saham Indonesia berdasarkan Lapis, Syariah, Sektor, dan Pencarian.
    """
    stocks = get_all_idx_stocks_enriched()
    filtered = []

    for s in stocks:
        # Filter Tier
        if tier_filter != "Semua":
            if "Lapis 1" in tier_filter and s["tier_code"] != "L1":
                continue
            elif "Lapis 2" in tier_filter and s["tier_code"] != "L2":
                continue
            elif "Lapis 3" in tier_filter and s["tier_code"] != "L3":
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
    tier = "Lapis 1 (Blue-Chip)" if clean_ticker in TIER_1_TICKERS else ("Lapis 2 (Mid-Cap)" if clean_ticker in TIER_2_TICKERS else "Lapis 3 (Small-Cap)")
    return {
        "code": f"{clean_ticker}.JK",
        "ticker": clean_ticker,
        "name": clean_ticker,
        "sector": "Umum",
        "board": "Utama",
        "tier": tier,
        "tier_code": "L1" if clean_ticker in TIER_1_TICKERS else ("L2" if clean_ticker in TIER_2_TICKERS else "L3"),
        "is_syariah": is_syariah,
        "syariah_label": "☪️ Syariah (ISSI)" if is_syariah else "⚪ Non-Syariah",
        "display_label": f"{clean_ticker} [{tier[:2]} | {'Syariah' if is_syariah else 'Non-Syariah'}]",
    }


def get_all_sectors() -> List[str]:
    """Mengambil daftar seluruh sektor unik emiten di BEI."""
    stocks = get_all_idx_stocks_enriched()
    sectors = sorted(list(set(s["sector"] for s in stocks if s.get("sector"))))
    return ["Semua"] + sectors


get_all_idx_stocks = get_all_idx_stocks_enriched
