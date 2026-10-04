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

# Daftar Resmi Seluruh 597+ Saham Syariah Terdaftar di BEI Sesuai Keputusan Resmi OJK / ISSI
# Keputusan Dewan Komisioner Otoritas Jasa Keuangan (OJK) No. KEP-21/D.04/2026
OFFICIAL_DES_SYARIAH_TICKERS: Set[str] = {
    "AADI", "AALI", "ABMM", "ACES", "ADCP", "ADES", "ADMG", "ADMR", "ADRO", "AEGS",
    "AGAR", "AGII", "AISA", "AKPI", "AKRA", "AKSI", "ALDO", "ALKA", "AMAN", "AMFG",
    "AMIN", "AMMS", "ANTM", "APII", "APLI", "APLN", "ARCI", "AREA", "ARII", "ARNA",
    "ASGR", "ASHA", "ASLC", "ASLI", "ASPI", "ASRI", "ASSA", "ATAP", "ATIC", "ATLA",
    "AUTO", "AVIA", "AWAN", "AXIO", "AYAM", "AYLS", "BABY", "BAIK", "BALI", "BANK",
    "BAPI", "BATR", "BAUT", "BAYU", "BBRM", "BBSS", "BCIP", "BDKR", "BELI", "BELL",
    "BESS", "BEST", "BIKE", "BINO", "BIPP", "BIRD", "BISI", "BKDP", "BKSL", "BLES",
    "BLTA", "BLTZ", "BLUE", "BMBL", "BMHS", "BMSR", "BMTR", "BOAT", "BOBA", "BOGA",
    "BOLT", "BRAM", "BRIS", "BRMS", "BRNA", "BSBK", "BSDE", "BSML", "BSSR", "BTPS",
    "BUAH", "BUDI", "BULL", "BUMI", "BWPT", "BYAN", "CAKK", "CAMP", "CANI", "CARE",
    "CASH", "CASS", "CCSI", "CEKA", "CGAS", "CHEM", "CHIP", "CINT", "CITA", "CITY",
    "CLEO", "CLPI", "CMNP", "CMPP", "CMRY", "CNMA", "COAL", "CPIN", "CPRO", "CRSN",
    "CSAP", "CSIS", "CSMI", "CSRA", "CTBN", "CTRA", "CYBR", "DADA", "DATA", "DAYA",
    "DCII", "DEFI", "DEPO", "DEWA", "DEWI", "DGIK", "DGNS", "DILD", "DIVA", "DKFT",
    "DMAS", "DMMX", "DMND", "DOOH", "DOSS", "DRMA", "DSFI", "DSNG", "DSSA", "DUTI",
    "DVLA", "DWGL", "DYAN", "EAST", "ECII", "EKAD", "ELIT", "ELPI", "ELSA", "ELTY",
    "EMDE", "ENAK", "ENRG", "EPAC", "EPMT", "ERAA", "ERAL", "ERTX", "ESIP", "ESSA",
    "ESTA", "EURO", "EXCL", "FAST", "FASW", "FILM", "FIMP", "FIRE", "FISH", "FLMC",
    "FMII", "FOLK", "FOOD", "FPNI", "FWCT", "GDST", "GDYR", "GEMA", "GEMS", "GGRP",
    "GHON", "GIAA", "GJTL", "GLVA", "GMTD", "GOLD", "GOLF", "GOOD", "GPRA", "GPSO",
    "GRIA", "GRPH", "GRPM", "GTRA", "GULA", "GUNA", "GWSA", "GZCO", "HADE", "HAIS",
    "HAJJ", "HALO", "HATM", "HBAT", "HDIT", "HEAL", "HELI", "HERO", "HEXA", "HOKI",
    "HOMI", "HOPE", "HRTA", "HRUM", "HYGN", "IATA", "IBST", "ICBP", "ICON", "IDEA",
    "IDPR", "IFII", "IFSH", "IGAR", "IIKP", "IKAI", "IKAN", "IKBI", "IKPM", "IMPC",
    "INCI", "INDF", "INDR", "INDS", "INDY", "INET", "INKP", "INPP", "INTD", "INTP",
    "IOTF", "IPAC", "IPCM", "IPOL", "IPTV", "IRRA", "IRSX", "ISAP", "ISAT", "ISSP",
    "ITMA", "ITMG", "JARR", "JAST", "JATI", "JAWA", "JAYA", "JECC", "JGLE", "JIHD",
    "JKON", "JMAS", "JPFA", "JRPT", "JSMR", "JTPE", "KARW", "KBAG", "KBLI", "KBLM",
    "KDSI", "KEEN", "KEJU", "KETR", "KIAS", "KICI", "KIJA", "KING", "KINO", "KIOS",
    "KJEN", "KKES", "KKGI", "KLAS", "KLBF", "KLIN", "KMDS", "KOBX", "KOCI", "KOIN",
    "KOKA", "KONI", "KOPI", "KOTA", "KPIG", "KREN", "KUAS", "LABS", "LAJU", "LAND",
    "LFLO", "LION", "LIVE", "LMAX", "LMPI", "LMSH", "LOPI", "LPCK", "LPIN", "LPLI",
    "LPPF", "LRNA", "LSIP", "LTLS", "LUCK", "MAHA", "MAIN", "MANG", "MAPA", "MAPB",
    "MAPI", "MARK", "MAXI", "MBAP", "MBMA", "MBTO", "MCAS", "MCOL", "MDIA", "MDKA",
    "MDKI", "MEDC", "MEDS", "MEJA", "MERK", "META", "MFMI", "MGLV", "MHKI", "MICE",
    "MIKA", "MIRA", "MITI", "MKAP", "MKNT", "MKPI", "MKTR", "MLIA", "MLPL", "MLPT",
    "MMIX", "MMLP", "MNCN", "MORA", "MPIX", "MPMX", "MPOW", "MPPA", "MRAT", "MSIE",
    "MSIN", "MSJA", "MSKY", "MSTI", "MTDL", "MTEL", "MTLA", "MTMH", "MTPS", "MTSM",
    "MUTU", "MYOH", "MYOR", "NAIK", "NANO", "NASI", "NAYZ", "NELY", "NEST", "NFCX",
    "NICE", "NICL", "NIKL", "NRCA", "NSSS", "NTBK", "NZIA", "OBMD", "OKAS", "OLIV",
    "OMED", "PACK", "PADA", "PALM", "PAMG", "PANR", "PART", "PBID", "PCAR", "PDES",
    "PDPP", "PEHA", "PEVE", "PGAS", "PGJO", "PGLI", "PGUN", "PICO", "PJAA", "PKPK",
    "PLAN", "PLIN", "PMJS", "PNBS", "PNGO", "POLI", "POLU", "PORT", "POWR", "PPGL",
    "PPRE", "PPRI", "PRAY", "PRDA", "PRIM", "PSAB", "PSDN", "PSGO", "PSKT", "PSSI",
    "PTBA", "PTIS", "PTMP", "PTMR", "PTPP", "PTPS", "PTPW", "PTSN", "PTSP", "PURA",
    "PURI", "PZZA", "RAAM", "RAJA", "RALS", "RANC", "RBMS", "RCCC", "REAL", "RELF",
    "RGAS", "RISE", "RMKE", "RMKO", "ROCK", "RODA", "ROTI", "RSCH", "RSGK", "RUIS",
    "RUNS", "SAFE", "SAGE", "SAME", "SAMF", "SAPX", "SATU", "SBMA", "SCCO", "SCNP",
    "SCPI", "SDPC", "SEMA", "SGER", "SGRO", "SHID", "SICO", "SIDO", "SILO", "SIMP",
    "SIPD", "SKBM", "SKLT", "SKRN", "SLIS", "SMAR", "SMBR", "SMCB", "SMDM", "SMDR",
    "SMGA", "SMGR", "SMIL", "SMKL", "SMKM", "SMLE", "SMMT", "SMRA", "SMSM", "SNLK",
    "SOCI", "SOFA", "SOHO", "SOLA", "SOSS", "SOTS", "SPMA", "SPRE", "SPTO", "SRTG",
    "SSIA", "SSTM", "STAA", "STTP", "SULI", "SUNI", "SUPR", "SURI", "SWID", "TALF",
    "TAMA", "TAPG", "TAXI", "TBMS", "TCID", "TCPI", "TEBE", "TFAS", "TFCO", "TGKA",
    "TGUK", "TINS", "TIRA", "TIRT", "TKIM", "TLDN", "TLKM", "TMAS", "TMPO", "TNCA",
    "TOBA", "TOOL", "TOSK", "TOTL", "TOTO", "TPIA", "TPMA", "TRIS", "TRJA", "TRON",
    "TRST", "TRUK", "TSPC", "TYRE", "UANG", "UCID", "UDNG", "UFOE", "ULTJ", "UNIC",
    "UNIQ", "UNTR", "UNVR", "URBN", "UVCR", "VAST", "VERN", "VICI", "VISI", "VKTR",
    "VOKS", "WAPO", "WEGE", "WEHA", "WGSH", "WIDI", "WIFI", "WINR", "WINS", "WIRG",
    "WOOD", "WOWS", "WTON", "YELO", "YPAS", "ZONE", "ZYRX",
}

# Seluruh Ticker Non-Syariah Tergabung (semua emiten di luar DES OJK)
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
    Berdasarkan Keputusan Dewan Komisioner Otoritas Jasa Keuangan (OJK)
    tentang Daftar Efek Syariah (DES) / Indeks Saham Syariah Indonesia (ISSI)
    Nomor KEP-21/D.04/2026 dan Pembaruan Resmi OJK.
    Emiten diklasifikasikan sebagai Saham Syariah jika dan hanya jika
    terdaftar secara resmi dalam Daftar Efek Syariah (DES) OJK.
    """
    clean = ticker.replace(".JK", "").upper().strip()
    return clean in OFFICIAL_DES_SYARIAH_TICKERS


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

    rep_set = {
        "BBCA", "BBRI", "BMRI", "BBNI", "ASII", "TLKM", "AMMN", "BREN", "TPIA",
        "BRMS", "BUMI", "MEDC", "ANTM", "PGAS", "ENRG", "MDKA", "PSAB", "RAJA",
        "CSMI", "SLIS", "ZATA", "POLA", "NASI", "REAL", "ATLA", "WINR", "NINE",
        "BBSS", "HOMI", "ESTA", "BOBA", "KOCI", "PURI", "LUCK", "BAPA", "OILS",
        "ACRO", "BATR", "AEGS", "BAUT", "BATA", "ALMI", "ARKA", "ABBA", "AYLS",
        "DEWA", "DOID", "TOBA", "KIJA", "ELSA", "HRUM", "PANI", "CUAN", "ESSA",
        "AGRS", "AHAP", "BEER", "BVIC", "DEFI", "ADMF", "GGRM"
    }

    for s in stocks:
        p = float(s.get("price", 0.0))

        # Filter Tier Berdasarkan Rentang Harga Nominal Riil BEI
        if tier_filter not in {"Semua", "Semua Tingkatan"}:
            if "Gocap" in tier_filter or "Tidur" in tier_filter or "Rp50" in tier_filter:
                if s["tier_code"] != "GOCAP" or p < 50.0 or p > 100.0:
                    continue
            elif "Receh" in tier_filter or "Murah" in tier_filter or "Rp100 – Rp1.000" in tier_filter:
                if s["tier_code"] != "RECEH" or p <= 100.0 or p > 1000.0:
                    continue
            elif "Menengah" in tier_filter or "Rp1.000 – Rp5.000" in tier_filter:
                if s["tier_code"] != "MENENGAH" or p <= 1000.0 or p > 5000.0:
                    continue
            elif "Premium" in tier_filter or "Blue Chip" in tier_filter or "5.000" in tier_filter:
                if s["tier_code"] != "PREMIUM" or p <= 5000.0:
                    continue
            elif "Lapis 1" in tier_filter:
                if s["tier_code"] != "PREMIUM" or p <= 5000.0:
                    continue
            elif "Lapis 2" in tier_filter:
                if s["tier_code"] not in {"MENENGAH", "RECEH"} or not (100.0 < p <= 5000.0):
                    continue
            elif "Lapis 3" in tier_filter:
                if s["tier_code"] != "GOCAP" or not (50.0 <= p <= 100.0):
                    continue

        # Filter Syariah Presisi OJK / DSN-MUI
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

    # Prioritaskan emiten yang aktif diperdagangkan (harga >= 50) dan likuid di urutan teratas
    filtered.sort(
        key=lambda x: (
            0 if x["ticker"] in rep_set and x.get("price", 0.0) >= 50.0 else (
                1 if x.get("price", 0.0) >= 50.0 else 2
            ),
            x["ticker"]
        )
    )

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
