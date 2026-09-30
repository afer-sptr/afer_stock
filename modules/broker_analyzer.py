"""
broker_analyzer.py
==================
Modul Analisis Broker dan Emiten Bursa Efek Indonesia (BEI / IDX).
Menyediakan 6 Fitur Mikrostruktur Pasar & Deteksi Institusional:
1. Deteksi Akumulasi Institusi Senyap (Silent Institutional Accumulation pada Saham Stagnan)
2. Prediksi Arah Pergerakan Jangka Pendek (3 Pilar: Teknikal & Holt-Winters, NLP VADER, Order Flow)
3. Deteksi Indikasi 'Pump and Dump' (Harga, Volume, dan Sentimen Berita)
4. Analisis Order Book Level 2: Deteksi 'Spoofing dan Layering'
5. Deteksi 'Marking the Close' (Analisis Data Intraday 1 Menit & Pre-Closing)
6. Deteksi 'Wash Trading' (Analisis Broker Summary & Transaksi Putar)
"""

from __future__ import annotations

import math
from datetime import datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Coba import modul Holt-Winters dan VADER
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        _vader_analyzer = SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
        _vader_analyzer = SentimentIntensityAnalyzer()
    HAS_VADER = True
except Exception:
    _vader_analyzer = None
    HAS_VADER = False

# ==============================================================================
# DIREKTORI PROFIL, SIFAT & IMPLIKASI HARGA BROKER BEI
# ==============================================================================
IDX_BROKER_DIRECTORY: Dict[str, Dict[str, str]] = {
    "ZP": {
        "code": "ZP",
        "name": "PT Maybank Sekuritas Indonesia",
        "category": "🏛️ Asing & Institusi Global",
        "archetype": "Smart Money / Akumulasi Senyap",
        "behavior": "Sering mengeksekusi akumulasi bertahap dalam senyap (silent accumulation) dengan volume masif tanpa menimbulkan lonjakan harga di pasar reguler. Berorientasi pada valuasi jangka menengah hingga panjang.",
        "future_price_impact": "🟢 BULLISH JANGKA MENENGAH: Jika ZP konsisten Net Buy saat harga sideways, menandakan fondasi kuat menuju markup harga berkelanjutan."
    },
    "BK": {
        "code": "BK",
        "name": "PT J.P. Morgan Sekuritas Indonesia",
        "category": "🏛️ Smart Money Asing (Tier 1 Global)",
        "archetype": "Global Big Fund & Sovereign Flow",
        "behavior": "Menyalurkan aliran dana manajer investasi raksasa dan sovereign/hedge fund global. Sangat selektif pada saham-saham likuid berkapitalisasi besar (Blue Chip) dengan fundamental prima.",
        "future_price_impact": "🟢 TREN NAIK BERKELANJUTAN: Net Buy konsisten dari BK menandakan arus dana asing masuk (foreign inflow) yang solid; tren harga cenderung stabil naik dan tahan terhadap koreksi sesaat."
    },
    "CS": {
        "code": "CS",
        "name": "PT Credit Suisse Sekuritas Indonesia",
        "category": "🏛️ Smart Money Asing (Tier 1)",
        "archetype": "Institutional Portfolio Rebalancing",
        "behavior": "Eksekutor institusi global yang sangat disiplin terhadap level target harga dan manajemen risiko portofolio internasional.",
        "future_price_impact": "🟢 SINYAL AWAL RALLY: Akumulasi masif CS sering menjadi pemicu rally harga saham lapis 1 dan lapis 2 di bursa."
    },
    "MS": {
        "code": "MS",
        "name": "PT Morgan Stanley Sekuritas Indonesia",
        "category": "🏛️ Smart Money Asing (Tier 1 Global)",
        "archetype": "MSCI Index & Macro Rebalancer",
        "behavior": "Kerap menjadi eksekutor utama penyesuaian bobot saham dalam indeks global (MSCI/FTSE). Transaksinya sering terjadi pada sesi crossing atau menit-menit akhir bursa.",
        "future_price_impact": "🟢 VOLATILITAS BESAR & INFLOW: Net Buy masif MS menjelang penyesuaian indeks menjamin lonjakan likuiditas dan kenaikan target harga di masa depan."
    },
    "AK": {
        "code": "AK",
        "name": "PT UBS Sekuritas Indonesia",
        "category": "🏛️ Institusi Global & Private Wealth",
        "archetype": "Discreet Institutional Accumulator",
        "behavior": "Akumulasi terukur tanpa memicu kepanikan beli di pasar. Membeli di area support dan menahan posisi untuk jangka panjang.",
        "future_price_impact": "🟢 PROYEKSI HARGA NAIK: Kehadiran AK di pucuk Net Buy saat volume transaksi meningkat merupakan sinyal konfirmasi bahwa fase penurunan (downtrend) telah berakhir."
    },
    "RX": {
        "code": "RX",
        "name": "PT Macquarie Sekuritas Indonesia",
        "category": "🏛️ Institusi Asing & Market Maker Waran",
        "archetype": "Structured Warrant Issuer & Arbitrageur",
        "behavior": "Penerbit utama Waran Terstruktur di BEI dan penyedia likuiditas lindung nilai (hedging). Juga memfasilitasi transaksi block trade institusi asing.",
        "future_price_impact": "⚪ NETRAL / 🟢 AKUMULASI ASING: Jika transaksinya murni Net Buy tanpa posisi waran terstruktur, menjadi sinyal akumulasi saham induk."
    },
    "KZ": {
        "code": "KZ",
        "name": "PT CLSA Sekuritas Indonesia",
        "category": "🏛️ Riset Unggulan & Institusi Regional",
        "archetype": "Institutional Research Driven",
        "behavior": "Didukung divisi riset yang disegani di kawasan Asia-Pasifik. Transaksinya sering mendahului laporan riset positif institusi.",
        "future_price_impact": "🟢 SINYAL PERTUMBUHAN HARGA: Akumulasi KZ biasanya mengonfirmasi bahwa katalis pertumbuhan laba emiten akan segera direspons positif oleh pasar."
    },
    "CC": {
        "code": "CC",
        "name": "PT Mandiri Sekuritas",
        "category": "🏦 BUMN & Institusi Domestik / HNW Terbesar",
        "archetype": "National Flagship / Domestic Big Fund",
        "behavior": "Broker pelat merah terbesar. Menampung dana institusi domestik (BPJS Ketenagakerjaan, Taspen, Asuransi, Dana Pensiun) serta nasabah ritel mapan (High Net Worth).",
        "future_price_impact": "🟢 DUKUNGAN LANTAI HARGA (SUPPORT KUAT): Jika CC melakukan Net Buy masif pada saat pasar terkoreksi, ini menandakan intervensi pasar domestik (stabilisasi harga) yang mencegah harga turun lebih dalam."
    },
    "NI": {
        "code": "NI",
        "name": "PT BNI Sekuritas",
        "category": "🏦 BUMN & Institusi Lokal",
        "archetype": "Conservative Value & Dividend Seeker",
        "behavior": "Berfokus pada saham-saham defensif, perbankan, dan infrastruktur dengan hasil dividen stabil.",
        "future_price_impact": "🟢 STABILITAS & KENAIKAN BERTAHAP: Net Buy NI menunjukkan kepercayaan institusi lokal terhadap kinerja fundamental jangka panjang emiten."
    },
    "OD": {
        "code": "OD",
        "name": "PT BRI Danareksa Sekuritas",
        "category": "🏦 BUMN & Penasihat Keuangan Negara",
        "archetype": "State Underwriter & Corporate Action Specialist",
        "behavior": "Sangat aktif dalam penanganan IPO BUMN, restrukturisasi, rights issue, dan merger korporasi strategis.",
        "future_price_impact": "🟢 SIKLUS AKSI KORPORASI: Net Buy OD sering menjadi sinyal awal bahwa emiten akan menggelar aksi korporasi yang mendongkrak valuasi harga."
    },
    "DX": {
        "code": "DX",
        "name": "PT Bahana Sekuritas",
        "category": "🏦 BUMN (Holding IFG)",
        "archetype": "Sovereign Fund & Pension Inflow",
        "behavior": "Pengelola portofolio institusi negara dan asuransi BUMN. Mengutamakan kepatuhan tata kelola (GCG) dan jarang berspekulasi.",
        "future_price_impact": "🟢 SINYAL NILAI WAJAR (VALUE REVERSAL): Pembelian DX di harga murah menandakan valuasi saham telah terdiskon jauh di bawah nilai intrinsiknya."
    },
    "YP": {
        "code": "YP",
        "name": "PT Mirae Asset Sekuritas Indonesia",
        "category": "👥 Kerumunan Ritel (Retail Crowd / Scalper)",
        "archetype": "Retail Herd & Day Traders",
        "behavior": "Basis nasabah ritel dan scalper terbesar di Indonesia. Transaksinya sangat reaktif terhadap berita harian, pergerakan chart jangka pendek, dan rumor di grup trading.",
        "future_price_impact": "⚠️ KONTRARIAN (RESIKO DISTRIBUSI): Jika YP mendominasi Net Buy di harga pucuk (Ritel FOMO), harga saham sangat rawan diguyur (dumping) oleh bandar. Sebaliknya, jika YP panik Net Sell masif, sering kali menandakan dasar harga (bottoming) sebelum rebound."
    },
    "PD": {
        "code": "PD",
        "name": "PT Indo Premier Sekuritas",
        "category": "👥 Ritel Domestik Massal (IPOT)",
        "archetype": "Mainstream Retail Sentiment",
        "behavior": "Nasabah ritel mandiri dari platform IPOT. Cenderung mengejar saham yang sedang breakout atau saham yang masuk daftar top gainer harian.",
        "future_price_impact": "⚠️ RESIKO TRAP PUCUK: Net Buy besar PD di pucuk menandakan barang beralih dari bandar ke tangan ritel; harga berpotensi berbalik turun tajam jika momentum memudar."
    },
    "XC": {
        "code": "XC",
        "name": "PT Ajaib Sekuritas Asia",
        "category": "👥 Ritel Pemula & Gen-Z (Fast Money)",
        "archetype": "Novice Retail & Social Media Traders",
        "behavior": "Mayoritas investor ritel muda dan pemula. Memiliki holding period sangat singkat, menyukai saham lapis 3 / saham gorengan yang sedang viral di media sosial.",
        "future_price_impact": "⚡ VOLATILITAS TINGGI TANPA KETAHANAN: Mampu memicu lonjakan harga cepat sesaat, namun jika terjadi koreksi, kepanikan jual XC dapat menjatuhkan harga langsung ke batas ARB."
    },
    "MG": {
        "code": "MG",
        "name": "PT Semesta Indovest Sekuritas",
        "category": "⚡ Bandar Kilat / Proprietary Day Trading",
        "archetype": "Fast Market Maker & Momentum Igniter",
        "behavior": "Dikenal di komunitas bursa sebagai broker 'Bandar Kilat'. Sering melakukan HAKA masif di pagi hari untuk menciptakan lonjakan volume dan menarik ritel, lalu membanting harga (HAKI) di sesi siang atau keesokan harinya.",
        "future_price_impact": "🚨 SANGAT WASPADA (HANYA SCALPING CEPAT): Kenaikan harga akibat Net Buy MG umumnya bersifat sementara (1-2 hari). Jangan pernah menjadikan saham ini sebagai investasi jangka panjang saat MG aktif."
    },
    "AZ": {
        "code": "AZ",
        "name": "PT Sucor Sekuritas",
        "category": "🚀 Komunitas Ritel Aktif & Swing Traders",
        "archetype": "Aggressive Retail & HNW Community",
        "behavior": "Memiliki basis komunitas trader aktif yang dipandu analisis teknikal momentum. Kerap memicu tren breakout saham mid-cap.",
        "future_price_impact": "🟢 MOMENTUM RALLY 3-7 HARI: Net Buy AZ yang didukung volume tebal sering kali mengawali reli swing trading selama beberapa hari bursa ke depan."
    },
    "XL": {
        "code": "XL",
        "name": "PT Stockbit Sekuritas Digital",
        "category": "👥 Komunitas Investor & Swing Trader",
        "archetype": "Analytical Retail Community",
        "behavior": "Nasabah teredukasi yang memanfaatkan fitur analisis mendalam. Menggabungkan strategi value investing dan swing momentum.",
        "future_price_impact": "🟢 SINYAL AKUMULASI RITEL TERDIDIK: Akumulasi bertahap XL menandakan sentimen positif yang mulai menyebar di kalangan investor ritel berkualitas."
    },
    "SQ": {
        "code": "SQ",
        "name": "PT BCA Sekuritas",
        "category": "💼 Ritel High Net Worth & Institusi Swasta",
        "archetype": "Affluent Private Wealth",
        "behavior": "Nasabah BCA Prioritas/Solitaire yang berinvestasi dengan modal relatif besar dan profil risiko tenang. Menyukai saham berkinerja sehat.",
        "future_price_impact": "🟢 PERTUMBUHAN STABIL: Akumulasi SQ menandakan aliran modal swasta lokal berbobot tinggi yang menopang kenaikan harga jangka menengah."
    },
    "LG": {
        "code": "LG",
        "name": "PT Trimegah Sekuritas Indonesia",
        "category": "🏛️ Institusi Swasta & Underwriter IPO",
        "archetype": "Private Equity & Promoter Allied",
        "behavior": "Kerap menjadi penjamin pelaksana emisi IPO baru dan memiliki hubungan erat dengan pemegang saham pengendali (promoter).",
        "future_price_impact": "🟢 PENGEREKAN HARGA SPONSOR: Net Buy LG pada saham binaannya sering menjadi sinyal bahwa pengelola harga (market maker resmi) sedang menjaga dan menaikkan harga."
    },
    "AI": {
        "code": "AI",
        "name": "PT UOB Kay Hian Sekuritas",
        "category": "🏛️ Smart Money Regional (Singapura/HK)",
        "archetype": "Regional Institutional Capital",
        "behavior": "Menghubungkan investor kaya dan hedge fund Asia Tenggara ke pasar saham Indonesia, khususnya sektor komoditas dan industri.",
        "future_price_impact": "🟢 AKUMULASI SEKTORAL: Net Buy AI sering kali menandakan masuknya sentimen positif regional ke sektor terkait."
    },
    "KK": {
        "code": "KK",
        "name": "PT Phillip Sekuritas Indonesia",
        "category": "👥 Ritel Mandiri & Reksa Dana Domestik",
        "archetype": "Retail Online & Mutual Fund Inflow",
        "behavior": "Platform ritel online terlama di Indonesia. Transaksinya mencerminkan pergerakan portofolio ritel mandiri dan reksa dana lokal.",
        "future_price_impact": "⚪ PENGIKUT TREN: Pergerakannya cenderung selaras dengan tren arah pasar secara keseluruhan."
    },
    "CP": {
        "code": "CP",
        "name": "PT KB Valbury Sekuritas",
        "category": "💼 Institusi Domestik & Fast Trader",
        "archetype": "Commodity & Fast Money Trader",
        "behavior": "Sangat aktif di saham-saham tambang, energi, dan komoditas siklikal.",
        "future_price_impact": "🟢 MOMENTUM SIKLIKAL: Menandakan kebangkitan siklus harga komoditas pada saham terkait."
    },
    "EP": {
        "code": "EP",
        "name": "PT MNC Sekuritas",
        "category": "💼 Ritel & Terafiliasi Konglomerasi MNC",
        "archetype": "Conglomerate Inflow & Retail Network",
        "behavior": "Memfasilitasi transaksi investor ritel dan ekosistem konglomerasi media, properti, dan perbankan MNC Group.",
        "future_price_impact": "🟢 PENGGERAK SAHAM TERAFFILIASI: Menjadi penentu utama pergerakan saham-saham di bawah naungan grup konglomerasi terkait."
    },
    "DR": {
        "code": "DR",
        "name": "PT RHB Sekuritas Indonesia",
        "category": "🏛️ Institusi Regional ASEAN (Malaysia)",
        "archetype": "Regional ASEAN Fund",
        "behavior": "Memiliki fokus pada saham-saham perkebunan CPO, konstruksi, dan perbankan di Asia Tenggara.",
        "future_price_impact": "🟢 PENGUATAN TREN REGIONAL: Net Buy DR sering memperkuat tren kenaikan saham komoditas agribisnis dan energi."
    },
    "CD": {
        "code": "CD",
        "name": "PT Mega Capital Sekuritas",
        "category": "💼 Terafiliasi Konglomerasi CT Corp",
        "archetype": "Private Conglomerate Desk",
        "behavior": "Menampung transaksi private investor dan ekosistem CT Corp (ritel, perbankan digital, media).",
        "future_price_impact": "🟢 DUKUNGAN STRATEGIS: Memberikan stabilitas likuiditas pada saham-saham terafiliasi konglomerasi."
    },
    "GR": {
        "code": "GR",
        "name": "PT Panin Sekuritas Tbk.",
        "category": "💼 Ritel HNW & Reksa Dana Panin",
        "archetype": "Value Investor Domestik",
        "behavior": "Terkenal dengan gaya investasi value investing (mencari saham murah dengan aset riil besar) dan nasabah HNW loyal.",
        "future_price_impact": "🟢 AKUMULASI JANGKA PANJANG: Net Buy GR menandakan saham terdiskon secara fundamental dan siap disimpan untuk target multi-bagger."
    },
    "HD": {
        "code": "HD",
        "name": "PT KGI Sekuritas Indonesia",
        "category": "🏛️ Institusi Regional Taiwan/Asia",
        "archetype": "Asia Regional Wealth Desk",
        "behavior": "Menyalurkan dana investor swasta Asia Timur pada saham-saham manufaktur dan ekspor.",
        "future_price_impact": "🟢 PENGUATAN EKSPOR: Mengindikasikan minat asing pada saham berbasis ekspor dan komoditas."
    },
    "DP": {
        "code": "DP",
        "name": "PT DBS Vickers Sekuritas Indonesia",
        "category": "🏛️ Smart Money Asing (Singapura)",
        "archetype": "Institutional Wealth Management",
        "behavior": "Mengelola portofolio institusi global dengan standar kepatuhan tinggi pada saham-saham LQ45.",
        "future_price_impact": "🟢 KREDIBILITAS TINGGI: Net Buy konsisten DP memberikan rasa aman bagi pelaku pasar bahwa saham tersebut aman secara tata kelola korporasi."
    },
    "CG": {
        "code": "CG",
        "name": "PT CGS International Sekuritas Indonesia",
        "category": "🏛️ Institusi Regional China/ASEAN",
        "archetype": "Cross-border Institutional Capital",
        "behavior": "Menjembatani aliran dana investasi institusi China dan Asia Tenggara ke pasar saham Indonesia.",
        "future_price_impact": "🟢 ARUS MODAL INTERNASIONAL: Net Buy CG mencerminkan penempatan modal asing jangka menengah pada sektor riil."
    },
    "XA": {
        "code": "XA",
        "name": "PT NH Korindo Sekuritas Indonesia",
        "category": "💼 Institusi Korea & Ritel Domestik",
        "archetype": "Korean Capital & Retail Desk",
        "behavior": "Menghubungkan ekosistem investasi Korea Selatan dengan bursa Indonesia, aktif di saham teknologi dan logistik.",
        "future_price_impact": "🟢 MOMENTUM SPESIFIK: Mengindikasikan ekspansi bisnis atau kemitraan dengan entitas korporasi internasional."
    },
    "KI": {
        "code": "KI",
        "name": "PT Ciptadana Sekuritas Asia",
        "category": "💼 Institusi Swasta Domestik Terkemuka",
        "archetype": "Independent Domestic Institution",
        "behavior": "Sekuritas independen mapan dengan nasabah institusi lokal dan manajer investasi terpercaya.",
        "future_price_impact": "🟢 KUALITAS INVESTASI: Net Buy KI menunjukkan saham memiliki rasio valuasi menarik dan likuiditas sehat."
    },
    "HP": {
        "code": "HP",
        "name": "PT Henan Putihrai Sekuritas",
        "category": "💼 Institusi Swasta & Komunitas Sharia",
        "archetype": "Boutique Investment & Sharia Specialist",
        "behavior": "Sekuritas butik senior yang aktif mengelola transaksi saham syariah dan nasabah loyal.",
        "future_price_impact": "🟢 AKUMULASI SYARIAH: Mendukung apresiasi harga pada saham-saham yang masuk dalam indeks JII / ISSI."
    },
    "SH": {
        "code": "SH",
        "name": "PT Shinhan Sekuritas Indonesia",
        "category": "🏛️ Institusi Korea Selatan",
        "archetype": "Global Financial Group Network",
        "behavior": "Menyalurkan dana konglomerasi dan perbankan Shinhan Financial Group ke instrumen ekuitas BEI.",
        "future_price_impact": "🟢 DUKUNGAN KEUANGAN ASING: Menandai saham-saham yang memiliki daya tarik bagi investor institusi Asia Timur."
    },
    "IF": {
        "code": "IF",
        "name": "PT Samuel Sekuritas Indonesia",
        "category": "💼 Riset Domestik & Institusi Swasta",
        "archetype": "Comprehensive Domestic Research",
        "behavior": "Divisi riset makro dan sektoral terkemuka yang menjadi rujukan fund manager domestik.",
        "future_price_impact": "🟢 SINYAL RISET POSITIF: Pembelian IF mendahului sentimen kenaikan laba bersih emiten."
    },
    "LS": {
        "code": "LS",
        "name": "PT Reliance Sekuritas Indonesia Tbk.",
        "category": "👥 Ritel Domestik & Asuransi",
        "archetype": "Domestic Retail & Insurance Desk",
        "behavior": "Terdaftar publik di BEI, melayani jaringan ritel mandiri dan unit asuransi grup Reliance.",
        "future_price_impact": "⚪ PENGIKUT PASAR: Transaksi cenderung dinamis mengikuti tren likuiditas harian bursa."
    }
}

# Klasifikasi Kode Broker BEI
INSTITUTIONAL_BROKERS = {
    code for code, data in IDX_BROKER_DIRECTORY.items() if "Asing" in data["category"] or "BUMN" in data["category"] or "Institusi" in data["category"]
}
RETAIL_BROKERS = {
    code for code, data in IDX_BROKER_DIRECTORY.items() if "Ritel" in data["category"] or "Bandar" in data["category"] or "Pemula" in data["category"]
}


def get_broker_info(broker_code: str) -> Dict[str, str]:
    """Mengambil metadata profil resmi broker, kategori, sifat transaksi, dan implikasi pergerakan harga."""
    code_clean = str(broker_code).upper().strip()
    if code_clean in IDX_BROKER_DIRECTORY:
        return IDX_BROKER_DIRECTORY[code_clean]
    
    # Fallback jika broker belum terdaftar di katalog
    is_inst = code_clean in INSTITUTIONAL_BROKERS
    is_ret = code_clean in RETAIL_BROKERS
    cat = "🏛️ Institusi Domestik/Asing" if is_inst else ("👥 Broker Ritel" if is_ret else "💼 Broker Swasta BEI")
    return {
        "code": code_clean,
        "name": f"Broker Anggota Bursa ({code_clean})",
        "category": cat,
        "archetype": "General Participant",
        "behavior": "Broker peserta perdagangan aktif di Bursa Efek Indonesia.",
        "future_price_impact": "⚪ NETRAL: Mengikuti dinamika supply dan demand pasar reguler."
    }


# ==============================================================================
# 1. DETEKSI AKUMULASI INSTITUSI PADA SAHAM STAGNAN (SILENT ACCUMULATION)
# ==============================================================================
def detect_silent_institutional_accumulation(
    df_ohlcv: pd.DataFrame,
    broker_summary_df: Optional[pd.DataFrame] = None,
    window: int = 14,
    volatility_thresh: float = 0.02
) -> Dict[str, Any]:
    """
    1. Deteksi Fase Konsolidasi (Stagnasi): Volatilitas sempit (< 2%) selama 10-14 hari bursa.
    2. Identifikasi Akumulasi Institusi Senyap: Jejak transaksi volume besar oleh broker institusi.
    3. Konfirmasi Sinyal: Metrik penilaian (Skor Akumulasi) membandingkan Net Buy institusi vs Net Sell ritel.
    """
    if df_ohlcv is None or len(df_ohlcv) < window:
        return {
            "is_stagnant": False,
            "is_accumulating": False,
            "accumulation_score": 0.0,
            "status": "DATA TIDAK CUKUP",
            "volatility_pct": 0.0,
            "consolidation_range": (0.0, 0.0),
            "top_institutional_buyers": [],
            "top_retail_sellers": [],
            "message": f"Data historis kurang dari {window} baris."
        }

    sub_df = df_ohlcv.tail(window).copy()
    high_max = float(sub_df["High"].max())
    low_min = float(sub_df["Low"].min())
    close_last = float(sub_df["Close"].iloc[-1])

    # Volatilitas rentang harga selama window
    if low_min > 0:
        volatility_pct = (high_max - low_min) / low_min
    else:
        volatility_pct = 0.0

    is_stagnant = volatility_pct <= volatility_thresh

    if broker_summary_df is None or broker_summary_df.empty:
        broker_summary_df = generate_synthetic_broker_summary(close_last, sub_df["Volume"].tail(window).sum())
    else:
        broker_summary_df = broker_summary_df.copy()
        if "broker_name" not in broker_summary_df.columns:
            broker_summary_df["broker_name"] = broker_summary_df["broker_code"].apply(lambda c: get_broker_info(c)["name"])
        if "category" not in broker_summary_df.columns:
            broker_summary_df["category"] = broker_summary_df["broker_code"].apply(lambda c: get_broker_info(c)["category"])
        if "archetype" not in broker_summary_df.columns:
            broker_summary_df["archetype"] = broker_summary_df["broker_code"].apply(lambda c: get_broker_info(c)["archetype"])
        if "future_price_impact" not in broker_summary_df.columns:
            broker_summary_df["future_price_impact"] = broker_summary_df["broker_code"].apply(lambda c: get_broker_info(c)["future_price_impact"])

    # Analisis Net Buy Institusi vs Net Sell Ritel
    inst_df = broker_summary_df[broker_summary_df["broker_code"].isin(INSTITUTIONAL_BROKERS)]
    retail_df = broker_summary_df[broker_summary_df["broker_code"].isin(RETAIL_BROKERS)]

    inst_net_buy = float(inst_df[inst_df["net_vol"] > 0]["net_vol"].sum())
    retail_net_sell = float(abs(retail_df[retail_df["net_vol"] < 0]["net_vol"].sum()))
    total_vol = float(broker_summary_df["buy_vol"].sum() + 1e-6)

    # Skor Akumulasi (0 - 100)
    # Membandingkan rasio net buy institusi dan net sell ritel
    inst_ratio = inst_net_buy / total_vol
    retail_ratio = retail_net_sell / total_vol
    
    # Formula skor berbasis konsentrasi
    score_raw = (inst_ratio * 60.0) + (retail_ratio * 40.0)
    accumulation_score = min(max(score_raw * 1.5, 5.0), 99.0) if is_stagnant else min(score_raw, 50.0)

    is_accumulating = is_stagnant and (accumulation_score >= 60.0 or inst_net_buy > retail_net_sell)

    status_str = "🟢 AKUMULASI INSTITUSI SENYAP" if is_accumulating else (
        "🟡 KONSOLIDASI (BELUM AKUMULASI)" if is_stagnant else "⚪ TIDAK STAGNAN (TREN AKTIF)"
    )

    # Urutkan top buyer & seller serta perkaya dengan metadata resmi dan implikasi harga
    top_inst_raw = inst_df.sort_values(by="net_vol", ascending=False).head(5).to_dict(orient="records")
    top_inst = []
    for item in top_inst_raw:
        b_info = get_broker_info(item["broker_code"])
        top_inst.append({
            "broker_code": item["broker_code"],
            "broker_name": b_info["name"],
            "category": b_info["category"],
            "archetype": b_info["archetype"],
            "net_vol": int(item["net_vol"]),
            "buy_vol": int(item.get("buy_vol", 0)),
            "sell_vol": int(item.get("sell_vol", 0)),
            "future_price_impact": b_info["future_price_impact"]
        })

    top_ret_raw = retail_df.sort_values(by="net_vol", ascending=True).head(5).to_dict(orient="records")
    top_ret = []
    for item in top_ret_raw:
        b_info = get_broker_info(item["broker_code"])
        top_ret.append({
            "broker_code": item["broker_code"],
            "broker_name": b_info["name"],
            "category": b_info["category"],
            "archetype": b_info["archetype"],
            "net_vol": int(item["net_vol"]),
            "buy_vol": int(item.get("buy_vol", 0)),
            "sell_vol": int(item.get("sell_vol", 0)),
            "future_price_impact": b_info["future_price_impact"]
        })

    return {
        "is_stagnant": is_stagnant,
        "is_accumulating": is_accumulating,
        "accumulation_score": round(accumulation_score, 1),
        "status": status_str,
        "volatility_pct": round(volatility_pct * 100, 2),
        "consolidation_range": (round(low_min, 2), round(high_max, 2)),
        "current_price": close_last,
        "inst_net_buy_lots": int(inst_net_buy),
        "retail_net_sell_lots": int(retail_net_sell),
        "top_institutional_buyers": top_inst,
        "top_retail_sellers": top_ret,
        "window_days": window
    }


# ==============================================================================
# 2. PREDIKSI ARAH PERGERAKAN JANGKA PENDEK (3 PILAR ANALISIS)
# ==============================================================================
def run_vader_sentiment(texts: List[str]) -> Dict[str, Any]:
    """Menghitung skor sentimen NLP dengan algoritma VADER."""
    if not texts:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "sentiment_label": "NETRAL"}

    compounds = []
    pos_scores = []
    neg_scores = []
    neu_scores = []

    for text in texts:
        if not text or not isinstance(text, str):
            continue
        if HAS_VADER and _vader_analyzer is not None:
            scores = _vader_analyzer.polarity_scores(text)
            compounds.append(scores["compound"])
            pos_scores.append(scores["pos"])
            neg_scores.append(scores["neg"])
            neu_scores.append(scores["neu"])
        else:
            # Fallback jika VADER belum terinisialisasi
            clean_t = text.lower()
            pos_words = ["profit", "laba", "naik", "cuan", "dividen", "akuisisi", "tumbuh", "rekor", "bullish", "lonjakan"]
            neg_words = ["rugi", "turun", "anjlok", "suspensi", "delisting", "gugat", "utang", "bearish", "merosot", "jatuh"]
            p_cnt = sum(1 for w in pos_words if w in clean_t)
            n_cnt = sum(1 for w in neg_words if w in clean_t)
            tot = p_cnt + n_cnt
            if tot > 0:
                cp = (p_cnt - n_cnt) / tot
            else:
                cp = 0.0
            compounds.append(cp)
            pos_scores.append(0.7 if cp > 0 else 0.1)
            neg_scores.append(0.7 if cp < 0 else 0.1)
            neu_scores.append(0.2)

    if not compounds:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "sentiment_label": "NETRAL"}

    mean_compound = float(np.mean(compounds))
    mean_pos = float(np.mean(pos_scores))
    mean_neg = float(np.mean(neg_scores))
    mean_neu = float(np.mean(neu_scores))

    if mean_compound >= 0.15:
        label = "🟢 POSITIF (BULLISH)"
    elif mean_compound <= -0.15:
        label = "🔴 NEGATIF (BEARISH)"
    else:
        label = "⚪ NETRAL"

    return {
        "compound": round(mean_compound, 3),
        "pos": round(mean_pos, 3),
        "neg": round(mean_neg, 3),
        "neu": round(mean_neu, 3),
        "sentiment_label": label
    }


def predict_short_term_trajectory(
    df_ohlcv: pd.DataFrame,
    broker_summary_df: Optional[pd.DataFrame] = None,
    news_titles: Optional[List[str]] = None,
    weights: Tuple[float, float, float] = (0.45, 0.25, 0.30),
    forecast_steps: int = 3
) -> Dict[str, Any]:
    """
    Memprediksi arah jangka pendek (Intraday s/d H+1) dengan mengintegrasikan 3 pilar:
    1. Pilar Teknikal & Momentum (EMA 9, EMA 20, RSI, MACD, Model Holt-Winters)
    2. Pilar Katalis Sentimen (NLP VADER)
    3. Pilar Mikrostruktur Pasar / Order Flow (Net Buy Top 10 Broker vs Net Sell Top 10 Broker)
    """
    if df_ohlcv is None or len(df_ohlcv) < 25:
        return {"status": "ERROR", "message": "Data historis minimal 25 baris."}

    df = df_ohlcv.copy()
    close = df["Close"]

    # 1. PILAR TEKNIKAL & MOMENTUM
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    curr_close = float(close.iloc[-1])
    curr_ema9 = float(ema9.iloc[-1])
    curr_ema20 = float(ema20.iloc[-1])
    curr_rsi = float(rsi.iloc[-1])
    curr_macd = float(macd_line.iloc[-1])
    curr_sig = float(signal_line.iloc[-1])

    # Skor Momentum Teknikal (-1.0 s/d +1.0)
    tech_score = 0.0
    if curr_ema9 > curr_ema20:
        tech_score += 0.35
    else:
        tech_score -= 0.35

    if curr_rsi > 50:
        tech_score += 0.30 * min((curr_rsi - 50) / 25, 1.0)
    else:
        tech_score -= 0.30 * min((50 - curr_rsi) / 25, 1.0)

    if curr_macd > curr_sig:
        tech_score += 0.35
    else:
        tech_score -= 0.35

    tech_score = max(min(tech_score, 1.0), -1.0)

    # Peramalan Holt-Winters (Exponential Smoothing)
    hw_forecast = []
    if HAS_STATSMODELS and len(close) >= 15:
        try:
            # Model Double Exponential Smoothing (Holt's Linear Trend)
            hw_model = ExponentialSmoothing(
                close.values,
                trend="add",
                seasonal=None,
                damped_trend=True
            ).fit(damping_trend=0.9)
            hw_forecast = [float(x) for x in hw_model.forecast(forecast_steps)]
        except Exception:
            slope = (curr_close - float(close.iloc[-5])) / 5.0
            hw_forecast = [curr_close + slope * (i + 1) for i in range(forecast_steps)]
    else:
        slope = (curr_close - float(close.iloc[-5])) / 5.0
        hw_forecast = [curr_close + slope * (i + 1) for i in range(forecast_steps)]

    # 2. PILAR SENTIMEN VADER
    sentiment_result = run_vader_sentiment(news_titles or [])
    sentiment_score = sentiment_result["compound"]  # -1.0 s/d +1.0

    # 3. PILAR MIKROSTRUKTUR (ORDER FLOW BROKER SUMMARY)
    if broker_summary_df is None or broker_summary_df.empty:
        broker_summary_df = generate_synthetic_broker_summary(curr_close, df["Volume"].tail(5).sum())

    top10_buy_df = broker_summary_df.nlargest(10, "buy_vol")
    top10_sell_df = broker_summary_df.nlargest(10, "sell_vol")
    top10_buy_lots = float(top10_buy_df["buy_vol"].sum())
    top10_sell_lots = float(top10_sell_df["sell_vol"].sum())
    net_order_flow = top10_buy_lots - top10_sell_lots
    total_top10 = top10_buy_lots + top10_sell_lots + 1e-6
    order_flow_ratio = (top10_buy_lots - top10_sell_lots) / total_top10  # -1.0 s/d +1.0

    lead_buy_code = str(top10_buy_df.iloc[0]["broker_code"]) if len(top10_buy_df) > 0 else "CC"
    lead_sell_code = str(top10_sell_df.iloc[0]["broker_code"]) if len(top10_sell_df) > 0 else "YP"
    lead_buy_info = get_broker_info(lead_buy_code)
    lead_sell_info = get_broker_info(lead_sell_code)

    # 4. PENGAMBILAN KEPUTUSAN TERPADU
    w_tech, w_sent, w_of = weights
    composite_score = (tech_score * w_tech) + (sentiment_score * w_sent) + (order_flow_ratio * w_of)

    if composite_score >= 0.25:
        direction = "🟢 BULLISH / POTENSI NAIK (BUY)"
        action = "STRONG BUY"
    elif composite_score >= 0.08:
        direction = "🟡 AKUMULASI PERLAHAN (ACCUMULATE)"
        action = "BUY ON WEAKNESS"
    elif composite_score <= -0.25:
        direction = "🔴 BEARISH / POTENSI TURUN (SELL)"
        action = "TAKE PROFIT / CUT LOSS"
    else:
        direction = "⚪ NETRAL / KONSOLIDASI (WAIT & SEE)"
        action = "HOLD / WAIT"

    # Rekomendasi Target Harga & SL
    atr_val = float((df["High"] - df["Low"]).tail(14).mean())
    tp_price = round(curr_close + (atr_val * (1.5 if composite_score > 0 else 0.8)), 0)
    sl_price = round(curr_close - (atr_val * 1.0), 0)

    return {
        "status": "SUCCESS",
        "current_price": curr_close,
        "composite_score": round(composite_score, 3),
        "direction": direction,
        "action": action,
        "tech_pilar": {
            "ema9": round(curr_ema9, 2),
            "ema20": round(curr_ema20, 2),
            "rsi": round(curr_rsi, 2),
            "macd": round(curr_macd, 2),
            "macd_signal": round(curr_sig, 2),
            "tech_score": round(tech_score, 3),
            "holt_winters_forecast": [round(x, 2) for x in hw_forecast],
        },
        "sentiment_pilar": sentiment_result,
        "order_flow_pilar": {
            "top10_buy_lots": int(top10_buy_lots),
            "top10_sell_lots": int(top10_sell_lots),
            "net_order_flow_lots": int(net_order_flow),
            "order_flow_score": round(order_flow_ratio, 3),
            "lead_buyer": lead_buy_info,
            "lead_seller": lead_sell_info,
        },
        "recommendation": {
            "target_price": tp_price,
            "stop_loss": sl_price,
            "potential_gain_pct": round(((tp_price - curr_close) / curr_close) * 100, 2),
            "risk_pct": round(((curr_close - sl_price) / curr_close) * 100, 2),
        }
    }


# ==============================================================================
# 3. DETEKSI INDIKASI PUMP AND DUMP (HARGA, VOLUME, & SENTIMEN)
# ==============================================================================
def detect_pump_and_dump(
    df_ohlcv: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    ticker: str = ""
) -> List[Dict[str, Any]]:
    """
    1. Fase Stagnan (Akumulasi): volatilitas < 2% selama minimum 14 hari bursa.
    2. Fase Pump: lonjakan harga > 10% dalam 1 hari & volume > 300% dari SMA 20 Volume.
    3. Validasi Sentimen VADER: Red flag jika lonjakan terjadi saat sentimen netral/negatif
       atau tanpa berita fundamental (corporate action) 3 hari terakhir.
    """
    if df_ohlcv is None or len(df_ohlcv) < 35:
        return []

    df = df_ohlcv.copy()
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Pct_Change"] = df["Close"].pct_change()
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_SMA20"] + 1e-6)

    alerts = []
    # Loop mengecek hari-hari perdagangan
    for i in range(25, len(df)):
        day_row = df.iloc[i]
        pct_chg = day_row["Pct_Change"]
        vol_ratio = day_row["Vol_Ratio"]

        # Syarat Fase 2 (Pump): Naik > 10% dan Volume > 300% (3x SMA20)
        if pct_chg >= 0.10 and vol_ratio >= 3.0:
            # Cek Fase 1 (Stagnan): 14 hari sebelum lonjakan harus memiliki volatilitas sempit < 2% (atau max-min / min < 0.04)
            prev_14 = df.iloc[i-14:i]
            prev_high = prev_14["High"].max()
            prev_low = prev_14["Low"].min()
            stagnant_range = (prev_high - prev_low) / (prev_low + 1e-6)

            # Validasi Sentimen VADER pada berita di sekitar tanggal kejadian
            event_date = df.index[i]
            date_str = str(event_date)[:10]

            sentiment_score = 0.0
            has_fundamental_news = False
            if news_df is not None and not news_df.empty:
                # Filter berita dalam jendela 3 hari sebelum kejadian
                # Jika tidak ada kolom date, gunakan teks yang ada
                news_subset = news_df.head(5)
                news_texts = [str(x) for x in news_subset.get("title", [])]
                vader_res = run_vader_sentiment(news_texts)
                sentiment_score = vader_res["compound"]
                has_fundamental_news = any(
                    any(k in t.lower() for k in ["akuisisi", "dividen", "kinerja", "laba", "merger", "kontrak"])
                    for t in news_texts
                )

            # Kriteria Red Flag:
            # Lonjakan masif tapi sentimen netral/negatif (compound <= 0.05) ATAU tanpa corporate action
            is_red_flag = (sentiment_score <= 0.05) or (not has_fundamental_news)

            if is_red_flag:
                alerts.append({
                    "ticker": ticker or "IDX",
                    "date": date_str,
                    "price_surge_pct": round(pct_chg * 100, 2),
                    "volume_surge_pct": round(vol_ratio * 100, 1),
                    "stagnant_range_pct": round(stagnant_range * 100, 2),
                    "sentiment_score": round(sentiment_score, 3),
                    "has_fundamental_news": "🟢 Ada" if has_fundamental_news else "🔴 Tidak Ada",
                    "severity": "🚨 RED FLAG (PUMP & DUMP SUSPECT)",
                    "recommendation": "HINDARI MEMBELI DI PUCUK / SEGERA EXIT JIKA MEMILIKI POSISI"
                })

    return alerts


# ==============================================================================
# 4. ANALISIS ORDER BOOK LEVEL 2: DETEKSI SPOOFING & LAYERING
# ==============================================================================
def detect_spoofing_and_layering(
    order_book_l2_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    1. Identifikasi antrean bid/ask 5x lebih besar dari rata-rata volume di 5 tingkat harga teratas.
    2. Lacak apakah antrean raksasa tersebut dibatalkan (cancel order) sepenuhnya < 5 detik sebelum harga menyentuh level tersebut.
    3. Hitung Cancel-to-Trade Ratio: Jika > 80% pada tingkat harga tertentu -> klasifikasikan sebagai indikasi Spoofing.
    4. Sediakan data untuk visualisasi grafik titik pasang & cabut.
    """
    if order_book_l2_df is None or order_book_l2_df.empty:
        order_book_l2_df = generate_synthetic_l2_order_book(current_price=1000.0)

    df = order_book_l2_df.copy()
    
    # Hitung rata-rata volume di 5 tingkat teratas
    avg_top5_bid = df["bid_volume"].head(5).mean()
    avg_top5_ask = df["ask_volume"].head(5).mean()

    spoof_events = []
    timeline_events = []

    # Deteksi anomali antrean raksasa (5x)
    for idx, row in df.iterrows():
        bid_vol = row.get("bid_volume", 0)
        ask_vol = row.get("ask_volume", 0)
        bid_p = row.get("bid_price", 0)
        ask_p = row.get("ask_price", 0)
        timestamp = row.get("timestamp", str(datetime.now().time())[:8])
        event_type = row.get("event_type", "NEW")
        seconds_to_touch = row.get("seconds_before_touch", 10.0)
        cancel_ratio = row.get("cancel_to_trade_ratio", 0.0)

        # Cek sisi Bid
        if bid_vol >= (5.0 * avg_top5_bid):
            is_spoof = (seconds_to_touch < 5.0) and (cancel_ratio > 0.80)
            status = "🚨 SPOOFING DETECTED" if is_spoof else "⚠️ GIANT BID ORDER"
            spoof_events.append({
                "timestamp": str(timestamp),
                "side": "BID",
                "price": bid_p,
                "volume": int(bid_vol),
                "ratio_to_avg": round(bid_vol / (avg_top5_bid + 1e-6), 1),
                "cancel_delay_sec": round(seconds_to_touch, 1),
                "cancel_to_trade_ratio": round(cancel_ratio * 100, 1),
                "status": status
            })
            timeline_events.append({
                "time": str(timestamp),
                "price": bid_p,
                "type": "BID_SPOOF" if is_spoof else "BID_ORDER",
                "volume": int(bid_vol),
                "color": "#ff0055" if is_spoof else "#00ffcc"
            })

        # Cek sisi Ask
        if ask_vol >= (5.0 * avg_top5_ask):
            is_spoof = (seconds_to_touch < 5.0) and (cancel_ratio > 0.80)
            status = "🚨 SPOOFING DETECTED" if is_spoof else "⚠️ GIANT ASK ORDER"
            spoof_events.append({
                "timestamp": str(timestamp),
                "side": "ASK",
                "price": ask_p,
                "volume": int(ask_vol),
                "ratio_to_avg": round(ask_vol / (avg_top5_ask + 1e-6), 1),
                "cancel_delay_sec": round(seconds_to_touch, 1),
                "cancel_to_trade_ratio": round(cancel_ratio * 100, 1),
                "status": status
            })
            timeline_events.append({
                "time": str(timestamp),
                "price": ask_p,
                "type": "ASK_SPOOF" if is_spoof else "ASK_ORDER",
                "volume": int(ask_vol),
                "color": "#ffaa00" if is_spoof else "#ff5555"
            })

    # Summary rasio pembatalan
    total_spoof_count = sum(1 for e in spoof_events if "SPOOFING" in e["status"])
    overall_spoof_risk = "TINGGI (TERDETEKSI MANIPULASI)" if total_spoof_count > 0 else "RENDAH (NORMAL)"

    return {
        "status": "SUCCESS",
        "avg_top5_bid_volume": int(avg_top5_bid),
        "avg_top5_ask_volume": int(avg_top5_ask),
        "total_spoof_events": total_spoof_count,
        "overall_spoof_risk": overall_spoof_risk,
        "spoof_details": spoof_events,
        "timeline_events": timeline_events,
    }


# ==============================================================================
# 5. DETEKSI MARKING THE CLOSE (DATA INTRADAY 1 MENIT)
# ==============================================================================
def detect_marking_the_close(
    df_intraday_1m: pd.DataFrame,
    ticker: str = ""
) -> Dict[str, Any]:
    """
    1. Hitung VWAP pukul 09:00 - 15:50 WIB.
    2. Bandingkan VWAP dengan harga penutupan di 16:00 WIB (Pre-Closing).
    3. Sinyal 'Marking the Close' jika harga 16:00 menyimpang > 3% dari VWAP
       DAN 40% dari total volume harian dieksekusi hanya pada 10 menit terakhir (15:50-16:00 WIB).
    """
    if df_intraday_1m is None or df_intraday_1m.empty:
        df_intraday_1m = generate_synthetic_intraday_1m(current_price=1000.0)

    df = df_intraday_1m.copy()
    # Pastikan kolom datetime tersedia
    if "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"])
    else:
        df["dt"] = pd.to_datetime(df.index)

    df["time_val"] = df["dt"].dt.time

    # Filter jam perdagangan normal BEI: 09:00 - 15:50 WIB
    normal_mask = (df["time_val"] >= dtime(9, 0)) & (df["time_val"] < dtime(15, 50))
    pre_close_mask = (df["time_val"] >= dtime(15, 50)) & (df["time_val"] <= dtime(16, 0))

    df_normal = df[normal_mask]
    df_pre_close = df[pre_close_mask]

    if df_normal.empty or df_pre_close.empty:
        # Fallback jika timestamp tidak dalam rentang jam bursa tepat
        split_idx = int(len(df) * 0.90)
        df_normal = df.iloc[:split_idx]
        df_pre_close = df.iloc[split_idx:]

    # 1. VWAP Sesi Normal (09:00 - 15:50)
    vol_normal = float(df_normal["volume"].sum())
    pv_normal = float((df_normal["close"] * df_normal["volume"]).sum())
    vwap_normal = pv_normal / (vol_normal + 1e-6)

    # 2. Volume & Harga Penutupan Sesi Pre-Closing (15:50 - 16:00)
    vol_pre_close = float(df_pre_close["volume"].sum())
    total_daily_vol = vol_normal + vol_pre_close + 1e-6
    close_1600 = float(df_pre_close["close"].iloc[-1])

    # 3. Kriteria Deteksi
    pct_vol_in_last_10m = (vol_pre_close / total_daily_vol) * 100.0
    price_deviation_pct = ((close_1600 - vwap_normal) / vwap_normal) * 100.0

    # Syarat: Deviasi > 3% dan Volume 10m terakhir >= 40%
    is_marking_the_close = (abs(price_deviation_pct) >= 3.0) and (pct_vol_in_last_10m >= 40.0)

    status_str = "🚨 TERINDIKASI MARKING THE CLOSE" if is_marking_the_close else "🟢 NORMAL (TIDAK ADA MANIPULASI PENUTUPAN)"

    return {
        "ticker": ticker or "IDX",
        "is_marking_the_close": is_marking_the_close,
        "status": status_str,
        "vwap_normal_hours": round(vwap_normal, 2),
        "closing_price_1600": round(close_1600, 2),
        "price_deviation_pct": round(price_deviation_pct, 2),
        "vol_pre_close_lots": int(vol_pre_close),
        "total_daily_vol_lots": int(total_daily_vol),
        "vol_in_last_10m_pct": round(pct_vol_in_last_10m, 2),
        "intraday_df": df
    }


# ==============================================================================
# 6. DETEKSI WASH TRADING (ANALISIS BROKER SUMMARY)
# ==============================================================================
def detect_wash_trading(
    broker_summary_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    1. Hitung total transaksi kotor (gross_vol = buy_vol + sell_vol) dari suatu broker.
    2. Flag broker jika:
       - Transaksi kotor broker mendominasi >= 30% dari total volume saham harian, TETAPI
       - Nilai net_vol (buy_vol - sell_vol) mendekati nol (< 5% dari transaksi kotornya).
    """
    if broker_summary_df is None or broker_summary_df.empty:
        broker_summary_df = generate_synthetic_broker_summary(1000.0, 500000)

    df = broker_summary_df.copy()
    df["gross_vol"] = df["buy_vol"] + df["sell_vol"]
    df["net_vol"] = df["buy_vol"] - df["sell_vol"]

    total_market_gross = float(df["gross_vol"].sum() + 1e-6)
    df["dominance_pct"] = (df["gross_vol"] / total_market_gross) * 100.0
    df["net_to_gross_pct"] = (df["net_vol"].abs() / (df["gross_vol"] + 1e-6)) * 100.0

    # Pastikan seluruh baris memiliki metadata nama perusahaan sekuritas, kategori, arketipe, dan implikasi harga
    df["broker_name"] = df["broker_code"].apply(lambda c: get_broker_info(c)["name"])
    df["category"] = df["broker_code"].apply(lambda c: get_broker_info(c)["category"])
    df["archetype"] = df["broker_code"].apply(lambda c: get_broker_info(c)["archetype"])
    df["future_price_impact"] = df["broker_code"].apply(lambda c: get_broker_info(c)["future_price_impact"])

    flagged_brokers = []
    for _, row in df.iterrows():
        b_code = row["broker_code"]
        dom = row["dominance_pct"]
        net_ratio = row["net_to_gross_pct"]
        b_info = get_broker_info(b_code)

        # Kriteria: Dominasi >= 30% DAN net_to_gross < 5%
        if dom >= 30.0 and net_ratio < 5.0:
            flagged_brokers.append({
                "broker_code": b_code,
                "broker_name": b_info["name"],
                "category": b_info["category"],
                "gross_volume": int(row["gross_vol"]),
                "buy_volume": int(row["buy_vol"]),
                "sell_volume": int(row["sell_vol"]),
                "net_volume": int(row["net_vol"]),
                "dominance_pct": round(dom, 2),
                "net_to_gross_pct": round(net_ratio, 2),
                "type": b_info["category"],
                "future_price_impact": b_info["future_price_impact"],
                "verdict": "🚨 TERINDIKASI WASH TRADING (TRANSAKSI PUTAR)"
            })

    is_wash_trading_present = len(flagged_brokers) > 0
    return {
        "is_detected": is_wash_trading_present,
        "verdict": "🚨 TERDETEKSI WASH TRADING" if is_wash_trading_present else "🟢 BERSIH DARI WASH TRADING",
        "flagged_brokers": flagged_brokers,
        "total_analyzed_brokers": len(df),
        "broker_summary_table": df.sort_values(by="gross_vol", ascending=False).to_dict(orient="records")
    }


# ==============================================================================
# DATA GENERATOR HELPER UNTUK SIMULASI REALISTIS BEI
# ==============================================================================
def generate_synthetic_broker_summary(current_price: float, total_volume: float) -> pd.DataFrame:
    """Membuat data broker summary yang realistis berdasarkan kode broker BEI."""
    brokers = [
        ("YP", 0.08, 0.12), ("PD", 0.06, 0.10), ("XC", 0.04, 0.08), ("NI", 0.03, 0.06), # Ritel
        ("ZP", 0.14, 0.02), ("CS", 0.12, 0.03), ("MS", 0.10, 0.02), ("BK", 0.09, 0.01), # Institusi Asing
        ("AK", 0.08, 0.04), ("KZ", 0.07, 0.03), ("CC", 0.05, 0.05), ("OD", 0.04, 0.04),
        ("XL", 0.03, 0.04), ("AZ", 0.02, 0.03), ("CP", 0.02, 0.02), ("LG", 0.03, 0.02)
    ]
    records = []
    base_vol = max(total_volume, 100000.0)
    for b_code, buy_pct, sell_pct in brokers:
        b_vol = int(base_vol * buy_pct)
        s_vol = int(base_vol * sell_pct)
        b_info = get_broker_info(b_code)
        records.append({
            "broker_code": b_code,
            "broker_name": b_info["name"],
            "category": b_info["category"],
            "buy_vol": b_vol,
            "sell_vol": s_vol,
            "net_vol": b_vol - s_vol,
            "future_price_impact": b_info["future_price_impact"]
        })
    return pd.DataFrame(records)


def generate_synthetic_l2_order_book(current_price: float) -> pd.DataFrame:
    """Membuat data simulasi Order Book Level 2 BEI dengan jejak spoofing."""
    base_p = current_price if current_price > 0 else 1000.0
    now = datetime.now()
    records = []
    
    # 10 level harga bid & ask
    for i in range(10):
        t_sec = now - timedelta(seconds=(10 - i) * 3)
        b_price = base_p - (i + 1) * 5
        a_price = base_p + (i + 1) * 5
        
        # Buat antrean raksasa di level 3
        if i == 2:
            b_vol = 85000  # 6x dari rata-rata normal ~12000
            a_vol = 14000
            sec_touch = 3.2  # dibatalkan < 5 detik
            cancel_ratio = 0.89  # > 80%
            ev_type = "CANCEL"
        else:
            b_vol = int(np.random.randint(8000, 16000))
            a_vol = int(np.random.randint(8000, 16000))
            sec_touch = float(np.random.uniform(15.0, 60.0))
            cancel_ratio = float(np.random.uniform(0.1, 0.4))
            ev_type = "TRADE"

        records.append({
            "timestamp": t_sec.strftime("%H:%M:%S"),
            "bid_price": b_price,
            "bid_volume": b_vol,
            "ask_price": a_price,
            "ask_volume": a_vol,
            "event_type": ev_type,
            "seconds_before_touch": sec_touch,
            "cancel_to_trade_ratio": cancel_ratio
        })
    return pd.DataFrame(records)


def generate_synthetic_intraday_1m(current_price: float) -> pd.DataFrame:
    """Membuat data intraday 1 menit dari 09:00 hingga 16:00 WIB."""
    base_p = current_price if current_price > 0 else 1000.0
    times = pd.date_range("2026-09-18 09:00", "2026-09-18 16:00", freq="1min")
    records = []
    curr = base_p
    
    for t in times:
        is_pre_close = (t.hour == 15 and t.minute >= 50) or (t.hour == 16 and t.minute == 0)
        # Sesi normal fluktuasi stabil
        if not is_pre_close:
            curr += np.random.normal(0, 1.5)
            vol = np.random.randint(200, 800)
        else:
            # Sesi Pre-Closing: lonjakan volume masif
            curr += np.random.normal(0.8, 2.0)
            vol = np.random.randint(15000, 45000)
            
        records.append({
            "datetime": t,
            "close": max(curr, 50.0),
            "volume": vol
        })
    return pd.DataFrame(records)


# ==============================================================================
# 7. STREAMLIT UI RENDERER: ANALISIS BROKER DAN EMITEN
# ==============================================================================
def render_broker_emiten_page(
    ticker: str,
    df_ohlcv: pd.DataFrame,
    info: Dict[str, Any],
    news_titles: Optional[List[str]] = None,
    broker_summary_df: Optional[pd.DataFrame] = None,
    order_book_l2_df: Optional[pd.DataFrame] = None,
    intraday_1m_df: Optional[pd.DataFrame] = None
):
    """Merender antarmuka interaktif lengkap untuk menu Analisis Broker dan Emiten."""
    st.markdown('<div class="main-title">🏛️ Analisis Broker dan Emiten BEI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deteksi Akumulasi Senyap Institusi, Prediksi 3 Pilar (Teknikal+Holt-Winters, NLP VADER, Order Flow), Deteksi Pump & Dump, Spoofing/Layering, Marking the Close & Wash Trading</div>', unsafe_allow_html=True)

    curr_p = float(df_ohlcv["Close"].iloc[-1]) if df_ohlcv is not None and not df_ohlcv.empty else 1000.0
    comp_name = info.get("longName") or info.get("shortName") or ticker
    tier_label = info.get("tier", "Saham BEI")

    st.info(f"📌 **Emiten Terpilih**: **{ticker}** ({comp_name}) | **Rp {curr_p:,.0f}** | {tier_label}")

    # Tabs untuk 7 Fitur Utama
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🕵️ Akumulasi Senyap Institusi",
        "🎯 Prediksi 3 Pilar & Holt-Winters",
        "🚨 Deteksi Pump & Dump",
        "🛡️ Spoofing & Layering Order Book",
        "⏰ Marking the Close (Intraday)",
        "🔄 Deteksi Wash Trading",
        "📖 Profil & Sifat Broker (Smart Money Guide)"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: DETEKSI AKUMULASI INSTITUSI SENYAP
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("### 🕵️ Deteksi Akumulasi Institusi Senyap pada Saham Stagnan")
        st.caption("Mendeteksi fase konsolidasi sempit (< 2% dalam 10-14 hari) yang diiringi pengumpulan barang diam-diam oleh broker institusi.")

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            window_days = st.slider("Rentang Hari Konsolidasi:", min_value=7, max_value=21, value=14, step=1)
        with col_cfg2:
            vol_threshold = st.slider("Batas Maksimal Volatilitas (%):", min_value=1.0, max_value=5.0, value=2.0, step=0.5) / 100.0

        silent_res = detect_silent_institutional_accumulation(
            df_ohlcv=df_ohlcv,
            broker_summary_df=broker_summary_df,
            window=window_days,
            volatility_thresh=vol_threshold
        )

        # Kartu Metrik
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Status Fase", "KONSOLIDASI" if silent_res["is_stagnant"] else "TREN NORMAL", f"Volatilitas {silent_res['volatility_pct']}%")
        with m2:
            st.metric("Skor Akumulasi", f"{silent_res['accumulation_score']}/100", silent_res["status"].split()[0])
        with m3:
            st.metric("Net Buy Institusi", f"{silent_res['inst_net_buy_lots']:,} Lot")
        with m4:
            st.metric("Net Sell Ritel", f"{silent_res['retail_net_sell_lots']:,} Lot")

        # Visualisasi Plotly Area Konsolidasi & Akumulasi
        if df_ohlcv is not None and not df_ohlcv.empty:
            sub_ohlcv = df_ohlcv.tail(max(window_days + 15, 30))
            low_box, high_box = silent_res["consolidation_range"]

            fig_silent = go.Figure()
            # Candlestick
            fig_silent.add_trace(go.Candlestick(
                x=sub_ohlcv.index,
                open=sub_ohlcv["Open"],
                high=sub_ohlcv["High"],
                low=sub_ohlcv["Low"],
                close=sub_ohlcv["Close"],
                name="Candlestick",
                increasing_line_color="#00ffcc",
                decreasing_line_color="#ff0055"
            ))

            # Shaded box area konsolidasi institusi
            if silent_res["is_stagnant"]:
                start_date = sub_ohlcv.index[-window_days]
                end_date = sub_ohlcv.index[-1]
                fig_silent.add_shape(
                    type="rect",
                    x0=start_date,
                    y0=low_box,
                    x1=end_date,
                    y1=high_box,
                    fillcolor="rgba(0, 255, 204, 0.2)",
                    line=dict(color="#00ffcc", width=2, dash="dash"),
                    name="Zona Akumulasi Institusi"
                )
                fig_silent.add_annotation(
                    x=end_date,
                    y=high_box,
                    text=f"📦 Zona Konsolidasi Institusi (Rp {low_box:,.0f} - {high_box:,.0f})",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="#00ffcc",
                    font=dict(color="#00ffcc", size=11)
                )

            fig_silent.update_layout(
                title=f"Grafik Candlestick {ticker} & Sorotan Zona Akumulasi Institusi",
                template="plotly_dark",
                height=450,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_silent, use_container_width=True)

        # Tabel Top Broker dengan Metadata Nama Resmi & Implikasi Arah Harga
        c_inst, c_ret = st.columns(2)
        with c_inst:
            st.markdown("#### 🏛️ Top 5 Net Buy Broker Institusi (Smart Money)")
            if silent_res["top_institutional_buyers"]:
                st.dataframe(
                    pd.DataFrame(silent_res["top_institutional_buyers"]),
                    column_order=["broker_code", "broker_name", "category", "net_vol", "buy_vol", "sell_vol", "future_price_impact"],
                    column_config={
                        "broker_code": st.column_config.TextColumn("Kode", width="small"),
                        "broker_name": st.column_config.TextColumn("Nama Resmi Sekuritas", width="medium"),
                        "category": st.column_config.TextColumn("Kategori", width="small"),
                        "net_vol": st.column_config.NumberColumn("Net Lot", format="%d"),
                        "buy_vol": st.column_config.NumberColumn("Beli (Lot)", format="%d"),
                        "sell_vol": st.column_config.NumberColumn("Jual (Lot)", format="%d"),
                        "future_price_impact": st.column_config.TextColumn("Implikasi Arah Harga Masa Depan", width="large"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Tidak ada transaksi net buy institusi yang dominan.")
        with c_ret:
            st.markdown("#### 👥 Top 5 Net Sell Broker Ritel (Kerumunan)")
            if silent_res["top_retail_sellers"]:
                st.dataframe(
                    pd.DataFrame(silent_res["top_retail_sellers"]),
                    column_order=["broker_code", "broker_name", "category", "net_vol", "buy_vol", "sell_vol", "future_price_impact"],
                    column_config={
                        "broker_code": st.column_config.TextColumn("Kode", width="small"),
                        "broker_name": st.column_config.TextColumn("Nama Resmi Sekuritas", width="medium"),
                        "category": st.column_config.TextColumn("Kategori", width="small"),
                        "net_vol": st.column_config.NumberColumn("Net Lot", format="%d"),
                        "buy_vol": st.column_config.NumberColumn("Beli (Lot)", format="%d"),
                        "sell_vol": st.column_config.NumberColumn("Jual (Lot)", format="%d"),
                        "future_price_impact": st.column_config.TextColumn("Implikasi Arah Harga Masa Depan", width="large"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Tidak ada distribusi ritel yang dominan.")

        # Kartu Penjelasan Sifat Broker & Analisis Arah Harga Masa Depan
        with st.expander("💡 Penjelasan Sifat Broker & Analisis Arah Harga Masa Depan (Smart Money vs Kerumunan)", expanded=True):
            st.markdown("##### 📌 Karakteristik & Prediksi Arah Harga Broker Teratas:")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**🏛️ Institusi Pembeli Teratas (Smart Money):**")
                for item in silent_res["top_institutional_buyers"]:
                    b_inf = get_broker_info(item["broker_code"])
                    st.info(
                        f"**{item['broker_code']} — {b_inf['name']}**\n\n"
                        f"• **Kategori / Peran**: {b_inf['category']} ({b_inf['archetype']})\n"
                        f"• **Sifat Transaksi**: {b_inf['behavior']}\n"
                        f"• **Proyeksi Arah Harga**: **{b_inf['future_price_impact']}**"
                    )
            with col_b2:
                st.markdown("**👥 Ritel / Penjual Teratas (Kerumunan):**")
                for item in silent_res["top_retail_sellers"]:
                    b_inf = get_broker_info(item["broker_code"])
                    st.warning(
                        f"**{item['broker_code']} — {b_inf['name']}**\n\n"
                        f"• **Kategori / Peran**: {b_inf['category']} ({b_inf['archetype']})\n"
                        f"• **Sifat Transaksi**: {b_inf['behavior']}\n"
                        f"• **Proyeksi Arah Harga**: **{b_inf['future_price_impact']}**"
                    )

    # --------------------------------------------------------------------------
    # TAB 2: PREDIKSI 3 PILAR (TEKNIKAL + HOLT-WINTERS, NLP VADER, ORDER FLOW)
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("### 🎯 Prediksi Arah Pergerakan Jangka Pendek (Intraday s/d H+1)")
        st.caption("Integrasi 3 pilar: Teknikal Momentum & Holt-Winters, Analisis Sentimen NLP VADER, dan Order Flow Broker Summary.")

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            w_tech = st.slider("Bobot Pilar Teknikal (%):", 10, 70, 45, 5) / 100.0
        with col_w2:
            w_sent = st.slider("Bobot Pilar Sentimen (%):", 10, 50, 25, 5) / 100.0
        with col_w3:
            w_of = st.slider("Bobot Order Flow (%):", 10, 60, 30, 5) / 100.0

        # Normalisasi bobot
        tot_w = w_tech + w_sent + w_of
        w_tech, w_sent, w_of = w_tech / tot_w, w_sent / tot_w, w_of / tot_w

        pred_res = predict_short_term_trajectory(
            df_ohlcv=df_ohlcv,
            broker_summary_df=broker_summary_df,
            news_titles=news_titles or [],
            weights=(w_tech, w_sent, w_of)
        )

        if pred_res.get("status") == "SUCCESS":
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Arah Prediksi", pred_res["direction"].split()[0], pred_res["action"])
            with p2:
                st.metric("Skor Komposit", f"{pred_res['composite_score']:+.3f}")
            with p3:
                st.metric("Target Take Profit", f"Rp {pred_res['recommendation']['target_price']:,.0f}", f"+{pred_res['recommendation']['potential_gain_pct']}%")
            with p4:
                st.metric("Batas Cut Loss", f"Rp {pred_res['recommendation']['stop_loss']:,.0f}", f"-{pred_res['recommendation']['risk_pct']}%")

            # Candlestick Interaktif dengan Sinyal & Proyeksi Holt-Winters
            sub_df = df_ohlcv.tail(40).copy()
            fig_pred = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])

            fig_pred.add_trace(go.Candlestick(
                x=sub_df.index,
                open=sub_df["Open"],
                high=sub_df["High"],
                low=sub_df["Low"],
                close=sub_df["Close"],
                name="Candlestick",
                increasing_line_color="#00ffcc",
                decreasing_line_color="#ff0055"
            ), row=1, col=1)

            # EMA 9 & EMA 20
            ema9_s = sub_df["Close"].ewm(span=9, adjust=False).mean()
            ema20_s = sub_df["Close"].ewm(span=20, adjust=False).mean()
            fig_pred.add_trace(go.Scatter(x=sub_df.index, y=ema9_s, line=dict(color="#ffea00", width=1.5), name="EMA 9"), row=1, col=1)
            fig_pred.add_trace(go.Scatter(x=sub_df.index, y=ema20_s, line=dict(color="#00b0ff", width=1.5), name="EMA 20"), row=1, col=1)

            # Proyeksi Holt-Winters
            hw_forecast = pred_res["tech_pilar"]["holt_winters_forecast"]
            if hw_forecast:
                last_dt = sub_df.index[-1]
                future_dates = [last_dt + timedelta(days=i+1) for i in range(len(hw_forecast))]
                forecast_x = [last_dt] + future_dates
                forecast_y = [float(sub_df["Close"].iloc[-1])] + hw_forecast

                fig_pred.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    line=dict(color="#ff00e5", width=2.5, dash="dashdot"),
                    name="Proyeksi Holt-Winters (H+1 s/d H+3)"
                ), row=1, col=1)

            # Volume Bar
            fig_pred.add_trace(go.Bar(
                x=sub_df.index,
                y=sub_df["Volume"],
                marker_color="rgba(0, 176, 255, 0.4)",
                name="Volume"
            ), row=2, col=1)

            fig_pred.update_layout(
                title=f"Prediksi 3 Pilar {ticker} (Teknikal Holt-Winters, Sentimen VADER, Order Flow)",
                template="plotly_dark",
                height=520,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Rincian 3 Pilar
            st.markdown("#### 📋 Rincian Evaluasi 3 Pilar Pengambilan Keputusan:")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown("**1. Pilar Teknikal & Holt-Winters**")
                st.write(f"• EMA 9 / EMA 20: **{pred_res['tech_pilar']['ema9']} / {pred_res['tech_pilar']['ema20']}**")
                st.write(f"• RSI (14): **{pred_res['tech_pilar']['rsi']}**")
                st.write(f"• MACD: **{pred_res['tech_pilar']['macd']}** (Signal: {pred_res['tech_pilar']['macd_signal']})")
                st.write(f"• Skor Teknikal: **{pred_res['tech_pilar']['tech_score']}**")
            with col_p2:
                st.markdown("**2. Pilar Sentimen NLP VADER**")
                sent = pred_res["sentiment_pilar"]
                st.write(f"• Label Sentimen: **{sent['sentiment_label']}**")
                st.write(f"• Compound Score: **{sent['compound']}**")
                st.write(f"• Positif: **{sent['pos']}** | Negatif: **{sent['neg']}**")
            with col_p3:
                st.markdown("**3. Pilar Order Flow (Broker Summary)**")
                of = pred_res["order_flow_pilar"]
                st.write(f"• Top 10 Net Buy: **{of['top10_buy_lots']:,} Lot**")
                st.write(f"• Top 10 Net Sell: **{of['top10_sell_lots']:,} Lot**")
                st.write(f"• Net Order Flow: **{of['net_order_flow_lots']:,} Lot**")
                st.write(f"• Skor Order Flow: **{of['order_flow_score']}**")
                if "lead_buyer" in of:
                    l_b = of["lead_buyer"]
                    st.write(f"• Broker Akumulasi Utama: **{l_b['code']} — {l_b['name']}** ({l_b['category']})")
                    st.caption(f"  └ *Proyeksi*: {l_b['future_price_impact']}")

    # --------------------------------------------------------------------------
    # TAB 3: DETEKSI PUMP AND DUMP
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("### 🚨 Deteksi Indikasi 'Pump and Dump' (Harga, Volume, & Sentimen)")
        st.caption("Mendeteksi pola lonjakan harga > 10% dan volume > 300% setelah fase stagnan tanpa didukung sentimen berita fundamental.")

        alerts = detect_pump_and_dump(df_ohlcv=df_ohlcv, ticker=ticker)

        if alerts:
            st.error(f"⚠️ Terdeteksi **{len(alerts)}** indikasi anomali Pump & Dump pada riwayat emiten **{ticker}**!")
            df_alert = pd.DataFrame(alerts)
            st.dataframe(
                df_alert,
                column_config={
                    "ticker": "Ticker",
                    "date": "Tanggal Kejadian",
                    "price_surge_pct": st.column_config.NumberColumn("Lonjakan Harga", format="+%.1f%%"),
                    "volume_surge_pct": st.column_config.NumberColumn("Lonjakan Volume (vs SMA20)", format="%.0f%%"),
                    "stagnant_range_pct": st.column_config.NumberColumn("Rentang Stagnan 14H", format="%.2f%%"),
                    "sentiment_score": "Skor Sentimen VADER",
                    "has_fundamental_news": "Berita Fundamental",
                    "severity": "Status Peringatan",
                    "recommendation": "Rekomendasi Aksi",
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(f"✅ Tidak terdeteksi anomali Pump and Dump agresif pada {ticker}. Pergerakan harga dan volume dalam batas wajar.")

        with st.expander("💡 Peran & Karakteristik Broker dalam Skema Pump & Dump", expanded=False):
            st.markdown(
                "• **Broker Penggerak Pump (Bandar Kilat / Scalper)**: Kerap dipicu oleh broker agresif seperti **MG (PT Semesta Indovest Sekuritas)**, **AZ (PT Sucor Sekuritas)**, atau **CP (PT KB Valbury Sekuritas)** yang memompa likuiditas dan menyapu antrean offer (HAKA) di awal sesi.\n"
                "• **Broker Penerima Dump (Kerumunan Ritel)**: Ritel pengguna broker **YP (Mirae Asset Sekuritas)**, **PD (Indo Premier Sekuritas)**, **XC (Ajaib Sekuritas)**, dan **XL (Stockbit Sekuritas)** kerap menjadi pihak yang menyerap barang di pucuk (*FOMO*) karena tergiur rumor media sosial.\n"
                "• **Proyeksi Arah Harga**: Saham yang mengalami Pump tanpa katalis fundamental hampir selalu diikuti oleh **DISTRIBUSI MASIF & PENURUNAN HARGA TAJAM (Anjlok)** dalam 1 hingga 3 hari bursa berikutnya."
            )

    # --------------------------------------------------------------------------
    # TAB 4: SPOOFING DAN LAYERING ORDER BOOK LEVEL 2
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("### 🛡️ Analisis Order Book Level 2: Deteksi 'Spoofing dan Layering'")
        st.caption("Mendeteksi antrean palsu berukuran raksasa (5x rata-rata) yang dibatalkan kilat (< 5 detik) sebelum tereksekusi dengan rasio cancel-to-trade > 80%.")

        spoof_res = detect_spoofing_and_layering(order_book_l2_df)

        sp1, sp2, sp3, sp4 = st.columns(4)
        with sp1:
            st.metric("Tingkat Risiko Spoofing", spoof_res["overall_spoof_risk"].split()[0], spoof_res["overall_spoof_risk"])
        with sp2:
            st.metric("Total Event Spoofing", f"{spoof_res['total_spoof_events']} Kejadian")
        with sp3:
            st.metric("Rata-rata Bid Top 5", f"{spoof_res['avg_top5_bid_volume']:,} Lot")
        with sp4:
            st.metric("Rata-rata Ask Top 5", f"{spoof_res['avg_top5_ask_volume']:,} Lot")

        # Visualisasi Grafik Waktu Pasang & Cabut Antrean Palsu
        timeline = spoof_res.get("timeline_events", [])
        if timeline:
            df_tl = pd.DataFrame(timeline)
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=df_tl["time"],
                y=df_tl["price"],
                mode="markers+text",
                marker=dict(
                    size=np.clip(df_tl["volume"] / 3000, 10, 30),
                    color=df_tl["color"],
                    symbol="diamond",
                    line=dict(width=1, color="white")
                ),
                text=df_tl["type"] + " (" + df_tl["volume"].astype(str) + " Lot)",
                textposition="top center",
                name="Antrean Order Book"
            ))

            fig_tl.update_layout(
                title="Peta Titik Waktu Pemasangan & Pembatalan Antrean Order Book (Spoofing Timeline)",
                template="plotly_dark",
                height=380,
                xaxis_title="Waktu Transaksi",
                yaxis_title="Tingkat Harga (IDR)",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_tl, use_container_width=True)

        if spoof_res["spoof_details"]:
            st.markdown("#### 🚨 Tabel Bukti Aktivitas Spoofing & Layering:")
            st.dataframe(pd.DataFrame(spoof_res["spoof_details"]), use_container_width=True, hide_index=True)

        with st.expander("💡 Peran & Karakteristik Broker dalam Spoofing & Layering", expanded=False):
            st.markdown(
                "• **Mekanisme Manipulasi Antrean**: Market maker memasang tebal antrean Bid palsu untuk menciptakan persepsi semu bahwa ada institusi besar yang siap menampung harga (*Artificial Demand*).\n"
                "• **Pembatalan Mendadak**: Begitu kerumunan ritel terpancing melakukan HAKA di atasnya, antrean raksasa tersebut dibatalkan kilat (< 5 detik) dan pelaku melakukan HAKI (buang barang) ke antrean ritel di bawahnya.\n"
                "• **Proyeksi Arah Harga**: Keberadaan spoofing menandakan **ARAH HARGA AKAN BERBALIK TURUN (False Breakout)** segera setelah ritel kehabisan daya beli."
            )

    # --------------------------------------------------------------------------
    # TAB 5: MARKING THE CLOSE (DATA INTRADAY 1 MENIT)
    # --------------------------------------------------------------------------
    with tab5:
        st.markdown("### ⏰ Deteksi 'Marking the Close' (Analisis Data Intraday 1 Menit)")
        st.caption("Mendeteksi penyimpangan harga penutupan di 16:00 WIB > 3% dari VWAP sesi 09:00-15:50 WIB yang terkonsentrasi pada 40% volume harian dalam 10 menit terakhir.")

        mtc_res = detect_marking_the_close(df_intraday_1m=intraday_1m_df, ticker=ticker)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Status Pre-Closing", mtc_res["status"].split()[0], mtc_res["status"])
        with mc2:
            st.metric("VWAP Sesi Normal (09:00-15:50)", f"Rp {mtc_res['vwap_normal_hours']:,.0f}")
        with mc3:
            st.metric("Harga Close (16:00)", f"Rp {mtc_res['closing_price_1600']:,.0f}", f"Deviasi {mtc_res['price_deviation_pct']:+.2f}%")
        with mc4:
            st.metric("Volume 10 Menit Terakhir", f"{mtc_res['vol_in_last_10m_pct']:.1f}%", f"{mtc_res['vol_pre_close_lots']:,} Lot")

        # Visualisasi Grafik Intraday 1 Menit & VWAP
        df_intra = mtc_res["intraday_df"]
        fig_mtc = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        
        fig_mtc.add_trace(go.Scatter(
            x=df_intra["dt"],
            y=df_intra["close"],
            line=dict(color="#00ffcc", width=2),
            name="Harga Intraday 1 Menit"
        ), row=1, col=1)

        # Garis Horizontal VWAP Normal
        fig_mtc.add_hline(
            y=mtc_res["vwap_normal_hours"],
            line_dash="dash",
            line_color="#ffea00",
            annotation_text=f"VWAP 09:00-15:50 (Rp {mtc_res['vwap_normal_hours']:,.0f})",
            row=1, col=1
        )

        # Volume Bar Intraday
        fig_mtc.add_trace(go.Bar(
            x=df_intra["dt"],
            y=df_intra["volume"],
            marker_color="rgba(0, 176, 255, 0.5)",
            name="Volume Intraday"
        ), row=2, col=1)

        fig_mtc.update_layout(
            title=f"Pergerakan Intraday 1 Menit & Sesi Pre-Closing 15:50-16:00 WIB ({ticker})",
            template="plotly_dark",
            height=420,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_mtc, use_container_width=True)

        with st.expander("💡 Peran & Karakteristik Broker dalam Marking the Close", expanded=False):
            st.markdown(
                "• **Motif Penutupan Sesi (Pre-Closing 15:50 - 16:00 WIB)**: Broker institusi atau manajer investasi kerap mengeksekusi order masif di sesi pre-closing untuk mempercantik nilai portofolio (*Window Dressing*) menjelang akhir bulan/kuartal.\n"
                "• **Proyeksi Arah Harga Masa Depan**: Jika harga penutupan diangkat/diturunkan tanpa volume transaksi yang organik sepanjang sesi siang, harga hampir pasti akan **REVERT (KOREKSI KEMBALI KE HARGA VWAP)** pada pembukaan sesi perdagangan esok hari (H+1)."
            )

    # --------------------------------------------------------------------------
    # TAB 6: DETEKSI WASH TRADING
    # --------------------------------------------------------------------------
    with tab6:
        st.markdown("### 🔄 Deteksi 'Wash Trading' (Analisis Broker Summary)")
        st.caption("Mendeteksi transaksi putar broker yang mendominasi >= 30% transaksi harian namun nilai net-nya mendekati nol (< 5% dari nilai kotor).")

        wash_res = detect_wash_trading(broker_summary_df)

        w1, w2, w3 = st.columns(3)
        with w1:
            st.metric("Status Wash Trading", wash_res["verdict"].split()[0], wash_res["verdict"])
        with w2:
            st.metric("Total Broker Teranalisis", f"{wash_res['total_analyzed_brokers']} Broker")
        with w3:
            st.metric("Broker Terflag Transaksi Putar", f"{len(wash_res['flagged_brokers'])} Broker")

        if wash_res["flagged_brokers"]:
            st.error("⚠️ Terdeteksi kode broker dengan pola transaksi putar (Wash Trading):")
            st.dataframe(pd.DataFrame(wash_res["flagged_brokers"]), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Tidak terdeteksi anomali Wash Trading pada broker summary saham ini. Distribusi transaksi wajar.")

        st.markdown("#### 📊 Rekapitulasi Broker Summary Lengkap:")
        st.dataframe(
            pd.DataFrame(wash_res["broker_summary_table"]),
            column_order=[
                "broker_code",
                "broker_name",
                "category",
                "buy_vol",
                "sell_vol",
                "gross_vol",
                "net_vol",
                "dominance_pct",
                "net_to_gross_pct",
                "future_price_impact"
            ],
            column_config={
                "broker_code": st.column_config.TextColumn("Kode", width="small"),
                "broker_name": st.column_config.TextColumn("Nama Resmi Sekuritas", width="medium"),
                "category": st.column_config.TextColumn("Kategori", width="small"),
                "buy_vol": st.column_config.NumberColumn("Volume Beli (Lot)", format="%d"),
                "sell_vol": st.column_config.NumberColumn("Volume Jual (Lot)", format="%d"),
                "gross_vol": st.column_config.NumberColumn("Total Kotor (Lot)", format="%d"),
                "net_vol": st.column_config.NumberColumn("Net Volume (Lot)", format="%d"),
                "dominance_pct": st.column_config.NumberColumn("Dominasi Pasar", format="%.2f%%"),
                "net_to_gross_pct": st.column_config.NumberColumn("Rasio Net/Gross", format="%.2f%%"),
                "future_price_impact": st.column_config.TextColumn("Implikasi Arah Pergerakan Harga Masa Depan", width="large"),
            },
            use_container_width=True,
            hide_index=True
        )

        with st.expander("💡 Penjelasan Karakteristik Broker & Proyeksi Pasar dari Tabel Broker Summary", expanded=True):
            st.markdown("##### 📌 Rangkuman Sifat Broker & Proyeksi Arah Harga Saham:")
            top_active = wash_res["broker_summary_table"][:6]
            for row in top_active:
                b_info = get_broker_info(row["broker_code"])
                st.info(
                    f"**{row['broker_code']} — {b_info['name']}** ({b_info['category']})\n\n"
                    f"• **Peran Pasar**: {b_info['archetype']}\n"
                    f"• **Dominasi Transaksi**: {row.get('dominance_pct', 0):.2f}% dari total volume bursa\n"
                    f"• **Sifat & Karakter Transaksi**: {b_info['behavior']}\n"
                    f"• **Proyeksi Arah Harga Masa Depan**: **{b_info['future_price_impact']}**"
                )

    # --------------------------------------------------------------------------
    # TAB 7: PROFIL & SIFAT BROKER (SMART MONEY GUIDE)
    # --------------------------------------------------------------------------
    with tab7:
        st.markdown("### 📖 Profil & Sifat Broker (Panduan Smart Money & Prediksi Arah Harga)")
        st.markdown(
            "Panduan komprehensif mengenai **sifat dan karakteristik broker di Bursa Efek Indonesia (BEI)**. "
            "Memahami siapa yang berada di balik transaksi memungkinkan investor mengantisipasi fase pasar (*Akumulasi*, *Markup*, *Distribusi*, atau *Markdown*) "
            "dan memproyeksikan ke mana arah pergerakan harga saham di masa depan."
        )
        
        # 1. Empat Karakter Utama Pelaku Pasar BEI
        st.markdown("#### 🧭 4 Karakter Utama Pelaku Pasar (Archetypes) di BEI:")
        c_arch1, c_arch2, c_arch3, c_arch4 = st.columns(4)
        with c_arch1:
            st.info(
                "**🏛️ Smart Money Asing**\n\n"
                "*(BK, CS, ZP, RX, AK, KZ, DP)*\n\n"
                "• **Sifat**: Akumulasi senyap, modal masif, orientasi jangka menengah-panjang.\n"
                "• **Arah Harga**: **BULLISH KUAT**. Net buy konsisten mengawali fase kenaikan harga berkelanjutan (*Markup*)."
            )
        with c_arch2:
            st.success(
                "**🏦 BUMN & Sovereign Fund**\n\n"
                "*(CC, NI, OD, DX)*\n\n"
                "• **Sifat**: Dana pensiun & asuransi negara, penstabil pasar saat krisis.\n"
                "• **Arah Harga**: **BOTTOM REVERSAL**. Pembelian masif di harga murah menandakan lantai support terkuat."
            )
        with c_arch3:
            st.warning(
                "**👥 Kerumunan Ritel**\n\n"
                "*(YP, PD, XC, KK, XL)*\n\n"
                "• **Sifat**: Cepat FOMO, sangat reaktif terhadap berita & medsos, holding singkat.\n"
                "• **Arah Harga**: **BEARISH TRAP**. Jika ritel memborong di pucuk saat asing jualan, harga rawan diguyur (distribusi)."
            )
        with c_arch4:
            st.error(
                "**⚡ Bandar Kilat / Scalper**\n\n"
                "*(MG, AZ, CP)*\n\n"
                "• **Sifat**: Memompa harga cepat di pagi hari lalu guyur di siang/besok hari.\n"
                "• **Arah Harga**: **VOLATILITAS EKSTREM**. HANYA untuk scalping kilat berdisiplin tinggi, sangat bahaya di-hold."
            )

        st.markdown("---")

        # 2. Matriks Pengambilan Keputusan Arah Harga
        st.markdown("#### 🎯 Matriks Sinyal Proyeksi Arah Harga Masa Depan:")
        matrix_data = [
            {
                "Kondisi Transaksi Broker": "🏛️ Asing/Institusi NET BUY + 👥 Ritel NET SELL",
                "Fase Pasar": "Akumulasi Smart Money",
                "Arah Harga Masa Depan": "🟢 SANGAT BULLISH (Harga Siap Naik)",
                "Strategi Tindakan": "Ikut akumulasi / Buy on Weakness dan pasang target hold menengah."
            },
            {
                "Kondisi Transaksi Broker": "👥 Ritel NET BUY MASIF + 🏛️ Asing/Institusi NET SELL",
                "Fase Pasar": "Distribusi ke Ritel (Jebakan Pucuk)",
                "Arah Harga Masa Depan": "🔴 SANGAT BEARISH (Rawan Guyuran / Anjlok)",
                "Strategi Tindakan": "Ambil Take Profit segera atau hindari masuk; risiko nyangkut di pucuk sangat tinggi."
            },
            {
                "Kondisi Transaksi Broker": "⚡ Bandar Kilat (MG) NET BUY Besar di Awal Hari",
                "Fase Pasar": "Fast Momentum Pump",
                "Arah Harga Masa Depan": "🟡 VOLATIL (Naik Tajam Intraday, Guyuran H+1)",
                "Strategi Tindakan": "Hanya ikuti untuk Scalping hit-and-run cepat; jangan menginapkan posisi."
            },
            {
                "Kondisi Transaksi Broker": "🏦 BUMN (CC, NI, OD) NET BUY Saat IHSG / Saham Crash",
                "Fase Pasar": "Stabilisasi Pasar (National Defense)",
                "Arah Harga Masa Depan": "🟢 REVERSAL (Pembentukan Titik Terendah / Bottom)",
                "Strategi Tindakan": "Mulai cicil beli bertahap pada saham BUMN/Blue Chip berfundamental unggul."
            },
            {
                "Kondisi Transaksi Broker": "🔄 1 Broker Gross Masif (>= 30%), tapi Net ~ 0 (< 5%)",
                "Fase Pasar": "Wash Trading / Transaksi Semu",
                "Arah Harga Masa Depan": "⚪ ILUSI LIKUIDITAS (Arah Semu)",
                "Strategi Tindakan": "Waspada manipulasi volume semu; jangan terkecoh oleh antrean tebal palsu."
            }
        ]
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 3. Pencarian & Katalog Interaktif Seluruh Broker BEI
        st.markdown("#### 🔍 Katalog & Direktori Profil Broker BEI:")
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            cat_filter = st.selectbox(
                "Filter Berdasarkan Kategori Broker:",
                ["Semua Kategori", "🏛️ Asing / Smart Money", "🏦 BUMN / Sovereign Fund", "👥 Kerumunan Ritel", "⚡ Bandar Kilat / Scalper", "💼 Swasta & Terafiliasi"],
                index=0
            )
        with col_f2:
            search_kw = st.text_input("Cari Kode Broker atau Nama Perusahaan (contoh: ZP, YP, Mandiri, Mirae, J.P. Morgan):", "").strip().upper()

        all_brokers_list = list(IDX_BROKER_DIRECTORY.values())
        filtered_brokers = []
        for b in all_brokers_list:
            if cat_filter != "Semua Kategori":
                if "Asing" in cat_filter and "Asing" not in b["category"]:
                    continue
                elif "BUMN" in cat_filter and "BUMN" not in b["category"]:
                    continue
                elif "Ritel" in cat_filter and "Ritel" not in b["category"]:
                    continue
                elif "Bandar" in cat_filter and "Bandar" not in b["category"] and "Fast" not in b["archetype"]:
                    continue
                elif "Swasta" in cat_filter and "Swasta" not in b["category"] and "Terafiliasi" not in b["category"]:
                    continue

            if search_kw:
                if search_kw not in b["code"] and search_kw not in b["name"].upper():
                    continue

            filtered_brokers.append({
                "Kode": b["code"],
                "Nama Resmi Perusahaan": b["name"],
                "Kategori": b["category"],
                "Arketipe": b["archetype"],
                "Sifat & Karakter Transaksi": b["behavior"],
                "Implikasi Arah Harga Masa Depan": b["future_price_impact"]
            })

        st.dataframe(
            pd.DataFrame(filtered_brokers),
            column_config={
                "Kode": st.column_config.TextColumn("Kode", width="small"),
                "Nama Resmi Perusahaan": st.column_config.TextColumn("Nama Perusahaan Sekuritas", width="medium"),
                "Kategori": st.column_config.TextColumn("Kategori", width="medium"),
                "Arketipe": st.column_config.TextColumn("Peran Pasar", width="medium"),
                "Sifat & Karakter Transaksi": st.column_config.TextColumn("Karakteristik & Perilaku", width="large"),
                "Implikasi Arah Harga Masa Depan": st.column_config.TextColumn("Proyeksi Arah Harga", width="large"),
            },
            use_container_width=True,
            hide_index=True
        )
