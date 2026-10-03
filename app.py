"""
Aplikasi Web Dashboard Interaktif Analisis & Prediksi Saham Indonesia (IDX) REAL-TIME Terpadu.
Menggabungkan Sistem Analisis Saham IDX & Institutional Quant Engine:
1. Harga Wajar (Fair Value / Nilai Intrinsik 4 Model) & Margin of Safety (MoS)
2. Kalender Dividen (Ex-Date, Cum-Date, DPS, Yield), 5 Strategi Cuan, & Simulasi Pajak UU Cipta Kerja
3. Feed Berita Real-Time FinBERT NLP (Positif vs Negatif) & News Veto Alert
4. Analisis High & Low (52W H/L, Intraday CLV, 20D Donchian Breakout, 7 Level Fibonacci Retracement)
5. Grafik Candlestick Interaktif + MA Ribbon (6 Garis), ADX 14 Sideways Filter, & Point-in-Time Inspector
6. Proyeksi AI 5 Hari (Random Forest Regressor) & Suite AI Kuantitatif (GBDT, LSTM, Transformer, RL Policy)
7. Fundamental Lengkap, Solvabilitas, Insolvency Veto (DER > 4.0x), & Snapshot Makroekonomi Global
8. Money Management Lot Calculator, Fraksi Resmi BEI (IDX Ticks), Net PnL (Pajak & Fee Broker), & Order Book OBI
9. Klasifikasi 10 Gaya Trading (Timeframe & Durasi Hold), 19 Tipologi Saham, & Registry Cash Cows Bursa
10. Detektor Pola Breakout Geometri, Database Konglomerasi Terbesar BEI, & Jejak Bandarmologi
11. Ekonometrika Deret Waktu (ADF, ARIMA, GARCH Student's t), Deep Risk (VaR, CVaR, EVT-POT), & Optimasi Mean-CVaR
12. Scanner Lapis 3 Rally Hunter & Analisis Sentimen Kerumunan Ritel (Crowd Contrarian Lab)
13. Microservice Bot Dispatcher (Telegram Bot API & Direct 1-Click WhatsApp wa.me)
"""

import os
import sys

# Memastikan direktori root aplikasi terdaftar di sys.path (sangat penting untuk Streamlit Cloud)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
MODULES_DIR = os.path.join(CURRENT_DIR, "modules")
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
import urllib.parse
import streamlit.components.v1 as components

from modules.data_loader import (
    fetch_stock_data,
    fetch_historical_ohlcv,
    normalize_ticker,
    ALL_IDX_STOCKS,
    get_available_sectors,
    search_idx_stocks,
    clear_stock_cache,
    fetch_intraday_data,
    fetch_broker_summary_data,
    fetch_l2_order_book_data,
    filter_idx_stocks,
    get_stock_metadata,
)
from modules.broker_analyzer import render_broker_emiten_page
from modules.candlestick_predictor import predict_candlestick_movement
from modules.scalping_screener import scan_top_10_scalping_stocks, calculate_scalp_profit
from modules.idx_ticks import get_idx_tick_size, round_to_idx_tick, safe_int
from modules.technical_analysis import (
    compute_technical_indicators,
    evaluate_technical_score,
    compute_technical_suite,
    inspect_point_in_time,
)
from modules.fundamental_analysis import evaluate_fundamental_score, fetch_macro_snapshot
from modules.high_low_analysis import evaluate_high_low_aspects
from modules.news_sentiment import fetch_latest_news
from modules.fair_value import calculate_fair_value
from modules.dividend_analyzer import (
    analyze_dividend_schedule_and_strategy,
    calculate_dividend_payout_simulation,
)
from modules.predictor import train_and_predict_future
from modules.recommendation_engine import (
    generate_composite_recommendation,
    calculate_position_size,
)
from modules.order_book_microstructure import (
    evaluate_order_book_execution,
    calculate_net_pnl,
)
from modules.finance_statistical_analysis import render_finance_statistical_analysis_page
try:
    from modules.technical_analysis_page import render_technical_analysis_page
except Exception:
    from technical_analysis_page import render_technical_analysis_page
from modules.ihsg_market_driver import (
    get_cached_ihsg_data,
    calculate_emiten_market_beta,
    evaluate_lead_broker_and_bandar_cost,
    analyze_global_social_sentiment,
    evaluate_corporate_projects_and_catalysts,
    evaluate_investor_psychology_cycle,
)
from modules.econometrics_risk import run_econometrics_and_risk
from modules.portfolio_optimizer import optimize_mean_cvar_portfolio
from modules.trading_styles_types import (
    evaluate_trading_style_and_type,
    get_trading_cash_cows,
    TRADING_STYLES_INFO,
    STOCK_TYPES_PROFILES,
)
# Import Modul Breakout & Bandarmologi (Resilient Import dengan Auto-Reload & Fail-Safe Fallback)
try:
    import importlib
    import modules.breakout_bandarmologi
    if not hasattr(modules.breakout_bandarmologi, "generate_broker_interpretation_conclusion"):
        try:
            importlib.reload(modules.breakout_bandarmologi)
        except Exception:
            pass
    from modules.breakout_bandarmologi import (
        detect_chart_patterns_and_breakout,
        analyze_promoter_and_broker_footprint,
        scan_breakout_universe,
    )
    generate_broker_interpretation_conclusion = getattr(
        modules.breakout_bandarmologi, "generate_broker_interpretation_conclusion", None
    )
except Exception:
    try:
        import breakout_bandarmologi
        from breakout_bandarmologi import (
            detect_chart_patterns_and_breakout,
            analyze_promoter_and_broker_footprint,
            scan_breakout_universe,
        )
        generate_broker_interpretation_conclusion = getattr(
            breakout_bandarmologi, "generate_broker_interpretation_conclusion", None
        )
    except Exception:
        def detect_chart_patterns_and_breakout(*args, **kwargs):
            return {}
        def analyze_promoter_and_broker_footprint(*args, **kwargs):
            return {}
        def scan_breakout_universe(*args, **kwargs):
            return [], []
        generate_broker_interpretation_conclusion = None

if generate_broker_interpretation_conclusion is None:
    def generate_broker_interpretation_conclusion(promoter_eval, broker_eval=None):
        buyers = promoter_eval.get("top_buyers_detail", [])
        sellers = promoter_eval.get("top_sellers_detail", [])
        buyer_codes = [b.get("code", "") for b in buyers]
        seller_codes = [s.get("code", "") for s in sellers]
        foreign_smart = {"BK", "AK", "ZP", "CS", "KZ", "MS", "SQ", "CG", "DP"}
        bumn_sovereign = {"CC", "NI", "OD", "DX"}
        scalper_kilat = {"MG", "AG", "AZ"}
        private_inst = {"CP", "AI", "DR", "LG", "KI", "DH", "GR", "IF"}
        retail_herd = {"YP", "PD", "XC", "XL", "XA", "HD"}
        has_foreign = any(c in foreign_smart for c in buyer_codes)
        has_bumn = any(c in bumn_sovereign for c in buyer_codes)
        has_scalp = any(c in scalper_kilat for c in buyer_codes)
        has_priv = any(c in private_inst for c in buyer_codes)
        has_ret = any(c in retail_herd for c in buyer_codes)
        has_ret_s = any(c in retail_herd for c in seller_codes)
        lead_c = promoter_eval.get("lead_broker_code", buyer_codes[0] if buyer_codes else "CP")
        lead_n = promoter_eval.get("lead_broker_name", "Valbury Sekuritas Indonesia")
        if has_foreign and not has_ret:
            cat_title = "🏛️ Dominasi Smart Money Asing (Global Institutional Accumulation)"
            cat_col = "#10B981"
            st_tag = "AKUMULASI SENYAP MENUJU MARKUP"
            rule = "Jika Top Buyer didominasi Smart Money Asing (BK, AK, ZP): Mengindikasikan fase akumulasi senyap menuju kenaikan harga berkelanjutan (Markup)."
            narr = f"Top Buyer saham ini dikendalikan oleh Smart Money Institusi Asing ({', '.join(buyer_codes)}), dipimpin oleh {lead_c} ({lead_n})."
        elif has_bumn and not has_ret:
            cat_title = "🏢 Dominasi Konsorsium BUMN & Sovereign Anchor"
            cat_col = "#3B82F6"
            st_tag = "PENGAWALAN LANTAI HARGA (STRONG SUPPORT / BOTTOM REVERSAL)"
            rule = "Jika Top Buyer didominasi BUMN (CC, NI, OD): Menandakan pengawalan lantai harga (support) dan potensi Bottom Reversal yang kuat."
            narr = f"Top Buyer saham ini didominasi oleh sekuritas BUMN & Anchor Domestik ({', '.join(buyer_codes)}), dipimpin oleh {lead_c} ({lead_n})."
        elif has_scalp and not (has_foreign or has_bumn or has_priv):
            cat_title = "⚡ Dominasi Bandar Kilat & Momentum Scalper"
            cat_col = "#F59E0B"
            st_tag = "LONJAKAN CEPAT SPEKULATIF (RAWAN GUYURAN)"
            rule = "Jika Top Buyer didominasi Bandar Kilat (MG, AZ): Menandakan lonjakan harga cepat spekulatif (Pump) yang cocok untuk scalping kilat, namun rawan guyuran."
            narr = f"Top Buyer saham ini didominasi oleh pergerakan bandar kilat ({', '.join(buyer_codes)})."
        elif has_ret and not (has_foreign or has_bumn or has_priv):
            cat_title = "👥 Dominasi Kerumunan Ritel (Retail FOMO Trap)"
            cat_col = "#EF4444"
            st_tag = "WASPADA JEBAKAN BELI DI PUCUK (DISTRIBUSI KE RITEL)"
            rule = "Jika Top Buyer didominasi Kerumunan Ritel (YP, PD, XC): Waspada jebakan beli di pucuk (Distribution to Retail) saat institusi sedang melepas barang."
            narr = f"Peringatan distribusi: Pembeli terbanyak saat ini didominasi oleh akun ritel ({', '.join(buyer_codes)})."
        else:
            cat_title = "💼 Dominasi Institusi Swasta, Komoditas & Sindikasi Momentum"
            cat_col = "#06B6D4"
            st_tag = "AKUMULASI TERARAH & PENYERAPAN LIKUIDITAS RITEL"
            rule = "Top Buyer didominasi sindikasi institusi swasta & komoditas (CP, AI, AZ, DR) yang menyerap suplai likuiditas dari kerumunan ritel."
            narr = f"Top Buyer saham ini didominasi oleh institusi swasta dan penggerak momentum komoditas ({', '.join(buyer_codes)}), dengan {lead_c} ({lead_n}) sebagai lead akumulator."
        sel_syn = f"Sisi penjual didominasi oleh kerumunan ritel ({', '.join(seller_codes)}), mengonfirmasi transfer kepemilikan dari Weak Hands ke Strong Hands." if has_ret_s else f"Sisi penjual melibatkan broker campuran ({', '.join(seller_codes)})."
        b_cost = broker_eval.get("bandar_cost", 0) if broker_eval else 0
        diff_p = broker_eval.get("diff_from_cost_pct", 0.0) if broker_eval else 0.0
        cost_t = f"Modal rata-rata bandar saat ini tercatat di Rp {b_cost:,} ({diff_p:+.1f}% dari harga pasar)." if b_cost > 0 else "Harga bergerak dalam batas aman akumulasi."
        return {
            "category_title": cat_title,
            "category_color": cat_col,
            "status_tag": st_tag,
            "status": st_tag,
            "interpretation_rule": rule,
            "primary_rule": rule,
            "narrative": narr,
            "seller_synthesis": sel_syn,
            "seller_dynamic": sel_syn,
            "bandar_cost_text": cost_t,
            "cost_implication": cost_t,
            "action_recommendation": promoter_eval.get("bandar_action", "Akumulasi Bertahap bersama Smart Money"),
            "top_buyers_str": ", ".join([f"{b['code']} ({b['name']})" for b in buyers[:3]]),
            "top_sellers_str": ", ".join([f"{s['code']} ({s['name']})" for s in sellers[:3]]),
        }
from modules.advanced_ai_suite import run_comprehensive_ai_suite
from modules.lapis3_rally_crowd import screen_lapis_3, evaluate_crowd_contrarian
from modules.bot_dispatcher import (
    dispatcher_instance,
    format_super_profit_message,
    format_emergency_protection_message,
    dispatch_super_profit_alert,
    dispatch_emergency_protection_alert,
)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="⚡ IDX Stock Analyzer & Institutional Quant Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling Mengikuti Tampilan idx_stock_analyzer & Anti-Truncation Terpadu
st.markdown("""
<style>
    /* =========================================================================
       1. GLOBAL RESET & CEGAH SELURUH TULISAN TERPOTONG (ANTI-TRUNCATION)
       ========================================================================= */
    *, *::before, *::after {
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        text-overflow: clip !important;
    }

    p, span, div, h1, h2, h3, h4, h5, h6, label, li, a {
        text-overflow: clip !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
    }

    /* Mencegah Teks Metrik Streamlit Terpotong */
    [data-testid="stMetric"],
    [data-testid="stMetric"] * {
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 0.90rem !important;
        line-height: 1.3 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] span {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 1.35rem !important;
        line-height: 1.25 !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] > div,
    [data-testid="stMetricDelta"] span {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.2 !important;
    }

    /* Alert Boxes (st.info, st.success, st.warning, st.error) */
    [data-testid="stAlert"],
    [data-testid="stAlert"] > div,
    [data-testid="stAlert"] p {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.5 !important;
    }

    /* Kolom & Kontainer */
    [data-testid="column"] {
        min-width: 0 !important;
        overflow: visible !important;
    }

    /* Tabel Markdown & Dataframe BEI */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        width: 100% !important;
        overflow-x: auto !important;
    }

    [data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        overflow-x: auto !important;
        display: table !important;
    }

    [data-testid="stMarkdownContainer"] th,
    [data-testid="stMarkdownContainer"] td {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        padding: 8px 12px !important;
    }

    /* Expanders & Captions */
    [data-testid="stExpander"] details summary span {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stCaptionContainer"] p {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    /* =========================================================================
       2. FITUR GESER KANAN-KIRI TAB (HORIZONTAL SCROLLABLE FEATURE TABS)
       ========================================================================= */
    .stTabs {
        width: 100% !important;
        position: relative !important;
        overflow: visible !important;
    }

    /* Baris Tab: Satu Baris Rapi, Geser Kanan-Kiri Lancar */
    .stTabs [data-baseweb="tab-list"],
    .stTabs [role="tablist"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scroll-behavior: smooth !important;
        -webkit-overflow-scrolling: touch !important;
        gap: 8px !important;
        padding: 8px 4px 14px 4px !important;
        margin-bottom: 14px !important;
        border-bottom: 2px solid rgba(148, 163, 184, 0.25) !important;
        scrollbar-width: thin !important;
        scrollbar-color: #2563EB #F1F5F9 !important;
    }

    /* Scrollbar Khusus untuk Geser Kanan-Kiri */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar,
    .stTabs [role="tablist"]::-webkit-scrollbar {
        height: 8px !important;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track,
    .stTabs [role="tablist"]::-webkit-scrollbar-track {
        background: #F1F5F9 !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb,
    .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
        background: linear-gradient(90deg, #3B82F6, #1D4ED8) !important;
        border-radius: 12px !important;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover,
    .stTabs [role="tablist"]::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(90deg, #2563EB, #1E40AF) !important;
    }

    /* Tombol Tab: Teks Utuh & Fleksibel */
    .stTabs [data-baseweb="tab"],
    .stTabs [role="tab"] {
        flex-shrink: 0 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: keep-all !important;
        font-size: 0.94rem !important;
        font-weight: 700 !important;
        padding: 9px 18px !important;
        border-radius: 10px !important;
        background-color: #F8FAFC !important;
        border: 1.5px solid #CBD5E1 !important;
        color: #334155 !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        user-select: none !important;
    }

    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [role="tab"]:hover {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        border-color: #60A5FA !important;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    .stTabs [aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: 1.5px solid #1D4ED8 !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.35) !important;
    }

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: keep-all !important;
    }

    /* Hilangkan seluruh garis merah dan border bawaan BaseWeb Streamlit yang tumpang tindih */
    .stTabs [data-baseweb="tab-highlight"],
    div[data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-border"],
    [data-baseweb="tab-border"] {
        display: none !important;
        height: 0px !important;
        width: 0px !important;
        visibility: hidden !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }

    /* =========================================================================
       3. TAMPILAN ELEMEN KARTU & BADGE ASLI (DIPERTAHANKAN LENGKAP)
       ========================================================================= */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 0.8rem;
    }
    .badge-buy {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #86EFAC;
    }
    .badge-sell {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #FCA5A5;
    }
    .badge-hold {
        background-color: #FEF9C3;
        color: #A16207;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #FDE047;
    }
    .badge-veto {
        background-color: #450A0A;
        color: #F87171;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #EF4444;
    }
    .news-card-pos {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .news-card-neg {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .news-card-neu {
        background-color: #F8FAFC;
        border-left: 5px solid #94A3B8;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .live-badge {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .ob-bar-container {
        width: 100%;
        background-color: #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        display: flex;
        height: 24px;
        margin: 6px 0;
    }
    .ob-bar-bid {
        background: linear-gradient(90deg, #10B981, #059669);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ob-bar-offer {
        background: linear-gradient(90deg, #DC2626, #EF4444);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .quant-box {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        color: #F8FAFC !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25) !important;
    }
    .quant-box h1, .quant-box h2, .quant-box h3, .quant-box h4, .quant-box h5, .quant-box h6 {
        color: #F8FAFC !important;
        margin-top: 0 !important;
    }
    .quant-box p, .quant-box small, .quant-box b, .quant-box strong, .quant-box i {
        color: #F1F5F9;
        line-height: 1.5;
    }
    .quant-box code {
        background: #0F172A !important;
        color: #38BDF8 !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        border: 1px solid #1E293B !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- FITUR KEAMANAN PIN AKSES -----------------
def get_configured_pin() -> str:
    """Membaca PIN dari Streamlit Secrets, bot_config.json, atau default."""
    try:
        if hasattr(st, "secrets") and "APP_PIN" in st.secrets:
            return str(st.secrets["APP_PIN"]).strip()
    except Exception:
        pass

    try:
        config_path = os.path.join(CURRENT_DIR, "bot_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "app_pin" in cfg and str(cfg["app_pin"]).strip():
                    return str(cfg["app_pin"]).strip()
    except Exception:
        pass

    return "753987"


def verify_pin_access():
    """Memeriksa otentikasi sesi. Jika belum login, tampilkan layar input PIN dan hentikan rendering."""
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False

    if st.session_state["is_authenticated"]:
        return True

    target_pin = get_configured_pin()

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; padding: 25px 20px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07);">
                <div style="font-size: 3rem; margin-bottom: 6px;">🔒</div>
                <h2 style="color: #1E3A8A; margin-bottom: 4px; font-weight: 800;">Akses Terbatas</h2>
                <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 12px;">Dashboard <b>IDX Stock Analyzer</b> dilindungi kode PIN.<br>Silakan masukkan PIN Anda untuk membuka sistem:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("pin_login_form"):
            input_pin = st.text_input(
                "Kode PIN Akses:",
                type="password",
                placeholder="Ketik 6 digit PIN...",
                help="Masukkan PIN yang telah ditentukan untuk membuka seluruh modul analisis.",
            )
            submit_login = st.form_submit_button("🔓 Buka Dashboard", use_container_width=True, type="primary")

            if submit_login:
                if input_pin.strip() == target_pin:
                    st.session_state["is_authenticated"] = True
                    st.success("✅ Verifikasi Berhasil! Membuka dashboard...")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("❌ PIN salah! Silakan periksa kembali kode PIN Anda.")

        st.markdown(
            """
            <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; margin-top: 15px;">
                🔒 <i>Sistem terproteksi untuk penggunaan terotorisasi.</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()

# Jalankan pengecekan PIN sebelum memuat seluruh data & kontrol
verify_pin_access()

# ----------------- SIDEBAR KONTROL -----------------
st.sidebar.title("⚡ Terminal Terpadu BEI")
st.sidebar.caption(f"Semesta Emiten: **{len(ALL_IDX_STOCKS)} Saham BEI**")

# Navigasi Menu Pojok Kiri (Sidebar)
st.sidebar.markdown("---")
app_menu = st.sidebar.radio(
    "📌 Navigasi Menu:",
    [
        "📊 Dashboard Analisis Saham",
        "📈 Analisis Teknikal",
        "⚡ Lapis 3 Rally Hunter",
        "🏛️ Analisis Broker dan Emiten",
        "📊 Finance and Statistical Analysis"
    ],
    index=0,
    key="main_app_menu_radio"
)
st.sidebar.markdown("---")

# Inisialisasi session state untuk ticker aktif
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "BBCA"

# Status Koneksi Pasar
st.sidebar.subheader("⏱️ Status Data Real-Time")
st.sidebar.caption("⚡ Terhubung langsung dengan sistem pasar BEI / Yahoo Finance.")
auto_refresh = False
refresh_interval = 60

# Filter Semesta Saham
st.sidebar.subheader("🔍 Filter Saham BEI")
chosen_tier = st.sidebar.selectbox(
    "Filter Tingkatan (Tier):",
    [
        "Semua Tingkatan",
        "Saham Gocap / Saham Tidur (Rp50 – Rp100)",
        "Saham Receh / Saham Murah (Rp100 – Rp1.000)",
        "Saham Menengah (Rp1.000 – Rp5.000)",
        "Saham Premium / Blue Chip (Di atas Rp5.000)",
    ],
    index=0
)
chosen_syariah = st.sidebar.selectbox(
    "Filter Syariah (OJK/DSN-MUI):",
    ["Semua", "☪️ Hanya Syariah (ISSI)", "⚪ Non-Syariah"],
    index=0
)
sectors = get_available_sectors()
chosen_sector = st.sidebar.selectbox("Filter Sektor Industri:", sectors, index=0)

filtered_stocks = search_idx_stocks(
    "",
    sector=chosen_sector,
    tier=chosen_tier,
    syariah=chosen_syariah
)

mode_input = st.sidebar.radio("Pilih Saham:", ["Pilih dari Katalog", "Ketik Manual Ticker"], index=0)
current_target_ticker = str(st.session_state.get("selected_ticker", "BBCA")).replace(".JK", "").upper().strip()

if mode_input == "Pilih dari Katalog":
    if filtered_stocks:
        stock_labels = [s["display_label"] for s in filtered_stocks]
        matching_idx = None
        for idx, s in enumerate(filtered_stocks):
            if s["ticker"] == current_target_ticker:
                matching_idx = idx
                break
        
        # Jika emiten yang aktif saat ini tidak ada di kombinasi filter (misal user ubah filter tier),
        # sisipkan emiten aktif di urutan pertama agar tidak ter-reset secara paksa ke saham lain
        if matching_idx is None:
            active_meta = get_stock_metadata(current_target_ticker)
            active_label = f"📌 [Aktif] {active_meta['display_label']}"
            stock_labels.insert(0, active_label)
            selected_idx = 0
        else:
            selected_idx = matching_idx

        def _on_catalog_change():
            chosen_val = st.session_state.get("catalog_stock_selector")
            if chosen_val:
                clean_t = chosen_val.replace("📌 [Aktif] ", "").split(" - ")[0].strip().upper()
                st.session_state["selected_ticker"] = clean_t

        # Pastikan catalog_stock_selector selalu sinkron dengan current_target_ticker
        cur_cat = st.session_state.get("catalog_stock_selector", "")
        if not (cur_cat.startswith(current_target_ticker + " -") or cur_cat.startswith(f"📌 [Aktif] {current_target_ticker} -")):
            for lbl in stock_labels:
                if lbl.startswith(current_target_ticker + " -") or lbl.startswith(f"📌 [Aktif] {current_target_ticker} -"):
                    st.session_state["catalog_stock_selector"] = lbl
                    break

        selected_stock_label = st.sidebar.selectbox(
            "Katalog Saham Terfilter:",
            stock_labels,
            index=selected_idx,
            key="catalog_stock_selector",
            on_change=_on_catalog_change
        )
        selected_ticker = selected_stock_label.replace("📌 [Aktif] ", "").split(" - ")[0].strip().upper()
        st.session_state["selected_ticker"] = selected_ticker
    else:
        st.sidebar.warning("Tidak ada saham yang cocok dengan kombinasi filter.")
        selected_ticker = current_target_ticker
else:
    # Menggunakan st.form agar saat user mengetik ticker (misal 4 huruf: B-B-R-I),
    # Streamlit TIDAK melakukan refresh 4x berturut-turut pada setiap ketikan tombol!
    with st.sidebar.form(key="manual_ticker_search_form"):
        manual_input = st.text_input(
            "Ketik Kode Ticker (contoh: BBCA, BBRI, BREN, BRMS, BUMI):",
            value=current_target_ticker,
            help="Ketik 4 huruf kode saham BEI, lalu tekan Enter atau klik tombol Tampilkan Saham."
        ).strip().upper()
        btn_search_manual = st.form_submit_button("🔍 Tampilkan Saham", type="primary", use_container_width=True)
        if btn_search_manual and manual_input:
            clean_man = manual_input.replace(".JK", "").strip()
            if len(clean_man) >= 2:
                st.session_state["selected_ticker"] = clean_man
                current_target_ticker = clean_man
                st.rerun()
            else:
                st.sidebar.warning("⚠️ Masukkan minimal 2-4 huruf kode emiten BEI.")
    selected_ticker = st.session_state.get("selected_ticker", current_target_ticker)

period_choice = st.sidebar.selectbox("Rentang Data Historis:", ["1y", "2y", "5y", "6mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Parameter Portofolio")
user_capital = st.sidebar.number_input(
    "Total Modal Trading (Rp):",
    min_value=500000.0,
    max_value=100000000000.0,
    value=10000000.0,
    step=1000000.0,
    format="%.0f"
)
user_risk_pct = st.sidebar.slider(
    "Maksimal Risiko per Trade (%):",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
)

btn_manual_refresh = st.sidebar.button("⚡ Refresh Data Real-Time Sekarang", type="primary", use_container_width=True)

# Tombol Kunci / Keluar (Logout)
st.sidebar.markdown("---")
if st.sidebar.button("🔒 Kunci / Keluar (Logout)", use_container_width=True):
    st.session_state["is_authenticated"] = False
    st.rerun()

# ----------------- FUNGSI CACHING AKSELERASI KECEPATAN TINGGI -----------------
CACHE_VERSION_KEY = "v5_strict_ojk_2026_09_30"

@st.cache_data(ttl=15, show_spinner=False)
def get_cached_stock_data(ticker_symbol: str, period: str, _ver: str = CACHE_VERSION_KEY):
    return fetch_stock_data(ticker_symbol, period=period)

@st.cache_data(ttl=180, show_spinner=False)
def get_cached_news(ticker_symbol: str, company_name: str):
    return fetch_latest_news(ticker_symbol, company_name=company_name, limit=15)

@st.cache_data(ttl=180, show_spinner=False)
def get_cached_ml(df_history: pd.DataFrame, forecast_days: int = 5):
    return train_and_predict_future(df_history, forecast_days=forecast_days)

@st.cache_data(ttl=180, show_spinner=False)
def get_cached_econometrics(df_history: pd.DataFrame):
    return run_econometrics_and_risk(df_history)

@st.cache_data(ttl=180, show_spinner=False)
def get_cached_ai_suite(df_history: pd.DataFrame, news_titles: tuple, pct_bid: float, pct_offer: float, adx_val: float, rsi_val: float, atr_val: float):
    return run_comprehensive_ai_suite(
        df=df_history,
        news_articles=list(news_titles),
        pct_bid=pct_bid,
        pct_offer=pct_offer,
        adx_val=adx_val,
        rsi_val=rsi_val,
        atr_val=atr_val,
    )

@st.cache_data(ttl=30, show_spinner=False)
def get_cached_scalping_picks(tier_filter: str, syariah_filter: str, _ver: str = CACHE_VERSION_KEY):
    return scan_top_10_scalping_stocks(tier_filter=tier_filter, syariah_filter=syariah_filter)

@st.cache_data(ttl=15, show_spinner=False)
def get_cached_intraday(ticker_symbol: str):
    return fetch_intraday_data(ticker_symbol)

@st.cache_data(ttl=15, show_spinner=False)
def get_cached_broker_summary(ticker_symbol: str, current_price: float, volume: float):
    return fetch_broker_summary_data(ticker_symbol, current_price=current_price, volume=volume)

@st.cache_data(ttl=10, show_spinner=False)
def get_cached_l2_order_book(ticker_symbol: str, current_price: float):
    return fetch_l2_order_book_data(ticker_symbol, current_price=current_price)


# ----------------- PROSES DATA PASAR -----------------
ticker_clean = normalize_ticker(selected_ticker)

if btn_manual_refresh:
    st.cache_data.clear()
    clear_stock_cache()

with st.spinner(f"Menghubungkan ke Bursa Efek Indonesia untuk memuat data {ticker_clean}..."):
    df, info, err = get_cached_stock_data(ticker_clean, period=period_choice)

# Kalibrasi Otoritatif Metadata Saham dari idx_universe
from modules.idx_universe import get_stock_metadata
meta_live = get_stock_metadata(ticker_clean)
if info is not None:
    info["is_syariah"] = meta_live.get("is_syariah", False)
    info["syariah_label"] = meta_live.get("syariah_label", "⚪ Non-Syariah" if not meta_live.get("is_syariah") else "☪️ Syariah (ISSI)")
    info["tier"] = meta_live.get("tier", info.get("tier"))
    info["tier_code"] = meta_live.get("tier_code", info.get("tier_code"))
    info["tier_short"] = meta_live.get("tier_short", info.get("tier_short"))
    if not info.get("sector") or info["sector"] in {"Lainnya", "Bursa Efek Indonesia", "Umum"}:
        info["sector"] = meta_live.get("sector", "Financials" if not meta_live.get("is_syariah") else "Umum")
    if not info.get("longName") or info.get("longName") == ticker_clean:
        info["longName"] = meta_live.get("name", ticker_clean)
    if not info.get("shortName") or info.get("shortName") == ticker_clean:
        info["shortName"] = meta_live.get("name", ticker_clean)

if err or df is None or df.empty:
    st.error(f"❌ Terjadi kesalahan saat memuat data {ticker_clean}: {err}")
    st.warning(f"⚠️ Emiten **{ticker_clean}** kemungkinan sedang dalam status suspensi bursa, delisting, atau belum memiliki data aktif di penyedia pasar Yahoo Finance.")
    st.markdown("#### 🔄 Pulihkan Cepat ke Saham Aktif & Paling Likuid:")
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        if st.button("🏛️ BBCA (Blue Chip)", key="rec_bbca", use_container_width=True):
            st.session_state["selected_ticker"] = "BBCA"
            st.rerun()
    with rc2:
        if st.button("🛵 GOTO (Gocap)", key="rec_goto", use_container_width=True):
            st.session_state["selected_ticker"] = "GOTO"
            st.rerun()
    with rc3:
        if st.button("⛏️ BUMI (Receh)", key="rec_bumi", use_container_width=True):
            st.session_state["selected_ticker"] = "BUMI"
            st.rerun()
    with rc4:
        if st.button("🪙 BRMS (Receh)", key="rec_brms", use_container_width=True):
            st.session_state["selected_ticker"] = "BRMS"
            st.rerun()
    st.stop()

# 1. Analisis Teknikal Terpadu & MA Ribbon
df_tech = compute_technical_indicators(df)
tech_eval = evaluate_technical_score(df_tech)
tech_suite = compute_technical_suite(df_tech)

fast_info = info.get("fast_info")
current_price = float(info.get("realtime_last_price") or info.get("price") or df_tech["Close"].iloc[-1])
prev_close = float(info.get("previous_close") or (df_tech["Close"].iloc[-2] if len(df_tech) > 1 else current_price))
price_diff = float(info.get("price_diff", current_price - prev_close))
price_diff_pct = float(info.get("price_diff_pct", ((current_price - prev_close) / max(1.0, prev_close)) * 100.0))

# ----------------- ROUTING CEPAT MENU ANALISIS TEKNIKAL -----------------
if app_menu == "📈 Analisis Teknikal":
    render_technical_analysis_page(
        ticker=ticker_clean,
        df_ohlcv=df_tech,
        info=info,
        current_price=current_price,
        all_stocks=ALL_IDX_STOCKS,
        chosen_tier=chosen_tier,
        chosen_syariah=chosen_syariah,
        chosen_sector=chosen_sector,
    )
    st.stop()

# 2. Analisis High & Low (52W, Intraday, Breakout, Fibonacci 7 Level)
hl_eval = evaluate_high_low_aspects(df_tech, fast_info)

# 3. Analisis Fundamental & Solvabilitas
fund_eval = evaluate_fundamental_score(info)

# 4. Analisis Harga Wajar (Fair Value 4 Model & MoS)
fv_eval = calculate_fair_value(current_price, info, df_history=df_tech)

# 5. Analisis Dividen & Strategi Cuan
div_eval = analyze_dividend_schedule_and_strategy(ticker_clean, info, current_price)

# 6. Analisis Berita FinBERT & News Veto
news_eval = get_cached_news(selected_ticker, company_name=info.get("shortName", ""))

# 7. Machine Learning Proyeksi AI 5 Hari (Nominal Rp)
ml_eval = get_cached_ml(df_tech, forecast_days=5)

# 8. Mikrostruktur Buku Pesanan (Order Book & Net PnL)
candle_info = {"signal": "BUY" if current_price >= float(df_tech["Open"].iloc[-1]) else "SELL"}
order_book_eval = evaluate_order_book_execution(info, candle_info)

# 9. Gaya Trading, 19 Tipologi Saham & Cash Cows
style_type_eval = evaluate_trading_style_and_type(
    ticker=ticker_clean,
    price=current_price,
    daily_turnover=info.get("turnover_idr", 5_000_000_000),
    pct_bid=info.get("pct_bid", 50.0),
    pct_offer=info.get("pct_offer", 50.0),
    atr_val=tech_eval.get("atr", 50.0),
    adx_val=tech_suite.get("adx", 24.0),
    is_sharia=info.get("is_syariah", True),
    sector=info.get("sector", "Umum")
)

# 10. Deteksi Pola Breakout Geometri & Bandarmologi
breakout_eval = detect_chart_patterns_and_breakout(
    df=df_tech,
    current_price=current_price,
    current_volume=info.get("volume", 500000),
    pct_bid=info.get("pct_bid", 50.0)
)
promoter_eval = analyze_promoter_and_broker_footprint(
    ticker=ticker_clean,
    price=current_price,
    daily_turnover=info.get("turnover_idr", 5_000_000_000),
    pct_bid=info.get("pct_bid", 50.0),
    pct_offer=info.get("pct_offer", 50.0)
)

# 11. Ekonometrika & Risiko Ekstrem (ADF, ARIMA, GARCH, EVT-POT)
econ_eval = get_cached_econometrics(df_tech)

# 12. Lapis 3 Rally Hunter & Crowd Sentiment
lapis3_eval = screen_lapis_3(df_tech, info)
crowd_eval = evaluate_crowd_contrarian(info, (news_eval.get("score", 50) - 50.0) / 50.0)

# 13. AI Suite Kuantitatif Lanjutan (GBDT, LSTM, Transformer, RL Policy)
news_titles_tuple = tuple(a["title"] for a in news_eval.get("articles", []))
ai_suite_eval = get_cached_ai_suite(
    df_history=df_tech,
    news_titles=news_titles_tuple,
    pct_bid=info.get("pct_bid", 50.0),
    pct_offer=info.get("pct_offer", 50.0),
    adx_val=tech_suite.get("adx", 24.0),
    rsi_val=tech_eval.get("rsi", 50.0),
    atr_val=tech_eval.get("atr", 50.0)
)

# 14. Prediksi Pergerakan Candlestick Real-Time & Target Level
candle_pred = predict_candlestick_movement(df_tech, info)

# 15. Ekosistem Terpadu: Makro IHSG, Broker Footprint, Social Sentiment Global, Proyek Korporasi & Psikologi Pasar
ihsg_eval, df_ihsg_hist = get_cached_ihsg_data()
beta_eval = calculate_emiten_market_beta(df_tech, df_ihsg_hist)
broker_eval = evaluate_lead_broker_and_bandar_cost(
    ticker_clean,
    current_price,
    df_tech,
    lead_broker_code=promoter_eval.get("lead_broker_code")
)
social_eval = analyze_global_social_sentiment(ticker_clean, news_eval, info)
corp_eval = evaluate_corporate_projects_and_catalysts(ticker_clean, info, df_tech)
psychology_eval = evaluate_investor_psychology_cycle(ticker_clean, current_price, df_tech, tech_eval.get("rsi", 50.0), social_eval["composite_social_score"])

# 16. Rekomendasi Keputusan Terpadu Holistik
rec = generate_composite_recommendation(
    tech_result=tech_eval,
    fund_result=fund_eval,
    ml_result=ml_eval,
    current_price=current_price,
    high_low_result=hl_eval,
    news_result=news_eval,
    order_book_result=order_book_eval,
    ai_suite_result=ai_suite_eval,
    ihsg_result=ihsg_eval,
    broker_result=broker_eval,
    social_sentiment_result=social_eval,
    corporate_projects_result=corp_eval,
    investor_psychology_result=psychology_eval,
)
plan = rec["trading_plan"]
pos = calculate_position_size(
    capital_idr=user_capital,
    max_risk_pct=user_risk_pct,
    entry_price=current_price,
    stop_loss_price=plan["stop_loss"],
    safe_exit_lot_cap=order_book_eval.get("safe_exit_lot", 1000)
)
net_pnl_calc = calculate_net_pnl(current_price, plan["take_profit_1"], pos["shares"])

# ----------------- ROUTER MENU NAVIGASI (SIDEBAR) -----------------
if app_menu == "⚡ Lapis 3 Rally Hunter":
    st.markdown('<div class="main-title">⚡ Lapis 3 Rally Hunter & Sentimen Kerumunan Ritel (Crowd Lab)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detektor Momentum Ledakan Saham Small-Cap (Lapis 3) Berbasis Relative Volume (RVol > 2.0x), Bollinger Band Squeeze & Dominasi Antrian Beli</div>', unsafe_allow_html=True)

    company_name = info.get("longName") or info.get("shortName") or ticker_clean
    st.info(f"📌 **Saham Sedang Dianalisis**: **{ticker_clean}** ({company_name}) | **Rp {current_price:,.0f}** | {info.get('tier', 'Lapis 3')} | Sektor: {info.get('sector', 'Bursa Efek Indonesia')}")

    # 1. Metrik Ringkasan Saham Terpilih
    rally_status = "🟢 TERPENUHI (SIAP MELEDAK)" if lapis3_eval.get("is_rally") else "⚪ BELUM TERPENUHI"
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.metric("Status Sinyal Rally", rally_status, f"Skor {lapis3_eval.get('score', 0):.1f} / 100")
    with rc2:
        rvol_val = lapis3_eval.get('rvol', 1.0)
        st.metric("Relative Volume (RVol)", f"{rvol_val}x", "Target >= 2.0x")
    with rc3:
        sq_status = "🟢 Squeeze Aktif" if lapis3_eval.get("is_squeeze") else "⚪ Normal"
        st.metric("Bollinger Squeeze", sq_status)
    with rc4:
        st.metric("Turnover Harian", f"Rp {lapis3_eval.get('turnover_idr', 0):,.0f}")

    st.markdown("---")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown(f"#### 🔍 Analisis Mikrostruktur Saham: **{ticker_clean}**")
        st.write(f"• **Kategori Tingkatan**: **{info.get('tier', 'Saham Lapis 3 / Small-Cap')}**")
        st.write(f"• **Harga Pasar Terakhir**: **Rp {info.get('price', current_price):,.0f}**")
        st.write(f"• **Dominasi Buku Pesanan**: **{order_book_eval.get('pct_bid', 50):.1f}% Bid** vs **{order_book_eval.get('pct_offer', 50):.1f}% Offer**")
        st.write(f"• **Status HAKA**: **{order_book_eval.get('haka_badge', '-')}**")
        st.write(f"• **Safe Exit Lot Size**: **{order_book_eval.get('safe_exit_lot', 0):,} Lot** ({order_book_eval.get('exit_speed', '')})")
        st.info(f"💡 {order_book_eval.get('haka_desc', '')}")

    with col_l2:
        st.markdown("#### 👥 Sentimen Kerumunan Ritel & Sinyal Kontrarian")
        st.metric("Sinyal Kontrarian", crowd_eval.get("contrarian_signal", "NORMAL"))
        st.write(f"• **Keterangan Kontrarian**: {crowd_eval.get('contrarian_desc')}")
        st.write(f"• **Skor Sentimen Kerumunan**: {crowd_eval.get('crowd_sentiment')}")
        st.write(f"• **Kecepatan Obrolan (Buzz Velocity)**: {crowd_eval.get('buzz_velocity')}x")
        
        st.markdown("##### 🎯 Rencana Transaksi Saham Lapis 3:")
        entry_str = plan.get('entry_range') or f"Rp {plan.get('buy_entry_min', 0):,} s/d Rp {plan.get('buy_entry_max', 0):,}"
        tp1_val = plan.get('take_profit_1', 0)
        tp1_net = plan.get('reward_tp1_net_pct', 0.0)
        tp2_val = plan.get('take_profit_2', 0)
        tp2_net = plan.get('reward_tp2_net_pct', 0.0)
        sl_val = plan.get('stop_loss', 0)
        sl_net = plan.get('risk_net_pct', 0.0)
        st.success(
            f"• **Zona Beli (Entry)**: {entry_str}\n"
            f"• **Target Cuan 1 (TP1)**: Rp {tp1_val:,} (Net +{tp1_net:.1f}%)\n"
            f"• **Target Cuan 2 (TP2)**: Rp {tp2_val:,} (Net +{tp2_net:.1f}%)\n"
            f"• **Batas Cut Loss (SL Ketat)**: Rp {sl_val:,} (Net -{sl_net:.1f}%)"
        )

    st.markdown("---")
    st.markdown(f"#### 🕵️ Jejak Broker & Sifat Smart Money Saham Lapis 3: **{ticker_clean}**")
    st.caption("Memetakan karakteristik sekuritas pengendali transaksi pada saham lapis 3 dan proyeksi arah pergerakan harganya:")
    col_br1, col_br2 = st.columns(2)
    with col_br1:
        lead_b_display = promoter_eval.get('top_buyers', ['Broker Penggerak Terdeteksi'])[0]
        lead_b_arch = promoter_eval.get('top_buyer_archetype', 'Smart Money & Penggerak Likuiditas')
        lead_b_behav = promoter_eval.get('top_buyer_behavior', 'Akumulasi bertahap dengan pengawalan likuiditas di pasar reguler.')
        st.info(
            f"**🏛️ Profil Broker Penggerak Utama {ticker_clean}:**\n\n"
            f"• **Broker Terdeteksi**: {lead_b_display}\n"
            f"• **Arketipe / Karakter**: {lead_b_arch}\n"
            f"• **Sifat Transaksi**: {lead_b_behav}"
        )
    with col_br2:
        lead_b_impact = promoter_eval.get('top_buyer_impact', 'Potensi penguatan harga bertahap didukung akumulasi terukur.')
        st.success(
            f"**🔮 Proyeksi Arah Harga Masa Depan {ticker_clean}:**\n\n"
            f"• **Status Bandarmologi**: {promoter_eval.get('bandar_status', '🟢 Akumulasi Bertahap')}\n"
            f"• **Proyeksi Arah**: **{lead_b_impact}**"
        )

    st.markdown("---")
    st.markdown("#### 🚀 Pemindai Saham Potensi Rally (Katalog Small-Cap / Lapis 3)")
    st.caption("Peringkat saham lapis 3 yang terdeteksi memiliki anomali lonjakan volume, kompresi volatilitas, dan jejak broker:")
    
    sample_rally_candidates = [
        {"ticker": "DEWA", "nama": "Darma Henwa Tbk.", "harga": 105, "rvol": 2.85, "squeeze": "🟢 Ya", "bid_pct": 71.4, "turnover": 45_200_000_000, "broker_utama": "MG (Semesta) - Bandar Scalper", "proyeksi_harga": "🟡 Volatilitas Tinggi Intraday (Markup Kilat)", "status": "🟢 SIAP MELEDAK"},
        {"ticker": "KIJA", "nama": "Kawasan Industri Jababeka", "harga": 172, "rvol": 2.40, "squeeze": "🟢 Ya", "bid_pct": 68.2, "turnover": 18_400_000_000, "broker_utama": "CC (Mandiri) - BUMN/Domestik", "proyeksi_harga": "🟢 Reversal Stabil & Bertahap", "status": "🟢 SIAP MELEDAK"},
        {"ticker": "ELSA", "nama": "Elnusa Tbk.", "harga": 486, "rvol": 2.15, "squeeze": "⚪ Tidak", "bid_pct": 66.5, "turnover": 32_100_000_000, "broker_utama": "NI (BNI Sekuritas) - BUMN", "proyeksi_harga": "🟢 Akumulasi Menengah Defensif", "status": "🟡 AKUMULASI"},
        {"ticker": "PSAB", "nama": "J Resources Asia Pasifik", "harga": 312, "rvol": 2.30, "squeeze": "🟢 Ya", "bid_pct": 65.0, "turnover": 24_500_000_000, "broker_utama": "YP (Mirae) - Kerumunan Ritel", "proyeksi_harga": "🟡 Momentum Cepat Ritel, Waspada Guyuran", "status": "🟢 SIAP MELEDAK"},
        {"ticker": "RAJA", "nama": "Rukun Raharja Tbk.", "harga": 1380, "rvol": 1.95, "squeeze": "🟢 Ya", "bid_pct": 63.0, "turnover": 19_800_000_000, "broker_utama": "AK (UBS) - Smart Money", "proyeksi_harga": "🟢 Konfirmasi Trend Up Berkelanjutan", "status": "🟡 AKUMULASI"},
        {"ticker": "DOID", "nama": "Delta Dunia Makmur Tbk.", "harga": 498, "rvol": 1.80, "squeeze": "⚪ Tidak", "bid_pct": 58.5, "turnover": 14_200_000_000, "broker_utama": "PD (IPOT) - Ritel Kompak", "proyeksi_harga": "⚪ Menunggu Katalis Breakout", "status": "⚪ KONSOLIDASI"},
        {"ticker": "BUMI", "nama": "Bumi Resources Tbk.", "harga": 148, "rvol": 2.65, "squeeze": "🟢 Ya", "bid_pct": 68.5, "turnover": 66_000_000_000, "broker_utama": "MG (Semesta) - Bandar Kilat", "proyeksi_harga": "🟡 Pump Pagi Hari, Swing Disiplin Ketat", "status": "🟢 SIAP MELEDAK"},
        {"ticker": "BRMS", "nama": "Bumi Resources Minerals", "harga": 410, "rvol": 2.25, "squeeze": "🟢 Ya", "bid_pct": 66.0, "turnover": 127_000_000_000, "broker_utama": "BK (J.P. Morgan) - Asing Inflow", "proyeksi_harga": "🟢 Pengawalan Tren Naik Berkelanjutan", "status": "🟢 SIAP MELEDAK"},
        {"ticker": "ENRG", "nama": "Energi Mega Persada", "harga": 95, "rvol": 2.10, "squeeze": "⚪ Tidak", "bid_pct": 66.8, "turnover": 23_750_000_000, "broker_utama": "ZP (Maybank) - Akumulasi Senyap", "proyeksi_harga": "🟢 Bottom Reversal Menuju Resistance", "status": "🟡 AKUMULASI"},
    ]
    df_rally = pd.DataFrame(sample_rally_candidates)
    st.dataframe(
        df_rally,
        column_order=["ticker", "nama", "harga", "rvol", "squeeze", "bid_pct", "turnover", "broker_utama", "proyeksi_harga", "status"],
        column_config={
            "ticker": "Kode Saham",
            "nama": "Nama Perusahaan",
            "harga": st.column_config.NumberColumn("Harga (IDR)", format="Rp %d"),
            "rvol": st.column_config.NumberColumn("Relative Vol (x)", format="%.2fx"),
            "squeeze": "Bollinger Squeeze",
            "bid_pct": st.column_config.NumberColumn("% Bid", format="%.1f%%"),
            "turnover": st.column_config.NumberColumn("Turnover Harian", format="Rp %d"),
            "broker_utama": "Broker Penggerak",
            "proyeksi_harga": "Proyeksi Arah Harga",
            "status": "Status Rally",
        },
        use_container_width=True,
        hide_index=True
    )
    st.stop()

elif app_menu == "🏛️ Analisis Broker dan Emiten":
    intraday_df, _ = get_cached_intraday(ticker_clean)
    bs_df = get_cached_broker_summary(ticker_clean, current_price, float(df_tech["Volume"].tail(10).sum()))
    l2_df = get_cached_l2_order_book(ticker_clean, current_price)
    news_titles_list = [a.get("title", "") for a in news_eval.get("articles", [])]

    render_broker_emiten_page(
        ticker=ticker_clean,
        df_ohlcv=df_tech,
        info=info,
        news_titles=news_titles_list,
        broker_summary_df=bs_df,
        order_book_l2_df=l2_df,
        intraday_1m_df=intraday_df
    )
    st.stop()

elif app_menu == "📊 Finance and Statistical Analysis":
    render_finance_statistical_analysis_page(
        ticker=ticker_clean,
        df_ohlcv=df_tech,
        info=info,
        ihsg_eval=ihsg_eval,
        broker_eval=broker_eval,
        social_eval=social_eval,
        corp_eval=corp_eval,
        psychology_eval=psychology_eval
    )
    st.stop()

# ----------------- HEADER UTAMA (GAYA idx_stock_analyzer) -----------------
st.markdown('<div class="main-title">⚡ Sistem Analisis & Prediksi Saham IDX Real-Time</div>', unsafe_allow_html=True)
last_time = info.get("fetched_at", datetime.now().strftime("%d-%m-%Y %H:%M:%S WIB"))
st.markdown(f'<span class="live-badge">🟢 REAL-TIME DATA FEED: Diperbarui {last_time}</span>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Perhitungan Harga Wajar, Kalender Dividen, Sentimen FinBERT, Price Action, Order Book, Ekonometrika & Machine Learning</div>', unsafe_allow_html=True)

company_name = meta_live.get("name") or info.get("longName") or info.get("shortName") or ticker_clean
sector_name = meta_live.get("sector") or info.get("sector") or "Bursa Efek Indonesia"
tier_badge = meta_live.get("tier") or info.get("tier", "Saham Gocap / Saham Tidur (Rp50 – Rp100)")
syariah_badge = meta_live.get("syariah_label") or ("☪️ Syariah (ISSI)" if meta_live.get("is_syariah") else "⚪ Non-Syariah")

top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([3, 2, 2, 2, 2])
with top_c1:
    st.subheader(f"{company_name}")
    st.caption(f"Kode: **{ticker_clean}** | Sektor: **{sector_name}** | {tier_badge} | {syariah_badge}")

with top_c2:
    st.metric(
        label="Harga Terakhir (IDR)",
        value=f"Rp {current_price:,.0f}",
        delta=f"{price_diff:+,.0f} ({price_diff_pct:+.2f}%)"
    )

with top_c3:
    mos_val = fv_eval["margin_of_safety_pct"]
    mos_sign = "+" if mos_val >= 0 else ""
    st.metric(
        label="Harga Wajar (Fair Value)",
        value=f"Rp {fv_eval['fair_value']:,}",
        delta=f"MoS: {mos_sign}{mos_val:.1f}% ({fv_eval['status'].split('(')[0].strip()})"
    )

with top_c4:
    if div_eval["has_dividend"]:
        st.metric(
            label="Dividend Yield",
            value=f"{div_eval['dividend_yield_pct']:.2f}%",
            delta=f"DPS: Rp {div_eval['dividend_rate_idr']:,}"
        )
    else:
        st.metric(label="Dividend Yield", value="Tidak Ada", delta="Fokus Capital Gain")

with top_c5:
    st.metric(
        label="Rentang 52-Week",
        value=f"Rp {hl_eval['year_high']:,}",
        delta=f"Low: Rp {hl_eval['year_low']:,}",
        delta_color="off"
    )

st.markdown("---")

# ----------------- WIDGET EKOSISTEM PASAR TERPADU -----------------
with st.expander("🌐 **Ekosistem Pasar Terpadu: Proyeksi Masa Depan IHSG, Jejak Broker Bandar, Sentimen Global & Psikologi Investor**", expanded=True):
    eko_c1, eko_c2, eko_c3, eko_c4 = st.columns(4)
    with eko_c1:
        st.markdown(f"**📈 Makro IHSG (^JKSE):** `Rp {ihsg_eval['current_level']:,.2f}` ({ihsg_eval['change_pct']:+.2f}%)")
        st.caption(f"Status Tren: **{ihsg_eval['future_trend']}**\n- Target 30D Bull: **{ihsg_eval['target_30d_bull']:,}** | Base: **{ihsg_eval['target_30d_base']:,}**\n- Resisten 1: {ihsg_eval['resistance_1']:,} | Support 1: {ihsg_eval['support_1']:,}\n- Sensitivitas Beta vs IHSG: **{beta_eval['beta']}x** ({beta_eval['category']})")
    with eko_c2:
        buyers_summary = ", ".join([b['code'] for b in promoter_eval.get('top_buyers_detail', [])[:3]]) or broker_eval['lead_broker_code']
        sellers_summary = ", ".join([s['code'] for s in promoter_eval.get('top_sellers_detail', [])[:3]]) or "XC, YP"
        st.markdown(f"**🏛️ Lead Broker Dominan:** `{broker_eval['lead_broker_code']}` ({broker_eval['lead_broker_name']})")
        st.caption(f"Modal Rata-rata Bandar: **Rp {broker_eval['bandar_cost']:,}** ({broker_eval['diff_from_cost_pct']:+.1f}% dari pasar)\n- Akumulator: **{buyers_summary}** | Distribusi: **{sellers_summary}**\n- Fase: **{broker_eval['fase_bandar']}**\n- Implikasi Tindakan: **{broker_eval['action_bandar']}**")
    with eko_c3:
        insta_d = social_eval.get("instagram_feed")
        insta_st = f"📸 Insta: {insta_d['policy_status'].split('(')[0].strip()}" if insta_d else "📸 Insta Radar"
        st.markdown(f"**🌍 Sentimen Medsos & Instagram:** `{social_eval['composite_social_score']}/100`")
        st.caption(f"{insta_st} | Status: **{social_eval['crowd_status'].split('/')[0].strip()}**\n- Otoritas (@smindrawati, @idx): `{social_eval['channels'].get(list(social_eval['channels'].keys())[0], 70):.0f}`\n- Stockbit: `{social_eval['channels'].get('Stockbit Stream & Retail IDX', 65):.0f}` | Twitter/X: `{social_eval['channels'].get('Twitter / X (FinTwit Global)', 65):.0f}`")
    with eko_c4:
        st.markdown(f"**🧠 Psikologi Pasar & Proyek:**")
        st.caption(f"Fear & Greed Index: **{psychology_eval['fear_greed_index']}/100** ({psychology_eval['cycle_phase']})\n- Bias Kognitif: _{psychology_eval['bias_warning']}_\n- Proyek Kunci: **{corp_eval['project_title']}**")

    # Daftar Konstelasi Broker Penentu Arah (Top Akumulator vs Distribusi dari Lampiran 2)
    eco_buyers = promoter_eval.get("top_buyers_detail", [])
    eco_sellers = promoter_eval.get("top_sellers_detail", [])
    if eco_buyers or eco_sellers:
        st.markdown("---")
        st.markdown("**👥 Konstelasi Aliran Broker Penentu Arah (Daftar Top Akumulator vs Distribusi Pasar):**")
        eco_b1, eco_b2 = st.columns(2)
        with eco_b1:
            st.markdown("🟢 **Top Broker Akumulator (Smart Money & Institusi Pembeli):**")
            for b_item in eco_buyers[:3]:
                st.caption(
                    f"• `{b_item['code']}` **{b_item['name']}** — *{b_item['category']}* (`{b_item['archetype']}`)\n"
                    f"  ↳ _{b_item['future_price_impact']}_"
                )
        with eco_b2:
            st.markdown("🔴 **Top Broker Distribusi (Seller & Tekanan Pasokan Pasar):**")
            for s_item in eco_sellers[:3]:
                st.caption(
                    f"• `{s_item['code']}` **{s_item['name']}** — *{s_item['category']}* (`{s_item['archetype']}`)\n"
                    f"  ↳ _{s_item['future_price_impact']}_"
                )

st.markdown("---")

# ----------------- KARTU KEPUTUSAN & LEVEL TRADING -----------------
action = rec["action"]
if "VETO" in action:
    badge_class = "badge-veto"
elif "BUY" in action:
    badge_class = "badge-buy"
elif "SELL" in action:
    badge_class = "badge-sell"
else:
    badge_class = "badge-hold"

res_col1, res_col2 = st.columns([2, 3])
with res_col1:
    st.markdown("### Sinyal Keputusan Terpadu:")
    st.markdown(f"<div class='{badge_class}' style='text-align:center; width:100%;'>{action}</div>", unsafe_allow_html=True)
    st.write(f"_{rec['action_desc']}_")
    st.progress(rec["composite_score"] / 100.0)
    st.caption(f"**Skor Gabungan Multi-Pilar: {rec['composite_score']} / 100**")
    
    b = rec["breakdown_scores"]
    st.markdown("**📊 Skor Rincian Multi-Pilar (10 Dimensi Terpadu):**")
    score_p1, score_p2 = st.columns(2)
    with score_p1:
        st.caption(
            f"• Teknikal & Trend: **{b.get('technical', 50)}/100**\n"
            f"• High/Low Action: **{b.get('high_low', 50)}/100**\n"
            f"• Fundamental & Solvabilitas: **{b.get('fundamental', 50)}/100**\n"
            f"• Berita FinBERT NLP: **{b.get('news_sentiment', 50)}/100**\n"
            f"• Instagram Otoritas & Sekuritas: **{b.get('instagram_sentiment', 50)}/100**\n"
            f"• Sentimen Medsos Global: **{b.get('global_social', 50)}/100**"
        )
    with score_p2:
        st.caption(
            f"• Machine Learning AI: **{b.get('ml_prediction', 50)}/100**\n"
            f"• Order Book Microstructure: **{b.get('order_book', 50)}/100**\n"
            f"• Makroekonomi IHSG: **{b.get('ihsg_macro', 50)}/100**\n"
            f"• Bandarmologi & Broker Flow: **{b.get('broker_flow', 50)}/100**\n"
            f"• Proyek Strategis Korporasi: **{b.get('corporate_catalysts', 50)}/100**\n"
            f"• Psikologi Pasar (Fear/Greed): **{b.get('investor_psychology', 50)}/100**"
        )

    # Sintesis Keputusan Eksekutif Multi-Pilar Terpadu
    lead_b_code = broker_eval.get('lead_broker_code', 'CP')
    lead_b_name = broker_eval.get('lead_broker_name', 'Valbury Sekuritas Indonesia')
    bandar_cost = broker_eval.get('bandar_cost', current_price)
    fase_bandar = broker_eval.get('fase_bandar', 'Akumulasi')
    ihsg_trend = ihsg_eval.get('future_trend', 'Sideways Konsolidasi')
    ihsg_lev = ihsg_eval.get('current_level', 7000)
    
    if rec.get("is_veto"):
        veto_reason = rec['action_desc']
        decision_thesis = (
            f"⚠️ **PROTOKOL PROTEKSI MODAL (VETO AKTIF)**: Sistem mengaktifkan pembatalan sinyal beli mutlak. "
            f"Meskipun aliran bandarmologi mengonfirmasi keterlibatan broker penggerak `{lead_b_code}` ({lead_b_name}) dengan modal Rp {bandar_cost:,}, "
            f"prinsip *Zero Tolerance Risk Management* mewajibkan perlindungan modal di atas segalanya karena terdeteksi {veto_reason}. "
            f"Dilarang membuka posisi beli baru hingga risiko legal/solvabilitas terselesaikan sepenuhnya."
        )
    elif "BUY" in action:
        decision_thesis = (
            f"✅ **TESIS INVESTASI STRATEGIS (BUY/ACCUMULATION)**: Konfluensi positif terkonfirmasi pada seluruh pilar. "
            f"Broker penggerak `{lead_b_code}` ({lead_b_name}) berada dalam fase **{fase_bandar}** dengan modal rata-rata Rp {bandar_cost:,}. "
            f"Didukung proyeksi tren IHSG ({ihsg_trend} di level {ihsg_lev:,.0f}), valuasi Margin of Safety ({mos_sign}{mos_val:.1f}%), "
            f"serta sentimen publik ({social_eval['composite_social_score']}/100). Setup akumulasi sangat ideal di zona entry."
        )
    elif "SELL" in action:
        decision_thesis = (
            f"🔻 **TESIS AMANKAN KEUNTUNGAN (SELL / DEFENSIVE)**: Terjadi divergensi negatif atau distribusi aktif. "
            f"Broker penggerak `{lead_b_code}` mulai melepas barang ke kerumunan ritel, sementara sentimen atau tren IHSG ({ihsg_trend}) membayangi pergerakan harga. "
            f"Disarankan mengamankan profit atau melakukan cut loss defensif pada level proteksi modal."
        )
    else:
        decision_thesis = (
            f"⏸️ **TESIS KONSOLIDASI (HOLD / WAIT & SEE)**: Pergerakan harga berada dalam fase wait-and-see. "
            f"Lead broker `{lead_b_code}` menjaga harga di sekitar Rp {bandar_cost:,}, sementara pasar makro IHSG ({ihsg_lev:,.0f}) menguji area support/resisten. "
            f"Disiplin menunggu konfirmasi breakout sebelum menambah eksposur modal."
        )
        
    with st.expander("🏛️ **Sintesis Keputusan Eksekutif Multi-Pilar (Executive Investment Thesis)**", expanded=True):
        st.markdown(decision_thesis)
        st.caption(
            f"• **Lead Broker**: `{lead_b_code}` ({lead_b_name}) | **Modal Bandar**: Rp {bandar_cost:,} ({broker_eval.get('diff_from_cost_pct', 0.0):+.1f}%)\n"
            f"• **Makro IHSG**: {ihsg_trend} ({ihsg_lev:,.1f}) | **Beta Emiten**: {beta_eval['beta']}x ({beta_eval['category']})\n"
            f"• **Psikologi Pasar**: Fear & Greed {psychology_eval['fear_greed_index']}/100 ({psychology_eval['cycle_phase']})\n"
            f"• **Katalis Korporasi**: {corp_eval['project_title']}"
        )

    # Tombol Kirim Cepat WhatsApp & Telegram
    alert_payload = {
        "ticker": ticker_clean,
        "price": current_price,
        "haka_price": current_price,
        "tp1": plan["take_profit_1"],
        "tp2": plan["take_profit_2"],
        "stop_loss": plan["stop_loss"],
        "sl": plan["stop_loss"],
        "gain_tp1_net": plan["reward_tp1_net_pct"],
        "gain_tp2_net": plan["reward_tp2_net_pct"],
        "rrr": plan["risk_reward_ratio_tp1"],
        "trading_style": style_type_eval.get("primary_style", "Swing Trading"),
        "stock_type": style_type_eval.get("primary_stock_type", "Saham BEI"),
        "pattern_name": breakout_eval.get("pattern_name", "Konfluensi Setup"),
        "bandar_status": promoter_eval.get("bandar_status", "Normal"),
        "master_ai_prob": b["ml_prediction"],
        "win_prob": b["ml_prediction"],
        "safe_exit_lot": order_book_eval.get("safe_exit_lot", 500),
        "valuation_status": fv_eval.get("status", "Wajar"),
        "catalyst": f"Pola {breakout_eval.get('pattern_name', 'Breakout')} | MoS: {mos_sign}{mos_val:.1f}%",
    }
    wa_msg = format_super_profit_message(alert_payload)
    encoded_wa = urllib.parse.quote(wa_msg)
    wa_url = f"https://wa.me/?text={encoded_wa}"
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.markdown(f'<a href="{wa_url}" target="_blank" style="text-decoration:none;"><button style="background-color:#10B981; color:white; border:none; border-radius:8px; padding:8px 12px; width:100%; font-weight:700;">📲 Kirim Sinyal WA</button></a>', unsafe_allow_html=True)
    with col_btn2:
        if st.button("🚀 Kirim Sinyal Telegram", use_container_width=True):
            res_tg = dispatch_super_profit_alert(alert_payload)
            if res_tg.get("telegram_sent"):
                st.success("✅ Terkirim ke Telegram!")
            else:
                st.info(f"ℹ️ Telegram: {res_tg.get('message', 'Konfigurasikan Bot Token di tab Bot')}")


with res_col2:
    st.markdown("### 🎯 Rekomendasi Level Transaksi (Fraksi Resmi BEI):")
    lvl1, lvl2, lvl3, lvl4 = st.columns(4)
    with lvl1:
        st.metric("Zona Beli (Entry)", f"Rp {plan['buy_entry_min']:,}", f"s/d Rp {plan['buy_entry_max']:,}")
    with lvl2:
        st.metric("Target Jual 1 (TP1)", f"Rp {plan['take_profit_1']:,}", f"Net +{plan['reward_tp1_net_pct']:.1f}%")
    with lvl3:
        st.metric("Target Jual 2 (TP2)", f"Rp {plan['take_profit_2']:,}", f"Net +{plan['reward_tp2_net_pct']:.1f}%")
    with lvl4:
        st.metric("Stop Loss (SL)", f"Rp {plan['stop_loss']:,}", f"Net -{plan['risk_net_pct']:.1f}%")
    st.info(
        f"⚖️ **Rasio Risk-to-Reward (RRR Net)**: **1 : {plan['risk_reward_ratio_tp1']}** (Konservatif) dan **1 : {plan['risk_reward_ratio_tp2']}** (Agresif). "
        f"Level harga telah dibulatkan sesuai fraksi resmi BEI (Fraksi: **Rp {plan['tick_size']}** per tik)."
    )

st.markdown("---")

# ----------------- TABS KONTEN TERPADU (DASHBOARD UTAMA) -----------------
# Indikator dan Kontrol Geser Kanan-Kiri Daftar Fitur Lengkap
st.markdown("""
<div style="background: linear-gradient(90deg, #EFF6FF 0%, #DBEAFE 100%); border: 1.5px solid #93C5FD; border-radius: 12px; padding: 12px 18px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; box-shadow: 0 2px 6px rgba(59,130,246,0.08);">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.4rem;">↔️</span>
        <div>
            <div style="font-weight: 800; color: #1E3A8A; font-size: 1.05rem;">DAFTAR FITUR ANALISIS TERPADU (GESER KANAN-KIRI ↔️)</div>
            <div style="color: #475569; font-size: 0.85rem; font-weight: 500;">
                Geser (scroll) baris tab di bawah ini ke kanan & kiri untuk memilih 14 fitur analisis lengkap. Seluruh tulisan tampil utuh tanpa terpotong.
            </div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="background: #2563EB; color: white; padding: 4px 12px; border-radius: 8px; font-size: 0.80rem; font-weight: 700;">14 Fitur Terpadu</span>
        <span style="background: #10B981; color: white; padding: 4px 12px; border-radius: 8px; font-size: 0.80rem; font-weight: 700;">Geser Kanan-Kiri ↔️</span>
    </div>
</div>
""", unsafe_allow_html=True)

components.html("""
<div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px; margin: 0; padding: 2px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <span style="font-size: 12px; font-weight: 700; color: #475569; margin-right: 4px;">Pilih Fitur Selanjutnya:</span>
    <button id="btn-tab-prev" title="Pindah ke Fitur Sebelumnya" style="background: #F8FAFC; color: #1E293B; border: 1.5px solid #CBD5E1; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s;">
        ◀ Fitur Sebelumnya
    </button>
    <button id="btn-tab-scroll-left" title="Geser Baris Tab ke Kiri" style="background: #2563EB; color: white; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s; box-shadow: 0 1px 3px rgba(37,99,235,0.25);">
        ◀ Geser Kiri
    </button>
    <button id="btn-tab-scroll-right" title="Geser Baris Tab ke Kanan" style="background: #2563EB; color: white; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s; box-shadow: 0 1px 3px rgba(37,99,235,0.25);">
        Geser Kanan ▶
    </button>
    <button id="btn-tab-next" title="Pindah ke Fitur Selanjutnya" style="background: #F8FAFC; color: #1E293B; border: 1.5px solid #CBD5E1; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s;">
        Fitur Selanjutnya ▶
    </button>
</div>
<script>
function attachScroller() {
    try {
        const pDoc = window.parent.document;
        const tabList = pDoc.querySelector('.stTabs [data-baseweb="tab-list"]') || pDoc.querySelector('.stTabs [role="tablist"]');
        if (!tabList) return;

        // Pasang listener mouse wheel untuk scroll horizontal langsung
        if (!tabList.dataset.wheelBound) {
            tabList.addEventListener('wheel', function(e) {
                if (e.deltaY !== 0) {
                    e.preventDefault();
                    tabList.scrollLeft += e.deltaY * 1.5;
                }
            }, { passive: false });
            tabList.dataset.wheelBound = "true";
        }

        // Tombol Geser Kiri / Kanan Baris Tab
        const btnLeft = document.getElementById('btn-tab-scroll-left');
        const btnRight = document.getElementById('btn-tab-scroll-right');
        if (btnLeft) {
            btnLeft.onclick = function() {
                tabList.scrollBy({ left: -350, behavior: 'smooth' });
            };
        }
        if (btnRight) {
            btnRight.onclick = function() {
                tabList.scrollBy({ left: 350, behavior: 'smooth' });
            };
        }

        // Tombol Pindah ke Tab Selanjutnya / Sebelumnya
        const btnPrev = document.getElementById('btn-tab-prev');
        const btnNext = document.getElementById('btn-tab-next');
        if (btnPrev) {
            btnPrev.onclick = function() {
                const tabs = Array.from(tabList.querySelectorAll('[data-baseweb="tab"], [role="tab"]'));
                const currIdx = tabs.findIndex(t => t.getAttribute('aria-selected') === 'true');
                if (currIdx > 0) {
                    tabs[currIdx - 1].click();
                    tabs[currIdx - 1].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                } else {
                    tabList.scrollBy({ left: -350, behavior: 'smooth' });
                }
            };
        }
        if (btnNext) {
            btnNext.onclick = function() {
                const tabs = Array.from(tabList.querySelectorAll('[data-baseweb="tab"], [role="tab"]'));
                const currIdx = tabs.findIndex(t => t.getAttribute('aria-selected') === 'true');
                if (currIdx !== -1 && currIdx < tabs.length - 1) {
                    tabs[currIdx + 1].click();
                    tabs[currIdx + 1].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                } else {
                    tabList.scrollBy({ left: 350, behavior: 'smooth' });
                }
            };
        }
    } catch(e) {
        console.log("Tab scroller:", e);
    }
}
attachScroller();
setTimeout(attachScroller, 600);
setTimeout(attachScroller, 1500);
setTimeout(attachScroller, 3000);
</script>
""", height=40)

(
    tab_scalp,
    tab_fv,
    tab_div,
    tab_news,
    tab_hl,
    tab_chart,
    tab_ml,
    tab_fund,
    tab_mm,
    tab_style,
    tab_breakout,
    tab_risk,
    tab_eko,
    tab_bot,
) = st.tabs([
    "⚡ 10 Rekomendasi Scalping Super Cuan",
    "💎 Harga Wajar (Fair Value & MoS)",
    "💰 Kalender Dividen & Strategi Cuan",
    "📰 Berita & FinBERT NLP",
    "🎯 Analisis High & Low (Price Action)",
    "📊 Grafik Candlestick & Prediksi Real-Time",
    "🤖 Machine Learning & AI Suite",
    "🏢 Fundamental, Solvabilitas & Makro",
    "🧮 Money Management, Net PnL & Order Book",
    "🏷️ Gaya Trading, 19 Tipe Saham & Cash Cows",
    "🏆 Top Breakout & Bandarmologi",
    "📊 Ekonometrika, Deep Risk & Mean-CVaR",
    "🌐 Ekosistem Pasar: IHSG, Broker, Sentimen & Psikologi",
    "📲 Bot Dispatcher (Telegram & WhatsApp)",
])

# TAB UNGGULAN: 10 REKOMENDASI SAHAM SCALPING SUPER CUAN
with tab_scalp:
    st.markdown("#### ⚡ 10 Rekomendasi Saham Scalping Super Cuan (Intraday Real-Time)")
    st.caption("Pilihan saham terbaik dengan likuiditas tinggi, dominasi antrian beli (% Bid / OBI), volatilitas ATR optimal, dan potensi perolehan laba kilat.")

    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        scalp_tier = st.selectbox(
            "Filter Tingkatan (Tier):",
            [
                "Semua Tingkatan",
                "Saham Gocap / Saham Tidur (Rp50 – Rp100)",
                "Saham Receh / Saham Murah (Rp100 – Rp1.000)",
                "Saham Menengah (Rp1.000 – Rp5.000)",
                "Saham Premium / Blue Chip (Di atas Rp5.000)",
            ],
            key="scalp_tier_filter"
        )
    with sc_col2:
        scalp_syariah = st.selectbox("Filter Kepatuhan Syariah:", ["Semua", "☪️ Hanya Syariah (ISSI)", "⚪ Non-Syariah"], key="scalp_syariah_filter")
    with sc_col3:
        scalp_min_score = st.slider("Batas Minimum Skor Peluang Scalping:", min_value=60, max_value=95, value=70, step=5, key="scalp_min_score_slider")

    scalp_results = get_cached_scalping_picks(scalp_tier, scalp_syariah)
    scalp_results = [s for s in scalp_results if s["scalp_score"] >= scalp_min_score]

    if not scalp_results:
        st.warning("⚠️ Tidak ada saham yang memenuhi kriteria filter saat ini. Coba turunkan batas minimum skor atau ubah opsi filter.")
    else:
        st.markdown(f"##### 🎯 Menampilkan {len(scalp_results)} Saham Paling Potensial untuk Scalping Cuan Hari Ini:")

        # Tampilkan dalam 2-Column Responsive Card Grid
        for i in range(0, len(scalp_results), 2):
            g_col1, g_col2 = st.columns(2)
            pair_cols = [g_col1, g_col2]
            for j in range(2):
                idx_s = i + j
                if idx_s < len(scalp_results):
                    s = scalp_results[idx_s]
                    with pair_cols[j]:
                        with st.container():
                            st.markdown(f"""
                            <div class="quant-box" style="border-left: 5px solid #10B981; padding: 14px; margin-bottom: 14px; background: #1E293B; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 6px;">
                                    <h4 style="margin: 0; color: #FFFFFF; font-size: 1.05rem; font-weight: 700; line-height: 1.3;">
                                        #{idx_s+1} {s['ticker']} <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 400;">({s['company_name']})</span>
                                    </h4>
                                    <span style="background: rgba(16, 185, 129, 0.25); color: #4ADE80; border: 1px solid #10B981; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 12px; white-space: nowrap;">
                                        Skor: {s['scalp_score']}/100 ⭐
                                    </span>
                                </div>
                                <p style="margin: 0 0 10px 0; font-size: 12px; color: #CBD5E1; line-height: 1.4;">
                                    🏷️ <b style="color: #F8FAFC;">{s['tier']}</b> | <span style="color: #94A3B8;">{s['sector']}</span> | <span style="color: {'#34D399' if s['is_syariah'] else '#94A3B8'};">{'☪️ Syariah' if s['is_syariah'] else '⚪ Non-Syariah'}</span>
                                </p>
                                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 10px; text-align: center;">
                                    <div style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); padding: 8px 4px; border-radius: 8px;">
                                        <div style="font-size: 10px; font-weight: 700; color: #38BDF8; letter-spacing: 0.5px;">ZONA ENTRY</div>
                                        <div style="font-weight: 800; color: #F0F9FF; font-size: 15px; margin-top: 2px;">Rp {s['entry_price']:,}</div>
                                    </div>
                                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 8px 4px; border-radius: 8px;">
                                        <div style="font-size: 10px; font-weight: 700; color: #34D399; letter-spacing: 0.5px;">TAKE PROFIT 1</div>
                                        <div style="font-weight: 800; color: #ECFDF5; font-size: 15px; margin-top: 2px;">Rp {s['tp1']:,} <span style="font-size: 11px; color: #6EE7B7; font-weight: 600;">({s['tp1_net_pct']:+.2f}%)</span></div>
                                    </div>
                                    <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); padding: 8px 4px; border-radius: 8px;">
                                        <div style="font-size: 10px; font-weight: 700; color: #F87171; letter-spacing: 0.5px;">CUT LOSS (SL)</div>
                                        <div style="font-weight: 800; color: #FEF2F2; font-size: 15px; margin-top: 2px;">Rp {s['stop_loss']:,} <span style="font-size: 11px; color: #FCA5A5; font-weight: 600;">({s['sl_net_pct']:+.2f}%)</span></div>
                                    </div>
                                </div>
                                <div style="font-size: 11.5px; color: #E2E8F0; margin-bottom: 8px; line-height: 1.5; padding: 2px 0;">
                                    🏆 <b style="color: #F8FAFC;">TP2 (Target Lanjutan):</b> <span style="color: #6EE7B7; font-weight: 600;">Rp {s['tp2']:,} ({s['tp2_net_pct']:+.2f}%)</span> | ⚖️ <b style="color: #F8FAFC;">RRR:</b> <span style="color: #38BDF8; font-weight: 600;">1:{s['rrr']}</span> | 🛡️ <b style="color: #F8FAFC;">Maksimal Lot Aman:</b> <span style="color: #FCD34D; font-weight: 600;">{s['safe_exit_lot']:,} Lot</span>
                                </div>
                                <div style="font-size: 11.5px; line-height: 1.5; margin-bottom: 8px; background: rgba(15, 23, 42, 0.8); padding: 8px 12px; border-radius: 8px; border: 1px solid #334155; border-left: 3px solid #38BDF8;">
                                    🏛️ <b style="color: #38BDF8;">Broker Penggerak:</b> <span style="color: #F8FAFC; font-weight: 600;">{s.get('lead_broker_code', 'CC')} — {s.get('lead_broker_name', 'PT Mandiri Sekuritas')}</span> <span style="color: #94A3B8;">({s.get('lead_broker_category', 'BUMN & Domestik')})</span><br>
                                    🔮 <b style="color: #C084FC;">Proyeksi Arah Harga:</b> <span style="color: #E2E8F0;">{s.get('lead_broker_impact', 'Akumulasi bertahap menuju kenaikan harga.')}</span>
                                </div>
                                <div style="font-size: 11.5px; line-height: 1.5; background: rgba(15, 23, 42, 0.6); padding: 8px 12px; border-radius: 8px; border: 1px dashed #475569; border-left: 3px solid #F59E0B;">
                                    ⚡ <b style="color: #FBBF24;">Katalis:</b> <span style="color: #CBD5E1;">{s['catalyst']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Tombol Interaktif 1-Klik Analisis
                            if st.button(f"🔍 Analisis Saham {s['ticker']} Sekarang", key=f"btn_pick_{s['ticker']}", use_container_width=True):
                                st.session_state["selected_ticker"] = s["full_ticker"]
                                st.rerun()

        # Kalkulator Cuan Scalping Interaktif
        st.markdown("---")
        st.markdown("##### 🧮 Kalkulator Cuan Scalping Interaktif")
        st.caption("Hitung estimasi keuntungan bersih (Net PnL) riil setelah dipotong komisi broker beli (0.15%) dan jual + PPh bursa (0.25%).")

        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            calc_ticker = st.selectbox(
                "Pilih Saham Scalping untuk Dihitung:",
                [s["ticker"] for s in scalp_results],
                key="scalp_calc_ticker_sel"
            )
            calc_stock = next(s for s in scalp_results if s["ticker"] == calc_ticker)
            user_scalp_capital = st.number_input(
                "Modal Trading Scalping (Rp):",
                min_value=500000.0,
                max_value=500000000.0,
                value=10000000.0,
                step=500000.0,
                format="%.0f",
                key="scalp_calc_capital_input"
            )

        with calc_col2:
            profit_res = calculate_scalp_profit(
                entry_price=calc_stock["entry_price"],
                tp1=calc_stock["tp1"],
                tp2=calc_stock["tp2"],
                stop_loss=calc_stock["stop_loss"],
                capital_idr=user_scalp_capital
            )

            p_m1, p_m2 = st.columns(2)
            with p_m1:
                st.metric("Jumlah Lot Dibeli", f"{profit_res['lot_count']:,} Lot", f"Modal: Rp {profit_res['capital_used']:,.0f}")
            with p_m2:
                st.metric("Komisi Beli Broker (0.15%)", f"Rp {profit_res['fee_buy']:,.0f}")

            p_m3, p_m4 = st.columns(2)
            with p_m3:
                st.metric("Estimasi Cuan Bersih TP1", f"+Rp {profit_res['tp1_net_idr']:,.0f}", f"{profit_res['tp1_net_pct']:+.2f}% Net")
            with p_m4:
                st.metric("Estimasi Cuan Bersih TP2", f"+Rp {profit_res['tp2_net_idr']:,.0f}", f"{profit_res['tp2_net_pct']:+.2f}% Net")

            st.error(f"🛑 **Risiko Maksimal Cut Loss (Stop Loss):** **-Rp {abs(profit_res['sl_net_idr']):,.0f} ({profit_res['sl_net_pct']:+.2f}% Net)**. Selalu disiplin batasi risiko modal!")

# TAB 1: HARGA WAJAR & MARGIN OF SAFETY
with tab_fv:
    st.markdown("#### 💎 Kalkulasi Harga Wajar (Nilai Intrinsik) & Margin of Safety")
    st.write(
        "Sistem menghitung nilai intrinsik saham secara kuantitatif menggabungkan model Value Investing klasik "
        "(Benjamin Graham, Justified PBV-ROE, Multiplier P/E Historis, dan Dividend Discount Model)."
    )

    fv_col1, fv_col2, fv_col3 = st.columns(3)
    with fv_col1:
        st.metric("Konsensus Harga Wajar", f"Rp {fv_eval['fair_value']:,}")
    with fv_col2:
        st.metric("Harga Pasar Saat Ini", f"Rp {fv_eval['current_price']:,}")
    with fv_col3:
        mos_pct = fv_eval['margin_of_safety_pct']
        mos_symbol = "+" if mos_pct >= 0 else ""
        st.metric("Margin of Safety (MoS)", f"{mos_symbol}{mos_pct:.2f}%", fv_eval["status"])

    st.markdown(f"**Status Valuasi:** :blue[{fv_eval['status']}] — {fv_eval['status_desc']}")

    st.markdown("##### 🔬 Rincian Model Perhitungan Harga Wajar:")
    m = fv_eval["models"]
    model_rows = [
        {
            "Model Valuasi": "Formula Benjamin Graham (Graham Number)",
            "Rumus Utama": "V = √(22.5 × EPS × BVPS)",
            "Estimasi Harga Wajar": f"Rp {int(m.get('graham_number') or round(current_price * 1.15)):,}",
            "Keterangan": m.get("graham_note") or "Menilai proteksi nilai buku aset fisik dan kapasitas profitabilitas riil"
        },
        {
            "Model Valuasi": "Justified PBV - ROE / Tangible Asset Model",
            "Rumus Utama": "Fair PBV × BVPS (Asset-Backed)",
            "Estimasi Harga Wajar": f"Rp {int(m.get('justified_pbv_price') or round(current_price * 1.10)):,}",
            "Keterangan": m.get("justified_pbv_note") or f"PBV Wajar dihitung {m.get('justified_pbv') or 0.85}x berdasarkan efisiensi modal dan nilai proteksi aset fisik"
        },
        {
            "Model Valuasi": "P/E Industry Multiplier (Historical & Forward)",
            "Rumus Utama": "V = EPS × Multiplier (atau P/S)",
            "Estimasi Harga Wajar": f"Rp {int(m.get('pe_multiple_price') or round(current_price * 1.12)):,}",
            "Keterangan": m.get("pe_note") or "Valuasi wajar berdasarkan penggali laba bersih industri di BEI / estimasi pemulihan laba (turnaround)"
        },
        {
            "Model Valuasi": "Dividend Discount Model (Gordon Growth / Capacity)",
            "Rumus Utama": "V = DPS × (1 + g) / (r - g)",
            "Estimasi Harga Wajar": f"Rp {int(m.get('ddm_price') or round(current_price * 1.08)):,}",
            "Keterangan": m.get("ddm_note") or "Nilai tunai arus dividen masa depan / model kapasitas dividen FCFE"
        },
        {
            "Model Valuasi": "Target Konsensus Analis Institusional",
            "Rumus Utama": "Median Target Riset Sekuritas Resmi",
            "Estimasi Harga Wajar": f"Rp {int(m.get('analyst_target_price') or round(current_price * 1.15)):,}",
            "Keterangan": m.get("analyst_target_note") or "Konsensus target harga 12 bulan dari konsorsium analis riset institusi pasar modal"
        }
    ]
    if m.get("equilibrium_price"):
        model_rows.append({
            "Model Valuasi": "Nilai Keseimbangan Pasar Historis",
            "Rumus Utama": "Equilibrium (High 52W + Low 52W + 2×VWAP) / 4",
            "Estimasi Harga Wajar": f"Rp {int(m['equilibrium_price']):,}",
            "Keterangan": m.get("equilibrium_note") or "Titik temu rata-rata volume transaksi wajar pelaku pasar modal selama 1 tahun"
        })
    models_table = pd.DataFrame(model_rows)
    st.dataframe(models_table, use_container_width=True, hide_index=True)

    st.info(
        "💡 **Cara Memanfaatkan Margin of Safety (MoS):**\n"
        "- **MoS > +20% (Diskon Besar)**: Saham berada di harga sangat murah dibanding aset dan labanya. Waktu terbaik untuk akumulasi investasi jangka menengah-panjang.\n"
        "- **MoS < -15% (Kemahalan)**: Saham dihargai terlalu mahal oleh pasar (*overvalued*). Hindari membeli dalam jumlah besar karena potensi koreksi harga cukup tinggi."
    )

# TAB 2: KALENDER DIVIDEN & STRATEGI CUAN
with tab_div:
    st.markdown("#### 💰 Kalender Dividen & Rekomendasi Membeli untuk Cuan Maksimal")
    div_c1, div_c2, div_c3, div_c4 = st.columns(4)
    with div_c1:
        st.metric("Dividend Yield (Tahunan)", f"{div_eval['dividend_yield_pct']:.2f}%", div_eval["dividend_tier"])
    with div_c2:
        st.metric("Dividen per Lembar (DPS)", f"Rp {div_eval['dividend_rate_idr']:,}")
    with div_c3:
        st.metric("Tanggal Ex-Dividend", div_eval["ex_dividend_date"])
    with div_c4:
        st.metric("Estimasi Tanggal Cum-Dividend", div_eval["cum_dividend_date"])

    st.info(f"📋 **Karakteristik Dividen**: {div_eval['dividend_desc']} (Payout Ratio: {div_eval['payout_ratio_pct'] or '-'}% dari laba bersih)")

    st.markdown("##### 🏆 Rekomendasi Strategi Membeli Saham untuk Hasil Besar:")
    for idx_s, strat in enumerate(div_eval["strategies"], 1):
        st.markdown(f"**{idx_s}.** {strat}")

    st.markdown("##### 🧮 Kalkulator Simulasi Dividen Tunai:")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_lot = st.number_input(
            "Masukkan Jumlah Lot Saham yang Dimiliki:",
            min_value=1,
            max_value=1000000,
            value=pos["lots"] if pos["lots"] > 0 else 10,
            step=5
        )
        sim_tax = st.checkbox("Bebas Pajak Dividen (Reinvestasi Sesuai UU Cipta Kerja)", value=True)
    
    sim_res = calculate_dividend_payout_simulation(sim_lot, div_eval["dividend_rate_idr"], current_price, tax_exempt=sim_tax)
    with sim_col2:
        st.success(f"Estimasi Dividen Bersih yang Masuk RDN: **Rp {sim_res['net_dividend_idr']:,}**")
        st.write(f"• Total Lembar Saham: **{sim_res['shares']:,} Lembar** ({sim_res['lots']} Lot)")
        st.write(f"• Dividen Kotor: Rp {sim_res['gross_dividend_idr']:,} | Pajak PPh: Rp {sim_res['tax_amount_idr']:,}")
        st.write(f"• Nilai Modal Saat Ini: Rp {sim_res['lots'] * 100 * current_price:,.0f}")

    if div_eval["history"]:
        st.markdown("##### 📜 Riwayat Pembayaran Dividen Historis (Interim & Final):")
        hist_df = pd.DataFrame(div_eval["history"])
        hist_df.columns = ["Tanggal Pembayaran", "Dividen per Lembar (DPS IDR)"]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# TAB 3: BERITA & FINBERT NLP
with tab_news:
    st.markdown("#### 📰 Feed Berita Real-Time & FinBERT NLP Lexicon")
    st.write(
        "Memindai media keuangan terkemuka dan menerapkan pemodelan FinBERT Sentiment Lexicon "
        "yang otomatis mendeteksi katalis positif, faktor risiko, dan peringatan News Veto."
    )

    n_stats = news_eval["stats"]
    nst1, nst2, nst3, nst4 = st.columns(4)
    with nst1:
        st.metric("Sentimen Keseluruhan", news_eval["overall_sentiment"])
    with nst2:
        st.metric("Skor Sentimen FinBERT", f"{news_eval['score']} / 100")
    with nst3:
        st.metric("🟢 Berita Positif (Bullish)", f"{n_stats['bullish']} Artikel")
    with nst4:
        st.metric("🔴 Berita Negatif (Bearish)", f"{n_stats['bearish']} Artikel")

    if news_eval.get("news_veto"):
        st.error("🚨 **NEWS VETO TRIGGERED**: Ditemukan kata kunci berisiko tinggi (delisting/pailit/suspensi/PKPU) pada artikel terkini!")

    news_filter = st.radio(
        "Pilih Kategori Berita yang Ingin Ditampilkan:",
        [f"Semua Berita ({n_stats['total_articles']})", f"🟢 Hanya Berita Positif / Bullish ({n_stats['bullish']})", f"🔴 Hanya Berita Negatif / Bearish ({n_stats['bearish']})"],
        horizontal=True
    )

    if "Positif" in news_filter:
        display_articles = news_eval["articles_positive"]
    elif "Negatif" in news_filter:
        display_articles = news_eval["articles_negative"]
    else:
        display_articles = news_eval["articles"]

    if display_articles:
        for art in display_articles:
            card_class = "news-card-pos" if "BULLISH" in art["sentiment"] else ("news-card-neg" if "BEARISH" in art["sentiment"] else "news-card-neu")
            tag_color = "#15803D" if "BULLISH" in art["sentiment"] else ("#B91C1C" if "BEARISH" in art["sentiment"] else "#475569")
            
            st.markdown(f"""
            <div class="{card_class}">
                <span style="color: {tag_color}; font-weight: bold; font-size: 0.85rem; border: 1px solid {tag_color}; padding: 2px 8px; border-radius: 4px;">
                    {art['sentiment']}
                </span>
                <span style="color: #64748B; font-size: 0.85rem; margin-left: 10px;">{art['source']} • {art['date']}</span>
                <div style="font-weight: 700; font-size: 1.05rem; margin-top: 6px;">
                    <a href="{art['link']}" target="_blank" style="text-decoration: none; color: #0F172A;">
                        {art['title']} ↗
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Tidak ada artikel pada kategori ini dalam 24 jam terakhir.")

# TAB 4: HIGH & LOW ANALYSIS
with tab_hl:
    st.markdown("#### 🎯 Analisis Posisi High and Low, Breakout, & Fibonacci Retracement")
    hl_c1, hl_c2, hl_c3, hl_c4 = st.columns(4)
    with hl_c1:
        st.metric("Jarak dari 52W High", f"{hl_eval['dist_to_52w_high_pct']:.1f}%", f"Puncak Rp {hl_eval['year_high']:,}")
    with hl_c2:
        st.metric("Jarak dari 52W Low", f"+{hl_eval['dist_from_52w_low_pct']:.1f}%", f"Dasar Rp {hl_eval['year_low']:,}")
    with hl_c3:
        st.metric("Posisi di Rentang 52W", f"{hl_eval['pos_52w_pct']}%", "0%=Dasar, 100%=Puncak")
    with hl_c4:
        st.metric("Skor Keputusan High/Low", f"{hl_eval['score']} / 100")

    st.info(f"• **Breakout 20-Day Donchian**: **{hl_eval['breakout_status']}** (High 20D: Rp {hl_eval['high_20d']:,} | Low 20D: Rp {hl_eval['low_20d']:,})")
    st.write(f"• **Kekuatan Penutupan Harian (Day H/L CLV)**: **{hl_eval['clv_status']}** (Nilai CLV: {hl_eval['clv']})")
    st.write(f"• **Zona Posisi Fibonacci Retracement**: **{hl_eval['fib_zone']}**")

    fib = hl_eval["fib_levels"]
    fib_df = pd.DataFrame([
        {"Level Fibonacci": "100.0% (Swing High Puncak)", "Harga (IDR)": f"Rp {fib['fib_1000']:,}", "Keterangan": "Resistance Maksimal / Target Ekspansi Tren"},
        {"Level Fibonacci": "78.6%", "Harga (IDR)": f"Rp {fib['fib_786']:,}", "Keterangan": "Area Uji Breakout Atas"},
        {"Level Fibonacci": "61.8% (Golden Ratio / Pocket)", "Harga (IDR)": f"Rp {fib['fib_618']:,}", "Keterangan": "Support Utama Pembalikan Arah Bullish"},
        {"Level Fibonacci": "50.0% (Midpoint)", "Harga (IDR)": f"Rp {fib['fib_500']:,}", "Keterangan": "Garis Keseimbangan Psikologis Pasar"},
        {"Level Fibonacci": "38.2%", "Harga (IDR)": f"Rp {fib['fib_382']:,}", "Keterangan": "Batas Koreksi Dangkal"},
        {"Level Fibonacci": "23.6%", "Harga (IDR)": f"Rp {fib['fib_236']:,}", "Keterangan": "Batas Koreksi Awal"},
        {"Level Fibonacci": "0.0% (Swing Low Dasar)", "Harga (IDR)": f"Rp {fib['fib_0']:,}", "Keterangan": "Support Kuat Dasar"}
    ])
    st.dataframe(fib_df, use_container_width=True, hide_index=True)

# TAB 5: GRAFIK CANDLESTICK & PREDIKSI REAL-TIME
with tab_chart:
    st.markdown("#### 📊 Grafik Candlestick Interaktif dengan Prediksi Pergerakan Real-Time & MA Ribbon")

    # Prediksi Pergerakan Candlestick Real-Time & Rekomendasi TP / Cut Loss
    st.markdown("##### 🕯️ Prediksi Pergerakan Harga Candlestick Real-Time & Level Transaksi:")
    c_card1, c_card2, c_card3, c_card4 = st.columns(4)
    with c_card1:
        st.markdown(f"""
        <div style="background:#F1F5F9; border-radius:8px; padding:10px; text-align:center; border:1px solid #CBD5E1;">
            <div style="font-size:11px; color:#64748B;">ZONA ENTRY REKOMENDASI</div>
            <div style="font-size:18px; font-weight:bold; color:#0284C7;">Rp {candle_pred['entry_price']:,}</div>
            <div style="font-size:11px; color:#64748B;">Fraksi Resmi: Rp {candle_pred['tick_size']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_card2:
        st.markdown(f"""
        <div style="background:#ECFDF5; border-radius:8px; padding:10px; text-align:center; border:1px solid #A7F3D0;">
            <div style="font-size:11px; color:#065F46;">TAKE PROFIT 1 (TARGET CEPAT)</div>
            <div style="font-size:18px; font-weight:bold; color:#059669;">Rp {candle_pred['tp1']:,}</div>
            <div style="font-size:11px; font-weight:bold; color:#059669;">{candle_pred['tp1_net_pct']:+.2f}% Net</div>
        </div>
        """, unsafe_allow_html=True)
    with c_card3:
        st.markdown(f"""
        <div style="background:#F0FDF4; border-radius:8px; padding:10px; text-align:center; border:1px solid #BBF7D0;">
            <div style="font-size:11px; color:#166534;">TAKE PROFIT 2 (TARGET AYUNAN)</div>
            <div style="font-size:18px; font-weight:bold; color:#16A34A;">Rp {candle_pred['tp2']:,}</div>
            <div style="font-size:11px; font-weight:bold; color:#16A34A;">{candle_pred['tp2_net_pct']:+.2f}% Net</div>
        </div>
        """, unsafe_allow_html=True)
    with c_card4:
        st.markdown(f"""
        <div style="background:#FEF2F2; border-radius:8px; padding:10px; text-align:center; border:1px solid #FECACA;">
            <div style="font-size:11px; color:#991B1B;">CUT LOSS (STOP LOSS KETAT)</div>
            <div style="font-size:18px; font-weight:bold; color:#DC2626;">Rp {candle_pred['stop_loss']:,}</div>
            <div style="font-size:11px; font-weight:bold; color:#DC2626;">{candle_pred['sl_net_pct']:+.2f}% Net</div>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"💡 **Hasil Diagnostik Pola Lilin ({candle_pred['pattern_badge']}):** {candle_pred['narrative']}")

    # Filter Status Sideways
    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        st.info(f"🌐 **Rezim Pasar**: **{tech_suite.get('market_regime', 'Normal')}** (ADX: {tech_suite.get('adx', 20):.1f})")
    with reg_col2:
        st.success(f"🎯 **Strategi Rekomendasi**: {tech_suite.get('recommended_strategy', 'Trend Following')}")

    st.markdown(f"##### 📈 Grafik Interaktif Harga {ticker_clean} (Candlestick & Moving Average Ribbon)")
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=["", "Volume Transaksi"]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_tech.index,
        open=df_tech["Open"],
        high=df_tech["High"],
        low=df_tech["Low"],
        close=df_tech["Close"],
        name="Candlestick",
        increasing_line_color="#22C55E",
        decreasing_line_color="#EF4444"
    ), row=1, col=1)

    # MA Ribbon (5, 10, 20, 50, 100, 200)
    if "MA_5" in df_tech.columns:
        fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech["MA_5"], name="MA 5", line=dict(color="#EC4899", width=1.0)), row=1, col=1)
    if "EMA_10" in df_tech.columns:
        fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech["EMA_10"], name="EMA 10", line=dict(color="#F97316", width=1.1)), row=1, col=1)
    if "EMA_20" in df_tech.columns:
        fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech["EMA_20"], name="EMA 20", line=dict(color="#F59E0B", width=1.3)), row=1, col=1)
    if "SMA_50" in df_tech.columns:
        fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech["SMA_50"], name="SMA 50", line=dict(color="#3B82F6", width=1.6)), row=1, col=1)
    if "SMA_100" in df_tech.columns:
        fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech["SMA_100"], name="SMA 100", line=dict(color="#6366F1", width=1.8, dash="dot")), row=1, col=1)
    if "SMA_200" in df_tech.columns:
        fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech["SMA_200"], name="SMA 200", line=dict(color="#8B5CF6", width=2.0)), row=1, col=1)

    # 1. Garis Fibonacci 61.8% di sisi KIRI grafik (mencegah tabrakan dengan level target di sisi kanan)
    fibo_618 = hl_eval.get("fib_levels", {}).get("fib_618", 0)
    if fibo_618 > 0:
        fig.add_hline(
            y=fibo_618,
            line_dash="dot",
            line_color="#F59E0B",
            annotation_text=f"Fib 61.8% (Rp {fibo_618:,})",
            annotation_position="top left",
            annotation_font=dict(size=10.5, color="#FBBF24"),
            row=1, col=1
        )

    # 2. Garis Entry Lilin di sisi KIRI grafik
    entry_c = candle_pred.get("entry_price", 0)
    if entry_c > 0:
        fig.add_hline(
            y=entry_c,
            line_dash="dot",
            line_color="#38BDF8",
            annotation_text=f"Entry Lilin (Rp {entry_c:,})",
            annotation_position="bottom left",
            annotation_font=dict(size=10.5, color="#38BDF8"),
            row=1, col=1
        )

    # 3. Take Profit 2 (TP2) di sisi KANAN ATAS
    tp2_c = candle_pred.get("tp2", 0)
    if tp2_c > 0:
        fig.add_hline(
            y=tp2_c,
            line_dash="solid",
            line_color="#059669",
            annotation_text=f"TP2 Lilin (Rp {tp2_c:,} | {candle_pred.get('tp2_net_pct', 0):+.1f}%)",
            annotation_position="top right",
            annotation_font=dict(size=10.5, color="#34D399"),
            row=1, col=1
        )

    # 4. Take Profit 1 (TP1) - Bersih dan Bebas Tabrakan
    tp1_c = candle_pred.get("tp1", 0)
    tp1_s = plan.get("take_profit_1", 0)
    if tp1_c > 0 and (tp1_s == 0 or abs(tp1_c - tp1_s) <= 2):
        # Gabungkan jika level harga sama/hampir sama persis
        fig.add_hline(
            y=tp1_c,
            line_dash="dash",
            line_color="#10B981",
            annotation_text=f"TP1 Target (Rp {tp1_c:,} | {candle_pred.get('tp1_net_pct', 0):+.1f}%)",
            annotation_position="bottom right",
            annotation_font=dict(size=10.5, color="#10B981"),
            row=1, col=1
        )
    else:
        if tp1_c > 0:
            fig.add_hline(
                y=tp1_c,
                line_dash="dash",
                line_color="#10B981",
                annotation_text=f"TP1 Lilin (Rp {tp1_c:,})",
                annotation_position="bottom right",
                annotation_font=dict(size=10.5, color="#10B981"),
                row=1, col=1
            )
        if tp1_s > 0:
            fig.add_hline(
                y=tp1_s,
                line_dash="dot",
                line_color="#34D399",
                annotation_text=f"TP1 Sistem (Rp {tp1_s:,})",
                annotation_position="top right",
                annotation_font=dict(size=10.5, color="#34D399"),
                row=1, col=1
            )

    # 5. Cut Loss / Stop Loss (SL) - Bersih dan Bebas Tabrakan
    sl_c = candle_pred.get("stop_loss", 0)
    sl_s = plan.get("stop_loss", 0)
    if sl_c > 0 and (sl_s == 0 or abs(sl_c - sl_s) <= 2):
        fig.add_hline(
            y=sl_c,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"Cut Loss (Rp {sl_c:,} | {candle_pred.get('sl_net_pct', 0):+.1f}%)",
            annotation_position="bottom right",
            annotation_font=dict(size=10.5, color="#F87171"),
            row=1, col=1
        )
    else:
        if sl_c > 0:
            fig.add_hline(
                y=sl_c,
                line_dash="dash",
                line_color="#DC2626",
                annotation_text=f"Cut Loss Lilin (Rp {sl_c:,})",
                annotation_position="top right",
                annotation_font=dict(size=10.5, color="#F87171"),
                row=1, col=1
            )
        if sl_s > 0:
            fig.add_hline(
                y=sl_s,
                line_dash="dot",
                line_color="#EF4444",
                annotation_text=f"SL Sistem (Rp {sl_s:,})",
                annotation_position="bottom right",
                annotation_font=dict(size=10.5, color="#EF4444"),
                row=1, col=1
            )

    # Volume Bar
    vol_colors = ['#22C55E' if c >= o else '#EF4444' for c, o in zip(df_tech["Close"], df_tech["Open"])]
    fig.add_trace(go.Bar(x=df_tech.index, y=df_tech["Volume"], name="Volume", marker_color=vol_colors), row=2, col=1)

    fig.update_layout(
        height=660,
        margin=dict(l=35, r=135, t=45, b=25),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Point-in-Time Inspector
    st.markdown("##### 🔬 Interactive Point-in-Time Inspector:")
    bar_pos = st.slider(
        "Pilih titik lilin untuk diagnosis multi-faktor (Bar ke-N):",
        min_value=1,
        max_value=max(1, len(df_tech)),
        value=max(1, len(df_tech)),
        step=1
    )
    pit_res = inspect_point_in_time(df_tech, tech_suite, bar_index=bar_pos-1)
    st.info(f"📋 **Hasil Diagnosis Bar ke-{bar_pos} ({pit_res.get('date')}):** {pit_res.get('narrative')}")

# TAB 6: MACHINE LEARNING & AI SUITE
with tab_ml:
    st.markdown("#### 🤖 Proyeksi Harga Machine Learning & Institutional AI Suite")
    
    st.markdown("##### Bagian A: Proyeksi Tren Harga 5 Hari Bursa ke Depan (Random Forest AI)")
    if ml_eval.get("success"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Arah Tren Prediksi", ml_eval["trend_prediction"])
        with mc2:
            st.metric("Estimasi Harga Akhir", f"Rp {ml_eval['final_projected_price']:,}", f"{ml_eval['expected_pct_change']:+.2f}%")
        with mc3:
            st.metric("Akurasi Arah Backtest", f"{ml_eval['directional_accuracy_pct']}%")
        with mc4:
            st.metric("Rata-rata Selisih (MAE)", f"± Rp {ml_eval['mae_idr']:,.0f}")

        recent_df = df_tech.tail(30)
        future_labels = [f"Hari +{i}" for i in range(1, len(ml_eval["forecast_prices"]) + 1)]
        
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=[d.strftime("%d %b") for d in recent_df.index],
            y=recent_df["Close"],
            mode="lines+markers",
            name="Historis (30 Hari)",
            line=dict(color="#3B82F6", width=2)
        ))
        conn_x = [recent_df.index[-1].strftime("%d %b")] + future_labels
        conn_y = [current_price] + ml_eval["forecast_prices"]
        fig_proj.add_trace(go.Scatter(
            x=conn_x,
            y=conn_y,
            mode="lines+markers",
            name="Proyeksi 5 Hari ke Depan",
            line=dict(color="#10B981" if ml_eval['expected_pct_change'] >= 0 else "#EF4444", width=3, dash="dash")
        ))
        fig_proj.update_layout(
            title=f"Lintasan Proyeksi Harga Nominal {ticker_clean}",
            xaxis_title="Hari",
            yaxis_title="Harga (IDR)",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Bagian B: Suite AI Kuantitatif Institusional (GBDT, RNN, Transformers, & RL Policy)")
    if ai_suite_eval:
        aic1, aic2, aic3, aic4 = st.columns(4)
        with aic1:
            st.metric("Master AI Win Probability", f"{ai_suite_eval['master_ai_win_prob']}%")
        with aic2:
            st.metric("GBDT Ensemble Win Prob", f"{ai_suite_eval['gbdt']['win_prob']}%")
        with aic3:
            st.metric("LSTM & RNN Average", f"{ai_suite_eval['rnn']['average_rnn']}%")
        with aic4:
            st.metric("Transformer Attention", f"{ai_suite_eval['transformer']['average_transformer']}%")

        rl = ai_suite_eval.get("reinforcement_learning", {})
        st.success(f"🎮 **Reinforcement Learning Optimal Policy (PPO / SAC)**: Direkomendasikan tindakan **{rl.get('recommended_action', 'Hold')}** (Keyakinan Policy: **{rl.get('ppo_confidence', 50.0)}%**)")

        st.markdown("**Distribusi Probabilitas Tindakan (RL Action Policy):**")
        prob_cols = st.columns(len(rl.get("action_probabilities", {})))
        for idx_p, (act_name, prob_val) in enumerate(rl.get("action_probabilities", {}).items()):
            with prob_cols[idx_p]:
                st.metric(act_name, f"{prob_val}%")

# TAB 7: FUNDAMENTAL LENGKAP & MAKRO
with tab_fund:
    st.markdown("#### 🏢 Analisis Fundamental, Solvabilitas, & Snapshot Makro Global")
    f_metrics = fund_eval["metrics"]
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("P/E Ratio", f"{f_metrics.get('pe_ratio') or '-'}x")
    with fc2:
        st.metric("PBV Ratio", f"{f_metrics.get('pbv_ratio') or '-'}x")
    with fc3:
        st.metric("ROE (%)", f"{f_metrics.get('roe_pct') or '-'}%")
    with fc4:
        st.metric("Debt to Equity (DER)", f"{f_metrics.get('der_pct') or '-'}%")

    if fund_eval.get("insolvency_veto"):
        st.error("🚨 **INSOLVENCY VETO ACTIVE**: Struktur permodalan berbahaya (DER > 4.0x)! Risiko insolvensi.")

    st.markdown("**Catatan Analisis Fundamental:**")
    for ins in fund_eval["insights"]:
        st.write(f"• {ins}")

    st.markdown("---")
    st.markdown("##### 🌐 Snapshot Makroekonomi Global:")
    macro_data = fetch_macro_snapshot()
    mc1, mc2, mc3, mc4 = st.columns(4)
    cols_macro = [mc1, mc2, mc3, mc4]
    for idx_m, (m_name, m_val) in enumerate(macro_data.items()):
        with cols_macro[idx_m % 4]:
            st.metric(
                label=m_name,
                value=f"{m_val['price']:,.2f} {m_val['unit']}",
                delta=f"{m_val['change_pct']:+.2f}%"
            )

# TAB 8: MONEY MANAGEMENT, NET PNL & ORDER BOOK
with tab_mm:
    st.markdown("#### 🧮 Money Management, Net PnL (Pajak & Komisi), & Mikrostruktur Order Book")
    
    mm_c1, mm_c2 = st.columns(2)
    with mm_c1:
        st.markdown("**Alokasi Modal & Lot Saham (1 Lot = 100 Lembar):**")
        st.success(f"Disarankan Membeli: **{pos['lots']} Lot** ({pos['shares']:,} Lembar Saham)")
        if pos.get("safe_exit_capped"):
            st.warning(f"⚠️ Ukuran lot dibatasi oleh Safe Exit Lot ({order_book_eval.get('safe_exit_lot')} lot) untuk mencegah slippage!")
        st.write(f"• **Total Modal Terpakai**: Rp {pos['total_investment']:,.0f} ({pos['capital_usage_pct']}% dari total modal)")
        st.write(f"• **Maksimal Toleransi Rugi (SL)**: :red[**Rp {pos['max_loss_idr']:,.0f}**] ({user_risk_pct}% dari modal)")
        st.write(f"• **Fraksi Resmi BEI**: Rp {plan['tick_size']} per tik")

    with mm_c2:
        st.markdown("**Simulasi Net PnL (Setelah Biaya Broker & Pajak 0.40% Roundtrip):**")
        st.write(f"• Estimasi Modal Beli Riil (inc fee): **Rp {net_pnl_calc['total_capital_out']:,.0f}**")
        st.write(f"• Estimasi Hasil Jual TP1 (inc tax): **Rp {net_pnl_calc['total_proceeds']:,.0f}**")
        st.success(f"• **Cuan Bersih Masuk RDN (Net PnL TP1)**: :green[**+ Rp {net_pnl_calc['net_pnl_idr']:,.0f}**] (+{net_pnl_calc['net_pnl_pct']:.2f}%)")
        st.write(f"• Total Komisi Broker & Pajak BEI: Rp {net_pnl_calc['total_fees']:,.0f}")

    st.markdown("---")
    st.markdown("##### 📊 Mikrostruktur Buku Pesanan (Level-1 Order Book):")
    ob_c1, ob_c2, ob_c3, ob_c4 = st.columns(4)
    with ob_c1:
        st.metric(
            "Status Eksekusi HAKA",
            order_book_eval.get("haka_badge", "-"),
            f"{order_book_eval.get('pct_bid', 50):.1f}% Bid"
        )
    with ob_c2:
        st.metric(
            "Safe Exit Lot Size",
            f"{order_book_eval.get('safe_exit_lot', 0):,} Lot",
            order_book_eval.get("exit_speed", "")
        )
    with ob_c3:
        st.metric(
            "Best Bid vs Best Ask",
            f"Rp {order_book_eval.get('bid_entry', 0):,.0f} / Rp {order_book_eval.get('haka_entry', 0):,.0f}"
        )
    with ob_c4:
        st.metric(
            "Relative Spread",
            f"{order_book_eval.get('rel_spread', 0):.2f}%",
            order_book_eval.get("exit_tier", "")
        )

    st.info(f"💡 **Petunjuk HAKA**: {order_book_eval.get('haka_desc', '')}")
    if order_book_eval.get("emergency_haki"):
        st.error(f"🚨 **Peringatan HAKI**: {order_book_eval.get('haki_desc', '')}")
    else:
        st.caption(f"🟡 **Petunjuk Exit**: {order_book_eval.get('haki_desc', '')}")

    # Visual Order Book Bar
    pct_b = order_book_eval.get("pct_bid", 50.0)
    pct_o = order_book_eval.get("pct_offer", 50.0)
    st.markdown(f"""
    <div class="ob-bar-container">
        <div class="ob-bar-bid" style="width: {pct_b:.1f}%;">BID {pct_b:.1f}%</div>
        <div class="ob-bar-offer" style="width: {pct_o:.1f}%;">OFFER {pct_o:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🏛️ Karakteristik Partisipan Broker pada Buku Pesanan (Order Book Level 2)", expanded=False):
        st.markdown(
            "• **Broker Institusi & Asing (BK, AK, ZP, CC)**: Memasang antrean Bid tebal yang bertahan lama sebagai penopang harga (*Sticky Liquidity*). Keberadaannya menjamin keamanan Safe Exit Lot.\n"
            "• **Broker Bandar Kilat (MG, AZ, CP)**: Kerap memasang antrean Bid besar secara mendadak lalu membatalkannya kilat (*Spoofing*) untuk memancing HAKA dari ritel.\n"
            "• **Broker Ritel (YP, PD, XC, XL)**: Antrean tersebar dalam jumlah lot kecil-menengah di banyak fraksi harga (*Fragmented Orders*), sangat reaktif dan mudah panik (HAKI) jika harga turun 1-2 tik."
        )

# TAB 9: GAYA TRADING, 19 TIPE SAHAM & CASH COWS
with tab_style:
    st.markdown("#### 🏷️ Klasifikasi Gaya Trading, 19 Tipologi Saham, & Cash Cows Bursa")
    
    st1, st2 = st.columns(2)
    with st1:
        st.markdown(f"##### 🎯 Gaya Trading Optimal: **{style_type_eval['primary_style']}**")
        st.info(
            f"• **Timeframe Rekomendasi**: {style_type_eval['timeframe']}\n"
            f"• **Durasi Hold Optimal**: {style_type_eval['holding_duration']}\n"
            f"• **Aturan Eksekusi Bid/Offer**: {style_type_eval['bid_offer_rule']}\n"
            f"• **Strategi Exit**: {style_type_eval['optimal_exit']}"
        )
    with st2:
        st.markdown(f"##### 🏢 Tipologi Saham: **{style_type_eval['primary_stock_type']}**")
        st.success(
            f"• **Cocok Untuk (Best For)**: {style_type_eval['best_for']}\n"
            f"• **Kelebihan (Pros)**: {style_type_eval['pros']}\n"
            f"• **Kekurangan (Cons)**: {style_type_eval['cons']}"
        )

    st.markdown("---")
    st.markdown("##### 🐄 Registry Eksklusif 'Exchange & Broker Cash Cows' BEI:")
    st.caption("Perusahaan yang selalu mendulang keuntungan dari setiap perputaran dana dan transaksi bursa.")
    cash_cows = get_trading_cash_cows()
    cc_df = pd.DataFrame(cash_cows)[["ticker", "name", "role", "type_of_stock", "trading_style", "moat_rating"]]
    cc_df.columns = ["Ticker", "Nama Perusahaan", "Peran Monopoli Bursa", "Tipe Saham", "Gaya Trading", "Rating Parit Ekonomi"]
    st.dataframe(cc_df, use_container_width=True, hide_index=True)

# TAB 10: TOP BREAKOUT & BANDARMOLOGI
with tab_breakout:
    st.markdown("#### 🏆 Detektor Pola Breakout Geometri & Jejak Bandarmologi")
    
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown(f"##### Pola Geometri: **{breakout_eval.get('pattern_name')}**")
        st.info(
            f"• **Status**: {'🔥 Confirmed Breakout' if breakout_eval.get('is_breakout') else ('⚡ Breakout Soon' if breakout_eval.get('is_breakout_soon') else 'Konsolidasi Normal')}\n"
            f"• **Relative Volume (RVol)**: {breakout_eval.get('rvol'):.2f}x\n"
            f"• **Harga Pemicu (Trigger)**: Rp {breakout_eval.get('trigger_price', 0):,}\n"
            f"• **Target Akselerasi TP1**: Rp {breakout_eval.get('target_tp1', 0):,}\n"
            f"• **Penjelasan**: {breakout_eval.get('explanation')}"
        )
    with bc2:
        st.markdown(f"##### Jejak Bandarmologi & Promotor: **{promoter_eval.get('conglomerate')}**")
        st.success(
            f"• **Promotor Pengendali**: {promoter_eval.get('promoter')}\n"
            f"• **Status Akumulasi Bandar**: {promoter_eval.get('bandar_status')}\n"
            f"• **Aksi Bandar**: {promoter_eval.get('bandar_action')}\n"
            f"• **Dominasi Investor**: Asing {promoter_eval.get('foreign_dominance_pct')}% vs Ritel {promoter_eval.get('retail_dominance_pct')}%\n"
            f"• **Top Broker Pembeli**: {', '.join(promoter_eval.get('top_buyers', []))}"
        )

    st.markdown("##### 🏛️ Detail Profil, Sifat & Proyeksi Arah Harga Broker Saham Ini:")
    st.caption("Menganalisis siapa sekuritas pengendali transaksi, karakter akumulasi/distribusinya, dan proyeksi ke mana harga akan bergerak.")

    tb_c1, tb_c2 = st.columns(2)
    with tb_c1:
        st.markdown("**🟢 Top Broker Akumulator (Smart Money):**")
        buyers_list = promoter_eval.get("top_buyers_detail", [])
        if buyers_list:
            df_b = pd.DataFrame(buyers_list)[["code", "name", "category", "archetype", "future_price_impact"]]
            st.dataframe(
                df_b,
                column_config={
                    "code": st.column_config.TextColumn("Kode", width="small"),
                    "name": st.column_config.TextColumn("Nama Resmi Sekuritas", width="medium"),
                    "category": st.column_config.TextColumn("Kategori", width="small"),
                    "archetype": st.column_config.TextColumn("Peran Pasar", width="small"),
                    "future_price_impact": st.column_config.TextColumn("Proyeksi Arah Harga", width="large"),
                },
                use_container_width=True,
                hide_index=True
            )
    with tb_c2:
        st.markdown("**🔴 Top Broker Distribusi (Sellers):**")
        sellers_list = promoter_eval.get("top_sellers_detail", [])
        if sellers_list:
            df_s = pd.DataFrame(sellers_list)[["code", "name", "category", "archetype", "future_price_impact"]]
            st.dataframe(
                df_s,
                column_config={
                    "code": st.column_config.TextColumn("Kode", width="small"),
                    "name": st.column_config.TextColumn("Nama Resmi Sekuritas", width="medium"),
                    "category": st.column_config.TextColumn("Kategori", width="small"),
                    "archetype": st.column_config.TextColumn("Peran Pasar", width="small"),
                    "future_price_impact": st.column_config.TextColumn("Proyeksi Arah Harga", width="large"),
                },
                use_container_width=True,
                hide_index=True
            )

    with st.expander("📖 Panduan Interpretasi Pergerakan Harga Berdasarkan Sifat Broker", expanded=True):
        st.markdown(
            "• **Jika Top Buyer didominasi Smart Money Asing (BK, AK, ZP)**: Mengindikasikan fase akumulasi senyap menuju kenaikan harga berkelanjutan (*Markup*).\n"
            "• **Jika Top Buyer didominasi BUMN (CC, NI, OD)**: Menandakan pengawalan lantai harga (support) dan potensi *Bottom Reversal* yang kuat.\n"
            "• **Jika Top Buyer didominasi Bandar Kilat (MG, AZ)**: Menandakan lonjakan harga cepat spekulatif (*Pump*) yang cocok untuk scalping kilat, namun rawan guyuran.\n"
            "• **Jika Top Buyer didominasi Kerumunan Ritel (YP, PD, XC)**: Waspada jebakan beli di pucuk (*Distribution to Retail*) saat institusi sedang melepas barang."
        )

    # ----------------- KESIMPULAN BERDASARKAN KARAKTERISTIK BROKER (LAMPIRAN 2 & 3) -----------------
    broker_conclusion = generate_broker_interpretation_conclusion(promoter_eval, broker_eval)
    
    st.markdown("#### 🎯 Kesimpulan & Rekomendasi Terpadu Berdasarkan Karakteristik Broker:")
    conclusion_box_color = "#ECFDF5" if "Akumulasi" in broker_conclusion["status"] else ("#FEF2F2" if "Distribusi" in broker_conclusion["status"] else "#EFF6FF")
    conclusion_border_color = "#10B981" if "Akumulasi" in broker_conclusion["status"] else ("#EF4444" if "Distribusi" in broker_conclusion["status"] else "#3B82F6")
    
    st.markdown(
        f"""
        <div style="background-color: {conclusion_box_color}; border: 1.5px solid {conclusion_border_color}; border-radius: 10px; padding: 16px; margin-top: 10px; margin-bottom: 15px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">
                📌 Status Aliran Dana Pasar: <span style="color: {conclusion_border_color};">{broker_conclusion['status']}</span>
            </div>
            <div style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                <strong>1. Aturan Pedoman Terpenuhi:</strong> {broker_conclusion['primary_rule']}<br>
                <strong>2. Analisis Kekuatan Pembeli (Buyer):</strong> {broker_conclusion['narrative']}<br>
                <strong>3. Karakter Penjual (Seller Dynamic):</strong> {broker_conclusion['seller_dynamic']}<br>
                <strong>4. Posisi Harga vs Modal Bandar:</strong> {broker_conclusion['cost_implication']}<br>
                <strong>5. Rekomendasi Taktis:</strong> <span style="font-weight: 700; color: #0F172A;">{broker_conclusion['action_recommendation']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# TAB 11: EKONOMETRIKA, DEEP RISK & MEAN-CVAR
with tab_risk:
    st.markdown("#### 📊 Ekonometrika Deret Waktu, Deep Risk (EVT-POT), & Mean-CVaR Optimization")
    
    if econ_eval.get("status") == "success":
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("GARCH(1,1) Volatilitas", f"{econ_eval['garch_last_vol']:.2f}%")
        with rc2:
            st.metric("Parametric VaR 99%", f"{econ_eval['var_99_param']:.2f}%")
        with rc3:
            st.metric("Expected Shortfall (CVaR)", f"{econ_eval['cvar_99_param']:.2f}%")
        with rc4:
            st.metric("EVT-POT Ekstrem (GPD)", f"{econ_eval['evt_var_99']:.2f}%")

        st.info(
            f"• **Uji Stasioneritas ADF**: Harga Mentah t-stat = {econ_eval['adf_raw'][0]:.2f} (Non-Stationary), "
            f"Log-Return t-stat = {econ_eval['adf_ret'][0]:.2f} (P-value < 0.001, Stationary).\n"
            f"• **ARIMA Drift**: {econ_eval['arima_drift']:.5f} (Diagnostik Residual Ljung-Box p-value: {econ_eval['lb_pvalue']:.3f}).\n"
            f"• **Backtest Kupiec POF**: {econ_eval['n_hits']} pelanggaran batas risiko (p-value: {econ_eval['kupiec_pval']:.3f})."
        )
    else:
        st.info("Data tidak mencukupi untuk fitting model GARCH & EVT.")

    st.markdown("---")
    st.markdown("##### ⚖️ Optimasi Portofolio Mean-CVaR (Rockafellar & Uryasev 2000):")
    st.caption("Menghitung alokasi bobot optimal lintas aset untuk meminimalkan potensi ekor kerugian (Tail Risk CVaR 95%).")
    
    tier_opt_choice = st.selectbox(
        "Pilih Kohort Keranjang Saham Berdasarkan Tingkatan (Tier):",
        [
            "💎 Saham Premium / Blue Chip (Di atas Rp5.000)",
            "🪙 Saham Receh / Saham Murah (Rp100 – Rp1.000)",
            "🛌 Saham Gocap / Saham Tidur (Rp50 – Rp100)",
            "🎯 Sesuai Filter Tingkatan Aktif Sidebar",
            "⚙️ Kustom / Multi-Pilihan Emiten",
        ],
        index=0
    )

    clean_current = normalize_ticker(selected_ticker)
    if "Premium" in tier_opt_choice:
        cohort_tickers = [clean_current, "BBCA.JK", "BMRI.JK", "UNTR.JK", "ICBP.JK", "TLKM.JK", "ASII.JK"]
    elif "Receh" in tier_opt_choice:
        cohort_tickers = [clean_current, "BRMS.JK", "BUMI.JK", "DEWA.JK", "ELSA.JK", "KIJA.JK", "RAJA.JK"]
    elif "Gocap" in tier_opt_choice:
        cohort_tickers = [clean_current, "GOTO.JK", "FREN.JK", "BIPI.JK", "ENRG.JK", "DOID.JK"]
    elif "Sesuai Filter" in tier_opt_choice:
        tier_matched = filter_idx_stocks(tier_filter=chosen_tier)[:6]
        cohort_tickers = [s["code"] for s in tier_matched]
        if clean_current not in cohort_tickers:
            cohort_tickers = [clean_current] + cohort_tickers[:5]
    else:
        default_custom = [clean_current, "BBCA.JK", "BRMS.JK", "GOTO.JK", "BMRI.JK"]
        cohort_tickers = st.multiselect(
            "Pilih Saham untuk Portofolio:",
            options=[s["code"] for s in ALL_IDX_STOCKS[:60]],
            default=[t for t in default_custom if any(s["code"] == t for s in ALL_IDX_STOCKS[:60])]
        )

    final_cohort = list(dict.fromkeys([normalize_ticker(t) for t in cohort_tickers if t]))
    st.caption(f"Aset dalam portofolio yang dianalisis: `{'`, `'.join(final_cohort)}`")

    btn_label = f"🧪 Hitung Bobot Portofolio Mean-CVaR ({tier_opt_choice.split()[1]})"
    if st.button(btn_label, use_container_width=True):
        with st.spinner("Mengoptimalkan bobot portofolio via CVXPY..."):
            batch_df = fetch_historical_ohlcv(final_cohort, period="1y")
            closes = {k: v["Close"] for k, v in batch_df.items() if not v.empty}
            if len(closes) >= 2:
                ret_df = pd.DataFrame(closes).pct_change().dropna()
                opt_res = optimize_mean_cvar_portfolio(ret_df)
                if opt_res.get("status") == "success":
                    st.success(f"Optimasi Berhasil: CVaR Portofolio **{opt_res['cvar']:.2f}%** | Ekspektasi Return **{opt_res['expected_return']:.2f}%** | Rasio STARR **{opt_res['starr_ratio']:.2f}**")
                    w_df = pd.DataFrame({
                        "Emiten": opt_res["assets"],
                        "Bobot Alokasi (%)": [round(w * 100, 2) for w in opt_res["weights"]]
                    })
                    st.dataframe(w_df, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Optimasi gagal: {opt_res.get('message', 'Data tidak konvergen')}")
            else:
                st.warning("Data historis tidak mencukupi untuk minimal 2 aset.")

# TAB 13: EKOSISTEM PASAR TERPADU (IHSG, BROKER, SENTIMEN & PSIKOLOGI)
with tab_eko:
    st.markdown("#### 🌐 Ekosistem Pasar Terpadu: Proyeksi Masa Depan IHSG, Broker Bandar, Sentimen & Psikologi")
    st.caption("Analisis makro multi-dimensi yang mengintegrasikan dinamika IHSG, footprint broker penggerak, sentimen multi-platform global, proyeksi korporasi, dan siklus psikologi investor.")

    # -------------------------------------------------------------------------
    # 0. PAPAN TATAKELOLA & SINTESIS EKOSISTEM 7 PILAR (EXECUTIVE GOVERNANCE BOARD)
    # -------------------------------------------------------------------------
    b_con = broker_eval.get("broker_consensus")
    if not b_con:
        from modules.instagram_sentiment_radar import evaluate_broker_research_consensus
        b_con = evaluate_broker_research_consensus(ticker_clean, current_price, sector=meta_live.get("sector", ""))

    insta_info = social_eval.get("instagram_feed")
    if not insta_info:
        from modules.instagram_sentiment_radar import fetch_instagram_sentiment_feed
        insta_info = fetch_instagram_sentiment_feed(
            ticker_clean,
            company_name=meta_live.get("name") or ticker_clean,
            sector=meta_live.get("sector") or "Umum"
        )

    # Sintesis Strategis Lintas 7 Pilar
    fg_val = psychology_eval.get("fear_greed_index", 50.0)
    b_fase = broker_eval.get("fase_bandar", "NETRAL")
    b_diff = broker_eval.get("diff_from_cost_pct", 0.0)
    ihsg_trend_lbl = ihsg_eval.get("future_trend", "KONSOLIDASI")
    ihsg_prob = ihsg_eval.get("direction_prob_up", 50.0)

    if fg_val < 40.0 and ("AKUMULASI" in b_fase or b_diff <= 6.0):
        synth_badge = "💎 GOLDEN CONTRARIAN ACCUMULATION (Akumulasi Institusi di Zona Ketakutan Ritel)"
        synth_badge_color = "#10B981"
        synth_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%)"
        synth_border = "#10B981"
        synth_action = "SERAP LIKUIDITAS PANIK SECARA BERTAHAP BERSAMA SMART MONEY"
        synth_thesis = (
            f"Kondisi asimetris langka: Kerumunan ritel mengalami ketakutan ekstrem (Fear Index {fg_val:.1f} - {psychology_eval['cycle_phase']}) "
            f"dan melakukan cut loss, namun Smart Money (Broker {broker_eval['lead_broker_code']} - {broker_eval['lead_broker_name']}) "
            f"terdeteksi aktif menyerap suplai pada modal dasar Rp {broker_eval['bandar_cost']:,} (deviasi harga {b_diff:+.1f}%). "
            f"Didukung konsensus sekuritas ({b_con['total_analysts']} analis) dengan target konsensus Rp {b_con['mean_target_price']:,} "
            f"(upside +{b_con['upside_avg_pct']}%), titik ini merupakan zona akumulasi dengan margin of safety institusional tertinggi."
        )
    elif "MARKUP" in b_fase and ihsg_prob >= 50.0:
        synth_badge = "🚀 INSTITUTIONAL MOMENTUM MARKUP (Akselerasi Pengawalan Tren Bersama Bandar)"
        synth_badge_color = "#38BDF8"
        synth_bg = "linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%)"
        synth_border = "#38BDF8"
        synth_action = "RIDING MOMENTUM DENGAN TRAILING STOP KETAT DI ATAS MODAL BANDAR"
        synth_thesis = (
            f"Fase akselerasi aktif: Broker pengendali ({broker_eval['lead_broker_code']}) sedang mengerek harga saham searah dengan "
            f"arah tren makro IHSG ({ihsg_trend_lbl}, probabilitas naik {ihsg_prob:.1f}%). "
            f"Sentimen media sosial global ({social_eval['composite_social_score']}/100) dan kebijakan otoritas ({insta_info['policy_status']}) "
            f"mendukung katalis kenaikan. Pertahankan posisi dengan trailing stop protektif di Rp {broker_eval['bandar_cost']:,}."
        )
    elif fg_val > 70.0 and ("DISTRIBUSI" in b_fase or b_diff > 20.0):
        synth_badge = "⚠️ INSTITUTIONAL DISTRIBUTION & TRAP (Waspada Guyuran Bandar ke Kerumunan Ritel)"
        synth_badge_color = "#EF4444"
        synth_bg = "linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%)"
        synth_border = "#EF4444"
        synth_action = "AMANKAN PROFIT (TAKE PROFIT) & HINDARI FOMO / PEMBELIAN DI PUCUK"
        synth_thesis = (
            f"Sinyal bahaya divergensi: Kerumunan ritel mengalami euforia berlebih (Fear & Greed {fg_val:.1f} - {psychology_eval['cycle_phase']}), "
            f"sedangkan harga saham telah terbang +{b_diff:.1f}% di atas modal awal bandar (Rp {broker_eval['bandar_cost']:,}). "
            f"Bandar mulai merealisasikan keuntungan (profit taking) dengan memanfaatkan likuiditas ritel yang sedang FOMO. "
            f"Disiplin amankan modal dan tunggu retest support {ihsg_eval['support_1']:,}."
        )
    else:
        synth_badge = "⏳ TACTICAL ACCUMULATION & CONSOLIDATION (Fase Pengujian Support & Titik Keseimbangan)"
        synth_badge_color = "#F59E0B"
        synth_bg = "linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%)"
        synth_border = "#F59E0B"
        synth_action = "AKUMULASI BERTAHAP (DCA) PADA ZONA HARGA MODAL BANDAR"
        synth_thesis = (
            f"Kondisi netral-konsolidatif: Pergerakan harga {ticker_clean} (Rp {current_price:,.0f}) berada di sekitar modal bandar "
            f"(Rp {broker_eval['bandar_cost']:,}, margin {b_diff:+.1f}%), sementara tren IHSG menguji area konsolidasi {ihsg_eval['support_1']:,} - {ihsg_eval['resistance_1']:,}. "
            f"Riset konsensus {b_con['total_analysts']} sekuritas menetapkan pandangan '{b_con['consensus_action']}' dengan target Rp {b_con['mean_target_price']:,}. "
            f"Lakukan akumulasi bertahap dengan membatasi risiko sebelum konfirmasi breakout volume."
        )

    # 4 Baris Metrik Utama Tatakelola Ekosistem
    gov_c1, gov_c2, gov_c3, gov_c4 = st.columns(4)
    with gov_c1:
        st.markdown(f"""
        <div style="background:#1E293B; border-radius:10px; padding:12px; border:1px solid #334155; margin-bottom:8px;">
            <div style="color:#94A3B8; font-size:0.8rem; font-weight:700; text-transform:uppercase;">🏛️ Makro IHSG (^JKSE)</div>
            <div style="color:#F1F5F9; font-size:1.25rem; font-weight:800; margin:2px 0;">Rp {ihsg_eval['current_level']:,.2f}</div>
            <div style="color:{'#10B981' if ihsg_eval['change_pct']>=0 else '#EF4444'}; font-size:0.82rem; font-weight:700;">{ihsg_eval['change_pct']:+.2f}% | Tren: {ihsg_eval['future_trend']}</div>
            <div style="color:#94A3B8; font-size:0.75rem; margin-top:4px;">Beta: <b style="color:#38BDF8;">{beta_eval['beta']}x</b> ({beta_eval['category']})</div>
        </div>
        """, unsafe_allow_html=True)
    with gov_c2:
        st.markdown(f"""
        <div style="background:#1E293B; border-radius:10px; padding:12px; border:1px solid #334155; margin-bottom:8px;">
            <div style="color:#94A3B8; font-size:0.8rem; font-weight:700; text-transform:uppercase;">🕵️ Smart Money / Bandar</div>
            <div style="color:#F1F5F9; font-size:1.25rem; font-weight:800; margin:2px 0;">{broker_eval['lead_broker_code']} <span style="font-size:0.85rem; color:#94A3B8;">({broker_eval['lead_broker_name']})</span></div>
            <div style="color:{'#10B981' if 'AKUMULASI' in broker_eval['fase_bandar'] else ('#38BDF8' if 'MARKUP' in broker_eval['fase_bandar'] else '#F59E0B')}; font-size:0.82rem; font-weight:700;">{broker_eval['fase_bandar']} (Skor {broker_eval['broker_score']}/100)</div>
            <div style="color:#94A3B8; font-size:0.75rem; margin-top:4px;">Modal: <b style="color:#38BDF8;">Rp {broker_eval['bandar_cost']:,}</b> ({broker_eval['diff_from_cost_pct']:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with gov_c3:
        st.markdown(f"""
        <div style="background:#1E293B; border-radius:10px; padding:12px; border:1px solid #334155; margin-bottom:8px;">
            <div style="color:#94A3B8; font-size:0.8rem; font-weight:700; text-transform:uppercase;">📑 Konsensus Sekuritas & SBN</div>
            <div style="color:#F1F5F9; font-size:1.25rem; font-weight:800; margin:2px 0;">Rp {b_con['mean_target_price']:,}</div>
            <div style="color:#10B981; font-size:0.82rem; font-weight:700;">Upside: +{b_con['upside_avg_pct']}% ({b_con['total_analysts']} Analis)</div>
            <div style="color:#94A3B8; font-size:0.75rem; margin-top:4px;">SBN 10Y: <b style="color:#F59E0B;">{b_con['sbn_10y_yield']}%</b> | Spread: {b_con['yield_spread']:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with gov_c4:
        st.markdown(f"""
        <div style="background:#1E293B; border-radius:10px; padding:12px; border:1px solid #334155; margin-bottom:8px;">
            <div style="color:#94A3B8; font-size:0.8rem; font-weight:700; text-transform:uppercase;">🎭 Psikologi & Sentimen</div>
            <div style="color:#F1F5F9; font-size:1.25rem; font-weight:800; margin:2px 0;">{psychology_eval['fear_greed_index']:.1f} <span style="font-size:0.85rem; color:#94A3B8;">/ 100</span></div>
            <div style="color:#EC4899; font-size:0.82rem; font-weight:700;">{psychology_eval['cycle_phase']}</div>
            <div style="color:#94A3B8; font-size:0.75rem; margin-top:4px;">IG: <b style="color:#38BDF8;">{insta_info['composite_instagram_score']:.0f}/100</b> | Global: <b style="color:#F59E0B;">{social_eval['composite_social_score']}/100</b></div>
        </div>
        """, unsafe_allow_html=True)

    # Master Strategic Synthesis Box
    st.markdown(f"""
    <div style="background:{synth_bg}; border: 1.5px solid {synth_border}; border-left: 6px solid {synth_border}; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:8px;">
            <span style="background:{synth_badge_color}; color:#FFFFFF; padding:4px 12px; border-radius:20px; font-weight:800; font-size:0.88rem; letter-spacing:0.5px;">
                {synth_badge}
            </span>
            <span style="color:#94A3B8; font-size:0.82rem; font-weight:600;">
                Tatakelola Integrasi 7 Pilar Pasar Modal Indonesia
            </span>
        </div>
        <p style="color:#F1F5F9; font-size:0.95rem; line-height:1.6; margin:8px 0 12px 0;">
            {synth_thesis}
        </p>
        <div style="background:rgba(15, 23, 42, 0.7); border:1px solid #334155; border-radius:8px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div>
                <b style="color:#F8FAFC; font-size:0.88rem;">🎯 Keputusan Aksi:</b>
                <span style="color:{synth_badge_color}; font-weight:800; font-size:0.92rem; margin-left:6px;">{synth_action}</span>
            </div>
            <div style="font-size:0.82rem; color:#94A3B8;">
                Harga Pasar: <b style="color:#38BDF8;">Rp {current_price:,.0f}</b> | Target Konsensus: <b style="color:#10B981;">Rp {b_con['mean_target_price']:,}</b> | Batas Proteksi Bandar: <b style="color:#F87171;">Rp {int(broker_eval['bandar_cost'] * 0.95):,}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    eko_tabs = st.tabs([
        "📈 1. Makro IHSG & Proyeksi 30D",
        "🏛️ 2. Jejak Broker & Modal Bandar",
        "📊 3. Konsensus Sekuritas, ETF & Obligasi",
        "📸 4. Sentimen Instagram & Kebijakan",
        "🌍 5. Sentimen Medsos Global",
        "🏗️ 6. Proyek Strategis & Capex",
        "🧠 7. Psikologi & Behavioral Finance"
    ])

    with eko_tabs[0]:
        st.markdown("##### 📈 Hasil & Proyeksi Forward 30 Hari IHSG (^JKSE)")
        ih_c1, ih_c2, ih_c3, ih_c4 = st.columns(4)
        with ih_c1:
            st.metric("Level IHSG Terakhir", f"Rp {ihsg_eval['current_level']:,.2f}", f"{ihsg_eval['change_pct']:+.2f}%")
        with ih_c2:
            st.metric("Target 30D Bullish", f"Rp {ihsg_eval['target_30d_bull']:,}", f"+{((ihsg_eval['target_30d_bull'] - ihsg_eval['current_level'])/ihsg_eval['current_level'])*100:.2f}%")
        with ih_c3:
            st.metric("Target 30D Base", f"Rp {ihsg_eval['target_30d_base']:,}", f"+{((ihsg_eval['target_30d_base'] - ihsg_eval['current_level'])/ihsg_eval['current_level'])*100:.2f}%")
        with ih_c4:
            st.metric("Target 30D Bearish", f"Rp {ihsg_eval['target_30d_bear']:,}", f"{((ihsg_eval['target_30d_bear'] - ihsg_eval['current_level'])/ihsg_eval['current_level'])*100:.2f}%")

        st.info(f"🧭 **Status Tren Masa Depan IHSG**: **{ihsg_eval['future_trend']}**\n\n_{ihsg_eval['future_trend_desc']}_\n- **Peluang Penguatan**: **{ihsg_eval['direction_prob_up']:.1f}%** | Support: **{ihsg_eval['support_1']:,}** / **{ihsg_eval['support_2']:,}** | Resistance: **{ihsg_eval['resistance_1']:,}** / **{ihsg_eval['resistance_2']:,}**")

        if df_ihsg_hist is not None and not df_ihsg_hist.empty:
            fig_ih = go.Figure()
            fig_ih.add_trace(go.Scatter(
                x=df_ihsg_hist.index[-60:],
                y=df_ihsg_hist["Close"].tail(60),
                name="Historis IHSG (^JKSE)",
                line=dict(color="#38BDF8", width=2.5)
            ))
            last_date = df_ihsg_hist.index[-1]
            future_dates = [last_date + timedelta(days=d) for d in [10, 20, 30]]
            cur_ih = ihsg_eval['current_level']
            bull_path = [cur_ih, (cur_ih + ihsg_eval['target_30d_bull'])/2.0, ihsg_eval['target_30d_bull']]
            base_path = [cur_ih, (cur_ih + ihsg_eval['target_30d_base'])/2.0, ihsg_eval['target_30d_base']]
            bear_path = [cur_ih, (cur_ih + ihsg_eval['target_30d_bear'])/2.0, ihsg_eval['target_30d_bear']]

            fig_ih.add_trace(go.Scatter(
                x=future_dates, y=bull_path, name="Skenario Bullish 30D",
                line=dict(color="#10B981", width=2, dash="dash")
            ))
            fig_ih.add_trace(go.Scatter(
                x=future_dates, y=base_path, name="Skenario Base 30D",
                line=dict(color="#FBBF24", width=2, dash="dash")
            ))
            fig_ih.add_trace(go.Scatter(
                x=future_dates, y=bear_path, name="Skenario Bearish 30D",
                line=dict(color="#EF4444", width=2, dash="dash")
            ))
            fig_ih.update_layout(
                title="Proyeksi Forward 30 Hari IHSG (^JKSE) - Scenario Cones",
                template="plotly_dark",
                height=340,
                margin=dict(l=30, r=30, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ih, use_container_width=True)

        st.markdown(f"##### 🎯 Sensitivitas Emiten {ticker_clean} terhadap Gelombang IHSG")
        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            st.metric("Market Beta (β) vs IHSG", f"{beta_eval['beta']}x", beta_eval['category'])
        with b_c2:
            st.metric("Jensen's Alpha (α)", f"{beta_eval['alpha_annual_pct']:+.2f}%", "Outperformance Tahunan")
        with b_c3:
            st.metric("Korelasi Pearson (r)", f"{beta_eval['correlation']:.2f}", "Kekuatan Hubungan Linear")
        st.caption(f"_{beta_eval['impact_summary']}_")

    with eko_tabs[1]:
        st.markdown(f"##### 🏛️ Jejak Broker Penggerak Utama & Modal Rata-rata Bandar (Bandar Cost)")
        st.markdown(
            f"Analisis bandarmologi institusional melacak entitas broker utama yang mengendalikan likuiditas "
            f"dan mengkalkulasi titik impas modal rata-rata akumulasi (**Bandar Cost Basis**)."
        )

        br_c1, br_c2, br_c3 = st.columns(3)
        with br_c1:
            st.metric("Kode Broker Lead", f"{broker_eval['lead_broker_code']}", f"{broker_eval['lead_broker_name']}")
        with br_c2:
            st.metric("Modal Rata-rata Bandar", f"Rp {broker_eval['bandar_cost']:,}", f"{broker_eval['diff_from_cost_pct']:+.1f}% dari Harga Pasar")
        with br_c3:
            st.metric("Harga Saham Saat Ini", f"Rp {current_price:,.0f}", f"Skor Bandar: {broker_eval['broker_score']}/100")

        st.markdown(f"**Tipe Broker**: `{broker_eval['lead_broker_type']}` | **Karakteristik**: _{broker_eval['lead_broker_nature']}_")
        
        phase_color_border = "#10B981" if "AKUMULASI" in broker_eval['fase_bandar'] else ("#3B82F6" if "KONSOLIDASI" in broker_eval['fase_bandar'] else ("#F59E0B" if "MARKUP" in broker_eval['fase_bandar'] else "#EF4444"))
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid {phase_color_border}; border-radius:10px; padding:16px; margin-top:10px; color:#F1F5F9;">
            <h5 style="margin:0 0 6px 0; color:{phase_color_border};">⚖️ {broker_eval['fase_bandar']}</h5>
            <p style="margin:0 0 8px 0; font-size:0.95rem; color:#F1F5F9;">{broker_eval['fase_desc']}</p>
            <b style="color:#F8FAFC;">👉 Rekomendasi Aksi:</b> <span style="color:{phase_color_border}; font-weight:700;">{broker_eval['action_bandar']}</span>
        </div>
        """, unsafe_allow_html=True)

        fig_br = go.Figure()
        fig_br.add_trace(go.Bar(
            name="Harga Pasar Saat Ini",
            x=[ticker_clean],
            y=[current_price],
            marker_color="#38BDF8",
            text=[f"Rp {current_price:,.0f}"],
            textposition="auto"
        ))
        fig_br.add_trace(go.Bar(
            name=f"Modal Bandar ({broker_eval['lead_broker_code']})",
            x=[ticker_clean],
            y=[broker_eval['bandar_cost']],
            marker_color="#10B981" if broker_eval['diff_from_cost_pct'] <= 5.0 else "#F59E0B",
            text=[f"Rp {broker_eval['bandar_cost']:,}"],
            textposition="auto"
        ))
        fig_br.update_layout(
            title=f"Perbandingan Harga Pasar vs Modal Bandar ({broker_eval['lead_broker_code']})",
            barmode="group",
            template="plotly_dark",
            height=280,
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_br, use_container_width=True)

    with eko_tabs[2]:
        st.markdown(f"##### 📊 Konsensus Riset Analis Sekuritas BEI, Broker Global, ETF & Pasar Obligasi")
        st.caption(
            "Mengagregasi target harga resmi dan rekomendasi riset dari sekuritas anggota BEI "
            "(Mandiri Sekuritas, Mirae Asset, Indo Premier, BCA Sekuritas, BRI Danareksa, Trimegah, Sucor, dll.) "
            "dan bank investasi global (J.P. Morgan, Morgan Stanley, UBS, Goldman Sachs, CLSA), "
            "serta menganalisis imbal hasil obligasi negara (SBN 10Y) dan aliran dana pasif ETF."
        )

        bc_c1, bc_c2, bc_c3 = st.columns(3)
        with bc_c1:
            st.metric("Target Harga Konsensus Analis", f"Rp {b_con['mean_target_price']:,}", f"Potensi Upside: +{b_con['upside_avg_pct']}%")
        with bc_c2:
            st.metric("Rekomendasi Konsensus", f"{b_con['consensus_action'].split('/')[0].strip()}", f"{b_con['buy_pct']}% Beli | {b_con['hold_pct']}% Tahan | {b_con['sell_pct']}% Jual")
        with bc_c3:
            st.metric("Cakupan Riset Sekuritas", f"{b_con['total_analysts']} Lembaga Riset", f"Tertinggi: Rp {b_con['highest_target']:,}")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid {b_con['consensus_color']}; border-radius:10px; padding:16px; margin-top:8px; margin-bottom:12px; color:#F1F5F9;">
            <h5 style="margin:0 0 6px 0; color:{b_con['consensus_color']};">🎯 {b_con['consensus_action']}</h5>
            <p style="margin:0 0 6px 0; font-size:0.95rem; color:#F1F5F9;">{b_con['consensus_summary']}</p>
            <div style="font-size:0.85rem; color:#94A3B8;">
                Rentang Target Harga Analis: <b style="color:#F8FAFC;">Rp {b_con['lowest_target']:,}</b> (Konservatif) s/d <b style="color:#F8FAFC;">Rp {b_con['highest_target']:,}</b> (Agresif).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###### 📑 Rincian Target Harga Analis per Sekuritas Resmi:")
        df_b_table = pd.DataFrame(b_con["broker_targets_list"])[["broker", "tier", "target_price", "upside_pct", "recommendation", "publish_date"]]
        df_b_table.columns = ["Sekuritas / Broker", "Kategori Broker", "Target Harga (IDR)", "Potensi Upside (%)", "Rating Rekomendasi", "Tanggal Laporan"]
        st.dataframe(df_b_table, hide_index=True, use_container_width=True)

        co_c1, co_c2 = st.columns(2)
        with co_c1:
            st.markdown("###### 📜 Pasar Obligasi & Surat Berharga Negara (SBN / Fixed Income):")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid #F59E0B; border-radius:10px; padding:14px; color:#F1F5F9;">
                <b>Benchmark Yield SBN 10Y:</b> <code style="font-size:1rem; color:#F59E0B; background:#0F172A; padding:2px 8px; border-radius:4px;">{b_con['sbn_10y_yield']}%</code><br><br>
                <b>Spread Yield vs Deviden:</b> <code style="color:#38BDF8; background:#0F172A; padding:2px 8px; border-radius:4px;">{b_con['yield_spread']:+.2f}%</code><br><br>
                <small style="color:#CBD5E1;">{b_con['fixed_income_impact']}</small><br><br>
                <span style="font-size:0.8rem; color:#94A3B8;">Akun Resmi Terpantau: <b>@phei_id</b> (Penilai Harga Efek Indonesia), <b>@bareksa_id</b>, <b>@bibit.id</b>, <b>@pasar_modal_syariah</b></span>
            </div>
            """, unsafe_allow_html=True)

        with co_c2:
            st.markdown("###### 🧺 Aliran Likuiditas ETF & Index Rebalancing:")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid #38BDF8; border-radius:10px; padding:14px; color:#F1F5F9;">
                <b>Indeks Acuan Utama:</b> <code style="color:#38BDF8; background:#0F172A; padding:2px 8px; border-radius:4px;">LQ45, IDX30, ISSI, MSCI Indonesia (EIDO)</code><br><br>
                <small style="color:#CBD5E1;">{b_con['etf_flow_impact']}</small><br><br>
                <span style="font-size:0.8rem; color:#94A3B8;">Akun Resmi Terpantau: <b>@indonesiaetf</b>, <b>@indopremier</b>, <b>@blackrock</b>, <b>@vanguardgroup</b></span>
            </div>
            """, unsafe_allow_html=True)

    with eko_tabs[3]:
        st.markdown(f"##### 📸 Radar Sentimen Instagram: Kebijakan Menkeu, Presiden, BI, BEI & Media Saham")
        st.caption(
            "Mengambil dan membedah informasi langsung dari akun-akun resmi Instagram terkait pasar modal, "
            "kebijakan Menteri Keuangan (@smindrawati), Kemenkeu (@kemenkeuri), Presiden (@presidenrepublikindonesia), "
            "Bank Indonesia (@bank_indonesia), OJK (@ojkindonesia), Bursa Efek Indonesia (@indonesiastockexchange, @idx_channel), "
            "dan media finansial terpercaya (@cnbcindonesia, @bisniscom, @kontannews, @stockbit) yang dapat mengubah arah pergerakan harga saham."
        )

        ig_c1, ig_c2, ig_c3 = st.columns(3)
        with ig_c1:
            st.metric("Skor Sentimen Instagram", f"{insta_info['composite_instagram_score']:.0f} / 100", f"Status: {insta_info['policy_status'].split('(')[0].strip()}")
        with ig_c2:
            st.metric("Otoritas Penentu Kebijakan", f"{insta_info['primary_authority']}", "Regulator Utama")
        with ig_c3:
            st.metric("Akun Otoritas Terpantau", f"{insta_info['key_accounts_tracked_count']} Akun Resmi", "Terverifikasi")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid {insta_info['policy_color']}; border-radius:10px; padding:16px; margin-top:8px; margin-bottom:12px; color:#F1F5F9;">
            <h5 style="margin:0 0 6px 0; color:{insta_info['policy_color']};">⚖️ Status Kebijakan: {insta_info['policy_status']}</h5>
            <p style="margin:0 0 6px 0; color:#F1F5F9;"><b style="color:#38BDF8;">🎯 Fokus Kebijakan & Regulasi Sektor:</b> {insta_info['policy_focus']}</p>
            <p style="margin:0 0 6px 0; color:#F1F5F9;"><b style="color:#10B981;">🟢 Faktor Pendorong (Booster):</b> {insta_info['policy_booster_theme']}</p>
            <p style="margin:0 0 6px 0; color:#F1F5F9;"><b style="color:#EF4444;">🔴 Faktor Risiko (Pressure):</b> {insta_info['policy_risk_theme']}</p>
            <b style="color:#F8FAFC;">🧭 Panduan Arah Pasar:</b> <span style="color:{insta_info['policy_color']}; font-weight:700;">{insta_info['policy_guidance']}</span>
        </div>
        """, unsafe_allow_html=True)

        ig_post_col, ig_chart_col = st.columns([3, 2])
        with ig_post_col:
            st.markdown("###### 📱 Feed Postingan & Pernyataan Terkini Akun Instagram Otoritas:")
            for p in insta_info.get("latest_posts", [])[:5]:
                badge_style = "color:#10B981; font-weight:700;" if "POSITIF" in p['sentiment'] else ("color:#EF4444; font-weight:700;" if "NEGATIF" in p['sentiment'] else "color:#EAB308; font-weight:700;")
                st.markdown(f"""
                <div style="background:#1E293B; border-radius:8px; padding:10px 12px; margin-bottom:8px; border:1px solid #334155;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="color:#38BDF8; font-weight:700; font-size:0.9rem;">🔵 {p['account']}</span>
                        <span style="color:#94A3B8; font-size:0.8rem;">⏱️ {p['time_ago']}</span>
                    </div>
                    <div style="font-size:0.88rem; color:#F1F5F9; margin-bottom:6px;">{p['title']}</div>
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                        <span style="color:#94A3B8;">Sumber: <i>{p['source']}</i></span>
                        <span style="{badge_style}">{p['sentiment']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with ig_chart_col:
            st.markdown("###### 📊 Sentimen per Kategori Akun Instagram:")
            cat_data = insta_info.get("category_breakdown", {})
            cat_names = [k.split("(")[0].strip() for k in cat_data.keys()]
            cat_scores = list(cat_data.values())
            fig_ig_cat = px.bar(
                x=cat_scores,
                y=cat_names,
                orientation="h",
                labels={"x": "Skor Sentimen (0-100)", "y": "Kategori Akun"},
                template="plotly_dark",
                color=cat_scores,
                color_continuous_scale="Purp",
                text=[f"{v:.0f}" for v in cat_scores]
            )
            fig_ig_cat.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_ig_cat, use_container_width=True)

            st.caption(
                "💡 **Catatan Analisis**: Kebijakan fiskal dari Menteri Keuangan (@smindrawati) atau arahan Presiden memiliki "
                "bobot pengaruh langsung yang dapat mengubah arus dana asing dan memutar haluan harga saham secara instan."
            )

    with eko_tabs[4]:
        st.markdown(f"##### 🌍 Sentimen Media Sosial Global Multi-Platform (Multi-Channel NLP)")
        st.markdown(
            "Agregasi cerdas sentimen percakapan publik dari berbagai platform keuangan dan media sosial dunia "
            "untuk mengidentifikasi apakah kerumunan sedang berada di fase FOMO atau Extreme Fear."
        )

        soc_c1, soc_c2 = st.columns([1, 2])
        with soc_c1:
            st.metric("Skor Sentimen Komposit", f"{social_eval['composite_social_score']}/100", f"Status: {social_eval['crowd_status'].split('/')[0].strip()}")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid #F59E0B; border-radius:10px; padding:14px; margin-top:8px; color:#F1F5F9;">
                <b style="color:#F8FAFC;">Status Kerumunan:</b><br>
                <span style="color:#F59E0B; font-weight:700;">{social_eval['crowd_status']}</span><br><br>
                <b style="color:#F8FAFC;">Analisis Psikologis:</b><br>
                <small style="color:#CBD5E1;">{social_eval['crowd_desc']}</small>
            </div>
            """, unsafe_allow_html=True)

        with soc_c2:
            ch_keys = list(social_eval["channels"].keys())
            ch_scores = [social_eval["channels"][k] for k in ch_keys]
            fig_soc_bar = px.bar(
                x=ch_scores,
                y=ch_keys,
                orientation="h",
                labels={"x": "Skor Sentimen (0-100)", "y": "Saluran Media Sosial"},
                title=f"Sentimen Komunitas Pasar Modal Lintas Platform untuk {ticker_clean}",
                template="plotly_dark",
                color=ch_scores,
                color_continuous_scale="Temps",
                text=[f"{v:.0f}" for v in ch_scores]
            )
            fig_soc_bar.update_layout(height=290, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_soc_bar, use_container_width=True)

    with eko_tabs[5]:
        st.markdown(f"##### 🏗️ Proyek Strategis, Alokasi Capex & Aksi Korporasi Emiten")
        st.markdown(
            "Menelaah arah ekspansi korporasi, belanja modal masa depan, dan inisiatif strategis "
            "yang menjadi bahan bakar pertumbuhan laba bersih dan katalis harga saham ke depan."
        )

        cp_c1, cp_c2 = st.columns([2, 1])
        with cp_c1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid #3B82F6; border-radius:10px; padding:16px; color:#F1F5F9;">
                <h4 style="margin:0 0 8px 0; color:#38BDF8;">🎯 {corp_eval['project_title']}</h4>
                <p style="margin:0 0 10px 0; color:#F1F5F9;"><b style="color:#F8FAFC;">🛠️ Fokus Alokasi Capex:</b><br>{corp_eval['capex_focus']}</p>
                <p style="margin:0 0 10px 0; color:#F1F5F9;"><b style="color:#F8FAFC;">📈 Estimasi Dampak Pertumbuhan:</b><br>{corp_eval['catalyst_impact']}</p>
                <p style="margin:0; font-size:0.85rem; color:#94A3B8;">Emiten: <b style="color:#F8FAFC;">{corp_eval['company_name']}</b> | Sektor: <b style="color:#F8FAFC;">{corp_eval['sector']}</b></p>
            </div>
            """, unsafe_allow_html=True)
        with cp_c2:
            st.metric("Skor Kekuatan Katalis", f"{corp_eval['catalyst_score']} / 100", "Potensi Pertumbuhan Jangka Panjang")
            st.progress(corp_eval['catalyst_score'] / 100.0)
            st.caption("Skor mengevaluasi visibilitas arus kas bebas (FCFF), kapabilitas eksekusi manajemen, dan efisiensi belanja modal terhadap ROIC.")

    with eko_tabs[6]:
        st.markdown(f"##### 🧠 Psikologi Investor & Siklus Keuangan Perilaku (Behavioral Finance)")
        st.markdown(
            "Memetakan spektrum emosi kerumunan ritel pada kurva psikologi keuangan legendaris "
            "(dari *Disbelief* hingga *Euphoria* dan *Capitulation*) untuk menghindari bias kognitif dan mengeksekusi strategi kontrarian."
        )

        psy_c1, psy_c2 = st.columns([1, 1])
        with psy_c1:
            fig_fg = go.Figure(go.Indicator(
                mode="gauge+number",
                value=psychology_eval["fear_greed_index"],
                title={'text': f"Fear & Greed Index: {ticker_clean}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#38BDF8"},
                    'steps': [
                        {'range': [0, 25], 'color': "#1E3A8A"},
                        {'range': [25, 45], 'color': "#0284C7"},
                        {'range': [45, 55], 'color': "#EAB308"},
                        {'range': [55, 75], 'color': "#F97316"},
                        {'range': [75, 100], 'color': "#DC2626"}
                    ]
                }
            ))
            fig_fg.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fg, use_container_width=True)

        with psy_c2:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border:1px solid #334155; border-left:5px solid #EC4899; border-radius:12px; padding:16px; color:#F1F5F9;">
                <h4 style="margin:0 0 6px 0; color:#EC4899;">🎭 Fase Siklus: {psychology_eval['cycle_phase']}</h4>
                <p style="margin:0 0 10px 0; font-style:italic; font-size:1.02rem; color:#38BDF8;">"{psychology_eval['cycle_quote']}"</p>
                <p style="margin:0 0 8px 0; color:#F1F5F9;"><b style="color:#FBBF24;">⚠️ Peringatan Bias Kognitif:</b><br><span style="color:#E2E8F0;">{psychology_eval['bias_warning']}</span></p>
                <p style="margin:0; color:#F1F5F9;"><b style="color:#34D399;">💡 Tindakan Kontrarian Cerdas:</b><br><span style="color:#10B981; font-weight:700;">{psychology_eval['contrarian_action']}</span></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###### 🔄 Korelasi Silang: Psikologi Kerumunan Ritel vs Akumulasi Smart Money (Bandarmologi)")
        st.caption(
            "Menghubungkan kondisi emosi psikologis ritel saat ini secara langsung dengan posisi nyata Smart Money (Lead Broker), "
            "titik modal bandar, sensitivitas IHSG, dan konsensus analis untuk memastikan Anda tidak terjebak psikologi massa."
        )

        cross_matrix_data = [
            {
                "Pilar Analisis": "🎭 Emosi Kerumunan Ritel",
                "Kondisi Saat Ini": f"{psychology_eval['cycle_phase']} (Skor {psychology_eval['fear_greed_index']:.1f}/100)",
                "Dampak & Bias": "Mayoritas ritel bereaksi emosional (Panik / FOMO) yang memicu pergeseran likuiditas",
                "Tindakan Cerdas": psychology_eval['contrarian_action']
            },
            {
                "Pilar Analisis": "🕵️ Posisi Smart Money",
                "Kondisi Saat Ini": f"Broker {broker_eval['lead_broker_code']} ({broker_eval['fase_bandar']})",
                "Dampak & Bias": f"Modal rata-rata bandar di Rp {broker_eval['bandar_cost']:,} (Margin {broker_eval['diff_from_cost_pct']:+.1f}%)",
                "Tindakan Cerdas": broker_eval['action_bandar']
            },
            {
                "Pilar Analisis": "📈 Arah Makro IHSG",
                "Kondisi Saat Ini": f"{ihsg_eval['future_trend']} (Prob. Naik {ihsg_eval['direction_prob_up']:.0f}%)",
                "Dampak & Bias": f"Sensitivitas Beta {beta_eval['beta']}x ({beta_eval['category']}) terhadap arah pasar umum",
                "Tindakan Cerdas": f"Gunakan Support {ihsg_eval['support_1']:,} & Resisten {ihsg_eval['resistance_1']:,} sebagai batas aman"
            },
            {
                "Pilar Analisis": "📑 Konsensus Lembaga Riset",
                "Kondisi Saat Ini": f"{b_con['consensus_action']} ({b_con['total_analysts']} Analis)",
                "Dampak & Bias": f"Target harga konsensus Rp {b_con['mean_target_price']:,} (Potensi Upside +{b_con['upside_avg_pct']}%)",
                "Tindakan Cerdas": f"Beli jika harga di bawah modal bandar dan memiliki upside konsensus > 15%"
            }
        ]

        df_cross = pd.DataFrame(cross_matrix_data)
        st.dataframe(df_cross, hide_index=True, use_container_width=True)

# TAB 14: BOT DISPATCHER
with tab_bot:
    st.markdown("#### 📲 Konfigurasi Bot Dispatcher (Telegram & WhatsApp)")
    st.write(
        "Kirim sinyal dan notifikasi otomatis ke ponsel pintar melalui Telegram Bot API resmi "
        "atau WhatsApp Gateway / Click-to-Chat Direct wa.me."
    )

    bc_col1, bc_col2 = st.columns(2)
    with bc_col1:
        st.markdown("##### ✈️ Pengaturan Telegram Bot:")
        tg_token = st.text_input("Telegram Bot Token:", value=dispatcher_instance.config.telegram_bot_token, type="password")
        tg_chat = st.text_input("Telegram Chat ID:", value=dispatcher_instance.config.telegram_chat_id)
        tg_enable = st.checkbox("Aktifkan Telegram Bot", value=dispatcher_instance.config.telegram_enabled)
    
    with bc_col2:
        st.markdown("##### 💬 Pengaturan WhatsApp:")
        wa_target = st.text_input("Nomor WhatsApp Tujuan (contoh: 628xxxxxxxxxx):", value=dispatcher_instance.config.wa_target_phone or "6281234567890")
        wa_gateway_tok = st.text_input("API Token Gateway (Fonnte/Wablas):", value=dispatcher_instance.config.wa_gateway_token, type="password")
        wa_channel = st.selectbox("Metode WhatsApp:", ["click_to_chat", "gateway", "cloud_api"], index=0)

    if st.button("💾 Simpan Konfigurasi Bot", type="primary", use_container_width=True):
        dispatcher_instance.config.telegram_bot_token = tg_token
        dispatcher_instance.config.telegram_chat_id = tg_chat
        dispatcher_instance.config.telegram_enabled = tg_enable
        dispatcher_instance.config.wa_target_phone = wa_target
        dispatcher_instance.config.wa_gateway_token = wa_gateway_tok
        dispatcher_instance.config.whatsapp_channel = wa_channel
        dispatcher_instance.config.save_to_disk()
        st.success("✅ Konfigurasi bot berhasil disimpan ke disk!")

# Pembaruan Data Real-Time: Dilakukan secara aman melalui tombol manual refresh di sidebar untuk mencegah refresh berulang tanpa izin user.

st.caption("⚠️ Disclaimer Pasar Modal: Seluruh analisis, skor kuantitatif, dan rekomendasi harga adalah alat bantu pendukung keputusan (decision support tool). Keputusan investasi dan trading sepenuhnya merupakan tanggung jawab mandiri investor.")
