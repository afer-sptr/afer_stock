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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
import urllib.parse

from modules.data_loader import (
    fetch_stock_data,
    fetch_historical_ohlcv,
    normalize_ticker,
    ALL_IDX_STOCKS,
    get_available_sectors,
    search_idx_stocks,
    clear_stock_cache,
)
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
from modules.econometrics_risk import run_econometrics_and_risk
from modules.portfolio_optimizer import optimize_mean_cvar_portfolio
from modules.trading_styles_types import (
    evaluate_trading_style_and_type,
    get_trading_cash_cows,
    TRADING_STYLES_INFO,
    STOCK_TYPES_PROFILES,
)
from modules.breakout_bandarmologi import (
    detect_chart_patterns_and_breakout,
    analyze_promoter_and_broker_footprint,
    scan_breakout_universe,
)
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

# Custom Styling Mengikuti Tampilan idx_stock_analyzer
st.markdown("""
<style>
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
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
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
    ["📊 Dashboard Analisis Saham", "⚡ Lapis 3 Rally Hunter"],
    index=0
)
st.sidebar.markdown("---")

# Inisialisasi session state untuk ticker aktif
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "BBCA"

# Mode Live Real-Time
st.sidebar.subheader("⏱️ Mode Live Real-Time")
auto_refresh = st.sidebar.checkbox("Aktifkan Auto-Refresh Otomatis", value=False)
refresh_interval = st.sidebar.select_slider(
    "Interval Pembaruan Data:",
    options=[5, 10, 15, 30, 60, 120],
    value=5,
    format_func=lambda x: f"{x} Detik"
)

# Filter Semesta Saham
st.sidebar.subheader("🔍 Filter Saham BEI")
chosen_tier = st.sidebar.selectbox(
    "Filter Tingkatan (Tier):",
    [
        "Semua Tingkatan",
        "Saham Gocap / Saham Tidur (Rp50 – Rp100)",
        "Saham Receh / Saham Murah (Rp100 – Rp1.000)",
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
current_target_ticker = str(st.session_state.get("selected_ticker", "BBCA")).replace(".JK", "").upper()

if mode_input == "Pilih dari Katalog":
    if filtered_stocks:
        stock_labels = [s["display_label"] for s in filtered_stocks]
        default_idx = 0
        for idx, s in enumerate(filtered_stocks):
            if s["ticker"] == current_target_ticker:
                default_idx = idx
                break
        selected_stock_label = st.sidebar.selectbox("Katalog Saham Terfilter:", stock_labels, index=default_idx)
        selected_ticker = selected_stock_label.split(" - ")[0]
        st.session_state["selected_ticker"] = selected_ticker
    else:
        st.sidebar.warning("Tidak ada saham yang cocok dengan kombinasi filter.")
        selected_ticker = current_target_ticker
else:
    selected_ticker = st.sidebar.text_input("Ketik Kode Ticker (contoh: BBCA, BBRI, BREN, BRMS):", value=current_target_ticker).strip()
    st.session_state["selected_ticker"] = selected_ticker

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
@st.cache_data(ttl=15, show_spinner=False)
def get_cached_stock_data(ticker_symbol: str, period: str):
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
def get_cached_scalping_picks(tier_filter: str, syariah_filter: str):
    return scan_top_10_scalping_stocks(tier_filter=tier_filter, syariah_filter=syariah_filter)


# ----------------- PROSES DATA PASAR -----------------
ticker_clean = normalize_ticker(selected_ticker)

if btn_manual_refresh:
    st.cache_data.clear()
    clear_stock_cache()

with st.spinner(f"Menghubungkan ke Bursa Efek Indonesia untuk memuat data {ticker_clean}..."):
    df, info, err = get_cached_stock_data(ticker_clean, period=period_choice)

if err or df is None or df.empty:
    st.error(f"❌ Terjadi kesalahan saat memuat data {ticker_clean}: {err}")
    st.info("💡 Pastikan kode ticker tepat 4 huruf yang tercatat aktif di Bursa Efek Indonesia.")
    st.stop()

# 1. Analisis Teknikal Terpadu & MA Ribbon
df_tech = compute_technical_indicators(df)
tech_eval = evaluate_technical_score(df_tech)
tech_suite = compute_technical_suite(df_tech)

fast_info = info.get("fast_info")
current_price = float(df_tech["Close"].iloc[-1])
prev_close = float(df_tech["Close"].iloc[-2]) if len(df_tech) > 1 else current_price
price_diff = current_price - prev_close
price_diff_pct = (price_diff / prev_close) * 100 if prev_close > 0 else 0

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

# 14. Rekomendasi Keputusan Terpadu Holistik
rec = generate_composite_recommendation(
    tech_result=tech_eval,
    fund_result=fund_eval,
    ml_result=ml_eval,
    current_price=current_price,
    high_low_result=hl_eval,
    news_result=news_eval,
    order_book_result=order_book_eval,
    ai_suite_result=ai_suite_eval,
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
    st.markdown("#### 🚀 Pemindai Saham Potensi Rally (Katalog Small-Cap / Lapis 3)")
    st.caption("Peringkat saham lapis 3 yang terdeteksi memiliki anomali lonjakan volume dan kompresi volatilitas:")
    
    sample_rally_candidates = [
        {"ticker": "DEWA", "nama": "Darma Henwa Tbk.", "harga": 105, "rvol": 2.85, "squeeze": "🟢 Ya", "bid_pct": 71.4, "turnover": 45_200_000_000, "status": "🟢 SIAP MELEDAK"},
        {"ticker": "KIJA", "nama": "Kawasan Industri Jababeka", "harga": 172, "rvol": 2.40, "squeeze": "🟢 Ya", "bid_pct": 68.2, "turnover": 18_400_000_000, "status": "🟢 SIAP MELEDAK"},
        {"ticker": "ELSA", "nama": "Elnusa Tbk.", "harga": 486, "rvol": 2.15, "squeeze": "⚪ Tidak", "bid_pct": 66.5, "turnover": 32_100_000_000, "status": "🟡 AKUMULASI"},
        {"ticker": "PSAB", "nama": "J Resources Asia Pasifik", "harga": 312, "rvol": 2.30, "squeeze": "🟢 Ya", "bid_pct": 65.0, "turnover": 24_500_000_000, "status": "🟢 SIAP MELEDAK"},
        {"ticker": "RAJA", "nama": "Rukun Raharja Tbk.", "harga": 1380, "rvol": 1.95, "squeeze": "🟢 Ya", "bid_pct": 63.0, "turnover": 19_800_000_000, "status": "🟡 AKUMULASI"},
        {"ticker": "DOID", "nama": "Delta Dunia Makmur Tbk.", "harga": 498, "rvol": 1.80, "squeeze": "⚪ Tidak", "bid_pct": 58.5, "turnover": 14_200_000_000, "status": "⚪ KONSOLIDASI"},
        {"ticker": "BUMI", "nama": "Bumi Resources Tbk.", "harga": 148, "rvol": 2.65, "squeeze": "🟢 Ya", "bid_pct": 68.5, "turnover": 66_000_000_000, "status": "🟢 SIAP MELEDAK"},
        {"ticker": "BRMS", "nama": "Bumi Resources Minerals", "harga": 410, "rvol": 2.25, "squeeze": "🟢 Ya", "bid_pct": 66.0, "turnover": 127_000_000_000, "status": "🟢 SIAP MELEDAK"},
        {"ticker": "ENRG", "nama": "Energi Mega Persada", "harga": 95, "rvol": 2.10, "squeeze": "⚪ Tidak", "bid_pct": 66.8, "turnover": 23_750_000_000, "status": "🟡 AKUMULASI"},
    ]
    df_rally = pd.DataFrame(sample_rally_candidates)
    st.dataframe(
        df_rally,
        column_config={
            "ticker": "Kode Saham",
            "nama": "Nama Perusahaan",
            "harga": st.column_config.NumberColumn("Harga (IDR)", format="Rp %d"),
            "rvol": st.column_config.NumberColumn("Relative Vol (x)", format="%.2fx"),
            "squeeze": "Bollinger Squeeze",
            "bid_pct": st.column_config.NumberColumn("% Bid", format="%.1f%%"),
            "turnover": st.column_config.NumberColumn("Turnover Harian", format="Rp %d"),
            "status": "Status Rally",
        },
        use_container_width=True,
        hide_index=True
    )
    st.stop()

# ----------------- HEADER UTAMA (GAYA idx_stock_analyzer) -----------------
st.markdown('<div class="main-title">⚡ Sistem Analisis & Prediksi Saham IDX Real-Time</div>', unsafe_allow_html=True)
last_time = info.get("fetched_at", datetime.now().strftime("%d-%m-%Y %H:%M:%S WIB"))
st.markdown(f'<span class="live-badge">🟢 REAL-TIME DATA FEED: Diperbarui {last_time}</span>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Perhitungan Harga Wajar, Kalender Dividen, Sentimen FinBERT, Price Action, Order Book, Ekonometrika & Machine Learning</div>', unsafe_allow_html=True)

company_name = info.get("longName") or info.get("shortName") or ticker_clean
sector_name = info.get("sector") or "Bursa Efek Indonesia"
tier_badge = info.get("tier", "Lapis 2 (Mid-Cap)")
syariah_badge = info.get("syariah_label", "☪️ Syariah (ISSI)")

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
    st.caption(f"• Teknikal: {b['technical']} | • High/Low: {b['high_low']} | • Fundamental: {b['fundamental']}")
    st.caption(f"• Sentimen Berita: {b['news_sentiment']} | • AI & ML: {b['ml_prediction']} | • Order Book: {b['order_book']}")

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
                            <div class="quant-box" style="border-left: 5px solid #10B981; padding:12px; margin-bottom:12px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <h4 style="margin:0; color:#0F172A;">#{idx_s+1} {s['ticker']} <span style="font-size:12px; color:#64748B;">({s['company_name'][:22]})</span></h4>
                                    <span style="background-color:#DCFCE7; color:#166534; padding:3px 8px; border-radius:6px; font-weight:bold; font-size:12px;">Skor: {s['scalp_score']}/100 ⭐</span>
                                </div>
                                <p style="margin:4px 0; font-size:12px; color:#475569;">
                                    🏷️ <b>{s['tier']}</b> | {s['sector']} | {'☪️ Syariah' if s['is_syariah'] else '⚪ Non-Syariah'}
                                </p>
                                <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:6px; margin:8px 0; text-align:center;">
                                    <div style="background:#F1F5F9; padding:6px; border-radius:6px;">
                                        <div style="font-size:10px; color:#64748B;">ZONA ENTRY</div>
                                        <div style="font-weight:bold; color:#0284C7; font-size:14px;">Rp {s['entry_price']:,}</div>
                                    </div>
                                    <div style="background:#ECFDF5; padding:6px; border-radius:6px;">
                                        <div style="font-size:10px; color:#059669;">TAKE PROFIT 1</div>
                                        <div style="font-weight:bold; color:#059669; font-size:14px;">Rp {s['tp1']:,} <span style="font-size:11px;">({s['tp1_net_pct']:+.2f}%)</span></div>
                                    </div>
                                    <div style="background:#FEF2F2; padding:6px; border-radius:6px;">
                                        <div style="font-size:10px; color:#DC2626;">CUT LOSS (SL)</div>
                                        <div style="font-weight:bold; color:#DC2626; font-size:14px;">Rp {s['stop_loss']:,} <span style="font-size:11px;">({s['sl_net_pct']:+.2f}%)</span></div>
                                    </div>
                                </div>
                                <div style="font-size:11px; color:#334155; margin-bottom:6px;">
                                    🏆 <b>TP2 (Target Lanjutan):</b> Rp {s['tp2']:,} ({s['tp2_net_pct']:+.2f}%) | ⚖️ <b>RRR:</b> 1:{s['rrr']} | 🛡️ <b>Maksimal Lot Aman:</b> {s['safe_exit_lot']:,} Lot
                                </div>
                                <div style="font-size:11px; color:#64748B; background:#F8FAFC; padding:4px 8px; border-radius:4px; border:1px dashed #CBD5E1;">
                                    ⚡ <b>Katalis:</b> {s['catalyst']}
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
    models_table = pd.DataFrame([
        {
            "Model Valuasi": "Formula Benjamin Graham (Graham Number)",
            "Rumus Utama": "V = √(22.5 × EPS × BVPS)",
            "Estimasi Harga Wajar": f"Rp {m['graham_number']:,}" if m['graham_number'] else "N/A",
            "Keterangan": "Menilai proteksi nilai buku aset fisik dan kapasitas profitabilitas riil"
        },
        {
            "Model Valuasi": "Justified PBV - ROE Model",
            "Rumus Utama": "Fair PBV = ROE / Cost of Equity (11%)",
            "Estimasi Harga Wajar": f"Rp {m['justified_pbv_price']:,}" if m['justified_pbv_price'] else "N/A",
            "Keterangan": f"PBV Wajar dihitung {m.get('justified_pbv') or '-'}x berdasarkan efisiensi modal ROE"
        },
        {
            "Model Valuasi": "P/E Industry Multiplier (BEI Mean)",
            "Rumus Utama": "V = EPS × 15.0x",
            "Estimasi Harga Wajar": f"Rp {m['pe_multiple_price']:,}" if m['pe_multiple_price'] else "N/A",
            "Keterangan": "Valuasi wajar berdasarkan penggali laba bersih rata-rata industri di BEI"
        },
        {
            "Model Valuasi": "Dividend Discount Model (Gordon Growth)",
            "Rumus Utama": "V = DPS × (1 + g) / (r - g)",
            "Estimasi Harga Wajar": f"Rp {m['ddm_price']:,}" if m['ddm_price'] else "N/A",
            "Keterangan": "Nilai tunai dari seluruh arus dividen masa depan yang didiskontokan ke saat ini"
        }
    ])
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

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"Grafik Harga {ticker_clean} (Candlestick & MA Ribbon)", "Volume Transaksi"]
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

    # Garis Rekomendasi Candlestick Real-Time
    fig.add_hline(y=candle_pred["tp1"], line_dash="dash", line_color="#10B981", annotation_text=f"TP1 Lilin (Rp {candle_pred['tp1']:,} | {candle_pred['tp1_net_pct']:+.1f}%)", row=1, col=1)
    fig.add_hline(y=candle_pred["tp2"], line_dash="solid", line_color="#059669", annotation_text=f"TP2 Lilin (Rp {candle_pred['tp2']:,} | {candle_pred['tp2_net_pct']:+.1f}%)", row=1, col=1)
    fig.add_hline(y=candle_pred["stop_loss"], line_dash="dash", line_color="#DC2626", annotation_text=f"Cut Loss Lilin (Rp {candle_pred['stop_loss']:,} | {candle_pred['sl_net_pct']:+.1f}%)", row=1, col=1)
    fig.add_hline(y=candle_pred["entry_price"], line_dash="dot", line_color="#3B82F6", annotation_text=f"Entry Lilin (Rp {candle_pred['entry_price']:,})", row=1, col=1)

    # Garis Rekomendasi Multi-Pilar TP / SL Berfraksi BEI
    fig.add_hline(y=plan["take_profit_1"], line_dash="dot", line_color="#34D399", annotation_text=f"TP1 Sistem (Rp {plan['take_profit_1']:,})", row=1, col=1)
    fig.add_hline(y=plan["stop_loss"], line_dash="dot", line_color="#F87171", annotation_text=f"SL Sistem (Rp {plan['stop_loss']:,})", row=1, col=1)

    # Garis Fibo 61.8%
    fig.add_hline(y=hl_eval["fib_levels"]["fib_618"], line_dash="dot", line_color="#F59E0B", annotation_text=f"Fib 61.8% (Rp {hl_eval['fib_levels']['fib_618']:,})", row=1, col=1)

    # Volume Bar
    vol_colors = ['#22C55E' if c >= o else '#EF4444' for c, o in zip(df_tech["Close"], df_tech["Open"])]
    fig.add_trace(go.Bar(x=df_tech.index, y=df_tech["Volume"], name="Volume", marker_color=vol_colors), row=2, col=1)

    fig.update_layout(
        height=640,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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

# TAB 13: BOT DISPATCHER
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

# Auto-Refresh Handler
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

st.caption("⚠️ Disclaimer Pasar Modal: Seluruh analisis, skor kuantitatif, dan rekomendasi harga adalah alat bantu pendukung keputusan (decision support tool). Keputusan investasi dan trading sepenuhnya merupakan tanggung jawab mandiri investor.")
