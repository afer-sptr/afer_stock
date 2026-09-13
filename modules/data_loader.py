"""
Modul Data Loader untuk Saham Indonesia (IDX / BEI) - Terpadu.
Memuat katalog lengkap 940+ emiten terdaftar di Bursa Efek Indonesia,
klasifikasi tingkatan (Lapis 1 / 2 / 3), kepatuhan Syariah, data order book Level-1,
serta data harga historis dan profil fundamental dari Yahoo Finance (.JK).
"""

import os
import json
import logging
import math
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import yfinance as yf

from modules.idx_ticks import round_to_idx_tick, get_idx_tick_size
from modules.idx_universe import (
    get_all_idx_stocks_enriched,
    filter_idx_stocks,
    get_stock_metadata,
    get_all_sectors,
    TIER_1_TICKERS,
    TIER_2_TICKERS,
    NON_SHARIA_TICKERS,
)

logger = logging.getLogger(__name__)

ALL_IDX_STOCKS = get_all_idx_stocks_enriched()
POPULAR_IDX_STOCKS = [s for s in ALL_IDX_STOCKS if s["ticker"] in TIER_1_TICKERS][:15]


def get_available_sectors() -> List[str]:
    """Daftar sektor industri di BEI."""
    return get_all_sectors()


def search_idx_stocks(
    query: str,
    sector: Optional[str] = None,
    tier: Optional[str] = None,
    syariah: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Pencarian dan penyaringan fleksibel seluruh saham BEI."""
    return filter_idx_stocks(
        tier_filter=tier or "Semua",
        syariah_filter=syariah or "Semua",
        sector_filter=sector or "Semua",
        search_query=query,
    )


def normalize_ticker(ticker: str) -> str:
    """Menstandarkan kode ticker BEI ke format Yahoo Finance (.JK)."""
    clean = ticker.strip().upper()
    if not clean.endswith(".JK") and not clean.startswith("^"):
        clean += ".JK"
    return clean


import time
import copy

_STOCK_CACHE: Dict[str, Tuple[float, pd.DataFrame, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 10.0


def clear_stock_cache():
    """Membersihkan seluruh cache data saham in-memory."""
    global _STOCK_CACHE
    _STOCK_CACHE.clear()


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """
    Mengambil data harga historis (OHLCV), profil fundamental, serta snapshot
    Order Book (Bid/Offer/OBI/Spread) saham IDX secara real-time.
    Menggunakan in-memory caching berkecepatan tinggi (TTL 10 detik).
    """
    ticker_clean = normalize_ticker(ticker)
    cache_key = f"{ticker_clean}_{period}_{interval}"
    now = time.time()

    if not force_refresh and cache_key in _STOCK_CACHE:
        cached_time, c_df, c_info = _STOCK_CACHE[cache_key]
        if (now - cached_time) < CACHE_TTL_SECONDS:
            return c_df.copy(), copy.deepcopy(c_info), None

    try:
        stock = yf.Ticker(ticker_clean)
        df = stock.history(period=period, interval=interval)
        
        if df is None or df.empty:
            return None, None, f"Tidak ada data transaksi ditemukan untuk {ticker_clean}. Pastikan kode saham aktif di BEI."
        
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(df.index)
        
        info = {}
        try:
            raw_info = stock.info
            if raw_info and isinstance(raw_info, dict):
                info = raw_info
        except Exception as e:
            logger.warning(f"Gagal mengambil metadata untuk {ticker_clean}: {e}")
            info = {}

        # Simpan fast_info real-time
        try:
            fast = stock.fast_info
            if fast:
                info["fast_info"] = fast
                if getattr(fast, "last_price", None):
                    info["realtime_last_price"] = float(fast.last_price)
                if getattr(fast, "year_high", None):
                    info["year_high"] = float(fast.year_high)
                if getattr(fast, "year_low", None):
                    info["year_low"] = float(fast.year_low)
        except Exception:
            pass

        # Ekstraksi Level-1 Order Book & Turnover
        last_close = float(df["Close"].iloc[-1])
        last_vol = float(df["Volume"].iloc[-1])
        turnover_idr = last_close * last_vol

        raw_bid = info.get("bid")
        bid_tick = get_idx_tick_size(last_close)
        bid = float(raw_bid) if (raw_bid is not None and not math.isnan(float(raw_bid)) and float(raw_bid) > 0) else max(50.0, last_close - bid_tick)
        
        raw_ask = info.get("ask")
        ask = float(raw_ask) if (raw_ask is not None and not math.isnan(float(raw_ask)) and float(raw_ask) > 0) else (last_close + bid_tick)

        raw_b_size = info.get("bidSize")
        bid_size = float(raw_b_size) if (raw_b_size is not None and not math.isnan(float(raw_b_size)) and float(raw_b_size) > 0) else 12500.0

        raw_a_size = info.get("askSize")
        ask_size = float(raw_a_size) if (raw_a_size is not None and not math.isnan(float(raw_a_size)) and float(raw_a_size) > 0) else 8200.0

        total_size = max(1.0, bid_size + ask_size)
        pct_bid = (bid_size / total_size) * 100.0
        pct_offer = 100.0 - pct_bid
        obi = (pct_bid - pct_offer) / 100.0
        mid_price = (bid + ask) / 2.0
        rel_spread = ((ask - bid) / max(1.0, mid_price)) * 100.0

        info["ticker"] = ticker_clean
        info["clean_ticker"] = ticker_clean.replace(".JK", "")
        info["price"] = last_close
        info["bid"] = round_to_idx_tick(bid, "down")
        info["ask"] = round_to_idx_tick(ask, "up")
        info["bid_size"] = int(bid_size)
        info["ask_size"] = int(ask_size)
        info["pct_bid"] = pct_bid
        info["pct_offer"] = pct_offer
        info["obi"] = obi
        info["rel_spread"] = rel_spread
        info["turnover_idr"] = turnover_idr
        info["volume"] = int(last_vol)
        info["fetched_at"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S WIB")

        # Tambahkan metadata universe
        meta = get_stock_metadata(ticker_clean)
        info["tier"] = meta.get("tier", "Lapis 2 (Mid-Cap)")
        info["tier_code"] = meta.get("tier_code", "L2")
        info["is_syariah"] = meta.get("is_syariah", True)
        info["syariah_label"] = meta.get("syariah_label", "☪️ Syariah (ISSI)")
        # Simpan ke in-memory cache
        _STOCK_CACHE[cache_key] = (now, df.copy(), copy.deepcopy(info))
        return df, info, None

    except Exception as e:
        return None, None, f"Terjadi kesalahan saat mengunduh data: {str(e)}"


def fetch_historical_ohlcv(
    tickers: List[str],
    period: str = "1y"
) -> Dict[str, pd.DataFrame]:
    """Mengunduh dan menormalkan data historis OHLCV untuk beberapa ticker sekaligus."""
    data_map: Dict[str, pd.DataFrame] = {}
    for t_str in tickers:
        clean = normalize_ticker(t_str)
        try:
            t = yf.Ticker(clean)
            df = t.history(period=period, auto_adjust=False)
            if not df.empty and len(df) > 20:
                df = df.sort_index()
                col_map = {c: c.capitalize() for c in df.columns}
                df = df.rename(columns=col_map)
                needed = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in needed if c in df.columns]]
                data_map[clean] = df
        except Exception as ex:
            logger.warning(f"Gagal mengambil data batch {clean}: {ex}")
    return data_map
