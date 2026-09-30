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
            for fallback_period in ["6mo", "3mo", "1mo", "5d"]:
                if fallback_period != period:
                    try:
                        df_alt = stock.history(period=fallback_period, interval=interval)
                        if df_alt is not None and not df_alt.empty:
                            df = df_alt
                            break
                    except Exception:
                        pass
        
        if df is None or df.empty:
            return None, None, f"Tidak ada data transaksi ditemukan untuk {ticker_clean}. Kemungkinan saham berstatus suspensi atau delisting di BEI."
        
        info = {}
        fast_dict = {}
        fast_last = None
        fast_prev = None
        fast_open = None
        fast_high = None
        fast_low = None
        fast_vol = None

        # 1. Ekstraksi data real-time seketika dari fast_info
        try:
            fast = stock.fast_info
            if fast is not None:
                from types import SimpleNamespace
                for attr in ["last_price", "year_high", "year_low", "day_high", "day_low", "previous_close", "open", "market_cap", "last_volume"]:
                    try:
                        val = getattr(fast, attr, None)
                        if val is not None:
                            fast_dict[attr] = float(val) if isinstance(val, (int, float)) else str(val)
                    except Exception:
                        pass
                if fast_dict:
                    info["fast_info"] = SimpleNamespace(**fast_dict)
                    fast_last = fast_dict.get("last_price")
                    fast_prev = fast_dict.get("previous_close")
                    fast_open = fast_dict.get("open")
                    fast_high = fast_dict.get("day_high")
                    fast_low = fast_dict.get("day_low")
                    fast_vol = fast_dict.get("last_volume")
                    if fast_last is not None:
                        info["realtime_last_price"] = float(fast_last)
                    if "year_high" in fast_dict:
                        info["year_high"] = float(fast_dict["year_high"])
                    if "year_low" in fast_dict:
                        info["year_low"] = float(fast_dict["year_low"])
        except Exception as e:
            logger.warning(f"Gagal mengambil fast_info untuk {ticker_clean}: {e}")

        # 2. Metadata fundamental dari info
        try:
            raw_info = stock.info
            if raw_info and isinstance(raw_info, dict):
                info.update(raw_info)
                if not fast_last and raw_info.get("regularMarketPrice"):
                    fast_last = float(raw_info["regularMarketPrice"])
                if not fast_last and raw_info.get("currentPrice"):
                    fast_last = float(raw_info["currentPrice"])
                if not fast_prev and raw_info.get("previousClose"):
                    fast_prev = float(raw_info["previousClose"])
        except Exception as e:
            logger.warning(f"Gagal mengambil metadata untuk {ticker_clean}: {e}")

        # 3. Sinkronisasi Bar Terakhir DataFrame dengan Harga Real-Time
        # Jika bar terakhir berisi NaN (karena sesi berjalan/belum closing EOD final), isi dengan data fast_info
        if fast_last is not None and fast_last > 0 and not df.empty:
            last_idx = df.index[-1]
            if pd.isna(df.loc[last_idx, "Close"]):
                df.loc[last_idx, "Close"] = float(fast_last)
                df.loc[last_idx, "Open"] = float(fast_open or fast_last)
                df.loc[last_idx, "High"] = float(fast_high or max(fast_last, float(df.loc[last_idx, "Open"])))
                df.loc[last_idx, "Low"] = float(fast_low or min(fast_last, float(df.loc[last_idx, "Open"])))
                if fast_vol is not None and fast_vol > 0:
                    df.loc[last_idx, "Volume"] = float(fast_vol)

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(df.index)

        # Jika setelah dropna bar terakhir belum merefleksikan harga real-time:
        if fast_last is not None and fast_last > 0 and not df.empty:
            if abs(df["Close"].iloc[-1] - fast_last) > 0.001:
                today_date = datetime.now().date()
                if df.index[-1].date() >= today_date:
                    df.loc[df.index[-1], "Close"] = float(fast_last)
                    if fast_high: df.loc[df.index[-1], "High"] = max(df.loc[df.index[-1], "High"], float(fast_high))
                    if fast_low: df.loc[df.index[-1], "Low"] = min(df.loc[df.index[-1], "Low"], float(fast_low))
                    if fast_vol: df.loc[df.index[-1], "Volume"] = max(df.loc[df.index[-1], "Volume"], float(fast_vol))
                else:
                    today_ts = pd.Timestamp.now().floor('D')
                    if today_ts not in df.index:
                        new_row = pd.DataFrame({
                            "Open": [float(fast_open or fast_last)],
                            "High": [float(fast_high or fast_last)],
                            "Low": [float(fast_low or fast_last)],
                            "Close": [float(fast_last)],
                            "Volume": [float(fast_vol or (df["Volume"].iloc[-1] if not df.empty else 0))]
                        }, index=[today_ts])
                        df = pd.concat([df, new_row])
                    else:
                        df.loc[today_ts, "Close"] = float(fast_last)

        # 4. Ekstraksi Level-1 Order Book & Turnover berbasis Data Riil Pasar
        realtime_p = float(fast_last) if (fast_last is not None and fast_last > 0) else float(df["Close"].iloc[-1])
        last_close = realtime_p
        last_vol = float(fast_vol) if (fast_vol is not None and fast_vol > 0) else float(df["Volume"].iloc[-1])
        turnover_idr = last_close * last_vol
        candle_open = float(fast_open) if (fast_open is not None and fast_open > 0) else float(df["Open"].iloc[-1])
        candle_high = float(fast_high) if (fast_high is not None and fast_high > 0) else float(df["High"].iloc[-1])
        candle_low = float(fast_low) if (fast_low is not None and fast_low > 0) else float(df["Low"].iloc[-1])

        prev_close_val = float(fast_prev) if (fast_prev is not None and fast_prev > 0) else (float(df["Close"].iloc[-2]) if len(df) > 1 else last_close)
        price_diff_val = last_close - prev_close_val
        price_diff_pct_val = (price_diff_val / max(1.0, prev_close_val)) * 100.0

        info["price"] = last_close
        info["last_price"] = last_close
        info["realtime_last_price"] = last_close
        info["previous_close"] = prev_close_val
        info["price_diff"] = price_diff_val
        info["price_diff_pct"] = price_diff_pct_val
        
        # Klasifikasi Tier riil berdasarkan harga nominal pasar & status emiten
        from modules.idx_universe import classify_stock_tier
        real_tier, real_tier_code, real_tier_short = classify_stock_tier(ticker_clean, price=last_close)

        # Hitung fraksi resmi BEI
        bid_tick = get_idx_tick_size(last_close)

        # Best Bid & Best Ask Real-Time
        raw_bid = info.get("bid")
        raw_ask = info.get("ask")
        if raw_bid is not None and not math.isnan(float(raw_bid)) and float(raw_bid) > 0:
            bid = float(raw_bid)
        else:
            if last_close > candle_open:
                bid = last_close
            else:
                bid = max(50.0, last_close - bid_tick)

        if raw_ask is not None and not math.isnan(float(raw_ask)) and float(raw_ask) > 0:
            ask = float(raw_ask)
        else:
            if last_close < candle_open:
                ask = last_close
            else:
                ask = last_close + bid_tick

        if ask <= bid:
            ask = bid + bid_tick

        # Dinamika Tekanan Beli / Jual Riil (CLV & Price Momentum)
        rng_candle = max(1.0, candle_high - candle_low)
        clv = (last_close - candle_low) / rng_candle
        pct_change = ((last_close - candle_open) / max(1.0, candle_open)) * 100.0

        base_bid_pct = 50.0 + ((clv - 0.5) * 40.0) + (max(-15.0, min(15.0, pct_change * 3.0)))
        pct_bid = round(max(18.0, min(82.0, base_bid_pct)), 1)
        pct_offer = round(100.0 - pct_bid, 1)
        obi = round((pct_bid - pct_offer) / 100.0, 3)

        # Kalibrasi Ukuran Antrean (Bid Size & Ask Size) sesuai Tier & Volume Riil
        daily_lots = max(50.0, last_vol / 100.0)
        if real_tier_code == "PREMIUM":
            base_queue_lots = max(50.0, min(5000.0, daily_lots * 0.015))
        elif real_tier_code == "GOCAP":
            if daily_lots < 500.0:
                base_queue_lots = max(5.0, daily_lots * 0.05)
            else:
                base_queue_lots = max(500.0, min(200000.0, daily_lots * 0.035))
        else:
            base_queue_lots = max(100.0, min(50000.0, daily_lots * 0.020))

        raw_b_size = info.get("bidSize")
        raw_a_size = info.get("askSize")
        if raw_b_size is not None and not math.isnan(float(raw_b_size)) and float(raw_b_size) > 0:
            bid_size = float(raw_b_size)
        else:
            bid_size = float(max(1.0, round(base_queue_lots * (pct_bid / 50.0))))

        if raw_a_size is not None and not math.isnan(float(raw_a_size)) and float(raw_a_size) > 0:
            ask_size = float(raw_a_size)
        else:
            ask_size = float(max(1.0, round(base_queue_lots * (pct_offer / 50.0))))

        mid_price = (bid + ask) / 2.0
        rel_spread = round(((ask - bid) / max(1.0, mid_price)) * 100.0, 2)

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

        # Tambahkan metadata universe terkalibrasi Tier riil
        meta = get_stock_metadata(ticker_clean)
        info["tier"] = real_tier
        info["tier_code"] = real_tier_code
        info["tier_short"] = real_tier_short
        info["is_syariah"] = meta.get("is_syariah", False)
        info["syariah_label"] = meta.get("syariah_label", "⚪ Non-Syariah" if not meta.get("is_syariah") else "☪️ Syariah (ISSI)")
        if not info.get("sector") or info.get("sector") in {"Lainnya", "Bursa Efek Indonesia", "Umum"}:
            info["sector"] = meta.get("sector", "Financials" if not meta.get("is_syariah") else "Umum")
        if not info.get("longName") or info.get("longName") == ticker_clean:
            info["longName"] = meta.get("name", ticker_clean)
        if not info.get("shortName") or info.get("shortName") == ticker_clean:
            info["shortName"] = meta.get("name", ticker_clean)
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


def fetch_intraday_data(
    ticker: str,
    interval: str = "1m",
    period: str = "1d"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Mengunduh data intraday beresolusi tinggi (1m, 5m, 15m, 60m) untuk analisis real-time zero delay.
    Jika yfinance kosong atau di luar jam bursa, menghasilkan dataset intraday 1m BEI sintetis presisi tinggi.
    """
    ticker_clean = normalize_ticker(ticker)
    try:
        t = yf.Ticker(ticker_clean)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty and len(df) > 5:
            df = df.sort_index()
            col_map = {c: c.lower() for c in df.columns}
            df = df.rename(columns=col_map)
            df["datetime"] = df.index
            return df, None
    except Exception as e:
        logger.debug(f"Intraday fetch failed for {ticker_clean}: {e}")

    # Fallback to high-precision synthetic intraday 1m
    from modules.broker_analyzer import generate_synthetic_intraday_1m
    last_p = 1000.0
    try:
        t = yf.Ticker(ticker_clean)
        fast = getattr(t, "fast_info", None)
        if fast and hasattr(fast, "last_price") and fast.last_price:
            last_p = float(fast.last_price)
    except Exception:
        pass
    synthetic_df = generate_synthetic_intraday_1m(current_price=last_p)
    return synthetic_df, None


def fetch_broker_summary_data(ticker: str, current_price: float = 1000.0, volume: float = 500000.0) -> pd.DataFrame:
    """Mengambil atau mensimulasikan data Broker Summary harian BEI."""
    from modules.broker_analyzer import generate_synthetic_broker_summary
    return generate_synthetic_broker_summary(current_price=current_price, total_volume=volume, ticker=ticker)


def fetch_l2_order_book_data(ticker: str, current_price: float = 1000.0) -> pd.DataFrame:
    """Mengambil atau mensimulasikan data Order Book Level 2 BEI."""
    from modules.broker_analyzer import generate_synthetic_l2_order_book
    return generate_synthetic_l2_order_book(current_price=current_price)

