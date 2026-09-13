"""
idx_ticks.py
============
Mesin Fraksi Harga Resmi Bursa Efek Indonesia (BEI / IDX Ticks Engine)
Sesuai Peraturan BEI No. II-A tentang Perdagangan Efek Bersifat Ekuitas:
  - Kelompok 1: Harga < Rp 200          -> Fraksi Rp 1
  - Kelompok 2: Rp 200 s.d. < Rp 500    -> Fraksi Rp 2
  - Kelompok 3: Rp 500 s.d. < Rp 2.000  -> Fraksi Rp 5
  - Kelompok 4: Rp 2.000 s.d. < Rp 5.000-> Fraksi Rp 10
  - Kelompok 5: Harga >= Rp 5.000       -> Fraksi Rp 25
"""

import math
from typing import Any


def get_idx_tick_size(price: float) -> int:
    """Mengembalikan besaran fraksi harga resmi BEI berdasarkan kelompok harga."""
    if price is None:
        return 1
    try:
        p = float(price)
        if math.isnan(p) or math.isinf(p) or p <= 0:
            return 1
        if p < 200:
            return 1
        elif p < 500:
            return 2
        elif p < 2000:
            return 5
        elif p < 5000:
            return 10
        else:
            return 25
    except Exception:
        return 1


def round_to_idx_tick(price: float, round_direction: str = "nearest") -> int:
    """
    Membulatkan harga ke fraksi kelipatan resmi BEI terdekat.
    round_direction: 'nearest', 'up' (ceil), atau 'down' (floor)
    """
    if price is None:
        return 50
    try:
        p = float(price)
        if math.isnan(p) or math.isinf(p) or p <= 0:
            return 50
        tick = get_idx_tick_size(p)
        if round_direction == "up":
            val = math.ceil(p / tick) * tick
        elif round_direction == "down":
            val = math.floor(p / tick) * tick
        else:
            val = round(p / tick) * tick
        return int(val) if not math.isnan(val) and not math.isinf(val) else 50
    except Exception:
        return 50


def safe_int(val: Any, default: int = 0) -> int:
    """Mengonversi nilai apapun secara aman ke integer tanpa error NaN/None/Inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(round(f))
    except Exception:
        return default
