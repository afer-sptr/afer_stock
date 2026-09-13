"""
Modul Kalkulasi Harga Wajar (Fair Value / Nilai Intrinsik) Saham Indonesia (IDX).
Menggunakan model kuantitatif Value Investing:
1. Formula Benjamin Graham (Graham Number)
2. Justified PBV - ROE Multiple
3. Historical P/E Industry Multiplier
4. Dividend Discount Model (DDM / Gordon Growth)
Menghitung Konsensus Harga Wajar dan Margin of Safety (MoS %).
"""

from typing import Dict, Any, Optional
import math


def calculate_fair_value(
    current_price: float,
    info: Dict[str, Any],
    df_history: Any = None
) -> Dict[str, Any]:
    """
    Menghitung estimasi harga wajar (nilai intrinsik) saham secara real-time.
    """
    eps = info.get("trailingEps") or info.get("forwardEps")
    bvps = info.get("bookValue")
    roe = info.get("returnOnEquity")      # format decimal (0.18 = 18%)
    pe = info.get("trailingPE")
    div_rate = info.get("dividendRate")  # DPS dalam Rupiah

    models = {}
    valid_values = []

    # 1. Formula Benjamin Graham: V = sqrt(22.5 * EPS * BVPS)
    # Berlaku jika EPS > 0 dan BVPS > 0
    if eps and bvps and eps > 0 and bvps > 0:
        graham_val = math.sqrt(22.5 * eps * bvps)
        models["graham_number"] = round(graham_val)
        valid_values.append(graham_val)
    else:
        models["graham_number"] = None

    # 2. Justified PBV - ROE Model
    # Asumsi Cost of Equity standar di BEI (Indonesia) = 11% (Risk Free ~6.5% + Equity Risk Premium ~4.5%)
    # Fair PBV = ROE / Cost_of_Equity -> Fair Price = Fair PBV * BVPS
    if roe and bvps and roe > 0 and bvps > 0:
        cost_of_equity = 0.11
        justified_pbv = roe / cost_of_equity
        justified_price = justified_pbv * bvps
        models["justified_pbv_price"] = round(justified_price)
        models["justified_pbv"] = round(justified_pbv, 2)
        valid_values.append(justified_price)
    else:
        models["justified_pbv_price"] = None
        models["justified_pbv"] = None

    # 3. P/E Multiplier Wajar (Rata-rata BEI ~ 15x untuk emiten profit)
    if eps and eps > 0:
        fair_pe_mult = 15.0
        pe_fair_price = eps * fair_pe_mult
        models["pe_multiple_price"] = round(pe_fair_price)
        valid_values.append(pe_fair_price)
    else:
        models["pe_multiple_price"] = None

    # 4. Dividend Discount Model (DDM / Gordon Growth)
    # Jika membagikan dividen rutin: V = DPS * (1 + g) / (r - g)
    # r = 10.5%, g = 4.5% (pertumbuhan ekonomi Indonesia jangka panjang)
    if div_rate and div_rate > 0:
        r = 0.105
        g = 0.045
        ddm_price = (div_rate * (1 + g)) / (r - g)
        models["ddm_price"] = round(ddm_price)
        # Batasi agar DDM tidak terlalu ekstrim
        if ddm_price > 0 and ddm_price < current_price * 4:
            valid_values.append(ddm_price)
    else:
        models["ddm_price"] = None

    # 5. Konsensus Nilai Wajar Gabungan (Fair Value Blend)
    if valid_values:
        fair_value = round(sum(valid_values) / len(valid_values))
    else:
        # Fallback jika data EPS/BVPS nihil: gunakan pendekatan valuasi pasar
        fair_value = round(current_price)

    # 6. Kalkulasi Margin of Safety (MoS %)
    # MoS = (Fair Value - Current Price) / Fair Value * 100%
    if fair_value > 0:
        mos_pct = round(((fair_value - current_price) / fair_value) * 100, 2)
    else:
        mos_pct = 0.0

    # Penentuan Status Valuasi Real-Time
    if mos_pct >= 25.0:
        status = "DISKON SANGAT BESAR (DEEP UNDERVALUED)"
        status_desc = "Harga pasar saat ini jauh di bawah nilai intrinsiknya. Menawarkan potensi keuntungan jangka panjang yang sangat besar dengan proteksi modal tinggi."
        badge_color = "green"
    elif mos_pct >= 10.0:
        status = "UNDERVALUED (DISKON MENARIK)"
        status_desc = "Harga saham berada di bawah harga wajar. Waktu yang baik untuk akumulasi beli."
        badge_color = "teal"
    elif mos_pct >= -10.0:
        status = "FAIR VALUE (HARGA WAJAR)"
        status_desc = "Harga pasar saat ini sudah mencerminkan nilai wajar fundamental emiten."
        badge_color = "yellow"
    else:
        status = "OVERVALUED (HARGA MAHAL/PREMIUM)"
        status_desc = "Harga pasar sudah melebihi nilai intrinsiknya. Berhati-hati terhadap risiko koreksi harga kembali ke nilai wajarnya."
        badge_color = "red"

    return {
        "fair_value": fair_value,
        "current_price": round(current_price),
        "margin_of_safety_pct": mos_pct,
        "status": status,
        "status_desc": status_desc,
        "badge_color": badge_color,
        "models": models,
        "eps_idr": round(eps, 2) if eps else None,
        "bvps_idr": round(bvps, 2) if bvps else None
    }
