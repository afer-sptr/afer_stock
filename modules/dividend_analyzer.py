"""
Modul Analisis Dividen & Strategi Cuan Maksimal Saham Indonesia (IDX).
Mengekstrak tanggal Ex-Date, estimasi Cum-Date, Dividend Yield, DPS, DPR,
serta menyusun Rekomendasi Waktu Beli Optimal untuk memaksimalkan hasil
dari kombinasi Dividen Tunai dan Capital Gain tanpa terkena Dividend Trap.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


def analyze_dividend_schedule_and_strategy(
    ticker_code: str,
    info: Dict[str, Any],
    current_price: float,
    stock_obj: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Menganalisis jadwal dividen, riwayat pembayaran, dan strategi membeli optimal.
    """
    if stock_obj is None:
        try:
            stock_obj = yf.Ticker(ticker_code)
        except Exception:
            stock_obj = None

    # 1. Ekstraksi Metrik Dividen
    div_rate = info.get("dividendRate")     # DPS total tahunan dalam Rupiah
    div_yield_raw = info.get("dividendYield") or 0.0
    div_yield_pct = (div_yield_raw * 100 if div_yield_raw < 1 else div_yield_raw) if div_yield_raw else 0.0
    payout_ratio_raw = info.get("payoutRatio")
    payout_ratio_pct = round(payout_ratio_raw * 100, 1) if payout_ratio_raw else None

    # 2. Tanggal Ex-Dividen & Estimasi Cum-Date
    ex_timestamp = info.get("exDividendDate")
    ex_date_str = "-"
    cum_date_str = "-"
    days_to_ex = None

    if ex_timestamp:
        try:
            ex_dt = datetime.fromtimestamp(ex_timestamp)
            ex_date_str = ex_dt.strftime("%d %B %Y")
            # Cum-Date di BEI adalah 1 hari kerja bursa sebelum Ex-Date
            # Mundur 1 hari (jika hari senin, mundur ke jumat)
            if ex_dt.weekday() == 0: # Senin
                cum_dt = ex_dt - timedelta(days=3)
            elif ex_dt.weekday() == 6: # Minggu
                cum_dt = ex_dt - timedelta(days=2)
            else:
                cum_dt = ex_dt - timedelta(days=1)
            cum_date_str = cum_dt.strftime("%d %B %Y")
            
            diff_days = (ex_dt - datetime.now()).days
            days_to_ex = diff_days
        except Exception:
            pass

    # 3. Riwayat Pembayaran Dividen Historis
    history_list = []
    if stock_obj is not None:
        try:
            div_series = stock_obj.dividends
            if div_series is not None and not div_series.empty:
                # Ambil 6 dividen terakhir
                recent_divs = div_series.tail(6)
                for dt_idx, dps_val in recent_divs.items():
                    dt_str = pd.to_datetime(dt_idx).strftime("%d %b %Y")
                    history_list.append({
                        "tanggal": dt_str,
                        "dps_rupiah": float(dps_val)
                    })
                history_list.reverse()
        except Exception:
            pass

    # 4. Rekomendasi Strategi Membeli untuk Cuan Maksimal
    strategy_recommendations = []
    has_attractive_dividend = div_yield_pct >= 4.0

    if div_yield_pct >= 7.0:
        div_tier = "DIVIDEN JUMBO (SUPER HIGH YIELD)"
        div_desc = f"Dividend Yield mencapai {div_yield_pct:.2f}%, sangat besar dibanding bunga deposito perbankan (~3-4%)."
    elif div_yield_pct >= 4.0:
        div_tier = "DIVIDEN MENARIK (HEALTHY YIELD)"
        div_desc = f"Dividend Yield sebesar {div_yield_pct:.2f}%, stabil dan menguntungkan untuk investor pasif."
    elif div_yield_pct > 0:
        div_tier = "DIVIDEN MODERAT"
        div_desc = f"Dividend Yield sebesar {div_yield_pct:.2f}%, perusahaan lebih fokus mengalokasikan laba untuk ekspansi bisnis."
    else:
        div_tier = "TIDAK ADA DIVIDEN"
        div_desc = "Emiten belum membagikan dividen tunai dalam periode terkini."

    # Rumus Strategi Dividen vs Capital Gain
    if has_attractive_dividend:
        strategy_recommendations.append(
            "WAKTU BELI OPTIMAL: Akumulasi 1 s/d 2 bulan sebelum masa RUPS Tahunan saat harga masih berada di area support. "
            "Jangan pernah membeli di saat harga sudah mendekati hari H Cum-Date karena risiko 'Dividend Trap' sangat tinggi."
        )
        strategy_recommendations.append(
            f"STRATEGI MENANGKAN CUAN MAKSIMAL: Jika harga saham naik melampaui persentase yield (+{div_yield_pct:.1f}% ke atas) "
            "menjelang Cum-Date, disarankan merealisasikan Capital Gain (jual sebelum Cum-Date). Anda mengunci keuntungan modal yang "
            "lebih besar tanpa risiko penurunan tajam pada Ex-Date."
        )
        strategy_recommendations.append(
            "DIVIDEND REINVESTMENT (DRIP): Jika Anda berinvestasi jangka panjang, gunakan dana dividen tunai yang masuk ke RDN "
            "untuk membeli kembali saham saat terjadi koreksi pasca Ex-Date, guna melipatgandakan jumlah lot Anda secara eksponensial."
        )
    else:
        strategy_recommendations.append(
            "Emiten ini lebih cocok difokuskan pada strategi SWING TRADING berbasis Capital Gain (Memanfaatkan fluktuasi teknikal "
            "dan momentum breakout) daripada dividen tunai."
        )

    return {
        "has_dividend": div_yield_pct > 0,
        "dividend_rate_idr": round(div_rate, 2) if div_rate else (history_list[0]["dps_rupiah"] if history_list else 0.0),
        "dividend_yield_pct": round(div_yield_pct, 2),
        "payout_ratio_pct": payout_ratio_pct,
        "ex_dividend_date": ex_date_str,
        "cum_dividend_date": cum_date_str,
        "days_to_ex": days_to_ex,
        "dividend_tier": div_tier,
        "dividend_desc": div_desc,
        "history": history_list,
        "strategies": strategy_recommendations
    }


def calculate_dividend_payout_simulation(
    lots: int,
    dps_idr: float,
    current_price: float,
    tax_exempt: bool = True
) -> Dict[str, Any]:
    """
    Menghitung simulasi penerimaan dividen tunai bersih.
    Standar BEI: 1 Lot = 100 Lembar.
    Pajak dividen Indonesia: 0% jika diinvestasikan kembali (UU Cipta Kerja), atau 10% PPh Final.
    """
    total_shares = lots * 100
    gross_payout = total_shares * dps_idr
    tax_rate = 0.0 if tax_exempt else 0.10
    tax_amount = gross_payout * tax_rate
    net_payout = gross_payout - tax_amount
    total_capital = total_shares * current_price
    effective_yield = (net_payout / total_capital * 100) if total_capital > 0 else 0.0

    return {
        "lots": lots,
        "shares": total_shares,
        "gross_dividend_idr": round(gross_payout),
        "tax_amount_idr": round(tax_amount),
        "net_dividend_idr": round(net_payout),
        "effective_yield_pct": round(effective_yield, 2)
    }
