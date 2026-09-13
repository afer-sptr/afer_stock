"""
Modul Analisis Fundamental Saham Indonesia (IDX) - Terpadu.
Mengevaluasi valuasi, profitabilitas, solvabilitas, rasio utang, dividen,
kapitalisasi pasar, peringatan Insolvency Veto (DER > 4.0x), serta indikator makro.
"""

from typing import Dict, Any, Optional
import yfinance as yf


def evaluate_fundamental_score(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Menilai metrik fundamental emiten IDX berdasarkan rasio keuangan standar
    dan kriteria proteksi institusional.
    """
    if not info:
        return {
            "score": 50,
            "status": "DATA TIDAK LENGKAP",
            "valuation": "UNKNOWN",
            "insights": ["Informasi fundamental tidak tersedia dari data feed."],
            "metrics": {},
            "insolvency_veto": False,
        }

    # Ambil metrik dasar dengan fallback aman
    pe = info.get("trailingPE") or info.get("forwardPE")
    pbv = info.get("priceToBook")
    roe = info.get("returnOnEquity")  # decimal: 0.15 = 15%
    der = info.get("debtToEquity")    # percentage in yfinance (misal: 110.0 = 110%)
    npm = info.get("profitMargins")   # decimal: 0.20 = 20%
    curr_ratio = info.get("currentRatio")
    div_yield = info.get("dividendYield") or 0.0
    market_cap = info.get("marketCap") or 0
    sector = info.get("sector", "Lainnya")
    company_name = info.get("longName") or info.get("shortName", "Emiten IDX")

    cap_trillion = market_cap / 1e12
    if cap_trillion >= 50:
        cap_tier = "Lapis 1 (Big Cap / Bluechip)"
    elif cap_trillion >= 10:
        cap_tier = "Lapis 2 (Mid Cap)"
    else:
        cap_tier = "Lapis 3 (Small Cap)"

    score = 50
    insights = []
    is_financial = "Financial" in sector or "Bank" in company_name

    # 1. P/E Ratio
    if pe is not None and pe > 0:
        if pe < 10:
            score += 15
            insights.append(f"P/E Ratio {pe:.1f}x: Valuasi sangat murah (Undervalued)")
        elif pe <= 18:
            score += 10
            insights.append(f"P/E Ratio {pe:.1f}x: Valuasi wajar untuk bursa BEI")
        elif pe > 30:
            score -= 10
            insights.append(f"P/E Ratio {pe:.1f}x: Valuasi premium/relatif mahal")
    else:
        insights.append("P/E Ratio negatif atau belum tersedia (Perusahaan merugi)")
        score -= 10

    # 2. PBV Ratio
    if pbv is not None and pbv > 0:
        if pbv < 1.0:
            score += 12
            insights.append(f"PBV {pbv:.2f}x: Saham dijual di bawah nilai buku aset (Undervalued)")
        elif pbv <= 2.5:
            score += 8
            insights.append(f"PBV {pbv:.2f}x: Valuasi PBV wajar")
        elif pbv > 5.0 and not is_financial:
            score -= 8
            insights.append(f"PBV {pbv:.2f}x: Cukup premium dibanding aset buku riil")

    # 3. ROE & Net Margin
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct >= 18:
            score += 15
            insights.append(f"ROE {roe_pct:.1f}%: Efisiensi modal istimewa (High Quality Business)")
        elif roe_pct >= 10:
            score += 8
            insights.append(f"ROE {roe_pct:.1f}%: Profitabilitas modal sehat")
        elif roe_pct < 5:
            score -= 10
            insights.append(f"ROE {roe_pct:.1f}%: Efisiensi modal rendah")

    if npm is not None:
        npm_pct = npm * 100
        if npm_pct >= 15:
            score += 10
            insights.append(f"Net Profit Margin {npm_pct:.1f}%: Marjin laba bersih tebal")
        elif npm_pct < 5:
            score -= 5
            insights.append(f"Net Profit Margin {npm_pct:.1f}%: Marjin laba bersih tipis")

    # 4. Solvabilitas & Insolvency Veto (DER > 400% / 4.0x)
    der_val = float(der) if der is not None else 80.0
    insolvency_veto = False
    if not is_financial:
        if der_val > 400.0 or (roe is not None and roe < -0.20):
            insolvency_veto = True
            score -= 30
            insights.append(f"🚨 INSOLVENCY VETO: DER mencapai {der_val:.1f}% (> 4.0x)! Risiko gagal bayar utang ekstrem.")
        elif der_val > 200.0:
            score -= 15
            insights.append(f"DER {der_val:.1f}%: Beban hutang tinggi (Risiko Solvabilitas)")
        elif der_val < 80.0:
            score += 10
            insights.append(f"DER {der_val:.1f}%: Beban hutang rendah & neraca sangat sehat")

    # 5. Current Ratio
    if curr_ratio is not None:
        if curr_ratio >= 1.5:
            score += 8
            insights.append(f"Current Ratio {curr_ratio:.2f}x: Likuiditas jangka pendek prima")
        elif curr_ratio < 1.0 and not is_financial:
            score -= 8
            insights.append(f"Current Ratio {curr_ratio:.2f}x: Likuiditas lancar di bawah 1.0x")

    # 6. Dividen
    if div_yield > 0:
        yield_pct = div_yield * 100 if div_yield < 1 else div_yield
        if yield_pct >= 4.0:
            score += 10
            insights.append(f"Dividend Yield {yield_pct:.1f}%: Menarik bagi pencari passive income")
        elif yield_pct >= 2.0:
            score += 5
            insights.append(f"Dividend Yield {yield_pct:.1f}%: Rutin membagikan dividen tunai")
    else:
        insights.append("Fokus ekspansi pertumbuhan (tidak ada dividen saat ini)")

    score = min(max(score, 0), 100)

    if insolvency_veto:
        status = "⚠️ INSOLVENT / SANGAT BERISIKO"
        valuation = "DISTRESS RISK"
    elif score >= 75:
        status = "SANGAT SEHAT & ATRAKTIF"
        valuation = "UNDERVALUED / EXCELLENT"
    elif score >= 60:
        status = "SEHAT & POTENSIAL"
        valuation = "FAIR VALUE"
    elif score <= 35:
        status = "BERISIKO / OVERVALUED"
        valuation = "HIGH RISK / EXPENSIVE"
    else:
        status = "MODERAT"
        valuation = "NEUTRAL"

    metrics = {
        "pe_ratio": round(pe, 2) if pe else None,
        "pbv_ratio": round(pbv, 2) if pbv else None,
        "roe_pct": round(roe * 100, 2) if roe else None,
        "der_pct": round(der_val, 2) if der is not None else None,
        "npm_pct": round(npm * 100, 2) if npm else None,
        "current_ratio": round(curr_ratio, 2) if curr_ratio else None,
        "div_yield_pct": round((div_yield * 100 if div_yield < 1 else div_yield), 2) if div_yield else 0.0,
        "market_cap_idr": market_cap,
        "market_cap_trillion": round(cap_trillion, 2),
        "cap_tier": cap_tier,
        "sector": sector,
        "company_name": company_name,
    }

    return {
        "score": score,
        "status": status,
        "valuation": valuation,
        "insights": insights,
        "metrics": metrics,
        "insolvency_veto": insolvency_veto,
    }


def fetch_macro_snapshot() -> Dict[str, Any]:
    """Mengambil indikator makroekonomi utama: Kurs USD/IDR, S&P 500, Minyak Mentah, Emas."""
    macro_items = {
        "USD/IDR": {"symbol": "IDR=X", "unit": "IDR"},
        "S&P 500": {"symbol": "^GSPC", "unit": "Pts"},
        "Minyak Mentah (WTI)": {"symbol": "CL=F", "unit": "USD/Bbl"},
        "Emas Dunia": {"symbol": "GC=F", "unit": "USD/Oz"},
    }
    results = {}
    for name, meta in macro_items.items():
        try:
            t = yf.Ticker(meta["symbol"])
            hist = t.history(period="5d")
            if not hist.empty and len(hist) >= 2:
                last_p = float(hist["Close"].iloc[-1])
                prev_p = float(hist["Close"].iloc[-2])
                chg = last_p - prev_p
                chg_pct = (chg / prev_p) * 100.0
                results[name] = {
                    "price": last_p,
                    "change": chg,
                    "change_pct": chg_pct,
                    "unit": meta["unit"],
                }
            else:
                results[name] = {"price": 0.0, "change": 0.0, "change_pct": 0.0, "unit": meta["unit"]}
        except Exception:
            results[name] = {"price": 0.0, "change": 0.0, "change_pct": 0.0, "unit": meta["unit"]}
    return results
