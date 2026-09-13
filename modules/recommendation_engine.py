"""
Modul Recommendation Engine Saham Indonesia (IDX) - Multi-Aspek Holistik & Institusional.
Menggabungkan 6 Pilar Utama:
  1. Teknikal & MA Ribbon (25%)
  2. High & Low / Price Action (20%)
  3. Fundamental & Solvabilitas (20%)
  4. Sentimen Berita FinBERT & Veto Alert (15%)
  5. Machine Learning & AI Suite (10%)
  6. Mikrostruktur Buku Pesanan / Order Book (10%)
Dilengkapi:
  - Pembulatan Level Transaksi ke Fraksi Resmi BEI (IDX Ticks: Rp 1, 2, 5, 10, 25)
  - Proyeksi Cuan Bersih (Net PnL) setelah potongan biaya komisi broker dan pajak transaksi bursa (0.40% roundtrip)
  - Proteksi Veto Otomatis (Insolvency Veto DER > 4.0x & News Veto suspensi/delisting/pailit)
"""

from typing import Dict, Any, Tuple, Optional
import math
from modules.idx_ticks import round_to_idx_tick, get_idx_tick_size


def calculate_trading_levels(
    current_price: float,
    support_near: float,
    support_strong: float,
    resistance_near: float,
    resistance_strong: float,
    atr: float,
    high_low_data: Optional[Dict[str, Any]] = None,
    order_book_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Menghitung level harga beli ideal, target take profit (TP1 & TP2),
    dan batas stop loss dengan mengintegrasikan support/resistance, ATR,
    level Fibonacci, serta pembulatan presisi fraksi resmi BEI (IDX Ticks).
    """
    # 1. Entry Range (Zona Beli)
    if support_near < current_price:
        raw_entry_min = support_near
        raw_entry_max = min(current_price, support_near + (0.5 * atr))
    else:
        raw_entry_min = current_price - (0.5 * atr)
        raw_entry_max = current_price

    buy_entry_min = round_to_idx_tick(raw_entry_min, "down")
    buy_entry_max = round_to_idx_tick(raw_entry_max, "nearest")

    # 2. Stop Loss (Cut Loss)
    raw_sl = support_near - (1.0 * atr)
    if high_low_data and "low_20d" in high_low_data:
        raw_sl = min(raw_sl, high_low_data["low_20d"] - (0.5 * atr))

    sl_max_loss = current_price * 0.93   # Maksimum toleransi batas kerugian 7%
    sl_min_loss = current_price * 0.975  # Minimum jarak kerugian 2.5%
    raw_sl_final = min(sl_min_loss, max(sl_max_loss, raw_sl))
    stop_loss = round_to_idx_tick(raw_sl_final, "down")

    # 3. Target Jual 1 (TP1 - Konservatif)
    if resistance_near > current_price:
        raw_tp1 = resistance_near
    else:
        raw_tp1 = current_price + (1.5 * atr)

    if raw_tp1 <= current_price * 1.03:
        raw_tp1 = current_price * 1.04

    tp1 = round_to_idx_tick(raw_tp1, "up")

    # 4. Target Jual 2 (TP2 - Agresif)
    if high_low_data and high_low_data.get("year_high", 0) > current_price * 1.05:
        target_candidate = high_low_data["year_high"]
        raw_tp2 = min(target_candidate, current_price * 1.15)
    elif resistance_strong > tp1:
        raw_tp2 = resistance_strong
    else:
        raw_tp2 = tp1 + (1.5 * atr)

    if raw_tp2 <= tp1:
        raw_tp2 = tp1 * 1.05

    tp2 = round_to_idx_tick(raw_tp2, "up")

    # 5. Risk-to-Reward Ratio (Gross & Net setelah Biaya Transaksi 0.40%)
    fee_roundtrip_pct = 0.40  # 0.15% beli + 0.25% jual (termasuk pajak PPh 0.1%)
    risk_gross_pct = max(0.1, ((current_price - stop_loss) / current_price) * 100.0)
    reward_tp1_gross_pct = max(0.0, ((tp1 - current_price) / current_price) * 100.0)
    reward_tp2_gross_pct = max(0.0, ((tp2 - current_price) / current_price) * 100.0)

    reward_tp1_net_pct = max(0.0, reward_tp1_gross_pct - fee_roundtrip_pct)
    reward_tp2_net_pct = max(0.0, reward_tp2_gross_pct - fee_roundtrip_pct)
    risk_net_pct = risk_gross_pct + fee_roundtrip_pct

    rrr_tp1 = round(reward_tp1_net_pct / max(0.1, risk_net_pct), 2)
    rrr_tp2 = round(reward_tp2_net_pct / max(0.1, risk_net_pct), 2)

    return {
        "buy_entry_min": buy_entry_min,
        "buy_entry_max": buy_entry_max,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "stop_loss": stop_loss,
        "risk_reward_ratio_tp1": rrr_tp1,
        "risk_reward_ratio_tp2": rrr_tp2,
        "risk_pct": round(risk_gross_pct, 1),
        "reward_tp1_pct": round(reward_tp1_gross_pct, 1),
        "reward_tp2_pct": round(reward_tp2_gross_pct, 1),
        "reward_tp1_net_pct": round(reward_tp1_net_pct, 1),
        "reward_tp2_net_pct": round(reward_tp2_net_pct, 1),
        "risk_net_pct": round(risk_net_pct, 1),
        "tick_size": get_idx_tick_size(current_price),
    }


def generate_composite_recommendation(
    tech_result: Dict[str, Any],
    fund_result: Dict[str, Any],
    ml_result: Dict[str, Any],
    current_price: float,
    high_low_result: Optional[Dict[str, Any]] = None,
    news_result: Optional[Dict[str, Any]] = None,
    order_book_result: Optional[Dict[str, Any]] = None,
    ai_suite_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mengintegrasikan seluruh pilar keputusan multi-aspek ke dalam skor terpadu (0 - 100).
    Dilengkapi deteksi Insolvency Veto dan News Veto institusional.
    """
    tech_score = tech_result.get("score", 50)
    fund_score = fund_result.get("score", 50)
    hl_score = high_low_result.get("score", 50) if high_low_result else 50
    news_score = news_result.get("score", 50) if news_result else 50

    # Skor ML (Random Forest + AI Suite jika ada)
    ml_score = 50
    if ml_result.get("success"):
        pct_change = ml_result.get("expected_pct_change", 0.0)
        calc_score = int(50 + (pct_change * 8))
        ml_score = max(10, min(95, calc_score))

    if ai_suite_result and "master_ai_win_prob" in ai_suite_result:
        ai_prob = ai_suite_result["master_ai_win_prob"]
        ml_score = round((ml_score * 0.4) + (ai_prob * 0.6))

    # Skor Order Book (% Bid)
    ob_score = 50
    if order_book_result:
        ob_score = round(float(order_book_result.get("pct_bid", 50.0)))

    # Pembobotan Multi-Pilar Terkalibrasi:
    # Teknikal 25%, High/Low 20%, Fundamental 20%, Sentimen Berita 15%, ML/AI 10%, Order Book 10%
    raw_composite = (
        (tech_score * 0.25)
        + (hl_score * 0.20)
        + (fund_score * 0.20)
        + (news_score * 0.15)
        + (ml_score * 0.10)
        + (ob_score * 0.10)
    )
    composite_score = int(round(raw_composite))

    # Cek Kondisi Veto Proteksi Modal (Insolvency Veto & News Veto)
    insolvency_veto = bool(fund_result.get("insolvency_veto", False))
    news_veto = bool(news_result.get("news_veto", False)) if news_result else False
    is_veto = insolvency_veto or news_veto

    # Penentuan Label Keputusan
    if is_veto:
        action = "🔴 VETO / HIGH RISK"
        if insolvency_veto and news_veto:
            action_desc = "VETO AKTIF: Beban utang sangat berbahaya (DER > 4.0x) DAN muncul berita krisis/hukum bursa. Dilarang membeli!"
        elif insolvency_veto:
            action_desc = "INSOLVENCY VETO: Struktur permodalan tidak sehat (DER > 4.0x atau ekuitas negatif). Risiko pailit tinggi."
        else:
            action_desc = "NEWS VETO: Terdeteksi pemberitaan hukum/suspensi/delisting/PKPU. Posisi beli dibatalkan secara otomatis."
        badge_color = "red"
    elif composite_score >= 76 and ob_score >= 55:
        action = "STRONG BUY"
        action_desc = "Konvergensi positif menyeluruh: momentum teknikal kuat, posisi breakout, fundamental sehat, sentimen berita positif, dan dominasi order book beli."
        badge_color = "green"
    elif composite_score >= 60:
        action = "BUY"
        action_desc = "Setup akumulasi menguntungkan dengan rasio risk/reward sehat dan dukungan sentimen positif."
        badge_color = "cyan"
    elif composite_score <= 35:
        action = "STRONG SELL"
        action_desc = "Tekanan jual dominan di area resisten, sentimen negatif, atau risiko penurunan lanjutan tinggi."
        badge_color = "red"
    elif composite_score <= 47:
        action = "SELL / TAKE PROFIT"
        action_desc = "Momentum harga melemah atau antrean offer tebal menahan kenaikan. Disarankan mengamankan modal."
        badge_color = "orange"
    else:
        action = "HOLD / WAIT & SEE"
        action_desc = "Kondisi konsolidasi/netral. Dinamika harga dan sentimen berimbang, disarankan menunggu konfirmasi tren."
        badge_color = "yellow"

    # Perhitungan Level Transaksi Berfraksi BEI
    levels = calculate_trading_levels(
        current_price=current_price,
        support_near=tech_result.get("support_near", current_price * 0.95),
        support_strong=tech_result.get("support_strong", current_price * 0.90),
        resistance_near=tech_result.get("resistance_near", current_price * 1.05),
        resistance_strong=tech_result.get("resistance_strong", current_price * 1.10),
        atr=tech_result.get("atr", current_price * 0.02),
        high_low_data=high_low_result,
        order_book_data=order_book_result,
    )

    return {
        "action": action,
        "action_desc": action_desc,
        "badge_color": badge_color,
        "composite_score": composite_score,
        "is_veto": is_veto,
        "insolvency_veto": insolvency_veto,
        "news_veto": news_veto,
        "breakdown_scores": {
            "technical": tech_score,
            "high_low": hl_score,
            "fundamental": fund_score,
            "news_sentiment": news_score,
            "ml_prediction": ml_score,
            "order_book": ob_score,
        },
        "trading_plan": levels
    }


def calculate_position_size(
    capital_idr: float,
    max_risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    safe_exit_lot_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Kalkulator alokasi modal dan ukuran lot saham (1 Lot = 100 lembar) berbasis risiko portofolio
    dengan batas pengaman Safe Exit Lot untuk menghindari risiko slippage likuiditas.
    """
    if entry_price <= stop_loss_price or capital_idr <= 0:
        return {
            "lots": 0,
            "shares": 0,
            "total_investment": 0,
            "capital_usage_pct": 0,
            "max_loss_idr": 0,
            "safe_exit_capped": False,
        }

    risk_budget = capital_idr * (max_risk_pct / 100.0)
    risk_per_share = entry_price - stop_loss_price
    
    raw_shares = risk_budget / risk_per_share
    lots = math.floor(raw_shares / 100)

    max_affordable_lots = math.floor(capital_idr / (entry_price * 100))
    final_lots = min(lots, max_affordable_lots)

    safe_exit_capped = False
    if safe_exit_lot_cap and safe_exit_lot_cap > 0 and final_lots > safe_exit_lot_cap:
        final_lots = safe_exit_lot_cap
        safe_exit_capped = True

    shares = final_lots * 100
    total_investment = shares * entry_price
    max_loss_idr = shares * risk_per_share
    usage_pct = (total_investment / capital_idr) * 100 if capital_idr > 0 else 0

    return {
        "lots": final_lots,
        "shares": shares,
        "total_investment": round(total_investment),
        "capital_usage_pct": round(usage_pct, 1),
        "max_loss_idr": round(max_loss_idr),
        "safe_exit_capped": safe_exit_capped,
    }
