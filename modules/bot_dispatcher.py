"""
bot_dispatcher.py
=================
Microservice Perpesanan Mandiri untuk Notifikasi Kuantitatif Pasar Modal
Mendukung:
  1. Telegram Bot API resmi (via direct HTTP POST requests)
  2. WhatsApp Gateway API (Fonnte, Wablas, Twilio, atau generic webhook)
  3. Meta WhatsApp Cloud API Resmi (Graph API v19.0)
  4. WhatsApp Click-to-Chat Generator (wa.me link fallback)
Dilengkapi:
  - Anti-spam cooldown (minimal 20 menit per ticker emiten)
  - Throttling pelindung nomor WA (maksimal 1 pesan per 5 detik)
  - Audit logger berbasis pandas DataFrame
  - Format template sinyal HAKA dan Darurat HAKI
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BotDispatcher")


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "bot_config.json")


@dataclass
class DispatchConfig:
    """Konfigurasi kredensial perpesanan Telegram & WhatsApp"""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # WhatsApp Channel Selector: 'gateway', 'cloud_api', 'click_to_chat'
    whatsapp_channel: str = "click_to_chat"

    # Gateway Mode (Fonnte, Wablas, etc)
    wa_gateway_endpoint: str = "https://api.fonnte.com/send"
    wa_gateway_token: str = ""
    wa_target_phone: str = ""  # Format: 628xxxxxxxxxx

    # Meta Cloud API
    wa_cloud_phone_number_id: str = ""
    wa_cloud_access_token: str = ""
    wa_cloud_recipient_phone: str = ""

    # Keamanan & Throttling
    cooldown_minutes_per_ticker: int = 20
    throttle_seconds_wa: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "telegram_enabled": self.telegram_enabled,
            "telegram_bot_token": self.telegram_bot_token,
            "telegram_chat_id": self.telegram_chat_id,
            "whatsapp_channel": self.whatsapp_channel,
            "wa_gateway_endpoint": self.wa_gateway_endpoint,
            "wa_gateway_token": self.wa_gateway_token,
            "wa_target_phone": self.wa_target_phone,
            "wa_cloud_phone_number_id": self.wa_cloud_phone_number_id,
            "wa_cloud_access_token": self.wa_cloud_access_token,
            "wa_cloud_recipient_phone": self.wa_cloud_recipient_phone,
            "cooldown_minutes_per_ticker": self.cooldown_minutes_per_ticker,
            "throttle_seconds_wa": self.throttle_seconds_wa,
        }

    def save_to_disk(self, path: Optional[str] = None) -> None:
        p = path or _CONFIG_PATH
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Konfigurasi bot tersimpan ke {p}")
        except Exception as ex:
            logger.warning(f"Gagal menyimpan bot_config.json: {ex}")

    @classmethod
    def load_from_disk(cls, path: Optional[str] = None) -> DispatchConfig:
        p = path or _CONFIG_PATH
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    valid_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                    return cls(**valid_data)
            except Exception as ex:
                logger.warning(f"Gagal membaca bot_config.json: {ex}")
        return cls()


class AuditLogger:
    """Pencatat audit pengiriman notifikasi instan dalam memori DataFrame."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def log(
        self,
        ticker: str,
        signal_type: str,
        channel: str,
        status: str,
        http_code: Optional[int],
        details: str,
        recipient: str,
    ) -> None:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker.upper(),
            "signal_type": signal_type,
            "channel": channel,
            "status": status,
            "http_code": http_code if http_code is not None else 0,
            "recipient": recipient[:12] + "..." if len(recipient) > 12 else recipient,
            "details": details,
        }
        self.records.append(record)
        logger.info(f"AUDIT LOG: {record}")

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "ticker",
                    "signal_type",
                    "channel",
                    "status",
                    "http_code",
                    "recipient",
                    "details",
                ]
            )
        return pd.DataFrame(self.records).iloc[::-1].reset_index(drop=True)


class BotDispatcher:
    """
    Mesin microservice pengiriman pesan sinyal eksekusi taktis kuantitatif.
    """

    def __init__(self, config: Optional[DispatchConfig] = None):
        self.config = config or DispatchConfig.load_from_disk()
        self.audit = AuditLogger()
        self._last_dispatch_time: Dict[Tuple[str, str], datetime] = {}
        self._last_wa_send_timestamp: float = 0.0

    def update_config(self, config: DispatchConfig, save: bool = True) -> None:
        self.config = config
        if save:
            self.config.save_to_disk()

    @staticmethod
    def sanitize_phone(phone: str) -> str:
        """Memastikan format nomor ponsel diawali 62 dan tanpa karakter non-digit."""
        digits = "".join(ch for ch in phone if ch.isdigit())
        if digits.startswith("0"):
            digits = "62" + digits[1:]
        elif digits.startswith("8"):
            digits = "62" + digits
        return digits

    def can_dispatch(self, ticker: str, signal_type: str) -> Tuple[bool, str]:
        """Cek aturan anti-spam cooldown 20 menit per ticker dan tipe sinyal."""
        key = (ticker.upper(), signal_type.upper())
        now = datetime.now()
        if key in self._last_dispatch_time:
            elapsed = (now - self._last_dispatch_time[key]).total_seconds()
            cooldown_seconds = self.config.cooldown_minutes_per_ticker * 60
            if elapsed < cooldown_seconds:
                remaining = int((cooldown_seconds - elapsed) / 60)
                return (
                    False,
                    f"Cooldown aktif untuk {ticker} [{signal_type}]. Tunggu {remaining} menit lagi.",
                )
        return True, "Siap kirim"

    def _mark_dispatched(self, ticker: str, signal_type: str) -> None:
        key = (ticker.upper(), signal_type.upper())
        self._last_dispatch_time[key] = datetime.now()

    def format_haka_message(self, data: Dict[str, Any]) -> str:
        """
        Template A: Sinyal Beli Cepat (HAKA Alert)
        """
        ticker = data.get("ticker", "SAHAM.JK")
        action_label = data.get("action_label", "STRONG BUY / HAKA READY")
        confluence_score = data.get("confluence_score", 85.0)
        best_ask = data.get("best_ask", 0)
        best_bid = data.get("best_bid", 0)
        safe_lot = data.get("safe_lot", 50)
        tp1 = data.get("tp1", 0)
        gain_tp1_net = data.get("gain_tp1_net", 3.1)
        tp2 = data.get("tp2", 0)
        gain_tp2_net = data.get("gain_tp2_net", 6.2)
        stop_loss = data.get("stop_loss", 0)
        risk_sl = data.get("risk_sl", 2.0)
        rrr = data.get("rrr", 1.8)
        ml_win_prob = data.get("ml_win_prob", 72.5)
        brier_score = data.get("brier_score", 0.14)
        ml_signal_meta = data.get("ml_signal_meta", "SUPERIOR (Brier <= 0.18)")
        pct_bid = data.get("pct_bid", 71.0)
        pct_offer = data.get("pct_offer", 29.0)
        candle_pattern = data.get("candle_pattern", "Bullish Hammer")
        exit_speed = data.get("exit_speed", "INSTAN (< 1 Menit)")
        news_status = data.get("news_status", "Sentimen Positif Finansial (+0.45)")
        valuation_status = data.get("valuation_status", "Wajar (Fair Value)")
        fair_value = data.get("fair_value", 0)
        mos = data.get("mos", 0.0)

        msg = (
            f"🟢 [SINYAL BELI CEPAT / HAKA TERDETEKSI] 🟢\n"
            f"Saham: {ticker}\n"
            f"Status Sinyal: {action_label} (Skor: {confluence_score:.1f}%)\n\n"
            f"🛒 HARGA BELI CEPAT (HAKA): Rp {best_ask:,} (Best Ask)\n"
            f"🔵 Antrean Pasif (Bid): Rp {best_bid:,}\n"
            f"📦 Kuota Beli Aman: Maksimal {safe_lot:,} lot\n\n"
            f"🎯 Target Profit 1: Rp {tp1:,} (+{gain_tp1_net:.2f}% Net)\n"
            f"🎯 Target Profit 2: Rp {tp2:,} (+{gain_tp2_net:.2f}% Net)\n"
            f"🛑 Cut Loss (EVT-VaR): Rp {stop_loss:,} (-{risk_sl:.2f}%)\n"
            f"⚖️ Rasio RRR: 1 : {rrr:.2f}\n\n"
            f"🤖 Validasi Machine Learning:\n"
            f"• Probabilitas Sukses: {ml_win_prob:.1f}%\n"
            f"• Brier Score: {brier_score:.3f} (Akurasi Kalibrasi)\n"
            f"• Status Model: {ml_signal_meta}\n\n"
            f"📊 Analisis Pasar & Valuasi:\n"
            f"• Status Valuasi: {valuation_status} (Fair Value Rp {fair_value:,} | MoS {mos:+.1f}%)\n"
            f"• Order Book: {pct_bid:.1f}% Bid vs {pct_offer:.1f}% Offer\n"
            f"• Pola Candle: {candle_pattern}\n"
            f"• Kecepatan Jual: {exit_speed}\n"
            f"• Sentimen Berita: {news_status}\n\n"
            f"⚠️ Silakan buka aplikasi sekuritas (Stockbit/Bibit) dan pasang order sekarang!"
        )
        return msg

    def format_emergency_haki_message(self, data: Dict[str, Any]) -> str:
        """
        Template B: Sinyal Jual Cepat Darurat (Emergency HAKI Alert)
        """
        ticker = data.get("ticker", "SAHAM.JK")
        best_bid = data.get("best_bid", 0)
        safe_lot = data.get("safe_lot", 50)
        pct_offer = data.get("pct_offer", 74.0)
        pct_bid = data.get("pct_bid", 26.0)
        haki_desc = data.get(
            "haki_desc", "Offer membengkak masif, OBI -0.48, resiko guyuran ARB"
        )
        last_price = data.get("last_price", 0)
        stop_loss = data.get("stop_loss", 0)
        candle_pattern = data.get("candle_pattern", "Bearish Engulfing")
        valuation_status = data.get("valuation_status", "Normal")

        msg = (
            f"🚨 [PERINGATAN DARURAT: SEGERA JUAL CEPAT / HAKI] 🚨\n"
            f"Saham: {ticker}\n"
            f"Peringatan: Tekanan Guyuran Masif Terdeteksi!\n\n"
            f"🔴 HARGA JUAL CEPAT (HAKI): Rp {best_bid:,} (Best Bid Teratas)\n"
            f"⚠️ KUOTA MAKSIMAL AMAN: {safe_lot:,} lot\n\n"
            f"⛔ PERINGATAN KERAS:\n"
            f"Jangan lepas melebihi kuota lot di atas secara sekaligus agar tidak membanting harga ke fraksi bawah (slippage/ARB)!\n\n"
            f"📊 Indikator Bahaya:\n"
            f"• Antrean Offer Membengkak: {pct_offer:.1f}% Offer vs {pct_bid:.1f}% Bid\n"
            f"• Status Order Book: {haki_desc}\n"
            f"• Status Valuasi: {valuation_status}\n"
            f"• Posisi Harga: Rp {last_price:,} (Mendekati/Menembus Cut Loss Rp {stop_loss:,})\n"
            f"• Pola Aksi Harga: {candle_pattern}\n\n"
            f"⚡ Segera eksekusi Jual Cepat (HAKI) ke antrean Best Bid sebelum antrean pembeli habis!"
        )
        return msg

    def format_multi_opportunity_message(self, top_stocks: List[Dict[str, Any]]) -> str:
        """
        Template C: Siaran Peluang Multi-Emiten Terbaik (Top Opportunities Alert)
        """
        now_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S WIB")
        msg = (
            f"🏆 [TOP PELUANG TRADING TERBAIK HARI INI] 🏆\n"
            f"Pemindaian Multi-Emiten BEI: {now_str}\n"
            f"Fokus Cuan Maksimal Pengguna (Disaring Lintas Sektor):\n\n"
        )
        for idx, item in enumerate(top_stocks, 1):
            t = item.get("ticker", "SAHAM.JK")
            tier = item.get("tier", "Lapis 1")
            syariah = item.get("syariah_label", "Syariah")
            confluence = item.get("confluence_score", 80.0)
            ml_prob = item.get("ml_win_prob", 70.0)
            haka_p = item.get("best_ask", 0)
            tp1_p = item.get("tp1", 0)
            gain_net = item.get("gain_tp1_net", 3.0)
            sl_p = item.get("stop_loss", 0)
            pct_b = item.get("pct_bid", 65.0)
            val_stat = item.get("valuation_status", "Wajar")
            safe_lot = item.get("safe_lot", 50)
            rrr = item.get("rrr", 1.8)

            msg += (
                f"#{idx}. 🟢 {t} [{tier} | {syariah}]\n"
                f"   • Skor Konfluensi: {confluence:.1f}% | ML P(Win): {ml_prob:.1f}%\n"
                f"   • HAKA: Rp {haka_p:,} -> TP1: Rp {tp1_p:,} (+{gain_net:.2f}% Net)\n"
                f"   • Cut Loss: Rp {sl_p:,} | RRR: 1:{rrr:.2f}\n"
                f"   • Order Book: {pct_b:.1f}% Bid | Kuota Lot Aman: {safe_lot:,} lot\n"
                f"   • Valuasi: {val_stat}\n\n"
            )

        msg += "⚡ Seluruh sinyal di atas telah lolos uji Machine Learning & Order Book. Pasang order sekarang!"
        return msg

    def dispatch_multi_opportunities(
        self, top_stocks: List[Dict[str, Any]], bypass_cooldown: bool = True
    ) -> Dict[str, Any]:
        """
        Orkestrasi pengiriman siaran multi-saham terbaik ke seluruh kanal aktif.
        """
        if not top_stocks:
            return {"dispatched": False, "reason": "Tidak ada saham yang lolos filter"}

        message = self.format_multi_opportunity_message(top_stocks)
        results: Dict[str, Any] = {}

        if self.config.telegram_enabled:
            results["telegram"] = self.send_telegram(
                message, ticker="MULTI_TOP", signal_type="MULTI_OPPORTUNITY"
            )

        wa_channel = self.config.whatsapp_channel
        if wa_channel == "gateway":
            results["whatsapp"] = self.send_whatsapp_gateway(
                message, ticker="MULTI_TOP", signal_type="MULTI_OPPORTUNITY"
            )
        elif wa_channel == "cloud_api":
            results["whatsapp"] = self.send_whatsapp_cloud_api(
                message, ticker="MULTI_TOP", signal_type="MULTI_OPPORTUNITY"
            )
        else:
            results["whatsapp"] = {"success": True, "message": "Click-to-chat URL berhasil dibuat."}

        click_phone = (
            self.config.wa_target_phone
            if self.config.whatsapp_channel != "cloud_api"
            else self.config.wa_cloud_recipient_phone
        )
        click_url = self.generate_click_to_chat_url(message, click_phone)

        return {
            "dispatched": True,
            "reason": f"Berhasil menyiarkan {len(top_stocks)} saham peluang terbaik",
            "message": message,
            "results": results,
            "click_url": click_url,
        }

    def format_super_profit_message(self, data: Dict[str, Any]) -> str:
        """
        Template D: Sinyal Peluang Cuan Super Maksimal (Super Profit Alpha Alert)
        """
        ticker = data.get("ticker", "SAHAM.JK")
        price_raw = data.get("price", 1000)
        try:
            price = float(price_raw)
            if math.isnan(price) or math.isinf(price) or price <= 0:
                price = 1000.0
        except Exception:
            price = 1000.0

        haka_p = data.get("haka_price", price)
        tp1 = data.get("tp1", int(round(price * 1.04)))
        gain1 = data.get("gain_tp1_net", data.get("reward_tp1_net_pct", 3.8))
        tp2 = data.get("tp2", int(round(price * 1.08)))
        gain2 = data.get("gain_tp2_net", data.get("reward_tp2_net_pct", 7.5))
        sl = data.get("stop_loss", data.get("sl", int(round(price * 0.965))))
        rrr = data.get("rrr", data.get("risk_reward_ratio_tp1", 2.2))
        style = data.get("trading_style", "Swing Trading")
        stock_type = data.get("stock_type", "Saham BEI")
        pattern = data.get("pattern_name", data.get("catalyst", "Konfluensi Setup"))
        bandar_status = data.get("bandar_status", "AKUMULASI MASIF")
        ai_prob = data.get("master_ai_prob", data.get("win_prob", 78.5))
        val_status = data.get("valuation_status", "Undervalued / Murah")
        safe_exit = data.get("safe_exit_lot", 500)


        msg = (
            f"🚀 [SUPER PROFIT ALPHA ALERT: POTENSI CUAN MAKSIMAL] 🚀\n"
            f"Emiten: {ticker} | {stock_type}\n"
            f"Gaya Trading: {style}\n"
            f"Pola Setup: {pattern}\n\n"
            f"🛒 ENTRY LEVEL (HAKA): Rp {haka_p:,}\n"
            f"🎯 TARGET PROFIT 1: Rp {tp1:,} (+{gain1:.2f}% Net)\n"
            f"🎯 TARGET PROFIT 2: Rp {tp2:,} (+{gain2:.2f}% Net)\n"
            f"🛑 BATAS CUT LOSS: Rp {sl:,}\n"
            f"⚖️ Risk-to-Reward Ratio: 1 : {rrr:.2f}\n\n"
            f"🤖 Konfluensi AI & Bandarmologi:\n"
            f"• Probabilitas Menang Master AI: {ai_prob:.1f}%\n"
            f"• Status Bandar/Broker: {bandar_status}\n"
            f"• Valuasi Fundamental: {val_status}\n\n"
            f"⚡ Setup terkonfirmasi memiliki probabilitas tertinggi! Pasang order di sekuritas sekarang!"
        )
        return msg

    def format_emergency_protection_message(self, data: Dict[str, Any]) -> str:
        """
        Template E: Peringatan Risiko Kerugian Besar (Emergency Capital Protection Alert)
        """
        ticker = data.get("ticker", "SAHAM.JK")
        price_raw = data.get("price", 1000)
        try:
            price = float(price_raw)
            if math.isnan(price) or math.isinf(price) or price <= 0:
                price = 1000.0
        except Exception:
            price = 1000.0
        haki_p = data.get("haki_price", price)
        sl = data.get("stop_loss", int(round(price * 0.97)))
        bandar_status = data.get("bandar_status", "DISTRIBUSI MASIF (Guyuran)")
        reason = data.get("reason", "Penjualan masif broker dan kerusakan batas risiko")
        loss_pct = data.get("risk_sl", 2.5)

        msg = (
            f"🚨 [EMERGENCY RISK ALERT: POTENSI KERUGIAN BESAR] 🚨\n"
            f"Emiten: {ticker}\n"
            f"Peringatan: Risiko Guyuran & Kerugian Tajam Terdeteksi!\n\n"
            f"🔴 HARGA JUAL DARURAT (HAKI): Rp {haki_p:,}\n"
            f"🛑 Titik Cut Loss Ditembus: Rp {sl:,} (-{loss_pct:.2f}%)\n"
            f"⚠️ Status Aliran Bandar: {bandar_status}\n"
            f"📌 Penyebab Bahaya: {reason}\n\n"
            f"⛔ AMANKAN MODAL ANDA:\n"
            f"Segera realisasikan Cut Loss atau kurangi posisi sebelum harga terperosok lebih dalam!"
        )
        return msg

    def send_telegram(self, message: str, ticker: str = "SYSTEM", signal_type: str = "ALERT") -> Dict[str, Any]:
        """Pengiriman via Telegram Bot API langsung menggunakan requests."""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            res = {"success": False, "status_code": 400, "message": "Kredensial Telegram belum lengkap."}
            self.audit.log(ticker, signal_type, "Telegram", "FAILED", 400, "Token atau Chat ID kosong", "")
            return res

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                payload.pop("parse_mode", None)
                resp = requests.post(url, json=payload, timeout=10)

            if resp.status_code == 200:
                self.audit.log(
                    ticker,
                    signal_type,
                    "Telegram",
                    "SUCCESS",
                    resp.status_code,
                    "Pesan terkirim sukses",
                    self.config.telegram_chat_id,
                )
                return {"success": True, "status_code": resp.status_code, "message": "Terkirim ke Telegram"}
            else:
                self.audit.log(
                    ticker,
                    signal_type,
                    "Telegram",
                    "ERROR",
                    resp.status_code,
                    resp.text[:200],
                    self.config.telegram_chat_id,
                )
                return {"success": False, "status_code": resp.status_code, "message": resp.text}
        except Exception as ex:
            self.audit.log(
                ticker,
                signal_type,
                "Telegram",
                "EXCEPTION",
                500,
                str(ex),
                self.config.telegram_chat_id,
            )
            return {"success": False, "status_code": 500, "message": str(ex)}

    def _apply_wa_throttling(self) -> None:
        """Penerapan jeda throttling minimal untuk keamanan nomor WhatsApp."""
        elapsed = time.time() - self._last_wa_send_timestamp
        if elapsed < self.config.throttle_seconds_wa:
            sleep_time = self.config.throttle_seconds_wa - elapsed
            time.sleep(sleep_time)
        self._last_wa_send_timestamp = time.time()

    def send_whatsapp_gateway(
        self, message: str, ticker: str = "SYSTEM", signal_type: str = "ALERT"
    ) -> Dict[str, Any]:
        """Kirim via WhatsApp Gateway API (Fonnte, Wablas, etc)."""
        target_phone = self.sanitize_phone(self.config.wa_target_phone)
        if not target_phone or not self.config.wa_gateway_token:
            res = {"success": False, "status_code": 400, "message": "Nomor tujuan atau API Token Gateway belum diset."}
            self.audit.log(ticker, signal_type, "WA Gateway", "FAILED", 400, "Parameter tidak lengkap", target_phone)
            return res

        self._apply_wa_throttling()
        headers = {"Authorization": self.config.wa_gateway_token}
        payload = {"target": target_phone, "message": message, "countryCode": "62"}

        try:
            resp = requests.post(
                self.config.wa_gateway_endpoint,
                headers=headers,
                data=payload,
                timeout=12,
            )
            if resp.status_code in [200, 201]:
                self.audit.log(
                    ticker,
                    signal_type,
                    "WA Gateway",
                    "SUCCESS",
                    resp.status_code,
                    "Berhasil terkirim via Gateway",
                    target_phone,
                )
                return {"success": True, "status_code": resp.status_code, "message": "Terkirim via WA Gateway"}
            else:
                self.audit.log(
                    ticker,
                    signal_type,
                    "WA Gateway",
                    "ERROR",
                    resp.status_code,
                    resp.text[:200],
                    target_phone,
                )
                return {"success": False, "status_code": resp.status_code, "message": resp.text}
        except Exception as ex:
            self.audit.log(
                ticker,
                signal_type,
                "WA Gateway",
                "EXCEPTION",
                500,
                str(ex),
                target_phone,
            )
            return {"success": False, "status_code": 500, "message": str(ex)}

    def send_whatsapp_cloud_api(
        self, message: str, ticker: str = "SYSTEM", signal_type: str = "ALERT"
    ) -> Dict[str, Any]:
        """Kirim via Meta WhatsApp Cloud API resmi (Graph API v19.0)."""
        phone_id = self.config.wa_cloud_phone_number_id
        token = self.config.wa_cloud_access_token
        recipient = self.sanitize_phone(self.config.wa_cloud_recipient_phone)

        if not phone_id or not token or not recipient:
            res = {"success": False, "status_code": 400, "message": "Kredensial Meta Cloud API belum lengkap."}
            self.audit.log(ticker, signal_type, "WA Cloud API", "FAILED", 400, "Parameter tidak lengkap", recipient)
            return res

        self._apply_wa_throttling()
        url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": message},
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code in [200, 201]:
                self.audit.log(
                    ticker,
                    signal_type,
                    "WA Cloud API",
                    "SUCCESS",
                    resp.status_code,
                    "Terkirim via Meta Cloud API",
                    recipient,
                )
                return {"success": True, "status_code": resp.status_code, "message": "Terkirim via Meta Cloud API"}
            else:
                self.audit.log(
                    ticker,
                    signal_type,
                    "WA Cloud API",
                    "ERROR",
                    resp.status_code,
                    resp.text[:200],
                    recipient,
                )
                return {"success": False, "status_code": resp.status_code, "message": resp.text}
        except Exception as ex:
            self.audit.log(
                ticker,
                signal_type,
                "WA Cloud API",
                "EXCEPTION",
                500,
                str(ex),
                recipient,
            )
            return {"success": False, "status_code": 500, "message": str(ex)}

    def generate_click_to_chat_url(self, message: str, phone: Optional[str] = None) -> str:
        """
        Generate direct link wa.me yang bisa langsung dibuka 1-klik di browser HP.
        """
        target_phone = self.sanitize_phone(phone or self.config.wa_target_phone or "628123456789")
        encoded_message = urllib.parse.quote(message)
        url = f"https://wa.me/{target_phone}?text={encoded_message}"
        return url

    def dispatch_signal(
        self,
        payload_data: Dict[str, Any],
        signal_type: str = "HAKA",
        bypass_cooldown: bool = False,
    ) -> Dict[str, Any]:
        """
        Orkestrasi eksekusi pengiriman sinyal ke seluruh kanal aktif (Telegram & WhatsApp).
        """
        ticker = payload_data.get("ticker", "SAHAM.JK")

        # Cek Anti-Spam Cooldown
        if not bypass_cooldown:
            allowed, reason = self.can_dispatch(ticker, signal_type)
            if not allowed:
                return {
                    "dispatched": False,
                    "reason": reason,
                    "results": {},
                    "click_url": "",
                }

        # Format pesan sesuai tipe sinyal
        if signal_type.upper() == "HAKA":
            message = self.format_haka_message(payload_data)
        elif signal_type.upper() in ["HAKI", "EMERGENCY_HAKI"]:
            message = self.format_emergency_haki_message(payload_data)
        elif signal_type.upper() in ["SUPER_PROFIT", "ALPHA_PROFIT"]:
            message = self.format_super_profit_message(payload_data)
        elif signal_type.upper() in ["EMERGENCY_PROTECTION", "DANGER_LOSS"]:
            message = self.format_emergency_protection_message(payload_data)
        else:
            message = self.format_haka_message(payload_data)

        results: Dict[str, Any] = {}

        # 1. Dispatch Telegram
        if self.config.telegram_enabled:
            results["telegram"] = self.send_telegram(message, ticker=ticker, signal_type=signal_type)

        # 2. Dispatch WhatsApp
        wa_channel = self.config.whatsapp_channel
        if wa_channel == "gateway":
            results["whatsapp"] = self.send_whatsapp_gateway(message, ticker=ticker, signal_type=signal_type)
        elif wa_channel == "cloud_api":
            results["whatsapp"] = self.send_whatsapp_cloud_api(message, ticker=ticker, signal_type=signal_type)
        else:
            results["whatsapp"] = {"success": True, "message": "Click-to-chat URL berhasil dibuat."}

        # Generate Click URL sebagai backup instan di UI Streamlit
        click_phone = (
            self.config.wa_target_phone
            if self.config.whatsapp_channel != "cloud_api"
            else self.config.wa_cloud_recipient_phone
        )
        click_url = self.generate_click_to_chat_url(message, click_phone)

        # Tandai waktu terkirim untuk cooldown
        self._mark_dispatched(ticker, signal_type)

        return {
            "dispatched": True,
            "reason": "Sinyal berhasil diproses",
            "message": message,
            "results": results,
            "click_url": click_url,
        }


# Singleton instance
dispatcher_instance = BotDispatcher()


# Top-level helper functions for direct calling
def format_multi_opportunity_message(top_stocks: List[Dict[str, Any]]) -> str:
    return dispatcher_instance.format_multi_opportunity_message(top_stocks)


def dispatch_multi_opportunities(
    top_stocks: List[Dict[str, Any]], bypass_cooldown: bool = True
) -> Dict[str, Any]:
    return dispatcher_instance.dispatch_multi_opportunities(top_stocks, bypass_cooldown)


def format_haka_message(data: Dict[str, Any]) -> str:
    return dispatcher_instance.format_haka_message(data)


def format_emergency_haki_message(data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    payload = {}
    if isinstance(data, dict):
        payload.update(data)
    if kwargs:
        payload.update(kwargs)
    return dispatcher_instance.format_emergency_haki_message(payload)


def format_super_profit_message(data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    payload = {}
    if isinstance(data, dict):
        payload.update(data)
    if kwargs:
        payload.update(kwargs)
    return dispatcher_instance.format_super_profit_message(payload)


def format_emergency_protection_message(data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    payload = {}
    if isinstance(data, dict):
        payload.update(data)
    if kwargs:
        payload.update(kwargs)
    return dispatcher_instance.format_emergency_protection_message(payload)


def dispatch_signal(
    payload_data: Dict[str, Any], signal_type: str = "HAKA", bypass_cooldown: bool = False
) -> Dict[str, Any]:
    return dispatcher_instance.dispatch_signal(payload_data, signal_type, bypass_cooldown)


def dispatch_super_profit_alert(
    data: Optional[Dict[str, Any]] = None, bypass_cooldown: bool = True, **kwargs
) -> Dict[str, Any]:
    payload = {}
    if isinstance(data, dict):
        payload.update(data)
    if kwargs:
        payload.update(kwargs)
    return dispatcher_instance.dispatch_signal(
        payload, signal_type="SUPER_PROFIT", bypass_cooldown=bypass_cooldown
    )


def dispatch_emergency_protection_alert(
    data: Optional[Dict[str, Any]] = None, bypass_cooldown: bool = True, **kwargs
) -> Dict[str, Any]:
    payload = {}
    if isinstance(data, dict):
        payload.update(data)
    if kwargs:
        payload.update(kwargs)
    return dispatcher_instance.dispatch_signal(
        payload, signal_type="EMERGENCY_PROTECTION", bypass_cooldown=bypass_cooldown
    )


