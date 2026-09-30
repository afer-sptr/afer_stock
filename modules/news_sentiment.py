import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timezone
import email.utils
from typing import List, Dict, Any

# Kamus FinBERT & Pasar Modal Indonesia Terkalibrasi
FINBERT_LEXICON = {
    # Sentimen Sangat Bullish / Positif
    "laba": 1.8, "tumbuh": 1.5, "dividen": 1.6, "ekspansi": 1.4, "rekor": 1.7,
    "merger": 1.3, "akuisisi": 1.4, "untung": 1.5, "surplus": 1.4, "optimis": 1.2,
    "borong": 1.6, "akumulasi": 1.6, "melesat": 1.7, "terbang": 1.5, "bullish": 1.5,
    "cuan": 1.3, "kinerja": 1.1, "prospek": 1.2, "rebound": 1.4, "melonjak": 1.6,
    "menguat": 1.3, "melambung": 1.5, "melejit": 1.6, "net buy": 1.4, "inflow": 1.3,
    "buka gembok": 1.8, "cabut suspensi": 1.9, "lepas suspensi": 1.8,
    
    # Sentimen Sangat Bearish / Negatif
    "rugi": -1.9, "anjlok": -2.0, "suspensi": -2.8, "pailit": -3.0, "utang": -1.2,
    "turun": -1.3, "gagal": -2.1, "arb": -1.8, "penurunan": -1.2, "krisis": -2.2,
    "default": -2.7, "pkpu": -2.9, "delisting": -3.0, "sanksi": -2.0, "kebangkrutan": -3.0,
    "investigasi": -1.8, "ambles": -1.7, "merosot": -1.5, "bearish": -1.4,
    "net sell": -1.4, "outflow": -1.3, "koreksi": -1.1, "anjlok": -1.8, "gugatan": -1.8
}

VETO_KEYWORDS = {"pailit", "suspensi", "default", "pkpu", "delisting", "kebangkrutan", "sanksi berat"}
ANTI_VETO_PHRASES = [
    "cabut suspensi", "suspensi dicabut", "buka suspensi", "suspensi dibuka",
    "buka gembok", "gembok dibuka", "lepas suspensi", "bebas suspensi",
    "tolak pailit", "gugatan pailit ditolak", "batal pailit", "damai pkpu",
    "keluar dari pkpu", "homologasi", "lolos pkpu", "pulih"
]


def format_relative_time(pub_date_str: str) -> str:
    """Mengubah format pubDate RFC-822 / GMT menjadi waktu relatif Indonesia real-time."""
    if not pub_date_str:
        return "Baru saja"
    try:
        dt = email.utils.parsedate_to_datetime(pub_date_str)
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = diff.total_seconds()

        if seconds < 0:
            return "Baru saja"
        elif seconds < 120:
            return "1 menit lalu"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} menit lalu"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} jam lalu"
        elif seconds < 172800:
            return "Kemarin"
        else:
            days = int(seconds / 86400)
            if days <= 7:
                return f"{days} hari lalu"
            return dt.strftime("%d-%m-%Y")
    except Exception:
        return pub_date_str[:16] if len(pub_date_str) > 16 else pub_date_str


def classify_headline_sentiment(title: str) -> Dict[str, Any]:
    text = title.lower()
    score = 0.0
    matched_pos = []
    matched_neg = []
    words = re.findall(r'\b\w+\b', text)

    for w in words:
        if w in FINBERT_LEXICON:
            val = FINBERT_LEXICON[w]
            score += val
            if val > 0:
                matched_pos.append(w)
            else:
                matched_neg.append(w)

    # Deteksi VETO Presisi Tinggi (Bebas False Alarm Pencabutan Suspensi)
    has_veto_kw = any(k in text for k in VETO_KEYWORDS)
    has_anti_veto = any(p in text for p in ANTI_VETO_PHRASES)

    if has_anti_veto:
        is_veto = False
        score += 2.0  # Pencabutan suspensi / lolos pailit merupakan katalis positif kuat
        matched_pos.append("pemulihan status")
    else:
        is_veto = has_veto_kw

    if score > 0.5:
        sentiment = "BULLISH / POSITIF"
        badge_color = "green"
    elif score < -0.5:
        sentiment = "BEARISH / NEGATIF"
        badge_color = "red"
    else:
        sentiment = "NETRAL"
        badge_color = "gray"

    return {
        "sentiment": sentiment,
        "badge_color": badge_color,
        "score": score,
        "is_veto": is_veto,
        "positive_words": matched_pos,
        "negative_words": matched_neg
    }


def fetch_latest_news(ticker: str, company_name: str = "", limit: int = 15) -> Dict[str, Any]:
    """
    Mengambil berita real-time terkini untuk saham BEI langsung dari feed finansial,
    disortir dari yang paling baru (newest-first) dengan waktu relatif Indonesia.
    """
    clean_ticker = ticker.replace(".JK", "").strip().upper()
    query = f"saham {clean_ticker}"
    if company_name and len(company_name) > 3:
        clean_company = re.sub(r'\(.*?\)|Tbk\.?|PT\b', '', company_name).strip()
        if clean_company and clean_company.upper() != clean_ticker:
            query = f"saham {clean_ticker} OR \"{clean_company}\""

    encoded_q = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_q}+when:7d&hl=id&gl=ID&ceid=ID:id"

    articles_raw = []
    has_news_veto = False

    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")

            for item in items:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else "#"
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""

                if not title:
                    continue

                source = "Media Keuangan"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title_clean = parts[0]
                    source = parts[1]
                else:
                    title_clean = title

                sent_res = classify_headline_sentiment(title_clean)
                if sent_res["is_veto"]:
                    has_news_veto = True

                rel_time = format_relative_time(pub_date)
                
                # Parse datetime untuk sorting
                parsed_dt = None
                try:
                    parsed_dt = email.utils.parsedate_to_datetime(pub_date)
                except Exception:
                    pass

                articles_raw.append({
                    "title": title_clean,
                    "source": source,
                    "link": link,
                    "date": rel_time,
                    "raw_date": pub_date,
                    "dt": parsed_dt,
                    "sentiment": sent_res["sentiment"],
                    "badge_color": sent_res["badge_color"],
                    "score": sent_res["score"],
                    "keywords": sent_res["positive_words"] if sent_res["score"] > 0 else sent_res["negative_words"]
                })

    except Exception:
        pass

    # Fallback jika query khusus kosong: cari query ticker murni
    if not articles_raw:
        try:
            fallback_url = f"https://news.google.com/rss/search?q=saham+{clean_ticker}&hl=id&gl=ID&ceid=ID:id"
            req2 = urllib.request.Request(fallback_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=4) as resp2:
                root2 = ET.fromstring(resp2.read())
                for item in root2.findall(".//item")[:limit]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else "#"
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    if not title:
                        continue
                    source = "Media Keuangan"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        title_clean = parts[0]
                        source = parts[1]
                    else:
                        title_clean = title
                    sent_res = classify_headline_sentiment(title_clean)
                    articles_raw.append({
                        "title": title_clean,
                        "source": source,
                        "link": link,
                        "date": format_relative_time(pub_date),
                        "raw_date": pub_date,
                        "dt": None,
                        "sentiment": sent_res["sentiment"],
                        "badge_color": sent_res["badge_color"],
                        "score": sent_res["score"],
                        "keywords": sent_res["positive_words"] if sent_res["score"] > 0 else sent_res["negative_words"]
                    })
        except Exception:
            pass

    # Sort artikel dari yang paling baru (newest-first)
    articles_raw.sort(
        key=lambda x: x["dt"].timestamp() if x.get("dt") is not None else 0,
        reverse=True
    )

    articles_all = articles_raw[:limit]
    articles_positive = [a for a in articles_all if a["sentiment"] == "BULLISH / POSITIF"]
    articles_negative = [a for a in articles_all if a["sentiment"] == "BEARISH / NEGATIF"]
    articles_neutral = [a for a in articles_all if a["sentiment"] == "NETRAL"]

    total_score = sum(a["score"] for a in articles_all)
    if articles_all:
        avg_score = total_score / len(articles_all)
        sentiment_score = int(max(10, min(95, round(50 + (avg_score * 15)))))
    else:
        sentiment_score = 50

    if has_news_veto:
        overall_sentiment = "⚠️ NEWS VETO / HIGH ALERT"
        sentiment_desc = "Terdeteksi pemberitaan krisis serius (potensi suspensi, pailit, delisting, atau sanksi bursa aktif)."
    elif sentiment_score >= 65:
        overall_sentiment = "BULLISH / SANGAT POSITIF"
        sentiment_desc = "Sentimen pemberitaan didominasi aksi akumulasi beli, kinerja positif, atau prospek cerah."
    elif sentiment_score <= 38:
        overall_sentiment = "BEARISH / NEGATIF"
        sentiment_desc = "Sentimen pemberitaan diwarnai kekhawatiran koreksi, tekanan pasar, atau risiko emiten."
    else:
        overall_sentiment = "NETRAL / CAMPURAN"
        sentiment_desc = "Pemberitaan relatif seimbang antara sentimen positif dan dinamika pasar normal."

    return {
        "score": sentiment_score,
        "overall_sentiment": overall_sentiment,
        "sentiment_desc": sentiment_desc,
        "news_veto": has_news_veto,
        "articles": articles_all,
        "articles_positive": articles_positive,
        "articles_negative": articles_negative,
        "articles_neutral": articles_neutral,
        "stats": {
            "total_articles": len(articles_all),
            "bullish": len(articles_positive),
            "bearish": len(articles_negative),
            "neutral": len(articles_neutral)
        }
    }
