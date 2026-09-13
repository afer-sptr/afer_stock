"""
quant_styles_types.py
=====================
Engine Kuantitatif untuk Klasifikasi Gaya Trading (Trading Styles),
19 Tipologi Saham (Types of Stocks), dan Perusahaan Monopoli Transaksi Bursa.
Mencakup:
  1. 10 Gaya Trading (Scalping, Day, Swing, Position, Trend, Breakout, Range, News, Algo, Portfolio)
     beserta rekomendasi Timeframe Optimal & Durasi Hold untuk profit maksimal.
  2. 19 Tipologi Saham (Blue-Chip, Growth, Value, Dividend, Tech, Penny, dll)
     lengkap dengan Best For, Pros, Cons, dan Prediksi Harga Kuantitatif.
  3. Registry Khusus "Exchange & Broker Cash Cows" (Emiten pencetak laba dari tiap transaksi bursa).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# ==============================================================================
# 1. DEFINISI & METRIK 10 GAYA TRADING (TRADING STYLES)
# ==============================================================================
TRADING_STYLES_INFO = {
    "Scalping": {
        "description": "Perdagangan ultra-cepat hit-and-run memanfaatkan spread mikro dan lonjakan antrean bid/offer.",
        "timeframe": "1 Menit - 5 Menit",
        "holding_duration": "30 Detik s.d. 15 Menit (Intraday Sangat Singkat)",
        "optimal_exit": "Makan offer fraksi terdekat lalu pasang antrean jual di fraksi atas.",
        "bid_offer_rule": "Hanya masuk jika antrean Bid > 65% dan spread maksimal 1-2 fraksi harga. Wajib disiplin batas Safe Exit Lot.",
        "min_turnover_idr": 10_000_000_000,  # Min 10M per hari untuk likuiditas instan
    },
    "Day Trading": {
        "description": "Membuka dan menutup posisi dalam 1 hari bursa (zero overnight risk) mengikuti momentum pembukaan s.d. sesi 2.",
        "timeframe": "5 Menit - 15 Menit",
        "holding_duration": "30 Menit s.d. 4 Jam (Tutup Posisi Sebelum 15:50 WIB)",
        "optimal_exit": "TP pada resisten harian sesi 1 atau trailing stop 1% saat penutupan sesi 2.",
        "bid_offer_rule": "Amati OBI (Order Book Imbalance). Masuk saat antrean Offer terkikis cepat pada fraksi kunci.",
        "min_turnover_idr": 5_000_000_000,
    },
    "Swing Trading": {
        "description": "Menangkap ayunan gelombang harga dari support struktural menuju resisten utama dalam beberapa hari.",
        "timeframe": "1 Jam - Daily (Harian)",
        "holding_duration": "2 Hari s.d. 10 Hari Bursa",
        "optimal_exit": "TP1 pada +3% s.d. +7% Net, TP2 trailing stop di EMA 20.",
        "bid_offer_rule": "Beli bertahap (akumulasi pasif) di area Bid support, hindari mengejar harga saat overbought.",
        "min_turnover_idr": 2_000_000_000,
    },
    "Breakout Trading": {
        "description": "Masuk tepat saat harga menembus level resisten multi-minggu atau pola squeeze didukung lonjakan volume masif.",
        "timeframe": "15 Menit - Daily",
        "holding_duration": "1 Hari s.d. 5 Hari (Momentum Post-Breakout)",
        "optimal_exit": "Take Profit saat ekspansi Bollinger Bands melebar ekstrem atau muncul jarum suntik atas (upper shadow).",
        "bid_offer_rule": "HAKA agresif pada fraksi breakout saat volume transaksi melesat > 2x rata-rata 20 hari.",
        "min_turnover_idr": 3_000_000_000,
    },
    "Trend Trading": {
        "description": "Mengikuti arah tren dominan searah susunan MA Ribbon (EMA 10 > EMA 20 > SMA 50 > SMA 200).",
        "timeframe": "Daily - 4 Jam",
        "holding_duration": "2 Minggu s.d. 3 Bulan (Selama tren bertahan di atas EMA 20)",
        "optimal_exit": "Exit saat lilin ditutup menembus ke bawah SMA 50 atau terjadi Death Cross.",
        "bid_offer_rule": "Buy on Weakness saat harga pullback menguji EMA 10/20 dengan volume menyusut.",
        "min_turnover_idr": 1_000_000_000,
    },
    "Range Trading": {
        "description": "Strategi osilator jual di batas atas (resisten) dan beli di batas bawah (support) saat pasar bergerak menyamping (sideways).",
        "timeframe": "30 Menit - Daily",
        "holding_duration": "3 Hari s.d. 2 Minggu (Di dalam channel horizontal)",
        "optimal_exit": "TP tepat sebelum batas resisten horizontal, SL ketat jika support jebol.",
        "bid_offer_rule": "Pasang bid pasif berlapis di zona support bawah. Jangan HAKA di tengah area range.",
        "min_turnover_idr": 1_000_000_000,
    },
    "Position Trading": {
        "description": "Investasi jangka menengah-panjang berdasarkan valuasi fundamental murah, katalis ekspansi, dan siklus makro.",
        "timeframe": "Daily - Weekly (Mingguan)",
        "holding_duration": "1 Bulan s.d. 6 Bulan",
        "optimal_exit": "Realisasi profit saat harga mencapai Fair Value intrinsik Benjamin Graham atau target valuasi konsensus.",
        "bid_offer_rule": "Cicil beli teratur tanpa terpengaruh fluktuasi fraksi harian.",
        "min_turnover_idr": 500_000_000,
    },
    "News Trading": {
        "description": "Memanfaatkan katalis informasi mendadak (laporan keuangan melesat, dividen jumbo, merger/akuisisi, tender offer).",
        "timeframe": "1 Menit - 1 Jam",
        "holding_duration": "15 Menit s.d. 2 Hari (Menangkap reaksi euforia pasar)",
        "optimal_exit": "Sell on News saat sentimen positif telah di-priced-in oleh kerumunan retail.",
        "bid_offer_rule": "Eksekusi kilat (HAKA) pada detik pertama rilis keterbukaan informasi bursa.",
        "min_turnover_idr": 2_000_000_000,
    },
    "Algorithmic Trading": {
        "description": "Eksekusi kuantitatif berbasis aturan sistematis matematis, sinyal ML Meta-Labeling, dan arbitrase mikrostruktur.",
        "timeframe": "1 Menit - 15 Menit",
        "holding_duration": "Bervariasi (Sesuai sinyal model machine learning)",
        "optimal_exit": "Otomatis tertutup saat Brier score atau probabilitas menang model turun di bawah ambang batas.",
        "bid_offer_rule": "Smart Order Routing membagi order besar ke dalam lot-lot kecil (TWAP/VWAP) untuk mencegah market impact.",
        "min_turnover_idr": 5_000_000_000,
    },
    "Portfolio Trading": {
        "description": "Alokasi multi-aset berbasis optimasi matematika Mean-CVaR (Rockafellar-Uryasev) dan diversifikasi risiko.",
        "timeframe": "Weekly - Monthly",
        "holding_duration": "3 Bulan s.d. 12 Bulan (Rebalancing Periodik)",
        "optimal_exit": "Rebalancing kuartalan saat bobot portofolio bergeser dari alokasi optimum CVXPY.",
        "bid_offer_rule": "Eksekusi alokasi bobot proporsional di seluruh emiten terpilih.",
        "min_turnover_idr": 1_000_000_000,
    },
}


# ==============================================================================
# 2. DEFINISI 19 TIPOLOGI SAHAM (TYPES OF STOCKS)
# ==============================================================================
STOCK_TYPES_PROFILES = {
    "Blue-Chip Stocks": {
        "best_for": "Investor jangka panjang, dana pensiun, dan pencari stabilitas modal dengan dividen konsisten.",
        "pros": "Likuiditas raksasa, tata kelola GCG unggul, neraca keuangan sehat, tahan terhadap krisis ekonomi.",
        "cons": "Pertumbuhan persentase harga cenderung lebih lambat dibanding saham lapis bawah (low beta).",
    },
    "Growth Stocks": {
        "best_for": "Trader dan investor agresif yang mengejar pertumbuhan apresiasi modal (capital gain) eksponensial.",
        "pros": "Laba bersih dan pendapatan tumbuh di atas rata-rata industri, ekspansi bisnis agresif.",
        "cons": "Valuasi PER/PBV sering kali mahal, volatilitas tinggi saat koreksi pasar, dividen kecil atau nihil.",
    },
    "Value Stocks": {
        "best_for": "Pengikut prinsip Value Investing Benjamin Graham & Warren Buffett yang sabar.",
        "pros": "Harga terdiskon jauh di bawah nilai buku (PBV < 1) atau aset riilnya, Margin of Safety tebal.",
        "cons": "Risiko Value Trap (harga bisa murah dalam waktu lama jika tidak ada katalis pembuka nilai).",
    },
    "Dividend Stocks": {
        "best_for": "Pencari arus kas pasif (passive income) dan dividen yield tinggi (> 6-10% per tahun).",
        "pros": "Pembagian dividen tunai rutin setiap tahun, arus kas operasional sangat kuat dan stabil.",
        "cons": "Potensi dividend trap saat harga rontok pasca-cum date (ex-date drop), pertumbuhan ekspansi terbatas.",
    },
    "Quality Stocks": {
        "best_for": "Investor konservatif yang mengutamakan Return on Equity (ROE > 18%) dan keunggulan parit ekonomi (Moat).",
        "pros": "Tingkat profitabilitas superior, brand power dominan, utang rendah, penetapan harga (pricing power) kuat.",
        "cons": "Jarang sekali diperdagangkan dengan harga sangat murah di pasar reguler.",
    },
    "Mega-Cap Stocks": {
        "best_for": "Pengendali indeks bursa, reksadana institusi, dan investor asing skala global.",
        "pros": "Kapitalisasi pasar > Rp 100 Triliun, menjadi penggerak utama IHSG, likuiditas tanpa risiko slippage.",
        "cons": "Pergerakan harga sangat dipengaruhi oleh aliran dana asing (foreign flow) dan dinamika makro global.",
    },
    "Large-Cap Stocks": {
        "best_for": "Portofolio inti (core portfolio) untuk pertumbuhan seimbang dan risiko terukur.",
        "pros": "Kapitalisasi pasar Rp 50T - 100T, stabilitas tinggi dan terdaftar di indeks LQ45 / IDX30.",
        "cons": "Kurang lincah untuk trading kilat scalping berorientasi ratusan persen per pekan.",
    },
    "Mid-Cap Stocks": {
        "best_for": "Swing trader dan investor pencari 'Sweet Spot' antara stabilitas dan potensi lonjakan laba.",
        "pros": "Kapitalisasi pasar Rp 5T - 50T, potensi melesat menjadi large-cap, fleksibilitas bisnis tinggi.",
        "cons": "Volatilitas lebih tinggi daripada blue-chip saat terjadi pengetatan likuiditas bursa.",
    },
    "Small-Cap Stocks": {
        "best_for": "Trader momentum agresif dan pencari peluang multi-bagger dari saham berkapitalisasi kecil.",
        "pros": "Kapitalisasi pasar < Rp 5T, harga mudah melesat kencang saat ada katalis volume dan akumulasi bandar.",
        "cons": "Likuiditas antrean tipis, rentan terkena bantingan harga (slippage) jika melebihi safe exit lot.",
    },
    "Micro-Cap Stocks": {
        "best_for": "Spekulator tingkat tinggi yang siap dengan rasio untung-rugi asimetris tinggi.",
        "pros": "Kapitalisasi sangat kecil (< Rp 1 Triliun), potensi kenaikan persentase puluhan persen dalam hitungan hari.",
        "cons": "Risiko delisting, tata kelola minim, volatilitas liar, sangat mudah digerakkan oleh satu entitas.",
    },
    "Tech Stocks": {
        "best_for": "Investor masa depan yang percaya pada disrupsi digital, kecerdasan buatan, dan platform internet.",
        "pros": "Skalabilitas bisnis tidak terbatas, margin laba kotor tinggi saat fase mature tercapai.",
        "cons": "Sensitif terhadap kenaikan suku bunga bank sentral, sebagian besar masih dalam fase pembakaran modal.",
    },
    "Healthcare Stocks": {
        "best_for": "Investor defensif yang ingin tetap cuan di segala siklus ekonomi (tahan resesi).",
        "pros": "Permintaan layanan kesehatan, rumah sakit, dan farmasi bersifat inelastis (selalu dibutuhkan).",
        "cons": "Ketergantungan pada regulasi BPJS, harga bahan baku obat impor yang rentan terhadap kurs USD/IDR.",
    },
    "Consumer Staples": {
        "best_for": "Investor defensif pencari perlindungan modal dari inflasi dan siklus penurunan ekonomi.",
        "pros": "Produk makanan, minuman, dan kebutuhan sehari-hari dibeli konsumen setiap hari tanpa henti.",
        "cons": "Margin laba tertekan saat harga komoditas bahan baku (gandum, CPO, gula) melonjak tinggi.",
    },
    "Cyclical Stocks": {
        "best_for": "Trader siklus komoditas (Batubara, Nikel, Minyak, Tembaga, Properti) yang pandai memanfaatkan momentum timing.",
        "pros": "Laba bersih melonjak fantastis saat harga komoditas global berada di fase puncak supercycle.",
        "cons": "Laba bisa anjlok drastis dan harga saham jatuh dalam saat siklus komoditas global berbalik arah turun.",
    },
    "International Stocks": {
        "best_for": "Pencari lindung nilai terhadap pelemahan nilai tukar Rupiah (Dollar earning exporters).",
        "pros": "Pendapatan utama dalam mata uang Dolar AS (USD) sehingga diuntungkan saat Rupiah melemah.",
        "cons": "Tergantung pada permintaan ekonomi negara tujuan ekspor (Tiongkok, AS, Uni Eropa, India).",
    },
    "Emerging-Market Stocks": {
        "best_for": "Pengincar pertumbuhan ekonomi pasar berkembang Indonesia (demografi muda & konsumsi domestik kuat).",
        "pros": "Pertumbuhan PDB Indonesia di atas rata-rata negara maju (> 5% YoY), potensi masuknya arus dana asing.",
        "cons": "Rentan terhadap sentimen 'Risk-Off' global saat indeks DXY Dolar AS menguat tajam.",
    },
    "REIT Stocks": {
        "best_for": "Pencari imbal hasil aset properti riil tanpa harus repot mengelola gedung secara fisik.",
        "pros": "Mewajibkan distribusi laba sewa properti minimal 90% dalam bentuk dividen berkala kepada pemegang unit.",
        "cons": "Tingkat likuiditas perdagangan di bursa domestik masih relatif terbatas dibanding saham biasa.",
    },
    "Preferred Stocks": {
        "best_for": "Investor yang menginginkan prioritas pembagian dividen di atas pemegang saham biasa.",
        "pros": "Memiliki hak klaim aset dan dividen tetap lebih dulu dibanding pemegang saham umum.",
        "cons": "Biasanya tidak memiliki hak suara dalam RUPS dan apresiasi harga saham tidak setinggi saham biasa.",
    },
    "Penny Stocks": {
        "best_for": "Scalper kilat berpengalaman yang hanya bertransaksi pada fraksi harga di bawah Rp 100 (saham gocap/akselerasi).",
        "pros": "Kenaikan 1-2 tik saja sudah menghasilkan persentase cuan bersih yang signifikan (+2% s.d. +5%).",
        "cons": "Risiko likuiditas terkunci di batas bawah (Rp 50 atau fraksi bawah papan akselerasi/pemantauan khusus).",
    },
}


# ==============================================================================
# 3. REGISTRY: EMITEN PENERIMA MANFAAT SETIAP TRANSAKSI BURSA (CASH COWS)
# ==============================================================================
TRADING_CASH_COWS_REGISTRY = [
    {
        "ticker": "BBCA.JK",
        "name": "Bank Central Asia Tbk.",
        "role": "Bank Kustodian & Penyedia Rekening Dana Nasabah (RDN) Terbesar di Indonesia",
        "revenue_source": "Setiap investor mentransfer dana trading dan mendiamkan dana di RDN, BCA menikmati likuiditas dana murah (CASA) serta fee transfer kliring perputaran transaksi.",
        "type_of_stock": "Mega-Cap Stocks",
        "trading_style": "Position Trading",
        "moat_rating": "🌟🌟🌟🌟🌟 (Super Monopoly)",
    },
    {
        "ticker": "BBRI.JK",
        "name": "Bank Rakyat Indonesia Tbk.",
        "role": "Bank Kustodian RDN, Induk Usaha BRI Danareksa Sekuritas & Kliring KPEI",
        "revenue_source": "Mendulang fee perantara pedagang efek (brokerage fee) melalui Danareksa Sekuritas dan pengendapan margin saldo RDN jutaan investor ritel.",
        "type_of_stock": "Mega-Cap Stocks",
        "trading_style": "Dividend Stocks",
        "moat_rating": "🌟🌟🌟🌟🌟",
    },
    {
        "ticker": "BMRI.JK",
        "name": "Bank Mandiri Tbk.",
        "role": "Bank Pembayaran Kliring KPEI & Induk Mandiri Sekuritas (Top Broker BEI)",
        "revenue_source": "Mandiri Sekuritas memungut komisi fee transaksi beli/jual setiap hari, sementara Bank Mandiri mengelola kliring penyelesaian dana transaksi bursa.",
        "type_of_stock": "Mega-Cap Stocks",
        "trading_style": "Trend Trading",
        "moat_rating": "🌟🌟🌟🌟🌟",
    },
    {
        "ticker": "BBNI.JK",
        "name": "Bank Negara Indonesia Tbk.",
        "role": "Bank Pembayaran Kliring KPEI & Induk BNI Sekuritas",
        "revenue_source": "Menerima fee brokerage harian dari transaksi jutaan lot saham dan obligasi di platform BNI Sekuritas (BIONS).",
        "type_of_stock": "Large-Cap Stocks",
        "trading_style": "Swing Trading",
        "moat_rating": "🌟🌟🌟🌟",
    },
    {
        "ticker": "PANS.JK",
        "name": "Panin Sekuritas Tbk.",
        "role": "Perusahaan Efek & Sekuritas Publik Mandiri Murni (Direct Broker Play)",
        "revenue_source": "Murni mencetak laba langsung dari komisi perantara perdagangan efek saham setiap kali investor melakukan transaksi beli/jual.",
        "type_of_stock": "Dividend Stocks",
        "trading_style": "Value Stocks",
        "moat_rating": "🌟🌟🌟🌟",
    },
    {
        "ticker": "TRIM.JK",
        "name": "Trimegah Sekuritas Indonesia Tbk.",
        "role": "Sekuritas Publik & Agen Penjual Efek Transaksi Aktif",
        "revenue_source": "Meraup brokerage fee, margin financing interest fee, serta underwriting fee penerbitan obligasi dan IPO saham baru.",
        "type_of_stock": "Mid-Cap Stocks",
        "trading_style": "Breakout Trading",
        "moat_rating": "🌟🌟🌟",
    },
    {
        "ticker": "YULE.JK",
        "name": "Yulie Sekuritas Indonesia Tbk.",
        "role": "Perantara Perdagangan Efek & Manajemen Investasi",
        "revenue_source": "Pendapatan komisi transaksi efek bursa dan pengelolaan portofolio nasabah.",
        "type_of_stock": "Small-Cap Stocks",
        "trading_style": "Range Trading",
        "moat_rating": "🌟🌟",
    },
    {
        "ticker": "GOTO.JK",
        "name": "GoTo Gojek Tokopedia Tbk.",
        "role": "Ekosistem Fintech Super App (Integrasi Investasi Bibit & GoPay RDN)",
        "revenue_source": "Menghasilkan fee transaksi dari jutaan pengguna mikro-retail yang bertransaksi reksadana, saham, dan pembayaran melalui dompet digital GoPay.",
        "type_of_stock": "Tech Stocks",
        "trading_style": "Scalping",
        "moat_rating": "🌟🌟🌟🌟",
    },
    {
        "ticker": "ARTO.JK",
        "name": "Bank Jago Tbk.",
        "role": "Bank Digital Partner Utama Platform RDN Stockbit & Bibit",
        "revenue_source": "Setiap pengguna baru membuka akun saham di Stockbit/Bibit, rekening RDN otomatis dibuat di Bank Jago, menyuntikkan dana murah dan fee transaksi ekosistem.",
        "type_of_stock": "Tech Stocks",
        "trading_style": "Swing Trading",
        "moat_rating": "🌟🌟🌟🌟",
    },
    {
        "ticker": "EMTK.JK",
        "name": "Elang Mahkota Teknologi Tbk.",
        "role": "Pemegang Saham Strategis Ekosistem Bareksa & DANA",
        "revenue_source": "Mendapat manfaat dari aliran modal transaksi retail di marketplace finansial Bareksa dan saluran pembayaran digital.",
        "type_of_stock": "Mid-Cap Stocks",
        "trading_style": "Position Trading",
        "moat_rating": "🌟🌟🌟",
    },
]


# ==============================================================================
# 4. ENGINE EVALUASI DINAMIS GAYA TRADING & TIPE SAHAM PER EMITEN
# ==============================================================================
def evaluate_trading_style_and_type(
    ticker: str,
    price: float,
    daily_turnover: float,
    pct_bid: float,
    pct_offer: float,
    atr_val: float,
    adx_val: float,
    is_sharia: bool = True,
    market_cap: Optional[float] = None,
    sector: str = "Umum",
) -> Dict[str, Any]:
    """
    Menentukan Trading Style optimal, Timeframe, Durasi Hold, dan Tipologi Saham
    berdasarkan matriks mikrostruktur, volatilitas, turnover, dan likuiditas.
    """
    # Sanitasi input dari kemungkinan NaN / None / Inf
    try:
        price = float(price) if price is not None and not np.isnan(price) and not np.isinf(price) else 1000.0
    except Exception:
        price = 1000.0
    if price <= 0:
        price = 1000.0

    try:
        daily_turnover = float(daily_turnover) if daily_turnover is not None and not np.isnan(daily_turnover) else 5_000_000_000.0
    except Exception:
        daily_turnover = 5_000_000_000.0

    pct_bid = float(pct_bid) if pct_bid is not None and not np.isnan(pct_bid) else 50.0
    pct_offer = float(pct_offer) if pct_offer is not None and not np.isnan(pct_offer) else 50.0
    atr_val = float(atr_val) if atr_val is not None and not np.isnan(atr_val) else 50.0
    adx_val = float(adx_val) if adx_val is not None and not np.isnan(adx_val) else 20.0

    clean_t = ticker.replace(".JK", "").upper().strip()
    atr_pct = (atr_val / max(price, 1.0)) * 100.0 if price > 0 else 2.0

    # 1. Tentukan Trading Style Utama & Alternatif
    if price < 150 and daily_turnover > 3_000_000_000:
        primary_style = "Scalping"
        secondary_style = "Breakout Trading"
    elif atr_pct > 3.5 and daily_turnover > 8_000_000_000:
        primary_style = "Day Trading"
        secondary_style = "Scalping"
    elif adx_val > 25 and atr_pct > 2.0:
        primary_style = "Trend Trading"
        secondary_style = "Swing Trading"
    elif adx_val < 20 and daily_turnover > 1_000_000_000:
        primary_style = "Range Trading"
        secondary_style = "Swing Trading"
    elif daily_turnover > 15_000_000_000 and price > 2000:
        primary_style = "Portfolio Trading"
        secondary_style = "Position Trading"
    elif daily_turnover > 5_000_000_000 and atr_pct > 2.5:
        primary_style = "Swing Trading"
        secondary_style = "Breakout Trading"
    else:
        primary_style = "Swing Trading"
        secondary_style = "Position Trading"

    style_meta = TRADING_STYLES_INFO.get(primary_style, TRADING_STYLES_INFO["Swing Trading"])

    # 2. Tentukan Tipe Saham (Type of Stock)
    # Aturan estimasi kapitalisasi pasar jika tidak diberikan
    est_mcap = market_cap if market_cap else (price * 10_000_000_000 if price > 2000 else price * 3_000_000_000)

    stock_type_list = []
    if price < 100:
        stock_type_list.append("Penny Stocks")
        stock_type_list.append("Micro-Cap Stocks")
    elif est_mcap > 100_000_000_000_000 or clean_t in {"BBCA", "BBRI", "BMRI", "TLKM", "ASII", "BREN", "TPIA", "AMMN"}:
        stock_type_list.append("Mega-Cap Stocks")
        stock_type_list.append("Blue-Chip Stocks")
    elif est_mcap > 50_000_000_000_000 or clean_t in {"BBNI", "ICBP", "INDF", "UNVR", "KLBF", "AMRT", "PGAS", "ADRO", "BRIS"}:
        stock_type_list.append("Large-Cap Stocks")
        stock_type_list.append("Blue-Chip Stocks")
    elif est_mcap > 5_000_000_000_000:
        stock_type_list.append("Mid-Cap Stocks")
    else:
        stock_type_list.append("Small-Cap Stocks")

    # Sektor spesifik
    sec_lower = sector.lower()
    if "tech" in sec_lower or clean_t in {"GOTO", "BUKA", "EMTK", "WIFI", "DMMX", "BELI"}:
        stock_type_list.append("Tech Stocks")
    elif "health" in sec_lower or clean_t in {"KLBF", "MIKA", "HEAL", "SILO", "SIDO", "PRDA"}:
        stock_type_list.append("Healthcare Stocks")
    elif "consumer" in sec_lower or clean_t in {"ICBP", "INDF", "UNVR", "MYOR", "AMRT", "CMRY"}:
        stock_type_list.append("Consumer Staples")
    elif "energy" in sec_lower or "basic" in sec_lower or clean_t in {"ADRO", "PTBA", "MEDC", "INCO", "ANTM", "MDKA"}:
        stock_type_list.append("Cyclical Stocks")

    # Nilai / Dividen
    if clean_t in {"BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP"}:
        stock_type_list.append("Quality Stocks")
    if clean_t in {"PTBA", "ITMG", "ADRO", "HEXA", "BJTM", "BJBR", "ASII", "UNTR", "MBAP"}:
        stock_type_list.append("Dividend Stocks")
    if clean_t in {"BMTR", "MNCN", "INDY", "DOID", "ABMM", "SMSM"}:
        stock_type_list.append("Value Stocks")

    primary_stock_type = stock_type_list[0] if stock_type_list else "Mid-Cap Stocks"
    type_profile = STOCK_TYPES_PROFILES.get(primary_stock_type, STOCK_TYPES_PROFILES["Mid-Cap Stocks"])

    # 3. Prediksi Harga Kuantitatif Sederhana Berbasis Drift Volatilitas & Sentimen
    # Drift momentum mingguan diestimasi dari volatilitas dan bias order book
    order_bias = float(np.clip((pct_bid - pct_offer) / 100.0, -1.0, 1.0))
    drift_pct = float((order_bias * 1.5) + (1.0 if primary_style in ["Trend Trading", "Breakout Trading"] else 0.5))

    val_1w = price * (1.0 + (drift_pct / 100.0))
    val_1m = price * (1.0 + (drift_pct * 2.8 / 100.0))

    pred_1w_price = int(round(val_1w)) if not np.isnan(val_1w) and not np.isinf(val_1w) else int(price)
    pred_1m_price = int(round(val_1m)) if not np.isnan(val_1m) and not np.isinf(val_1m) else int(price)

    return {
        "ticker": ticker,
        "clean_ticker": clean_t,
        "price": price,
        "primary_style": primary_style,
        "secondary_style": secondary_style,
        "timeframe": style_meta["timeframe"],
        "holding_duration": style_meta["holding_duration"],
        "optimal_exit": style_meta["optimal_exit"],
        "bid_offer_rule": style_meta["bid_offer_rule"],
        "style_description": style_meta["description"],
        "primary_stock_type": primary_stock_type,
        "all_stock_types": stock_type_list,
        "best_for": type_profile["best_for"],
        "pros": type_profile["pros"],
        "cons": type_profile["cons"],
        "pred_1w_price": pred_1w_price,
        "pred_1m_price": pred_1m_price,
        "pred_change_pct": drift_pct * 2.8,
    }


def get_trading_cash_cows() -> List[Dict[str, Any]]:
    """Mengembalikan daftar perusahaan penerima manfaat transaksi bursa."""
    return TRADING_CASH_COWS_REGISTRY
