"""
Command Line Interface (CLI) untuk Analisis dan Prediksi Saham Indonesia (IDX) REAL-TIME.
Cakupan Komprehensif: Harga Wajar (Fair Value), Kalender Dividen & Strategi, Berita Positif vs Negatif.
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from modules.data_loader import (
    fetch_stock_data, 
    normalize_ticker, 
    ALL_IDX_STOCKS, 
    search_idx_stocks
)
from modules.technical_analysis import compute_technical_indicators, evaluate_technical_score
from modules.fundamental_analysis import evaluate_fundamental_score
from modules.high_low_analysis import evaluate_high_low_aspects
from modules.news_sentiment import fetch_latest_news
from modules.fair_value import calculate_fair_value
from modules.dividend_analyzer import analyze_dividend_schedule_and_strategy, calculate_dividend_payout_simulation
from modules.predictor import train_and_predict_future
from modules.recommendation_engine import generate_composite_recommendation, calculate_position_size

console = Console(force_terminal=True)


def display_analysis(ticker_input: str, capital: float = 10000000.0, risk_pct: float = 2.0):
    ticker = normalize_ticker(ticker_input)
    console.print(f"\n[bold cyan][*] Menghubungkan ke data feed real-time untuk emiten [yellow]{ticker}[/yellow]...[/bold cyan]")

    # 1. Unduh Data Pasar Real-time
    df, info, err = fetch_stock_data(ticker, period="2y", interval="1d")
    if err or df is None:
        console.print(f"[bold red][X] Gagal memuat data: {err}[/bold red]")
        return

    # 2. Proses Analisis Seluruh Aspek
    console.print("[dim green][*] Menghitung Harga Wajar, Kalender Dividen, Sentimen Berita, dan Sinyal Trading...[/dim green]")
    df_tech = compute_technical_indicators(df)
    fast_info = info.get("fast_info")
    current_price = float(df_tech["Close"].iloc[-1])

    tech_eval = evaluate_technical_score(df_tech)
    hl_eval = evaluate_high_low_aspects(df_tech, fast_info)
    fund_eval = evaluate_fundamental_score(info)
    fv_eval = calculate_fair_value(current_price, info, df_history=df_tech)
    div_eval = analyze_dividend_schedule_and_strategy(ticker, info, current_price)
    news_eval = fetch_latest_news(ticker, company_name=info.get("shortName", ""), limit=10)
    ml_eval = train_and_predict_future(df_tech, forecast_days=5)

    rec = generate_composite_recommendation(
        tech_result=tech_eval,
        fund_result=fund_eval,
        ml_result=ml_eval,
        current_price=current_price,
        high_low_result=hl_eval,
        news_result=news_eval
    )
    plan = rec["trading_plan"]
    pos_size = calculate_position_size(capital, risk_pct, current_price, plan["stop_loss"])

    # 3. Header Emiten Real-Time
    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector", "IDX Sector")
    prev_close = float(df_tech["Close"].iloc[-2]) if len(df_tech) > 1 else current_price
    day_change = current_price - prev_close
    day_change_pct = (day_change / prev_close) * 100 if prev_close > 0 else 0
    change_color = "green" if day_change >= 0 else "red"
    change_symbol = "[+]" if day_change >= 0 else "[-]"
    timestamp = info.get("fetched_at", "")

    mos_pct = fv_eval['margin_of_safety_pct']
    mos_sign = "+" if mos_pct >= 0 else ""

    console.print()
    console.print(Panel(
        f"[bold white]{company_name} ({ticker})[/bold white] - Sektor: [italic]{sector}[/italic]\n"
        f"Harga Real-Time: [bold {change_color}]Rp {current_price:,.0f} {change_symbol} {abs(day_change):,.0f} ({day_change_pct:+.2f}%)[/bold {change_color}] | "
        f"Update: [dim]{timestamp}[/dim]\n"
        f"💎 [bold cyan]Harga Wajar (Fair Value): Rp {fv_eval['fair_value']:,}[/bold cyan] (Margin of Safety: [bold green]{mos_sign}{mos_pct:.1f}%[/bold green] - {fv_eval['status'].split('(')[0]})\n"
        f"💰 [bold yellow]Dividend Yield: {div_eval['dividend_yield_pct']:.2f}%[/bold yellow] (DPS: Rp {div_eval['dividend_rate_idr']:,} | Ex-Date: {div_eval['ex_dividend_date']})",
        title="[bold yellow]PROFIL REAL-TIME EMITEN BEI[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED
    ))

    # 4. Kotak Keputusan Utama
    action = rec["action"]
    badge_color = rec["badge_color"]
    comp_score = rec["composite_score"]
    b = rec["breakdown_scores"]

    decision_text = (
        f"[bold {badge_color}]KEPUTUSAN: {action}[/bold {badge_color}]\n"
        f"[italic white]{rec['action_desc']}[/italic white]\n"
        f"[bold]Skor Gabungan: [{badge_color}]{comp_score} / 100[/{badge_color}][/bold]\n"
        f"Rincian: Teknikal {b['technical']} | High/Low {b['high_low']} | Fundamental {b['fundamental']} | Berita {b['news_sentiment']} | ML {b['ml_prediction']}"
    )
    console.print(Panel(decision_text, title="[bold]REKOMENDASI KEPUTUSAN TERPADU[/bold]", border_style=badge_color, box=box.HEAVY))

    # 5. Tabel Trading Plan
    plan_table = Table(title="[bold green]RENCANA LEVEL TRANSAKSI (TRADING PLAN)[/bold green]", box=box.DOUBLE_EDGE)
    plan_table.add_column("Level / Titik Kunci", style="bold cyan")
    plan_table.add_column("Rekomendasi Harga (IDR)", justify="right", style="bold white")
    plan_table.add_column("Rentang / Potensi (%)", justify="right")
    plan_table.add_column("Keterangan Strategi", style="italic")

    plan_table.add_row(
        "Zona Beli (Entry Range)",
        f"Rp {plan['buy_entry_min']:,} - Rp {plan['buy_entry_max']:,}",
        f"+/- {abs(current_price - plan['buy_entry_min']) / current_price * 100:.1f}%",
        "Area akumulasi optimal di dekat support / pullback wajar",
        style="green"
    )
    plan_table.add_row(
        "Target Jual 1 (TP1 - Konservatif)",
        f"Rp {plan['take_profit_1']:,}",
        f"+{plan['reward_tp1_pct']:.2f}%",
        f"Area resistance terdekat (Risk/Reward: {plan['risk_reward_ratio_tp1']}:1)",
        style="cyan"
    )
    plan_table.add_row(
        "Target Jual 2 (TP2 - Agresif)",
        f"Rp {plan['take_profit_2']:,}",
        f"+{plan['reward_tp2_pct']:.2f}%",
        f"Target ekspansi tren / 52W High (Risk/Reward: {plan['risk_reward_ratio_tp2']}:1)",
        style="blue"
    )
    plan_table.add_row(
        "Stop Loss (Cut Loss / SL)",
        f"Rp {plan['stop_loss']:,}",
        f"-{plan['risk_pct']:.2f}%",
        "Batas toleransi risiko di bawah support (Proteksi modal)",
        style="red"
    )
    console.print(plan_table)

    # 6. Strategi Dividen untuk Hasil Besar
    if div_eval["has_dividend"]:
        div_panel = (
            f"• [bold]Dividend Yield[/bold]: [bold green]{div_eval['dividend_yield_pct']:.2f}% per tahun[/bold green] ({div_eval['dividend_tier']})\n"
            f"• [bold]Dividen per Lembar (DPS)[/bold]: [bold yellow]Rp {div_eval['dividend_rate_idr']:,}[/bold yellow] | Payout Ratio: {div_eval['payout_ratio_pct'] or '-'}%\n"
            f"• [bold]Tanggal Ex-Dividend[/bold]: {div_eval['ex_dividend_date']} | [bold]Estimasi Cum-Date[/bold]: {div_eval['cum_dividend_date']}\n\n"
            f"[bold cyan]🏆 REKOMENDASI MEMBELI AGAR CUAN MAKSIMAL:[/bold cyan]\n"
        )
        for idx_s, st_text in enumerate(div_eval["strategies"], 1):
            div_panel += f"  {idx_s}. {st_text}\n"
        console.print(Panel(div_panel.strip(), title="[bold yellow]💰 KALENDER DIVIDEN & STRATEGI CUAN MAKSIMAL[/bold yellow]", border_style="yellow", box=box.ROUNDED))

    # 7. Berita Positif vs Negatif
    console.print("\n[bold cyan]📰 FEED BERITA REAL-TIME (GOOGLE NEWS FINANCE)[/bold cyan]")
    
    # Berita Positif
    if news_eval["articles_positive"]:
        pos_table = Table(title="[bold green]🟢 BERITA POSITIF / BULLISH (KATALIS KENAIKAN HARGA)[/bold green]", box=box.SIMPLE_HEAD)
        pos_table.add_column("Sumber", style="dim")
        pos_table.add_column("Judul Berita", style="bold white")
        for a in news_eval["articles_positive"][:3]:
            pos_table.add_row(a["source"], a["title"])
        console.print(pos_table)

    # Berita Negatif
    if news_eval["articles_negative"]:
        neg_table = Table(title="[bold red]🔴 BERITA NEGATIF / BEARISH (FAKTOR RISIKO & WASPADA)[/bold red]", box=box.SIMPLE_HEAD)
        neg_table.add_column("Sumber", style="dim")
        neg_table.add_column("Judul Berita", style="bold white")
        for a in news_eval["articles_negative"][:3]:
            neg_table.add_row(a["source"], a["title"])
        console.print(neg_table)

    # 8. Simulasi Manajemen Modal
    mm_text = (
        f"Simulasi Modal: [bold yellow]Rp {capital:,.0f}[/bold yellow] | Toleransi Risiko: [bold red]{risk_pct}% (Rp {capital * (risk_pct/100):,.0f})[/bold red]\n"
        f"• Alokasi Disarankan: [bold green]{pos_size['lots']} Lot ({pos_size['shares']:,} Lembar)[/bold green]\n"
        f"• Total Modal Digunakan: [bold white]Rp {pos_size['total_investment']:,.0f}[/bold white] ({pos_size['capital_usage_pct']}% dari modal)\n"
        f"• Maksimum Potensi Rugi jika kena Cut Loss: [bold red]Rp {pos_size['max_loss_idr']:,.0f}[/bold red]\n"
        f"• Potensi Cuan pada Target TP1: [bold green]Rp {(plan['take_profit_1'] - current_price) * pos_size['shares']:,.0f}[/bold green]"
    )
    console.print(Panel(mm_text, title="[bold magenta]SIMULASI MANAJEMEN MODAL & UKURAN LOT[/bold magenta]", border_style="magenta", box=box.ROUNDED))


def main():
    parser = argparse.ArgumentParser(description="Analisis Saham Real-Time IDX 5 Aspek")
    parser.add_argument("--ticker", "-t", type=str, help="Kode saham BEI (contoh: BBCA, BBRI, BREN)")
    parser.add_argument("--search", "-s", type=str, help="Cari saham berdasarkan kata kunci")
    parser.add_argument("--capital", "-c", type=float, default=10000000.0, help="Total modal dalam Rupiah (default: 10 juta)")
    parser.add_argument("--risk", "-r", type=float, default=2.0, help="Toleransi risiko persen (default: 2%)")
    args = parser.parse_args()

    if args.search:
        results = search_idx_stocks(args.search)
        console.print(f"\n[bold green]Hasil Pencarian Saham BEI untuk '{args.search}' ({len(results)} ditemukan):[/bold green]")
        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Ticker", style="bold cyan")
        table.add_column("Nama Perusahaan", style="white")
        table.add_column("Sektor", style="yellow")
        for s in results[:20]:
            table.add_row(s["ticker"], s["name"], s.get("sector", "-"))
        console.print(table)
        return

    ticker = args.ticker
    if not ticker:
        console.print(f"[bold yellow]🇮🇩 Terminal Saham Real-Time BEI ({len(ALL_IDX_STOCKS)} Emiten)[/bold yellow]\n")
        try:
            ticker = console.input("[bold green]Masukkan kode saham BEI (contoh: BBCA, BBRI, ADRO): [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nKeluar program.")
            sys.exit(0)

    if not ticker:
        ticker = "BBRI"

    display_analysis(ticker, capital=args.capital, risk_pct=args.risk)


if __name__ == "__main__":
    main()
