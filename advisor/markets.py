"""Market briefing: what the macro, equity, bond, credit, FX and commodity markets look like today,
and the conclusions the portfolio draws from them (each tied to a data source)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def load_briefing_data() -> pd.DataFrame:
    import yfinance as yf

    def fetch():
        df = yf.download(list(config.MARKET_TICKERS), period="3y", auto_adjust=True, progress=False)["Close"]
        return df.dropna(how="all")

    try:
        df, status = data._cached("market_briefing", fetch, ttl_hours=6)
        data._log("Yahoo Finance market briefing", status, f"{df.shape[1]} series: indices, yields, FX, commodities, crypto")
        return df
    except Exception as e:
        data._log("Yahoo Finance market briefing", "unavailable", str(e)[:120])
        return pd.DataFrame()


def _last(s):
    s = s.dropna() if s is not None else pd.Series(dtype=float)
    return float(s.iloc[-1]) if len(s) else None


def _chg(s, days):
    s = s.dropna() if s is not None else pd.Series(dtype=float)
    if len(s) < 5:
        return None
    cut = s.index.max() - pd.Timedelta(days=days)
    prev = s[s.index <= cut]
    return float(s.iloc[-1] / prev.iloc[-1] - 1) if len(prev) else None


def _diff(s, days):
    s = s.dropna() if s is not None else pd.Series(dtype=float)
    if len(s) < 5:
        return None
    prev = s[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] - prev.iloc[-1]) if len(prev) else None


def col(df, t):
    return df[t] if df is not None and t in df else None


def readings(mk: pd.DataFrame, macro: pd.DataFrame) -> dict:
    r = {}
    r["us_10y_pct"], r["us_3m_pct"], r["us_30y_pct"] = _last(col(mk, "^TNX")), _last(col(mk, "^IRX")), _last(col(mk, "^TYX"))
    r["us_10y_change_3m_pp"] = _diff(col(mk, "^TNX"), 90)
    if r["us_10y_pct"] is not None and r["us_3m_pct"] is not None:
        r["curve_10y_minus_3m_pp"] = r["us_10y_pct"] - r["us_3m_pct"]
    if macro is not None and not macro.empty:
        if "CPIAUCSL" in macro and macro["CPIAUCSL"].notna().sum() > 13:
            cpi = macro["CPIAUCSL"].dropna()
            r["us_inflation_pct"] = float((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100)
        r["us_unemployment_pct"] = _last(col(macro, "UNRATE"))
        r["us_real_10y_pct"] = _last(col(macro, "DFII10"))
        r["hy_spread_pct"] = _last(col(macro, "BAMLH0A0HYM2"))
        hy = col(macro, "BAMLH0A0HYM2")
        r["hy_spread_1y_median_pct"] = float(hy.dropna().iloc[-260:].median()) if hy is not None and hy.notna().any() else None
    for t, k in [("^GSPC", "sp500"), ("^NDX", "nasdaq100"), ("^FTSE", "ftse100"), ("^STOXX50E", "eurostoxx50"),
                 ("^N225", "nikkei225"), ("EEM", "emerging"), ("GC=F", "gold"), ("BZ=F", "brent_oil"), ("BTC-USD", "bitcoin"),
                 ("GBPUSD=X", "gbpusd"), ("DX-Y.NYB", "dollar_index"), ("IEF", "treasuries_7_10y"), ("HYG", "high_yield"), ("LQD", "ig_credit")]:
        s = col(mk, t)
        if s is None or s.dropna().empty:
            continue
        r[f"{k}_level"] = _last(s)
        r[f"{k}_1y_pct"] = None if _chg(s, 365) is None else _chg(s, 365) * 100
        r[f"{k}_3m_pct"] = None if _chg(s, 91) is None else _chg(s, 91) * 100
        r[f"{k}_off_high_pct"] = float((s.dropna().iloc[-1] / s.dropna().iloc[-252:].max() - 1) * 100)
    r["vix"] = _last(col(mk, "^VIX"))
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if v is not None}


def conclusions(r: dict, profile: dict, overlay_notes: list[str] | None = None) -> list[dict]:
    """Each row: the data point, where it came from, the reading, what it means, and what the portfolio does about it."""
    rows = []
    allowed = set(profile.get("allowed_assets") or config.DEFAULT_ALLOWED)

    def add(data_point, source, reading, meaning, action):
        rows.append({"Data point": data_point, "Source": source, "Reading": reading, "What it means": meaning, "What we do": action})

    c = r.get("curve_10y_minus_3m_pp")
    if c is not None:
        add("Yield curve (10y minus 3m)", "Yahoo Finance ^TNX, ^IRX", f"{c:+.2f}pp",
            "Inverted: markets expect rate cuts, often a recession warning." if c < 0 else "Normal shape: no recession signal from bonds.",
            "Keep a solid government bond floor and favour quality companies." if c < 0 else "Hold medium term Treasuries for protection and income.")
    ry = r.get("us_real_10y_pct")
    if ry is not None:
        add("10y real yield (after inflation)", "FRED DFII10", f"{ry:.2f}%",
            "Bonds pay a real return above inflation again." if ry > 1.5 else "Bonds barely beat inflation.",
            "Use Treasuries and inflation linked bonds as protection that also earns." if ry > 1.5 else "Keep bonds short; lean on gold and cash for protection.")
    inf = r.get("us_inflation_pct")
    if inf is not None:
        add("US inflation (CPI, 12 months)", "FRED CPIAUCSL", f"{inf:.1f}%",
            "Inflation is above the 2% target." if inf > 3 else "Inflation is close to target.",
            "Include inflation linked bonds and gold." if inf > 3 and ({"inflation_linked", "gold"} & allowed) else "Nominal Treasuries are fine for the bond floor.")
    sp, off = r.get("sp500_1y_pct"), r.get("sp500_off_high_pct")
    if sp is not None:
        add("S&P 500 over 1 year", "Yahoo Finance ^GSPC", f"{sp:+.1f}%, {off:.1f}% from its high",
            "Stocks have run hard; future returns may be lower and a pullback more likely." if sp > 15 else "Stocks have made modest progress.",
            "Invest the equity sleeve but hedge the downside." if sp > 15 else "Invest the equity sleeve at the policy weight.")
    nq = r.get("nasdaq100_1y_pct")
    if nq is not None and sp is not None:
        add("Nasdaq 100 vs S&P 500, 1 year", "Yahoo Finance ^NDX, ^GSPC", f"{nq:+.1f}% vs {sp:+.1f}%",
            "Technology is leading the market." if nq > sp else "Technology is lagging the wider market.",
            "Growth tilt supported, but cap single stocks and themes to avoid one big tech bet." if nq > sp else "Keep the tech tilt modest; broaden with index funds.")
    vix = r.get("vix")
    if vix is not None:
        add("VIX (expected stock volatility)", "Yahoo Finance ^VIX", f"{vix:.1f}",
            "Calm markets: insurance (put options) is cheap." if vix < 18 else "Nervous markets: insurance is expensive.",
            ("Buy protective puts now while they are cheap." if vix < 18 else "Use a collar: sell upside calls to pay for the puts.")
            if "options" in allowed else "Hedge with Treasuries, gold and managed futures rather than options.")
    hy, hym = r.get("hy_spread_pct"), r.get("hy_spread_1y_median_pct")
    if hy is not None:
        add("High yield credit spread", "FRED BAMLH0A0HYM2", f"{hy:.2f}% (1y median {hym:.2f}%)" if hym else f"{hy:.2f}%",
            "Risky borrowers pay little extra: credit is expensive." if hy < 3.5 else "Credit pays a decent premium for its risk.",
            "Limit high yield bonds; prefer government bonds." if hy < 3.5 else "Allow some corporate credit for extra income.")
    if profile.get("base_currency") == "GBP" and r.get("gbpusd_level"):
        add("GBP/USD", "Yahoo Finance GBPUSD=X", f"{r['gbpusd_level']:.4f} ({r.get('gbpusd_1y_pct', 0):+.1f}% in 1 year)",
            "You spend pounds but the portfolio is in dollars: currency moves change your returns.",
            "Hedge the bond sleeve's dollars back to pounds with a forward contract; leave most equities unhedged.")
    g = r.get("gold_1y_pct")
    if g is not None:
        add("Gold over 1 year", "Yahoo Finance GC=F", f"{g:+.1f}%",
            "Gold is in demand as a hedge against geopolitics and inflation." if g > 10 else "Gold has been quiet.",
            "Hold a small gold position as a diversifier." if "gold" in allowed else "Gold excluded by the client.")
    for n in overlay_notes or []:
        add("Macro overlay rule", "Yahoo Finance and FRED", "", n.split(":")[0], n.split(":", 1)[1].strip() if ":" in n else n)
    return rows


def plain_notes(r: dict) -> dict:
    """Rules based one liners per section (used when the AI is offline)."""
    n = {}
    if "us_10y_pct" in r:
        n["Interest rates and inflation"] = (f"Ten year US government bonds pay {r['us_10y_pct']:.2f}% and three month T bills "
                                             f"{r.get('us_3m_pct', 0):.2f}%. Inflation is {r.get('us_inflation_pct', float('nan')):.1f}%.")
    if "us_unemployment_pct" in r:
        n["Jobs"] = f"US unemployment is {r['us_unemployment_pct']:.1f}%: the jobs market is the economy's pulse."
    if "sp500_1y_pct" in r:
        n["Stock markets"] = (f"The S&P 500 is {r['sp500_1y_pct']:+.1f}% over a year and the Nasdaq 100 "
                              f"{r.get('nasdaq100_1y_pct', 0):+.1f}%. The VIX fear gauge reads {r.get('vix', 0):.1f}.")
    if "treasuries_7_10y_1y_pct" in r:
        n["Bonds and credit"] = (f"Medium term Treasuries returned {r['treasuries_7_10y_1y_pct']:+.1f}% in a year; "
                                 f"high yield bonds {r.get('high_yield_1y_pct', 0):+.1f}%.")
    if "gbpusd_level" in r:
        n["Currencies"] = f"One pound buys {r['gbpusd_level']:.3f} dollars ({r.get('gbpusd_1y_pct', 0):+.1f}% in a year)."
    if "gold_1y_pct" in r:
        n["Commodities"] = f"Gold is {r['gold_1y_pct']:+.1f}% over a year and Brent oil {r.get('brent_oil_1y_pct', 0):+.1f}%."
    if "bitcoin_1y_pct" in r:
        n["Crypto"] = f"Bitcoin is {r['bitcoin_1y_pct']:+.1f}% over a year and {r.get('bitcoin_off_high_pct', 0):.1f}% from its high."
    return n
