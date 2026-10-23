"""Public data layer: Yahoo Finance (prices, fundamentals, news), FRED (macro), Finnhub (analyst trends, news).

Every fetch is cached to disk. If a live call fails (venue wifi, rate limits), the last good snapshot is used
and the source is flagged as cached, so the demo never runs on invented numbers.
"""
from __future__ import annotations

import hashlib
import io
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from . import config

SOURCE_LOG: dict[str, dict] = {}


def _log(source: str, status: str, detail: str = ""):
    SOURCE_LOG[source] = {"status": status, "detail": detail, "time": datetime.now().strftime("%H:%M:%S")}


def _h(items) -> str:
    """Stable hash (Python's hash() is salted per process, which would break the disk cache)."""
    return hashlib.md5(",".join(sorted(items)).encode()).hexdigest()[:10]


def _cache_path(key: str):
    safe = "".join(c if c.isalnum() else "_" for c in key)
    return config.CACHE_DIR / f"{safe}.pkl"


def _cached(key: str, fn, ttl_hours: float = 12):
    """Return fresh cache, else live fetch, else stale cache. Returns (value, status)."""
    p = _cache_path(key)
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl_hours * 3600:
        with open(p, "rb") as f:
            return pickle.load(f), "cache"
    try:
        val = fn()
        if val is None or (hasattr(val, "empty") and val.empty) or (isinstance(val, dict) and not val):
            raise ValueError("empty response")
        with open(p, "wb") as f:
            pickle.dump(val, f)
        return val, "live"
    except Exception as e:  # fall back to last snapshot
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f), f"stale cache ({type(e).__name__})"
        raise


# ---------------------------------------------------------------- prices
def get_prices(tickers: list[str], years: int = 7) -> pd.DataFrame:
    import yfinance as yf

    def fetch():
        df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False, threads=True)
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tickers[0]})
        close = close.dropna(how="all")
        return close

    key = f"prices_{years}y_{_h(tickers)}"
    try:
        px, status = _cached(key, fetch, ttl_hours=6)
        good = [t for t in tickers if t in px.columns and px[t].notna().sum() > 120]
        _log("Yahoo Finance prices", status, f"{len(good)}/{len(tickers)} tickers, {px.index.min():%b %Y} to {px.index.max():%d %b %Y}")
        return px[good]
    except Exception as e:
        _log("Yahoo Finance prices", "unavailable", str(e)[:120])
        raise RuntimeError("Price data unavailable and no snapshot on disk. Check the internet connection.") from e


# ---------------------------------------------------------------- fundamentals
FUND_FIELDS = [
    "shortName", "sector", "industry", "marketCap", "currentPrice", "revenueGrowth", "earningsGrowth",
    "grossMargins", "operatingMargins", "profitMargins", "freeCashflow", "totalRevenue", "forwardPE",
    "trailingPE", "trailingPegRatio", "priceToSalesTrailing12Months", "recommendationMean",
    "numberOfAnalystOpinions", "targetMeanPrice", "beta", "longBusinessSummary",
]


def _one_info(t: str) -> dict:
    import yfinance as yf
    for attempt in range(2):
        try:
            info = yf.Ticker(t).info or {}
            if info:
                return {k: info.get(k) for k in FUND_FIELDS}
        except Exception:
            time.sleep(1 + attempt)
    return {}


def get_fundamentals(tickers: list[str]) -> pd.DataFrame:
    def fetch():
        with ThreadPoolExecutor(max_workers=5) as ex:
            rows = dict(zip(tickers, ex.map(_one_info, tickers)))
        df = pd.DataFrame.from_dict(rows, orient="index")
        return df.dropna(how="all")

    key = f"fund_{_h(tickers)}"
    try:
        df, status = _cached(key, fetch, ttl_hours=24)
        _log("Yahoo Finance fundamentals", status, f"{len(df)} companies, {len(FUND_FIELDS)} fields")
        df = df.reindex(tickers)
        return df
    except Exception as e:
        _log("Yahoo Finance fundamentals", "unavailable", str(e)[:120])
        return pd.DataFrame(index=tickers, columns=FUND_FIELDS)


# ---------------------------------------------------------------- macro (FRED)
def _fred_series(sid: str, start: str) -> pd.Series:
    key = os.getenv("FRED_API_KEY")
    if key:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": start},
            timeout=15,
        )
        r.raise_for_status()
        obs = r.json()["observations"]
        s = pd.Series({o["date"]: o["value"] for o in obs})
    else:  # public CSV endpoint, no key needed
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid, "cosd": start}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        s = df.set_index(df.columns[0])[df.columns[1]]
    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s, errors="coerce").dropna()


def get_macro() -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    def fetch():
        out = {}
        for sid in config.FRED_SERIES:
            try:
                out[sid] = _fred_series(sid, start)
            except Exception:
                pass
        return pd.DataFrame(out)

    try:
        df, status = _cached("fred_macro", fetch, ttl_hours=12)
        _log("FRED (St. Louis Fed)", status, f"{df.shape[1]} series, latest {df.index.max():%d %b %Y}")
        return df
    except Exception as e:
        _log("FRED (St. Louis Fed)", "unavailable", str(e)[:120])
        return pd.DataFrame()


YAHOO_MACRO = {"^TNX": "DGS10", "^FVX": "DGS5", "^IRX": "DGS3MO", "^VIX": "VIXCLS", "HYG": "HYG", "IEF": "IEF"}


def get_yahoo_macro() -> pd.DataFrame:
    """Rates, volatility and credit proxies straight from Yahoo Finance (primary macro source)."""
    import yfinance as yf

    def fetch():
        df = yf.download(list(YAHOO_MACRO), period="3y", auto_adjust=True, progress=False)["Close"]
        return df.rename(columns=YAHOO_MACRO).dropna(how="all")

    try:
        df, status = _cached("yahoo_macro", fetch, ttl_hours=6)
        _log("Yahoo Finance rates, VIX, credit", status, "^TNX, ^FVX, ^IRX, ^VIX, HYG/IEF")
        return df
    except Exception as e:
        _log("Yahoo Finance rates, VIX, credit", "unavailable", str(e)[:120])
        return pd.DataFrame()


def combine_macro(yahoo: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Yahoo first; FRED adds what Yahoo lacks (real yields, CPI, unemployment, HY spread)."""
    if yahoo is None or yahoo.empty:
        return fred if fred is not None else pd.DataFrame()
    if fred is None or fred.empty:
        return yahoo
    extra = fred[[c for c in fred.columns if c not in yahoo.columns]]
    return yahoo.join(extra, how="outer")


def macro_snapshot(macro: pd.DataFrame) -> dict:
    """Latest readings plus simple regime signals used by the allocation overlay."""
    if macro is None or macro.empty:
        return {}

    def last(c):
        return float(macro[c].dropna().iloc[-1]) if c in macro and macro[c].notna().any() else None

    def ago(c, days):
        if c not in macro or macro[c].notna().sum() == 0:
            return None
        s = macro[c].dropna()
        cut = s.index.max() - pd.Timedelta(days=days)
        prev = s[s.index <= cut]
        return float(prev.iloc[-1]) if len(prev) else None

    snap = {k: last(k) for k in list(config.FRED_SERIES) + ["DGS5"]}
    if snap.get("DGS2") is None and snap.get("DGS5") is not None:
        snap["DGS2"] = None  # curve below then uses 10y minus 3m
    if "HYG" in macro and "IEF" in macro:
        ratio = (macro["HYG"] / macro["IEF"]).dropna()
        if len(ratio) > 70:
            snap["CREDIT_HYG_IEF_3M"] = float((ratio.iloc[-1] / ratio.iloc[-64] - 1) * 100)
    if "CPIAUCSL" in macro and macro["CPIAUCSL"].notna().sum() > 13:
        cpi = macro["CPIAUCSL"].dropna()
        snap["CPI_YOY"] = float((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100)
    short = snap.get("DGS2") if snap.get("DGS2") is not None else snap.get("DGS3MO")
    snap["CURVE"] = (snap["DGS10"] - short) if snap.get("DGS10") is not None and short is not None else None
    snap["CURVE_LABEL"] = "10y minus 2y" if snap.get("DGS2") is not None else "10y minus 3m"
    snap["DGS10_CHG_3M"] = (snap["DGS10"] - ago("DGS10", 90)) if snap.get("DGS10") is not None and ago("DGS10", 90) is not None else None
    r_now, r_3m = snap.get("DFII10"), ago("DFII10", 90)
    snap["REAL_YIELD_CHG_3M"] = (r_now - r_3m) if r_now is not None and r_3m is not None else None
    hy = macro["BAMLH0A0HYM2"].dropna() if "BAMLH0A0HYM2" in macro else pd.Series(dtype=float)
    snap["HY_SPREAD_1Y_MEDIAN"] = float(hy.iloc[-260:].median()) if len(hy) else None
    snap["as_of"] = macro.index.max().strftime("%d %b %Y")
    return snap


# ---------------------------------------------------------------- news and analyst views
def _finnhub(path: str, **params):
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("no Finnhub key")
    r = requests.get(f"https://finnhub.io/api/v1/{path}", params={**params, "token": key}, timeout=10)
    r.raise_for_status()
    return r.json()


def _yahoo_news(t: str) -> list[dict]:
    import yfinance as yf
    out = []
    for n in (yf.Ticker(t).news or [])[:8]:
        c = n.get("content", n)
        title = c.get("title")
        if not title:
            continue
        prov = c.get("provider", {})
        out.append({
            "title": title,
            "source": prov.get("displayName") if isinstance(prov, dict) else c.get("publisher"),
            "date": (c.get("pubDate") or "")[:10] or (
                datetime.fromtimestamp(c["providerPublishTime"]).strftime("%Y-%m-%d") if c.get("providerPublishTime") else ""),
            "url": (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link"),
        })
    return out


def _one_news(t: str) -> list[dict]:
    try:
        to = datetime.now().strftime("%Y-%m-%d")
        frm = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        items = _finnhub("company-news", symbol=t, **{"from": frm, "to": to})
        return [{"title": i["headline"], "source": i.get("source"), "url": i.get("url"),
                 "date": datetime.fromtimestamp(i["datetime"]).strftime("%Y-%m-%d")} for i in items[:8]]
    except Exception:
        try:
            return _yahoo_news(t)
        except Exception:
            return []


def get_news(tickers: list[str]) -> dict[str, list[dict]]:
    def fetch():
        with ThreadPoolExecutor(max_workers=8) as ex:
            return dict(zip(tickers, ex.map(_one_news, tickers)))

    try:
        news, status = _cached(f"news_{_h(tickers)}", fetch, ttl_hours=3)
        src = "Finnhub company news" if os.getenv("FINNHUB_API_KEY") else "Yahoo Finance news"
        _log(src, status, f"{sum(len(v) for v in news.values())} headlines across {len(news)} companies")
        return news
    except Exception as e:
        _log("News", "unavailable", str(e)[:120])
        return {t: [] for t in tickers}


def _one_rec(t: str):
    try:
        trend = _finnhub("stock/recommendation", symbol=t)
        if trend:
            r = trend[0]
            total = sum(r.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]) or np.nan
            return {"buy_share": (r.get("strongBuy", 0) + r.get("buy", 0)) / total, "analysts": total, "period": r.get("period")}
    except Exception:
        pass
    return None


def get_analyst_trends(tickers: list[str]) -> pd.DataFrame:
    if not os.getenv("FINNHUB_API_KEY"):
        _log("Finnhub analyst ratings", "skipped", "no FINNHUB_API_KEY, using Yahoo consensus instead")
        return pd.DataFrame()

    def fetch():
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = dict(zip(tickers, ex.map(_one_rec, tickers)))
        return pd.DataFrame.from_dict({k: v for k, v in rows.items() if v}, orient="index")

    try:
        df, status = _cached(f"finnhub_rec_{_h(tickers)}", fetch, ttl_hours=24)
        _log("Finnhub analyst ratings", status, f"{len(df)} companies")
        return df
    except Exception as e:
        _log("Finnhub analyst ratings", "unavailable", str(e)[:120])
        return pd.DataFrame()
