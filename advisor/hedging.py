"""Hedging overlay: index put options (protective put or collar) priced from live Yahoo option chains,
with a Black Scholes fallback, plus a GBP/USD forward hedge for clients who think in pounds."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, data


def _bs(S, K, T, r, sigma, kind):
    if T <= 0 or sigma <= 0:
        return max(0.0, (K - S) if kind == "put" else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1) if kind == "put" else S * N(d1) - K * math.exp(-r * T) * N(d2)


def _chain(underlying: str, target_days: int = 120):
    """Nearest listed expiry to target_days and its chain, cached."""
    import yfinance as yf

    def fetch():
        tk = yf.Ticker(underlying)
        exps = tk.options
        if not exps:
            raise ValueError("no expiries")
        today = datetime.now()
        exp = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - today).days - target_days))
        ch = tk.option_chain(exp)
        return {"expiry": exp, "puts": ch.puts, "calls": ch.calls}

    val, status = data._cached(f"options_{underlying}_{target_days}", fetch, ttl_hours=6)
    data._log(f"Yahoo Finance option chain {underlying}", status, f"expiry {val['expiry']}")
    return val


def _pick(df: pd.DataFrame, strike: float):
    row = df.iloc[(df["strike"] - strike).abs().argsort().iloc[0]]
    bid, ask, last = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0), float(row.get("lastPrice", 0) or 0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    return float(row["strike"]), mid, float(row.get("impliedVolatility", np.nan) or np.nan)


def option_hedge(weights: pd.Series, rets: pd.DataFrame, px: pd.DataFrame, profile: dict, rf: float, vix: float | None) -> dict | None:
    if "options" not in (profile.get("allowed_assets") or []):
        return None
    amount = config.INVESTMENT_AMOUNT
    eq = [t for t in weights.index if t in config.STOCKS or (t in config.ETFS and config.ETFS[t][1] in (config.SLEEVE_INDEX,)) or t in ("USMV", "IBIT")]
    if not eq:
        return None
    tech = sum(weights[t] for t in eq if t in config.STOCKS or t == "QQQ") / max(sum(weights[t] for t in eq), 1e-9)
    und = "QQQ" if tech > 0.5 else "SPY"
    w_eq = weights[eq]
    port_eq = (rets[eq].iloc[-504:].fillna(0) * w_eq.values).sum(axis=1) / w_eq.sum()
    j = pd.concat([port_eq, rets[und].iloc[-504:]], axis=1).dropna()
    beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) if len(j) > 60 else 1.0
    exposure = float(w_eq.sum()) * amount
    S = float(px[und].dropna().iloc[-1])
    rt = profile.get("risk_tolerance", 6)
    style = profile.get("hedge_style", "auto")
    if style == "auto":
        style = "Protective put" if rt >= 7 and (vix or 20) < 22 else "Collar"
    put_k, call_k = S * (0.90 if rt >= 6 else 0.95), S * (1.15 if rt >= 7 else 1.08)
    days, source = 120, "live Yahoo Finance option chain"
    try:
        ch = _chain(und, 120)
        expiry = ch["expiry"]
        put_k, put_px, put_iv = _pick(ch["puts"], put_k)
        call_k, call_px, call_iv = _pick(ch["calls"], call_k)
        days = max(1, (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days)
        if not put_px or put_px <= 0:
            raise ValueError("no put price")
    except Exception:
        source = "Black Scholes estimate using the VIX (option chain unavailable)"
        expiry = (pd.Timestamp.now() + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        sig = (vix or 20) / 100 * (1.15 if und == "QQQ" else 1.0)
        put_k, call_k = round(put_k), round(call_k)
        put_px, call_px = _bs(S, put_k, days / 365, rf, sig * 1.1, "put"), _bs(S, call_k, days / 365, rf, sig * 0.9, "call")
        put_iv = call_iv = sig
    contracts = max(1, int(round(beta * exposure / (S * 100))))
    put_cost = contracts * 100 * put_px
    call_credit = contracts * 100 * call_px if style == "Collar" else 0.0
    net = put_cost - call_credit
    moves = np.linspace(-0.40, 0.30, 71)
    unhedged = beta * moves * exposure
    put_pay = contracts * 100 * np.maximum(put_k - S * (1 + moves), 0)
    call_pay = -contracts * 100 * np.maximum(S * (1 + moves) - call_k, 0) if style == "Collar" else 0
    hedged = unhedged + put_pay + call_pay - net
    return {
        "style": style, "underlying": und, "spot": S, "expiry": expiry, "days": days, "contracts": contracts,
        "put_strike": put_k, "put_price": put_px, "put_iv": put_iv, "call_strike": call_k if style == "Collar" else None,
        "call_price": call_px if style == "Collar" else None, "put_cost": put_cost, "call_credit": call_credit,
        "net_cost": net, "net_cost_pct": net / amount, "hedge_ratio": contracts * 100 * S / max(beta * exposure, 1),
        "beta": beta, "equity_exposure": exposure, "source": source,
        "protection_from": put_k / S - 1, "cap_at": (call_k / S - 1) if style == "Collar" else None,
        "payoff": pd.DataFrame({"move": moves, "Without hedge": unhedged, "With hedge": hedged}),
        "loss_at_minus_30": {"without": float(beta * -0.30 * exposure),
                             "with": float(beta * -0.30 * exposure + contracts * 100 * max(put_k - S * 0.7, 0) - net)},
    }


def fx_hedge(weights: pd.Series, profile: dict, gbpusd: float | None, us_3m: float | None, uk_3m: float | None) -> dict | None:
    if profile.get("base_currency") != "GBP" or not gbpusd:
        return None
    rt = profile.get("risk_tolerance", 6)
    bonds = sum(v for t, v in weights.items() if config.sleeve_of(t) in (config.SLEEVE_BONDS, config.SLEEVE_CREDIT))
    eq_ratio = 0.5 if rt <= 4 else 0.25 if rt <= 6 else 0.0
    eq = 1 - bonds
    usd_hedged = config.INVESTMENT_AMOUNT * (bonds * 1.0 + eq * eq_ratio)
    t = 0.25
    fwd = gbpusd * (1 + (us_3m or 0) / 100 * t) / (1 + (uk_3m or us_3m or 0) / 100 * t) if us_3m is not None else None
    return {"usd_to_hedge": usd_hedged, "gbp_equivalent": usd_hedged / gbpusd, "spot": gbpusd, "forward_3m": fwd,
            "bond_ratio": 1.0, "equity_ratio": eq_ratio,
            "why": "Bonds are there to be the calm part of the portfolio, so their currency risk is removed; equities stay mostly "
                   "unhedged because the dollar often rises when stocks fall, which cushions a pound based investor."}
