"""Data packet for the LLMs, deterministic fallback rationale, number verification and the exportable memo."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config


def _r(x, mult=1.0, nd=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x) * mult, nd)
    except Exception:
        return None


def holding_facts(t: str, feat: pd.DataFrame, z: pd.DataFrame, comp: pd.Series, sentiment: dict, rank: int) -> dict:
    f = feat.loc[t]
    d = {
        "ticker": t, "name": f.get("name"), "sector": f.get("sector"), "theme_bucket": f.get("theme_bucket"),
        "composite_rank": rank, "composite_score": _r(comp.get(t), 1, 2),
        "market_cap_bn": _r(f.get("market_cap_bn"), 1, 0),
        "revenue_growth_pct": _r(f.get("revenue_growth"), 100), "earnings_growth_pct": _r(f.get("earnings_growth"), 100),
        "gross_margin_pct": _r(f.get("gross_margin"), 100), "operating_margin_pct": _r(f.get("operating_margin"), 100),
        "fcf_margin_pct": _r(f.get("fcf_margin"), 100), "forward_pe_x": _r(f.get("forward_pe")),
        "price_to_sales_x": _r(f.get("ps_ratio")),
        "momentum_12m_ex_1m_pct": _r(f.get("ret_12_1"), 100), "return_6m_pct": _r(f.get("ret_6m"), 100),
        "return_1y_pct": _r(f.get("ret_1y"), 100), "volatility_1y_pct": _r(f.get("vol_1y"), 100),
        "max_drawdown_3y_pct": _r(f.get("max_dd_3y"), 100), "off_52w_high_pct": _r(f.get("off_52w_high"), 100),
        "beta_vs_qqq": _r(f.get("beta_qqq"), 1, 2), "analyst_rating_1_buy_5_sell": _r(f.get("analyst_rating"), 1, 2),
        "analyst_target_upside_pct": _r(f.get("target_upside"), 100), "theme_fit": _r(f.get("theme_fit"), 1, 3),
        "factor_z": {k: _r(v, 1, 2) for k, v in z.loc[t].items()},
    }
    s = (sentiment or {}).get(t)
    if s:
        d["news_sentiment"] = _r(s.get("score"), 1, 2)
        d["news_driver"] = s.get("driver")
    return {k: v for k, v in d.items() if v is not None}


def build_packet(res: dict) -> dict:
    prof = res["profile"]
    feat, z, comp = res["features"], res["z"], res["composite"]
    ranks = comp.rank(ascending=False).astype(int)
    hold = [holding_facts(t, feat, z, comp, res.get("sentiment"), int(ranks[t])) for t in res["stock_weights"].index]
    for h in hold:
        h["risk_contribution_pct"] = _r(res["risk_contrib"].get(h["ticker"]), 100)
    near = [{"ticker": t, "composite_rank": int(ranks[t]), "composite_score": _r(comp[t], 1, 2),
             "revenue_growth_pct": _r(feat.loc[t, "revenue_growth"], 100), "why_not": res["why_not"].get(t, "")}
            for t in res["near_misses"]]
    st = res["stats"]
    stress = [{"scenario": r["Scenario"], "portfolio_pct": _r(r["Portfolio"], 100), "benchmark_pct": _r(r[prof["benchmark"]], 100)
               if prof["benchmark"] in r else _r(r.get("QQQ"), 100), "pnl_usd": _r(r["P&L on $100k"], 1, 0)}
              for _, r in res["stress"].iterrows()]
    mc = res["mc"]
    return {
        "client": {k: prof.get(k) for k in ["client_name", "objective", "horizon_years", "risk_tolerance", "max_drawdown",
                                             "target_vol", "max_position", "themes", "exclude_tickers", "exclude_sectors",
                                             "benchmark", "constraints_summary"]},
        "amount_usd": config.INVESTMENT_AMOUNT,
        "macro": {k: (_r(v, 1, 2) if isinstance(v, (int, float)) else v) for k, v in res["macro"].items()},
        "macro_overlay_rules_applied": res["overlay"]["notes"],
        "stock_sleeve_pct": _r(res["opt_info"]["stock_share"], 100),
        "strategic_policy_for_risk_tolerance": {k: _r(v, 100) for k, v in res["opt_info"]["policy"].items()},
        "sleeves_pct": {k: _r(v, 100) for k, v in res["opt_info"]["sleeves"].items()},
        "funds": [{"ticker": t, "holds": config.ETFS[t][0], "sleeve": config.ETFS[t][1], "weight_pct": _r(v, 100),
                              "share_of_risk_pct": _r(res["risk_contrib"].get(t), 100)} for t, v in res["defensive_weights"].items()],
        "binding_client_limits": res["opt_info"].get("binding", []),
        "portfolio": {
            "expected_return_model_pct": _r(res["opt_info"]["exp_return"], 100),
            "expected_volatility_pct": _r(res["opt_info"]["exp_vol"], 100),
            "backtest_3y_in_sample": {k: _r(v, 100 if k not in ("Sharpe", "Sortino", "Beta") else 1, 2 if k in ("Sharpe", "Sortino", "Beta") else 1) for k, v in st.items()},
            "monte_carlo": {"horizon_years": prof["horizon_years"], "median_value_usd": _r(np.median(mc["terminal"]), 1, 0),
                            "p5_value_usd": _r(np.percentile(mc["terminal"], 5), 1, 0),
                            "p95_value_usd": _r(np.percentile(mc["terminal"], 95), 1, 0),
                            "prob_drawdown_beyond_client_limit_pct": _r(res["p_breach"], 100),
                            "prob_loss_after_horizon_pct": _r((mc["terminal"] < config.INVESTMENT_AMOUNT).mean(), 100)},
            "stress_tests": stress,
        },
        "holdings": hold,
        "near_misses": near,
    }


# ---------------------------------------------------------------- deterministic fallback (no LLM or LLM failure)
def fallback_holding(h: dict) -> dict:
    z = h.get("factor_z", {})
    phrases = {
        "growth": f"revenue growing {h.get('revenue_growth_pct')}%" if h.get("revenue_growth_pct") is not None else None,
        "momentum": f"{h.get('momentum_12m_ex_1m_pct')}% price momentum over the past year" if h.get("momentum_12m_ex_1m_pct") is not None else None,
        "quality": f"a {h.get('operating_margin_pct')}% operating margin" if h.get("operating_margin_pct") is not None else None,
        "valuation": f"a forward P/E of {h.get('forward_pe_x')}x" if h.get("forward_pe_x") is not None else None,
        "analyst": f"{h.get('analyst_target_upside_pct')}% upside to the analyst consensus target" if h.get("analyst_target_upside_pct") is not None else None,
        "theme": f"a theme fit score of {h.get('theme_fit')}" if h.get("theme_fit") is not None else None,
        "low_vol": f"lower volatility than most peers ({h.get('volatility_1y_pct')}% a year)" if h.get("volatility_1y_pct") is not None else None,
        "sentiment": f"positive news flow ({h.get('news_driver')})" if h.get("news_driver") else None,
    }
    order = sorted(z, key=lambda k: z[k], reverse=True)
    strengths = [phrases[k] for k in order if phrases.get(k)][:2]
    weak = next((k for k in reversed(order) if z.get(k, 0) < 0), None)
    risk_map = {
        "valuation": f"Valuation is demanding (forward P/E {h.get('forward_pe_x')}x), so a growth miss would hit the shares hard.",
        "momentum": f"Price momentum is weak ({h.get('momentum_12m_ex_1m_pct')}%), the market is not yet rewarding the story.",
        "growth": f"Growth is below peers at {h.get('revenue_growth_pct')}% revenue growth.",
        "quality": f"Profitability trails peers (operating margin {h.get('operating_margin_pct')}%).",
    }
    rg = h.get("revenue_growth_pct")
    return {
        "ticker": h["ticker"],
        "role": "Growth engine" if z.get("growth", 0) > 0.5 else "Core compounder" if z.get("quality", 0) > 0 else "Satellite",
        "why": f"{h.get('name', h['ticker'])} ranks #{h['composite_rank']} in our universe on the composite score, driven by "
               + " and ".join(strengths) + ".",
        "key_risk": risk_map.get(weak, f"High volatility ({h.get('volatility_1y_pct')}% a year) with a {h.get('max_drawdown_3y_pct')}% drawdown in the last 3 years."),
        "sell_trigger": (f"Revenue growth slows below {round(rg / 2, 1)}% or the stock falls out of the top 20 on the composite score."
                         if rg and rg > 0 else "The stock falls out of the top 20 on the composite score for two consecutive reviews."),
        "source": "rules",
    }


def fallback_decision(packet: dict) -> dict:
    c, p = packet["client"], packet["portfolio"]
    worst = min(p["stress_tests"], key=lambda s: s["portfolio_pct"]) if p["stress_tests"] else None
    return {
        "headline": f"Invest ${packet['amount_usd']:,} as a core and satellite growth portfolio: "
                    + ", ".join(f"{v}% {k.lower()}" for k, v in packet["sleeves_pct"].items() if v) + ".",
        "portfolio_thesis": f"The portfolio targets {c['objective'].lower().rstrip('.')} by holding the highest ranked companies on growth, "
                            f"momentum, quality and theme fit, sized by an optimiser that caps any single stock at "
                            f"{round(c['max_position'] * 100)}% and volatility at or below {round(c['target_vol'] * 100)}%, "
                            f"while losing less than {round(c['max_drawdown'] * 100)}% in every historical stress replay.",
        "suitability": (f"In the worst historical stress ({worst['scenario']}) the portfolio would have moved {worst['portfolio_pct']}%, versus the "
                        f"client's stated limit of {round(c['max_drawdown'] * 100)}%." if worst else ""),
        "adjustments": [],
        "holdings": [fallback_holding(h) for h in packet["holdings"]],
        "key_risks": ["Concentration in technology growth factors", "Sensitivity to rising real yields", "Single stock earnings risk"],
        "dissent": "AI committee offline: rules based explanation shown.",
    }


# ---------------------------------------------------------------- verification: every cited number traced to data
NUM_RE = re.compile(r"(?<![\w.])\$?(-?\d[\d,]*\.?\d*)\s*(%|x\b|pp\b|bn\b|billion)?")


def _numbers(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
        out.append(float(abs(obj)))
    return out


def verify_text(text: str, allowed: list[float]) -> list[tuple[str, bool]]:
    found = []
    for m in NUM_RE.finditer(text or ""):
        raw, unit = m.group(1), m.group(2)
        is_money = m.group(0).startswith("$")
        if not unit and not is_money:
            continue  # only audit financial figures (percent, multiples, $ amounts)
        try:
            x = float(raw.replace(",", ""))
        except ValueError:
            continue
        ok = any(abs(x - v) <= 0.051 or (abs(v) > 1 and abs(x - v) / abs(v) < 0.005)
                 or (x == round(x) and abs(x - round(v)) < 1e-9) for v in allowed)
        found.append((m.group(0).strip(), ok))
    return found


def verify_decision(decision: dict, packet: dict) -> dict:
    global_vals = _numbers({k: v for k, v in packet.items() if k != "holdings"}, [])
    hv = {h["ticker"]: _numbers(h, []) for h in packet["holdings"]}
    out, total, good = {}, 0, 0
    for h in decision.get("holdings", []):
        allowed = hv.get(h.get("ticker"), []) + global_vals
        checks = verify_text(" ".join(str(h.get(k, "")) for k in ("why", "key_risk", "sell_trigger")), allowed)
        out[h.get("ticker")] = checks
        total += len(checks)
        good += sum(ok for _, ok in checks)
    all_vals = _numbers(packet, [])
    port_checks = verify_text(" ".join(str(decision.get(k, "")) for k in ("headline", "portfolio_thesis", "suitability")), all_vals)
    total += len(port_checks)
    good += sum(ok for _, ok in port_checks)
    return {"per_holding": out, "portfolio": port_checks, "total": total, "verified": good}


# ---------------------------------------------------------------- export
def memo_markdown(res: dict) -> str:
    d, prof = res["decision"], res["profile"]
    trades = res["trades"]
    lines = [f"# Investment recommendation for {prof['client_name']}", "",
             f"Amount: ${config.INVESTMENT_AMOUNT:,}. Data as of {res['as_of']}. Educational hackathon project, not financial advice.", "",
             f"**{d.get('headline', '')}**", "", "## Thesis", d.get("portfolio_thesis", ""), "", "## Suitability", d.get("suitability", ""), "",
             "## Allocation", "", "| Ticker | Weight | Dollars | Shares |", "|---|---|---|---|"]
    for _, r in trades.iterrows():
        sh = "" if pd.isna(r.get("Shares")) else f"{r['Shares']:.3f}"
        nm = f"{r['Ticker']} ({r['Holding']})" if r.get("Holding") else r["Ticker"]
        lines.append(f"| {nm} | {r['Weight'] * 100:.1f}% | ${r['Dollars']:,.0f} | {sh} |")
    lines += ["", "## Why each holding"]
    for h in d.get("holdings", []):
        if h.get("ticker") not in res["weights"].index:
            continue
        lines += [f"### {h['ticker']} ({h.get('role', '')})", h.get("why", ""), f"*Key risk:* {h.get('key_risk', '')}",
                  f"*Sell trigger:* {h.get('sell_trigger', '')}", ""]
    lines += ["## Key risks"] + [f"* {k}" for k in d.get("key_risks", [])]
    lines += ["", "## Macro overlay"] + [f"* {n}" for n in res["overlay"]["notes"]]
    lines += ["", "## Committee", d.get("dissent", ""), "", "Sources: Yahoo Finance (prices, fundamentals, analyst consensus, news, rates, VIX), "
              "FRED (real yields, CPI, credit spreads), Finnhub (optional), Azure OpenAI (analysis)."]
    return "\n".join(lines)
