"""End to end run: data -> scores -> optimiser -> risk -> AI committee -> adjustments -> verified recommendation."""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from . import analytics as A
from . import config, data, hedging, llm, markets, report

BUCKET_TO_THEMES = {
    "AI semiconductors": ["Artificial intelligence"],
    "AI infrastructure and power": ["Artificial intelligence", "Energy transition and power"],
    "Cloud and cybersecurity software": ["Cloud software", "Cybersecurity"],
    "Digital platforms": ["Digital consumer platforms"],
    "Fintech and ecommerce": ["Fintech", "Digital consumer platforms"],
    "Healthcare innovation": ["Healthcare innovation"],
    "Autonomy and robotics": ["Robotics and autonomy"],
}


def load_market_data(progress=lambda m: None) -> dict:
    data.SOURCE_LOG.clear()
    progress("Pulling 7 years of daily prices from Yahoo Finance")
    px = data.get_prices(config.ALL_TICKERS, years=7)
    progress("Pulling fundamentals and analyst consensus from Yahoo Finance")
    fund = data.get_fundamentals([t for t in config.STOCKS if t in px.columns])
    progress("Pulling rates, VIX and credit (Yahoo Finance) and macro series (FRED)")
    macro = data.combine_macro(data.get_yahoo_macro(), data.get_macro())
    progress("Pulling recent company news")
    news = data.get_news([t for t in config.STOCKS if t in px.columns])
    analyst = data.get_analyst_trends([t for t in config.STOCKS if t in px.columns])
    progress("Pulling global indices, yields, currencies, commodities and crypto for the market briefing")
    mk = markets.load_briefing_data()
    return {"px": px, "fund": fund, "macro": macro, "news": news, "analyst": analyst, "mk": mk, "sources": dict(data.SOURCE_LOG)}


def market_brief(md: dict, prof: dict, use_ai: bool = True) -> dict:
    snap = data.macro_snapshot(md["macro"])
    overlay = A.macro_overlay(snap, prof)
    r = markets.readings(md.get("mk"), md["macro"])
    concl = markets.conclusions(r, prof, overlay["notes"])
    out = {"readings": r, "conclusions": concl, "notes": markets.plain_notes(r), "story": None, "macro": snap}
    if use_ai and llm.available():
        try:
            out["story"] = llm.market_story(r, concl, prof)
        except Exception as e:
            out["error"] = f"AI market briefing unavailable, rules based notes shown: {e}"
    return out


def _exclude(feat: pd.DataFrame, prof: dict) -> dict:
    out = {}
    ex_t = set(prof.get("exclude_tickers", []))
    ex_s = [s.lower() for s in prof.get("exclude_sectors", [])]
    for t in feat.index:
        if t in ex_t:
            out[t] = "Excluded by client"
        sec = f"{feat.loc[t, 'sector']} {feat.loc[t, 'industry']}".lower()
        if any(s and s in sec for s in ex_s):
            out[t] = "Sector excluded by client"
    return out


def run(md: dict, prof: dict, use_ai: bool = True, progress=lambda m: None) -> dict:
    prof = dict(prof)
    prof["target_vol"] = config.target_vol_for(prof["risk_tolerance"], prof["max_drawdown"])
    res = {"profile": prof, "errors": [], "sources": md["sources"]}
    px, fund = md["px"], md["fund"]
    bench = prof.get("benchmark", "QQQ") if prof.get("benchmark") in px.columns else "QQQ"
    rets = px.pct_change(fill_method=None)
    snap = data.macro_snapshot(md["macro"])
    rf = (snap.get("DGS3MO") or 4.0) / 100
    res.update(macro=snap, rf=rf, as_of=px.index.max().strftime("%d %b %Y"), bench=bench)
    ai = use_ai and llm.available()

    # ---- thematic fit (embeddings)
    progress("Embedding every business description and matching it to the client's themes")
    theme_fit, tm = None, None
    if ai:
        try:
            tm = llm.theme_matrix(fund["longBusinessSummary"])
            theme_fit = tm[prof["themes"]].max(axis=1)
        except Exception as e:
            res["errors"].append(f"Embeddings failed, using theme buckets: {e}")
    if theme_fit is None:
        theme_fit = pd.Series({t: float(any(th in prof["themes"] for th in BUCKET_TO_THEMES.get(config.TICKER_THEME[t], [])))
                               for t in config.STOCKS})
    res["theme_matrix"] = tm

    # ---- news sentiment
    sentiment = {}
    if ai:
        progress("Reading recent headlines and scoring fundamental news sentiment")
        try:
            sentiment = llm.news_sentiment(md["news"])
        except Exception as e:
            res["errors"].append(f"News sentiment failed: {e}")
    res["sentiment"] = sentiment

    # ---- factors, macro overlay, composite
    progress("Scoring the universe on growth, momentum, quality, valuation, analysts, sentiment and theme")
    pm = A.price_metrics(px, bench="QQQ")
    feat = A.build_features(fund, pm, md["analyst"], sentiment, theme_fit)
    z = A.factor_scores(feat)
    overlay = A.macro_overlay(snap, prof)
    weights = dict(config.DEFAULT_FACTOR_WEIGHTS)
    weights["valuation"] *= overlay["valuation_mult"]
    rt = prof["risk_tolerance"]
    weights["low_vol"] = 0.04 * (10 - rt)          # cautious clients reward steadier stocks
    weights["quality"] += 0.02 * (10 - rt)
    if not sentiment:
        weights["sentiment"] = 0
    comp = A.composite(z, weights).sort_values(ascending=False)
    excluded = _exclude(feat, prof)
    eligible = [t for t in comp.index if t not in excluded]
    res.update(features=feat, z=z, factor_weights=weights, overlay=overlay, composite=comp, excluded=excluded, cash=overlay["cash"])
    res["conclusions"] = markets.conclusions(markets.readings(md.get("mk"), md["macro"]), prof, overlay["notes"])

    # ---- select and optimise
    progress("Optimising position sizes under the client's risk limits")
    n = prof["n_holdings"]
    caps, removed = {}, {}

    def build(selection):
        stress = A.scenario_matrix(selection + list(config.DEFENSIVE), px, bench)
        return A.optimise(selection, comp, rets, prof, overlay["cash"], rf, overlay["vol_mult"], caps, stress)

    stocks_ok = "single_stocks" in (prof.get("allowed_assets") or config.DEFAULT_ALLOWED)
    selection = eligible[:n] if stocks_ok else []
    w, info = build(selection)
    _risk(res, w, info, px, rets, prof, bench, rf)

    # ---- AI investment committee
    decision = None
    if ai:
        progress("Convening the AI investment committee: Growth PM, Risk Officer, Macro Strategist, CIO")
        try:
            packet = report.build_packet(_with_misc(res, eligible, n))
            out = llm.committee(packet)
            decision = out["decision"]
            res["memos"] = out["memos"]
            res["pre_adjust"] = {"weights": w.copy(), "stats": dict(res["stats"])}
            for adj in (decision.get("adjustments") or [])[:3]:
                t = str(adj.get("ticker", "")).upper()
                if t not in w.index or t not in config.STOCKS:
                    continue
                if adj.get("action") == "remove":
                    removed[t] = adj.get("reason", "")
                elif adj.get("action") == "cap" and adj.get("max_weight"):
                    caps[t] = float(np.clip(float(adj["max_weight"]), 0.01, prof["max_position"]))
            if removed or caps:
                progress("Applying the committee's changes and re-optimising")
                pool = [t for t in eligible if t not in removed]
                selection = pool[:n] if stocks_ok else []
                w, info = build(selection)
                _risk(res, w, info, px, rets, prof, bench, rf)
        except Exception as e:
            res["errors"].append(f"AI committee failed, rules based rationale used: {e}")
            traceback.print_exc()
            decision = None
    res["removed_by_committee"] = removed
    res["caps_by_committee"] = caps

    res = _with_misc(res, [t for t in eligible if t not in removed], n)
    packet = report.build_packet(res)
    fb = report.fallback_decision(packet)
    if decision is None:
        decision = fb
    else:  # make sure every final holding has an explanation (backfilled names get the rules based one)
        have = {h.get("ticker") for h in decision.get("holdings", [])}
        decision["holdings"] = [h for h in decision.get("holdings", []) if h.get("ticker") in w.index] + \
                               [dict(h, role=h["role"] + " (added after committee review)") for h in fb["holdings"] if h["ticker"] not in have]
        for k in ("headline", "portfolio_thesis", "suitability", "key_risks", "dissent"):
            decision.setdefault(k, fb[k])
    decision["holdings"] = [h for h in decision["holdings"] if h.get("ticker") in config.STOCKS]
    order = {t: i for i, t in enumerate(w.index)}
    decision["holdings"].sort(key=lambda h: order.get(h.get("ticker"), 99))
    res["decision"] = decision
    res["packet"] = packet
    res["verification"] = report.verify_decision(decision, packet)
    res["trades"] = A.trade_list(w, pm["price"], 0.0, config.INVESTMENT_AMOUNT, prof.get("fractional_shares", True))
    progress("Pricing the hedges: index options and currency")
    try:
        res["option_hedge"] = hedging.option_hedge(w, rets, px, prof, rf, snap.get("VIXCLS"))
    except Exception as e:
        res["option_hedge"] = None
        res["errors"].append(f"Option hedge could not be priced: {e}")
    rd = markets.readings(md.get("mk"), md["macro"])
    res["fx_hedge"] = hedging.fx_hedge(w, prof, rd.get("gbpusd_level"), rd.get("us_3m_pct"), snap.get("IR3TIB01GBM156N"))
    res["packet"] = report.build_packet(res)

    progress("Running the walk forward test (quarterly rebalances using only past data)")
    try:
        res["walk_forward"] = A.walk_forward(px, prof, rf, years=3)
    except Exception as e:
        res["walk_forward"] = {}
        res["errors"].append(f"Walk forward test failed: {e}")
    res["ai_used"] = ai and "memos" in res
    return res


def _risk(res, w, info, px, rets, prof, bench, rf):
    port = A.portfolio_returns(w, rets, 0.0, rf).iloc[-756:]
    b = rets[bench].reindex(port.index)
    res.update(weights=w, opt_info=info, port_daily=port, stats=A.perf_stats(port, rf, b),
               bench_stats=A.perf_stats(b, rf), bench_daily=b,
               spy_daily=rets["SPY"].reindex(port.index) if "SPY" in rets else None,
               risk_contrib=A.risk_contributions(w, rets), stress=A.stress_test(w, px, 0.0, config.INVESTMENT_AMOUNT, bench),
               stock_weights=w[[t for t in w.index if t in config.STOCKS]],
               defensive_weights=w[[t for t in w.index if t in config.DEFENSIVE]])
    long = A.portfolio_returns(w, rets, 0.0, rf).iloc[-1260:]
    long.attrs["drift"] = np.log1p(info["exp_return"]) / A.TRADING_DAYS
    mc = A.monte_carlo(long, prof["horizon_years"], config.INVESTMENT_AMOUNT)
    res["mc"] = mc
    res["p_breach"] = float((mc["worst_dd"] < -prof["max_drawdown"]).mean())


def _with_misc(res, eligible, n):
    comp, feat = res["composite"], res["features"]
    held = set(res["stock_weights"].index)
    near = [t for t in eligible if t not in held][:5]
    cutoff = comp[list(held)].min() if held else comp.max()
    z = res["z"]
    why = {}
    for t in comp.index:
        if t in held:
            continue
        if t in res["excluded"]:
            why[t] = res["excluded"][t]
            continue
        weakest = z.loc[t].sort_values().index[:2]
        why[t] = (f"Composite score {comp[t]:.2f} vs cutoff {cutoff:.2f}; weakest on {weakest[0]} "
                  f"({z.loc[t, weakest[0]]:+.2f} sd) and {weakest[1]} ({z.loc[t, weakest[1]]:+.2f} sd)")
    for t, r in res.get("removed_by_committee", {}).items():
        why[t] = f"Removed by the AI committee: {r}"
    res["near_misses"] = near
    res["why_not"] = why
    return res
