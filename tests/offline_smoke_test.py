"""Offline smoke test: runs the full pipeline on randomly generated prices and a fake LLM.
Only checks that the code runs end to end. Never used by the app itself.
Run: python -m tests.offline_smoke_test
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from advisor import config, data, llm, pipeline, report  # noqa: E402


def fake_market():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * 7)
    px = {}
    mkt = 0.0006 + 0.013 * rng.standard_normal(len(idx))
    crash = (idx >= "2022-01-03") & (idx <= "2022-10-14")
    mkt[crash] -= 0.0025                                   # a realistic 2022 style bear market
    mkt[(idx >= "2020-02-19") & (idx <= "2020-03-23")] -= 0.012
    for i, t in enumerate(config.ALL_TICKERS):
        if t in config.DEFENSIVE:
            r = 0.00015 + 0.004 * rng.standard_normal(len(idx)) * (0.05 if t == "SGOV" else 1) + (0.4 * mkt if t == "USMV" else 0)
        else:
            beta = 1.0 if t in ("QQQ", "SPY") else 0.8 + 0.8 * rng.random()
            r = beta * mkt + (0 if t in ("QQQ", "SPY") else 0.014 * rng.standard_normal(len(idx))) + 0.0002 * rng.standard_normal()
        s = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        if t == "ARM":
            s.iloc[:252 * 5] = np.nan  # recent IPO
        px[t] = s
    px = pd.DataFrame(px)
    fund = pd.DataFrame({t: {
        "shortName": f"{t} Corp", "sector": "Technology", "industry": "Software", "marketCap": 1e11 * (1 + rng.random()),
        "currentPrice": px[t].iloc[-1], "revenueGrowth": rng.normal(0.2, 0.15), "earningsGrowth": rng.normal(0.25, 0.3),
        "grossMargins": rng.uniform(0.3, 0.8), "operatingMargins": rng.uniform(-0.1, 0.45), "freeCashflow": 1e9 * rng.uniform(-1, 10),
        "totalRevenue": 1e10, "forwardPE": rng.uniform(15, 90), "priceToSalesTrailing12Months": rng.uniform(3, 30),
        "recommendationMean": rng.uniform(1.4, 2.8), "targetMeanPrice": px[t].iloc[-1] * rng.uniform(0.9, 1.3),
        "longBusinessSummary": f"{t} makes things for {config.TICKER_THEME[t]} " * 5} for t in config.STOCKS}).T
    macro = pd.DataFrame({"DGS10": 4.2, "DGS3MO": 4.0, "DGS5": 4.0, "VIXCLS": 17.0, "HYG": 80.0, "IEF": 95.0,
                          "DFII10": 1.9, "CPIAUCSL": 320.0}, index=idx[-700:])
    news = {t: [{"title": f"{t} beats estimates", "date": "2026-10-20"}] for t in config.STOCKS}
    lv = {"^TNX": 4.2, "^IRX": 4.0, "^TYX": 4.6, "^VIX": 16, "GBPUSD=X": 1.30, "EURUSD=X": 1.10, "DX-Y.NYB": 100}
    mk = pd.DataFrame({t: (lv[t] + 0.05 * np.cumsum(rng.standard_normal(len(idx[-756:])))) if t in lv
                       else 100 * np.exp(np.cumsum(0.0004 + 0.01 * rng.standard_normal(len(idx[-756:]))))
                       for t in config.MARKET_TICKERS}, index=idx[-756:])
    macro = macro.assign(UNRATE=4.1, BAMLH0A0HYM2=3.0, CPIAUCSL=np.linspace(300, 320, len(macro)))
    return {"px": px, "fund": fund, "macro": macro, "news": news, "analyst": pd.DataFrame(), "mk": mk, "sources": {"fake": {"status": "test"}}}


class FakeEmb:
    def __init__(self, inp):
        self.data = [type("D", (), {"embedding": list(np.random.default_rng(abs(hash(t)) % 999).random(16))}) for t in inp]


def fake_chat(system, user, max_tokens=0, json_mode=False):
    if "onboarding" in system:
        return json.dumps({"client_name": "Test", "horizon_years": 12, "risk_tolerance": 8, "max_drawdown": 0.35,
                           "themes": ["Artificial intelligence", "Cloud software"], "exclude_tickers": ["TSLA"]})
    if "Know Your Client meeting. You have" in system:
        return json.dumps({"observations": "You want growth but already own a lot of NVDA.", "questions": ["Will you keep the NVDA shares?", "Any big purchases soon?"]})
    if "finishing a Know Your Client" in system:
        return json.dumps({"risk_tolerance": 7, "max_drawdown": 0.28, "options_ok": True, "objective": "Grow wealth over 10 years.",
                           "summary": "Young professional.", "rationale": "Long horizon but a concentrated NVDA position."})
    if "explaining today's markets" in system:
        return json.dumps({"big_picture": "Rates are high and stocks are strong.", "sections": {"Stock markets": "Stocks rose."}})
    if "news analyst" in system:
        return json.dumps({t: {"score": 0.4, "driver": "beat estimates", "n": 1} for t in json.loads(user)})
    if "Chief Investment Officer" in system:
        pk = json.loads(user.split("DATA PACKET:\n")[1].split("\n\n===")[0])
        hs = pk["holdings"]
        return json.dumps({"headline": "Buy the leaders.", "portfolio_thesis": "Growth.", "suitability": "Fits.",
                           "adjustments": [{"ticker": hs[0]["ticker"], "action": "remove", "reason": "too volatile"},
                                           {"ticker": hs[1]["ticker"], "action": "cap", "max_weight": 0.05, "reason": "crowded"}],
                           "holdings": [{"ticker": h["ticker"], "role": "Growth engine",
                                         "why": f"Revenue growth {h.get('revenue_growth_pct')}% and margin 99.9%.",
                                         "key_risk": "x", "sell_trigger": "y"} for h in hs],
                           "key_risks": ["a"], "dissent": "none"})
    return "memo text"


if __name__ == "__main__":
    llm.available = lambda: True
    llm.chat = fake_chat
    llm.client = lambda: type("C", (), {"embeddings": type("E", (), {"create": staticmethod(lambda input, model: FakeEmb(input))})})()
    config.CACHE_DIR = config.CACHE_DIR.parent / "data_cache_test"
    config.CACHE_DIR.mkdir(exist_ok=True)
    md = fake_market()
    prof = llm.parse_profile("test")
    res = pipeline.run(md, prof, use_ai=True, progress=print)
    print("errors:", res["errors"])
    print(res["trades"].to_string())
    print("stats", {k: round(v, 3) for k, v in res["stats"].items()})
    print(res["stress"].to_string())
    print("removed", res["removed_by_committee"], "caps", res["caps_by_committee"])
    print("verification", res["verification"]["verified"], "/", res["verification"]["total"])
    print("walk forward", {k: round(v, 3) for k, v in res["walk_forward"].get("stats", {}).items()})
    print(report.memo_markdown(res)[:1500])
    for rt, dd in [(2, 0.15), (5, 0.25), (9, 0.40)]:
        p2 = dict(prof, risk_tolerance=rt, max_drawdown=dd)
        r2 = pipeline.run(md, p2, use_ai=False, progress=lambda m: None)
        oi = r2["opt_info"]
        print(f"rt={rt} dd={dd}: stocks {oi['stock_share']:.0%} ({len(r2['stock_weights'])} names), defensive {oi['defensive_share']:.0%}, "
              f"vol {oi['exp_vol']:.1%} <= {oi['target_vol']:.1%}, worst stress {r2['stress']['Portfolio'].min():.1%}, converged {oi['converged']}, binding {oi['binding']}")
        print("   ", r2["defensive_weights"].round(3).to_dict(), "| top stocks", list(r2["stock_weights"].index[:5]))
    print(r2["decision"]["headline"])
