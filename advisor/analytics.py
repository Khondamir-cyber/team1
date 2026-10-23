"""Quant engine: factor scores, macro overlay, constrained optimiser, risk analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config

TRADING_DAYS = 252


# ---------------------------------------------------------------- price based metrics
def max_drawdown(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float((s / s.cummax() - 1).min())


def price_metrics(px: pd.DataFrame, bench: str = "QQQ") -> pd.DataFrame:
    rets = px.pct_change(fill_method=None)
    rows = {}
    b = rets[bench] if bench in rets else None
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < 130:
            continue
        r = rets[t].dropna()
        row = {
            "price": float(s.iloc[-1]),
            "ret_1m": float(s.iloc[-1] / s.iloc[-22] - 1),
            "ret_3m": float(s.iloc[-1] / s.iloc[-64] - 1),
            "ret_6m": float(s.iloc[-1] / s.iloc[-127] - 1),
            "ret_12_1": float(s.iloc[-22] / s.iloc[-253] - 1) if len(s) > 253 else np.nan,
            "ret_1y": float(s.iloc[-1] / s.iloc[-253] - 1) if len(s) > 253 else np.nan,
            "vol_1y": float(r.iloc[-252:].std() * np.sqrt(TRADING_DAYS)),
            "max_dd_3y": max_drawdown(s.iloc[-756:]),
            "off_52w_high": float(s.iloc[-1] / s.iloc[-252:].max() - 1),
            "history_years": round(len(s) / TRADING_DAYS, 1),
        }
        if b is not None and t != bench:
            j = pd.concat([r, b], axis=1).dropna().iloc[-504:]
            row["beta_qqq"] = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) if len(j) > 60 else np.nan
        rows[t] = row
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------- factor scores
def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.notna().sum() < 3:
        return pd.Series(0.0, index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    s = s.clip(lo, hi)
    sd = s.std()
    z = (s - s.mean()) / (sd if sd > 0 else 1)
    return z.clip(-clip, clip)


def build_features(fund: pd.DataFrame, pm: pd.DataFrame, analyst: pd.DataFrame | None,
                   sentiment: dict | None, theme: pd.Series | None) -> pd.DataFrame:
    idx = [t for t in config.STOCKS if t in pm.index]
    f = pd.DataFrame(index=idx)
    fund = fund.reindex(idx)
    num = lambda c: pd.to_numeric(fund.get(c), errors="coerce") if c in fund else pd.Series(np.nan, index=idx)
    f["name"] = fund.get("shortName")
    f["sector"] = fund.get("sector")
    f["industry"] = fund.get("industry")
    f["theme_bucket"] = [config.TICKER_THEME.get(t) for t in idx]
    f["market_cap_bn"] = num("marketCap") / 1e9
    f["revenue_growth"] = num("revenueGrowth")
    f["earnings_growth"] = num("earningsGrowth")
    f["gross_margin"] = num("grossMargins")
    f["operating_margin"] = num("operatingMargins")
    f["fcf_margin"] = num("freeCashflow") / num("totalRevenue")
    f["forward_pe"] = num("forwardPE")
    f["ps_ratio"] = num("priceToSalesTrailing12Months")
    f["peg"] = num("trailingPegRatio")
    f["analyst_rating"] = num("recommendationMean")  # 1 strong buy, 5 sell
    f["analyst_count"] = num("numberOfAnalystOpinions")
    price = pm["price"].reindex(idx)
    f["target_upside"] = num("targetMeanPrice") / price - 1
    for c in ["price", "ret_3m", "ret_6m", "ret_12_1", "ret_1y", "vol_1y", "max_dd_3y", "off_52w_high", "beta_qqq", "history_years"]:
        f[c] = pm[c].reindex(idx) if c in pm else np.nan
    if analyst is not None and not analyst.empty:
        f["finnhub_buy_share"] = analyst["buy_share"].reindex(idx)
    f["news_sentiment"] = pd.Series({t: (sentiment or {}).get(t, {}).get("score", np.nan) for t in idx})
    f["theme_fit"] = theme.reindex(idx) if theme is not None else np.nan
    return f


def factor_scores(f: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=f.index)
    z["growth"] = pd.concat([zscore(f["revenue_growth"]), 0.5 * zscore(f["earnings_growth"])], axis=1).mean(axis=1)
    z["momentum"] = pd.concat([zscore(f["ret_12_1"]), zscore(f["ret_6m"])], axis=1).mean(axis=1)
    z["quality"] = pd.concat([zscore(f["gross_margin"]), zscore(f["operating_margin"]), zscore(f["fcf_margin"])], axis=1).mean(axis=1)
    # valuation: cheaper forward P/E and P/S score higher; loss makers (no forward P/E) get a mild penalty
    fpe = f["forward_pe"].where(f["forward_pe"] > 0)
    val = pd.concat([-zscore(np.log(fpe)), -zscore(np.log(f["ps_ratio"].where(f["ps_ratio"] > 0)))], axis=1).mean(axis=1)
    z["valuation"] = val.where(fpe.notna(), val.fillna(0) - 0.5)
    analyst_parts = [-zscore(f["analyst_rating"]), zscore(f["target_upside"])]
    if "finnhub_buy_share" in f:
        analyst_parts.append(zscore(f["finnhub_buy_share"]))
    z["analyst"] = pd.concat(analyst_parts, axis=1).mean(axis=1)
    z["low_vol"] = pd.concat([-zscore(f["vol_1y"]), zscore(f["max_dd_3y"]), -zscore(f["beta_qqq"])], axis=1).mean(axis=1)
    z["sentiment"] = zscore(f["news_sentiment"]) if f["news_sentiment"].notna().sum() >= 3 else 0.0
    z["theme"] = zscore(f["theme_fit"]) if f["theme_fit"].notna().sum() >= 3 else 0.0
    return z.fillna(0.0)


def composite(z: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights).reindex(z.columns).fillna(0)
    w = w / w.sum()
    return (z * w).sum(axis=1)


# ---------------------------------------------------------------- macro overlay
def macro_overlay(snap: dict, profile: dict) -> dict:
    """Transparent rules turning the macro backdrop into portfolio settings."""
    notes, cash_add, val_mult, vol_mult = [], 0.0, 1.0, 1.0
    ry, d10 = snap.get("DFII10"), snap.get("DGS10_CHG_3M")
    if (ry is not None and ry > 2.0) or (d10 is not None and d10 > 0.40):
        val_mult = 1.75
        parts = ([f"10y real yield {ry:.2f}%"] if ry is not None else []) + ([f"10y yield {d10:+.2f}pp over 3m"] if d10 is not None else [])
        notes.append(f"Rates headwind ({', '.join(parts)}): valuation factor weight raised 75% to penalise the longest duration growth names.")
    else:
        parts = ([f"10y real yield {ry:.2f}%"] if ry is not None else []) + ([f"10y yield {d10:+.2f}pp over 3m"] if d10 is not None else [])
        notes.append(f"Rates backdrop neutral ({', '.join(parts) or 'no data'}): no extra valuation penalty.")
    hy, hy_med, cr = snap.get("BAMLH0A0HYM2"), snap.get("HY_SPREAD_1Y_MEDIAN"), snap.get("CREDIT_HYG_IEF_3M")
    if (hy is not None and hy_med is not None and hy > hy_med + 1.0) or (cr is not None and cr < -3):
        cash_add += 0.05
        vol_mult *= 0.9
        notes.append("Credit stress signal (high yield underperforming Treasuries): +5% T bill buffer, volatility budget cut 10%.")
    else:
        notes.append("Credit markets calm: no defensive cash added.")
    vix = snap.get("VIXCLS")
    if vix is not None and vix > 25:
        cash_add += 0.03
        notes.append(f"VIX at {vix:.1f} (above 25): +3% T bill buffer to fund buying on weakness.")
    elif vix is not None:
        notes.append(f"VIX at {vix:.1f}: volatility regime normal.")
    if snap.get("CURVE") is not None and snap["CURVE"] < 0:
        notes.append(f"Yield curve inverted ({snap['CURVE_LABEL']} {snap['CURVE']:.2f}pp): recession risk flagged to the committee.")
    cash = float(min(0.15, profile.get("cash_buffer", 0.03) + cash_add))
    return {"cash": cash, "valuation_mult": val_mult, "vol_mult": vol_mult, "notes": notes}


# ---------------------------------------------------------------- optimiser
def shrunk_cov(rets: pd.DataFrame, delta: float = 0.3, group: list[str] | None = None) -> pd.DataFrame:
    """Sample covariance shrunk toward constant correlation, applied within the stock group only
    (so stock to bond diversification is measured, not averaged away)."""
    s = rets.cov() * TRADING_DAYS
    cols = list(s.columns)
    sd = np.sqrt(np.diag(s))
    corr = rets.corr().fillna(0).values
    g = np.array([c in (group if group is not None else cols) for c in cols])
    gi = np.where(g)[0]
    target = s.values.copy()
    if len(gi) > 1:
        sub = corr[np.ix_(gi, gi)]
        avg_rho = (sub.sum() - len(gi)) / (len(gi) * (len(gi) - 1))
        target[np.ix_(gi, gi)] = avg_rho * np.outer(sd[gi], sd[gi])
        target[gi, gi] = sd[gi] ** 2
    return pd.DataFrame(delta * target + (1 - delta) * s.values, index=cols, columns=cols)


def scenario_matrix(tickers: list[str], px: pd.DataFrame, bench: str = "QQQ") -> pd.DataFrame:
    """Return of every asset in every historical stress window (beta x benchmark when it did not trade yet)."""
    rets = px.pct_change(fill_method=None)
    rows = {}
    for name, (a, b) in config.STRESS_SCENARIOS.items():
        win = px.loc[(px.index >= a) & (px.index <= b)]
        if len(win) < 2 or bench not in win or win[bench].isna().all():
            continue
        bret = float(win[bench].dropna().iloc[-1] / win[bench].dropna().iloc[0] - 1)
        row = {}
        for t in tickers:
            s = win[t].dropna() if t in win else pd.Series(dtype=float)
            if len(s) >= 2 and s.index[0] <= win.index[0] + pd.Timedelta(days=5):
                row[t] = float(s.iloc[-1] / s.iloc[0] - 1)
            else:
                j = pd.concat([rets[t], rets[bench]], axis=1).dropna().iloc[-504:]
                beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) if len(j) > 60 else 1.0
                row[t] = beta * bret
                row.setdefault("_proxied", []).append(t)
        rows[name] = row
    return rows


def optimise(stocks: list[str], score: pd.Series, rets: pd.DataFrame, profile: dict, cash: float,
             rf: float, vol_mult: float = 1.0, caps: dict | None = None, stress: dict | None = None) -> tuple[pd.Series, dict]:
    """Score tilted mean variance over the chosen stocks plus the defensive sleeve, subject to:
    weights sum to 100%, single stock cap, theme cap, volatility ceiling, T bill floor, and (when given) a loss
    no worse than the client's max drawdown in every historical stress window."""
    defensive = [t for t in config.ETFS if t in rets.columns and rets[t].iloc[-504:].notna().sum() > 120]
    tickers = list(stocks) + defensive
    r = rets[tickers].iloc[-504:].fillna(0)
    cov = shrunk_cov(r, group=list(stocks)).values
    s = score.reindex(stocks).clip(-2, 2).fillna(0).values
    mu = np.concatenate([rf + 0.06 + 0.04 * s,                      # equity premium 6%, +4% per 1 sd of composite score
                         [rf + config.ETFS[t][2] for t in defensive]])
    rt = profile.get("risk_tolerance", 8)
    # Risk is controlled mainly by the client's hard limits (volatility ceiling, stress losses, caps):
    # maximise expected return inside that risk budget, with a mild diversification penalty.
    risk_aversion = 0.5 + 0.15 * (10 - rt)
    target_vol = profile.get("target_vol", 0.28) * vol_mult
    max_dd = profile.get("max_drawdown", 0.35)
    n_s = len(stocks)
    pol0 = {**config.strategic_policy(rt), **(profile.get("policy") or {})}
    per_stock = min(profile.get("max_position", 0.15), pol0["max_single_stocks"] / min(5, max(n_s, 1)))  # at least 5 names
    bounds = [(0.0, min(per_stock, (caps or {}).get(t, 1.0))) for t in stocks]
    bounds += [(cash if t == config.CASH_PROXY else 0.0, config.ETFS[t][3]) for t in defensive]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: target_vol ** 2 - w @ cov @ w}]
    themes = pd.Series([config.TICKER_THEME.get(t, "Other") for t in stocks])
    for th in themes.unique():
        m = np.concatenate([(themes == th).values.astype(float), np.zeros(len(defensive))])
        cons.append({"type": "ineq", "fun": lambda w, m=m: profile.get("max_theme_weight", 0.4) - m @ w})
    # strategic policy: core index floor, government bond floor, single stock and diversifier ceilings
    pol = {**config.strategic_policy(rt), **(profile.get("policy") or {})}
    sl = np.array([config.sleeve_of(t) for t in tickers])
    m_st, m_ix = (sl == config.SLEEVE_STOCKS).astype(float), (sl == config.SLEEVE_INDEX).astype(float)
    m_bd, m_dv = (sl == config.SLEEVE_BONDS).astype(float), (sl == config.SLEEVE_DIVERS).astype(float)
    cons += [{"type": "ineq", "fun": lambda w: pol["max_single_stocks"] - m_st @ w},
             {"type": "ineq", "fun": lambda w: m_ix @ w - pol["min_index"]},
             {"type": "ineq", "fun": lambda w: m_bd @ w - pol["min_bonds"]},
             {"type": "ineq", "fun": lambda w: pol["max_diversifiers"] - m_dv @ w}]
    scen = {}
    for name, row in (stress or {}).items():
        vec = np.array([row.get(t, 0.0) for t in tickers])
        scen[name] = vec
        cons.append({"type": "ineq", "fun": lambda w, v=vec: v @ w + max_dd * 0.98})
    obj = lambda w: -(w @ mu - 0.5 * risk_aversion * (w @ cov @ w))

    def feasible(w):
        return all(c["fun"](w) >= -1e-4 for c in cons if c["type"] == "ineq") and abs(w.sum() - 1) < 1e-4

    best = None
    def policy_start(stock_share):
        w = np.zeros(len(tickers))
        w[:n_s] = stock_share / max(n_s, 1)
        ix = [i for i, t in enumerate(tickers) if t in ("VOO", "QQQ")] or [i for i in range(len(tickers)) if m_ix[i]]
        w[ix] = (1 - stock_share - pol["min_bonds"]) / max(len(ix), 1)
        cash_j = tickers.index(config.CASH_PROXY) if config.CASH_PROXY in tickers else None
        if cash_j is not None:
            w[cash_j] += pol["min_bonds"]
        return w
    starts = [policy_start(pol["max_single_stocks"] * 0.8), policy_start(pol["max_single_stocks"] * 0.3)]
    for w0 in starts:
        res = minimize(obj, np.clip(w0, [b[0] for b in bounds], [b[1] for b in bounds]), method="SLSQP",
                       bounds=bounds, constraints=cons, options={"maxiter": 800, "ftol": 1e-10})
        if feasible(res.x) and (best is None or res.fun < best.fun):
            best = res
    info = {"target_vol": target_vol, "max_dd": max_dd}
    if best is not None:
        wv = np.clip(best.x, 0, None)
        info["converged"] = True
    else:  # guaranteed fallback: shrink the risky part of the policy portfolio until every limit holds, rest in T bills
        info["converged"] = False
        base = policy_start(pol["max_single_stocks"] * 0.5)
        cash_i = tickers.index(config.CASH_PROXY) if config.CASH_PROXY in tickers else None
        wv = base
        for k in np.linspace(1, 0, 41):
            cand = base.copy()
            cand[(m_st + m_ix + m_dv) > 0] *= k
            if cash_i is not None:
                cand[cash_i] += 1 - cand.sum()
            if feasible(cand):
                wv = cand
                break
        info["note"] = "Optimiser fell back to a scaled equal weight portfolio to respect every client limit."
    wv = wv / wv.sum()
    w = pd.Series(wv, index=tickers)
    info["exp_return"] = float(wv @ mu)
    info["exp_vol"] = float(np.sqrt(wv @ cov @ wv))
    info["mu"] = dict(zip(tickers, mu))
    info["stock_share"] = float(wv[:n_s].sum())
    info["defensive_share"] = float(wv[n_s:].sum())
    info["sleeves"] = {name: float(m @ wv) for name, m in [(config.SLEEVE_INDEX, m_ix), (config.SLEEVE_STOCKS, m_st),
                                                           (config.SLEEVE_BONDS, m_bd), (config.SLEEVE_DIVERS, m_dv)]}
    info["policy"] = pol
    info["stress_returns"] = {k: float(v @ wv) for k, v in scen.items()}
    binding = []
    if info["exp_vol"] > target_vol - 0.005:
        binding.append(f"volatility ceiling of {target_vol * 100:.1f}%")
    for k, v in info["stress_returns"].items():
        if v < -max_dd * 0.98 + 0.006:
            binding.append(f"max loss of {max_dd * 100:.0f}% in the {k.split(' (')[0]} replay")
    sv = info["sleeves"]
    if sv[config.SLEEVE_BONDS] < pol["min_bonds"] + 0.005:
        binding.append(f"government bond floor of {pol['min_bonds'] * 100:.0f}%")
    if sv[config.SLEEVE_STOCKS] > pol["max_single_stocks"] - 0.005:
        binding.append(f"single stock satellite cap of {pol['max_single_stocks'] * 100:.0f}%")
    if sv[config.SLEEVE_INDEX] < pol["min_index"] + 0.005:
        binding.append(f"index fund core floor of {pol['min_index'] * 100:.0f}%")
    info["binding"] = binding
    w = w[w > 0.004]
    w = (w / w.sum()).sort_values(ascending=False) if w.sum() else w
    return w, info


def risk_contributions(w: pd.Series, rets: pd.DataFrame) -> pd.Series:
    t = list(w.index)
    cov = shrunk_cov(rets[t].iloc[-504:].fillna(0), group=[x for x in t if x in config.STOCKS]).values
    wv = w.values
    port_var = wv @ cov @ wv
    return pd.Series(wv * (cov @ wv) / port_var, index=t)


# ---------------------------------------------------------------- portfolio analytics
def portfolio_returns(w: pd.Series, rets: pd.DataFrame, cash: float, rf: float, cash_rets: pd.Series | None = None) -> pd.Series:
    r = rets[list(w.index)]
    pr = (r.fillna(0) * w.values).sum(axis=1)
    cr = cash_rets.reindex(pr.index).fillna(rf / TRADING_DAYS) if cash_rets is not None else rf / TRADING_DAYS
    return pr + cash * cr


def perf_stats(r: pd.Series, rf: float, bench: pd.Series | None = None) -> dict:
    r = r.dropna()
    if r.empty:
        return {}
    years = len(r) / TRADING_DAYS
    growth = (1 + r).prod()
    cagr = growth ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    out = {"CAGR": cagr, "Volatility": vol, "Sharpe": (cagr - rf) / vol if vol else np.nan,
           "Max drawdown": max_drawdown((1 + r).cumprod()), "Total return": growth - 1}
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    out["Sortino"] = (cagr - rf) / downside if downside else np.nan
    if bench is not None:
        j = pd.concat([r, bench], axis=1).dropna()
        out["Beta"] = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())
        out["Tracking error"] = float((j.iloc[:, 0] - j.iloc[:, 1]).std() * np.sqrt(TRADING_DAYS))
    return out


def stress_test(w: pd.Series, px: pd.DataFrame, cash: float, amount: float, bench: str = "QQQ") -> pd.DataFrame:
    """Replay historical crises on today's weights. Stocks without history in a window use beta x benchmark."""
    rows = []
    betas = {}
    rets = px.pct_change(fill_method=None)
    for t in w.index:
        j = pd.concat([rets[t], rets[bench]], axis=1).dropna().iloc[-504:]
        betas[t] = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) if len(j) > 60 else 1.0
    for name, (a, b) in config.STRESS_SCENARIOS.items():
        win = px.loc[(px.index >= a) & (px.index <= b)]
        if len(win) < 2 or bench not in win or win[bench].isna().all():
            continue
        bret = float(win[bench].dropna().iloc[-1] / win[bench].dropna().iloc[0] - 1)
        port, proxied = 0.0, []
        for t, wt in w.items():
            s = win[t].dropna() if t in win else pd.Series(dtype=float)
            if len(s) >= 2 and s.index[0] <= win.index[0] + pd.Timedelta(days=5):
                rt = float(s.iloc[-1] / s.iloc[0] - 1)
            else:
                rt = betas[t] * bret
                proxied.append(t)
            port += wt * rt
        rows.append({"Scenario": name, "Window": f"{a} to {b}", "Portfolio": port, bench: bret,
                     "P&L on $100k": port * amount, "Proxied (no history)": ", ".join(proxied) or "none"})
    return pd.DataFrame(rows)


def monte_carlo(daily: pd.Series, years: int, amount: float, n_paths: int = 2000, block: int = 21, seed: int = 7):
    """Block bootstrap of the portfolio's own daily returns (keeps fat tails and volatility clustering)."""
    rng = np.random.default_rng(seed)
    r = daily.dropna().values
    steps = years * TRADING_DAYS
    n_blocks = int(np.ceil(steps / block))
    starts = rng.integers(0, len(r) - block, size=(n_paths, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_paths, -1)[:, :steps]
    # de-mean then add a long run drift so results are not just a replay of a hot bull market
    path_r = r[idx] - r.mean() + daily.attrs.get("drift", r.mean())
    wealth = amount * np.cumprod(1 + path_r, axis=1)
    monthly = wealth[:, ::21]
    peak = np.maximum.accumulate(wealth, axis=1)
    worst_dd = (wealth / peak - 1).min(axis=1)
    pct = np.percentile(monthly, [5, 25, 50, 75, 95], axis=0)
    return {"months": np.arange(monthly.shape[1]), "pct": pct, "terminal": wealth[:, -1], "worst_dd": worst_dd}


def walk_forward(px: pd.DataFrame, profile: dict, rf: float, years: int = 3, bench: str = "QQQ") -> dict:
    """Honest test of the price based part of the method: quarterly rebalance using only data known at the time."""
    rets = px.pct_change(fill_method=None)
    stocks = [t for t in config.STOCKS if t in px.columns]
    start = px.index[-1] - pd.DateOffset(years=years)
    dates = pd.date_range(start, px.index[-1], freq="QS")
    chunks, holdings = [], []
    for i, d in enumerate(dates):
        hist = px.loc[:d]
        avail = [t for t in stocks if hist[t].dropna().shape[0] > 260]
        if len(avail) < 8:
            continue
        mom = pd.Series({t: hist[t].dropna().iloc[-22] / hist[t].dropna().iloc[-253] - 1 for t in avail})
        vol = rets.loc[:d, avail].iloc[-252:].std()
        sc = zscore(mom) - 0.5 * zscore(vol)
        pick = list(sc.sort_values(ascending=False).index[: profile.get("n_holdings", 12)])
        w, _ = optimise(pick, sc, rets.loc[:d], profile, cash=profile.get("cash_buffer", 0.03), rf=rf)  # no stress replay: it would look ahead
        end = dates[i + 1] if i + 1 < len(dates) else px.index[-1] + pd.Timedelta(days=1)
        seg = rets.loc[(rets.index > d) & (rets.index < end)]
        chunks.append(portfolio_returns(w, seg, 0.0, rf))
        st_ = [t for t in w.index if t in config.STOCKS]
        holdings.append({"date": d.strftime("%b %Y"), "stocks": ", ".join(st_), "defensive_pct": round(100 * w.drop(st_).sum(), 1)})
    if not chunks:
        return {}
    strat = pd.concat(chunks)
    b = rets[bench].reindex(strat.index)
    return {"strategy": strat, "bench": b, "stats": perf_stats(strat, rf, b), "bench_stats": perf_stats(b, rf), "log": holdings}


def trade_list(w: pd.Series, prices: pd.Series, cash: float, amount: float, fractional: bool) -> pd.DataFrame:
    rows = []
    for t, wt in w.items():
        dollars = wt * amount
        p = float(prices.get(t, np.nan))
        shares = dollars / p if fractional else np.floor(dollars / p)
        rows.append({"Ticker": t, "Weight": wt, "Dollars": shares * p, "Price": p, "Shares": shares})
    df = pd.DataFrame(rows)
    df.insert(1, "Holding", [config.ETFS[t][0] if t in config.ETFS else "" for t in df["Ticker"]])
    df.insert(2, "Sleeve", [config.sleeve_of(t) for t in df["Ticker"]])
    invested = df["Dollars"].sum()
    left = amount - invested
    if abs(left) > 0.5:
        df = pd.concat([df, pd.DataFrame([{"Ticker": "Uninvested cash (rounding)", "Weight": left / amount, "Dollars": left}])], ignore_index=True)
    return df
