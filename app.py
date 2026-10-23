"""Velocity: the AI growth analyst. Streamlit front end.
Run: streamlit run app.py
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px_
import plotly.graph_objects as go
import streamlit as st

from advisor import config, llm, pipeline, report

st.set_page_config(page_title="Velocity | AI Growth Analyst", page_icon="📈", layout="wide")

PURPLE, DEEP, YELLOW, GREEN, RED, GREY = "#5b2a9b", "#2a0f4c", "#f5d90a", "#1f9d6b", "#d64545", "#8a8a9a"
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1400px;}}
.hero {{background: linear-gradient(120deg, {DEEP} 0%, {PURPLE} 100%); color: white; padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;}}
.hero h1 {{color: white; margin: 0; font-size: 2.1rem;}}
.hero p {{color: #e6dcf5; margin: 4px 0 0 0; font-size: 1.02rem;}}
.hero b {{color: {YELLOW};}}
.kpi {{background: #f6f2fc; border-radius: 12px; padding: 12px 16px; border-left: 5px solid {PURPLE};}}
.kpi .v {{font-size: 1.55rem; font-weight: 700; color: {DEEP};}}
.kpi .l {{font-size: .82rem; color: #5a5a6a;}}
.pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; margin-right:6px; background:#ece4f8; color:{DEEP};}}
.ok {{background:#e2f5ec; color:#11643f;}} .bad {{background:#fbe5e5; color:#8a1f1f;}} .warn {{background:#fff6cc; color:#6b5a00;}}
.headline {{font-size: 1.25rem; font-weight: 650; color: {DEEP}; border-left: 6px solid {YELLOW}; padding: 6px 14px; background: #fbf9e6; border-radius: 6px;}}
@media (prefers-color-scheme: dark) {{
  .kpi {{background:#231a33;}} .kpi .v {{color:#f1eaff;}} .kpi .l {{color:#b9b0c9;}}
  .headline {{background:#2b2714; color:#fff6c2;}} .pill {{background:#33244d; color:#e9ddff;}}
}}
</style>""", unsafe_allow_html=True)


def pct(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def esc(text) -> str:
    """Escape $ so Streamlit markdown does not render dollar amounts as LaTeX."""
    return str(text or "").replace("\\$", "$").replace("$", "\\$")


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


ss = st.session_state
ss.setdefault("profile", dict(config.DEFAULT_PROFILE))
ss.setdefault("profile_text", config.DEFAULT_PROFILE_TEXT)
ss.setdefault("chat", [])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Connections")
    if not (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        key = st.text_input("Azure OpenAI key", type="password", help="Or set AZURE_OPENAI_API_KEY in a .env file")
        if key:
            llm.set_api_key(key)
    fh = st.text_input("Finnhub key (optional)", type="password", value=os.getenv("FINNHUB_API_KEY", ""))
    if fh:
        os.environ["FINNHUB_API_KEY"] = fh
    st.markdown(f'<span class="pill {"ok" if llm.available() else "bad"}">Azure OpenAI {"connected" if llm.available() else "no key"}</span>'
                f'<span class="pill ok">Yahoo Finance</span><span class="pill ok">FRED</span>'
                f'<span class="pill {"ok" if os.getenv("FINNHUB_API_KEY") else "warn"}">Finnhub {"on" if os.getenv("FINNHUB_API_KEY") else "off"}</span>',
                unsafe_allow_html=True)
    st.caption(f"Model: {config.CHAT_MODEL} · Embeddings: {config.EMBED_MODEL}")
    use_ai = st.toggle("Use AI investment committee", value=True)
    if st.button("Refresh market data", width="stretch"):
        for p in config.CACHE_DIR.glob("*.pkl"):
            if p.name != "embeddings.pkl":
                p.unlink()
        ss.pop("md", None)
        st.toast("Cache cleared; next build pulls live data.")
    st.divider()
    st.caption("Educational hackathon project. Not financial advice.")

st.markdown("""<div class="hero"><h1>Velocity · the AI growth analyst</h1>
<p>Turns a client's own words into a <b>$100,000</b> growth portfolio: live public data, quant scoring, an
<b>AI investment committee</b> that debates the picks, and every number in the explanation <b>traced back to source data</b>.</p></div>""",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ 01 client
st.subheader("01 · The client")
c1, c2 = st.columns([1.05, 1])
with c1:
    ss.profile_text = st.text_area("In the client's own words", ss.profile_text, height=190)
    if st.button("🧠 Read the brief with AI", width="stretch", disabled=not llm.available()):
        with st.spinner("Turning the brief into an investment policy..."):
            try:
                ss.profile = llm.parse_profile(ss.profile_text)
                ss.pop("res", None)
            except Exception as e:
                st.error(f"Could not parse the brief: {e}")
    if not llm.available():
        st.caption("Add an Azure OpenAI key in the sidebar to parse free text; defaults below are used meanwhile.")
with c2:
    p = ss.profile
    st.markdown(f"**{p['client_name']}** · high growth investor")
    st.markdown(f"*{p['objective']}*")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Horizon", f"{p['horizon_years']}y")
    kpi(k2, "Risk tolerance", f"{p['risk_tolerance']}/10")
    kpi(k3, "Max drawdown", pct(p["max_drawdown"], 0))
    kpi(k4, "Max position", pct(p["max_position"], 0))
    st.markdown(" ".join(f'<span class="pill">{t}</span>' for t in p["themes"]) +
                "".join(f'<span class="pill bad">no {x}</span>' for x in p.get("exclude_tickers", []) + p.get("exclude_sectors", [])),
                unsafe_allow_html=True)
    if p.get("assumptions"):
        st.caption("Assumed: " + "; ".join(map(str, p["assumptions"])))
    with st.expander("Adjust the investment policy"):
        a1, a2 = st.columns(2)
        p["horizon_years"] = a1.slider("Horizon (years)", 1, 30, int(p["horizon_years"]))
        p["risk_tolerance"] = a2.slider("Risk tolerance", 1, 10, int(p["risk_tolerance"]))
        p["max_drawdown"] = a1.slider("Max drawdown the client accepts", 0.10, 0.60, float(p["max_drawdown"]), 0.01)
        p["target_vol"] = config.target_vol_for(p["risk_tolerance"], p["max_drawdown"])
        a2.metric("Volatility ceiling (set from risk tolerance and drawdown)", pct(p["target_vol"]))
        p["max_position"] = a1.slider("Max single position", 0.05, 0.30, float(p["max_position"]), 0.01)
        p["n_holdings"] = a2.slider("Number of stocks", 6, 20, int(p["n_holdings"]))
        p["cash_buffer"] = a1.slider("Minimum T bill holding", 0.0, 0.15, float(p["cash_buffer"]), 0.01)
        pol = config.strategic_policy(p["risk_tolerance"])
        st.markdown("**Strategic allocation (core and satellite)**, set from risk tolerance:")
        if st.checkbox("Customise sleeve limits", value=bool(p.get("policy"))):
            b1, b2, b3 = st.columns(3)
            p["policy"] = {
                "min_bonds": b1.slider("Government bonds and cash, at least", 0.0, 0.8, float((p.get("policy") or pol)["min_bonds"]), 0.01),
                "min_index": b2.slider("Index funds, at least", 0.0, 0.8, float((p.get("policy") or pol)["min_index"]), 0.01),
                "max_single_stocks": b3.slider("Single stocks, at most", 0.0, 0.8, float((p.get("policy") or pol)["max_single_stocks"]), 0.01),
                "max_diversifiers": pol["max_diversifiers"]}
        else:
            p["policy"] = None
            st.caption(f"Government bonds and cash at least {pct(pol['min_bonds'], 0)} · index funds at least {pct(pol['min_index'], 0)} · "
                       f"single stocks at most {pct(pol['max_single_stocks'], 0)} · gold and min volatility at most {pct(pol['max_diversifiers'], 0)}")
        p["benchmark"] = a2.selectbox("Benchmark", ["QQQ", "SPY"], index=0 if p["benchmark"] == "QQQ" else 1)
        p["themes"] = st.multiselect("Themes", list(config.THEMES), default=p["themes"])
        p["exclude_tickers"] = [t.strip().upper() for t in st.text_input("Exclude tickers (comma separated)", ", ".join(p["exclude_tickers"])).split(",") if t.strip()]
        p["fractional_shares"] = st.checkbox("Fractional shares", value=p.get("fractional_shares", True))

if st.button("🚀 Build the $100,000 portfolio", type="primary", width="stretch"):
    with st.status("Running the analyst...", expanded=True) as status:
        try:
            if "md" not in ss:
                ss.md = pipeline.load_market_data(progress=lambda m: status.write(f"▸ {m}"))
            ss.res = pipeline.run(ss.md, dict(ss.profile), use_ai=use_ai, progress=lambda m: status.write(f"▸ {m}"))
            ss.chat = []
            status.update(label="Recommendation ready", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Run failed", state="error")
            st.exception(e)

if "res" not in ss:
    st.info("Describe the client, then build the portfolio. First run pulls about 7 years of data for 40 tickers (about a minute); later runs use the cache.")
    st.stop()

res = ss.res
prof = res["profile"]
d = res["decision"]
w = res["weights"]
feat = res["features"]
for e in res["errors"]:
    st.warning(e)

tabs = st.tabs(["04 · Recommendation", "03 · How the AI decided", "Risk check", "02 · The data", "💬 Ask the analyst"])

# ------------------------------------------------------------------ recommendation
with tabs[0]:
    st.markdown(f'<div class="headline">{esc(d.get("headline", ""))}</div>', unsafe_allow_html=True)
    st.write("")
    v = res["verification"]
    st_ = res["stats"]
    k = st.columns(6)
    sv = res["opt_info"]["sleeves"]
    kpi(k[0], "Index · stocks · bonds", f"{sv[config.SLEEVE_INDEX] * 100:.0f} · {sv[config.SLEEVE_STOCKS] * 100:.0f} · {sv[config.SLEEVE_BONDS] * 100:.0f}%")
    kpi(k[1], "Model expected return", pct(res["opt_info"]["exp_return"]))
    kpi(k[2], "Expected volatility", pct(res["opt_info"]["exp_vol"]))
    worst = res["stress"]["Portfolio"].min() if len(res["stress"]) else np.nan
    kpi(k[3], "Worst historical stress", pct(worst))
    kpi(k[4], f"Chance of a fall beyond {pct(prof['max_drawdown'], 0)} in {prof['horizon_years']}y", pct(res["p_breach"], 0))
    kpi(k[5], "Figures traced to data", f"{v['verified']}/{v['total']}")

    left, right = st.columns([1.1, 1])
    with left:
        t = res["trades"].copy()
        t["Sleeve"] = t["Ticker"].map(config.sleeve_of)
        t = t[t["Dollars"] > 0]
        colors = {config.SLEEVE_INDEX: "#7c3aed", config.SLEEVE_STOCKS: PURPLE, config.SLEEVE_BONDS: "#1f6f9d", config.SLEEVE_DIVERS: YELLOW, "(?)": GREY}
        fig = px_.treemap(t, path=["Sleeve", "Ticker"], values="Dollars", color="Sleeve", color_discrete_map=colors)
        fig.update_traces(texttemplate="<b>%{label}</b><br>$%{value:,.0f}", textfont_size=14)
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=420)
        st.plotly_chart(fig, width="stretch")
    with right:
        show = res["trades"].copy()
        show["Weight"] = show["Weight"].map(lambda x: pct(x))
        show["Dollars"] = show["Dollars"].map(lambda x: f"${x:,.0f}")
        show["Price"] = show["Price"].map(lambda x: "" if pd.isna(x) else f"${x:,.2f}")
        show["Shares"] = show["Shares"].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
        st.dataframe(show, hide_index=True, width="stretch", height=420)

    st.markdown("#### Why this portfolio")
    st.markdown(esc(d.get("portfolio_thesis", "")))
    st.markdown(f"**Suitability check.** {esc(d.get('suitability', ''))}")
    oi = res["opt_info"]
    lim = (f"Hard limits enforced by the optimiser: max {pct(prof['max_position'], 0)} per stock, volatility at most "
           f"{pct(oi['target_vol'])}, a loss smaller than {pct(prof['max_drawdown'], 0)} in every historical stress replay, "
           f"government bonds and cash at least {pct(oi['policy']['min_bonds'], 0)}, index funds at least {pct(oi['policy']['min_index'], 0)}, "
           f"single stocks at most {pct(oi['policy']['max_single_stocks'], 0)}.")
    if oi.get("binding"):
        lim += " The limits that shaped this portfolio: " + "; ".join(oi["binding"]) + "."
    st.info(lim)
    dw = res["defensive_weights"]
    why_sleeve = {config.SLEEVE_INDEX: "broad, cheap market growth so the outcome never rests on a handful of stocks",
                  config.SLEEVE_BONDS: "protection in equity sell offs and dry powder to rebalance into weakness",
                  config.SLEEVE_DIVERS: "assets that behave differently from tech stocks"}
    for sname, reason in why_sleeve.items():
        items = [(t, v) for t, v in dw.items() if config.sleeve_of(t) == sname]
        if items:
            st.markdown(f"**{sname}, {pct(sum(v for _, v in items))}:** " + ", ".join(
                f"{t} ({config.ETFS[t][0]}) {pct(v)}" for t, v in items) + f". Why: {reason}.")
    if res["removed_by_committee"] or res["caps_by_committee"]:
        chg = [f"removed **{t}** ({r})" for t, r in res["removed_by_committee"].items()] + \
              [f"capped **{t}** at {pct(c, 0)}" for t, c in res["caps_by_committee"].items()]
        st.markdown("🛡️ **Committee changes applied before sizing:** " + "; ".join(chg))

    st.markdown("#### Why each holding")
    cols = st.columns(2)
    for i, h in enumerate(d.get("holdings", [])):
        tk = h.get("ticker")
        if tk not in w.index:
            continue
        checks = v["per_holding"].get(tk, [])
        okn = sum(ok for _, ok in checks)
        badge = (f'<span class="pill ok">✓ {okn}/{len(checks)} figures verified</span>' if checks and okn == len(checks)
                 else f'<span class="pill warn">⚠ {okn}/{len(checks)} figures verified</span>' if checks else '<span class="pill">no figures cited</span>')
        src = '<span class="pill">rules based</span>' if h.get("source") == "rules" else '<span class="pill">AI committee</span>'
        with cols[i % 2].container(border=True):
            nm = feat.loc[tk, "name"] if tk in feat.index else tk
            st.markdown(f"**{tk}** · {nm} · <span class='pill'>{h.get('role', '')}</span> **{pct(w[tk])}** "
                        f"(\\${w[tk] * config.INVESTMENT_AMOUNT:,.0f})", unsafe_allow_html=True)
            st.markdown(esc(h.get("why", "")))
            st.markdown(f"⚠️ *Risk:* {esc(h.get('key_risk', ''))}  \n🚪 *Sell if:* {esc(h.get('sell_trigger', ''))}")
            st.markdown(badge + src, unsafe_allow_html=True)
            bad = [s for s, ok in checks if not ok]
            if bad:
                st.caption("Not found in source data: " + ", ".join(bad))
    if d.get("key_risks"):
        st.markdown("#### Portfolio risks")
        for r in d["key_risks"]:
            st.markdown(f"* {esc(r)}")
    m1, m2 = st.columns(2)
    m1.download_button("⬇️ Client memo (Markdown)", report.memo_markdown(res), file_name="velocity_recommendation.md", width="stretch")
    m2.download_button("⬇️ Trade list (CSV)", res["trades"].to_csv(index=False), file_name="velocity_trades.csv", width="stretch")

# ------------------------------------------------------------------ how the AI decided
with tabs[1]:
    st.markdown("#### The pipeline")
    st.markdown(
        "1. **Profile agent** turns the client's words into a structured investment policy.  \n"
        f"2. **Embeddings** ({config.EMBED_MODEL}) compare every company's business description with the client's themes.  \n"
        "3. **News agent** reads recent headlines per company and scores what they mean for fundamentals.  \n"
        "4. **Quant engine** z scores 40 stocks on 7 factors, applies the macro overlay, then a constrained optimiser sizes positions.  \n"
        "5. **AI investment committee**: Growth PM, Risk Officer and Macro Strategist each write a memo; the CIO decides and can remove or cap names, then the optimiser re runs.  \n"
        "6. **Verifier** checks every figure in the explanation against the source data.")
    st.markdown("#### Macro overlay applied")
    for n in res["overlay"]["notes"]:
        st.markdown(f"* {n}")
    fw = pd.Series(res["factor_weights"])
    fw = (fw / fw.sum()).sort_values()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Factor weights")
        st.plotly_chart(px_.bar(fw, orientation="h", color_discrete_sequence=[PURPLE]).update_layout(
            showlegend=False, height=330, margin=dict(t=5, b=5), xaxis_tickformat=".0%", xaxis_title="", yaxis_title=""), width="stretch")
    with c2:
        st.markdown("#### Factor scores (z) across the universe, best first")
        z = res["z"].loc[res["composite"].index]
        zz = z.copy()
        zz.index = [f"{'★ ' if t in w.index else ''}{t}" for t in z.index]
        fig = px_.imshow(zz.T, color_continuous_scale=["#d64545", "#f7f7f7", PURPLE], zmin=-2, zmax=2, aspect="auto")
        fig.update_layout(height=330, margin=dict(t=5, b=5), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    if res.get("theme_matrix") is not None:
        st.markdown("#### Theme fit from embeddings (cosine similarity of business description vs theme)")
        tm = res["theme_matrix"].loc[[t for t in res["composite"].index if t in res["theme_matrix"].index]]
        fig = px_.imshow(tm.T, color_continuous_scale="Purples", aspect="auto")
        fig.update_layout(height=360, margin=dict(t=5, b=5))
        st.plotly_chart(fig, width="stretch")

    if res.get("memos"):
        st.markdown("#### The AI investment committee")
        mc = st.columns(3)
        icons = {"Growth PM": "🚀", "Risk Officer": "🛡️", "Macro Strategist": "🌍"}
        for col, (name, memo) in zip(mc, res["memos"].items()):
            with col.container(border=True, height=420):
                st.markdown(f"**{icons.get(name, '')} {name}**")
                st.markdown(esc(memo))
        st.markdown(f"**🏛️ CIO decision.** {esc(d.get('dissent', ''))}")
        adj = d.get("adjustments") or []
        if adj:
            st.dataframe(pd.DataFrame(adj), hide_index=True, width="stretch")
        pre = res.get("pre_adjust")
        if pre and (res["removed_by_committee"] or res["caps_by_committee"]):
            comp_df = pd.DataFrame({"Before committee": pre["weights"], "After committee": w}).fillna(0)
            fig = go.Figure([go.Bar(name=c, x=comp_df.index, y=comp_df[c], marker_color=col)
                             for c, col in zip(comp_df.columns, [GREY, PURPLE])])
            fig.update_layout(barmode="group", height=320, yaxis_tickformat=".0%", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("AI committee not run (no Azure key or toggle off): explanations are rules based.")

    st.markdown("#### Why not ...?")
    others = [t for t in res["composite"].index if t not in w.index]
    pick = st.selectbox("Pick a stock we did not buy", others)
    if pick:
        st.write(res["why_not"].get(pick, ""))
        cmp_ = pd.DataFrame({pick: res["z"].loc[pick], "Average holding": res["z"].loc[list(res["stock_weights"].index)].mean()})
        st.plotly_chart(px_.bar(cmp_, barmode="group", color_discrete_sequence=[RED, PURPLE]).update_layout(
            height=300, margin=dict(t=10), yaxis_title="z score", xaxis_title=""), width="stretch")

    if res.get("sentiment"):
        st.markdown("#### News sentiment (AI read of recent headlines)")
        sdf = pd.DataFrame(res["sentiment"]).T.reset_index().rename(columns={"index": "Ticker"})
        st.dataframe(sdf.sort_values("score", ascending=False), hide_index=True, width="stretch")

# ------------------------------------------------------------------ risk
with tabs[2]:
    bench = res["bench"]
    st.markdown(f"#### Growth of $100,000: today's portfolio replayed over the last 3 years vs {bench}")
    st.caption("In sample illustration: it uses today's picks, so it flatters the strategy. The walk forward test below is the honest check.")
    g = pd.DataFrame({"Portfolio": res["port_daily"], bench: res["bench_daily"]})
    if res.get("spy_daily") is not None and bench != "SPY":
        g["SPY"] = res["spy_daily"]
    g = config.INVESTMENT_AMOUNT * (1 + g.fillna(0)).cumprod()
    st.plotly_chart(px_.line(g, color_discrete_sequence=[PURPLE, YELLOW, GREY]).update_layout(
        height=360, yaxis_tickprefix="$", margin=dict(t=10), legend_title="", yaxis_title="", xaxis_title=""), width="stretch")
    stats = pd.DataFrame({"Portfolio": res["stats"], bench: res["bench_stats"]})
    fmt = stats.copy().astype(object)
    for r in fmt.index:
        fmt.loc[r] = [f"{x:.2f}" if r in ("Sharpe", "Sortino", "Beta") else pct(x) for x in stats.loc[r]]
    st.dataframe(fmt, width="stretch")

    wf = res.get("walk_forward") or {}
    if wf:
        st.markdown("#### Walk forward test: the method, rebalanced quarterly using only data known at the time")
        st.caption("Uses only the price based signals (momentum, volatility) and the same optimiser, because historical fundamentals and news are not in the free data.")
        gg = config.INVESTMENT_AMOUNT * (1 + pd.DataFrame({"Velocity method": wf["strategy"], bench: wf["bench"]}).fillna(0)).cumprod()
        st.plotly_chart(px_.line(gg, color_discrete_sequence=[PURPLE, YELLOW]).update_layout(
            height=320, yaxis_tickprefix="$", margin=dict(t=10), legend_title="", yaxis_title="", xaxis_title=""), width="stretch")
        a, b = wf["stats"], wf["bench_stats"]
        q = st.columns(4)
        kpi(q[0], "Method CAGR", pct(a["CAGR"]))
        kpi(q[1], f"{bench} CAGR", pct(b["CAGR"]))
        kpi(q[2], "Method max drawdown", pct(a["Max drawdown"]))
        kpi(q[3], f"{bench} max drawdown", pct(b["Max drawdown"]))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Historical stress tests on today's portfolio")
        sdf = res["stress"]
        if len(sdf):
            fig = go.Figure([go.Bar(name="Portfolio", x=sdf["Scenario"], y=sdf["Portfolio"], marker_color=PURPLE),
                             go.Bar(name=bench, x=sdf["Scenario"], y=sdf[bench], marker_color=YELLOW)])
            fig.add_hline(y=-prof["max_drawdown"], line_dash="dash", line_color=RED, annotation_text="client limit")
            fig.update_layout(barmode="group", height=360, yaxis_tickformat=".0%", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Stocks without history in a window are proxied by beta × benchmark: " +
                       "; ".join(f"{r['Scenario'].split(' (')[0]}: {r['Proxied (no history)']}" for _, r in sdf.iterrows()))
    with c2:
        st.markdown(f"#### {prof['horizon_years']} year Monte Carlo (2,000 paths)")
        mcr = res["mc"]
        x = mcr["months"] / 12
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=mcr["pct"][4], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mcr["pct"][0], fill="tonexty", fillcolor="rgba(91,42,155,0.15)", line=dict(width=0), name="5th to 95th"))
        fig.add_trace(go.Scatter(x=x, y=mcr["pct"][3], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mcr["pct"][1], fill="tonexty", fillcolor="rgba(91,42,155,0.3)", line=dict(width=0), name="25th to 75th"))
        fig.add_trace(go.Scatter(x=x, y=mcr["pct"][2], line=dict(color=PURPLE, width=3), name="Median"))
        fig.update_layout(height=360, yaxis_tickprefix="$", xaxis_title="Years", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        term = mcr["terminal"]
        st.caption(f"Median \\${np.median(term):,.0f}; 5th percentile \\${np.percentile(term, 5):,.0f}; 95th \\${np.percentile(term, 95):,.0f}. "
                   f"Block bootstrap of the portfolio's own daily returns (5y), drift set to the model expected return "
                   f"{pct(res['opt_info']['exp_return'])}.")
    st.markdown("#### Where the risk comes from")
    rc = res["risk_contrib"].sort_values(ascending=False)
    rcdf = pd.DataFrame({"Weight": w.reindex(rc.index), "Share of portfolio risk": rc})
    st.plotly_chart(px_.bar(rcdf, barmode="group", color_discrete_sequence=[GREY, PURPLE]).update_layout(
        height=320, yaxis_tickformat=".0%", margin=dict(t=10), legend_title="", xaxis_title="", yaxis_title=""), width="stretch")

# ------------------------------------------------------------------ data
with tabs[3]:
    st.markdown("#### Public sources used in this run")
    src = pd.DataFrame(res["sources"]).T.reset_index().rename(columns={"index": "Source"})
    st.dataframe(src, hide_index=True, width="stretch")
    m = res["macro"]
    st.markdown(f"#### Macro dashboard (as of {m.get('as_of', 'n/a')})")
    q = st.columns(6)
    fmt_ = lambda x, suf="%": "n/a" if x is None else f"{x:.2f}{suf}"
    kpi(q[0], "10y Treasury", fmt_(m.get("DGS10")))
    kpi(q[1], "3m T bill (risk free)", fmt_(m.get("DGS3MO")))
    kpi(q[2], "10y real yield", fmt_(m.get("DFII10")))
    kpi(q[3], m.get("CURVE_LABEL", "Curve"), fmt_(m.get("CURVE"), "pp"))
    kpi(q[4], "VIX", fmt_(m.get("VIXCLS"), ""))
    kpi(q[5], "HYG vs IEF, 3m", fmt_(m.get("CREDIT_HYG_IEF_3M")))
    st.markdown("#### Universe: fundamentals and price data")
    cols = ["name", "theme_bucket", "market_cap_bn", "revenue_growth", "earnings_growth", "gross_margin", "operating_margin",
            "fcf_margin", "forward_pe", "ret_12_1", "vol_1y", "max_dd_3y", "analyst_rating", "target_upside", "theme_fit", "news_sentiment"]
    u = feat[[c for c in cols if c in feat]].copy()
    u.insert(0, "score", res["composite"])
    u = u.sort_values("score", ascending=False)
    pcts = ["revenue_growth", "earnings_growth", "gross_margin", "operating_margin", "fcf_margin", "ret_12_1", "vol_1y", "max_dd_3y", "target_upside"]
    st.dataframe(u.style.format({**{c: "{:.1%}" for c in pcts}, "market_cap_bn": "{:,.0f}", "forward_pe": "{:.1f}",
                                  "analyst_rating": "{:.2f}", "theme_fit": "{:.3f}", "news_sentiment": "{:+.2f}", "score": "{:+.2f}"}, na_rep="n/a"),
                 width="stretch", height=520)
    with st.expander("Recent headlines fed to the news agent"):
        for t in w.index:
            for n in ss.md["news"].get(t, [])[:4]:
                st.markdown(f"**{t}** · {n.get('date', '')} · [{n['title']}]({n.get('url') or '#'}) · {n.get('source') or ''}")
    with st.expander("Data packet sent to the AI committee (JSON)"):
        st.json(res["packet"], expanded=False)

# ------------------------------------------------------------------ chat
with tabs[4]:
    st.markdown("Ask anything about the recommendation. Answers are grounded in the same data packet the committee used.")
    if not ss.chat:
        st.caption("Try: *What happens to this portfolio if Nvidia falls 40%?* · *Why so little in Tesla?* · *How would you de-risk this for a 5 year horizon?*")
    for msg in ss.chat:
        st.chat_message(msg["role"]).markdown(esc(msg["content"]))
    q = st.chat_input("Ask the analyst", disabled=not llm.available())
    if q:
        ss.chat.append({"role": "user", "content": q})
        st.chat_message("user").markdown(esc(q))
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                ans = llm.ask(q, res["packet"], ss.chat)
            except Exception as e:
                ans = f"The analyst is unavailable right now: {e}"
            st.markdown(esc(ans))
        ss.chat.append({"role": "assistant", "content": ans})
