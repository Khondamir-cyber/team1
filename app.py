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

from advisor import config, kyc, llm, pipeline, report

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

st.markdown("""<div class="hero"><h1>Velocity · your AI private banker</h1>
<p>Know the client first. Read the markets. Show the evidence. Then build a hedged, multi asset portfolio for
<b>$100,000</b>, debated by an <b>AI investment committee</b> with every number <b>traced back to source data</b>.</p></div>""",
            unsafe_allow_html=True)

STEPS = ["1 · Know your client", "2 · The markets today", "3 · What the data tells us", "4 · Your portfolio"]
ss.setdefault("step", 1)
ss.setdefault("answers", dict(kyc.DEFAULT_ANSWERS))
nav = st.columns(4)
for i, lab in enumerate(STEPS, start=1):
    done = i < ss.step
    nav[i - 1].markdown(f'<div class="kpi" style="opacity:{1 if i <= ss.step else .45}; border-left-color:{YELLOW if i == ss.step else PURPLE}">'
                        f'<div class="l">{"✓ " if done else ""}{lab}</div></div>', unsafe_allow_html=True)
st.write("")


def goto(n):
    ss.step = n
    st.rerun()


def ensure_data(status):
    if "md" not in ss:
        ss.md = pipeline.load_market_data(progress=lambda m: status.write(f"▸ {m}"))


# ------------------------------------------------------------------ STEP 1: KYC
if ss.step == 1:
    a = ss.answers
    st.subheader("Know your client")
    st.caption("A private banker never starts with products. These questions cover goals, money needs, wealth, behaviour in a crash and preferences.")
    with st.form("kyc"):
        st.markdown("##### About you")
        c = st.columns(4)
        a["name"] = c[0].text_input("Name", a["name"])
        a["age"] = c[1].number_input("Age", 18, 100, int(a["age"]))
        a["occupation"] = c[2].text_input("Occupation", a["occupation"])
        a["residence"] = c[3].selectbox("Tax residence", ["United Kingdom", "United States", "European Union", "Other"],
                                        index=["United Kingdom", "United States", "European Union", "Other"].index(a["residence"]))
        c = st.columns(4)
        a["base_currency"] = c[0].radio("You spend and measure wealth in", ["GBP", "USD"], index=0 if a["base_currency"] == "GBP" else 1, horizontal=True)
        a["dependants"] = c[1].selectbox("Dependants", ["None", "Partner", "Children", "Partner and children"],
                                         index=["None", "Partner", "Children", "Partner and children"].index(a["dependants"]))
        st.markdown("##### Your goals")
        c = st.columns([1.2, 1, 1.4])
        a["objective"] = c[0].selectbox("Main objective for this money", kyc.OBJECTIVES, index=kyc.OBJECTIVES.index(a["objective"]))
        a["horizon_years"] = c[1].slider("Years until you need most of it", 1, 30, int(a["horizon_years"]))
        a["goal_text"] = c[2].text_input("What is the money ultimately for?", a["goal_text"])
        c = st.columns(3)
        a["withdrawals"] = c[0].selectbox("Withdrawals", ["No withdrawals planned", "Occasional lump sums", "Regular income needed"],
                                          index=["No withdrawals planned", "Occasional lump sums", "Regular income needed"].index(a["withdrawals"]))
        a["income_pct"] = c[1].slider("If regular income: % a year", 0.0, 0.08, float(a["income_pct"]), 0.005)
        a["liquidity_2y"] = c[2].number_input("Amount you may need within 2 years ($)", 0, 100000, int(a["liquidity_2y"]), 1000)
        st.markdown("##### Your financial situation")
        c = st.columns(4)
        a["wealth_share"] = c[0].selectbox("This $100k as a share of your total wealth", list(kyc.WEALTH_SHARE), index=list(kyc.WEALTH_SHARE).index(a["wealth_share"]))
        a["emergency_fund"] = c[1].selectbox("Emergency savings outside this money", ["Yes, 6 months or more", "Yes, under 6 months", "No"],
                                             index=["Yes, 6 months or more", "Yes, under 6 months", "No"].index(a["emergency_fund"]))
        a["income_stability"] = c[2].selectbox("Income", ["Stable salary", "Variable or self employed", "Retired"],
                                               index=["Stable salary", "Variable or self employed", "Retired"].index(a["income_stability"]))
        a["concentrated"] = c[3].radio("Hold a large position in one company (e.g. employer shares)?", ["Yes", "No"],
                                       index=0 if a["concentrated"] == "Yes" else 1, horizontal=True)
        c = st.columns(4)
        a["concentrated_ticker"] = c[0].text_input("Which company (ticker)?", a["concentrated_ticker"])
        a["concentrated_value"] = c[1].number_input("Roughly worth ($)", 0, 10_000_000, int(a["concentrated_value"]), 5000)
        st.markdown("##### You and risk")
        c = st.columns(4)
        a["risk_words"] = c[0].select_slider("How would you describe yourself as an investor?", list(kyc.RISK_WORDS), value=a["risk_words"])
        a["reaction"] = c[1].selectbox("Your portfolio falls 20% in a month. You...", list(kyc.REACTION), index=list(kyc.REACTION).index(a["reaction"]))
        a["max_loss"] = c[2].slider("Largest fall you could live with before losing sleep", 0.05, 0.60, float(a["max_loss"]), 0.01, format="%.2f")
        a["experience"] = c[3].selectbox("Investing experience", list(kyc.EXPERIENCE), index=list(kyc.EXPERIENCE).index(a["experience"]))
        a["options_knowledge"] = st.radio("Have you used options (puts, calls) before?", ["Yes", "A little", "No"],
                                          index=["Yes", "A little", "No"].index(a["options_knowledge"]), horizontal=True)
        st.markdown("##### Your preferences")
        c = st.columns(2)
        a["themes"] = c[0].multiselect("Themes you believe in", list(config.THEMES), default=a["themes"])
        a["exclusions"] = c[1].multiselect("Never invest in", list(kyc.EXCLUSIONS), default=a["exclusions"])
        a["allowed_assets"] = st.multiselect("Assets you are comfortable with (government bonds and cash are always allowed)",
                                             list(config.ASSET_CLASSES), default=a["allowed_assets"], format_func=lambda k: config.ASSET_CLASSES[k])
        c = st.columns([1, 2])
        a["esg"] = c[0].selectbox("Sustainability (ESG)", ["Not important", "Somewhat important", "Very important"],
                                  index=["Not important", "Somewhat important", "Very important"].index(a["esg"]))
        a["notes"] = c[1].text_input("Anything else your banker should know?", a["notes"])
        submitted = st.form_submit_button("Review my answers", type="primary", width="stretch")
    if a["esg"] == "Very important" and "Fossil fuels" not in a["exclusions"]:
        a["exclusions"] = a["exclusions"] + ["Fossil fuels"]
    if submitted:
        ss.score = kyc.score(a)
        ss.followups = None
        if llm.available():
            with st.spinner("Your banker is reading your answers..."):
                try:
                    ss.followups = llm.kyc_followups(a, ss.score)
                except Exception as e:
                    st.warning(f"AI banker unavailable, using standard follow ups: {e}")
        if not ss.followups:
            ss.followups = {"observations": " ".join(ss.score["contradictions"] + ss.score["capacity_notes"]) or "Your answers are consistent.",
                            "questions": kyc.fallback_questions(a, ss.score)}
    if ss.get("score"):
        sc = ss.score
        st.markdown("#### Your risk profile so far")
        c = st.columns([1, 1.3])
        with c[0]:
            fig = go.Figure(go.Bar(x=[sc["willingness"], sc["ability"]], y=["Willingness to take risk", "Ability to take risk"],
                                   orientation="h", marker_color=[PURPLE, YELLOW], text=[f"{sc['willingness']}/10", f"{sc['ability']}/10"]))
            fig.update_layout(height=200, xaxis_range=[0, 10], margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"We invest at the lower of the two: **risk score {sc['risk_tolerance']}/10** before follow ups.")
        with c[1]:
            st.markdown(f"**🎩 Your banker:** {esc(ss.followups.get('observations', ''))}")
            qa = []
            for i, q in enumerate(ss.followups.get("questions", [])):
                ans = st.text_input(q, key=f"fu_{i}")
                qa.append({"question": q, "answer": ans})
        if st.button("Confirm my profile", type="primary", width="stretch"):
            final = None
            if llm.available():
                with st.spinner("Setting your final risk profile..."):
                    try:
                        final = llm.kyc_finalise(a, sc, qa)
                    except Exception as e:
                        st.warning(f"AI unavailable, rules based profile used: {e}")
            ss.profile = kyc.build_profile(a, sc, final)
            ss.pop("res", None)
            ss.pop("brief", None)
        if ss.get("profile", {}).get("kyc") is not None:
            p = ss.profile
            st.success(f"**{p['client_name']}: risk score {p['risk_tolerance']}/10, max acceptable fall {pct(p['max_drawdown'], 0)}, "
                       f"volatility ceiling {pct(p['target_vol'])}.** {esc(p.get('risk_rationale', ''))}")
            pol = config.strategic_policy(p["risk_tolerance"])
            st.caption(f"Policy: government bonds and cash at least {pct(pol['min_bonds'], 0)}, index funds at least {pct(pol['min_index'], 0)}, "
                       f"single stocks at most {pct(pol['max_single_stocks'], 0)}, at least {pct(p['cash_buffer'], 0)} in T bills for liquidity"
                       + (f", no {', '.join(p['exclude_tickers'])} (already held)" if p["exclude_tickers"] else "") + ".")
            if st.button("Next: the markets today →", type="primary", width="stretch"):
                with st.status("Pulling live market data...", expanded=True) as status:
                    try:
                        ensure_data(status)
                        ss.brief = pipeline.market_brief(ss.md, ss.profile, use_ai)
                        status.update(label="Market data ready", state="complete")
                    except Exception as e:
                        status.update(label="Data load failed", state="error")
                        st.exception(e)
                        st.stop()
                goto(2)
    st.stop()

# ------------------------------------------------------------------ STEP 2: markets
if ss.step == 2:
    b, mk, macro = ss.brief, ss.md.get("mk"), ss.md["macro"]
    r, story = b["readings"], b.get("story") or {}
    if b.get("error"):
        st.warning(b["error"])
    st.subheader("The markets today")
    if story.get("big_picture"):
        st.markdown(f'<div class="headline">{esc(story["big_picture"])}</div>', unsafe_allow_html=True)
    notes = {**b["notes"], **(story.get("sections") or {})}

    def last_year(df, cols, rebase=False):
        cols = [c for c in cols if df is not None and c in df]
        if not cols:
            return None
        x = df[cols].loc[df.index >= df.index.max() - pd.Timedelta(days=365)].ffill().dropna(how="all")
        if rebase:
            x = x / x.bfill().iloc[0] * 100
        return x.rename(columns=config.MARKET_TICKERS)

    def section(title, df, ytitle, note_key=None, fmt=None):
        st.markdown(f"#### {title}")
        cc = st.columns([1.6, 1])
        if df is not None and not df.empty:
            fig = px_.line(df, color_discrete_sequence=[PURPLE, YELLOW, "#1f6f9d", RED, GREEN, GREY])
            fig.update_layout(height=290, margin=dict(t=10, b=10), legend_title="", yaxis_title=ytitle, xaxis_title="")
            cc[0].plotly_chart(fig, width="stretch")
        else:
            cc[0].info("Data unavailable right now.")
        cc[1].markdown(esc(notes.get(note_key or title, "")))
        return cc[1]

    side = section("Interest rates and inflation", last_year(mk, ["^IRX", "^TNX", "^TYX"]), "Yield, %")
    if macro is not None and "CPIAUCSL" in macro:
        cpi = (macro["CPIAUCSL"].dropna().pct_change(12) * 100).dropna().iloc[-36:]
        if len(cpi):
            side.plotly_chart(px_.line(cpi, color_discrete_sequence=[RED]).update_layout(
                height=160, margin=dict(t=25, b=5), showlegend=False, title="US inflation, % a year (FRED)", yaxis_title="", xaxis_title=""), width="stretch")
    if macro is not None and "UNRATE" in macro:
        section("Jobs", macro[["UNRATE"]].dropna().iloc[-36:].rename(columns={"UNRATE": "US unemployment rate (FRED)"}), "%")
    side = section("Stock markets", last_year(mk, ["^GSPC", "^NDX", "^FTSE", "^STOXX50E", "^N225", "EEM"], rebase=True), "Rebased to 100")
    vx = last_year(mk, ["^VIX"])
    if vx is not None:
        side.plotly_chart(px_.line(vx, color_discrete_sequence=[RED]).update_layout(
            height=160, margin=dict(t=25, b=5), showlegend=False, title="VIX fear gauge", yaxis_title="", xaxis_title=""), width="stretch")
    side = section("Bonds and credit", last_year(mk, ["IEF", "LQD", "HYG"], rebase=True), "Rebased to 100")
    if macro is not None and "BAMLH0A0HYM2" in macro:
        hy = macro["BAMLH0A0HYM2"].dropna().iloc[-500:]
        side.plotly_chart(px_.line(hy, color_discrete_sequence=[RED]).update_layout(
            height=160, margin=dict(t=25, b=5), showlegend=False, title="High yield spread, % (FRED)", yaxis_title="", xaxis_title=""), width="stretch")
    section("Currencies", last_year(mk, ["GBPUSD=X", "EURUSD=X"]), "Dollars per unit")
    section("Commodities", last_year(mk, ["GC=F", "BZ=F"], rebase=True), "Rebased to 100")
    if "crypto" in (ss.profile.get("allowed_assets") or []):
        section("Crypto", last_year(mk, ["BTC-USD"]), "US$")
    c = st.columns(2)
    if c[0].button("← Back to the questionnaire", width="stretch"):
        goto(1)
    if c[1].button("Next: what the data tells us →", type="primary", width="stretch"):
        goto(3)
    st.stop()

# ------------------------------------------------------------------ STEP 3: conclusions
if ss.step == 3:
    b = ss.brief
    st.subheader("What the data tells us, and what we will do about it")
    st.caption("Every conclusion is tied to a specific public data point. These rules feed straight into the portfolio construction.")
    for row in b["conclusions"]:
        with st.container(border=True):
            cc = st.columns([1.3, 1, 2, 2])
            cc[0].markdown(f"**{row['Data point']}**  \n<span class='pill'>{row['Source']}</span>", unsafe_allow_html=True)
            cc[1].markdown(f"### {esc(row['Reading'])}" if row["Reading"] else "")
            cc[2].markdown(f"**Means:** {esc(row['What it means'])}")
            cc[3].markdown(f"**We will:** {esc(row['What we do'])}")
    with st.expander("Every number used (raw readings)"):
        st.json(b["readings"], expanded=False)
    c = st.columns(2)
    if c[0].button("← Back to the markets", width="stretch"):
        goto(2)
    if c[1].button("🚀 Build my portfolio", type="primary", width="stretch"):
        with st.status("Your banker and the AI committee are at work...", expanded=True) as status:
            try:
                ensure_data(status)
                ss.res = pipeline.run(ss.md, dict(ss.profile), use_ai=use_ai, progress=lambda m: status.write(f"▸ {m}"))
                ss.chat = []
                status.update(label="Portfolio ready", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Run failed", state="error")
                st.exception(e)
                st.stop()
        goto(4)
    st.stop()

# ------------------------------------------------------------------ STEP 4: portfolio
if "res" not in ss:
    goto(3)
if st.button("← Back to the evidence"):
    goto(3)

res = ss.res
prof = res["profile"]
d = res["decision"]
w = res["weights"]
feat = res["features"]
for e in res["errors"]:
    st.warning(e)

tabs = st.tabs(["Your portfolio", "How the AI decided", "Risk check", "How every number is made", "Data", "💬 Ask your banker"])

# ------------------------------------------------------------------ recommendation
with tabs[0]:
    st.markdown(f'<div class="headline">{esc(d.get("headline", ""))}</div>', unsafe_allow_html=True)
    st.write("")
    v = res["verification"]
    st_ = res["stats"]
    k = st.columns(6)
    sv = res["opt_info"]["sleeves"]
    kpi(k[0], "Index · stocks · bonds", f"{sv[config.SLEEVE_INDEX] * 100:.0f} · {sv[config.SLEEVE_STOCKS] * 100:.0f} · {sv[config.SLEEVE_BONDS] * 100:.0f}%")
    kpi(k[1], "Expected return a year (assumption)", pct(res["opt_info"]["exp_return"]))
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

    st.markdown("#### A note from your banker")
    with st.container(border=True):
        st.markdown(f"<div style='font-size:1.08rem; line-height:1.7'>{str(d.get('letter', '')).replace('$', '&#36;').replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True)
    st.markdown("#### What each part of your money does")
    plain = {config.SLEEVE_INDEX: "Grows with the whole market at very low cost, so your result never depends on a few companies.",
             config.SLEEVE_STOCKS: "A small group of fast growing companies from the themes you believe in, each one capped in size.",
             config.SLEEVE_BONDS: "Your cushion: tends to hold up when shares fall, pays interest, and gives us cash to buy when prices are low.",
             config.SLEEVE_CREDIT: "Loans to large companies that pay more interest than government bonds.",
             config.SLEEVE_REAL: "Real assets like gold, property and commodities, which often rise when inflation does.",
             config.SLEEVE_ALTS: "Steadier funds and strategies that tend to behave differently from technology shares.",
             config.SLEEVE_CRYPTO: "A small, capped position in Bitcoin, at your request."}
    tr = res["trades"]
    for sname in config.SLEEVES:
        rows_ = tr[tr["Ticker"].map(config.sleeve_of) == sname]
        if rows_.empty:
            continue
        names = ", ".join(rows_["Ticker"])
        st.markdown(f"**\\${rows_['Dollars'].sum():,.0f} · {sname}** ({names})  \n{plain[sname]}")
    with st.expander("Technical details: suitability, limits and committee reasoning"):
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
        for sname in config.SLEEVES:
            items = [(t, v) for t, v in dw.items() if config.sleeve_of(t) == sname]
            if items:
                st.markdown(f"**{sname}, {pct(sum(v for _, v in items))}:** " + ", ".join(
                    f"{t} ({config.ETFS[t][0]}) {pct(v)}" for t, v in items))
    oh, fx = res.get("option_hedge"), res.get("fx_hedge")
    if oh or fx:
        st.markdown("#### 🛡️ Hedges")
    if oh:
        with st.container(border=True):
            hc = st.columns([1, 1.3])
            with hc[0]:
                desc = (f"Buy {oh['contracts']} {oh['underlying']} put{'s' if oh['contracts'] > 1 else ''} at strike {oh['put_strike']:.0f}"
                        + (f", sell {oh['contracts']} call{'s' if oh['contracts'] > 1 else ''} at {oh['call_strike']:.0f}" if oh["call_strike"] else "")
                        + f", expiring {oh['expiry']}.")
                st.markdown(f"**Insurance on your shares ({oh['style'].lower()}).** If the market falls more than "
                            f"{abs(oh['protection_from']) * 100:.0f}% before {oh['expiry']}, the insurance pays out"
                            + (f"; to make it cheaper we give up gains above {oh['cap_at'] * 100:.0f}%" if oh.get("cap_at") is not None else "")
                            + f". In practice: {desc}")
                q = st.columns(2)
                kpi(q[0], "Net cost (from T bills)", f"${oh['net_cost']:,.0f} ({pct(oh['net_cost_pct'], 2)})")
                kpi(q[1], "Protection starts", pct(oh["protection_from"], 0, sign=True))
                q = st.columns(2)
                lw = oh["loss_at_minus_30"]
                kpi(q[0], f"If {oh['underlying']} falls 30%: loss without", f"${-lw['without']:,.0f}")
                kpi(q[1], "with the hedge", f"${-lw['with']:,.0f}")
                st.caption(f"Priced from: {oh['source']}. Equity beta to {oh['underlying']}: {oh['beta']:.2f}. "
                           + (f"Upside given up above {pct(oh['cap_at'], 0, sign=True)}. " if oh.get("cap_at") is not None else "")
                           + "Options trade in 100 share contracts, so the hedge ratio is " + pct(oh["hedge_ratio"], 0) + ".")
            with hc[1]:
                pf = oh["payoff"].set_index("move")
                fig = px_.line(pf, color_discrete_sequence=[GREY, PURPLE])
                fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_tickformat=".0%", yaxis_tickprefix="$",
                                  xaxis_title=f"{oh['underlying']} move by expiry", yaxis_title="Equity P&L", legend_title="")
                st.plotly_chart(fig, width="stretch")
    if fx:
        with st.container(border=True):
            st.markdown(f"**Currency hedge (you think in pounds).** Sell \\${fx['usd_to_hedge']:,.0f} forward against GBP for 3 months "
                        f"(about £{fx['gbp_equivalent']:,.0f}); spot GBP/USD {fx['spot']:.4f}"
                        + (f", 3 month forward about {fx['forward_3m']:.4f}" if fx.get("forward_3m") else "")
                        + f". Hedged: 100% of bonds, {pct(fx['equity_ratio'], 0)} of equities. {fx['why']}")
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

# ------------------------------------------------------------------ methodology
with tabs[3]:
    st.markdown("#### How every number on these pages is produced")
    st.caption("Nothing is typed in by hand. Each figure is either read from a public data source, computed from that data with the "
               "formula below, a stated assumption, an AI judgement (labelled), or one of the client's own answers.")
    oi, mc_, oh_ = res["opt_info"], res["mc"], res.get("option_hedge")
    pol_ = oi["policy"]
    fwn = ", ".join(f"{k} {v / sum(res['factor_weights'].values()) * 100:.0f}%" for k, v in res["factor_weights"].items() if v)
    worst_ = res["stress"]["Portfolio"].min() if len(res["stress"]) else np.nan
    st_ = res["stats"]
    rows_m = [
        ("Expected volatility", pct(oi["exp_vol"]), "Computed from data",
         "Square root of w x C x w, where w are the portfolio weights and C is the covariance matrix of the daily returns of every holding over "
         "the last 2 years, annualised by 252 trading days. Correlations between individual stocks are pulled 30% toward their average to cut noise.",
         "Yahoo Finance adjusted daily closes"),
        ("Expected return a year", pct(oi["exp_return"]), "Stated assumption",
         f"Weighted average of long run assumptions: 3 month T bill yield ({pct(res['rf'], 2)}) plus a premium. Stocks: 6% plus 4% per standard "
         "deviation of their composite score (capped at plus or minus 2). Funds: S&P 500 5%, Nasdaq 100 5.5%, international 5%, Treasuries 0 to 1.2%, "
         "inflation linked 1%, corporate bonds 1.5 to 2.5%, gold and commodities 1.5%, managed futures 2%, min volatility 4%. Used to rank, not a promise.",
         "T bill yield from Yahoo Finance ^IRX or FRED; premia are our assumptions"),
        ("Worst historical stress", pct(worst_), "Computed from data",
         "Today's weights multiplied by each holding's actual price change between the start and end dates of each crisis. Funds or stocks that did "
         "not exist yet use their 2 year beta times the index move.",
         "Yahoo Finance prices; crisis dates: Feb to Mar 2020, Jan to Oct 2022, 24 to 27 Jan 2025, Feb to Apr 2025"),
        (f"Chance of a fall beyond {pct(prof['max_drawdown'], 0)} in {prof['horizon_years']} years", pct(res["p_breach"], 0), "Data plus assumption",
         "Monte Carlo: 2,000 simulated paths built by resampling 21 day blocks of this portfolio's own daily returns over the last 5 years, with the "
         "average return reset to the expected return assumption. The figure is the share of paths whose worst peak to trough fall is bigger than the limit.",
         "Yahoo Finance daily returns; expected return assumption above"),
        ("Monte Carlo median and range", f"${np.median(mc_['terminal']):,.0f} (5th to 95th: ${np.percentile(mc_['terminal'], 5):,.0f} to ${np.percentile(mc_['terminal'], 95):,.0f})",
         "Data plus assumption", "Middle, 5th and 95th percentile of the 2,000 simulated end values.", "As above"),
        ("3 year return, Sharpe, max drawdown (Risk check)", f"CAGR {pct(st_.get('CAGR'))}, Sharpe {st_.get('Sharpe', float('nan')):.2f}, max fall {pct(st_.get('Max drawdown'))}",
         "Computed from data, in sample", "Today's weights applied to actual daily returns over the last 3 years. Sharpe is (annual return minus T bill "
         "yield) divided by annual volatility. Flattering, because today's picks are known winners; see the walk forward test.", "Yahoo Finance daily returns"),
        ("Walk forward test", "Risk check tab", "Computed from data, out of sample",
         "Every quarter for 3 years, using only prices known on that date: rank stocks by 12 month momentum minus volatility, pick the top names, "
         "run the same optimiser, hold until the next quarter. Compared with the benchmark over the same days.", "Yahoo Finance daily prices"),
        ("Composite stock score and rank", "AI decided tab", "Data plus AI",
         f"Average of z scores (value minus the universe average, divided by the standard deviation, extremes trimmed at the 5th and 95th "
         f"percentiles) for each factor, then weighted: {fwn}.",
         "Yahoo Finance fundamentals, prices and analyst targets; AI news reading; embeddings"),
        ("Theme fit", "AI decided tab", "AI measurement",
         f"Cosine similarity between the {config.EMBED_MODEL} embedding of the company's business description and the embedding of each theme "
         "the client chose; the highest match counts.", "Yahoo Finance business descriptions; Azure OpenAI embeddings"),
        ("News sentiment", "AI decided tab", "AI judgement",
         "The AI reads up to 8 recent headlines per company and scores what they mean for the business from minus 1 to plus 1.",
         "Yahoo Finance or Finnhub headlines; Azure OpenAI"),
        ("Risk score", f"{prof['risk_tolerance']}/10", "Client answers",
         f"Willingness ({prof.get('willingness', 'n/a')}) is the average of three answers: self description, reaction to a 20% fall, largest "
         f"acceptable fall. Ability ({prof.get('ability', 'n/a')}) is the average of horizon and share of wealth, minus points for no emergency "
         "fund, variable income, regular withdrawals, near term cash needs or a concentrated position. The score is the lower of the two; the AI "
         "banker may move it by at most 2 points after the follow up questions.", "Questionnaire"),
        ("Volatility ceiling", pct(oi["target_vol"]), "Rule from client answers",
         "The lower of 6% plus 2.5% per risk point, and 80% of the largest fall the client accepts.", "Risk score and questionnaire"),
        ("Bond floor, stock cap, index floor", f"{pct(pol_['min_bonds'], 0)} / {pct(pol_['max_single_stocks'], 0)} / {pct(pol_['min_index'], 0)}",
         "Rule from client answers", "Government bonds and cash at least 65% minus 5.5% per risk point (between 10% and 60%); single stocks at most "
         "5% per risk point; index funds at least 20%.", "Risk score"),
        ("Portfolio weights and dollar amounts", "Your portfolio tab", "Computed",
         "SciPy SLSQP optimiser: maximise expected return minus a small risk penalty, subject to every limit above, the stress test limit in every "
         "crisis, a per stock cap, a 40% cap per theme and a minimum T bill holding for near term cash needs. Dollars are weight times $100,000; "
         "shares are dollars divided by the latest close.", "All of the above"),
        ("Share of portfolio risk", "Risk check tab", "Computed from data",
         "Each holding's weight times its covariance with the whole portfolio, divided by the portfolio variance; adds up to 100%.", "Yahoo Finance daily returns"),
    ]
    if oh_:
        rows_m += [
            ("Hedge cost", f"${oh_['net_cost']:,.0f}", "Data" if oh_["source"].startswith("live") else "Model estimate",
             f"Contracts = beta of our shares to {oh_['underlying']} ({oh_['beta']:.2f}) times the dollars in shares, divided by 100 times the index "
             "price, rounded. Cost = contracts x 100 x option mid price ((bid + ask) / 2), minus the call sold for a collar.", oh_["source"]),
            ("Loss if the index falls 30%, with and without hedge", f"${-oh_['loss_at_minus_30']['without']:,.0f} vs ${-oh_['loss_at_minus_30']['with']:,.0f}",
             "Computed", "Without: beta x minus 30% x dollars in shares. With: the same plus what the puts pay at expiry, minus the net cost.", "As above"),
        ]
    if res.get("fx_hedge"):
        rows_m.append(("Currency forward rate", f"{res['fx_hedge']['forward_3m']:.4f}" if res["fx_hedge"].get("forward_3m") else "n/a", "Computed from data",
                       "Spot GBP/USD x (1 + US 3 month rate x 0.25) / (1 + UK 3 month rate x 0.25).", "Yahoo Finance GBPUSD=X, ^IRX; FRED UK 3 month rate"))
    rows_m.append(("Figures traced to data", f"{res['verification']['verified']}/{res['verification']['total']}", "Automatic check",
                   "Every percentage, multiple or dollar figure the AI writes is matched against the numbers computed above (within rounding). "
                   "Anything not found is flagged under the holding.", "The data packet sent to the AI"))
    mdf = pd.DataFrame(rows_m, columns=["Number shown", "Value now", "Type", "How it is calculated", "Inputs and source"])
    st.dataframe(mdf, hide_index=True, width="stretch", height=640,
                 column_config={"How it is calculated": st.column_config.TextColumn(width="large")})

# ------------------------------------------------------------------ data
with tabs[4]:
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
with tabs[5]:
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
