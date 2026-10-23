# Velocity: presentation brief for Claude

Team 1 · Microsoft x LSEG London Hackathon 2026 · pitch slot 5 to 7 minutes including a live demo

## How to use this file

1. Open Claude, start a new chat, and attach this file.
2. Paste the prompt in the box below and send it.
3. When Claude asks for screenshots or live numbers, take them from the running app (Khondamir has it on his laptop) and paste them in.

> **Prompt to paste:**
> Build a 16:9 slide deck for our hackathon pitch using the attached brief. Follow the slide plan exactly: 8 slides, one idea
> per slide, big numbers, very little text, speaker notes on every slide with the talk track given. Use only facts in the
> brief; wherever it says [LIVE: ...] leave a clearly marked placeholder for me to fill from the app. Visual style: deep
> purple #2a0f4c and #5b2a9b with yellow #f5d90a accents and white backgrounds, matching the hackathon's own slides. Do not
> use hyphens or dashes in slide text. Make it as a Claude Slides deck I can edit and export to PowerPoint.

## The brief we were given (what the judges asked for)

**Challenge:** a client has $100,000 to invest. Using public data and LLMs, build a tool that recommends where the money
should go and explains why.

**Every team must cover:**

| # | Section | What the judges want |
|---|---|---|
| 01 | The client | Who they are and what they want |
| 02 | The data | Which public data the tool uses |
| 03 | What the AI does | How the LLMs turn the data into a recommendation |
| 04 | The recommendation and why | Where the $100,000 goes and how the tool explains it |
| 05 | Live demo | Show it working, not just slides |

**Scoring (each 1 to 10, total out of 50; a Microsoft judge scores live; top 3 across all rooms reach the final):**

| Criterion | 10 out of 10 looks like |
|---|---|
| Client focus | Solution clearly designed around the client's objectives |
| Use of data and AI | AI produces analysis that would be hard to do manually |
| Quality of recommendation | Specific, well reasoned, logic easy to follow |
| Originality | A novel approach that stands out from the field |
| Execution | Polished live demo, presented with confidence |

The deck must hit all five sections and make each scoring criterion obvious to a judge.

## The product: Velocity, the AI growth analyst

**One line:** Velocity turns a client's own words into a $100,000 growth portfolio, has an AI investment committee debate
it, and proves every number in its explanation against live public data.

**Client we chose:** a high growth investor in public markets (not a startup investor). Default persona: Alex Morgan, 32,
product engineer in London, $100,000 lump sum from vested shares, 10+ year horizon, no need to withdraw, can sit through a
35% fall but not 50%, believes in AI, cloud software and digital platforms, listed liquid stocks only, no crypto or tobacco,
no single stock above 15%, wants to beat the Nasdaq 100. Every limit is editable in the app, and the client is typed in
plain English, so a cautious client gets a very different portfolio.

**How it works, in order:**

1. **Profile agent (LLM):** reads the client's brief in plain English and turns it into a structured investment policy:
   horizon, risk tolerance, max drawdown, position cap, themes, exclusions, plus a list of anything it had to assume.
2. **Live public data:** Yahoo Finance (7 years of daily prices for 40 growth stocks and 9 index, bond and diversifier ETFs, fundamentals,
   analyst consensus and price targets, company news, 10 year and 3 month Treasury yields, VIX, high yield vs Treasury
   credit); FRED from the St. Louis Fed (10 year real yield, CPI, unemployment, credit spreads); Finnhub optional.
   Everything is cached so the demo survives bad wifi.
3. **Embeddings (text-embedding-3-large):** every company's business description is compared with the client's themes,
   so "AI" exposure is measured from what the company actually does, not from a label.
4. **News agent (LLM):** reads recent headlines per company and scores what they mean for fundamentals over 6 to 12 months.
5. **Quant engine:** scores every stock on 8 factors (growth, momentum, quality, valuation, analyst views, news
   sentiment, theme fit, low volatility). The weights shift with the client: a cautious client puts more weight on
   quality and low volatility.
6. **Macro overlay:** transparent rules from the rates, credit and volatility data (for example, if real yields are high
   the valuation penalty rises; if credit is stressed the T bill floor rises).
7. **Core and satellite portfolio with the client's limits as hard rules:** every portfolio has four sleeves:
   **core index funds** (Vanguard S&P 500 VOO, Invesco Nasdaq 100 QQQ, Vanguard international VXUS), **growth stock
   picks** as a capped satellite, **government bonds and cash** (T bills SGOV, 1 to 3 year SHY, 7 to 10 year IEF,
   inflation linked TIP) and **diversifiers** (gold GLD, minimum volatility stocks USMV). The client's risk tolerance sets
   the strategic policy (for risk tolerance 8: bonds and cash at least 21%, index funds at least 20%, single stocks at most
   40%). The optimiser then maximises expected return subject to that policy, a single stock cap, a theme cap, a
   volatility ceiling, and **a loss smaller than the client's max drawdown in every historical crisis replayed on today's
   portfolio**: COVID crash 2020, the 2022 rate shock, the DeepSeek AI selloff of January 2025 and the 2025 tariff shock.
8. **AI investment committee (the originality hook):** three LLM agents write memos in parallel: a Growth PM makes the
   bull case, a Risk Officer attacks concentration and drawdowns, a Macro Strategist reads rates and credit. A CIO agent
   decides, can remove or cap stocks, and the optimiser re runs with those changes. The debate and the before/after
   weights are shown in the app.
9. **Verifier:** every percentage and multiple the AI writes in its explanation is automatically checked against the
   source data; each holding shows a "figures verified" badge. This is our answer to AI hallucination in finance.

**What the client gets:** dollar amount and number of shares per stock and per fund, grouped into the four sleeves; a headline recommendation; a
portfolio thesis; a suitability check against their own drawdown limit; one card per holding with its role, why it is
there (with verified figures), its key risk and a measurable sell trigger; a "Why not X?" explainer for any stock left
out; a downloadable client memo and trade list; and an "Ask the analyst" chat grounded in the same data.

**Risk evidence in the app:** growth of $100,000 over 3 years vs QQQ and SPY (flagged as in sample), an honest walk
forward test of the method rebalanced quarterly using only data known at the time, the four crisis replays against the
client's limit, a 2,000 path Monte Carlo over the client's horizon, and each holding's share of total risk.

**Built with:** Python, Streamlit, Azure OpenAI (chat model gpt-6.1-sol, embeddings text-embedding-3-large), yfinance,
FRED, SciPy optimiser, Plotly. Code: github.com/Khondamir-cyber/team1

## Slide plan (8 slides, about 6 minutes; the live demo sits inside slide 6)

| # | Slide | On the slide | Talk track (speaker notes) | Time |
|---|---|---|---|---|
| 1 | Title | "Velocity: the AI growth analyst". Subtitle: "From a client's own words to a verified $100,000 portfolio". Team 1, names | "Every robo adviser asks you to fill in a form. Velocity listens to you, then shows its working." | 0:20 |
| 2 | 01 The client | Persona card for Alex Morgan with 4 big numbers: 10+ years, 35% max fall, 15% max per stock, $100,000. Quote of his brief | "Alex wants growth, but he told us exactly how much pain he can take. Those words become hard rules in our optimiser, not a questionnaire score." | 0:40 |
| 3 | 02 The data | Logo strip or table: Yahoo Finance, FRED, Finnhub; what each gives; "40 growth stocks, 6 defensive ETFs, 7 years daily, refreshed live" | "All public, all free, all cached so nothing breaks on stage." | 0:30 |
| 4 | 03 What the AI does | Pipeline diagram, left to right: Client brief → Profile agent → Data + Embeddings + News agent → Quant engine and macro overlay → Optimiser with client limits → AI investment committee → Verifier → Recommendation | "Five LLM jobs, each doing something a human analyst would take days to do, and a quant engine keeping them honest." | 0:50 |
| 5 | The committee and the verifier (originality) | Three agent icons (Growth PM, Risk Officer, Macro Strategist) feeding a CIO; a before/after weights mini chart; a green badge "[LIVE: 20/25] figures traced to source data" | "Our AI argues with itself before it advises you, and then we fact check it. Every number in the explanation is traced to the data." | 0:40 |
| 6 | 04 The recommendation: live demo | Screenshot of the treemap and trade list as backup [LIVE: screenshot] with 3 big numbers: [LIVE: % index funds / % stocks / % bonds and cash], [LIVE: expected return], [LIVE: worst crisis loss vs 35% limit] | Switch to the app. Demo order: read Alex's brief with AI; build the portfolio and narrate the live steps; recommendation tab and verified badges; committee memos; "Why not ...?" on a famous stock left out; ask the analyst "What if Nvidia falls 40%?". Then, if time: change the brief to a cautious client and rebuild to show the bond sleeve grow and the stock satellite shrink. | 2:30 |
| 7 | Risk check | Crisis replay bar chart with the client's limit line [LIVE: screenshot]; walk forward result [LIVE: method CAGR vs QQQ CAGR, max drawdowns] | "We replay four real crises on today's portfolio and the optimiser is not allowed to break the client's limit in any of them." | 0:30 |
| 8 | Why Velocity wins | Four ticks mapped to the judging criteria: client's words become hard limits (client focus); embeddings plus agent debate (data and AI); verified, specific, sized trades (quality); committee plus fact check (originality). "Educational hackathon project. Not financial advice." | "Velocity is designed around the client, explains itself, and proves its numbers." | 0:20 |

## Screenshots to collect from the app (after a fresh build)

1. Client section with the policy card filled in by the AI.
2. Recommendation tab: headline, treemap and trade list.
3. One holding card with the green "figures verified" badge.
4. "How the AI decided" tab: the three committee memos side by side and the before/after chart.
5. Risk check tab: crisis replay chart with the red client limit line, and the Monte Carlo fan chart.

## Rules for the deck

* Use only facts in this brief or numbers read from the app; never invent performance figures.
* Label in sample backtests as illustrative; the walk forward test is the honest check.
* Keep "Educational hackathon project. Not financial advice." on the title or closing slide.
* No hyphens or dashes in slide text.
