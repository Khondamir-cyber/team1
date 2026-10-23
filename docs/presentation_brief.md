# Velocity, your AI private banker: presentation brief for Claude

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

## The product: Velocity, your AI private banker

**One line:** Velocity works like a private banker: it gets to know the client first, explains today's markets in plain
English, shows which data drove each decision, then builds a hedged, multi asset $100,000 portfolio that an AI investment
committee debates and every number of which is traced back to public data.

**The four step client journey (this is the live demo):**

1. **Know your client.** A questionnaire like a private bank's onboarding: age, job, tax residence, base currency (GBP or
   USD), goal and horizon, withdrawals, money needed within 2 years, share of total wealth, emergency savings, income
   stability, concentrated positions (Alex already holds about $40,000 of NVDA employer shares), how they would react to a
   20% fall, the largest fall they can live with, experience with options, themes they believe in, exclusions, ESG, and
   which asset classes they are comfortable with. Velocity scores **willingness** and **ability** to take risk separately
   and invests at the **lower** of the two (standard private banking practice). The AI banker then reads the answers,
   spots contradictions (for example "very risk averse" but "grow as fast as possible", or adding more NVDA on top of
   employer shares) and asks 2 to 4 follow up questions before setting the final risk score with a plain English rationale.
2. **The markets today.** Charts with one or two plain sentences each: US interest rates (3 month, 10 and 30 year),
   inflation, unemployment, world stock markets (S&P 500, Nasdaq 100, FTSE 100, Euro Stoxx 50, Nikkei 225, emerging
   markets), the VIX fear gauge, bonds and credit (Treasuries, investment grade, high yield, credit spreads), currencies
   (GBP/USD, EUR/USD), gold and oil, and Bitcoin only if the client allows crypto. An AI "big picture" summary on top.
3. **What the data tells us.** A card per conclusion: the data point, its source (for example "FRED DFII10" or "Yahoo
   Finance ^VIX"), the reading, what it means in plain English, and what the portfolio will do about it (for example: VIX
   low, so insurance is cheap, so buy protective puts now; inflation above target, so hold inflation linked bonds and gold).
4. **Your portfolio.** Multi asset and core and satellite, only from asset classes the client allowed: core index funds
   (VOO, QQQ, VXUS), capped growth stock picks scored by the AI, government bonds and cash (SGOV, SHY, IEF, TLT, TIP),
   corporate credit (LQD, HYG), real assets (REITs VNQ, gold GLD, commodities DBC), alternatives and hedges (managed
   futures DBMF, minimum volatility USMV) and Bitcoin ETF IBIT if allowed. The client's risk score sets the bond floor and
   the stock cap; the optimiser also enforces a volatility ceiling and a loss smaller than the client's limit in every
   historical crisis replay (COVID 2020, 2022 rate shock, DeepSeek January 2025, 2025 tariff shock).
   **Hedges like a private bank:** a protective put or collar on the equity exposure priced from the live Yahoo Finance
   option chain (Black Scholes fallback), with a payoff chart and the loss with and without the hedge if the index falls
   30%; and for a client who thinks in pounds, a GBP/USD forward hedge on 100% of the bonds and part of the equities.
   Then the **AI investment committee** (Growth PM, Risk Officer, Macro Strategist, CIO) debates and amends the stock
   picks, a **verifier** checks every figure in the explanation against the data, and the client can **ask the banker**
   questions in a chat.

**Data used:** Yahoo Finance (prices, fundamentals, analyst targets, news, indices, yields, VIX, FX, gold, oil, Bitcoin,
option chains), FRED from the St. Louis Fed (inflation, unemployment, real yields, credit spreads, UK rates), Finnhub
optional. **AI used:** Azure OpenAI gpt-6.1-sol for the banker follow ups, market explanation, news reading and the
committee; text-embedding-3-large to measure how well each company fits the client's themes.

**Built with:** Python, Streamlit, Azure OpenAI, yfinance, FRED, SciPy optimiser, Plotly. Code: github.com/Khondamir-cyber/team1

## Slide plan (8 slides, about 6 minutes; the live demo sits inside slides 4 to 6)

| # | Slide | On the slide | Talk track (speaker notes) | Time |
|---|---|---|---|---|
| 1 | Title | "Velocity: your AI private banker". Subtitle: "Know the client. Read the market. Show the evidence. Then invest." Team 1, names | "Most robo advisers ask five questions and sell you a model portfolio. A private banker listens first. So does Velocity." | 0:20 |
| 2 | 01 The client | Alex Morgan, 32, London, thinks in pounds, $100,000, 10 years, would lose sleep below a 30% fall, already holds about $40,000 of NVDA employer shares. Willingness vs ability bar | "Alex says he wants maximum growth, but his answers say he is more careful than that, and he already has a big NVDA bet. Velocity catches both." | 0:40 |
| 3 | 02 The data | Table: Yahoo Finance (what), FRED (what), Finnhub optional; "50+ tickers, 7 years daily, live option chains" | "All public, all free, cached so nothing breaks on stage." | 0:25 |
| 4 | 03 What the AI does | Pipeline: KYC questionnaire → AI banker follow ups → market briefing → data conclusions → embeddings and news agent → optimiser with client limits → hedges → AI investment committee → verifier | "Each AI step does something a human analyst would take days to do, and rules keep it honest." | 0:40 |
| 5 | Live demo, part 1 | Screenshot backups: questionnaire with follow ups [LIVE], market charts [LIVE], conclusion cards [LIVE] | Demo: fill or show the questionnaire, read the banker's follow ups and answer one; show 2 market charts and their plain notes; show 2 conclusion cards ("VIX low, so we buy puts"). | 1:40 |
| 6 | 04 The recommendation: live demo, part 2 | Treemap by sleeve [LIVE], 3 big numbers: [LIVE: % bonds and cash], [LIVE: hedge cost and loss if index falls 30% with vs without], [LIVE: figures verified] | Demo: portfolio treemap, the hedge card and payoff chart, one verified holding card, the committee memos, "Why not NVDA?" (already held), ask the banker "What if markets fall 30%?" | 1:30 |
| 7 | Risk check | Crisis replay chart against the client's limit [LIVE], walk forward vs benchmark [LIVE] | "We replay four real crises on today's portfolio; the optimiser is not allowed to break the client's limit in any of them." | 0:25 |
| 8 | Why Velocity wins | Ticks mapped to judging: KYC drives every limit (client focus); embeddings, agents, live option pricing (data and AI); sized trades with hedges and verified reasons (quality); banker journey plus AI committee plus fact check (originality). "Educational hackathon project. Not financial advice." | "Velocity knows the client, explains the market, proves its numbers and protects the downside." | 0:20 |

## Screenshots to collect from the app (after a fresh build)

1. Step 1: the questionnaire, the willingness vs ability chart and the banker's follow up questions.
2. Step 2: two market charts with their plain English notes.
3. Step 3: the conclusion cards (data point, source, reading, what we do).
4. Step 4: treemap by sleeve, trade list, the hedge card with payoff chart, one "figures verified" holding card.
5. "How the AI decided": the three committee memos; "Risk check": crisis replay chart with the client limit line.

## Rules for the deck

* Use only facts in this brief or numbers read from the app; never invent performance figures.
* Label in sample backtests as illustrative; the walk forward test is the honest check.
* Keep "Educational hackathon project. Not financial advice." on the title or closing slide.
* No hyphens or dashes in slide text.
