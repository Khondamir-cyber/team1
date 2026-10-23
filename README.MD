# Velocity: the AI growth analyst

Microsoft x LSEG London Hackathon 2026. Educational project, not financial advice.

A client has $100,000. Velocity reads the client's brief in plain English, pulls live public market data,
scores 40 growth stocks, sizes a portfolio under the client's own risk limits, has an **AI investment committee**
debate and amend it, and explains every holding with figures that are **automatically checked against the source data**.

## Run it (5 minutes)

```bash
git clone https://github.com/Khondamir-cyber/team1.git
cd team1
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
# paste the team's Azure OpenAI key into .env (AZURE_OPENAI_API_KEY=...)
python check_setup.py              # checks the key, both models and Yahoo Finance
streamlit run app.py
```

The key can also be pasted in the app sidebar. The first build pulls 7 years of prices for 40 tickers (about a minute);
everything is cached in `data_cache/`, so later runs are fast and the demo survives bad wifi.
**Run one full build before presenting** so the cache is warm.

## How it maps to the brief

| Brief | What Velocity does |
|---|---|
| 01 The client | A high growth (public markets) investor: Alex Morgan, 32, $100k lump sum, 10+ year horizon, can sit through a 35% fall, wants AI, cloud and digital platforms, no single stock above 15%. The profile agent turns the brief into a structured investment policy; every limit is editable. |
| 02 The data | **Yahoo Finance** (7y daily prices, fundamentals, analyst consensus and targets, news, ^TNX, ^IRX, ^VIX, HYG/IEF credit), **FRED** (10y real yield, CPI, unemployment, high yield spread; no key needed), **Finnhub** optional (analyst rating trends, company news). |
| 03 What the AI does | (1) Profile agent: free text to investment policy. (2) `text-embedding-3-large` embeds every company's business description and measures its fit with the client's themes. (3) News agent scores what recent headlines mean for fundamentals. (4) Quant engine: 8 factor z scores, macro overlay, then a core and satellite optimiser: index funds (VOO, QQQ, VXUS), capped growth stock picks, government bonds and cash (SGOV, SHY, IEF, TIP) and diversifiers (GLD, USMV), with the bond floor and stock cap set by the client's risk tolerance and the client's max drawdown enforced in every historical crisis replay. (5) AI investment committee: Growth PM, Risk Officer and Macro Strategist write memos in parallel; the CIO decides and can remove or cap names, then the optimiser re runs. (6) Verifier traces every % and x figure in the explanation back to the data packet. |
| 04 Recommendation and why | Dollar and share amounts per stock plus a T bill buffer, a thesis, a suitability check against the client's drawdown limit, and per holding cards: role, why (with verified figures), key risk, measurable sell trigger. "Why not X?" explains any stock left out. Memo and trade list download. |
| 05 Live demo | `streamlit run app.py`, then the demo script below. |

## Why it scores well on the rubric

* **Client focus**: the client's own words set the themes, the position cap, the volatility budget and the drawdown test every output is checked against.
* **Use of data and AI**: embeddings for thematic fit and an agent debate that changes the portfolio are analysis that would be hard to do by hand.
* **Quality of recommendation**: every figure verified; stress tests (COVID 2020, 2022 rate shock, DeepSeek Jan 2025, tariff shock 2025); a 2,000 path Monte Carlo; an honest walk forward test of the method versus QQQ.
* **Originality**: the committee debate with dissent recorded, plus automatic fact checking of the AI's own explanation.

## Demo script (3 minutes)

1. Show the client brief, click **Read the brief with AI**: the policy card fills in (horizon, drawdown, themes, exclusions).
2. Click **Build the $100,000 portfolio**: narrate the live steps as they tick (data, embeddings, news, optimiser, committee).
3. **Recommendation** tab: treemap of the $100k, headline, the green "figures verified" badges on each holding.
4. **How the AI decided** tab: the three committee memos side by side, the CIO's changes and the before/after chart. Use **Why not ...?** on a famous stock that was left out.
5. **Risk check** tab: stress tests against the client's 35% line, Monte Carlo fan, walk forward versus QQQ.
6. **Ask the analyst**: "What happens to this portfolio if Nvidia falls 40%?"

## Code map

```
app.py                 Streamlit interface
advisor/config.py      universe, themes, stress windows, default client
advisor/data.py        Yahoo Finance, FRED, Finnhub with disk cache and stale fallback
advisor/analytics.py   factor scores, macro overlay, optimiser, backtest, stress, Monte Carlo, walk forward
advisor/llm.py         Azure OpenAI: profile agent, embeddings, news agent, committee, Q&A
advisor/report.py      data packet, rules based fallback, figure verifier, memo export
advisor/pipeline.py    runs everything end to end
check_setup.py         pre demo connectivity check
tests/offline_smoke_test.py   runs the pipeline on random data with a fake LLM (code check only)
```

If the AI is unreachable the app still produces the portfolio with rules based explanations, clearly labelled.
