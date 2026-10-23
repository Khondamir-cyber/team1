"""Azure OpenAI layer: profile parsing, embedding based theme fit, news sentiment, the AI investment committee, Q&A."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import config

_client = None
_key_override = None
LAST_ERROR = ""


def set_api_key(key: str | None):
    global _client, _key_override
    _key_override = key or None
    _client = None


def _key():
    return _key_override or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def available() -> bool:
    return bool(_key())


def client():
    global _client
    if _client is None:
        from openai import AzureOpenAI
        _client = AzureOpenAI(api_version=config.AZURE_API_VERSION, azure_endpoint=config.AZURE_ENDPOINT,
                              api_key=_key(), timeout=180, max_retries=2)
    return _client


def chat(system: str, user: str, max_tokens: int = 6000, json_mode: bool = False) -> str:
    global LAST_ERROR
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(model=config.CHAT_MODEL, messages=msgs, max_completion_tokens=max_tokens)
    if json_mode:
        try:
            r = client().chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            return r.choices[0].message.content or ""
        except Exception as e:  # some deployments reject response_format; retry plain
            LAST_ERROR = str(e)[:200]
    r = client().chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


def parse_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


# ---------------------------------------------------------------- 1. client profile
PROFILE_SYSTEM = """You are the onboarding analyst at a wealth manager. Convert the client's own words into a structured
investment policy. Return ONLY a JSON object with these keys:
client_name (string), objective (one sentence), horizon_years (int), risk_tolerance (int 1 to 10),
max_drawdown (decimal, e.g. 0.35), target_vol (decimal annual volatility ceiling consistent with the drawdown tolerance,
roughly max_drawdown x 0.8), max_position (decimal), n_holdings (int 8 to 20), cash_buffer (decimal 0 to 0.15),
themes (list chosen ONLY from: {themes}), exclude_tickers (list of US tickers), exclude_sectors (list),
benchmark ("QQQ" or "SPY"), constraints_summary (one sentence), assumptions (list of anything you had to assume).
Only use what the client said; when something is missing use the conservative default and list it under assumptions."""


def parse_profile(text: str) -> dict:
    prof = dict(config.DEFAULT_PROFILE)
    raw = parse_json(chat(PROFILE_SYSTEM.format(themes=", ".join(config.THEMES)), text, max_tokens=3000, json_mode=True))
    for k, v in raw.items():
        if v is not None and v != "":
            prof[k] = v
    # clamp to sane ranges so a bad parse can never break the optimiser
    prof["risk_tolerance"] = int(np.clip(int(prof["risk_tolerance"]), 1, 10))
    prof["max_drawdown"] = float(np.clip(float(prof["max_drawdown"]), 0.05, 0.8))
    prof["target_vol"] = float(np.clip(float(prof["target_vol"]), 0.08, 0.45))
    prof["max_position"] = float(np.clip(float(prof["max_position"]), 0.05, 0.3))
    prof["n_holdings"] = int(np.clip(int(prof["n_holdings"]), 6, 20))
    prof["cash_buffer"] = float(np.clip(float(prof["cash_buffer"]), 0, 0.15))
    prof["horizon_years"] = int(np.clip(int(prof["horizon_years"]), 1, 40))
    prof["themes"] = [t for t in prof.get("themes", []) if t in config.THEMES] or config.DEFAULT_PROFILE["themes"]
    prof["exclude_tickers"] = [str(t).upper() for t in prof.get("exclude_tickers", [])]
    return prof


# ---------------------------------------------------------------- 2. embeddings: thematic fit
def _embed(texts: list[str]) -> np.ndarray:
    cache_p = config.CACHE_DIR / "embeddings.pkl"
    cache = pickle.load(open(cache_p, "rb")) if cache_p.exists() else {}
    keys = [hashlib.md5((config.EMBED_MODEL + t).encode()).hexdigest() for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    for i in range(0, len(todo), 64):
        batch = todo[i:i + 64]
        resp = client().embeddings.create(input=[t[:6000] for _, t in batch], model=config.EMBED_MODEL)
        for (k, _), d in zip(batch, resp.data):
            cache[k] = np.array(d.embedding, dtype=np.float32)
    if todo:
        pickle.dump(cache, open(cache_p, "wb"))
    m = np.vstack([cache[k] for k in keys])
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def theme_matrix(summaries: pd.Series) -> pd.DataFrame:
    """Cosine similarity between every company's business description and every theme description."""
    s = summaries.dropna()
    s = s[s.str.len() > 40]
    comp = _embed(list(s.values))
    th = _embed(list(config.THEMES.values()))
    return pd.DataFrame(comp @ th.T, index=s.index, columns=list(config.THEMES))


# ---------------------------------------------------------------- 3. news sentiment
SENT_SYSTEM = """You are an equity news analyst. For each ticker you get recent headlines. Judge what they imply for the
company's fundamentals over the next 6 to 12 months (not the daily price move). Return ONLY JSON:
{"TICKER": {"score": number from -1 (very negative) to 1 (very positive), "driver": "max 12 words naming the specific
event", "n": number of headlines used}}. Ignore headlines that are generic market noise; if nothing material, score 0."""


def news_sentiment(news: dict[str, list[dict]]) -> dict:
    items = {t: [f"{n.get('date', '')} {n['title']}" for n in v[:8]] for t, v in news.items() if v}
    out = {}
    tickers = list(items)
    for i in range(0, len(tickers), 20):
        chunk = {t: items[t] for t in tickers[i:i + 20]}
        res = parse_json(chat(SENT_SYSTEM, json.dumps(chunk, indent=1), max_tokens=5000, json_mode=True))
        for t, v in res.items():
            if isinstance(v, dict) and t in chunk:
                v["score"] = float(np.clip(float(v.get("score", 0)), -1, 1))
                out[t] = v
    return out


# ---------------------------------------------------------------- 4. the AI investment committee
RULES = """Ground rules: use ONLY facts and numbers in the DATA PACKET. When you cite a number copy it exactly as it
appears in the packet (same rounding, same unit). Never invent figures, events, prices or forecasts. Be specific: name
tickers. Write plainly for a private client. Do not use hyphens or dashes as punctuation."""

AGENTS = {
    "Growth PM": "You are the Growth Portfolio Manager. Make the strongest evidence based case for this portfolio and each "
                 "holding: revenue growth, margins, momentum, theme fit. Flag any holding whose growth case is weak versus the "
                 "near misses. 250 words max.",
    "Risk Officer": "You are the Chief Risk Officer. Attack the portfolio against the client's own limits: concentration, "
                    "theme crowding (are we just long one AI factor?), drawdown in the stress tests versus the client's max "
                    "drawdown, valuation, loss making companies, high beta. Propose at most 3 concrete fixes, each as remove or "
                    "cap a named ticker with a reason. 250 words max.",
    "Macro Strategist": "You are the Macro Strategist. Using the macro snapshot and overlay rules, explain how rates, real "
                        "yields, credit and volatility affect these specific holdings. Name the most rate sensitive and most "
                        "cyclical holdings. 200 words max.",
}

CIO_SYSTEM = RULES + """
You are the Chief Investment Officer chairing the committee. You have the data packet and three memos. Decide.
Return ONLY a JSON object:
{
 "headline": "one sentence recommendation for the client",
 "portfolio_thesis": "3 to 4 sentences: why this mix suits this client",
 "suitability": "2 sentences checking it against the client's drawdown limit, horizon and exclusions, citing the stress test",
 "adjustments": [{"ticker": "...", "action": "remove" or "cap", "max_weight": decimal or null, "reason": "..."}],
 "holdings": [{"ticker": "...", "role": "Core compounder | Growth engine | Diversifier | Satellite",
               "why": "2 sentences citing at least two numbers from the packet", "key_risk": "1 sentence",
               "sell_trigger": "1 measurable condition that would make us sell"}],
 "key_risks": ["3 portfolio level risks"],
 "dissent": "1 to 2 sentences on where the Risk Officer disagreed and how you resolved it"
}
Rules: adjustments at most 3, only for current holdings, only if the risk case is genuinely convincing (an empty list is fine).
Include every current holding in "holdings". Do not mention portfolio weights or dollar amounts in "why" (they are set after
your adjustments by the optimiser)."""


def committee(packet: dict) -> dict:
    data = "DATA PACKET:\n" + json.dumps(packet, indent=1, default=str)

    def run(name):
        return name, chat(RULES + "\n" + AGENTS[name], data, max_tokens=4000)

    with ThreadPoolExecutor(max_workers=3) as ex:
        memos = dict(ex.map(run, AGENTS))
    memo_text = "\n\n".join(f"=== {k} memo ===\n{v}" for k, v in memos.items())
    decision = parse_json(chat(CIO_SYSTEM, data + "\n\n" + memo_text, max_tokens=8000, json_mode=True))
    return {"memos": memos, "decision": decision}


# ---------------------------------------------------------------- 5. ask the analyst
QA_SYSTEM = RULES + """
You are the client's AI analyst. Answer the question using the data packet of the recommended portfolio. If the packet
does not contain the answer, say so plainly and say which data would be needed. Max 180 words."""


def ask(question: str, packet: dict, history: list[dict]) -> str:
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
    return chat(QA_SYSTEM, f"DATA PACKET:\n{json.dumps(packet, default=str)}\n\nCONVERSATION SO FAR:\n{convo}\n\nQUESTION: {question}",
                max_tokens=3000)
