"""Know Your Client: questionnaire answers -> risk score (willingness vs ability) -> investment profile."""
from __future__ import annotations

from . import config

RISK_WORDS = {"Very cautious": 1, "Cautious": 3, "Balanced": 5, "Growth": 7, "Aggressive": 9}
REACTION = {"Sell everything": 1, "Sell some to limit losses": 3, "Do nothing and wait": 6, "Buy more at lower prices": 9}
WEALTH_SHARE = {"Under 10% of my wealth": 9, "10 to 25%": 7, "25 to 50%": 5, "Over 50%": 3}
EXPERIENCE = {"None": 3, "Some funds or pension investing": 5, "Active investor in stocks": 7, "Professional": 9}
OBJECTIVES = ["Maximise long term growth", "Growth with some protection", "Balanced growth and income",
              "Preserve capital and beat inflation", "Generate regular income"]
EXCLUSIONS = {"Tobacco": ["Tobacco"], "Weapons and defence": ["Aerospace & Defense"], "Fossil fuels": ["Oil & Gas", "Coal"],
              "Gambling": ["Gambling"], "Alcohol": ["Beverages"]}

DEFAULT_ANSWERS = {
    "name": "Alex Morgan", "age": 32, "occupation": "Product engineer", "residence": "United Kingdom", "base_currency": "GBP",
    "dependants": "None",
    "objective": "Maximise long term growth", "horizon_years": 10,
    "goal_text": "Build long term wealth; possibly a house deposit in 7 to 10 years.",
    "withdrawals": "No withdrawals planned", "income_pct": 0, "liquidity_2y": 5000,
    "wealth_share": "25 to 50%", "emergency_fund": "Yes, 6 months or more", "income_stability": "Stable salary",
    "concentrated": "Yes", "concentrated_ticker": "NVDA", "concentrated_value": 40000,
    "risk_words": "Growth", "reaction": "Do nothing and wait", "max_loss": 0.30,
    "experience": "Some funds or pension investing", "options_knowledge": "A little",
    "themes": ["Artificial intelligence", "Cloud software", "Digital consumer platforms"],
    "exclusions": ["Tobacco"], "esg": "Somewhat important",
    "allowed_assets": config.DEFAULT_ALLOWED,
    "notes": "Wants to beat the Nasdaq 100 but would hate to lose a third of the money.",
}


def score(a: dict) -> dict:
    """Private banking rule: invest at the LOWER of risk willingness (attitude) and risk ability (capacity)."""
    lm = a.get("max_loss", 0.3)
    loss_score = max(1, min(10, round(lm / 0.06)))  # 6% -> 1, 30% -> 5, 60% -> 10
    willingness = round((RISK_WORDS.get(a["risk_words"], 5) + REACTION.get(a["reaction"], 5) + loss_score) / 3, 1)
    h = a.get("horizon_years", 10)
    ability = 9 if h > 10 else 7 if h > 5 else 4 if h > 3 else 2
    ability = (ability + WEALTH_SHARE.get(a["wealth_share"], 5)) / 2
    notes = []
    if not str(a.get("emergency_fund", "")).startswith("Yes"):
        ability -= 1.5
        notes.append("No emergency fund: part of the money may be needed at a bad time.")
    if a.get("income_stability") == "Variable or self employed":
        ability -= 1
        notes.append("Variable income lowers the ability to ride out losses.")
    if a.get("withdrawals") == "Regular income needed":
        ability -= 1.5
        notes.append("Regular withdrawals mean losses get locked in.")
    if a.get("liquidity_2y", 0) > 0.2 * config.INVESTMENT_AMOUNT:
        ability -= 1
        notes.append("More than 20% may be needed within 2 years.")
    if a.get("concentrated") == "Yes" and a.get("concentrated_value", 0) > 0.25 * config.INVESTMENT_AMOUNT:
        ability -= 0.5
        notes.append(f"Already holds a concentrated position in {a.get('concentrated_ticker') or 'employer shares'}.")
    ability = round(max(1, min(10, ability)), 1)
    rt = int(max(1, min(10, round(min(willingness, ability)))))
    return {"willingness": willingness, "ability": ability, "risk_tolerance": rt, "capacity_notes": notes,
            "contradictions": contradictions(a, willingness, ability)}


def contradictions(a: dict, willingness: float, ability: float) -> list[str]:
    c = []
    growthy = a["objective"] in OBJECTIVES[:2]
    if growthy and RISK_WORDS.get(a["risk_words"], 5) <= 3:
        c.append("Wants maximum growth but describes themselves as cautious.")
    if a["reaction"] in ("Sell everything", "Sell some to limit losses") and a.get("max_loss", 0) >= 0.25:
        c.append(f"Says they could accept a {a['max_loss']:.0%} fall, but would sell after a 20% fall.")
    if a.get("horizon_years", 10) >= 10 and a.get("liquidity_2y", 0) > 0.3 * config.INVESTMENT_AMOUNT:
        c.append("Long horizon, yet a large share of the money may be needed within 2 years.")
    if "options" in a.get("allowed_assets", []) and a.get("options_knowledge") == "No":
        c.append("Allows options for hedging but has not used options before.")
    if abs(willingness - ability) >= 3:
        c.append(f"Willingness to take risk ({willingness}/10) and ability to take risk ({ability}/10) are far apart.")
    if a.get("concentrated") == "Yes" and a.get("concentrated_ticker"):
        c.append(f"Already exposed to {a['concentrated_ticker']} through employer shares; adding more would concentrate risk.")
    return c


def fallback_questions(a: dict, sc: dict) -> list[str]:
    q = []
    for c in sc["contradictions"][:3]:
        if "cautious" in c:
            q.append("You want maximum growth but call yourself cautious. If the portfolio fell 15% in a year, how would you feel and what would you do?")
        elif "would sell" in c:
            q.append("You said you could accept a large fall but would sell after 20%. Which is closer to the truth?")
        elif "within 2 years" in c:
            q.append("How certain is the money you may need within 2 years, and what is it for?")
        elif "options" in c:
            q.append("Would you be comfortable with us using index put options purely as insurance, if we explain the cost upfront?")
        elif "employer" in c:
            q.append(f"Do you plan to keep your {a.get('concentrated_ticker')} shares, and are you happy for us to avoid adding more of that company?")
        elif "far apart" in c:
            q.append("Your appetite for risk and your financial capacity for it differ. Which matters more to you for this money?")
    if len(q) < 2:
        q.append("Is there anything that would make you need this money earlier than planned?")
    return q[:4]


def build_profile(a: dict, sc: dict, final: dict | None = None) -> dict:
    final = final or {}
    rt = int(final.get("risk_tolerance") or sc["risk_tolerance"])
    rt = max(sc["risk_tolerance"] - 2, min(sc["risk_tolerance"] + 2, rt))  # AI may adjust, but within 2 points
    max_dd = float(final.get("max_drawdown") or a.get("max_loss", 0.3))
    max_dd = max(0.05, min(0.6, max_dd))
    sectors = sorted({s for ex in a.get("exclusions", []) for s in EXCLUSIONS.get(ex, [ex])})
    excl_t = [a["concentrated_ticker"].upper()] if a.get("concentrated") == "Yes" and a.get("concentrated_ticker") else []
    allowed = list(a.get("allowed_assets") or config.DEFAULT_ALLOWED)
    if a.get("options_knowledge") == "No" and not final.get("options_ok"):
        allowed = [x for x in allowed if x != "options"]
    liquidity = min(0.6, max(0.02, a.get("liquidity_2y", 0) / config.INVESTMENT_AMOUNT))
    if a.get("withdrawals") == "Regular income needed":
        liquidity = max(liquidity, min(0.3, 2 * a.get("income_pct", 0)))
    prof = dict(config.DEFAULT_PROFILE)
    prof.update({
        "client_name": a.get("name") or "Client",
        "objective": final.get("objective") or f"{a['objective']} over {a['horizon_years']} years. {a.get('goal_text', '')}".strip(),
        "horizon_years": int(a["horizon_years"]), "risk_tolerance": rt, "max_drawdown": max_dd,
        "target_vol": config.target_vol_for(rt, max_dd), "max_position": 0.15 if rt >= 6 else 0.08,
        "n_holdings": 12 if rt >= 5 else 8, "cash_buffer": round(liquidity, 3),
        "themes": [t for t in a.get("themes", []) if t in config.THEMES] or config.DEFAULT_PROFILE["themes"],
        "exclude_tickers": excl_t, "exclude_sectors": sectors, "allowed_assets": allowed,
        "base_currency": a.get("base_currency", "USD"), "benchmark": "QQQ" if rt >= 7 else "SPY",
        "constraints_summary": final.get("summary", ""), "risk_rationale": final.get("rationale", ""),
        "willingness": sc["willingness"], "ability": sc["ability"], "kyc": a,
    })
    return prof
