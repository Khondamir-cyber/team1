"""Static configuration: investable universe, themes, scenarios, defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# Azure OpenAI (hackathon portal). Values come from env vars or .env, never hard coded keys.
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://app-uoaihack2ix6ak.azurewebsites.net")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-6.1-sol")
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-large")

INVESTMENT_AMOUNT = 100_000

# Liquid, US listed growth universe grouped by theme. Missing tickers are dropped at runtime.
UNIVERSE = {
    "AI semiconductors": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "ARM", "MU"],
    "AI infrastructure and power": ["ANET", "VRT", "ETN", "CEG"],
    "Cloud and cybersecurity software": ["MSFT", "NOW", "SNOW", "DDOG", "PLTR", "CRWD", "ZS", "PANW", "NET"],
    "Digital platforms": ["AMZN", "GOOGL", "META", "NFLX", "UBER", "DASH"],
    "Fintech and ecommerce": ["SHOP", "MELI", "NU", "XYZ"],
    "Healthcare innovation": ["LLY", "ISRG", "VRTX"],
    "Autonomy and robotics": ["TSLA"],
}
TICKER_THEME = {t: theme for theme, ts in UNIVERSE.items() for t in ts}
STOCKS = list(TICKER_THEME)

BENCHMARKS = ["QQQ", "SPY"]
CASH_PROXY = "SGOV"  # iShares 0 to 3 Month Treasury Bond ETF, used for the liquidity sleeve

# Defensive sleeve the optimiser can use to respect the client's risk limits.
# (description, long run premium over T bills assumed for the optimiser, max weight)
DEFENSIVE = {
    "SGOV": ("0 to 3 month US T bills", 0.000, 1.00),
    "SHY": ("1 to 3 year US Treasuries", 0.003, 0.40),
    "IEF": ("7 to 10 year US Treasuries", 0.010, 0.35),
    "TIP": ("Inflation linked US Treasuries", 0.010, 0.25),
    "GLD": ("Gold", 0.015, 0.10),
    "USMV": ("US minimum volatility equities", 0.040, 0.30),
}
ALL_TICKERS = STOCKS + BENCHMARKS + list(DEFENSIVE)


def target_vol_for(risk_tolerance: int, max_drawdown: float) -> float:
    """Volatility ceiling implied by the client's risk tolerance and drawdown limit (the tighter of the two)."""
    return round(min(0.06 + 0.025 * risk_tolerance, 0.8 * max_drawdown), 3)

# Theme descriptions the embedding model compares each company's business summary against.
THEMES = {
    "Artificial intelligence": "Companies building or selling artificial intelligence: AI accelerators and GPUs, AI models, AI software platforms, data centre compute for training and inference.",
    "Cloud software": "Cloud computing platforms, software as a service, data platforms, observability, developer tools and subscription enterprise software.",
    "Cybersecurity": "Cybersecurity: endpoint protection, network security, zero trust, identity and cloud security.",
    "Digital consumer platforms": "Online marketplaces, ecommerce, digital advertising, streaming, ride hailing and delivery apps with network effects.",
    "Fintech": "Digital payments, online banking, lending platforms and financial technology for merchants and consumers.",
    "Energy transition and power": "Electricity generation, nuclear power, grid equipment, electrification, data centre power and cooling.",
    "Healthcare innovation": "Innovative medicines such as obesity and diabetes drugs, gene therapies, and robotic surgery systems.",
    "Robotics and autonomy": "Electric vehicles, autonomous driving, robotics, humanoid robots and factory automation.",
}

# FRED series used for the macro overlay.
FRED_SERIES = {
    "DGS10": "10y Treasury yield (%)",
    "DGS2": "2y Treasury yield (%)",
    "DGS3MO": "3m Treasury yield (%)",
    "DFII10": "10y real yield, TIPS (%)",
    "CPIAUCSL": "CPI index",
    "UNRATE": "Unemployment rate (%)",
    "BAMLH0A0HYM2": "High yield credit spread (%)",
    "VIXCLS": "VIX",
}

# Historical stress windows replayed against today's portfolio.
STRESS_SCENARIOS = {
    "COVID crash (Feb to Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Rate shock bear market (2022)": ("2022-01-03", "2022-10-14"),
    "AI selloff, DeepSeek shock (Jan 2025)": ("2025-01-24", "2025-01-27"),
    "Tariff shock (Feb to Apr 2025)": ("2025-02-19", "2025-04-08"),
}

# Factor weights for the composite score (high growth mandate). Normalised at runtime.
DEFAULT_FACTOR_WEIGHTS = {
    "growth": 0.28,
    "momentum": 0.18,
    "theme": 0.14,
    "quality": 0.14,
    "analyst": 0.10,
    "valuation": 0.08,
    "sentiment": 0.08,
    "low_vol": 0.0,   # set from risk tolerance at runtime
}

DEFAULT_PROFILE_TEXT = (
    "Alex Morgan, 32, product engineer in London. Has a $100,000 lump sum from vested company shares "
    "and wants it to compound as fast as possible over the next 10+ years. No need to draw on the money. "
    "Comfortable with big swings and can stomach a 35% fall without selling, but a 50% fall would be too much. "
    "Believes AI, cloud software and digital platforms will drive the next decade. "
    "Wants listed, liquid stocks only: no crypto, no tobacco, no single stock above 15%. "
    "Wants to beat the Nasdaq 100 over time."
)

DEFAULT_PROFILE = {
    "client_name": "Alex Morgan",
    "objective": "Maximise long term capital growth; outperform the Nasdaq 100 over a full cycle.",
    "horizon_years": 10,
    "risk_tolerance": 8,          # 1 to 10
    "max_drawdown": 0.35,         # loss the client says they can sit through
    "target_vol": 0.28,           # annualised volatility ceiling
    "max_position": 0.15,
    "min_position": 0.03,
    "max_theme_weight": 0.40,
    "n_holdings": 12,
    "cash_buffer": 0.03,
    "themes": ["Artificial intelligence", "Cloud software", "Digital consumer platforms"],
    "exclude_tickers": [],
    "exclude_sectors": ["Tobacco"],
    "benchmark": "QQQ",
    "fractional_shares": True,
}
