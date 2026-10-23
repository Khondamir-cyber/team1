"""Run before the demo: python check_setup.py
Checks the Azure OpenAI key, the chat and embedding models, and Yahoo Finance access."""
import time

from advisor import config, llm

print(f"Endpoint: {config.AZURE_ENDPOINT}\nChat model: {config.CHAT_MODEL}\nEmbedding model: {config.EMBED_MODEL}\n")

if not llm.available():
    print("✗ No AZURE_OPENAI_API_KEY found. Copy .env.example to .env and paste the key.")
else:
    t = time.time()
    try:
        print("✓ Chat:", llm.chat("Reply with the single word OK.", "ping", max_tokens=500).strip()[:40], f"({time.time() - t:.1f}s)")
    except Exception as e:
        print("✗ Chat failed:", e)
    try:
        v = llm.client().embeddings.create(input=["test"], model=config.EMBED_MODEL).data[0].embedding
        print(f"✓ Embeddings: vector length {len(v)}")
    except Exception as e:
        print("✗ Embeddings failed:", e)

try:
    import yfinance as yf
    px = yf.download(["NVDA", "QQQ"], period="5d", progress=False, auto_adjust=True)["Close"]
    print(f"✓ Yahoo Finance: NVDA last close {px['NVDA'].dropna().iloc[-1]:.2f} on {px.index.max():%d %b %Y}")
except Exception as e:
    print("✗ Yahoo Finance failed:", e)
