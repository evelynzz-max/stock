import streamlit as st
import pandas as pd
import time, random, json, re
from datetime import datetime, timedelta

from massive import RESTClient
from openai import OpenAI
from fpdf import FPDF


# ============================================================
# SETTINGS
# ============================================================
MAX_NEWS = 5          # 最多新闻条数
NEWS_TIMEOUT = 3      # 新闻 API 最大等待秒数
AI_TIMEOUT = 300       # DeepSeek 分析最大等待秒数

st.set_page_config(page_title="AI Stock Dashboard Pro", layout="wide")
st.title("📈 DeepSeek AI Stock Dashboard (No Chat Version)")
# ============================================================
# Load CSV
# ============================================================
df = pd.read_csv("ondo_stock.csv")
df["ticker"] = df["Stock Ticker"].str.strip()


# ============================================================
# API Clients
# ============================================================
MASSIVE_API_KEY = st.secrets["MASSIVE_API_KEY"]
massive = RESTClient(MASSIVE_API_KEY)

DEEPSEEK_KEY = st.secrets["DEEPSEEK_API_KEY"]
ai = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")


# ============================================================
# Logger
# ============================================================
log_box = st.empty()

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_box.write(f"🟦 **[{timestamp}] {msg}**")
    print(f"[{timestamp}] {msg}")


# ============================================================
# Utils
# ============================================================

def extract_json(text):
    """Extract JSON from DeepSeek output"""
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


# -------- Fetch News (with limit + timeout) --------
def fetch_news(ticker):
    log(f"Fetching news for {ticker} (limit={MAX_NEWS}, timeout={NEWS_TIMEOUT}s)...")
    start = time.time()

    try:
        resp = massive.list_ticker_news(
            ticker,
            limit=20,  # 先多获取，再过滤
            sort="published_utc",
            order="desc"
        )
    except Exception as e:
        log(f"❌ News fetch error: {e}")
        return []

    news_rows = []
    for n in resp:
        if len(news_rows) >= MAX_NEWS:
            break

        if time.time() - start > NEWS_TIMEOUT:
            log("⏳ News timeout reached.")
            break

        news_rows.append({
            "published": n.published_utc,
            "title": n.title,
            "url": n.article_url,
        })

    log(f"📰 News fetched: {len(news_rows)} items")
    return news_rows


# -------- Fetch Price History --------
def fetch_price_history(ticker, days=30):
    log(f"Fetching {days}d price history for {ticker}...")

    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    bars = []

    try:
        for bar in massive.list_aggs(
            ticker,
            1,
            "day",
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            adjusted="true",
            limit=days,
            sort="asc"
        ):
            bars.append(bar)
    except Exception as e:
        log(f"❌ Price fetch error: {e}")
        return None

    if not bars:
        log("❌ No price data returned.")
        return None

    df = pd.DataFrame([
        {
            "date": datetime.utcfromtimestamp(b.timestamp / 1000),
            "close": b.close,
            "volume": b.volume
        }
        for b in bars
    ])

    log(f"📈 Price rows fetched: {len(df)}")
    return df


# -------- DeepSeek AI Stock Analysis --------
def ai_stock_analysis(ticker, stock_meta, news_list, price_df):

    if price_df is None:
        return {"error": "No price data."}

    # Prepare metrics
    latest_price = float(price_df["close"].iloc[-1])
    pct_change_30 = round(
        (latest_price - float(price_df["close"].iloc[0])) /
        float(price_df["close"].iloc[0]) * 100,
        2
    )
    avg_volume = float(price_df["volume"].mean())
    daily_vol = float(price_df["volume"].iloc[-1])

    # Contract addresses
    eth = stock_meta["Ethereum Deployed Address"]
    bsc = stock_meta["BSC Deployed Address"]

    news_text = "\n".join(
        [f"- [{n['published']}] {n['title']}" for n in news_list]
    )

    prompt = f"""
You are a senior Wall Street equity analyst.

Analyze the stock using real market data + news + contract metadata.

Ticker: {ticker}
Ondo ETH Contract: {eth}
Ondo BSC Contract: {bsc}

Latest Price: {latest_price}
30D % Change: {pct_change_30}
Average Volume: {avg_volume}
Latest Volume: {daily_vol}

Recent News:
{news_text}

Return ONLY JSON:

{{
  "summary": "",
  "sentiment": "",
  "sector": "",
  "industry": "",
  "risk_level": "",
  "ai_tags": [],
  "key_drivers": [],
  "risk_factors": [],
  "short_term_watch": [],
  "medium_term_watch": [],
  "scores": {{
    "momentum": 0,
    "volatility": 0,
    "sentiment_score": 0,
    "macro_sensitivity": 0,
    "composite_rating": 0
  }}
}}
"""

    log("🤖 Calling DeepSeek V3.2 Reasoner...")

    start = time.time()
    try:
        resp = ai.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "enabled"}},
            timeout=AI_TIMEOUT  # timeout!
        )
    except Exception as e:
        return {"error": f"AI Error or Timeout: {e}"}

    raw = resp.choices[0].message.content
    log("📥 AI response received, parsing JSON...")

    json_text = extract_json(raw)

    if not json_text:
        return {"error": "Invalid AI JSON", "raw": raw}

    try:
        return json.loads(json_text)
    except:
        return {"error": "JSON parse error", "raw": raw}



# -------- PDF Export --------
def export_pdf(ticker, analysis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"AI Stock Report - {ticker}", ln=True)

    for k, v in analysis.items():
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(200, 10, txt=f"{k}:", ln=True)
        pdf.set_font("Arial", size=12)

        if isinstance(v, list):
            for item in v:
                pdf.multi_cell(0, 8, txt=f"- {item}")
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                pdf.multi_cell(0, 8, txt=f"{k2}: {v2}")
        else:
            pdf.multi_cell(0, 8, txt=str(v))

        pdf.ln(4)

    file_name = f"{ticker}_report.pdf"
    pdf.output(file_name)
    return file_name


# ============================================================
# Streamlit UI
# ============================================================


# Sidebar
st.sidebar.header("Stock Selector")
tickers = df["ticker"].tolist()
stock_selected = st.sidebar.selectbox("Pick a Stock", tickers)

stock_meta = df[df["ticker"] == stock_selected].iloc[0]

log("🚀 Pipeline started...")

# Fetch Data
price_df = fetch_price_history(stock_selected)
news_list = fetch_news(stock_selected)
analysis = ai_stock_analysis(stock_selected, stock_meta, news_list, price_df)

log("🎉 Pipeline completed.")


# ------------------------------------------------------------
# UI Rendering
# ------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 Basic Info")
    st.write(f"**Name:** {stock_meta['Name']}")
    st.write(f"**Ticker:** {stock_selected}")
    st.write(f"**Type:** {stock_meta['Type']}")
    st.write(f"**ETH Contract:** `{stock_meta['Ethereum Deployed Address']}`")
    st.write(f"**BSC Contract:** `{stock_meta['BSC Deployed Address']}`")

with col2:
    st.subheader("📊 30-Day Price")
    if price_df is not None:
        st.line_chart(price_df.set_index("date")[["close"]])
        st.bar_chart(price_df.set_index("date")[["volume"]])
    else:
        st.warning("No price data available")

# ------------------------------------------------------------
# AI Summary
# ------------------------------------------------------------
st.subheader("🤖 AI Stock Summary")

if "error" in analysis:
    st.error("AI Error: " + analysis["error"])
    if "raw" in analysis:
        st.write(analysis["raw"])
else:
    st.write("### Summary")
    st.write(analysis["summary"])

    st.write("### 🏷 AI Tags")
    st.write(", ".join(analysis["ai_tags"]))

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Sentiment", analysis["sentiment"])
        st.text("Sector: " + analysis["sector"])
    with cB:
        st.metric("Risk Level", analysis["risk_level"])
        st.text("Industry: " + analysis["industry"])
    with cC:
        st.write("### ⭐ Scores")
        for k, v in analysis["scores"].items():
            st.write(f"{k}: {v}")

    st.write("### 🚀 Key Drivers")
    st.write(analysis["key_drivers"])

    st.write("### ⚠️ Risk Factors")
    st.write(analysis["risk_factors"])

    st.write("### 🔍 Short-Term Watch")
    st.write(analysis["short_term_watch"])

    st.write("### 📈 Medium-Term Watch")
    st.write(analysis["medium_term_watch"])


# ------------------------------------------------------------
# PDF Export
# ------------------------------------------------------------
st.subheader("📄 Export Report")

if st.button("Export PDF"):
    file_name = export_pdf(stock_selected, analysis)
    st.success("Report generated!")
    with open(file_name, "rb") as f:
        st.download_button("Download PDF", f, file_name)
