# 🧠 FIST AI Forex Bot

**FIST** (Fundamental, Intermarket, Sentiment, Technical) is an intelligent trading bot that combines multi-factor analysis and machine learning to make automated trading decisions on MetaTrader5. Includes a real-time dashboard powered by Streamlit.

---

## ⚙️ How It Works

This bot uses:
- 📊 **Technical Indicators**: RSI, MACD, ATR, EMA50/200
- 🌍 **Fundamental Data**: Fed rates via FRED, GDP/inflation via World Bank
- 📰 **Sentiment Analysis**: News headlines (TextBlob) + Google Trends
- 🔁 **Intermarket Correlation**: USDJPY alignment
- 🤖 **ML Model**: LightGBM classifier trained with SMOTE and TimeSeries CV
- 🔁 **Live Mode**: Places real trades using MT5 with ATR-based SL/TP
- 🧮 **Train Mode**: Automatically backtests and tunes hyperparameters

---

## 🖥️ Streamlit Dashboard

The included `dashboard.py` file provides a real-time interface to:

- View latest trades
- Monitor equity curve & performance
- Adjust key config values (symbol, confidence threshold, risk%)

---

## 📁 Project Structure

```
├── Trading_Algo_Full_V3_Structured.py
├── dashboard.py
├── requirements.txt
├── .env.example
```

---

## 🚀 Getting Started

1. Clone the repo:
```bash
git clone https://github.com/yourusername/fist-ai-forex-bot.git
cd fist-ai-forex-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env.example` and fill in your real values.

4. Run the bot:
```bash
# Train mode
python Trading_Algo_Full_V3_Structured.py

# Streamlit dashboard
streamlit run dashboard.py
```

---

## ⚠️ Disclaimer

This bot is for **educational and research purposes only**. It is not financial advice. Trading is risky — use at your own discretion.

---

## 📫 Contact

Made by **Ademola E Adegbola**  
Reach me via GitHub or LinkedIn!