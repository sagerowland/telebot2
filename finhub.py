# finnhub.py
import os
import requests

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def get_insider_trades(symbol):
    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()
    trades = res.get("data", [])

    if not trades:
        return [f"❌ No recent insider trades for {symbol}"]

    messages = []
    for trade in trades[:3]:  # latest 3 trades
        exec_name = trade.get("name", "Unknown")
        date = trade.get("transactionDate", "?")
        shares = trade.get("share", 0)
        code = trade.get("transactionCode", "?")
        messages.append(f"📈 Insider Trade ({symbol})\n👤 {exec_name}\n📅 {date}\n🔁 Code: {code}\n📊 Shares: {shares}")
    return messages

def get_crypto_price(symbol="BTC"):
    url = f"https://finnhub.io/api/v1/crypto/price?symbol=BINANCE:{symbol.upper()}USDT&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()
    price = res.get("price")

    return f"💰 {symbol.upper()} = ${price}" if price else "❌ Crypto not found"

def get_stock_news(symbol):
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to=2025-01-01&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()

    if not res:
        return ["❌ No news found"]

    return [f"📰 {item['headline']}\n🔗 {item['url']}" for item in res[:3]]
