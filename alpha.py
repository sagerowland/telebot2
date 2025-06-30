# alpha.py
import os
import requests

ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def get_stock_price(symbol):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_API_KEY}"
    res = requests.get(url).json()
    quote = res.get("Global Quote", {})

    if not quote:
        return f"❌ No price data for {symbol}"

    price = quote.get("05. price", "N/A")
    change = quote.get("10. change percent", "N/A")
    return f"💵 {symbol.upper()} Price: ${price}\nChange: {change}"

def get_company_overview(symbol):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_API_KEY}"
    data = requests.get(url).json()

    if not data or "Name" not in data:
        return f"❌ No company overview for {symbol}"

    return f"""📊 {data.get('Name', symbol)} Overview
Sector: {data.get('Sector')}
Market Cap: {data.get('MarketCapitalization')}
P/E Ratio: {data.get('PERatio')}
EPS: {data.get('EPS')}"""
