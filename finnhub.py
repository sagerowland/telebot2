# finnhub.py
import os
import requests
from datetime import datetime

# No Binance API key needed for public endpoints
BINANCE_API_URL = "https://api.binance.com/api/v3"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")  # Only needed for stock functions

def get_crypto_price(symbol="BTC"):
    """Priority: Binance -> CoinGecko -> Finnhub"""
    symbol = symbol.upper().strip()
    
    # 1. Try Binance (no API key)
    try:
        url = f"{BINANCE_API_URL}/ticker/price?symbol={symbol}USDT"
        res = requests.get(url, timeout=5).json()
        return f"⚡ {symbol} = ${float(res['price']):,.4f} (Binance)"
    except Exception as e:
        print(f"Binance failed: {e}")

    # 2. Fallback to CoinGecko (no API key)
    try:
        url = f"{COINGECKO_API_URL}/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        res = requests.get(url, timeout=5).json()
        return f"🔄 {symbol} = ${res[symbol.lower()]['usd']:,.4f} (CoinGecko)"
    except Exception as e:
        print(f"CoinGecko failed: {e}")

    # 3. Final fallback to Finnhub (requires API key)
    if FINNHUB_API_KEY:
        try:
            url = f"https://finnhub.io/api/v1/crypto/price?symbol=BINANCE:{symbol}USDT&token={FINNHUB_API_KEY}"
            res = requests.get(url, timeout=5).json()
            return f"📊 {symbol} = ${res['price']:,.4f} (Finnhub)"
        except Exception as e:
            print(f"Finnhub failed: {e}")
    
    return f"❌ Could not fetch {symbol} price"

def get_insider_trades(symbol):
    if not FINNHUB_API_KEY:
        return ["❌ Finnhub API key not set"]
    try:
        url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_API_KEY}"
        res = requests.get(url).json()
        trades = res.get("data", [])
        if not trades:
            return [f"❌ No insider trades for {symbol}"]
        return [f"📈 {trade['name']} bought {trade['share']} shares" for trade in trades[:3]]
    except Exception as e:
        return [f"❌ Error fetching insider trades: {str(e)}"]
