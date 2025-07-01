import os
import requests
from datetime import datetime

# API Configs
BINANCE_API_URL = "https://api.binance.com/api/v3"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
FINNHUB_API_URL = "https://finnhub.io/api/v1"
REQUEST_TIMEOUT = 5  # seconds
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# --------------------------
# CRYPTO PRICE FUNCTIONS
# --------------------------
def get_crypto_price(symbol="BTC"):
    """Priority: Binance -> CoinGecko -> Finnhub"""
    symbol = symbol.upper().strip()
    
    # 1. Binance (No API key needed)
    try:
        url = f"{BINANCE_API_URL}/ticker/price?symbol={symbol}USDT"
        res = requests.get(url, timeout=REQUEST_TIMEOUT).json()
        if 'price' in res:
            return f"⚡ {symbol} = ${float(res['price']):,.4f} (Binance)"
    except Exception as e:
        print(f"Binance Error: {e}")

    # 2. CoinGecko (No API key needed)
    try:
        url = f"{COINGECKO_API_URL}/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        res = requests.get(url, timeout=REQUEST_TIMEOUT).json()
        if symbol.lower() in res and 'usd' in res[symbol.lower()]:
            return f"🔄 {symbol} = ${res[symbol.lower()]['usd']:,.4f} (CoinGecko)"
    except Exception as e:
        print(f"CoinGecko Error: {e}")

    # 3. Finnhub (Requires API key)
    if FINNHUB_API_KEY:
        try:
            url = f"{FINNHUB_API_URL}/crypto/price?symbol=BINANCE:{symbol}USDT&token={FINNHUB_API_KEY}"
            res = requests.get(url, timeout=REQUEST_TIMEOUT).json()
            if 'price' in res:
                return f"📊 {symbol} = ${res['price']:,.4f} (Finnhub)"
        except Exception as e:
            print(f"Finnhub Error: {e}")
    
    return f"❌ Could not fetch {symbol} price"

# --------------------------
# STOCK FUNCTIONS
# --------------------------
def get_insider_trades(symbol):
    """Get recent insider trades for a stock"""
    if not FINNHUB_API_KEY:
        return ["❌ Finnhub API key not configured"]
    
    try:
        url = f"{FINNHUB_API_URL}/stock/insider-transactions?symbol={symbol.upper()}&token={FINNHUB_API_KEY}"
        res = requests.get(url, timeout=REQUEST_TIMEOUT).json()
        
        if not res.get('data'):
            return [f"❌ No insider trades for {symbol.upper()}"]
            
        trades = []
        for trade in res['data'][:3]:  # Limit to 3 most recent
            trades.append(
                f"📈 {trade.get('name', 'Unknown')}\n"
                f"Shares: {trade.get('share', 'N/A')}\n"
                f"Date: {trade.get('transactionDate', 'N/A')}"
            )
        return trades
        
    except Exception as e:
        print(f"Insider Trades Error: {e}")
        return [f"❌ Failed to fetch insider trades for {symbol.upper()}"]

def get_stock_news(symbol):
    """Get recent stock news"""
    if not FINNHUB_API_KEY:
        return ["❌ Finnhub API key not configured"]
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"{FINNHUB_API_URL}/company-news?symbol={symbol.upper()}&from={today}&to={today}&token={FINNHUB_API_KEY}"
        res = requests.get(url, timeout=REQUEST_TIMEOUT).json()
        
        if not res:
            return ["❌ No news found today"]
            
        news = []
        for item in res[:3]:  # Limit to 3 most recent
            if item.get('headline') and item.get('url'):
                news.append(f"📰 {item['headline']}\n🔗 {item['url']}")
        return news if news else ["❌ No valid news items"]
        
    except Exception as e:
        print(f"Stock News Error: {e}")
        return [f"❌ Failed to fetch news for {symbol.upper()}"]
