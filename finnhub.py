import os
import requests
import logging
from datetime import datetime, timedelta
from time import sleep
from typing import List, Optional, Union

# --------------------------
# Configuration
# --------------------------
# API Configs
BINANCE_API_URL = "https://api.binance.com/api/v3"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
FINNHUB_API_URL = "https://finnhub.io/api/v1"
REQUEST_TIMEOUT = 5  # seconds
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Rate limiting (Finnhub: 30 calls/minute for free tier)
RATE_LIMIT_DELAY = 2  # seconds between API calls

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------
# Helper Functions
# --------------------------
def validate_symbol(symbol: str) -> str:
    """Validate and format symbol."""
    symbol = symbol.strip().upper()
    if not symbol.isalpha():
        raise ValueError(f"Invalid symbol: {symbol}")
    return symbol

def make_api_request(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 2
) -> Optional[dict]:
    """Generic API request helper with retry logic."""
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
            if attempt < retries:
                sleep(RATE_LIMIT_DELAY)
    return None

# --------------------------
# Crypto Price Functions
# --------------------------
def get_crypto_price(symbol: str = "BTC") -> str:
    """Get crypto price with fallback from multiple APIs.
    Priority: Binance -> CoinGecko -> Finnhub
    """
    try:
        symbol = validate_symbol(symbol)
    except ValueError as e:
        return f"❌ {e}"

    # 1. Try Binance
    try:
        url = f"{BINANCE_API_URL}/ticker/price"
        params = {"symbol": f"{symbol}USDT"}
        res = make_api_request(url, params=params)
        
        if res and "price" in res:
            price = float(res["price"])
            return f"⚡ {symbol} = ${price:,.4f} (Binance)"
    except Exception as e:
        logger.error(f"Binance failed: {e}")

    # 2. Try CoinGecko
    try:
        url = f"{COINGECKO_API_URL}/simple/price"
        params = {"ids": symbol.lower(), "vs_currencies": "usd"}
        res = make_api_request(url, params=params)
        
        if res and symbol.lower() in res and "usd" in res[symbol.lower()]:
            price = res[symbol.lower()]["usd"]
            return f"🔄 {symbol} = ${price:,.4f} (CoinGecko)"
    except Exception as e:
        logger.error(f"CoinGecko failed: {e}")

    # 3. Try Finnhub (if API key available)
    if FINNHUB_API_KEY:
        try:
            url = f"{FINNHUB_API_URL}/crypto/price"
            params = {
                "symbol": f"BINANCE:{symbol}USDT",
                "token": FINNHUB_API_KEY
            }
            res = make_api_request(url, params=params)
            
            if res and "price" in res:
                price = float(res["price"])
                return f"📊 {symbol} = ${price:,.4f} (Finnhub)"
        except Exception as e:
            logger.error(f"Finnhub failed: {e}")
    else:
        logger.warning("Finnhub API key not configured")

    return f"❌ Could not fetch {symbol} price from any source"

# --------------------------
# Stock Functions
# --------------------------
def get_insider_trades(symbol: str) -> List[str]:
    """Get recent insider trades for a stock."""
    if not FINNHUB_API_KEY:
        return ["❌ Finnhub API key not configured"]

    try:
        symbol = validate_symbol(symbol)
    except ValueError as e:
        return [f"❌ {e}"]

    try:
        url = f"{FINNHUB_API_URL}/stock/insider-transactions"
        params = {
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        }
        res = make_api_request(url, params=params)
        
        if not res or not res.get("data"):
            return [f"❌ No insider trades found for {symbol}"]
            
        trades = []
        for trade in res["data"][:3]:  # Limit to 3 most recent
            trade_info = [
                f"📈 {trade.get('name', 'Unknown')}",
                f"Shares: {trade.get('share', 'N/A')}",
                f"Date: {trade.get('transactionDate', 'N/A')}",
                f"Type: {trade.get('transactionType', 'N/A')}",
                f"Price: ${trade.get('price', 'N/A'):,.2f}"
            ]
            trades.append("\n".join(trade_info))
        return trades
        
    except Exception as e:
        logger.error(f"Failed to fetch insider trades: {e}")
        return [f"❌ Failed to fetch insider trades for {symbol}"]

def get_stock_news(
    symbol: str,
    days_back: int = 7,
    max_items: int = 3
) -> List[str]:
    """Get recent stock news."""
    if not FINNHUB_API_KEY:
        return ["❌ Finnhub API key not configured"]

    try:
        symbol = validate_symbol(symbol)
    except ValueError as e:
        return [f"❌ {e}"]

    try:
        today = datetime.now()
        from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        url = f"{FINNHUB_API_URL}/company-news"
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_API_KEY
        }
        res = make_api_request(url, params=params)
        
        if not res:
            return [f"❌ No news found for {symbol} in the last {days_back} days"]
            
        valid_news = [
            item for item in res
            if item.get("headline") and item.get("url")
        ][:max_items]
        
        if not valid_news:
            return [f"❌ No valid news items for {symbol}"]
            
        return [
            f"📰 {item['headline']}\n🔗 {item['url']}\n"
            f"📅 {item.get('datetime', 'N/A')}"
            for item in valid_news
        ]
        
    except Exception as e:
        logger.error(f"Failed to fetch stock news: {e}")
        return [f"❌ Failed to fetch news for {symbol}"]

# --------------------------
# Main Execution (for testing)
# --------------------------
if __name__ == "__main__":
    # Test crypto price
    print(get_crypto_price("BTC"))
    print(get_crypto_price("ETH"))
    print(get_crypto_price("INVALID!"))

    # Test stock functions
    print("\n".join(get_insider_trades("AAPL")))
    print("\n---\n".join(get_stock_news("TSLA"))))
