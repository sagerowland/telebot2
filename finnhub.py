import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import requests
from azure_integrations import (
    analyze_sentiment,
    store_user_preferences,
    get_user_preferences,
    log_to_azure_blob
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configs
BINANCE_API_URL = "https://api.binance.com/api/v3"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
FINNHUB_API_URL = "https://finnhub.io/api/v1"
REQUEST_TIMEOUT = 10  # Increased timeout
RATE_LIMIT_DELAY = 3  # Seconds between API calls
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def make_api_request(url: str, params: Optional[Dict] = None, retries: int = 2) -> Optional[Dict]:
    """Generic API request helper with retry logic."""
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
            if attempt < retries:
                time.sleep(RATE_LIMIT_DELAY)
    return None

def get_crypto_price(symbol: str = "BTC") -> str:
    """Get crypto price with fallback from multiple APIs."""
    symbol = symbol.upper().strip()
    
    # Try Binance
    try:
        res = make_api_request(
            f"{BINANCE_API_URL}/ticker/price",
            params={"symbol": f"{symbol}USDT"}
        )
        if res and 'price' in res:
            return f"⚡ {symbol} = ${safe_float(res['price']):,.4f} (Binance)"
    except Exception as e:
        logger.error(f"Binance error: {e}")

    # Try CoinGecko
    try:
        res = make_api_request(
            f"{COINGECKO_API_URL}/simple/price",
            params={"ids": symbol.lower(), "vs_currencies": "usd"}
        )
        if res and symbol.lower() in res and 'usd' in res[symbol.lower()]:
            return f"🔄 {symbol} = ${safe_float(res[symbol.lower()]['usd']):,.4f} (CoinGecko)"
    except Exception as e:
        logger.error(f"CoinGecko error: {e}")

    # Try Finnhub
    if FINNHUB_API_KEY:
        try:
            res = make_api_request(
                f"{FINNHUB_API_URL}/crypto/price",
                params={"symbol": f"BINANCE:{symbol}USDT", "token": FINNHUB_API_KEY}
            )
            if res and 'price' in res:
                return f"📊 {symbol} = ${safe_float(res['price']):,.4f} (Finnhub)"
        except Exception as e:
            logger.error(f"Finnhub error: {e}")
    else:
        logger.warning("Finnhub API key not configured")

    return f"❌ Could not fetch {symbol} price"

def get_insider_trades(symbol: str) -> List[str]:
    """Get recent insider trades with enhanced formatting."""
    if not FINNHUB_API_KEY:
        return ["❌ Finnhub API key not configured"]

    try:
        res = make_api_request(
            f"{FINNHUB_API_URL}/stock/insider-transactions",
            params={"symbol": symbol.upper(), "token": FINNHUB_API_KEY}
        )
        
        if not res or not res.get('data'):
            return [f"❌ No insider trades for {symbol.upper()}"]
            
        trades = []
        for trade in res['data'][:3]:  # Limit to 3 most recent
            price = trade.get('price', 'N/A')
            price_str = f"${safe_float(price):,.2f}" if price != 'N/A' else price
            
            trades.append(
                f"📈 {trade.get('name', 'Unknown')}\n"
                f"Shares: {trade.get('share', 'N/A')}\n"
                f"Date: {trade.get('transactionDate', 'N/A')}\n"
                f"Type: {trade.get('transactionType', 'N/A')}\n"
                f"Price: {price_str}"
            )
        return trades
        
    except Exception as e:
        logger.error(f"Failed to fetch insider trades: {e}")
        return [f"❌ Failed to fetch insider trades for {symbol.upper()}"]

def get_stock_news(symbol: str, days_back: int = 3) -> List[str]:
    """Get stock news with sentiment analysis."""
    if not FINNHUB_API_KEY:
        return ["❌ Finnhub API key not configured"]

    try:
        today = datetime.now()
        from_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        res = make_api_request(
            f"{FINNHUB_API_URL}/company-news",
            params={
                "symbol": symbol.upper(),
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_API_KEY
            }
        )
        
        if not res:
            return [f"❌ No news found in the last {days_back} days"]
            
        news_items = []
        for item in res[:5]:  # Limit to 5 most recent
            if not (item.get('headline') and item.get('url')):
                continue
                
            headline = item['headline']
            url = item['url']
            
            # Get sentiment analysis
            sentiment = analyze_sentiment(headline)
            sentiment_emoji = "😊" if "positive" in sentiment.lower() else "😐" if "neutral" in sentiment.lower() else "😞"
            
            news_items.append(
                f"📰 {headline}\n"
                f"{sentiment_emoji} {sentiment}\n"
                f"🔗 {url}\n"
                f"📅 {item.get('datetime', 'N/A')}"
            )
            
        return news_items if news_items else ["❌ No valid news items"]
        
    except Exception as e:
        logger.error(f"Failed to fetch stock news: {e}")
        return [f"❌ Failed to fetch news for {symbol.upper()}"]
