import os
import io
import re
import random
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request
import telebot
from telebot.types import Update, InlineKeyboardMarkup, InlineKeyboardButton
from sqlalchemy import (
    create_engine,
    Column,
    BigInteger,
    String,
    Float,
    Integer,
    Boolean,
    UniqueConstraint,
    DateTime  # This was missing in original
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import requests
import openai
import google.generativeai as genai
from google.generativeai import configure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration and Initialization ---
load_dotenv()

class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    DATABASE_URL = os.getenv("DATABASE_URL")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    GEMINI_KEY = os.getenv("GEMINI_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    MONGODB_URI = os.getenv("MONGODB_URI")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
    RATE_LIMIT = int(os.getenv("RATE_LIMIT", 3))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", 60))
    AUTOSCAN_INTERVAL = int(os.getenv("AUTOSCAN_INTERVAL", 5))

# Initialize Flask and Telebot
app = Flask(__name__)
bot = telebot.TeleBot(Config.BOT_TOKEN, threaded=False)
update_lock = Lock()

# --- Database Models ---
Base = declarative_base()

class Tracked(Base):
    __tablename__ = 'tracked'
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger)
    username = Column(String)

class Keyword(Base):
    __tablename__ = 'keywords'
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger)
    keyword = Column(String)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger)
    ticker = Column(String)
    direction = Column(String)
    price = Column(Float)

class Portfolio(Base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger)
    ticker = Column(String)
    qty = Column(Float)
    price = Column(Float)

class UserSettings(Base):
    __tablename__ = 'user_settings'
    chat_id = Column(BigInteger, primary_key=True)
    autoscan_paused = Column(Boolean, default=False)
    scan_accounts = Column(Boolean, default=True)
    scan_keywords = Column(Boolean, default=True)
    scan_depth = Column(Integer, default=3)

class LastSeenUser(Base):
    __tablename__ = 'last_seen_user'
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger)
    username = Column(String)
    tweet_id = Column(String)
    __table_args__ = (UniqueConstraint('chat_id', 'username', name='uq_user_chat'),)

class LastSeenKeyword(Base):
    __tablename__ = 'last_seen_keyword'
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger)
    keyword = Column(String)
    tweet_id = Column(String)
    __table_args__ = (UniqueConstraint('chat_id', 'keyword', name='uq_keyword_chat'),)

class ProcessedUpdate(Base):
    __tablename__ = 'processed_updates'
    update_id = Column(Integer, primary_key=True)
    processed_at = Column(DateTime, default=datetime.utcnow)

# Initialize database
engine = create_engine(
    Config.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"sslmode": "require"}
)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(engine)

# Initialize MongoDB for analytics
mongo_client = MongoClient(Config.MONGODB_URI)
analytics_db = mongo_client.get_database("analytics")

# --- Helper Functions ---
def huggingface_generate(prompt):
    """Generate text using HuggingFace API"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        return response.json()[0]['generated_text']
    except Exception as e:
        logger.error(f"HuggingFace error: {e}")
        return "Sorry, I couldn't generate a response."

def get_company_overview(symbol):
    """Get company overview from Alpha Vantage"""
    try:
        # This would be replaced with actual Alpha Vantage API call
        return f"Company overview for {symbol} would appear here from Alpha Vantage"
    except Exception as e:
        logger.error(f"Error getting company overview: {e}")
        return "Could not retrieve company overview."

def get_insider_trades(symbol):
    """Get insider trades for a symbol"""
    try:
        # This would be replaced with actual insider trading data
        return [f"Insider trading data for {symbol} would appear here"]
    except Exception as e:
        logger.error(f"Error getting insider trades: {e}")
        return ["Could not retrieve insider trading data."]

def get_stock_news(symbol):
    """Get news for a stock"""
    try:
        # This would be replaced with actual news API call
        return [f"News for {symbol} would appear here"]
    except Exception as e:
        logger.error(f"Error getting stock news: {e}")
        return ["Could not retrieve news."]

# --- Rate Limiting ---
class RateLimiter:
    def __init__(self):
        self.user_limits = defaultdict(list)
    
    def check_limit(self, user_id, limit=Config.RATE_LIMIT, period=Config.RATE_LIMIT_PERIOD):
        now = datetime.now()
        # Clear old timestamps
        self.user_limits[user_id] = [
            t for t in self.user_limits[user_id] 
            if now - t < timedelta(seconds=period)
        ]
        if len(self.user_limits[user_id]) >= limit:
            return False
        self.user_limits[user_id].append(now)
        return True

limiter = RateLimiter()

# --- Twitter/Nitter Integration ---
class TwitterService:
    STATIC_INSTANCES = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.1d4.us",
        "https://nitter.moomoo.me",
        "https://nitter.pussthecat.org",
    ]
    
    EXTRA_INSTANCES = [
        "https://xcancel.com",
        "https://nitter.poast.org",
        "https://nitter.privacyredirect.com",
        "https://lightbrd.com",
        "https://nitter.space",
        "https://nitter.tiekoetter.com",
        "https://nitter.kareem.one",
        "https://nuku.trabun.org"
    ]
    
    @staticmethod
    def get_nitter_instances_from_html_status():
        url = "https://status.d420.de/"
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            instances = []
            table = soup.find("table")
            if not table:
                return instances
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                url_tag = cols[0].find("a")
                online = cols[1].text.strip()
                working = cols[2].text.strip()
                if url_tag and online == "✅" and working == "✅":
                    instance_url = url_tag["href"].rstrip("/")
                    if not instance_url.startswith("http"):
                        instance_url = "https://" + instance_url
                    instances.append(instance_url)
            return instances
        except Exception as e:
            logger.error(f"Error fetching Nitter status: {e}")
            return []
    
    @staticmethod
    def get_all_nitter_instances():
        scraped = TwitterService.get_nitter_instances_from_html_status()
        all_instances = set(scraped) | set(TwitterService.EXTRA_INSTANCES)
        if not all_instances:
            all_instances = set(TwitterService.EXTRA_INSTANCES) | set(TwitterService.STATIC_INSTANCES)
        return list(all_instances)
    
    @staticmethod
    def _fetch_feed(rss_url):
        try:
            feed = feedparser.parse(rss_url)
            if feed.entries:
                return feed.entries
        except Exception as e:
            logger.error(f"Error fetching feed from {rss_url}: {e}")
        return None
    
    @staticmethod
    def get_twitter_rss(username):
        try:
            rss_url = f"https://twiiit.com/{username}/rss"
            feed = feedparser.parse(rss_url)
            if feed.entries:
                return feed.entries
        except Exception as e:
            logger.error(f"Error with twiiit.com: {e}")
        
        instances = TwitterService.get_all_nitter_instances()
        urls = [f"{base}/{username}/rss" for base in instances]
        random.shuffle(urls)
        
        with ThreadPoolExecutor(max_workers=min(Config.MAX_WORKERS, len(urls))) as executor:
            future_to_url = {executor.submit(TwitterService._fetch_feed, url): url for url in urls}
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    return result
        
        for base_url in TwitterService.STATIC_INSTANCES + ["https://twitrss.me/twitter_user_to_rss"]:
            try:
                if "nitter" in base_url:
                    rss_url = f"{base_url}/{username}/rss"
                else:
                    rss_url = f"{base_url}/?user={username}"
                feed = feedparser.parse(rss_url)
                if feed.entries:
                    return feed.entries
            except Exception as e:
                logger.error(f"Error with {base_url}: {e}")
                continue
        return []
    
    @staticmethod
    def extract_image_url(entry):
        if hasattr(entry, 'media_content') and entry.media_content:
            return entry.media_content[0].get('url')
        elif hasattr(entry, 'links'):
            for link in entry.links:
                if link.get('type', '').startswith('image/'):
                    return link['href']
        return None
    
    @staticmethod
    def get_latest_tweet(username):
        tweets = TwitterService.get_twitter_rss(username)
        if tweets:
            return tweets[0]
        return None
    
    @staticmethod
    def get_tweets_for_query(query, limit=5):
        try:
            rss_url = f"https://twiiit.com/search/rss?f=tweets&q={query}"
            feed = feedparser.parse(rss_url)
            if feed.entries:
                return feed.entries[:limit]
        except Exception as e:
            logger.error(f"Error with twiiit.com search: {e}")
        
        instances = TwitterService.get_all_nitter_instances()
        random.shuffle(instances)
        
        for base_url in instances:
            try:
                rss = feedparser.parse(f"{base_url}/search/rss?f=tweets&q={query}")
                if rss.entries:
                    return rss.entries[:limit]
            except Exception as e:
                logger.error(f"Error with {base_url} search: {e}")
                continue
        
        for base_url in TwitterService.STATIC_INSTANCES:
            try:
                rss = feedparser.parse(f"{base_url}/search/rss?f=tweets&q={query}")
                if rss.entries:
                    return rss.entries[:limit]
            except Exception as e:
                logger.error(f"Error with static {base_url} search: {e}")
                continue
        return []

# --- Stock Market Utilities ---
class StockService:
    @staticmethod
    def get_stock_price(ticker):
        try:
            data = yf.Ticker(ticker)
            price = data.history(period="1d")['Close'][0]
            return price
        except Exception as e:
            logger.error(f"Error getting stock price for {ticker}: {e}")
            return None
    
    @staticmethod
    def get_stock_info(ticker):
        try:
            data = yf.Ticker(ticker)
            info = data.info
            return info
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {e}")
            return None
    
    @staticmethod
    def generate_stock_chart(ticker, period="1mo", interval="1d"):
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval)
            if data.empty:
                return None
            
            plt.figure(figsize=(10, 4))
            data['Close'].plot(title=f"{ticker} Close Price")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return buf
        except Exception as e:
            logger.error(f"Error generating chart for {ticker}: {e}")
            return None
    
    @staticmethod
    def generate_advanced_chart(ticker, period="1mo", rsi_period=14):
        try:
            data = yf.Ticker(ticker).history(period=period, interval="1d")
            if data.empty:
                return None
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot candlestick
            mpf.plot(data, type='candle', style='charles', ax=ax1, volume=ax2, show_nontrading=False)
            
            # Calculate and plot RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            ax2.plot(data.index, rsi, label='RSI', color='purple')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.axhline(70, color='red', linestyle='--')
            ax2.set_ylim(0, 100)
            ax2.legend()
            
            # Formatting
            ax1.set_title(f"{ticker} Price with RSI {rsi_period}")
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return buf
        except Exception as e:
            logger.error(f"Error generating advanced chart for {ticker}: {e}")
            return None
    
    @staticmethod
    def get_crypto_price(symbol):
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            price = ticker.history(period="1d")['Close'][0]
            return price
        except Exception as e:
            logger.error(f"Error getting crypto price for {symbol}: {e}")
            return None
    
    @staticmethod
    def analyze_sentiment(text):
        try:
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None

# --- AI Services ---
class AIService:
    @staticmethod
    def generate_with_azure(prompt):
        try:
            client = openai.AzureOpenAI(
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION,
                azure_endpoint=Config.AZURE_ENDPOINT,
            )
            response = client.chat.completions.create(
                model=Config.AZURE_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}")
            return None
    
    @staticmethod
    def generate_with_gemini(prompt):
        try:
            genai.configure(api_key=Config.GEMINI_KEY)
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None
    
    @staticmethod
    def generate_with_huggingface(prompt):
        try:
            API_URL = "https://api-inference.huggingface.co/models/gpt2"
            headers = {"Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}"}
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            return response.json()[0]['generated_text']
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return None
    
    @staticmethod
    def generate_response(prompt):
        # Try Azure first
        response = AIService.generate_with_azure(prompt)
        if response:
            return response
        
        # Fallback to Gemini
        response = AIService.generate_with_gemini(prompt)
        if response:
            return response
        
        # Final fallback to HuggingFace
        return AIService.generate_with_huggingface(prompt) or "Sorry, I couldn't generate a response."

# --- Portfolio Management ---
class PortfolioService:
    @staticmethod
    def add_stock(chat_id, ticker, qty, price):
        session = SessionLocal()
        try:
            stock = Portfolio(chat_id=chat_id, ticker=ticker, qty=qty, price=price)
            session.add(stock)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding stock: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def remove_stock(chat_id, ticker):
        session = SessionLocal()
        try:
            count = session.query(Portfolio).filter_by(chat_id=chat_id, ticker=ticker).delete()
            session.commit()
            return count > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing stock: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def get_portfolio(chat_id):
        session = SessionLocal()
        try:
            stocks = session.query(Portfolio).filter_by(chat_id=chat_id).all()
            return stocks
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return []
        finally:
            session.close()
    
    @staticmethod
    def calculate_portfolio_value(stocks):
        total = 0.0
        holdings = []
        for stock in stocks:
            value = stock.qty * stock.price
            holdings.append({
                'ticker': stock.ticker,
                'qty': stock.qty,
                'price': stock.price,
                'value': value
            })
            total += value
        return {
            'holdings': holdings,
            'total': total
        }

# --- Alert Management ---
class AlertService:
    @staticmethod
    def add_alert(chat_id, ticker, direction, price):
        session = SessionLocal()
        try:
            alert = Alert(chat_id=chat_id, ticker=ticker, direction=direction, price=price)
            session.add(alert)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding alert: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def remove_alert(chat_id, alert_id):
        session = SessionLocal()
        try:
            count = session.query(Alert).filter_by(chat_id=chat_id, id=alert_id).delete()
            session.commit()
            return count > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing alert: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def get_alerts(chat_id):
        session = SessionLocal()
        try:
            alerts = session.query(Alert).filter_by(chat_id=chat_id).all()
            return alerts
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
        finally:
            session.close()
    
    @staticmethod
    def check_alerts():
        session = SessionLocal()
        try:
            alerts = session.query(Alert).all()
            triggered = []
            
            # Group alerts by ticker to minimize API calls
            ticker_alerts = defaultdict(list)
            for alert in alerts:
                ticker_alerts[alert.ticker].append(alert)
            
            for ticker, alert_list in ticker_alerts.items():
                try:
                    current_price = StockService.get_stock_price(ticker)
                    if current_price is None:
                        continue
                    
                    for alert in alert_list:
                        if alert.direction == "ABOVE" and current_price > alert.price:
                            triggered.append((alert, current_price))
                        elif alert.direction == "BELOW" and current_price < alert.price:
                            triggered.append((alert, current_price))
                except Exception as e:
                    logger.error(f"Error checking alerts for {ticker}: {e}")
            
            return triggered
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
        finally:
            session.close()

# --- Twitter Account Management ---
class TwitterAccountService:
    @staticmethod
    def add_account(chat_id, username):
        session = SessionLocal()
        try:
            # Remove @ if present
            username = username.lstrip('@')
            
            if session.query(Tracked).filter_by(chat_id=chat_id, username=username).first():
                return False  # Already exists
            
            account = Tracked(chat_id=chat_id, username=username)
            session.add(account)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding account: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def remove_account(chat_id, username):
        session = SessionLocal()
        try:
            username = username.lstrip('@')
            count = session.query(Tracked).filter_by(chat_id=chat_id, username=username).delete()
            session.commit()
            return count > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing account: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def get_accounts(chat_id):
        session = SessionLocal()
        try:
            accounts = session.query(Tracked).filter_by(chat_id=chat_id).all()
            return [account.username for account in accounts]
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return []
        finally:
            session.close()
    
    @staticmethod
    def clear_accounts(chat_id):
        session = SessionLocal()
        try:
            count = session.query(Tracked).filter_by(chat_id=chat_id).delete()
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing accounts: {e}")
            return 0
        finally:
            session.close()

# --- Keyword Tracking ---
class KeywordService:
    @staticmethod
    def add_keyword(chat_id, keyword):
        session = SessionLocal()
        try:
            if session.query(Keyword).filter_by(chat_id=chat_id, keyword=keyword).first():
                return False  # Already exists
            
            kw = Keyword(chat_id=chat_id, keyword=keyword)
            session.add(kw)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding keyword: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def remove_keyword(chat_id, keyword):
        session = SessionLocal()
        try:
            count = session.query(Keyword).filter_by(chat_id=chat_id, keyword=keyword).delete()
            session.commit()
            return count > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing keyword: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def get_keywords(chat_id):
        session = SessionLocal()
        try:
            keywords = session.query(Keyword).filter_by(chat_id=chat_id).all()
            return [kw.keyword for kw in keywords]
        except Exception as e:
            logger.error(f"Error getting keywords: {e}")
            return []
        finally:
            session.close()
    
    @staticmethod
    def clear_keywords(chat_id):
        session = SessionLocal()
        try:
            count = session.query(Keyword).filter_by(chat_id=chat_id).delete()
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing keywords: {e}")
            return 0
        finally:
            session.close()

# --- User Settings ---
class UserSettingsService:
    @staticmethod
    def get_settings(chat_id):
        session = SessionLocal()
        try:
            settings = session.query(UserSettings).filter_by(chat_id=chat_id).first()
            if not settings:
                # Create default settings if none exist
                settings = UserSettings(
                    chat_id=chat_id,
                    autoscan_paused=False,
                    scan_accounts=True,
                    scan_keywords=True,
                    scan_depth=3
                )
                session.add(settings)
                session.commit()
            return settings
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
            return None
        finally:
            session.close()
    
    @staticmethod
    def update_settings(chat_id, **kwargs):
        session = SessionLocal()
        try:
            settings = session.query(UserSettings).filter_by(chat_id=chat_id).first()
            if not settings:
                settings = UserSettings(chat_id=chat_id)
                session.add(settings)
            
            for key, value in kwargs.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating settings: {e}")
            return False
        finally:
            session.close()
    
    @staticmethod
    def toggle_autoscan(chat_id):
        session = SessionLocal()
        try:
            settings = session.query(UserSettings).filter_by(chat_id=chat_id).first()
            if not settings:
                settings = UserSettings(chat_id=chat_id, autoscan_paused=True)
                session.add(settings)
                paused = True
            else:
                settings.autoscan_paused = not settings.autoscan_paused
                paused = settings.autoscan_paused
            session.commit()
            return paused
        except Exception as e:
            session.rollback()
            logger.error(f"Error toggling autoscan: {e}")
            return None
        finally:
            session.close()

# --- Autoscan Service ---
class AutoscanService:
    @staticmethod
    def clean_tweet_text(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def send_tweet_with_image(chat_id, entry, prefix):
        try:
            clean_text = AutoscanService.clean_tweet_text(entry.title)
            clean_url = entry.link.split('#')[0].split('?')[0]
            caption = f"{prefix}\n\n{clean_text}\n\n🔗 {clean_url}"
            
            if hasattr(entry, 'media_content') and entry.media_content:
                bot.send_photo(chat_id, entry.media_content[0]['url'], caption=caption)
            else:
                bot.send_message(chat_id, caption)
        except Exception as e:
            logger.error(f"Error sending tweet: {e}")
            # Fallback to minimal message
            bot.send_message(chat_id, f"{prefix}\n{entry.link.split('#')[0]}")
    
    @staticmethod
    def run_autoscan():
        logger.info("🍀 Starting autoscan with deduplication and rate limiting")
        session = SessionLocal()
        RATE_LIMIT_PER_RUN = 30
        sent_count = 0

        # Get all unique chat IDs that have tracked accounts or keywords
        all_chats = set(
            [r[0] for r in session.query(Tracked.chat_id).distinct()] +
            [r[0] for r in session.query(Keyword.chat_id).distinct()]
        )
        
        for chat_id in all_chats:
            settings = UserSettingsService.get_settings(chat_id)
            if not settings or settings.autoscan_paused:
                continue

            try:
                # Account autoscan
                if settings.scan_accounts:
                    users = session.query(Tracked).filter_by(chat_id=chat_id).all()
                    for user in users:
                        if sent_count >= RATE_LIMIT_PER_RUN:
                            logger.info("Rate limit reached. Pausing autoscan for this run.")
                            session.close()
                            return
                        
                        tweets = TwitterService.get_twitter_rss(user.username)
                        if not tweets:
                            continue
                        
                        last_seen = session.query(LastSeenUser).filter_by(
                            chat_id=chat_id, 
                            username=user.username
                        ).first()
                        
                        new_tweets = []
                        for entry in tweets[:settings.scan_depth]:
                            tweet_id = getattr(entry, "id", None) or entry.link
                            if last_seen and tweet_id == last_seen.tweet_id:
                                break
                            new_tweets.append((tweet_id, entry))
                        
                        # Send from oldest to newest
                        for tweet_id, entry in reversed(new_tweets):
                            try:
                                AutoscanService.send_tweet_with_image(
                                    chat_id, 
                                    entry, 
                                    f"🍀 Autoscan @{user.username}:"
                                )
                                sent_count += 1
                                
                                # Update last_seen after sending
                                if last_seen:
                                    last_seen.tweet_id = tweet_id
                                else:
                                    last_seen = LastSeenUser(
                                        chat_id=chat_id, 
                                        username=user.username, 
                                        tweet_id=tweet_id
                                    )
                                    session.add(last_seen)
                                session.commit()
                            except Exception as e:
                                logger.error(f"Error sending tweet for @{user.username} to chat {chat_id}: {e}")

                # Keyword autoscan
                if settings.scan_keywords and sent_count < RATE_LIMIT_PER_RUN:
                    keywords = session.query(Keyword).filter_by(chat_id=chat_id).all()
                    for kw in keywords:
                        if sent_count >= RATE_LIMIT_PER_RUN:
                            logger.info("Rate limit reached. Pausing autoscan for this run.")
                            session.close()
                            return
                        
                        tweets = TwitterService.get_tweets_for_query(kw.keyword, limit=settings.scan_depth)
                        if not tweets:
                            continue
                        
                        last_seen = session.query(LastSeenKeyword).filter_by(
                            chat_id=chat_id, 
                            keyword=kw.keyword
                        ).first()
                        
                        new_tweets = []
                        for entry in tweets:
                            tweet_id = getattr(entry, "id", None) or entry.link
                            if last_seen and tweet_id == last_seen.tweet_id:
                                break
                            new_tweets.append((tweet_id, entry))
                        
                        for tweet_id, entry in reversed(new_tweets):
                            try:
                                AutoscanService.send_tweet_with_image(
                                    chat_id, 
                                    entry, 
                                    f"🍀 Autoscan keyword '{kw.keyword}':"
                                )
                                sent_count += 1
                                
                                # Update last_seen after sending
                                if last_seen:
                                    last_seen.tweet_id = tweet_id
                                else:
                                    last_seen = LastSeenKeyword(
                                        chat_id=chat_id, 
                                        keyword=kw.keyword, 
                                        tweet_id=tweet_id
                                    )
                                    session.add(last_seen)
                                session.commit()
                            except Exception as e:
                                logger.error(f"Error sending keyword '{kw.keyword}' in chat {chat_id}: {e}")
            except Exception as e:
                logger.error(f"Error scanning chat {chat_id}: {e}")
        
        session.close()
        logger.info(f"🍀 Autoscan completed. Sent {sent_count} messages.")

# --- Alert Checker ---
def check_alerts_job():
    logger.info("🔔 Starting alert check")
    triggered_alerts = AlertService.check_alerts()
    
    for alert, current_price in triggered_alerts:
        try:
            direction = "above" if alert.direction == "ABOVE" else "below"
            message = (
                f"🚨 Alert triggered!\n"
                f"{alert.ticker} is now ${current_price:.2f} ({direction} ${alert.price:.2f})\n"
                f"Alert ID: {alert.id}"
            )
            bot.send_message(alert.chat_id, message)
            
            # Remove the alert after triggering
            AlertService.remove_alert(alert.chat_id, alert.id)
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    logger.info(f"🔔 Alert check completed. Found {len(triggered_alerts)} triggered alerts.")

# --- Webhook and Flask Routes ---
@app.route(f"/{Config.BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    if not request.is_json:
        return "Invalid content", 400
        
    with update_lock:
        update = Update.de_json(request.get_json())
        session = SessionLocal()
        try:
            if session.query(ProcessedUpdate).filter_by(update_id=update.update_id).first():
                return "OK", 200
            bot.process_new_updates([update])
            session.add(ProcessedUpdate(update_id=update.update_id))
            session.commit()
        except Exception as e:
            logger.error(f"Error processing update: {e}")
            session.rollback()
            return "Error", 500
        finally:
            session.close()
    return "OK", 200

@app.route("/")
def index():
    return "Bot is alive!", 200

@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    try:
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=f"{Config.WEBHOOK_URL}/{Config.BOT_TOKEN}")
        return "Webhook set successfully", 200
    except Exception as e:
        return f"Error setting webhook: {e}", 500

# --- Command Handlers ---
@bot.message_handler(commands=['start', 'help'])
def handle_help(message):
    help_text = """
🤖 *Finance & News Telegram Bot*

📊 *Stock Tools*
• `/price <TICKER>` - Current stock price
• `/info <TICKER>` - Company summary
• `/chart <TICKER> [PERIOD] [INTERVAL]` - Price chart
• `/graph <TICKER> [PERIOD] [RSI_PERIOD]` - Advanced chart
• `/news <TICKER>` - Recent news
• `/insider <TICKER>` - Insider trading

💰 *Portfolio Tracker*
• `/addstock <TICKER> <QTY> <PRICE>` - Add stock
• `/removestock <TICKER>` - Remove stock
• `/viewportfolio` - View holdings

🔔 *Price Alerts*
• `/alert <TICKER> ABOVE|BELOW <PRICE>` - Set alert
• `/listalerts` - View alerts
• `/removealert <ID>` - Delete alert

🐦 *Twitter Tracking*
• `/add @username` - Track account
• `/remove @username` - Untrack
• `/list` - Tracked accounts
• `/last @username` - Latest tweet
• `/top [N] @username` - Top tweets

🔍 *Keyword Tracking*
• `/addkeyword <word>` - Track keyword
• `/removekeyword <word>` - Untrack
• `/listkeywords` - Tracked keywords

🔄 *Autoscan Control*
• `/pause` - Pause autoscan
• `/resume` - Resume autoscan
• `/scanmode all|accounts|keywords` - Set mode
• `/setscandepth <1-20>` - Set depth
• `/myautoscan` - View settings

🧠 *AI Assistant*
• `/ai <prompt>` - Ask anything
• `/gpt`, `/gemini` - Aliases

Type /help to see this menu again.
"""
    bot.reply_to(message, help_text, parse_mode="Markdown")

@bot.message_handler(commands=['price'])
def price_handler(message):
    if not limiter.check_limit(message.from_user.id):
        return  # Silent ignore for rate limits
    
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "💲 Usage: /price <TICKER>")
        return
    
    ticker = args[1].upper()
    price = StockService.get_stock_price(ticker)
    
    if price is not None:
        bot.reply_to(message, f"💲 {ticker} price: ${price:.2f}")
    else:
        bot.reply_to(message, f"❌ Error fetching price for {ticker}.")

@bot.message_handler(commands=['info'])
def info_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "ℹ️ Usage: /info <TICKER>")
        return
    
    ticker = args[1].upper()
    info = StockService.get_stock_info(ticker)
    
    if info and 'longBusinessSummary' in info:
        bot.reply_to(message, f"ℹ️ {ticker} info:\n{info['longBusinessSummary']}")
    else:
        bot.reply_to(message, f"❌ Error fetching info for {ticker}.")

@bot.message_handler(commands=['chart'])
def chart_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "📊 Usage: /chart <TICKER> [PERIOD] [INTERVAL]")
        return
    
    ticker = args[1].upper()
    period = args[2] if len(args) > 2 else "1mo"
    interval = args[3] if len(args) > 3 else "1d"
    
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    
    if period not in valid_periods:
        bot.reply_to(message, f"❌ Invalid period. Valid: {', '.join(valid_periods)}")
        return
    if interval not in valid_intervals:
        bot.reply_to(message, f"❌ Invalid interval. Valid: {', '.join(valid_intervals)}")
        return
    
    chart = StockService.generate_stock_chart(ticker, period, interval)
    if chart:
        bot.send_photo(message.chat.id, chart, caption=f"📊 {ticker} Chart ({period}, {interval})")
    else:
        bot.reply_to(message, "❌ Error generating chart.")

@bot.message_handler(commands=['graph'])
def graph_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "📈 Usage: /graph <TICKER> [PERIOD] [RSI_PERIOD]")
        return
    
    ticker = args[1].upper()
    period = args[2] if len(args) > 2 else "1mo"
    rsi_period = args[3] if len(args) > 3 else 14
    
    try:
        rsi_period = int(rsi_period)
        if rsi_period < 5 or rsi_period > 30:
            raise ValueError("RSI period out of range")
    except ValueError:
        bot.reply_to(message, "❌ RSI period must be between 5 and 30")
        return
    
    chart = StockService.generate_advanced_chart(ticker, period, rsi_period)
    if chart:
        bot.send_photo(message.chat.id, chart, caption=f"📈 {ticker} Advanced Chart ({period}, RSI {rsi_period})")
    else:
        bot.reply_to(message, "❌ Error generating advanced chart.")

@bot.message_handler(commands=['crypto'])
def crypto_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "💰 Usage: /crypto <SYMBOL>")
        return
    
    symbol = args[1].upper()
    price = StockService.get_crypto_price(symbol)
    
    if price is not None:
        bot.reply_to(message, f"💰 {symbol} price: ${price:.2f}")
    else:
        bot.reply_to(message, f"❌ Error fetching price for {symbol}.")

@bot.message_handler(commands=['sentiment'])
def sentiment_handler(message):
    text = message.text[len("/sentiment "):].strip()
    if not text:
        bot.reply_to(message, "💬 Usage: /sentiment <text>")
        return
    
    sentiment = StockService.analyze_sentiment(text)
    if sentiment:
        response = (
            f"💬 Sentiment analysis:\n\n"
            f"Text: {text[:200]}...\n\n"
            f"Positive: {sentiment['pos']:.2f}\n"
            f"Neutral: {sentiment['neu']:.2f}\n"
            f"Negative: {sentiment['neg']:.2f}\n"
            f"Compound: {sentiment['compound']:.2f}"
        )
        bot.reply_to(message, response)
    else:
        bot.reply_to(message, "❌ Error analyzing sentiment.")

@bot.message_handler(commands=['tweets'])
def tweets_handler(message):
    query = message.text[len('/tweets '):].strip()
    if not query:
        bot.reply_to(message, "🐦 Usage: /tweets <query>")
        return
    
    entries = TwitterService.get_tweets_for_query(query)
    if not entries:
        bot.reply_to(message, "🐦 No tweets found.")
    else:
        for entry in entries[:5]:  # Limit to 5 tweets
            AutoscanService.send_tweet_with_image(message.chat.id, entry, f"🐦 Search: {query}")

# --- Portfolio Command Handlers ---
@bot.message_handler(commands=['addstock'])
def addstock_handler(message):
    args = message.text.split()
    if len(args) != 4:
        bot.reply_to(message, "💼 Usage: /addstock <TICKER> <QTY> <PRICE>")
        return
    
    ticker = args[1].upper()
    try:
        qty = float(args[2])
        price = float(args[3])
    except ValueError:
        bot.reply_to(message, "💼 QTY and PRICE must be numbers.")
        return
    
    if PortfolioService.add_stock(message.chat.id, ticker, qty, price):
        bot.reply_to(message, f"💼 Added {qty} {ticker} at ${price:.2f}.")
    else:
        bot.reply_to(message, f"❌ Error adding {ticker} to portfolio.")

@bot.message_handler(commands=['removestock'])
def removestock_handler(message):
    args = message.text.split()
    if len(args) != 2:
        bot.reply_to(message, "💼 Usage: /removestock <TICKER>")
        return
    
    ticker = args[1].upper()
    if PortfolioService.remove_stock(message.chat.id, ticker):
        bot.reply_to(message, f"💼 Removed {ticker} from portfolio.")
    else:
        bot.reply_to(message, f"💼 {ticker} not found in portfolio.")

@bot.message_handler(commands=['viewportfolio'])
def viewportfolio_handler(message):
    stocks = PortfolioService.get_portfolio(message.chat.id)
    if not stocks:
        bot.reply_to(message, "💼 Your portfolio is empty.")
        return
    
    portfolio = PortfolioService.calculate_portfolio_value(stocks)
    response = ["💼 Your Portfolio:"]
    
    for holding in portfolio['holdings']:
        response.append(
            f"{holding['qty']} {holding['ticker']} @ ${holding['price']:.2f} = ${holding['value']:.2f}"
        )
    
    response.append(f"\nTotal Value: ${portfolio['total']:.2f}")
    bot.reply_to(message, "\n".join(response))

# --- Alert Command Handlers ---
@bot.message_handler(commands=['alert'])
def alert_handler(message):
    args = message.text.split()
    if len(args) != 4:
        bot.reply_to(message, "🔔 Usage: /alert <TICKER> ABOVE|BELOW <PRICE>")
        return
    
    ticker = args[1].upper()
    direction = args[2].upper()
    try:
        price = float(args[3])
    except ValueError:
        bot.reply_to(message, "🔔 Price must be a number.")
        return
    
    if direction not in ("ABOVE", "BELOW"):
        bot.reply_to(message, "🔔 Direction must be ABOVE or BELOW.")
        return
    
    if AlertService.add_alert(message.chat.id, ticker, direction, price):
        bot.reply_to(message, f"🔔 Alert set for {ticker} {direction} ${price:.2f}")
    else:
        bot.reply_to(message, "❌ Error setting alert.")

@bot.message_handler(commands=['listalerts'])
def listalerts_handler(message):
    alerts = AlertService.get_alerts(message.chat.id)
    if not alerts:
        bot.reply_to(message, "🔔 No alerts set.")
        return
    
    response = ["🔔 Your Alerts:"]
    for alert in alerts:
        response.append(f"ID:{alert.id} {alert.ticker} {alert.direction} ${alert.price:.2f}")
    
    bot.reply_to(message, "\n".join(response))

@bot.message_handler(commands=['removealert'])
def removealert_handler(message):
    args = message.text.split()
    if len(args) != 2:
        bot.reply_to(message, "❌ Usage: /removealert <ID>")
        return
    
    try:
        alert_id = int(args[1])
    except ValueError:
        bot.reply_to(message, "❌ Alert ID must be a number.")
        return
    
    if AlertService.remove_alert(message.chat.id, alert_id):
        bot.reply_to(message, f"❌ Removed alert ID {alert_id}.")
    else:
        bot.reply_to(message, f"❌ No alert found with ID {alert_id}.")

# --- Twitter Account Command Handlers ---
@bot.message_handler(commands=['add'])
def add_account_handler(message):
    args = message.text.split()
    if len(args) != 2 or not args[1].startswith('@'):
        bot.reply_to(message, "🐦 Usage: /add @username")
        return
    
    username = args[1][1:]  # Remove @
    if TwitterAccountService.add_account(message.chat.id, username):
        bot.reply_to(message, f"🐦 Now tracking @{username}")
    else:
        bot.reply_to(message, f"🐦 Already tracking @{username}")

@bot.message_handler(commands=['remove'])
def remove_account_handler(message):
    args = message.text.split()
    if len(args) != 2 or not args[1].startswith('@'):
        bot.reply_to(message, "🐦 Usage: /remove @username")
        return
    
    username = args[1][1:]  # Remove @
    if TwitterAccountService.remove_account(message.chat.id, username):
        bot.reply_to(message, f"🐦 Stopped tracking @{username}")
    else:
        bot.reply_to(message, f"🐦 @{username} wasn't being tracked")

@bot.message_handler(commands=['list'])
def list_accounts_handler(message):
    accounts = TwitterAccountService.get_accounts(message.chat.id)
    if not accounts:
        bot.reply_to(message, "🐦 No accounts being tracked.")
        return
    
    response = "🐦 Tracked Accounts:\n" + "\n".join([f"• @{username}" for username in accounts])
    bot.reply_to(message, response)

@bot.message_handler(commands=['clear'])
def clear_accounts_handler(message):
    count = TwitterAccountService.clear_accounts(message.chat.id)
    bot.reply_to(message, f"🐦 Cleared {count} tracked accounts.")

@bot.message_handler(commands=['last'])
def last_tweet_handler(message):
    args = message.text.split()
    if len(args) < 2 or not args[1].startswith('@'):
        bot.reply_to(message, "🕓 Usage: /last @username")
        return
    
    username = args[1][1:]
    entry = TwitterService.get_latest_tweet(username)
    if not entry:
        bot.reply_to(message, f"🕓 No tweets found for @{username}.")
    else:
        AutoscanService.send_tweet_with_image(
            message.chat.id, 
            entry, 
            f"🕓 Latest from @{username}:"
        )

@bot.message_handler(commands=['top'])
def top_tweets_handler(message):
    args = message.text.split()
    n = 3
    username = None
    
    for arg in args[1:]:
        if arg.isdigit():
            n = min(int(arg), 10)  # Limit to 10 tweets
        elif arg.startswith('@'):
            username = arg[1:]
    
    if not username:
        bot.reply_to(message, "🏆 Usage: /top [N] @username")
        return
    
    tweets = TwitterService.get_twitter_rss(username)
    if not tweets:
        bot.reply_to(message, f"🏆 No tweets found for @{username}.")
        return
    
    for entry in tweets[:n]:
        AutoscanService.send_tweet_with_image(
            message.chat.id, 
            entry, 
            f"🏆 Top from @{username}:"
        )

# --- Keyword Command Handlers ---
@bot.message_handler(commands=['checkdb'])
def check_db_handler(message):
    try:
        # Create a test document
        test_document = {"_id": "test", "value": "MongoDB is working!"}
        
        # Insert the test document into a test collection
        analytics_db.test_collection.insert_one(test_document)
        
        # Retrieve the test document
        retrieved_document = analytics_db.test_collection.find_one({"_id": "test"})
        
        # Check if the retrieved document matches the inserted document
        if retrieved_document and retrieved_document["value"] == test_document["value"]:
            bot.reply_to(message, "✅ MongoDB is working! Test document was successfully stored and retrieved.")
        else:
            bot.reply_to(message, "❌ MongoDB test failed. Document could not be retrieved.")
        
        # Clean up: Remove the test document
        analytics_db.test_collection.delete_one({"_id": "test"})
    except Exception as e:
        bot.reply_to(message, f"❌ MongoDB error: {str(e)}")
        
@bot.message_handler(commands=['addkeyword'])
def add_keyword_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "🔍 Usage: /addkeyword <word>")
        return
    
    keyword = " ".join(args[1:])
    if KeywordService.add_keyword(message.chat.id, keyword):
        bot.reply_to(message, f"🔍 Now tracking keyword: {keyword}")
    else:
        bot.reply_to(message, f"🔍 Already tracking keyword: {keyword}")

@bot.message_handler(commands=['removekeyword'])
def remove_keyword_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "🔍 Usage: /removekeyword <word>")
        return
    
    keyword = " ".join(args[1:])
    if KeywordService.remove_keyword(message.chat.id, keyword):
        bot.reply_to(message, f"🔍 Stopped tracking keyword: {keyword}")
    else:
        bot.reply_to(message, f"🔍 Wasn't tracking keyword: {keyword}")

@bot.message_handler(commands=['listkeywords'])
def list_keywords_handler(message):
    keywords = KeywordService.get_keywords(message.chat.id)
    if not keywords:
        bot.reply_to(message, "🔍 No keywords being tracked.")
        return
    
    response = "🔍 Tracked Keywords:\n" + "\n".join([f"• {keyword}" for keyword in keywords])
    bot.reply_to(message, response)

@bot.message_handler(commands=['clearkeywords'])
def clear_keywords_handler(message):
    count = KeywordService.clear_keywords(message.chat.id)
    bot.reply_to(message, f"🔍 Cleared {count} tracked keywords.")

# --- AI Command Handlers ---
@bot.message_handler(commands=['ai', 'gpt', 'gemini'])
def ai_handler(message):
    user_input = message.text.partition(" ")[2]
    if not user_input:
        bot.reply_to(message, "🧠 Please provide a prompt after /ai.")
        return
    
    bot.reply_to(message, "💡 Thinking...")
    response = AIService.generate_response(user_input)
    bot.reply_to(message, response)

# --- Autoscan Command Handlers ---
@bot.message_handler(commands=['pause', 'pauseautoscan'])
def pause_autoscan_handler(message):
    if UserSettingsService.update_settings(message.chat.id, autoscan_paused=True):
        bot.reply_to(message, "⏸️ Autoscan paused.")
    else:
        bot.reply_to(message, "❌ Error pausing autoscan.")

@bot.message_handler(commands=['resume', 'resumeautoscan'])
def resume_autoscan_handler(message):
    if UserSettingsService.update_settings(message.chat.id, autoscan_paused=False):
        bot.reply_to(message, "▶️ Autoscan resumed.")
    else:
        bot.reply_to(message, "❌ Error resuming autoscan.")

@bot.message_handler(commands=['toggleautoscan'])
def toggle_autoscan_handler(message):
    paused = UserSettingsService.toggle_autoscan(message.chat.id)
    if paused is not None:
        if paused:
            bot.reply_to(message, "⏸️ Autoscan paused.")
        else:
            bot.reply_to(message, "▶️ Autoscan resumed.")
    else:
        bot.reply_to(message, "❌ Error toggling autoscan.")

@bot.message_handler(commands=['scanmode'])
def scanmode_handler(message):
    args = message.text.split()
    if len(args) < 2 or args[1].lower() not in ("all", "accounts", "keywords"):
        bot.reply_to(message, "🛠️ Usage: /scanmode <all|accounts|keywords>")
        return
    
    mode = args[1].lower()
    if mode == "all":
        settings = {'scan_accounts': True, 'scan_keywords': True}
    elif mode == "accounts":
        settings = {'scan_accounts': True, 'scan_keywords': False}
    else:  # keywords
        settings = {'scan_accounts': False, 'scan_keywords': True}
    
    if UserSettingsService.update_settings(message.chat.id, **settings):
        bot.reply_to(message, f"🛠️ Scan mode set to {mode}.")
    else:
        bot.reply_to(message, "❌ Error updating scan mode.")

@bot.message_handler(commands=['setscandepth'])
def setscandepth_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "🔢 Usage: /setscandepth <number_of_tweets>")
        return
    
    try:
        depth = int(args[1])
        if depth < 1 or depth > 20:
            raise ValueError("Depth out of range")
    except ValueError:
        bot.reply_to(message, "🔢 Scan depth must be between 1 and 20.")
        return
    
    if UserSettingsService.update_settings(message.chat.id, scan_depth=depth):
        bot.reply_to(message, f"🔢 Scan depth set to {depth}.")
    else:
        bot.reply_to(message, "❌ Error setting scan depth.")

@bot.message_handler(commands=['myautoscan'])
def myautoscan_handler(message):
    settings = UserSettingsService.get_settings(message.chat.id)
    if not settings:
        bot.reply_to(message, "⚙️ Error retrieving settings.")
        return
    
    status = "⏸️ Paused" if settings.autoscan_paused else "▶️ Active"
    mode = []
    if settings.scan_accounts: mode.append("accounts")
    if settings.scan_keywords: mode.append("keywords")
    
    response = (
        f"⚙️ Autoscan Settings:\n"
        f"Status: {status}\n"
        f"Scan Mode: {', '.join(mode) if mode else 'none'}\n"
        f"Scan Depth: {settings.scan_depth}"
    )
    bot.reply_to(message, response)

# --- Menu and Fallback ---
@bot.message_handler(commands=['menu'])
def menu_handler(message):
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("📊 Stocks", callback_data="stocks"),
        InlineKeyboardButton("🐦 Twitter", callback_data="twitter"),
    )
    markup.add(
        InlineKeyboardButton("🔔 Alerts", callback_data="alerts"),
        InlineKeyboardButton("💼 Portfolio", callback_data="portfolio"),
    )
    markup.add(
        InlineKeyboardButton("🧠 AI", callback_data="ai"),
        InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
    )
    bot.send_message(message.chat.id, "📱 Main Menu:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    if call.data == "stocks":
        help_text = (
            "📊 Stock Commands:\n"
            "/price <TICKER> - Current price\n"
            "/info <TICKER> - Company info\n"
            "/chart <TICKER> - Price chart\n"
            "/graph <TICKER> - Advanced chart\n"
            "/news <TICKER> - Latest news"
        )
        bot.send_message(call.message.chat.id, help_text)
    elif call.data == "twitter":
        help_text = (
            "🐦 Twitter Commands:\n"
            "/add @user - Track account\n"
            "/remove @user - Untrack\n"
            "/list - Tracked accounts\n"
            "/last @user - Latest tweet\n"
            "/top [N] @user - Top tweets\n"
            "/tweets <query> - Search"
        )
        bot.send_message(call.message.chat.id, help_text)
    elif call.data == "alerts":
        help_text = (
            "🔔 Alert Commands:\n"
            "/alert TICKER ABOVE|BELOW PRICE\n"
            "/listalerts - View alerts\n"
            "/removealert ID - Delete alert"
        )
        bot.send_message(call.message.chat.id, help_text)
    elif call.data == "portfolio":
        help_text = (
            "💼 Portfolio Commands:\n"
            "/addstock TICKER QTY PRICE\n"
            "/removestock TICKER\n"
            "/viewportfolio - View holdings"
        )
        bot.send_message(call.message.chat.id, help_text)
    elif call.data == "ai":
        bot.send_message(call.message.chat.id, "🧠 Ask me anything with /ai <prompt>")
    elif call.data == "settings":
        help_text = (
            "⚙️ Autoscan Settings:\n"
            "/pause - Pause scanning\n"
            "/resume - Resume scanning\n"
            "/scanmode - Set scan mode\n"
            "/setscandepth - Set depth\n"
            "/myautoscan - View settings"
        )
        bot.send_message(call.message.chat.id, help_text)
    else:
        bot.send_message(call.message.chat.id, "🤷 Unknown action. Try /help")

@bot.message_handler(func=lambda message: message.text and message.text.startswith("/"))
def unknown_command_handler(message):
    bot.reply_to(message, "❓ Unknown command. Use /help to see all available commands.")

# --- Scheduler Setup ---
scheduler = BackgroundScheduler()
scheduler.add_job(AutoscanService.run_autoscan, 'interval', minutes=Config.AUTOSCAN_INTERVAL)
scheduler.add_job(check_alerts_job, 'interval', minutes=1)  # Check alerts every minute
scheduler.start()

# --- Main Execution ---
if __name__ == "__main__":
    # Set webhook on startup
    try:
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=f"{Config.WEBHOOK_URL}/{Config.BOT_TOKEN}")
    except Exception as e:
        logger.error(f"Error setting webhook: {e}")
    
    # Start Flask app
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
