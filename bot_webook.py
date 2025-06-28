import os
import io
import random
from flask import Flask, request
import telebot
from telebot.types import Update, InputFile
from sqlalchemy import create_engine, Column, BigInteger, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import feedparser
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Load environment ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

# --- Database setup ---
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

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

class Settings(Base):
    __tablename__ = 'settings'
    chat_id = Column(BigInteger, primary_key=True)
    interval = Column(Integer, default=300)
    timezone = Column(String, default='America/New_York')
    quiet_start = Column(String)
    quiet_end = Column(String)
    schedule_type = Column(String)
    schedule_time = Column(String)

Base.metadata.create_all(engine)

# --- Nitter RSS with fallback ---
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
    "https://nitter.moomoo.me",
    "https://nitter.pussthecat.org",
]
def get_twitter_rss(username):
    for base_url in NITTER_INSTANCES + ["https://twitrss.me/twitter_user_to_rss"]:
        try:
            if "nitter" in base_url:
                rss_url = f"{base_url}/{username}/rss"
            else:
                rss_url = f"{base_url}/?user={username}"
            feed = feedparser.parse(rss_url)
            if feed.entries:
                return feed.entries
        except Exception:
            continue
    return []

def get_latest_tweet(username):
    tweets = get_twitter_rss(username)
    if tweets:
        entry = tweets[0]
        return {'id': entry.link, 'text': entry.title, 'url': entry.link}
    return None

def get_tweets_for_query(query, limit=5):
    random.shuffle(NITTER_INSTANCES)
    for base_url in NITTER_INSTANCES:
        try:
            rss = feedparser.parse(f"{base_url}/search/rss?f=tweets&q={query}")
            if rss.entries:
                return [{"text": e.title, "url": e.link} for e in rss.entries[:limit]]
        except Exception:
            continue
    return []

# --- Flask webhook ---
@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    update = Update.de_json(request.stream.read().decode("utf-8"))
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/")
def index():
    return "Bot is alive!", 200

# --- Telegram handlers (see previous messages for all command implementations) ---
# ... ALL BOT HANDLERS FROM PREVIOUS CODE GO HERE ...
# For brevity, use the full bot_webhook.py code from the previous answer!

# --- Set webhook on startup ---
def set_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=f"{WEBHOOK_URL}/{BOT_TOKEN}")

if __name__ == "__main__":
    set_webhook()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)