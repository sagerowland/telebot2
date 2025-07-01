import os
import io
import random
import time
from flask import Flask, request
import telebot
from telebot.types import Update
from sqlalchemy import create_engine, Column, BigInteger, String, Float, Integer, Boolean, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import feedparser
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
import google.generativeai as genai
from google.generativeai import configure, GenerativeModel
import os
import telebot
from alpha import get_stock_price, get_company_overview
from finnhub import get_insider_trades, get_crypto_price, get_stock_news
import mplfinance as mpf
import matplotlib.dates as mdates
import sys
import logging

# --- Nitter instance discovery and fallback logic ---
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
STATIC_NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
    "https://nitter.moomoo.me",
    "https://nitter.pussthecat.org",
]

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
        print(f"Error fetching Nitter status: {e}")
        return []

def get_all_nitter_instances():
    scraped = get_nitter_instances_from_html_status()
    all_instances = set(scraped) | set(EXTRA_INSTANCES)
    if not all_instances:
        all_instances = set(EXTRA_INSTANCES) | set(STATIC_NITTER_INSTANCES)
    return list(all_instances)

def _fetch_feed(rss_url):
    try:
        feed = feedparser.parse(rss_url)
        if feed.entries:
            return feed.entries
    except Exception:
        pass
    return None

def get_twitter_rss(username):
    try:
        rss_url = f"https://twiiit.com/{username}/rss"
        feed = feedparser.parse(rss_url)
        if feed.entries:
            return feed.entries
    except Exception:
        pass
    instances = get_all_nitter_instances()
    urls = [f"{base}/{username}/rss" for base in instances]
    random.shuffle(urls)
    with ThreadPoolExecutor(max_workers=min(10, len(urls))) as executor:
        future_to_url = {executor.submit(_fetch_feed, url): url for url in urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                return result
    for base_url in STATIC_NITTER_INSTANCES + ["https://twitrss.me/twitter_user_to_rss"]:
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

def extract_image_url(entry):
    if hasattr(entry, 'media_content') and entry.media_content:
        return entry.media_content[0].get('url')
    elif hasattr(entry, 'links'):
        for link in entry.links:
            if link.get('type', '').startswith('image/'):
                return link['href']
    return None

def get_latest_tweet(username):
    tweets = get_twitter_rss(username)
    if tweets:
        return tweets[0]
    return None

def get_tweets_for_query(query, limit=5):
    try:
        rss_url = f"https://twiiit.com/search/rss?f=tweets&q={query}"
        feed = feedparser.parse(rss_url)
        if feed.entries:
            return feed.entries[:limit]
    except Exception:
        pass
    instances = get_all_nitter_instances()
    random.shuffle(instances)
    for base_url in instances:
        try:
            rss = feedparser.parse(f"{base_url}/search/rss?f=tweets&q={query}")
            if rss.entries:
                return rss.entries[:limit]
        except Exception:
            continue
    for base_url in STATIC_NITTER_INSTANCES:
        try:
            rss = feedparser.parse(f"{base_url}/search/rss?f=tweets&q={query}")
            if rss.entries:
                return rss.entries[:limit]
        except Exception:
            continue
    return []

# --- Load environment ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
GEMINI_KEY = os.getenv("GEMINI_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

configure(api_key=GEMINI_KEY)

gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

bot = telebot.TeleBot(BOT_TOKEN, threaded=False)
app = Flask(__name__)

# --- Database setup ---
Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"sslmode": "require"}
)
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

Base.metadata.create_all(engine)

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    raw = request.stream.read().decode("utf-8")
    print(f"Received update: {raw}")
    try:
        update = Update.de_json(raw)
        bot.process_new_updates([update])
    except Exception as e:
        print(f"Error processing update: {e}")
    return "OK", 200

@app.route("/")
def index():
    return "Bot is alive!", 200

def send_tweet_with_image(chat_id, entry, prefix):
    text = entry.title
    url = entry.link
    image_url = extract_image_url(entry)
    caption = f"{prefix}\n\n{text}\n{url}"
    try:
        if image_url:
            bot.send_photo(chat_id, image_url, caption=caption)
        else:
            bot.send_message(chat_id, caption)
    except Exception as e:
        print(f"Error sending message to chat {chat_id}: {e}")

# --- AUTOSCAN CONTROL COMMANDS ---
def pause_autoscan_handler(message):
    session = SessionLocal()
    settings = session.query(UserSettings).filter_by(chat_id=message.chat.id).first()
    if not settings:
        settings = UserSettings(chat_id=message.chat.id, autoscan_paused=True)
        session.add(settings)
    else:
        settings.autoscan_paused = True
    session.commit()
    session.close()
    bot.reply_to(message, "⏸️ Autoscan paused.")

def resume_autoscan_handler(message):
    session = SessionLocal()
    settings = session.query(UserSettings).filter_by(chat_id=message.chat.id).first()
    if not settings:
        settings = UserSettings(chat_id=message.chat.id, autoscan_paused=False)
        session.add(settings)
    else:
        settings.autoscan_paused = False
    session.commit()
    session.close()
    bot.reply_to(message, "▶️ Autoscan resumed.")

def toggle_autoscan_handler(message):
    session = SessionLocal()
    settings = session.query(UserSettings).filter_by(chat_id=message.chat.id).first()
    if not settings:
        settings = UserSettings(chat_id=message.chat.id, autoscan_paused=True)
        session.add(settings)
        paused = True
    else:
        settings.autoscan_paused = not settings.autoscan_paused
        paused = settings.autoscan_paused
    session.commit()
    session.close()
    if paused:
        bot.reply_to(message, "⏸️ Autoscan paused.")
    else:
        bot.reply_to(message, "▶️ Autoscan resumed.")

@bot.message_handler(commands=['pause', 'pauseautoscan'])
def pause_handler(message):
    pause_autoscan_handler(message)

@bot.message_handler(commands=['resume', 'resumeautoscan'])
def resume_handler(message):
    resume_autoscan_handler(message)

@bot.message_handler(commands=['toggleautoscan'])
def toggleautoscan_command_handler(message):
    toggle_autoscan_handler(message)

@bot.message_handler(commands=['scanmode'])
def scanmode_handler(message):
    args = message.text.split()
    if len(args) < 2 or args[1].lower() not in ("all", "accounts", "keywords"):
        bot.reply_to(message, "🛠️ Usage: /scanmode <all|accounts|keywords>")
        return
    mode = args[1].lower()
    session = SessionLocal()
    settings = session.query(UserSettings).filter_by(chat_id=message.chat.id).first()
    if not settings:
        settings = UserSettings(chat_id=message.chat.id)
        session.add(settings)
    if mode == "all":
        settings.scan_accounts = True
        settings.scan_keywords = True
    elif mode == "accounts":
        settings.scan_accounts = True
        settings.scan_keywords = False
    elif mode == "keywords":
        settings.scan_accounts = False
        settings.scan_keywords = True
    session.commit()
    session.close()
    bot.reply_to(message, f"🛠️ Scan mode set to {mode}.")

@bot.message_handler(commands=['setscandepth'])
def setscandepth_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "🔢 Usage: /setscandepth <number_of_tweets>")
        return
    try:
        depth = int(args[1])
        if depth < 1 or depth > 20:
            bot.reply_to(message, "🔢 Scan depth must be between 1 and 20.")
            return
        session = SessionLocal()
        settings = session.query(UserSettings).filter_by(chat_id=message.chat.id).first()
        if not settings:
            settings = UserSettings(chat_id=message.chat.id, scan_depth=depth)
            session.add(settings)
        else:
            settings.scan_depth = depth
        session.commit()
        session.close()
        bot.reply_to(message, f"🔢 Scan depth set to {depth}.")
    except Exception:
        bot.reply_to(message, "❌ Invalid scan depth.")

@bot.message_handler(commands=['myautoscan'])
def myautoscan_handler(message):
    session = SessionLocal()
    settings = session.query(UserSettings).filter_by(chat_id=message.chat.id).first()
    if not settings:
        bot.reply_to(message, "⚙️ No custom autoscan settings yet.")
        session.close()
        return
    status = "Paused" if settings.autoscan_paused else "Active"
    mode = []
    if settings.scan_accounts: mode.append("accounts")
    if settings.scan_keywords: mode.append("keywords")
    bot.reply_to(
        message,
        f"⚙️ Autoscan status: {status}\n"
        f"Scan: {', '.join(mode) if mode else 'none'}\n"
        f"Scan depth: {settings.scan_depth}"
    )
    session.close()

# --- AUTOSCAN SCHEDULER FUNCTION ---
def autoscan():
    print("🍀 Autoscan running with deduplication, rate-limit, error handling, and user controls!")
    session = SessionLocal()
    RATE_LIMIT_PER_RUN = 30
    sent_count = 0

    all_chats = set(
        [r[0] for r in session.query(Tracked.chat_id).distinct()] +
        [r[0] for r in session.query(Keyword.chat_id).distinct()]
    )
    for chat_id in all_chats:
        settings = session.query(UserSettings).filter_by(chat_id=chat_id).first()
        paused = settings.autoscan_paused if settings else False
        scan_accounts = settings.scan_accounts if settings else True
        scan_keywords = settings.scan_keywords if settings else True
        scan_depth = settings.scan_depth if settings else 3
        if paused:
            continue

        try:
            if scan_accounts:
                users = session.query(Tracked).filter_by(chat_id=chat_id).all()
                for user in users:
                    if sent_count >= RATE_LIMIT_PER_RUN:
                        print("Rate limit reached. Pausing autoscan for this run.")
                        session.close()
                        return
                    tweets = get_twitter_rss(user.username)
                    if not tweets:
                        continue
                    last_seen = session.query(LastSeenUser).filter_by(chat_id=chat_id, username=user.username).first()
                    new_tweets = []
                    for entry in tweets[:scan_depth]:
                        tweet_id = getattr(entry, "id", None) or entry.link
                        if last_seen and tweet_id == last_seen.tweet_id:
                            break
                        new_tweets.append((tweet_id, entry))
                    # Send from oldest to newest
                    for tweet_id, entry in reversed(new_tweets):
                        try:
                            send_tweet_with_image(chat_id, entry, f"🍀 Autoscan @{user.username}:")
                            sent_count += 1
                            # Always update last_seen after sending
                            if last_seen:
                                last_seen.tweet_id = tweet_id
                            else:
                                last_seen = LastSeenUser(chat_id=chat_id, username=user.username, tweet_id=tweet_id)
                                session.add(last_seen)
                            session.commit()
                        except Exception as e:
                            print(f"Error sending tweet for @{user.username} to chat {chat_id}: {e}")
    session.close()
    
                        return
                    tweets = get_tweets_for_query(kw.keyword, limit=scan_depth)
                    if not tweets:
                        continue
                    last_seen = session.query(LastSeenKeyword).filter_by(chat_id=chat_id, keyword=kw.keyword).first()
                    new_tweets = []
                    for entry in tweets:
                        tweet_id = getattr(entry, "id", None) or entry.link
                        if last_seen and tweet_id == last_seen.tweet_id:
                            break
                        new_tweets.append((tweet_id, entry))
                    for tweet_id, entry in reversed(new_tweets):
                        try:
                            send_tweet_with_image(chat_id, entry, f"🍀 Autoscan keyword '{kw.keyword}':")
                            sent_count += 1
                        except Exception as e:
                            print(f"Error sending keyword '{kw.keyword}' in chat {chat_id}: {e}")
                        if not last_seen:
                            last_seen = LastSeenKeyword(chat_id=chat_id, keyword=kw.keyword, tweet_id=tweet_id)
                            session.add(last_seen)
                        else:
                            last_seen.tweet_id = tweet_id
                        session.commit()
        except Exception as e:
            print(f"Error scanning chat {chat_id}: {e}")
    session.close()

# --- APScheduler setup ---
scheduler = BackgroundScheduler()
scheduler.add_job(autoscan, 'interval', minutes=5)
scheduler.start()

# --- Set webhook on startup ---
def set_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=f"{WEBHOOK_URL}/{BOT_TOKEN}")

set_webhook()

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message, "👋 Hello! I'm your finance & news bot.\nType /help to see what I can do.")

@bot.message_handler(commands=['help'])
def handle_help(message):
    help_text = """
🤖 *Telegram Bot Commands*

Welcome to your Twitter-scanning, stock-tracking, and AI-powered Telegram bot.

---

📊 *Stock Tools*
• `/price <TICKER>` — Get the current price of a stock (e.g., `/price TSLA`)
• `/info <TICKER>` — Company summary and info
• `/overview <TICKER>` — Detailed company overview (Alpha Vantage)
• `/chart <TICKER> [PERIOD] [INTERVAL]` — Show price chart (e.g. `1mo`, `1d`)
• `/graph <TICKER> [PERIOD] [RSI_PERIOD]` — Advanced candlestick chart with indicators
• `/news <TICKER>` — Get recent news for a stock
• `/insider <TICKER>` — Show recent insider trading activity

💰 *Cryptocurrency*
• `/crypto <SYMBOL>` — Get cryptocurrency price (e.g., `/crypto BTC`)

💼 *Portfolio Tracker*
• `/addstock <TICKER> <QTY> <PRICE>` — Add stock to your virtual portfolio
• `/removestock <TICKER>` — Remove a stock
• `/viewportfolio` — Show all holdings and total value

🔔 *Price Alerts*
• `/alert <TICKER> ABOVE|BELOW <PRICE>` — Set a price alert
• `/listalerts` — View your current alerts
• `/removealert <ID>` — Delete a specific alert

🧠 *AI Assistant*
• `/gemini <prompt>` — Ask anything with AI
• Aliases: `/ai`, `/gpt`

🐦 *Twitter Tracking*
• `/add @username` — Track a Twitter account
• `/remove @username` — Untrack a user
• `/list` — List tracked accounts
• `/clear` — Clear all tracked accounts
• `/last @username` — Show their most recent tweet
• `/top [N] [@username]` — Show top recent tweets

🔍 *Keyword Tracking*
• `/addkeyword <word>` — Track a keyword on Twitter
• `/removekeyword <word>` — Stop tracking it
• `/listkeywords` — View tracked keywords

💬 *Search & Sentiment*
• `/tweets <query>` — Search recent tweets
• `/sentiment <text>` — Analyze tone of text

🔄 *Autoscan Control*
• `/pause` or `/pauseautoscan` — Pause autoscan
• `/resume` or `/resumeautoscan` — Resume autoscan
• `/toggleautoscan` — Toggle scanning on/off
• `/scanmode all|accounts|keywords` — Choose what gets scanned
• `/setscandepth <1–20>` — Set depth of tweets per scan
• `/myautoscan` — View autoscan settings

🧹 *Data Management*
• `/cleardb` — Erase all tracked data (irreversible)

📱 *Interface*
• `/menu` — Show interactive button menu

🧪 *Coming Soon*
• `/setinterval` — Set scan interval for alerts (min 60s)
• `/setquiet` — Set quiet hours for notifications
• `/settimezone` — Set your timezone for reports
• `/setschedule` — Schedule daily/weekly reports
• `/mysettings` — View all your settings
• `/status` — Show bot status
• `/mute` — Mute notifications temporarily
• `/unmute` — Unmute notifications
• `/trending` — Show trending hashtags
• `/export` — Export tracked accounts/keywords
• `/import` — Import from CSV

🆘 *General*
• `/start` — Welcome message
• `/help` — Show this menu
"""
    bot.reply_to(message, help_text, parse_mode="Markdown")

@bot.message_handler(commands=['overview'])
def handle_overview(message):
    parts = message.text.split()
    symbol = parts[1] if len(parts) > 1 else "AAPL"
    bot.send_message(message.chat.id, get_company_overview(symbol))

@bot.message_handler(commands=['insider'])
def handle_insider(message):
    parts = message.text.split()
    symbol = parts[1] if len(parts) > 1 else "AAPL"
    for msg in get_insider_trades(symbol):
        bot.send_message(message.chat.id, msg)

@bot.message_handler(commands=['crypto'])
def handle_crypto(message):
    parts = message.text.split()
    symbol = parts[1] if len(parts) > 1 else "BTC"
    bot.send_message(message.chat.id, get_crypto_price(symbol))

@bot.message_handler(commands=['news'])
def handle_news(message):
    parts = message.text.split()
    symbol = parts[1] if len(parts) > 1 else "AAPL"
    for msg in get_stock_news(symbol):
        bot.send_message(message.chat.id, msg)

@bot.message_handler(commands=['price'])
def price_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "💲 Usage: /price <TICKER>")
        return
    ticker = args[1].upper()
    try:
        data = yf.Ticker(ticker)
        price = data.history(period="1d")['Close'][0]
        bot.reply_to(message, f"💲 {ticker} price: ${price:.2f}")
    except Exception as e:
        bot.reply_to(message, f"❌ Error fetching price for {ticker}.")

@bot.message_handler(commands=['info'])
def info_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "ℹ️ Usage: /info <TICKER>")
        return
    ticker = args[1].upper()
    try:
        data = yf.Ticker(ticker)
        info = data.info
        summary = info.get('longBusinessSummary', 'No info available.')
        bot.reply_to(message, f"ℹ️ {ticker} info:\n{summary}")
    except Exception:
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
        bot.reply_to(message, f"❌ Invalid period '{period}'. Valid: {', '.join(valid_periods)}")
        return
    if interval not in valid_intervals:
        bot.reply_to(message, f"❌ Invalid interval '{interval}'. Valid: {', '.join(valid_intervals)}")
        return
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty:
            bot.reply_to(message, "❌ No data found for this period/interval.")
            return
        plt.figure(figsize=(10,4))
        data['Close'].plot(title=f"{ticker} Close Price")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        bot.send_photo(message.chat.id, buf, caption=f"📊 {ticker} Chart ({period}, {interval})")
    except Exception:
        bot.reply_to(message, "❌ Error generating chart.")

@bot.message_handler(commands=['sentiment'])
def sentiment_handler(message):
    text = message.text[len("/sentiment "):].strip()
    if not text:
        bot.reply_to(message, "💬 Usage: /sentiment <text>")
        return
    try:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(text)
        bot.reply_to(
            message,
            f"💬 Sentiment for:\n{text}\n\n"
            f"Positive: {vs['pos']}\nNeutral: {vs['neu']}\nNegative: {vs['neg']}\nCompound: {vs['compound']}"
        )
    except Exception:
        bot.reply_to(message, "❌ Error analyzing sentiment.")

@bot.message_handler(commands=['tweets'])
def tweets_handler(message):
    query = message.text[len('/tweets '):].strip()
    if not query:
        bot.reply_to(message, "🐦 Usage: /tweets <query>")
        return
    entries = get_tweets_for_query(query)
    if not entries:
        bot.reply_to(message, "🐦 No tweets found.")
    else:
        for entry in entries:
            send_tweet_with_image(message.chat.id, entry, f"🐦 {query}:")

@bot.message_handler(commands=['add'])
def add_handler(message):
    args = message.text.split()
    if len(args) < 2 or not args[1].startswith('@'):
        bot.reply_to(message, "➕ Usage: /add @username")
        return
    username = args[1][1:]
    session = SessionLocal()
    exists = session.query(Tracked).filter_by(chat_id=message.chat.id, username=username).first()
    if exists:
        bot.reply_to(message, f"➕ @{username} is already tracked.")
    else:
        tracked = Tracked(chat_id=message.chat.id, username=username)
        session.add(tracked)
        session.commit()
        bot.reply_to(message, f"➕ Now tracking @{username}.")
    session.close()

@bot.message_handler(commands=['remove'])
def remove_handler(message):
    args = message.text.split()
    if len(args) < 2 or not args[1].startswith('@'):
        bot.reply_to(message, "➖ Usage: /remove @username")
        return
    username = args[1][1:]
    session = SessionLocal()
    count = session.query(Tracked).filter_by(chat_id=message.chat.id, username=username).delete()
    session.commit()
    if count:
        bot.reply_to(message, f"➖ Stopped tracking @{username}.")
    else:
        bot.reply_to(message, f"➖ @{username} was not being tracked.")
    session.close()

@bot.message_handler(commands=['list'])
def list_handler(message):
    session = SessionLocal()
    users = session.query(Tracked).filter_by(chat_id=message.chat.id).all()
    if users:
        bot.reply_to(message, "📋 Tracked Twitter accounts:\n" + "\n".join([f"@{u.username}" for u in users]))
    else:
        bot.reply_to(message, "📋 No Twitter accounts tracked.")
    session.close()

@bot.message_handler(commands=['clear'])
def clear_handler(message):
    session = SessionLocal()
    count = session.query(Tracked).filter_by(chat_id=message.chat.id).delete()
    session.commit()
    bot.reply_to(message, f"🧹 Removed {count} tracked Twitter accounts.")
    session.close()

@bot.message_handler(commands=['cleardb'])
def cleardb_handler(message):
    session = SessionLocal()
    session.query(Tracked).filter_by(chat_id=message.chat.id).delete()
    session.query(Keyword).filter_by(chat_id=message.chat.id).delete()
    session.query(Alert).filter_by(chat_id=message.chat.id).delete()
    session.query(Portfolio).filter_by(chat_id=message.chat.id).delete()
    session.commit()
    bot.reply_to(message, "🗑️ All your bot data has been erased. This cannot be undone.")
    session.close()

@bot.message_handler(commands=['addkeyword'])
def addkeyword_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "➕ Usage: /addkeyword word")
        return
    word = args[1]
    session = SessionLocal()
    exists = session.query(Keyword).filter_by(chat_id=message.chat.id, keyword=word).first()
    if exists:
        bot.reply_to(message, f"➕ '{word}' already tracked.")
    else:
        keyword = Keyword(chat_id=message.chat.id, keyword=word)
        session.add(keyword)
        session.commit()
        bot.reply_to(message, f"➕ Now tracking keyword '{word}'.")
    session.close()

@bot.message_handler(commands=['removekeyword'])
def removekeyword_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "➖ Usage: /removekeyword word")
        return
    word = args[1]
    session = SessionLocal()
    count = session.query(Keyword).filter_by(chat_id=message.chat.id, keyword=word).delete()
    session.commit()
    if count:
        bot.reply_to(message, f"➖ Removed keyword '{word}'.")
    else:
        bot.reply_to(message, f"➖ Keyword '{word}' was not tracked.")
    session.close()

@bot.message_handler(commands=['listkeywords'])
def listkeywords_handler(message):
    session = SessionLocal()
    keywords = session.query(Keyword).filter_by(chat_id=message.chat.id).all()
    if keywords:
        bot.reply_to(message, "📋 Tracked keywords:\n" + "\n".join([k.keyword for k in keywords]))
    else:
        bot.reply_to(message, "📋 No keywords tracked.")
    session.close()

@bot.message_handler(commands=['graph'])
def graph_handler(message):

    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "📈 Usage: /graph <TICKER> [PERIOD] [RSI_PERIOD]")
        return

    ticker = args[1].upper()
    period = args[2] if len(args) > 2 else "3mo"
    rsi_period = int(args[3]) if len(args) > 3 else 14

    valid_periods = [
        "1d", "5d", "1mo", "3mo", "6mo",
        "1y", "2y", "5y", "10y", "ytd", "max"
    ]

    if period not in valid_periods:
        bot.reply_to(message, f"❌ Invalid period '{period}'. Valid: {', '.join(valid_periods)}")
        return

    # Try yfinance
    try:
        data = yf.Ticker(ticker).history(period=period, interval="1d")
        if data.empty:
            raise Exception("No data from yfinance.")
    except Exception:
        # Fallback to Alpha Vantage
        bot.reply_to(message, "🔄 yfinance failed, trying Alpha Vantage...")
        ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={ALPHA_KEY}&outputsize=compact"
        r = requests.get(url)
        r.raise_for_status()
        json_data = r.json()
        time_series = json_data.get("Time Series (Daily)")
        if not time_series:
            bot.reply_to(message, "❌ Alpha Vantage returned no data.")
            return
        df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "6. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        data = df.sort_index()

    # Calculate moving averages
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # Calculate Bollinger Bands
    rolling_mean = data["Close"].rolling(window=20).mean()
    rolling_std = data["Close"].rolling(window=20).std()
    data["BB_upper"] = rolling_mean + (2 * rolling_std)
    data["BB_lower"] = rolling_mean - (2 * rolling_std)

    # Calculate RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Create additional plots
    apds = [
        mpf.make_addplot(data["SMA20"], color='blue', width=1),
        mpf.make_addplot(data["EMA20"], color='purple', width=1),
        mpf.make_addplot(data["BB_upper"], color='grey', linestyle='--'),
        mpf.make_addplot(data["BB_lower"], color='grey', linestyle='--'),
    ]

    # Create candlestick & volume figure
    fig, axes = mpf.plot(
        data,
        type='candle',
        volume=True,
        addplot=apds,
        returnfig=True,
        figsize=(12,8),
        style='yahoo',
        title=f"{ticker} Price, SMA, EMA, BB, Volume"
    )

    # RSI subplot below
    ax_rsi = fig.add_axes([0.1, 0.05, 0.8, 0.2])
    ax_rsi.plot(data.index, data["RSI"], label=f'RSI ({rsi_period})', color='orange')
    ax_rsi.axhline(70, color='red', linestyle='--')
    ax_rsi.axhline(30, color='green', linestyle='--')
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(True)
    ax_rsi.legend()

    # Save figure to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    bot.send_photo(message.chat.id, buf, caption=f"📈 {ticker} Candlestick Chart with SMA, EMA, Bollinger Bands, Volume, RSI ({rsi_period})")


@bot.message_handler(commands=['alert'])
def alert_handler(message):
    args = message.text.split()
    if len(args) < 4:
        bot.reply_to(message, "🔔 Usage: /alert TICKER <ABOVE|BELOW> <PRICE>")
        return
    ticker = args[1].upper()
    direction = args[2].upper()
    try:
        price = float(args[3])
    except ValueError:
        bot.reply_to(message, "🔔 Invalid price.")
        return
    if direction not in ("ABOVE", "BELOW"):
        bot.reply_to(message, "🔔 Direction must be ABOVE or BELOW.")
        return
    session = SessionLocal()
    alert = Alert(chat_id=message.chat.id, ticker=ticker, direction=direction, price=price)
    session.add(alert)
    session.commit()
    bot.reply_to(message, f"🔔 Alert set: {ticker} {direction} ${price:.2f}")
    session.close()

@bot.message_handler(commands=['listalerts'])
def listalerts_handler(message):
    session = SessionLocal()
    alerts = session.query(Alert).filter_by(chat_id=message.chat.id).all()
    if alerts:
        lines = [f"{a.id}: {a.ticker} {a.direction} ${a.price:.2f}" for a in alerts]
        bot.reply_to(message, "📋 Your alerts:\n" + "\n".join(lines))
    else:
        bot.reply_to(message, "📋 No alerts set.")
    session.close()

@bot.message_handler(commands=['removealert'])
def removealert_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "❌ Usage: /removealert ID")
        return
    try:
        alert_id = int(args[1])
    except ValueError:
        bot.reply_to(message, "❌ Invalid alert ID.")
        return
    session = SessionLocal()
    count = session.query(Alert).filter_by(chat_id=message.chat.id, id=alert_id).delete()
    session.commit()
    if count:
        bot.reply_to(message, f"❌ Alert {alert_id} removed.")
    else:
        bot.reply_to(message, f"❌ No alert with ID {alert_id}.")
    session.close()

@bot.message_handler(commands=['addstock'])
def addstock_handler(message):
    args = message.text.split()
    if len(args) < 4:
        bot.reply_to(message, "💰 Usage: /addstock TICKER QTY PRICE")
        return
    ticker = args[1].upper()
    try:
        qty = float(args[2])
        price = float(args[3])
    except ValueError:
        bot.reply_to(message, "💰 Invalid quantity or price.")
        return
    session = SessionLocal()
    entry = Portfolio(chat_id=message.chat.id, ticker=ticker, qty=qty, price=price)
    session.add(entry)
    session.commit()
    bot.reply_to(message, f"💰 Added {qty} {ticker} at ${price:.2f} to portfolio.")
    session.close()

@bot.message_handler(commands=['removestock'])
def removestock_handler(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "🗑️ Usage: /removestock TICKER")
        return
    ticker = args[1].upper()
    session = SessionLocal()
    count = session.query(Portfolio).filter_by(chat_id=message.chat.id, ticker=ticker).delete()
    session.commit()
    if count:
        bot.reply_to(message, f"🗑️ Removed {ticker} from portfolio.")
    else:
        bot.reply_to(message, f"🗑️ {ticker} not found in portfolio.")
    session.close()

@bot.message_handler(commands=['viewportfolio'])
def viewportfolio_handler(message):
    session = SessionLocal()
    entries = session.query(Portfolio).filter_by(chat_id=message.chat.id).all()
    if not entries:
        bot.reply_to(message, "📊 Portfolio is empty.")
        session.close()
        return
    lines = []
    total = 0.0
    for entry in entries:
        lines.append(f"{entry.ticker}: {entry.qty} @ ${entry.price:.2f}")
        total += entry.qty * entry.price
    bot.reply_to(message, "📊 Your portfolio:\n" + "\n".join(lines) + f"\nTotal invested: ${total:.2f}")
    session.close()
    
@bot.message_handler(commands=['ai', 'gpt', 'gemini'])
def handle_gemini(message):
    user_input = message.text.partition(" ")[2]
    if not user_input:
        bot.reply_to(message, "Please provide a prompt after /gemini.")
        return

    bot.reply_to(message, "💡 Thinking...")

    try:
        response = gemini_model.generate_content(user_input)
        bot.reply_to(message, response.text)
    except Exception as e:
        if "429" in str(e):
            bot.reply_to(message, "⚠️ Gemini rate limit reached. Switching to Hugging Face...")
            hf_output = huggingface_generate(user_input)
            bot.reply_to(message, hf_output)
        else:
            bot.reply_to(message, f"Gemini error: {str(e)}")
        
def huggingface_generate(prompt):
    API_URL = "https://api-inference.huggingface.co/models/distilbert/distilgpt2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 100}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        generated_text = response.json()[0]['generated_text']
        return generated_text
    else:
        print("Hugging Face API ERROR:")
        print("Status code:", response.status_code)
        print("Response body:", response.text)
        return "⚠️ Hugging Face API error. Try again later."
        
@bot.message_handler(commands=['setinterval'])
def setinterval_handler(message):
    bot.reply_to(message, "⏱️ Setinterval is not yet implemented. (Will set scan interval for alerts, minimum 60s)")

@bot.message_handler(commands=['setquiet'])
def setquiet_handler(message):
    bot.reply_to(message, "🤫 Setquiet is not yet implemented. (Will set quiet hours for notifications)")

@bot.message_handler(commands=['settimezone'])
def settimezone_handler(message):
    bot.reply_to(message, "🗺️ Settimezone is not yet implemented. (Will set your timezone for reports)")

@bot.message_handler(commands=['setschedule'])
def setschedule_handler(message):
    bot.reply_to(message, "🗓️ Setschedule is not yet implemented. (Will schedule daily/weekly reports)")

@bot.message_handler(commands=['mysettings'])
def mysettings_handler(message):
    bot.reply_to(message, "⚙️ Mysettings is not yet implemented. (Will show your settings)")

@bot.message_handler(commands=['status'])
def status_handler(message):
    bot.reply_to(message, "🚦 Status is not yet implemented. (Will show bot status)")

@bot.message_handler(commands=['mute'])
def mute_handler(message):
    bot.reply_to(message, "🔇 Mute is not yet implemented. (Will mute notifications for a user)")

@bot.message_handler(commands=['unmute'])
def unmute_handler(message):
    bot.reply_to(message, "🔊 Unmute is not yet implemented. (Will unmute notifications for a user)")

@bot.message_handler(commands=['last'])
def last_handler(message):
    args = message.text.split()
    if len(args) < 2 or not args[1].startswith('@'):
        bot.reply_to(message, "Usage: /last @username")
        return
    username = args[1][1:]
    entry = get_latest_tweet(username)
    if entry:
        send_tweet_with_image(message.chat.id, entry, f"🐦 Last tweet from @{username}:")
    else:
        bot.reply_to(message, f"❌ Could not retrieve tweets for @{username}. (Account may be protected, rate-limited, or unavailable.)")

@bot.message_handler(commands=['top'])
def top_handler(message):
    args = message.text.split()
    limit = 5
    username = None
    if len(args) > 1:
        if args[1].startswith('@'):
            username = args[1][1:]
        else:
            try:
                limit = int(args[1])
                if len(args) > 2 and args[2].startswith('@'):
                    username = args[2][1:]
            except ValueError:
                if args[1].startswith('@'):
                    username = args[1][1:]
    session = SessionLocal()
    if username:
        users = session.query(Tracked).filter_by(chat_id=message.chat.id, username=username).all()
    else:
        users = session.query(Tracked).filter_by(chat_id=message.chat.id).all()
    session.close()
    if not users:
        bot.reply_to(message, "📋 No Twitter accounts tracked." if not username else f"📋 @{username} is not tracked.")
        return
    tweets = []
    for user in users:
        user_tweets = get_twitter_rss(user.username)
        if user_tweets:
            tweets.extend([(user.username, e) for e in user_tweets[:limit]])
    tweets = [(u, e) for u, e in tweets if hasattr(e, "published_parsed")]
    tweets.sort(key=lambda x: x[1].published_parsed, reverse=True)
    if not tweets:
        bot.reply_to(message, "🐦 No recent tweets found.")
    else:
        for u, e in tweets[:limit]:
            send_tweet_with_image(message.chat.id, e, f"🐦 @{u}:")

@bot.message_handler(commands=['trending'])
def trending_handler(message):
    bot.reply_to(message, "🔥 Trending is not yet implemented. (Will show top trending hashtags)")

@bot.message_handler(commands=['export'])
def export_handler(message):
    bot.reply_to(message, "📤 Export is not yet implemented. (Will export tracked accounts/keywords)")

@bot.message_handler(commands=['import'])
def import_handler(message):
    bot.reply_to(message, "📥 Import is not yet implemented. (Will import tracked accounts/keywords from CSV)")

@bot.message_handler(commands=['menu'])
def handle_menu(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("📈 Price", callback_data="menu_price"),
        types.InlineKeyboardButton("💼 Portfolio", callback_data="menu_portfolio"),
        types.InlineKeyboardButton("🐦 Twitter", callback_data="menu_twitter"),
        types.InlineKeyboardButton("🔍 Keywords", callback_data="menu_keywords"),
        types.InlineKeyboardButton("🧠 AI Chat", callback_data="menu_ai"),
        types.InlineKeyboardButton("⚙️ Autoscan", callback_data="menu_autoscan")
    )
    bot.send_message(message.chat.id, "Choose an option:", reply_markup=markup)
    )

@bot.callback_query_handler(func=lambda call: True)
def handle_menu_callbacks(call):
    if call.data == "menu_price":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "📈 Use `/price <TICKER>` to check stock prices.\nExample: `/price AAPL`")
    elif call.data == "menu_portfolio":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "💼 Portfolio commands:\n• /addstock\n• /viewportfolio\n• /removestock")
    elif call.data == "menu_twitter":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "🐦 Twitter commands:\n• /add @user\n• /list\n• /top\n• /last @user")
    elif call.data == "menu_keywords":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "🔍 Keyword tracking:\n• /addkeyword <word>\n• /listkeywords\n• /removekeyword")
    elif call.data == "menu_ai":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "🧠 Ask the AI:\nUse `/gemini <your question>`")
    elif call.data == "menu_autoscan":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "⚙️ Autoscan controls:\n• /pauseautoscan\n• /resumeautoscan\n• /scanmode\n• /myautoscan")
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
