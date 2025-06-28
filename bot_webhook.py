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

# --- Flask webhook with logging ---
@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    raw = request.stream.read().decode("utf-8")
    print(f"Received update: {raw}")  # Debug log
    try:
        update = Update.de_json(raw)
        bot.process_new_updates([update])
    except Exception as e:
        print(f"Error processing update: {e}")
    return "OK", 200

@app.route("/")
def index():
    return "Bot is alive!", 200

# --- Telegram handlers (register OUTSIDE main block) ---
@bot.message_handler(commands=['start'])
def handle_start(message):
    print("Handling /start command!")  # Debug log
    bot.reply_to(message, "👋 Hello! I'm your finance & news bot.\nType /help to see what I can do.")

@bot.message_handler(commands=['help'])
def handle_help(message):
    bot.reply_to(
        message,
        "📚 *Available Commands:*\n"
        "-----------------------------\n"
        "💲 /price <ticker> - Get stock price\n"
        "ℹ️ /info <ticker> - Stock info\n"
        "📊 /chart <ticker> [period] [interval] - Stock chart\n"
        "💬 /sentiment <text> - Sentiment analysis\n"
        "🐦 /tweets <query> - Tweets for query\n"
        "➕ /add @user - Track Twitter account\n"
        "➖ /remove @user - Remove tracked account\n"
        "📋 /list - List tracked accounts\n"
        "🧹 /clear - Remove all tracked accounts\n"
        "🗑️ /cleardb - Clear all YOUR bot data\n"
        "➕ /addkeyword word - Track keyword\n"
        "📋 /listkeywords - Show tracked keywords\n"
        "➖ /removekeyword word - Remove keyword\n"
        "📈 /graph TICKER PERIOD [candle|line|rsi] - Stock graph\n"
        "🔔 /alert TICKER <ABOVE|BELOW> <PRICE> - Price alert\n"
        "📋 /listalerts - List your alerts\n"
        "❌ /removealert ID - Remove alert\n"
        "💰 /addstock TICKER QUANTITY PRICE - Add to portfolio\n"
        "🗑️ /removestock TICKER - Remove from portfolio\n"
        "📊 /viewportfolio - View portfolio\n"
        "⏱️ /setinterval seconds - Scan interval\n"
        "🤫 /setquiet <start> <end> - Quiet hours\n"
        "🗺️ /settimezone <TimeZoneName> - Set timezone\n"
        "🗓️ /setschedule <daily|weekly> <HH:MM> - Schedule reports\n"
        "⚙️ /mysettings - View settings\n"
        "🚦 /status - Bot status\n"
        "⏸️ /pause - Pause bot\n"
        "▶️ /resume - Resume bot\n"
        "🔇 /mute @user - Mute notifications\n"
        "🔊 /unmute @user - Unmute notifications\n"
        "📜 /last @user - Last tweet\n"
        "🔄 /toggleautoscan - Toggle auto-scan\n"
        "🔝 /top [num] - Top tweets from tracked\n"
        "🔥 /trending [num] - Trending hashtags\n"
        "📤 /export - Export accounts/keywords\n"
        "📥 /import - Import (reply to CSV)\n",
        parse_mode="Markdown"
    )

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
        # Sometimes info dict is empty if ticker is invalid
        summary = info.get('longBusinessSummary', 'No info available.')
        bot.reply_to(message, f"ℹ️ {ticker} info:\n{summary}")
    except Exception as e:
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
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty:
            bot.reply_to(message, "❌ No data found.")
            return
        plt.figure(figsize=(10,4))
        data['Close'].plot(title=f"{ticker} Close Price")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        bot.send_photo(message.chat.id, buf, caption=f"📊 {ticker} Chart ({period}, {interval})")
    except Exception as e:
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
    tweets = get_tweets_for_query(query)
    if not tweets:
        bot.reply_to(message, "🐦 No tweets found.")
    else:
        reply = "\n\n".join([f"🐦 {t['text']}\n{t['url']}" for t in tweets])
        bot.reply_to(message, reply[:4096])

# --- Twitter tracking ---
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

# --- Keyword tracking ---
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

# --- Alerts ---
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

# --- Portfolio ---
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

# --- Fallback handler ---
@bot.message_handler(func=lambda m: True, content_types=['text'])
def echo_all(message):
    bot.reply_to(message, "🤖 Unknown command or message. Use /help to see what I can do!")

# --- Set webhook on startup ---
def set_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=f"{WEBHOOK_URL}/{BOT_TOKEN}")

if __name__ == "__main__":
    set_webhook()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
