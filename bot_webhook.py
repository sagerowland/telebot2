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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Load environment ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

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

# --- Telegram handlers ---

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message, "👋 Hello! I'm your finance & news bot.\nType /help to see what I can do.")

@bot.message_handler(commands=['help'])
def handle_help(message):
    bot.reply_to(message, (
        "📈 `/price <ticker>` - Get the current price of a stock (e.g., `/price AAPL`)\n"
        "ℹ️ `/info <ticker>` - Get general information about a stock (e.g., `/info MSFT`)\n"
        "📊 `/chart <ticker> [period] [interval]` - Get a chart for a stock. "
        "Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. "
        "Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo. "
        "Defaults to 1mo period and 1d interval (e.g., `/chart GOOG 6mo 1wk`)\n"
        "💬 `/sentiment <text>` - Analyze the sentiment of a given text (e.g., `/sentiment The stock market is booming!`)\n"
        "🐦 `/tweets <query>` - (Conceptual) Fetch recent tweets related to a query (e.g., `/tweets Tesla`)\n\n"
        "--- **NEW COMMANDS** ---\n"
        "➕ `/add @user` - Track Twitter account\n"
        "➖ `/remove @user` - Remove tracked account\n"
        "📋 `/list` - List tracked accounts\n"
        "🧹 `/clear` - Remove all tracked accounts\n"
        "🗑️ `/cleardb` - Clear all YOUR bot data (DANGEROUS!)\n"
        "➕ `/addkeyword word` - Track keyword\n"
        "📋 `/listkeywords` - Show tracked keywords\n"
        "➖ `/removekeyword word` - Remove keyword\n"
        "📈 `/graph TICKER PERIOD [candle|line|rsi]` - Show stock graph (e.g., `/graph AAPL 1y candle`)\n"
        "🔔 `/alert TICKER <ABOVE|BELOW> <PRICE>` - Set a stock price alert (e.g., `/alert GOOGL ABOVE 150`)\n"
        "📋 `/listalerts` - List your active stock price alerts\n"
        "❌ `/removealert ID` - Remove a specific price alert by its ID\n"
        "💰 `/addstock TICKER QUANTITY PRICE` - Add stock to your virtual portfolio (e.g., `/addstock MSFT 10 300.50`)\n"
        "🗑️ `/removestock TICKER` - Remove stock from your portfolio\n"
        "📊 `/viewportfolio` - View your virtual stock portfolio performance\n"
        "⏱️ `/setinterval seconds` - Set scan interval (min 60s)\n"
        "🤫 `/setquiet <start_hour> <end_hour>` - Set quiet hours EST (HH:MM-HH:MM, e.g., `/setquiet 22:00 07:00`)\n"
        "🗺️ `/settimezone <TimeZoneName>` - Set your local timezone (e.g., `/settimezone Europe/London`). Use standard IANA names.\n"
        "🗓️ `/setschedule <daily|weekly> <HH:MM>` - Schedule daily/weekly reports (e.g., `/setschedule daily 09:00`)\n"
        "⚙️ `/mysettings` - View your settings\n"
        "🚦 `/status` - Show bot status\n"
        "⏸️ `/pause` - Pause bot\n"
        "▶️ `/resume` - Resume bot\n"
        "🔇 `/mute @user` - Mute user notifications\n"
        "🔊 `/unmute @user` - Unmute user notifications\n"
        "📜 `/last @user` - Show last tweet\n"
        "🔄 `/toggleautoscan` - Toggle auto-scan on/off\n"
        "🔝 `/top [num]` - Show top N recent tweets from tracked users\n"
        "🔥 `/trending [num]` - Show top N recent trending hashtags\n"
        "📤 `/export` - Export tracked accounts and keywords\n"
        "📥 `/import` - Import tracked accounts and keywords (reply to exported CSV)\n\n"
        "Need help again? Just type `/help`."
    ), parse_mode="Markdown")

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

@bot.message_handler(commands=['top'])
def top_handler(message):
    # Default to 5 if no number given
    args = message.text.split()
    try:
        n = int(args[1]) if len(args) > 1 else 5
    except ValueError:
        n = 5
    session = SessionLocal()
    users = session.query(Tracked).filter_by(chat_id=message.chat.id).all()
    if not users:
        bot.reply_to(message, "📋 No Twitter accounts tracked.")
        session.close()
        return
    all_tweets = []
    for user in users:
        entries = get_twitter_rss(user.username)
        for entry in entries[:3]:  # Only take a few per user to avoid spam
            all_tweets.append({
                "user": user.username,
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published_parsed", None)
            })
    session.close()
    if not all_tweets:
        bot.reply_to(message, "🔝 No recent tweets found for tracked users.")
        return
    # Sort by published date if available
    all_tweets = [t for t in all_tweets if t["published"]]
    all_tweets.sort(key=lambda t: t["published"], reverse=True)
    top_tweets = all_tweets[:n]
    reply = "\n\n".join([f"@{t['user']}: {t['title']}\n{t['link']}" for t in top_tweets])
    bot.reply_to(message, reply[:4096])

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
    bot.reply_to(message, "📈 Graph command is not implemented yet. Coming soon!")

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

@bot.message_handler(commands=['pause'])
def pause_handler(message):
    bot.reply_to(message, "⏸️ Pause is not yet implemented. (Will pause notifications or scans)")

@bot.message_handler(commands=['resume'])
def resume_handler(message):
    bot.reply_to(message, "▶️ Resume is not yet implemented. (Will resume notifications or scans)")

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
    tweet = get_latest_tweet(username)
    if tweet:
        bot.reply_to(message, f"🐦 Last tweet from @{username}:\n\n{tweet['text']}\n{tweet['url']}")
    else:
        bot.reply_to(message, f"❌ Could not retrieve tweets for @{username}. (Account may be protected, rate-limited, or unavailable.)")

@bot.message_handler(commands=['toggleautoscan'])
def toggleautoscan_handler(message):
    bot.reply_to(message, "🔄 Toggleautoscan is not yet implemented. (Will toggle auto-scan feature)")

@bot.message_handler(commands=['top'])
def top_handler(message):
    bot.reply_to(message, "🔝 Top is not yet implemented. (Will show top N tweets from tracked users)")

@bot.message_handler(commands=['trending'])
def trending_handler(message):
    bot.reply_to(message, "🔥 Trending is not yet implemented. (Will show top trending hashtags)")

@bot.message_handler(commands=['export'])
def export_handler(message):
    bot.reply_to(message, "📤 Export is not yet implemented. (Will export tracked accounts/keywords)")

@bot.message_handler(commands=['import'])
def import_handler(message):
    bot.reply_to(message, "📥 Import is not yet implemented. (Will import tracked accounts/keywords from CSV)")

# --- Set webhook on startup ---
def set_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=f"{WEBHOOK_URL}/{BOT_TOKEN}")

set_webhook()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
