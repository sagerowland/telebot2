# Telegram Bot (Flask, Gunicorn, SQLAlchemy, Stocks, Nitter)

## Features

- Webhook-based: Production-ready, runs on any $PORT (Render, Railway, Replit, etc)
- SQLAlchemy ORM DB (Postgres or SQLite)
- Twitter tracking with Nitter RSS + fallback to twitrss.me
- `/price`, `/info`, `/chart`, `/sentiment`, `/tweets`, and dozens more commands
- Supports all your listed commands (see `bot_webhook.py`)
- Easy deployment

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
