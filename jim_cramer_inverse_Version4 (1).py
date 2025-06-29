#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jim Cramer Inverse Strategy Module
Author: danielkotas (modularized for bot use)

This module contains a function run_cramer_strategy() for integration
with a Telegram bot. It downloads Jim Cramer's tweets, analyzes stock
mentions and sentiment, and simulates an "inverse" trading strategy.
"""

import os
import snscrape.modules.twitter as sntwitter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless servers like Render
import matplotlib.pyplot as plt
import stanza
import nltk
from textblob import TextBlob
import pickle
import _pickle as cPickle
import bz2
import yfinance as yf
import requests
import bs4 as bs
from datetime import datetime
import yaml
from portfolio_analytics_v2 import PortfolioAnalysis

# --- Helper functions ---
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data_load = cPickle.load(data)
    return data_load

def next_working_day(date):
    if date.weekday() in [5, 6]:
        return date + pd.Timedelta(7 - date.weekday(), unit = 'D')
    return date

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    names = []
    forbidden = [' Inc.', ', .Inc']
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        name = row.findAll('td')[1].text
        ticker = ticker.replace(".", "-")
        for char in forbidden:
            name = name.replace(char, "")
        tickers.append(ticker.replace('\n', ''))
        names.append(name.replace('\n', ''))
    df = pd.DataFrame({'ticker': tickers, 'name': names})
    return df

def asg_ticker(t, tickers_names):
    tickers_names = tickers_names.fillna('#N/A - Missing Ticker')
    found_tickers = []
    for char in t.split():
        if len(char) > 5 and char[:5] == 'https':
            t = t.replace(char, " ")
    disallowed_characters = "._!&,-?'0123456789();/`’"
    for char in disallowed_characters:
        t = t.replace(char, " ")
    t_split = t.split()
    tick = [x[1:] for x in t_split if x[0] == "$" and len (x) != 1]
    found_tickers += tick
    tick_2 = [x for x in t_split if x in list(tickers_names['ticker']) and x != 'A']
    found_tickers += tick_2
    for index, row in tickers_names.iterrows():
        if (row['name'] in t) or (row['alt_name'] in t) or (row['alt_name_2'] in t):
            found_tickers.append(row['ticker'])
    if found_tickers:
        return list(set(found_tickers))

def sent_class(text, nltk_flag=True, stanza_flag=False, textblob_flag=True, textblob_subjectivity=False, method='avg'):
    overall_sentiment = []
    if nltk_flag:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        nltk_sent = sia.polarity_scores(text)
        overall_sentiment.append(nltk_sent['compound'])
    if stanza_flag:
        nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
        stanza_raw = ()
        doc = nlp_stanza(text)
        for i, sentence in enumerate(doc.sentences):
            stanza_raw += (sentence.sentiment,)
        stanza_sent = np.mean(stanza_raw) - 1
        overall_sentiment.append(stanza_sent)
    if textblob_flag:
        tb_sent = TextBlob(text).sentiment.polarity
        if textblob_subjectivity:
            tb_sent = tb_sent * TextBlob(text).sentiment.subjectivity
        overall_sentiment.append(tb_sent)
    if method == 'avg':
        final_sent = np.mean(overall_sentiment)
    elif method == 'max':
        if abs(np.min(overall_sentiment)) > abs(np.max(overall_sentiment)):
            final_sent = np.min(overall_sentiment)
        else:
            final_sent = np.max(overall_sentiment)
    return final_sent

def weights_scaler(df, long_cap, short_cap):
    df_scaled = df.copy()
    df_scaled[df > 0] = df.div(df[df > 0].sum(axis=1), axis=0) * long_cap
    df_scaled[df < 0] = (df.div(df[df < 0].sum(axis=1), axis=0)) * short_cap
    return df_scaled

def tp_date(df, leg, tp):
    if not df.empty:
        nav = ((1+df).cumprod() - 1)*leg
        return (nav > tp).argmax()
    else:
        return 0

# --- Main function for bot integration ---
def run_cramer_strategy():
    """
    Runs the Jim Cramer Inverse strategy and returns a summary string,
    and saves a chart as 'cramer_nav.png' in the current directory.
    """
    # Download NLTK data if missing
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download("vader_lexicon")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt")

    # Load parameters
    with open("params.yaml", "r") as f:
        p = yaml.safe_load(f)

    # Scrape or load tweets
    if p.get('load_twitter', False):
        tweets_list = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper('from:jimcramer').get_items()):
            if tweet.date.date() < datetime.strptime(p['start_date'],'%Y-%m-%d').date():
                break
            tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
        compressed_pickle('tweets_cramer', tweets_list)
    else:
        tweets_list = decompress_pickle('tweets_cramer.pbz2')

    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime','Tweet Id','Text','Username'])

    # Stock data
    tickers_sp500 = save_sp500_tickers()
    df_alt_names = pd.read_excel('alt_names.xlsx')
    tickers_new = tickers_sp500.merge(df_alt_names, on='ticker', how='outer', sort=True).set_index('ticker', drop=False)
    end_date = pd.to_datetime(tweets_df.iloc[0,:]['Datetime'], format='%Y-%m-%d').date()

    if p.get('load_yahoo', False):
        prices = yf.download(list(tickers_new.ticker), start = p['start_date'], end=end_date)['Adj Close']
        compressed_pickle('prices', prices)
    else:
        prices = decompress_pickle('prices.pbz2')

    prices.index = pd.to_datetime(prices.index, format='%Y-%m-%d')
    df_returns = prices.pct_change()

    # Assigning stocks
    tweets_asg = tweets_df.copy()
    tweets_asg['assign'] = tweets_asg['Text'].apply(lambda x: asg_ticker(x, tickers_new))

    # Sentiment
    tweets_sent = tweets_asg[tweets_asg['assign'].notnull()].copy()
    tweets_sent['sent'] = tweets_sent['Text'].apply(
        lambda x: sent_class(
            x,
            nltk_flag=p['sent_params'].get('nltk', True),
            stanza_flag=p['sent_params'].get('stanza', False),
            textblob_flag=p['sent_params'].get('textblob', True),
            textblob_subjectivity=p['sent_params'].get('textblob_subjectivity', False),
            method=p['sent_params'].get('method', 'avg')
        )
    )
    tweets_sent['assign'] = tweets_sent['assign'].apply(lambda lst: [x if x != 'FB' else 'META' for x in lst])

    # Strategy
    tweets_clean = tweets_sent.copy()
    tweets_clean['Datetime'] = tweets_clean['Datetime'].apply(next_working_day)
    dates_clean = pd.to_datetime(tweets_clean['Datetime'].dt.date)
    tweets_clean.index = dates_clean
    dt_weights_index = pd.date_range(start=dates_clean.iloc[-1], end=dates_clean.iloc[0], freq='B')
    df_weights = pd.DataFrame(data=0, index=dt_weights_index, columns=tickers_new['ticker'])
    df_returns_r = df_returns.reindex(df_weights.index, fill_value=0)
    tweets_clean = tweets_clean.set_index(pd.RangeIndex(len(tweets_clean)), append=True)

    for i in tweets_clean.index:
        tickers = tweets_clean.loc[i, 'assign']
        if abs(tweets_clean.loc[i, 'sent']) < p['sent_min']:
            continue
        if p['weighting'] == 'direction':
            sentiment = np.sign(tweets_clean.loc[i, 'sent']) * -1
        else:
            sentiment = tweets_clean.loc[i, 'sent'] * -1
        for tick in tickers:
            if tick not in list(df_weights.columns):
                continue
            else:
                date = i[0] + pd.offsets.BDay(p['trading_lag'])
                if p['take_profit']:
                    tp_holding_period = tp_date(
                        df_returns_r.loc[date:, tick],
                        np.sign(sentiment),
                        p['take_profit_pct']
                    )
                    if tp_holding_period < p['holding_period'] and tp_holding_period != 0:
                        date_till = date + pd.offsets.BDay(tp_holding_period)
                    else:
                        date_till = date + pd.offsets.BDay(p['holding_period'])
                else:
                    date_till = date + pd.offsets.BDay(p['holding_period'])
                if p['allow_cumulating']:
                    df_weights.loc[date:date_till, tick] += sentiment
                elif p['allow_overwrite']:
                    df_weights.loc[date:date_till, tick] = sentiment
                elif (df_weights.loc[date:date_till, tick] == 0).all():
                    df_weights.loc[date:date_till, tick] = sentiment

    df_weights_scaled = weights_scaler(df_weights.fillna(0), **p['caps'])
    df_returns_portfolio = df_weights_scaled.mul(df_returns_r).fillna(0)
    df_port = pd.DataFrame(df_returns_portfolio.sum(axis=1), columns=['Jim Cramer'])
    df_port['S&P500'] = df_returns_r['SPY'].fillna(0)
    summary = PortfolioAnalysis(df_port['Jim Cramer'], benchmark=df_port['S&P500'], ann_factor=252)

    # Output chart and summary
    summary.navs.plot()
    plt.title('NAVs of Jim Cramer inverse strategy and S&P 500')
    plt.tight_layout()
    plt.savefig('cramer_nav.png')
    plt.close()
    df_summary = summary.summary_with_benchmark()
    return df_summary.to_string()