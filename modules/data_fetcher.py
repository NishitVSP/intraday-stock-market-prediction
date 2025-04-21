import yfinance as yf
import pandas as pd
import requests
import os
import json
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date, interval='1h'):
    """
    Fetch stock data from Yahoo Finance
    """
    print(f"Downloading data for {ticker}...")
    return yf.download(ticker, start=start_date, end=end_date, interval=interval)

def fetch_newsapi_news(keyword, start_date, end_date, api_key, cache_file):
    """
    Fetch news from NewsAPI with local caching
    """
    today = datetime.now()
    min_allowed_date = today - timedelta(days=30)
    if start_date < min_allowed_date:
        print(f"Adjusting start_date from {start_date.strftime('%Y-%m-%d')} to {min_allowed_date.strftime('%Y-%m-%d')} due to NewsAPI free plan limitations.")
        start_date = min_allowed_date
        
    if os.path.exists(cache_file):
        print("Loaded news from local cache.")
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception as e:
                print("Corrupted cache, will fetch fresh news.")
                os.remove(cache_file)
                
    print("Fetching news from NewsAPI.org...")
    all_articles = []
    page = 1
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={keyword.replace(' ', '%20')}&"
        f"from={start_date.strftime('%Y-%m-%d')}&"
        f"to={end_date.strftime('%Y-%m-%d')}&"
        "language=en&"
        "sortBy=publishedAt&"
        "pageSize=100&"
        f"page={page}&"
        f"apiKey={api_key}"
    )
    response = requests.get(url)
    data = response.json()
    
    if 'status' in data and data['status'] == 'error':
        print(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
        return []
        
    if 'articles' in data and data['articles']:
        all_articles.extend(data['articles'])
        
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f)
        
    return all_articles

def process_news_data(news_json):
    """
    Process news data into a pandas DataFrame
    """
    news_df = pd.DataFrame(news_json) if news_json is not None else pd.DataFrame()
    
    if not news_df.empty:
        news_df['datetime'] = pd.to_datetime(news_df['publishedAt']).dt.tz_convert('Asia/Kolkata')
        news_df['text'] = news_df['title'].fillna('') + '. ' + news_df['description'].fillna('')
    else:
        print("No news found for this period. All sentiment will be set to 0.")
        
    return news_df 