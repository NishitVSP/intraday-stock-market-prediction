import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        print("Loading FinBERT sentiment model...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
    def get_sentiment_score(self, text):
        """
        Analyze the sentiment of a text using FinBERT
        """
        if not text or not isinstance(text, str):
            return 0.0
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            sentiment_score = probs[0][2] - probs[0][0]  # Positive - Negative
            return sentiment_score.item()
            
    def analyze_news_sentiment(self, news_df):
        """
        Analyze sentiment for all news items in a DataFrame
        """
        if not news_df.empty:
            print("Analyzing sentiment for news headlines...")
            news_df['sentiment'] = news_df['text'].apply(self.get_sentiment_score)
            news_df['hour'] = news_df['datetime'].dt.floor('h')
            hourly_sentiment = news_df.groupby('hour')['sentiment'].mean()
        else:
            hourly_sentiment = pd.Series(dtype='float64')
            
        return hourly_sentiment
    
    def merge_sentiment_with_price(self, df, hourly_sentiment):
        """
        Merge sentiment data with price data
        """
        df = df.copy()
        df['Sentiment'] = 0.0
        
        if not hourly_sentiment.empty:
            for ts, score in hourly_sentiment.items():
                if ts in df.index:
                    df.at[ts, 'Sentiment'] = score
                    
        df['Sentiment'] = df['Sentiment'].fillna(0.0)
        return df 