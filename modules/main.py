from datetime import datetime, timedelta
import os
import pandas as pd

from .data_fetcher import fetch_stock_data, fetch_newsapi_news, process_news_data
from .sentiment_analysis import SentimentAnalyzer
from .model import StockPredictor

class StockPredictionPipeline:
    def __init__(self):
        self.newsapi_key = "7fd36820d0a7474b9f4a857b3b5c9efb"  # API key from original script
        self.sentiment_analyzer = SentimentAnalyzer()
        self.predictor = StockPredictor(lookback=24)
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs('static/images', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
    def run_pipeline(self, ticker, company_keyword):
        """Run the entire prediction pipeline"""
        # Set up dates
        today = datetime.now()
        prediction_date = today
        end_training_date = prediction_date - timedelta(hours=1)  # Data until the last hour
        start_date = end_training_date - timedelta(days=365)  # Using 365 days for training as in sentiments.py
        
        # Cache file for news data
        news_cache_file = f"data/newsapi_{company_keyword.replace(' ','_')}_{start_date.strftime('%Y%m%d')}_{end_training_date.strftime('%Y%m%d')}.json"
        
        # Model and prediction file paths
        date_str = prediction_date.strftime('%Y%m%d')
        model_path = f"models/{ticker.replace('.', '_')}_Model_for_{date_str}.keras"
        prediction_csv = f"data/{ticker.replace('.', '_')}_{date_str}_predictions.csv"
        
        # Step 1: Fetch stock data
        df = fetch_stock_data(ticker, start_date, end_training_date, interval='1h')
        if df.empty:
            return {"error": "No stock data available for the selected ticker. Please check if the ticker symbol is correct."}
        
        # Step 2: Fetch and process news data
        news_json = fetch_newsapi_news(company_keyword, start_date, end_training_date, self.newsapi_key, news_cache_file)
        news_df = process_news_data(news_json)
        
        # Step 3: Analyze sentiment
        hourly_sentiment = self.sentiment_analyzer.analyze_news_sentiment(news_df)
        df = self.sentiment_analyzer.merge_sentiment_with_price(df, hourly_sentiment)
        
        # Step 4: Prepare data, train model, and save
        X, y = self.predictor.prepare_data(df)
        self.predictor.train(X, y, epochs=100)  # Using 100 epochs as in sentiments.py
        self.predictor.save_model(model_path)
        
        # Step 5: Generate predictions
        predictions_df = self.predictor.predict_next_day(df, prediction_date)
        predictions_df.to_csv(prediction_csv)
        
        # Step 6: Plot results - only prediction plot, no combined plot
        pred_plot_path = self.predictor.plot_predictions_only(predictions_df, ticker, prediction_date)
        
        return {
            "ticker": ticker,
            "company_keyword": company_keyword,
            "prediction_date": prediction_date.strftime("%Y-%m-%d"),
            "predictions": predictions_df.to_dict(),
            "prediction_plot": pred_plot_path.replace('static/', ''),
            "model_path": model_path,
            "prediction_csv": prediction_csv
        } 