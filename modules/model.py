import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend to avoid Tkinter thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import pytz
import os

class StockPredictor:
    def __init__(self, lookback=24):
        self.lookback = lookback
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def prepare_data(self, df):
        """
        Prepare data for LSTM model
        """
        price_data = df['Close'].values.reshape(-1, 1)
        sentiment_data = df['Sentiment'].values.reshape(-1, 1)
        
        scaled_price = self.price_scaler.fit_transform(price_data)
        scaled_sentiment = self.sentiment_scaler.fit_transform(sentiment_data)
        
        combined_data = np.hstack((scaled_price, scaled_sentiment))
        
        X, y = [], []
        for i in range(self.lookback, len(combined_data)):
            X.append(combined_data[i-self.lookback:i])
            y.append(scaled_price[i, 0])
            
        return np.array(X), np.array(y)
        
    def build_model(self):
        """
        Build LSTM model
        """
        # Use functional API instead of Sequential to avoid warnings
        inputs = Input(shape=(self.lookback, 2))
        x = LSTM(50, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(50)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
        
    def train(self, X, y, epochs=60, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model
        """
        print("Building and training LSTM model...")
        self.model = self.build_model()
        self.model.summary()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split, 
            callbacks=[early_stop]
        )
        
        return self.model
        
    def save_model(self, filepath):
        """
        Save the trained model to a file
        """
        if self.model:
            self.model.save(filepath)
            print(f"Model saved as {filepath}")
            
    def load_model_from_file(self, filepath):
        """
        Load a trained model from a file
        """
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        return False
            
    def predict_next_day(self, df, prediction_date):
        """
        Generate predictions for the next day's market hours
        """
        if not self.model:
            raise ValueError("Model not trained or loaded")
            
        # Set up market hours for prediction day
        ist = pytz.timezone('Asia/Kolkata')
        prediction_day_ist = ist.localize(prediction_date)
        market_hours = []
        current_time = prediction_day_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = prediction_day_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # If current time is past market close, use tomorrow's market hours for prediction
        if datetime.now().hour > 15 or (datetime.now().hour == 15 and datetime.now().minute > 30):
            tomorrow = prediction_day_ist + timedelta(days=1)
            current_time = tomorrow.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = tomorrow.replace(hour=15, minute=30, second=0, microsecond=0)
            print(f"Current time is past market close, predicting for tomorrow: {tomorrow.strftime('%Y-%m-%d')}")
        
        while current_time <= market_close:
            market_hours.append(current_time)
            current_time += timedelta(minutes=30)
            
        hours_to_predict = len(market_hours)
        
        # Get the last sequence of price and sentiment data for prediction
        price_data = df['Close'].values.reshape(-1, 1)
        sentiment_data = df['Sentiment'].values.reshape(-1, 1)
        
        scaled_price = self.price_scaler.transform(price_data)
        scaled_sentiment = self.sentiment_scaler.transform(sentiment_data)
        
        last_price_seq = scaled_price[-self.lookback:].reshape(1, self.lookback, 1)
        last_sentiment_seq = scaled_sentiment[-self.lookback:].reshape(1, self.lookback, 1)
        current_sequence = np.concatenate([last_price_seq, last_sentiment_seq], axis=2)
        
        # Generate predictions
        predictions = []
        for _ in range(hours_to_predict):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            new_seq = np.roll(current_sequence, -1, axis=1)
            new_seq[0, -1, 0] = pred[0, 0]
            new_seq[0, -1, 1] = 0  # No future sentiment available
            current_sequence = new_seq
            
        # Convert predictions back to price scale
        predicted_prices = self.price_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'Predicted Price': predicted_prices.flatten()
        }, index=market_hours)
        
        return predictions_df
    
    def plot_predictions_only(self, predictions_df, ticker, prediction_date):
        """
        Plot only the predicted stock prices without historical data
        """
        plt.figure(figsize=(12, 6))
        plt.plot(predictions_df.index, predictions_df['Predicted Price'], 'r-o', label='Predicted Prices')
        plt.xlabel('Time (IST)')
        plt.ylabel('Stock Price (INR)')
        plt.title(f'{ticker} Stock Price Prediction for {prediction_date.strftime("%Y-%m-%d")} (Indian Market Hours)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save prediction plot
        pred_plot_path = f'static/images/{ticker}_prediction_{prediction_date.strftime("%Y%m%d")}.png'
        plt.savefig(pred_plot_path)
        plt.close('all')
        
        return pred_plot_path
        
    def plot_predictions(self, predictions_df, ticker, prediction_date, recent_data=None):
        """
        Plot the predicted stock prices
        """
        # Plot only predictions
        plt.figure(figsize=(12, 6))
        plt.plot(predictions_df.index, predictions_df['Predicted Price'], 'r-o', label='Predicted Prices')
        plt.xlabel('Time (IST)')
        plt.ylabel('Stock Price (INR)')
        plt.title(f'{ticker} Stock Price Prediction for {prediction_date.strftime("%Y-%m-%d")} (Indian Market Hours)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save prediction plot
        pred_plot_path = f'static/images/{ticker}_prediction_{prediction_date.strftime("%Y%m%d")}.png'
        plt.savefig(pred_plot_path)
        
        # Plot recent data with predictions if recent data is provided
        if recent_data is not None and len(recent_data) > 0:
            plt.figure(figsize=(14, 7))
            plt.plot(recent_data.index, recent_data.values, 'b-', label=f'Historical Prices (Last Trading Day)')
            plt.plot(predictions_df.index, predictions_df['Predicted Price'], 'r-o', label=f'Predicted Prices ({prediction_date.strftime("%Y-%m-%d")})')
            plt.axvline(x=recent_data.index[-1], color='gray', linestyle='--', label='Last Historical Data Point')
            plt.xlabel('Date and Time')
            plt.ylabel('Stock Price (INR)')
            plt.title(f'{ticker} Recent History and Prediction for {prediction_date.strftime("%Y-%m-%d")}')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save combined plot
            combined_plot_path = f'static/images/{ticker}_combined_{prediction_date.strftime("%Y%m%d")}.png'
            plt.savefig(combined_plot_path)
        
        plt.close('all')
        return pred_plot_path, combined_plot_path if recent_data is not None else (pred_plot_path, None) 