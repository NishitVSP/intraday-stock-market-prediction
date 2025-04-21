from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import traceback
from modules.main import StockPredictionPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock_prediction_app_secret_key'

# Ensure required directories exist
os.makedirs('static/images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Create pipeline instance
pipeline = StockPredictionPipeline()

# Default values matching sentiments.py
DEFAULT_TICKER = "TATAMOTORS.NS"
DEFAULT_COMPANY = "Tata Motors"

@app.route('/')
def home():
    return render_template('index.html', default_ticker=DEFAULT_TICKER, default_company=DEFAULT_COMPANY)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form.get('ticker', '').strip()
        company_keyword = request.form.get('company_keyword', '').strip()
        
        if not ticker or not company_keyword:
            return render_template('index.html', error="Ticker symbol and company keyword are required!", 
                                 default_ticker=DEFAULT_TICKER, default_company=DEFAULT_COMPANY)
        
        # Run the prediction pipeline
        result = pipeline.run_pipeline(ticker, company_keyword)
        
        if 'error' in result:
            return render_template('index.html', error=result['error'],
                                 default_ticker=DEFAULT_TICKER, default_company=DEFAULT_COMPANY)
            
        # Render the results page
        return render_template('results.html', result=result)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return render_template('index.html', error=f"Error processing your request: {str(e)}",
                             default_ticker=DEFAULT_TICKER, default_company=DEFAULT_COMPANY)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
