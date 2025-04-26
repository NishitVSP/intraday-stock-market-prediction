# Stock Price Prediction Web Application

This web application uses LSTM neural networks and sentiment analysis of news to predict stock prices. 

## Features

* Predict stock prices for the current day using historical data from the last 365 days
* Incorporate sentiment analysis from news related to the company
* Interactive web interface to input ticker symbol and company name
* Visualize predictions with charts and tables

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python app.py
```

4. Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

## Usage

1. Enter a valid stock ticker symbol (e.g., "TATAMOTORS.NS" for Tata Motors on the National Stock Exchange)
2. Enter the company name for news search (e.g., "Tata Motors")
3. Click "Generate Prediction" and wait for the system to process the data
4. View the prediction results displayed as charts and tables

## Project Structure

```
├── app.py                  # Flask application main file
├── requirements.txt        # Python dependencies
├── modules/                # Python modules for various components
│   ├── __init__.py         # Package initialization
│   ├── data_fetcher.py     # Functions to fetch stock and news data
│   ├── sentiment_analysis.py # Sentiment analysis using FinBERT
│   ├── model.py            # LSTM model definition and training
│   └── main.py             # Main orchestration module
├── static/                 # Static files
│   ├── css/
│   │   └── style.css       # Custom styles
│   └── images/             # Generated charts
├── templates/              # HTML templates
│   ├── index.html          # Home page with input form
│   └── results.html        # Results page with charts and tables
├── models/                 # Directory for saved models
└── data/                   # Directory for saved data and predictions
```

## Dependencies

* Flask - Web framework
* YFinance - Stock data API
* NewsAPI - News article API
* Transformers (FinBERT) - Sentiment analysis
* TensorFlow/Keras - LSTM model
* Matplotlib - Visualizations
* Pandas/NumPy - Data handling

## License

This project is provided for our data science project.

## Notes

* The News API in free tier has a limitation of only fetching news from the last 30 days
* Stock prediction involves inherent risks and uncertainties
* This application is for our data science project and should not be used for actual investment decisions

## Group NUMBER 17 : 
* Member 1 : Nishit Prajapati - nishit.prajapati@iitgn.ac.in
* Member 2 : Mitansh Patel - mitansh.patel@iitgn.ac.in
* Member 3 : Hardik Khobragade - hardik.k@iitgn.ac.in
* Member 4  : Ekansh Somani - ekansh.somani@iitgn.ac.in

