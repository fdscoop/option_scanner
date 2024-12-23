import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai
from scipy.stats import norm
from flask import Flask, request, jsonify
import os
import base64
import io
import logging
import requests
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NiftyOptionsAnalyzer:
    def __init__(self):
        """
        Enhanced Nifty Options and Trading Analysis with Secure Configuration
        """
        # Load API keys from environment variables
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.news_api_key or not self.openai_api_key:
            raise ValueError("Required API keys not found in environment variables")
            
        self.fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        self.historical_data = None

    def parse_custom_data(self, stock_name, data_string):
        """
        Parse custom data string into DataFrame
        """
        # Split data into groups of 6
        data_points = data_string.split(', ')
        rows = [data_points[i:i+6] for i in range(0, len(data_points), 6)]

        # Create DataFrame
        columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(rows, columns=columns)

        # Convert data types
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
          df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN if any conversion fails
        df.dropna(inplace=True)

        df.set_index('Datetime', inplace=True)
        self.historical_data = df

        return df
    def calculate_comprehensive_technical_indicators(self):
        """
        Advanced Technical Indicators for Professional Trading
        """
        df = self.historical_data.copy()

        # Trend Indicators
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()

        # Momentum Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['Stochastic_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['Stochastic_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()

        # Volatility Indicators
        df['Bollinger_High'] = ta.volatility.BollingerBands(df['Close'], window=20).bollinger_hband()
        df['Bollinger_Low'] = ta.volatility.BollingerBands(df['Close'], window=20).bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

        # Trend Strength and Direction
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

        return df

    def calculate_option_greeks(self, spot_price, strike_price, risk_free_rate=0.07, time_to_expiry=1.01/365, volatility=0.2):
        """
        Calculate Option Greeks for Index Options
        """
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

            # Greeks Calculation
            delta_call = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta_call = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)

            return {
                'price': call_price,
                'delta': delta_call,
                'gamma': gamma,
                'theta': theta_call,
                'vega': vega
            }

        def black_scholes_put(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            # Greeks Calculation
            delta_put = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta_put = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)

            return {
                'price': put_price,
                'delta': delta_put,
                'gamma': gamma,
                'theta': theta_put,
                'vega': vega
            }

        return {
            'call': black_scholes_call(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility),
            'put': black_scholes_put(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility)
        }

    def generate_analysis_plots(self, technical_data, breakeven_analysis, fib_levels):
        """
        Generate analysis plots and convert to base64
        """
        plt.figure(figsize=(15, 10))

        # Price and Moving Averages
        plt.subplot(2, 2, 1)
        plt.plot(technical_data.index, technical_data['Close'], label='Close Price')
        plt.plot(technical_data.index, technical_data['SMA_20'], label='20-day SMA')
        plt.plot(technical_data.index, technical_data['SMA_50'], label='50-day SMA')
        plt.title('Price and Moving Averages')
        plt.legend()

        # Breakeven Analysis
        plt.subplot(2, 2, 2)
        plt.plot(breakeven_analysis['scenarios']['price_points'],
                 breakeven_analysis['scenarios']['call_pnl'], label='Call PnL')
        plt.plot(breakeven_analysis['scenarios']['price_points'],
                 breakeven_analysis['scenarios']['put_pnl'], label='Put PnL')
        plt.title('Options Breakeven Analysis')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.legend()

        # Fibonacci Levels
        plt.subplot(2, 2, 3)
        level_values = [v for v in fib_levels.values()]
        plt.bar(list(fib_levels.keys()), level_values)
        plt.title('Fibonacci Levels')
        plt.xticks(rotation=45)

        # RSI
        plt.subplot(2, 2, 4)
        plt.plot(technical_data.index, technical_data['RSI'])
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('Relative Strength Index (RSI)')

        plt.tight_layout()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return plot_base64

    def fetch_financial_news(self, stock_name="Nifty"):
        """
        Fetch and analyze financial news with sentiment analysis
        """
        base_url = "https://newsapi.org/v2/everything"
        params = {
            "q": stock_name,
            "sortBy": "publishedAt",
            "apiKey": self.news_api_key
        }

        try:
            response = requests.get(base_url, params=params)
            news_data = response.json().get('articles', [])

            # Sentiment Analysis
            sid = SentimentIntensityAnalyzer()
            sentiments = []

            for article in news_data[:10]:  # Limit to 10 articles
                title = article.get('title', '')
                sentiment = sid.polarity_scores(title)
                sentiments.append({
                    'title': title,
                    'sentiment_score': sentiment['compound']
                })

            return pd.DataFrame(sentiments)

        except Exception as e:
            print(f"News Fetch Error: {e}")
            return pd.DataFrame()

    def perform_swot_analysis(self, technical_data):
        """
        Generate SWOT Analysis for Market Conditions
        """
        latest_data = technical_data.iloc[-1]

        swot = {
            'Strengths': [],
            'Weaknesses': [],
            'Opportunities': [],
            'Threats': []
        }

        # Strengths
        if latest_data['Close'] > latest_data['SMA_50']:
            swot['Strengths'].append("Strong Bullish Trend (Price above 50-day SMA)")
        if latest_data['RSI'] < 30:
            swot['Strengths'].append("Potential Oversold Condition (RSI < 30)")

        # Weaknesses
        if latest_data['RSI'] > 70:
            swot['Weaknesses'].append("Potential Overbought Condition (RSI > 70)")
        if latest_data['ATR'] > technical_data['ATR'].mean() * 1.5:
            swot['Weaknesses'].append("High Market Volatility")

        # Opportunities
        if latest_data['MACD'] > latest_data['MACD_Signal']:
            swot['Opportunities'].append("Positive MACD Crossover (Bullish Signal)")
        if latest_data['Stochastic_K'] > latest_data['Stochastic_D']:
            swot['Opportunities'].append("Stochastic Momentum Indicating Potential Uptrend")

        # Threats
        if latest_data['ADX'] > 25:
            swot['Threats'].append("Strong Trend Might Indicate Market Exhaustion")

        return swot

    def generate_ai_market_report(self, technical_data, sentiment_data, swot_analysis):
        """
        AI-Powered Comprehensive Market Analysis and Recommendation
        """
        try:
            # Updated OpenAI initialization
            client = openai.OpenAI(api_key=self.openai_api_key)

            # Prepare comprehensive market data
            latest_data = technical_data.iloc[-1]
            avg_sentiment = sentiment_data['sentiment_score'].mean() if not sentiment_data.empty else 0

            prompt = f"""
            Comprehensive Market Analysis Report:

            Technical Overview:
            - Current Price: {latest_data['Close']:.2f}
            - 20-day SMA: {latest_data['SMA_20']:.2f}
            - 50-day SMA: {latest_data['SMA_50']:.2f}
            - RSI: {latest_data['RSI']:.2f}
            - MACD: {latest_data['MACD']:.2f}
            - Market Sentiment: {avg_sentiment:.2f}

            SWOT Analysis Summary:
            Strengths: {', '.join(swot_analysis['Strengths'])}
            Weaknesses: {', '.join(swot_analysis['Weaknesses'])}
            Opportunities: {', '.join(swot_analysis['Opportunities'])}
            Threats: {', '.join(swot_analysis['Threats'])}

            Provide a detailed professional trading recommendation:
            1. Trading Strategy (Intraday/Positional/Long-term)
            2. Entry/Exit Recommendations
            3. Stop Loss Levels
            4. Risk Assessment
            5. Option Trading Suggestions (Put/Call)
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial market analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"AI Report Generation Error: {e}")
            return self.generate_manual_report(technical_data, sentiment_data, swot_analysis)

    def generate_manual_report(self, technical_data, sentiment_data, swot_analysis):
        """
        Generate a manual report when AI generation fails
        """
        latest_data = technical_data.iloc[-1]
        avg_sentiment = sentiment_data['sentiment_score'].mean() if not sentiment_data.empty else 0

        report = f"""
        Stocxer AI Market Analysis Report:

        Technical Overview:
        - Current Price: {latest_data['Close']:.2f}
        - 20-day SMA: {latest_data['SMA_20']:.2f}
        - 50-day SMA: {latest_data['SMA_50']:.2f}
        - RSI: {latest_data['RSI']:.2f}
        - MACD: {latest_data['MACD']:.2f}
        - Market Sentiment: {avg_sentiment:.2f}

        SWOT Analysis Summary:
        Strengths: {', '.join(swot_analysis['Strengths'])}
        Weaknesses: {', '.join(swot_analysis['Weaknesses'])}
        Opportunities: {', '.join(swot_analysis['Opportunities'])}
        Threats: {', '.join(swot_analysis['Threats'])}

        Trading Recommendations:
        1. Current Trend: {'Bullish' if latest_data['Close'] > latest_data['SMA_50'] else 'Bearish'}
        2. RSI Indication: {'Overbought' if latest_data['RSI'] > 70 else 'Oversold' if latest_data['RSI'] < 30 else 'Neutral'}
        3. Suggested Strategy: {'Hold' if abs(latest_data['RSI'] - 50) < 10 else 'Consider Entry/Exit'}

        Risk Assessment:
        - Volatility (ATR): {latest_data['ATR']:.2f}
        - Potential Support: {latest_data['SMA_20']:.2f}
        - Potential Resistance: {latest_data['Bollinger_High']:.2f}

        Disclaimer: This is an automated analysis. Always consult a financial advisor.
        """
        return report

    def calculate_fibonacci_levels(self, high, low):
        """
        Calculate Fibonacci retracement and extension levels
        """
        diff = high - low
        levels = {}

        # Retracement levels
        for ratio in self.fib_ratios:
            levels[f'retracement_{ratio}'] = high - (diff * ratio)

        # Extension levels
        for ratio in [1.618, 2.618, 3.618]:
            levels[f'extension_{ratio}'] = high + (diff * ratio)

        return levels

    def calculate_breakeven_analysis(self, spot_price, strike_price, premium, contract_size=75):
        """
        Calculate breakeven points and profit/loss scenarios for options
        """
        breakeven = {
            'call': {
                'breakeven_point': strike_price + premium,
                'max_loss': premium * contract_size,
                'max_profit': 'Unlimited'
            },
            'put': {
                'breakeven_point': strike_price - premium,
                'max_loss': premium * contract_size,
                'max_profit': strike_price - premium if strike_price > premium else 0
            }
        }

        # Calculate profit/loss at different price points
        price_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
        breakeven['scenarios'] = {
            'price_points': price_range,
            'call_pnl': [max(-premium, price - strike_price - premium) * contract_size for price in price_range],
            'put_pnl': [max(-premium, strike_price - price - premium) * contract_size for price in price_range]
        }

        return breakeven

    def game_theory_analysis(self, technical_data, sentiment_score):
        """
        Perform game theory analysis for market scenarios
        """
        latest_data = technical_data.iloc[-1]

        # Define payoff matrix for different strategies
        payoff_matrix = {
            'bull_market': {
                'long_call': 0.7,
                'long_put': -0.3,
                'iron_condor': 0.2,
                'covered_call': 0.5
            },
            'bear_market': {
                'long_call': -0.3,
                'long_put': 0.7,
                'iron_condor': 0.2,
                'covered_call': -0.1
            },
            'sideways_market': {
                'long_call': -0.1,
                'long_put': -0.1,
                'iron_condor': 0.6,
                'covered_call': 0.3
            }
        }

        # Calculate market scenario probabilities based on technical indicators
        market_probability = {
            'bull_market': min(max((latest_data['RSI'] - 50) / 30 +
                                 (sentiment_score + 1) / 4 +
                                 (latest_data['MACD'] > 0) * 0.2, 0), 1),
            'bear_market': min(max((50 - latest_data['RSI']) / 30 +
                                 (-sentiment_score + 1) / 4 +
                                 (latest_data['MACD'] < 0) * 0.2, 0), 1)
        }
        market_probability['sideways_market'] = 1 - market_probability['bull_market'] - market_probability['bear_market']

        # Calculate expected value for each strategy
        strategy_ev = {}
        for strategy in payoff_matrix['bull_market'].keys():
            ev = sum(prob * payoff_matrix[scenario][strategy]
                    for scenario, prob in market_probability.items())
            strategy_ev[strategy] = ev

        return {
            'market_probability': market_probability,
            'strategy_expected_values': strategy_ev,
            'recommended_strategy': max(strategy_ev.items(), key=lambda x: x[1])[0]
        }

    def comprehensive_trading_analysis(self, stock_name, data_string):
        """
        Enhanced Comprehensive Market Analysis Workflow with Custom Stock Name
        """
        # First parse the input data
        self.parse_custom_data(stock_name, data_string)

        # Technical Indicators
        technical_data = self.calculate_comprehensive_technical_indicators()

        # Convert technical_data index to string
        technical_data_serializable = technical_data.reset_index()
        technical_data_serializable['Datetime'] = technical_data_serializable['Datetime'].astype(str)

        # Financial News & Sentiment
        sentiment_data = self.fetch_financial_news(stock_name)
        avg_sentiment = sentiment_data['sentiment_score'].mean() if not sentiment_data.empty else 0

        # SWOT Analysis
        swot_analysis = self.perform_swot_analysis(technical_data)

        # Calculate Fibonacci levels
        high = technical_data['High'].max()
        low = technical_data['Low'].min()
        fib_levels = self.calculate_fibonacci_levels(high, low)

        # Latest price for further calculations
        latest_price = self.historical_data['Close'].iloc[-1]

        # Option Greeks and Breakeven Analysis
        strike_price = latest_price * 1.05  # Example strike price 5% above current
        option_greeks = self.calculate_option_greeks(
            spot_price=latest_price,
            strike_price=strike_price
        )

        # Assume example premium of 2% of spot price
        premium = latest_price * 0.02
        breakeven_analysis = self.calculate_breakeven_analysis(
            latest_price,
            strike_price,
            premium
        )

        # Game Theory Analysis
        game_theory_results = self.game_theory_analysis(technical_data, avg_sentiment)

        # AI Market Report
        market_report = self.generate_ai_market_report(
            technical_data,
            sentiment_data,
            swot_analysis
        )

<<<<<<< HEAD
        # Generate plots and convert to base64
        plot_buffer = self.generate_analysis_plots(technical_data, breakeven_analysis, fib_levels)

        return {
            'technical_data': technical_data_serializable.tail().to_dict(orient='records'),
            'fibonacci_levels': {k: float(v) for k, v in fib_levels.items()},
            'game_theory': {
                'market_probabilities': {k: float(v) for k, v in game_theory_results['market_probability'].items()},
                'strategy_expected_values': {k: float(v) for k, v in game_theory_results['strategy_expected_values'].items()},
                'recommended_strategy': game_theory_results['recommended_strategy']
            },
            'breakeven_analysis': {
                'call_breakeven': float(breakeven_analysis['call']['breakeven_point']),
                'put_breakeven': float(breakeven_analysis['put']['breakeven_point']),
                'max_loss_call': float(breakeven_analysis['call']['max_loss']),
                'max_loss_put': float(breakeven_analysis['put']['max_loss'])
            },
            'sentiment_analysis': sentiment_data.to_dict(orient='records'),
            'swot_analysis': swot_analysis,
            'option_greeks': {
                'call': {k: float(v) for k, v in option_greeks['call'].items()},
                'put': {k: float(v) for k, v in option_greeks['put'].items()}
            },
            'market_report': market_report,
            'plot_base64': plot_buffer
        }

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App Initialization
app = Flask(__name__)
analyzer = NiftyOptionsAnalyzer()
=======
        except Exception as e:
            logging.error(f"Error in comprehensive analysis: {str(e)}")
            raise
# Enhanced Flask routes
@app.route('/')
def home():
    """Root endpoint to verify the application is running"""
    return jsonify({
        "status": "success",
        "message": " Options Scanner AI is running"
    })
>>>>>>> c0e928b (Your commit message)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        # Validate input data
        if not data or 'stock_name' not in data or 'data_string' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: stock_name and data_string"
            }), 400

        stock_name = data['stock_name']
        data_string = data['data_string']

        analyzer = NiftyOptionsAnalyzer()
        result = analyzer.comprehensive_trading_analysis(stock_name, data_string)

        # Flatten and simplify the response with explicit type conversions
        flattened_response = {
            "status": "success",
            "technical_data": [{str(k): str(v) for k, v in item.items()} for item in result['technical_data']],
            "fibonacci_levels": {str(k): str(v) for k, v in result['fibonacci_levels'].items()},
            "game_theory": {
                "market_probabilities": {
                    str(k): str(v) for k, v in result['game_theory']['market_probabilities'].items()
                },
                "recommended_strategy": str(result['game_theory']['recommended_strategy'])
            },
            "breakeven_analysis": {
                str(k): str(v) for k, v in result['breakeven_analysis'].items()
            },
            "sentiment_analysis": [{str(k): str(v) for k, v in item.items()} for item in result['sentiment_analysis']],
            "swot_analysis": {
                str(k): [str(v) for v in result['swot_analysis'][k]]
                for k in result['swot_analysis']
            },
            "option_greeks": {
                str(k): {str(subk): str(subv) for subk, subv in v.items()}
                for k, v in result['option_greeks'].items()
            },
            "market_report": str(result['market_report']),
            "plot_base64": str(result['plot_base64'])
        }

        return jsonify(flattened_response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/")
def home():
    """
    Simple health check endpoint
    """
    return "Welcome to Stocxer AI"

if __name__ == "__main__":
    # Get port from environment variable (Heroku sets this automatically)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)