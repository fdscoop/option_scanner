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
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# Initialize Flask application
app = Flask(__name__)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NiftyOptionsAnalyzer:
    def __init__(self):
        """
        Enhanced Nifty Options and Trading Analysis with Secure Configuration
        """
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.news_api_key or not self.openai_api_key:
            logger.error("Missing required API keys")
            raise ValueError("Required API keys not found in environment variables")
            
        self.fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        self.historical_data = None
        self.analyzer = SentimentIntensityAnalyzer()
        
        # New configuration parameters
        self.min_profit_target = 50  # Minimum profit target in points
        self.max_loss_target = 30    # Maximum loss target in points
        self.trend_threshold = 0.6    # Threshold for trend strength
        
        logger.info("NiftyOptionsAnalyzer initialized successfully")

    def fetch_financial_news(self, stock_name, days=7):
        """
        Fetch and analyze financial news for the given stock
        
        Args:
            stock_name (str): Name of the stock to fetch news for
            days (int): Number of days of news to fetch (default: 7)
            
        Returns:
            pandas.DataFrame: DataFrame containing news data with sentiment scores
        """
        try:
            logger.info(f"Fetching news for {stock_name}")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # NewsAPI endpoint
            url = f"https://newsapi.org/v2/everything"
            
            # Parameters for the API request
            params = {
                'q': stock_name,
                'from': from_date,
                'to': to_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            news_data = response.json()
            
            if not news_data.get('articles'):
                logger.warning(f"No news articles found for {stock_name}")
                return pd.DataFrame(columns=['date', 'title', 'sentiment_score'])
            
            # Process articles
            processed_news = []
            for article in news_data['articles']:
                # Combine title and description for sentiment analysis
                text = f"{article['title']} {article.get('description', '')}"
                sentiment_scores = self.analyzer.polarity_scores(text)
                
                processed_news.append({
                    'date': article['publishedAt'],
                    'title': article['title'],
                    'sentiment_score': sentiment_scores['compound']
                })
            
            # Create DataFrame and sort by date
            news_df = pd.DataFrame(processed_news)
            news_df['date'] = pd.to_datetime(news_df['date'])
            news_df = news_df.sort_values('date', ascending=False)
            
            logger.info(f"Successfully fetched and analyzed {len(news_df)} news articles")
            return news_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return pd.DataFrame(columns=['date', 'title', 'sentiment_score'])
        except Exception as e:
            logger.error(f"Error processing news data: {str(e)}")
            return pd.DataFrame(columns=['date', 'title', 'sentiment_score'])
    pass

    def parse_array_data(self, stock_name, data_array, days_to_expiry=None):
        try:
            logger.info(f"Parsing array data for {stock_name}")
            
            df = pd.DataFrame(
                data_array,
                columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if df.isnull().any().any():
                null_counts = df.isnull().sum()
                logger.error(f"Missing values detected: {null_counts}")
                raise ValueError(f"Dataset contains missing values: {null_counts}")
            
            df.dropna(inplace=True)
            
            if len(df) < 5:
                raise ValueError(f"Insufficient data points for analysis. Required: 5, Got: {len(df)}")
            
            df.set_index('Datetime', inplace=True)
            
            if not (df['High'] >= df['Low']).all():
                logger.error("Invalid price data: High prices lower than Low prices detected")
                raise ValueError("Invalid price data detected")
            
            self.days_to_expiry = days_to_expiry if days_to_expiry is not None else 1
            self.historical_data = df
            
            logger.info(f"Successfully parsed {len(df)} data points")
            return df

        except Exception as e:
            logger.error(f"Error parsing data: {str(e)}")
            raise ValueError(f"Invalid data format: {str(e)}")

    def calculate_comprehensive_technical_indicators(self):
        try:
            if self.historical_data is None:
                raise ValueError("No historical data available")
                
            logger.info("Calculating technical indicators")
            df = self.historical_data.copy()
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Add technical indicators
            df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
            df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
            
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['Stochastic_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Stochastic_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
            
            df['Bollinger_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            df['Bollinger_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            df['DI_Positive'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_pos()
            df['DI_Negative'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_neg()
            
            df['Daily_Return'] = df['Close'].pct_change()
            df['Weekly_Return'] = df['Close'].pct_change(periods=5)
            df['Monthly_Return'] = df['Close'].pct_change(periods=20)
            
            if df.isnull().any().any():
                null_counts = df.isnull().sum()
                logger.warning(f"Missing values in indicators: {null_counts}")
                df.fillna(method='ffill', inplace=True)
                
            logger.info("Successfully calculated technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise ValueError(f"Error in technical analysis calculations: {str(e)}")



    def calculate_option_greeks(self, spot_price, strike_price, risk_free_rate=0.07, volatility=0.2):
        """
        Enhanced Option Greeks calculation with custom time to expiry
        """
        try:
            time_to_expiry = self.days_to_expiry / 365.0

            def black_scholes_call(S, K, T, r, sigma):
                d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)

                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
                vega = S * norm.pdf(d1) * np.sqrt(T)
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)

                return {
                    'price': price,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'rho': rho
                }

            def black_scholes_put(S, K, T, r, sigma):
                d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)

                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                delta = norm.cdf(d1) - 1
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
                vega = S * norm.pdf(d1) * np.sqrt(T)
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

                return {
                    'price': price,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'rho': rho
                }

            return {
                'call': black_scholes_call(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility),
                'put': black_scholes_put(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility)
            }

        except Exception as e:
            logging.error(f"Error calculating option greeks: {str(e)}")
            raise ValueError("Error in options calculations")
    def analyze_market_trends(self):
        """
        Analyze daily, weekly, and monthly trends using statistical models
        """
        try:
            df = self.historical_data.copy()

            # Calculate trends using different timeframes
            trends = {
                'daily': {
                    'data': df['Close'].pct_change(),
                    'window': 1
                },
                'weekly': {
                    'data': df['Close'].pct_change(5),
                    'window': 5
                },
                'monthly': {
                    'data': df['Close'].pct_change(20),
                    'window': 20
                }
            }

            trend_analysis = {}
            for timeframe, data in trends.items():
                series = data['data'].dropna()
                if len(series) > data['window']:
                    # Statistical analysis
                    mean = series.mean()
                    std = series.std()
                    skew = series.skew()

                    # Trend strength
                    positive_days = (series > 0).sum() / len(series)

                    # Simple trend determination
                    current_trend = "Bullish" if mean > 0 else "Bearish"
                    strength = abs(mean) / std

                    trend_analysis[timeframe] = {
                        'trend': current_trend,
                        'strength': strength,
                        'reliability': positive_days if current_trend == "Bullish" else (1 - positive_days),
                        'volatility': std
                    }

            return trend_analysis

        except Exception as e:
            logging.error(f"Error in trend analysis: {str(e)}")
            return {}

    def analyze_sentiment_impact(self, sentiment_data):
        """
        Enhanced sentiment analysis with market impact assessment
        """
        try:
            if sentiment_data.empty:
                return {
                    'impact': 'Neutral',
                    'score': 0,
                    'confidence': 0
                }

            # Calculate weighted sentiment score
            recent_sentiment = sentiment_data['sentiment_score'].iloc[-3:].mean() if len(sentiment_data) >= 3 else sentiment_data['sentiment_score'].mean()
            overall_sentiment = sentiment_data['sentiment_score'].mean()

            # Weight recent sentiment more heavily
            weighted_score = (recent_sentiment * 0.7) + (overall_sentiment * 0.3)

            # Calculate sentiment volatility
            sentiment_volatility = sentiment_data['sentiment_score'].std()

            # Determine impact level
            if abs(weighted_score) < 0.2:
                impact = 'Weak'
            elif abs(weighted_score) < 0.5:
                impact = 'Moderate'
            else:
                impact = 'Strong'

            # Calculate confidence based on consistency
            confidence = 1 - min(sentiment_volatility, 1)

            return {
                'impact': f"{'Bullish' if weighted_score > 0 else 'Bearish'} ({impact})",
                'score': weighted_score,
                'confidence': confidence,
                'recent_change': recent_sentiment - overall_sentiment
            }

        except Exception as e:
            logging.error(f"Error in sentiment impact analysis: {str(e)}")
            return {'impact': 'Error', 'score': 0, 'confidence': 0}

    def identify_entry_exit_zones(self, technical_data, option_greeks):
        """
        Identify optimal entry and exit zones based on technical and options data
        """
        try:
            latest_data = technical_data.iloc[-1]

            # Entry zones calculation
            entry_zones = {
                'call': {
                    'strong_buy': [],
                    'moderate_buy': [],
                    'avoid': []
                },
                'put': {
                    'strong_buy': [],
                    'moderate_buy': [],
                    'avoid': []
                }
            }

            current_price = latest_data['Close']

            # Call entry zones
            if latest_data['RSI'] < 30 and latest_data['MACD'] > latest_data['MACD_Signal']:
                entry_zones['call']['strong_buy'].append(current_price)
            elif latest_data['RSI'] < 40:
                entry_zones['call']['moderate_buy'].append(current_price)
            elif latest_data['RSI'] > 70:
                entry_zones['call']['avoid'].append(current_price)

            # Put entry zones
            if latest_data['RSI'] > 70 and latest_data['MACD'] < latest_data['MACD_Signal']:
                entry_zones['put']['strong_buy'].append(current_price)
            elif latest_data['RSI'] > 60:
                entry_zones['put']['moderate_buy'].append(current_price)
            elif latest_data['RSI'] < 30:
                entry_zones['put']['avoid'].append(current_price)

            # Calculate target zones based on ATR
            atr = latest_data['ATR']
            targets = {
                'call': {
                    'target1': current_price + (atr * 1.5),
                    'target2': current_price + (atr * 2.5),
                    'stop_loss': current_price - (atr * 1.0)
                },
                'put': {
                    'target1': current_price - (atr * 1.5),
                    'target2': current_price - (atr * 2.5),
                    'stop_loss': current_price + (atr * 1.0)
                }
            }

            return {
                'entry_zones': entry_zones,
                'targets': targets,
                'current_price': current_price
            }

        except Exception as e:
            logging.error(f"Error identifying entry/exit zones: {str(e)}")
            return {}

    def generate_ai_market_report(self, technical_data, sentiment_data, trend_analysis, entry_exit_zones):
        """
        Enhanced AI-powered market analysis using OpenAI's latest model
        """
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)

            # Prepare market data summary
            latest_data = technical_data.iloc[-1]
            sentiment_impact = self.analyze_sentiment_impact(sentiment_data)

            # Create detailed prompt for AI analysis
            prompt = f"""
            Generate a professional trading analysis based on the following data:

            Technical Indicators:
            - Price: {latest_data['Close']:.2f}
            - RSI: {latest_data['RSI']:.2f}
            - MACD: {latest_data['MACD']:.2f}
            - ADX: {latest_data['ADX']:.2f}

            Market Trends:
            {trend_analysis}

            Sentiment Analysis:
            - Impact: {sentiment_impact['impact']}
            - Score: {sentiment_impact['score']:.2f}
            - Confidence: {sentiment_impact['confidence']:.2f}

            Entry/Exit Zones:
            {entry_exit_zones}

            Provide a concise analysis with:
            1. KEY: Main trading bias (BULL/BEAR/NEUTRAL)
            2. ZONES: Key support/resistance levels
            3. ENTRY: Best entry points with reasoning
            4. EXIT: Target zones and stop-loss levels
            5. RISK: Key risk factors
            6. STRATEGY: Recommended option strategies
            """

            response = client.chat.completions.create(
                model="gpt-4",  # Using latest available model
                messages=[
                    {"role": "system", "content": "You are a professional market analyst and options trading expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Error generating AI market report: {str(e)}")
            return self.generate_fallback_report(technical_data, sentiment_data, trend_analysis)

    def generate_fallback_report(self, technical_data, sentiment_data, trend_analysis):
        """
        Fallback report generation when AI service is unavailable
        """
        try:
            latest_data = technical_data.iloc[-1]

            # Basic trend analysis
            trend = "BULL" if latest_data['Close'] > latest_data['EMA_200'] else "BEAR"
            strength = "Strong" if latest_data['ADX'] > 25 else "Weak"

            report = f"""
            KEY: {trend} ({strength})
            ZONES: Support: {latest_data['Bollinger_Low']:.2f}, Resistance: {latest_data['Bollinger_High']:.2f}
            ENTRY: {latest_data['Close']:.2f} ({trend} momentum)
            EXIT: Target +{latest_data['ATR']:.2f} points, Stop -{latest_data['ATR']/2:.2f} points
            RISK: ADX={latest_data['ADX']:.2f}, ATR={latest_data['ATR']:.2f}
            STRATEGY: {"Call options" if trend == "BULL" else "Put options"} with {self.days_to_expiry} DTE
            """

            return report

        except Exception as e:
            logging.error(f"Error generating fallback report: {str(e)}")
            return "Error generating market report"
    def perform_enhanced_swot_analysis(self, technical_data, sentiment_impact, trend_analysis):
        """
        Enhanced SWOT analysis with error handling
        """
        try:
            if technical_data is None or technical_data.empty:
                raise ValueError("Invalid technical data")
                
            latest_data = technical_data.iloc[-1]
            
            swot = {
                'Strengths': [],
                'Weaknesses': [],
                'Opportunities': [],
                'Threats': []
            }

            # Trend-based analysis
            if latest_data['Close'] > latest_data['EMA_200']:
                swot['Strengths'].append("ST:ABOVE_200EMA")
            else:
                swot['Weaknesses'].append("WT:BELOW_200EMA")

            # Momentum analysis
            if latest_data['RSI'] < 30:
                swot['Opportunities'].append("OP:OVERSOLD_RSI")
            elif latest_data['RSI'] > 70:
                swot['Threats'].append("TH:OVERBOUGHT_RSI")

            # Volume analysis
            vol_avg = technical_data['Volume'].rolling(20).mean().iloc[-1]
            if latest_data['Volume'] > vol_avg * 1.5:
                swot['Strengths'].append("ST:HIGH_VOL")
            elif latest_data['Volume'] < vol_avg * 0.5:
                swot['Weaknesses'].append("WT:LOW_VOL")

            # Sentiment integration
            if sentiment_impact['score'] > 0.5:
                swot['Strengths'].append("ST:STRONG_SENTIMENT")
            elif sentiment_impact['score'] < -0.5:
                swot['Weaknesses'].append("WT:WEAK_SENTIMENT")

            # Trend strength
            if latest_data['ADX'] > 25:
                if latest_data['DI_Positive'] > latest_data['DI_Negative']:
                    swot['Opportunities'].append("OP:STRONG_UPTREND")
                else:
                    swot['Threats'].append("TH:STRONG_DOWNTREND")

            return swot
            
        except Exception as e:
            logger.error(f"Error in SWOT analysis: {str(e)}")
            return {'Strengths': [], 'Weaknesses': [], 'Opportunities': [], 'Threats': []}

    def enhance_game_theory_analysis(self, technical_data, sentiment_score, trend_analysis):
        """
        Enhanced game theory analysis with trend integration
        """
        try:
            latest_data = technical_data.iloc[-1]

            # Define enhanced strategy payoffs
            payoff_matrix = {
                'bull_market': {
                    'long_call': 0.8,
                    'bull_spread': 0.6,
                    'iron_condor': 0.2,
                    'covered_call': 0.5
                },
                'bear_market': {
                    'long_put': 0.8,
                    'bear_spread': 0.6,
                    'iron_condor': 0.2,
                    'protective_put': 0.5
                },
                'neutral_market': {
                    'iron_condor': 0.7,
                    'butterfly': 0.6,
                    'calendar_spread': 0.5
                }
            }

            # Calculate market probabilities
            market_prob = {
                'bull_market': min(max((
                    (latest_data['RSI'] - 50) / 30 +
                    (sentiment_score + 1) / 4 +
                    (latest_data['MACD'] > 0) * 0.2 +
                    (latest_data['Close'] > latest_data['EMA_200']) * 0.3
                ), 0), 1),

                'bear_market': min(max((
                    (50 - latest_data['RSI']) / 30 +
                    (-sentiment_score + 1) / 4 +
                    (latest_data['MACD'] < 0) * 0.2 +
                    (latest_data['Close'] < latest_data['EMA_200']) * 0.3
                ), 0), 1)
            }

            market_prob['neutral_market'] = 1 - market_prob['bull_market'] - market_prob['bear_market']

            # Calculate expected values
            strategy_ev = {}
            risk_adjusted_ev = {}

            for market, prob in market_prob.items():
                for strategy, payoff in payoff_matrix[market].items():
                    if strategy not in strategy_ev:
                        strategy_ev[strategy] = 0
                        risk_adjusted_ev[strategy] = 0

                    strategy_ev[strategy] += prob * payoff
                    # Adjust EV based on ATR for risk consideration
                    risk_factor = 1 - (latest_data['ATR'] / latest_data['Close'])
                    risk_adjusted_ev[strategy] += prob * payoff * risk_factor

            # Find optimal strategy
            optimal_strategy = max(risk_adjusted_ev.items(), key=lambda x: x[1])

            return {
                'market_probability': market_prob,
                'strategy_expected_values': strategy_ev,
                'risk_adjusted_ev': risk_adjusted_ev,
                'recommended_strategy': optimal_strategy[0],
                'confidence_score': optimal_strategy[1]
            }

        except Exception as e:
            logging.error(f"Error in game theory analysis: {str(e)}")
            return {}

    def comprehensive_trading_analysis(self, stock_name, data_string, days_to_expiry=None):
        """
        Enhanced comprehensive analysis workflow
        """
        try:
            # Initialize analysis with days to expiry
            self.parse_custom_data(stock_name, data_string, days_to_expiry)

            # Technical analysis
            technical_data = self.calculate_comprehensive_technical_indicators()

            # Trend analysis
            trend_analysis = self.analyze_market_trends()

            # Sentiment analysis
            sentiment_data = self.fetch_financial_news(stock_name)
            sentiment_impact = self.analyze_sentiment_impact(sentiment_data)

            # Latest price for calculations
            latest_price = technical_data['Close'].iloc[-1]

            # Options analysis
            strike_price = latest_price * (1.05 if trend_analysis['daily']['trend'] == 'Bullish' else 0.95)
            option_greeks = self.calculate_option_greeks(latest_price, strike_price)

            # Entry/exit zones
            zones = self.identify_entry_exit_zones(technical_data, option_greeks)

            # Enhanced SWOT analysis
            swot_analysis = self.perform_enhanced_swot_analysis(
                technical_data,
                sentiment_impact,
                trend_analysis
            )

            # Game theory analysis
            game_theory = self.enhance_game_theory_analysis(
                technical_data,
                sentiment_impact['score'],
                trend_analysis
            )

            # AI market report
            market_report = self.generate_ai_market_report(
                technical_data,
                sentiment_data,
                trend_analysis,
                zones
            )

            return {
                'technical_summary': {
                    'trend': trend_analysis,
                    'current_price': float(latest_price),
                    'key_levels': {
                        'support': float(technical_data['Bollinger_Low'].iloc[-1]),
                        'resistance': float(technical_data['Bollinger_High'].iloc[-1])
                    }
                },
                'option_analysis': {
                    'greeks': option_greeks,
                    'zones': zones
                },
                'sentiment': sentiment_impact,
                'swot': swot_analysis,
                'game_theory': game_theory,
                'market_report': market_report
            }

        except Exception as e:
            logging.error(f"Error in comprehensive analysis: {str(e)}")
            raise
# Enhanced Flask routes
@app.route('/')
def home():
    """Root endpoint to verify the application is running"""
    return jsonify({
        "status": "success",
        "message": "Nifty Options Analyzer API is running"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        logger.info("Received analysis request")
        
        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request body must be a JSON object"
            }), 400
            
        if 'stock_name' not in data or 'data_array' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: stock_name and data_array"
            }), 400
            
        if not isinstance(data['data_array'], list):
            return jsonify({
                "status": "error",
                "message": "data_array must be a list of lists"
            }), 400
            
        days_to_expiry = data.get('days_to_expiry', None)
        analyzer = NiftyOptionsAnalyzer()
        
        def modified_analysis(self, stock_name, data_array, days_to_expiry=None):
            self.parse_array_data(stock_name, data_array, days_to_expiry)
            technical_data = self.calculate_comprehensive_technical_indicators()
            trend_analysis = self.analyze_market_trends()
            sentiment_data = self.fetch_financial_news(stock_name)
            sentiment_impact = self.analyze_sentiment_impact(sentiment_data)
            latest_price = technical_data['Close'].iloc[-1]
            strike_price = latest_price * (1.05 if trend_analysis['daily']['trend'] == 'Bullish' else 0.95)
            option_greeks = self.calculate_option_greeks(latest_price, strike_price)
            zones = self.identify_entry_exit_zones(technical_data, option_greeks)
            
            return [{
                'timestamp': technical_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(latest_price),
                'trend': trend_analysis['daily']['trend'],
                'strength': trend_analysis['daily']['strength'],
                'rsi': float(technical_data['RSI'].iloc[-1]),
                'macd': float(technical_data['MACD'].iloc[-1]),
                'support': float(technical_data['Bollinger_Low'].iloc[-1]),
                'resistance': float(technical_data['Bollinger_High'].iloc[-1]),
                'sentiment_score': sentiment_impact['score'],
                'call_delta': option_greeks['call']['delta'],
                'put_delta': option_greeks['put']['delta'],
                'recommended_entry': zones['current_price'],
                'stop_loss': zones['targets']['call']['stop_loss'],
                'target1': zones['targets']['call']['target1'],
                'target2': zones['targets']['call']['target2']
            }]
        
        NiftyOptionsAnalyzer.modified_analysis = modified_analysis
        
        result = analyzer.modified_analysis(
            data['stock_name'],
            data['data_array'],
            days_to_expiry
        )
        
        logger.info("Analysis completed successfully")
        return jsonify({
            "status": "success",
            "data": result
        })
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)