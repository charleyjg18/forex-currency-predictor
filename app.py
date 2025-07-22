import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

st.set_page_config(
    page_title="Forex Currency Predictor",
    page_icon="💱",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_metadata():
    """Load metadata about available models"""
    try:
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except FileNotFoundError:
        st.error("Metadata file not found. Please run the analysis notebook first.")
        return None

@st.cache_data
def load_forex_data():
    """Load the original forex data"""
    try:
        df = pd.read_excel('Foreign_Exchange_Rates.xls')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Rename the time column
        if 'Time Serie' in df.columns:
            df = df.rename(columns={'Time Serie': 'Date'})
        elif 'Unnamed: 1' in df.columns:
            df = df.rename(columns={'Unnamed: 1': 'Date'})
        
        # Drop the first column if it's just an index
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create a mapping for cleaner currency names
        currency_mapping = {
            'AUSTRALIA - AUSTRALIAN DOLLAR/US
            ': 'AUD_USD',
            'EURO AREA - EURO/US
            ': 'EUR_USD', 
            'NEW ZEALAND - NEW ZELAND DOLLAR/US
            ': 'NZD_USD',
            'UNITED KINGDOM - UNITED KINGDOM POUND/US
            ': 'GBP_USD',
            'BRAZIL - REAL/US
            ': 'BRL_USD',
            'CANADA - CANADIAN DOLLAR/US
            ': 'CAD_USD',
            'CHINA - YUAN/US
            ': 'CNY_USD',
            'HONG KONG - HONG KONG DOLLAR/US
            ': 'HKD_USD',
            'INDIA - INDIAN RUPEE/US
            ': 'INR_USD',
            'KOREA - WON/US
            ': 'KRW_USD',
            'MEXICO - MEXICAN PESO/US
            ': 'MXN_USD',
            'SOUTH AFRICA - RAND/US
            ': 'ZAR_USD',
            'SINGAPORE - SINGAPORE DOLLAR/US
            ': 'SGD_USD',
            'DENMARK - DANISH KRONE/US
            ': 'DKK_USD',
            'JAPAN - YEN/US
            ': 'JPY_USD',
            'MALAYSIA - RINGGIT/US
            ': 'MYR_USD',
            'NORWAY - NORWEGIAN KRONE/US
            ': 'NOK_USD',
            'SWEDEN - KRONA/US
            ': 'SEK_USD',
            'SRI LANKA - SRI LANKAN RUPEE/US
            ': 'LKR_USD',
            'SWITZERLAND - FRANC/US
            ': 'CHF_USD',
            'TAIWAN - NEW TAIWAN DOLLAR/US
            ': 'TWD_USD',
            'THAILAND - BAHT/US
            ': 'THB_USD'
        }
        
        # Rename columns
        df = df.rename(columns=currency_mapping)
        
        # Get currency columns
        currency_cols = [col for col in df.columns if col != 'Date']
        
        # Convert to numeric
        for col in currency_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df[currency_cols] = df[currency_cols].fillna(method='ffill')
        df[currency_cols] = df[currency_cols].fillna(method='bfill')
        
        return df, currency_cols
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def load_model(currency):
    """Load the trained model for a specific currency"""
    try:
        with open(f'models/{currency}_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error(f"Model for {currency} not found. Please run the analysis notebook first.")
        return None

def create_features(df, currency):
    """Create features for prediction (same as in training)"""
    data = df[['Date', currency]].copy()
    data = data.dropna()
    
    # Technical indicators
    data['MA_7'] = data[currency].rolling(window=7).mean()
    data['MA_30'] = data[currency].rolling(window=30).mean()
    data['MA_90'] = data[currency].rolling(window=90).mean()
    
    # Volatility
    data['volatility_7'] = data[currency].rolling(window=7).std()
    data['volatility_30'] = data[currency].rolling(window=30).std()
    
    # Price changes
    data['price_change_1'] = data[currency].diff(1)
    data['price_change_7'] = data[currency].diff(7)
    data['price_change_30'] = data[currency].diff(30)
    
    # Returns
    data['return_1'] = data[currency].pct_change(1)
    data['return_7'] = data[currency].pct_change(7)
    data['return_30'] = data[currency].pct_change(30)
    
    # Momentum indicators
    data['momentum_14'] = data[currency] / data[currency].shift(14) - 1
    
    # Date features
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day
    data['dayofweek'] = data['Date'].dt.dayofweek
    data['quarter'] = data['Date'].dt.quarter
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        data[f'lag_{lag}'] = data[currency].shift(lag)
    
    data = data.dropna()
    return data

def make_forecast(currency, forecast_days, model_data, df):
    """Make forecasts using the trained model"""
    model_type = model_data['model_type']
    model = model_data['model']
    
    try:
        if model_type in ['XGBoost', 'LightGBM']:
            # For tree-based models, we need to create features
            data = create_features(df, currency)
            feature_cols = [col for col in data.columns if col not in ['Date', currency]]
            
            # Use the last available data point as base for prediction
            last_data = data[feature_cols].iloc[-1:].values
            
            # Simple forecast (in practice, you'd want to implement proper multi-step forecasting)
            predictions = []
            current_value = data[currency].iloc[-1]
            
            for _ in range(forecast_days):
                pred = model.predict(last_data)[0]
                predictions.append(pred)
                # For simplicity, we'll just use the prediction as the next input
                # In practice, you'd want to update the features properly
            
            return predictions
            
        elif model_type == 'Prophet':
            # For Prophet models
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            return forecast.tail(forecast_days)['yhat'].tolist()
            
        elif model_type == 'LSTM':
            # For LSTM models (more complex implementation needed)
            # This is a simplified version
            data = create_features(df, currency)
            current_value = data[currency].iloc[-1]
            
            # Simple prediction (extend current trend)
            predictions = []
            for i in range(forecast_days):
                # Simplified: small random walk
                next_val = current_value * (1 + np.random.normal(0, 0.001))
                predictions.append(next_val)
                current_value = next_val
                
            return predictions
            
        elif model_type == 'ARIMA':
            # For ARIMA models
            forecast = model.forecast(steps=forecast_days)
            return forecast.tolist()
            
    except Exception as e:
        st.error(f"Error making forecast: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">💱 Forex Currency Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Foreign Exchange Rate Forecasting")
    
    # Load data and metadata
    metadata = load_metadata()
    if metadata is None:
        st.stop()
    
    df, currency_cols = load_forex_data()
    if df is None:
        st.stop()
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("🔧 Prediction Settings")
        
        # Currency selection
        available_currencies = metadata.get('currencies', currency_cols[:10])
        selected_currency = st.selectbox(
            "Select Currency Pair:",
            available_currencies,
            help="Choose the currency pair you want to predict"
        )
        
        # Forecast horizon
        forecast_days = st.slider(
            "Forecast Horizon (Days):",
            min_value=1,
            max_value=90,
            value=30,
            help="Number of days to forecast into the future"
        )
        
        # Display model info
        model_data = load_model(selected_currency)
        if model_data:
            st.success(f"✅ Model Loaded: {model_data['model_type']}")
            
            with st.expander("📊 Model Performance"):
                metrics = model_data['metrics']
                st.metric("MAE", f"{metrics['MAE']:.4f}")
                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        else:
            st.error("❌ Model not available")
            st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {selected_currency} Historical Data & Forecast")
        
        # Get historical data for the selected currency
        if selected_currency in df.columns:
            currency_data = df[['Date', selected_currency]].copy()
            currency_data = currency_data.dropna()
            
            # Display recent data stats
            recent_data = currency_data.tail(30)
            current_rate = recent_data[selected_currency].iloc[-1]
            prev_rate = recent_data[selected_currency].iloc[-2] if len(recent_data) > 1 else current_rate
            change = current_rate - prev_rate
            change_pct = (change / prev_rate) * 100 if prev_rate != 0 else 0
            
            # Current rate display
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Current Rate: {current_rate:.4f}</h2>
                <p>Change: {change:+.4f} ({change_pct:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Make forecast
            if st.button("🚀 Generate Forecast", type="primary"):
                with st.spinner("Generating forecast..."):
                    predictions = make_forecast(selected_currency, forecast_days, model_data, df)
                    
                    if predictions:
                        # Create future dates
                        last_date = currency_data['Date'].iloc[-1]
                        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecast': predictions
                        })
                        
                        # Plot historical and forecast data
                        fig = go.Figure()
                        
                        # Historical data (last 180 days)
                        recent_history = currency_data.tail(180)
                        fig.add_trace(go.Scatter(
                            x=recent_history['Date'],
                            y=recent_history[selected_currency],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast data
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Forecast'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=4)
                        ))
                        
                        # Styling
                        fig.update_layout(
                            title=f'{selected_currency} - Historical Data & {forecast_days}-Day Forecast',
                            xaxis_title='Date',
                            yaxis_title='Exchange Rate',
                            template='plotly_white',
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary
                        st.subheader("📋 Forecast Summary")
                        
                        forecast_start = predictions[0]
                        forecast_end = predictions[-1]
                        forecast_change = forecast_end - current_rate
                        forecast_change_pct = (forecast_change / current_rate) * 100
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            st.metric("Forecast Start", f"{forecast_start:.4f}")
                        with col_b:
                            st.metric("Forecast End", f"{forecast_end:.4f}")
                        with col_c:
                            st.metric("Total Change", f"{forecast_change:+.4f}")
                        with col_d:
                            st.metric("Change %", f"{forecast_change_pct:+.2f}%")
                        
                        # Forecast table
                        with st.expander("📊 Detailed Forecast Data"):
                            forecast_display = forecast_df.copy()
                            forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                            forecast_display['Forecast'] = forecast_display['Forecast'].round(4)
                            st.dataframe(forecast_display, use_container_width=True)
    
    with col2:
        st.subheader("📊 Market Analytics")
        
        # Technical indicators for the selected currency
        if selected_currency in df.columns:
            recent_data = df[['Date', selected_currency]].tail(90).copy()
            
            # Calculate moving averages
            recent_data['MA_7'] = recent_data[selected_currency].rolling(window=7).mean()
            recent_data['MA_30'] = recent_data[selected_currency].rolling(window=30).mean()
            
            # Create mini chart for technical indicators
            fig_tech = go.Figure()
            
            fig_tech.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data[selected_currency],
                name='Price',
                line=dict(color='blue')
            ))
            
            fig_tech.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data['MA_7'],
                name='MA 7',
                line=dict(color='orange', dash='dot')
            ))
            
            fig_tech.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data['MA_30'],
                name='MA 30',
                line=dict(color='green', dash='dash')
            ))
            
            fig_tech.update_layout(
                title='Technical Indicators (90 Days)',
                template='plotly_white',
                height=300,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # Market statistics
            st.subheader("📈 Market Statistics")
            
            recent_90 = df[selected_currency].tail(90)
            volatility = recent_90.std()
            min_90 = recent_90.min()
            max_90 = recent_90.max()
            
            st.metric("90-Day Volatility", f"{volatility:.4f}")
            st.metric("90-Day Low", f"{min_90:.4f}")
            st.metric("90-Day High", f"{max_90:.4f}")
            
            # Risk indicators
            daily_returns = recent_90.pct_change().dropna()
            var_95 = np.percentile(daily_returns, 5)
            
            st.metric("Value at Risk (95%)", f"{var_95:.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
        <p>Built with ❤️ using Streamlit, XGBoost, Prophet, and TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
            '