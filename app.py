import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚀 Forex Predictor Pro",
    page_icon="💱",
    layout="wide"
)

# Header
st.markdown("# 🚀 Forex Predictor Pro")
st.markdown("### Advanced Machine Learning Forex Forecasting System")

@st.cache_data
def load_models():
    """Load all available models"""
    model_files = glob.glob("models/*_model.pkl")
    models = {}
    
    for file_path in model_files:
        try:
            model_data = joblib.load(file_path)
            currency_name = model_data.get('currency_name', 'Unknown')
            models[currency_name] = model_data
        except:
            continue
    
    return models

@st.cache_data
def create_sample_data():
    """Create sample data for all currencies"""
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2023-12-31', freq='D')
    
    # Major currencies with realistic starting prices
    currencies = {
        'AUSTRALIA - AUSTRALIAN DOLLAR/US$': 0.72,
        'EURO AREA - EURO/US$': 1.18,
        'UNITED KINGDOM - UNITED KINGDOM POUND/US$': 1.28,
        'JAPAN - YEN/US$': 110.0,
        'CANADA - CANADIAN DOLLAR/US$': 1.33,
        'CHINA - YUAN/US$': 6.45,
        'HONG KONG - HONG KONG DOLLAR/US$': 7.75,
        'SWITZERLAND - FRANC/US$': 0.92,
        'SINGAPORE - SINGAPORE DOLLAR/US$': 1.35,
        'NORWAY - NORWEGIAN KRONE/US$': 8.5,
        'SWEDEN - KRONA/US$': 9.2,
        'DENMARK - DANISH KRONE/US$': 6.8,
        'NEW ZEALAND - NEW ZELAND DOLLAR/US$': 0.65,
        'SOUTH AFRICA - RAND/US$': 14.5,
        'BRAZIL - REAL/US$': 5.2,
        'MEXICO - MEXICAN PESO/US$': 19.5,
        'INDIA - INDIAN RUPEE/US$': 74.0,
        'KOREA - WON/US$': 1200.0,
        'MALAYSIA - RINGGIT/US$': 4.2,
        'THAILAND - BAHT/US$': 36.0,
        'TAIWAN - NEW TAIWAN DOLLAR/US$': 31.0,
        'SRI LANKA - SRI LANKAN RUPEE/US$': 190.0
    }
    
    data = pd.DataFrame(index=dates)
    
    for currency, start_price in currencies.items():
        # Create realistic price series
        n_days = len(dates)
        returns = np.random.normal(0, 0.01, n_days)  # 1% daily volatility
        
        # Add some trend and seasonality
        trend = np.linspace(0, 0.1, n_days)  # Small upward trend
        seasonal = 0.02 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        
        log_prices = np.cumsum(returns) + trend + seasonal
        prices = start_price * np.exp(log_prices)
        
        data[currency] = prices
    
    return data

def create_features(data, target_col):
    """Create features for prediction"""
    recent_data = data[target_col].dropna().tail(120)
    
    if len(recent_data) < 30:
        return None
    
    features = {}
    
    # Lag features
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        if lag < len(recent_data):
            features[f'lag_{lag}'] = recent_data.iloc[-1-lag]
    
    # Rolling statistics
    for window in [5, 10, 20, 30]:
        if window <= len(recent_data):
            window_data = recent_data.tail(window)
            features[f'rolling_mean_{window}'] = window_data.mean()
            features[f'rolling_std_{window}'] = window_data.std()
    
    # Price changes
    for period in [1, 7, 14, 30]:
        if period < len(recent_data):
            pct_change = recent_data.pct_change(period).iloc[-1]
            features[f'price_change_{period}d'] = pct_change if not pd.isna(pct_change) else 0.0
    
    # Technical indicators
    if len(recent_data) >= 5:
        features['sma_5'] = recent_data.tail(5).mean()
    if len(recent_data) >= 20:
        features['sma_20'] = recent_data.tail(20).mean()
    
    # Date features
    last_date = recent_data.index[-1]
    features['day_of_week'] = last_date.dayofweek
    features['month'] = last_date.month
    features['quarter'] = last_date.quarter
    features['month_sin'] = np.sin(2 * np.pi * last_date.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * last_date.month / 12)
    
    return features

def make_prediction(currency_name, forecast_days, models, sample_data):
    """Make forecast for selected currency"""
    if currency_name not in models:
        return None, None
    
    model_data = models[currency_name]
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_columns = model_data['feature_columns']
    target_col = model_data['target_col']
    
    # Get current price
    current_price = sample_data[target_col].iloc[-1]
    last_date = sample_data.index[-1]
    
    # Generate predictions
    predictions = []
    current_data = sample_data.copy()
    
    for i in range(min(forecast_days, 30)):  # Limit to 30 days
        # Create features
        features = create_features(current_data, target_col)
        if features is None:
            break
        
        # Align with model features
        aligned_features = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for col in feature_columns:
            if col in features:
                aligned_features.loc[0, col] = features[col]
        
        # Make prediction
        feature_array = aligned_features.values
        if scaler is not None:
            feature_array = scaler.transform(feature_array)
        
        pred = model.predict(feature_array)[0]
        predictions.append(pred)
        
        # Add to data for next iteration
        next_date = last_date + timedelta(days=i+1)
        current_data.loc[next_date, target_col] = pred
    
    # Create forecast dataframe
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=len(predictions),
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Price': predictions
    }).set_index('Date')
    
    return forecast_df, current_price

# Load data and models
with st.spinner('Loading models and data...'):
    models = load_models()
    sample_data = create_sample_data()

if not models:
    st.error("❌ No trained models found!")
    st.stop()

# Sidebar
st.sidebar.markdown("## 🎛️ Prediction Settings")

# Currency selection
currency_options = sorted(models.keys())
selected_currency = st.sidebar.selectbox(
    "📈 Select Currency Pair:",
    currency_options
)

# Forecast horizon
forecast_days = st.sidebar.slider(
    "📅 Forecast Horizon (Days):",
    min_value=1,
    max_value=30,
    value=7
)

# Model info
if selected_currency:
    model_info = models[selected_currency]
    st.sidebar.markdown("## 📊 Model Information")
    st.sidebar.write(f"**Type:** {model_info.get('model_type', 'Unknown')}")
    perf = model_info.get('performance', {})
    if 'MAPE' in perf:
        st.sidebar.write(f"**MAPE:** {perf['MAPE']:.2f}%")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 🔮 Generate Forecast")
    
    if st.button("🚀 Generate Prediction", type="primary"):
        with st.spinner(f'Generating {forecast_days}-day forecast...'):
            forecast_df, current_price = make_prediction(
                selected_currency, 
                forecast_days, 
                models, 
                sample_data
            )
            
            if forecast_df is not None:
                st.session_state.forecast_df = forecast_df
                st.session_state.current_price = current_price
                st.session_state.selected_currency = selected_currency
                st.success("✅ Forecast generated successfully!")
            else:
                st.error("❌ Failed to generate forecast")

with col2:
    st.markdown("## 📊 Portfolio")
    st.metric("Models", len(models))
    
    avg_mape = np.mean([m.get('performance', {}).get('MAPE', 0) for m in models.values()])
    st.metric("Avg MAPE", f"{avg_mape:.2f}%")

# Display results
if hasattr(st.session_state, 'forecast_df') and st.session_state.forecast_df is not None:
    st.markdown("---")
    st.markdown("## 📊 Forecast Results")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chart
        forecast_df = st.session_state.forecast_df
        target_col = models[st.session_state.selected_currency]['target_col']
        
        # Historical data (last 60 days)
        hist_data = sample_data[target_col].tail(60)
        
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Predicted_Price'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{st.session_state.selected_currency} - Price Forecast',
            xaxis_title='Date',
            yaxis_title='Exchange Rate',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary
        current_price = st.session_state.current_price
        final_price = forecast_df['Predicted_Price'].iloc[-1]
        change = final_price - current_price
        change_pct = (change / current_price) * 100
        
        st.markdown("### 🎯 Summary")
        st.metric("Current Price", f"{current_price:.4f}")
        st.metric("Predicted Price", f"{final_price:.4f}", f"{change:+.4f}")
        st.metric("Expected Change", f"{change_pct:+.2f}%")
        
        trend = "📈 Bullish" if change > 0 else "📉 Bearish"
        st.write(f"**Trend:** {trend}")
    
    # Data table
    st.markdown("### 📋 Detailed Forecast")
    display_df = forecast_df.copy()
    display_df['Date'] = display_df.index.strftime('%Y-%m-%d')
    display_df['Price'] = display_df['Predicted_Price'].round(4)
    display_df['Daily Change'] = display_df['Predicted_Price'].diff().round(4)
    
    st.dataframe(
        display_df[['Date', 'Price', 'Daily Change']],
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>🚀 <strong>Forex Predictor Pro</strong> - Powered by Machine Learning</p>
<p>⚠️ <em>For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)