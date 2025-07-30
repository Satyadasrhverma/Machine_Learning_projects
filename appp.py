import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
from datetime import datetime
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="🌦️ All India Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_KEY = "979180be6e731e506a54fcb46d0859ea"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #007bff;
    margin: 1rem 0;
}
.weather-metric {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load Model Function
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('balanced_rain_model.pkl', 'rb'))
        feature_columns = pickle.load(open('balanced_feature_columns.pkl', 'rb'))
        return model, feature_columns
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure model files are in the same folder.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# Weather API Function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_realtime_weather(city, api_key):
    try:
        city_encoded = urllib.parse.quote(city.strip())
        url = f"{BASE_URL}?q={city_encoded}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'main' not in data or 'weather' not in data:
                return None
                
            weather_data = {
                'city': city,
                'current_temp': data['main']['temp'],
                'temp_max': data['main']['temp_max'],
                'temp_min': data['main']['temp_min'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'clouds': data.get('clouds', {}).get('all', 0),
                'visibility': data.get('visibility', 10000) / 1000,
                'weather_desc': data['weather'][0]['description'],
                'feels_like': data['main']['feels_like']
            }
            return weather_data
        else:
            return None
    except Exception as e:
        st.error(f"Weather API Error: {str(e)}")
        return None

# Prediction Function
def predict_rain_rt(city, api_key, model, feature_columns):
    # Get weather data
    weather_data = get_realtime_weather(city, api_key)
    if not weather_data:
        return None, None, None

    # Current season
    current_month = datetime.now().month
    current_season = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon',
        9: 'Post-Monsoon', 10: 'Post-Monsoon', 11: 'Post-Monsoon'
    }[current_month]

    # Prepare features
    user_data = {
        'tmax': weather_data['temp_max'],
        'tmin': weather_data['temp_min'],
        'temp_range': weather_data['temp_max'] - weather_data['temp_min'],
        'avg_temp': (weather_data['temp_max'] + weather_data['temp_min']) / 2,
        'humidity': weather_data['humidity'],
        'pressure': weather_data['pressure'],
        'wind_speed': weather_data['wind_speed'],
        'clouds': weather_data['clouds'],
        'visibility': weather_data['visibility']
    }

    # Season encoding
    for season in ['Winter', 'Spring', 'Monsoon', 'Post-Monsoon']:
        user_data[f'season_{season}'] = 1 if current_season == season else 0

    # City encoding
    cities = [col.replace('city_', '') for col in feature_columns if col.startswith('city_')]
    if city not in cities:
        return None, None, None
        
    for c in cities:
        user_data[f'city_{c}'] = 1 if city == c else 0

    # Make prediction
    user_df = pd.DataFrame([user_data])
    user_df = user_df.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(user_df)
    probability = model.predict_proba(user_df)[0]

    return prediction[0], probability[1], weather_data

# Load model
model, feature_columns = load_model()

# Main App Layout
st.markdown('<div class="main-header"><h1>🌦️ All India Rainfall Prediction System</h1><p>Real-Time Weather API Based ML Prediction</p></div>', unsafe_allow_html=True)

# Sidebar for City Selection
with st.sidebar:
    st.header("🏙️ Select City")
    
    # Get available cities from model
    if feature_columns:
        available_cities = sorted([col.replace('city_', '') for col in feature_columns if col.startswith('city_')])
        
        # Popular cities for quick selection
        popular_cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Lucknow", 
                         "Kolkata", "Jaipur", "Pune", "Hyderabad", "Ahmedabad"]
        popular_in_model = [city for city in popular_cities if city in available_cities]
        
        st.subheader("🔥 Popular Cities")
        for city in popular_in_model[:5]:
            if st.button(f"📍 {city}", key=f"pop_{city}", use_container_width=True):
                st.session_state.selected_city = city
        
        st.markdown("---")
        
        # Dropdown selection
        st.subheader("📋 All Available Cities")
        selected_city = st.selectbox(
            "Choose from all cities:",
            ["Select a city..."] + available_cities,
            key="city_dropdown"
        )
        
        if selected_city != "Select a city...":
            st.session_state.selected_city = selected_city
        
        st.markdown("---")
        
        # Manual input
        st.subheader("✏️ Enter Manually")
        manual_city = st.text_input("Type city name:", key="manual_input")
        if manual_city and manual_city in available_cities:
            st.session_state.selected_city = manual_city
        elif manual_city and manual_city not in available_cities:
            st.error(f"❌ '{manual_city}' not in trained cities")
    
    # Info section
    st.markdown("---")
    st.info("💡 **Tip:** This system uses real-time weather data and machine learning to predict rainfall")
    
    if feature_columns:
        st.success(f"📊 Model trained on {len([c for c in feature_columns if c.startswith('city_')])} cities")

# Main Content Area
if 'selected_city' in st.session_state and st.session_state.selected_city:
    selected_city = st.session_state.selected_city
    
    st.header(f"🔮 Rainfall Prediction for {selected_city}")
    
    # Prediction button
    if st.button("🚀 Get Rainfall Prediction", type="primary", use_container_width=True):
        if model is not None and feature_columns is not None:
            with st.spinner(f"🌐 Fetching real-time weather data for {selected_city}..."):
                prediction, confidence, weather_data = predict_rain_rt(
                    selected_city, API_KEY, model, feature_columns
                )
            
            if prediction is not None and weather_data is not None:
                # Weather Data Display
                st.subheader("🌤️ Current Weather Conditions")
                
                # Weather metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="weather-metric">
                        <h3>🌡️</h3>
                        <h2>{weather_data['current_temp']:.1f}°C</h2>
                        <p>Temperature</p>
                        <small>Feels like {weather_data['feels_like']:.1f}°C</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="weather-metric">
                        <h3>💧</h3>
                        <h2>{weather_data['humidity']}%</h2>
                        <p>Humidity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="weather-metric">
                        <h3>🌪️</h3>
                        <h2>{weather_data['wind_speed']:.1f}</h2>
                        <p>Wind (m/s)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="weather-metric">
                        <h3>☁️</h3>
                        <h2>{weather_data['clouds']}%</h2>
                        <p>Cloud Cover</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional weather info
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("🏔️ Pressure", f"{weather_data['pressure']} hPa")
                    st.metric("👁️ Visibility", f"{weather_data['visibility']} km")
                with col6:
                    st.metric("🌡️ Max/Min", f"{weather_data['temp_max']}°C / {weather_data['temp_min']}°C")
                    current_month = datetime.now().month
                    current_season = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Post-Monsoon', 10: 'Post-Monsoon', 11: 'Post-Monsoon'}[current_month]
                    st.metric("🗓️ Season", current_season)
                
                st.info(f"📝 **Current Condition:** {weather_data['weather_desc'].title()}")
                
                # Prediction Results
                st.markdown("---")
                st.subheader("🔮 Rainfall Prediction Results")
                
                # Prediction display
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card" style="border-left-color: #28a745;">
                        <h2>🌧️ <strong>RAIN PREDICTED: YES</strong></h2>
                        <h3>आज बारिश होगी!</h3>
                        <p style="font-size: 1.2em;">Rain Probability: <strong>{confidence:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="prediction-card" style="border-left-color: #ffc107;">
                        <h2>☀️ <strong>RAIN PREDICTED: NO</strong></h2>
                        <h3>आज बारिश नहीं होगी</h3>
                        <p style="font-size: 1.2em;">No Rain Probability: <strong>{1-confidence:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.subheader("📊 Prediction Confidence")
                
                # Progress bar
                confidence_pct = confidence * 100
                st.progress(confidence_pct / 100)
                
                # Confidence chart
                fig = go.Figure(go.Bar(
                    x=['No Rain', 'Rain'],
                    y=[100-confidence_pct, confidence_pct],
                    marker_color=['#ffc107', '#007bff']
                ))
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability (%)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Weather Insights
                st.subheader("💡 Weather Analysis Insights")
                insights = []
                if weather_data['humidity'] > 80:
                    insights.append("🔹 **High humidity** detected - favorable conditions for precipitation")
                if weather_data['clouds'] > 70:
                    insights.append("🔹 **Heavy cloud cover** - increased probability of rain")
                if weather_data['wind_speed'] > 10:
                    insights.append("🔹 **Strong winds** - active weather system detected")
                if weather_data['pressure'] < 1010:
                    insights.append("🔹 **Low atmospheric pressure** - unstable weather conditions")
                if weather_data['temp_range'] > 15:
                    insights.append("🔹 **High temperature range** - potential for weather changes")
                
                if insights:
                    for insight in insights:
                        st.markdown(insight)
                else:
                    st.success("🔹 **Normal weather conditions** observed - stable atmospheric patterns")
                
            else:
                st.error(f"❌ Could not fetch weather data for {selected_city}. Please check:")
                st.write("• Internet connection")
                st.write("• City name spelling")
                st.write("• API service availability")
        else:
            st.error("❌ Model not loaded. Please check model files.")

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to the All India Rainfall Prediction System!
    
    ### How to use:
    1. **Select a city** from the sidebar
    2. **Click the prediction button** to get real-time weather analysis
    3. **View detailed results** with confidence levels and weather insights
    
    ### Features:
    - 🌐 **Real-time weather data** from OpenWeatherMap API
    - 🤖 **Machine Learning prediction** using advanced algorithms  
    - 📊 **Detailed weather analysis** with multiple parameters
    - 🎯 **High accuracy** based on balanced training data
    - 🏙️ **Multiple Indian cities** support
    
    **👈 Start by selecting a city from the sidebar!**
    """)
    
    # Sample cities showcase
    if feature_columns:
        available_cities = [col.replace('city_', '') for col in feature_columns if col.startswith('city_')]
        st.subheader("🌍 Available Cities")
        
        # Display cities in columns
        cities_cols = st.columns(4)
        for i, city in enumerate(available_cities[:20]):  # Show first 20 cities
            with cities_cols[i % 4]:
                if st.button(city, key=f"welcome_{city}"):
                    st.session_state.selected_city = city
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🌦️ Made with ❤️ using Streamlit & Machine Learning</p>
    <p>Powered by OpenWeatherMap API | Real-time Weather Data</p>
</div>
""", unsafe_allow_html=True)
# End of appp.py