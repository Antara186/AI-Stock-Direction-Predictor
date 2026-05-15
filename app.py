# =========================================================
# AI STOCK DIRECTION PREDICTION SYSTEM
# Advanced Quant Research Dashboard
# Built for IndiMinds 2026
# =========================================================
import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Stock Prediction System",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📊 AI Stock Direction Prediction System")

st.markdown("""
### Advanced Quantitative Research Dashboard

This platform uses:

- Engineered Market Features
- Technical Indicators
- Machine Learning Models
- Quantitative Analysis
- AI-Based Direction Prediction

to forecast future market direction.
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙ Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Stock CSV",
    type=["csv"]
)

# =========================================================
# FILE CHECK
# =========================================================

if uploaded_file is None:
    st.info("⬅ Upload stock dataset from sidebar")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv(uploaded_file)

except:
    st.error("❌ Invalid CSV File")
    st.stop()

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_cols = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing_cols = [
    c for c in required_cols
    if c not in df.columns
]

if missing_cols:
    st.error(f"❌ Missing Columns: {missing_cols}")
    st.stop()

# =========================================================
# DATE FORMAT
# =========================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

# =========================================================
# CLEAN DATA
# =========================================================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna().reset_index(drop=True)

# =========================================================
# DATA PREVIEW
# =========================================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

st.sidebar.subheader("🧠 Feature Selection")

available_features = [
    "RSI",
    "SMA",
    "EMA",
    "MACD",
    "Daily Return",
    "Volatility",
    "Price Change",
    "High Low Range",
    "Bollinger Bands"
]

selected_features = st.sidebar.multiselect(
    "Select Engineered Features",
    available_features,
    default=available_features
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

# DAILY RETURN
df["Daily_Return"] = df["Close"].pct_change()

# DYNAMIC WINDOWS
sma_window = min(50, max(5, len(df)//4))
vol_window = min(20, max(3, len(df)//10))

# SMA
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(sma_window).mean()

# EMA
df["EMA_12"] = df["Close"].ewm(span=12).mean()
df["EMA_26"] = df["Close"].ewm(span=26).mean()

# RSI
df["RSI"] = RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# MACD
macd = MACD(close=df["Close"])

df["MACD"] = macd.macd()
df["Signal_Line"] = macd.macd_signal()

# VOLATILITY
df["Volatility"] = (
    df["Daily_Return"]
    .rolling(vol_window)
    .std()
)

# EXTRA FEATURES
df["Price_Change"] = (
    df["Close"] - df["Open"]
)

df["High_Low_Range"] = (
    df["High"] - df["Low"]
)

# BOLLINGER BANDS
bb = BollingerBands(close=df["Close"])

df["BB_High"] = bb.bollinger_hband()
df["BB_Low"] = bb.bollinger_lband()

# =========================================================
# TARGET
# =========================================================

df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

# =========================================================
# FINAL CLEAN
# =========================================================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna().reset_index(drop=True)

# =========================================================
# SMART DATASET HANDLING
# =========================================================

total_rows = len(df)

st.sidebar.write(f"📦 Total Rows: {total_rows}")

if total_rows < 15:

    st.error(
        "❌ Dataset too small for prediction. "
        "Upload at least 15 rows."
    )

    st.stop()

elif total_rows < 40:

    st.warning(
        "⚠ Small dataset detected. "
        "Using lightweight prediction mode."
    )

    split_ratio = 0.7

elif total_rows < 100:

    st.info(
        "✅ Medium dataset detected."
    )

    split_ratio = 0.8

else:

    st.success(
        "🚀 Large dataset detected."
    )

    split_ratio = 0.85

# =========================================================
# DYNAMIC FEATURES
# =========================================================

features = []

if "RSI" in selected_features:
    features.append("RSI")

if "SMA" in selected_features:
    features.extend([
        "SMA_20",
        "SMA_50"
    ])

if "EMA" in selected_features:
    features.extend([
        "EMA_12",
        "EMA_26"
    ])

if "MACD" in selected_features:
    features.extend([
        "MACD",
        "Signal_Line"
    ])

if "Daily Return" in selected_features:
    features.append("Daily_Return")

if "Volatility" in selected_features:
    features.append("Volatility")

if "Price Change" in selected_features:
    features.append("Price_Change")

if "High Low Range" in selected_features:
    features.append("High_Low_Range")

if "Bollinger Bands" in selected_features:
    features.extend([
        "BB_High",
        "BB_Low"
    ])

# ALWAYS ADD VOLUME
features.append("Volume")

# =========================================================
# FEATURE CHECK
# =========================================================

if len(features) == 0:

    st.error("❌ Please select at least one feature")

    st.stop()

# =========================================================
# ML DATA
# =========================================================

X = df[features]

y = df["Target"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split = int(len(df) * split_ratio)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# =========================================================
# SAFETY CHECK
# =========================================================

if len(X_train) < 5 or len(X_test) < 2:

    st.error(
        "❌ Not enough data after split."
    )

    st.stop()

# =========================================================
# MODEL SELECTION
# =========================================================

st.sidebar.subheader("🤖 Model Selection")

model_option = st.sidebar.selectbox(
    "Choose Model",
    [
        "Random Forest",
        "Logistic Regression",
        "Ensemble"
    ]
)

# =========================================================
# MODELS
# =========================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

lr = LogisticRegression(
    max_iter=1000
)

# =========================================================
# TRAINING
# =========================================================

rf.fit(X_train, y_train)

lr.fit(X_train, y_train)

# =========================================================
# TEST PREDICTIONS
# =========================================================

if model_option == "Random Forest":

    y_pred = rf.predict(X_test)

elif model_option == "Logistic Regression":

    y_pred = lr.predict(X_test)

else:

    rf_pred = rf.predict(X_test)
    lr_pred = lr.predict(X_test)

    ensemble = (
        rf_pred + lr_pred
    ) / 2

    y_pred = np.where(
        ensemble >= 0.5,
        1,
        0
    )

# =========================================================
# METRICS
# =========================================================

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# =========================================================
# DASHBOARD METRICS
# =========================================================

st.subheader("📊 Model Accuracy Report")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Accuracy",
    f"{acc:.2%}"
)

c2.metric(
    "Precision",
    f"{prec:.2%}"
)

c3.metric(
    "Recall",
    f"{rec:.2%}"
)

# =========================================================
# LATEST PREDICTION
# =========================================================

latest = X.iloc[-1:].values

rf_prob = rf.predict_proba(latest)[0][1]

lr_prob = lr.predict_proba(latest)[0][1]

if model_option == "Random Forest":

    prob = rf_prob

elif model_option == "Logistic Regression":

    prob = lr_prob

else:

    prob = (rf_prob + lr_prob) / 2

pred = 1 if prob > 0.5 else 0

# =========================================================
# PREDICTION RESULT
# =========================================================

st.subheader("🔮 AI Market Prediction")

if pred == 1:
    st.success("📈 Tomorrow Market Direction → UP")
else:
    st.error("📉 Tomorrow Market Direction → DOWN")

st.write(
    "### Confidence:",
    f"{prob*100:.2f}%"
)

st.progress(float(prob))

# =========================================================
# BUY SELL SIGNAL
# =========================================================

latest_rsi = df["RSI"].iloc[-1]

if prob > 0.60 and latest_rsi < 70:
    st.success("🟢 BUY SIGNAL")

elif prob < 0.45 and latest_rsi > 30:
    st.warning("🔴 SELL SIGNAL")

else:
    st.info("🟡 HOLD SIGNAL")

# =========================================================
# CHART SELECTION
# =========================================================

st.sidebar.subheader("📈 Select Charts")

chart_options = st.sidebar.multiselect(
    "Choose Charts",
    [
        "Candlestick",
        "Price Trend",
        "RSI",
        "MACD",
        "Volume",
        "Bollinger Bands"
    ],
    default=[
        "Candlestick",
        "Price Trend",
        "RSI"
    ]
)

# =========================================================
# CANDLESTICK CHART
# =========================================================

if "Candlestick" in chart_options:

    st.subheader("🕯 Candlestick Chart")

    fig = go.Figure(data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
    ])

    fig.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# PRICE TREND
# =========================================================

if "Price Trend" in chart_options:

    st.subheader("📈 Price Trend Analysis")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price"
    ))

    if "SMA" in selected_features:

        fig2.add_trace(go.Scatter(
            x=df["Date"],
            y=df["SMA_20"],
            mode="lines",
            name="SMA 20"
        ))

    if "EMA" in selected_features:

        fig2.add_trace(go.Scatter(
            x=df["Date"],
            y=df["EMA_12"],
            mode="lines",
            name="EMA 12"
        ))

    fig2.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================================================
# RSI GRAPH
# =========================================================

if "RSI" in chart_options:

    st.subheader("📉 RSI Indicator")

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=df["Date"],
        y=df["RSI"],
        mode="lines",
        name="RSI"
    ))

    fig3.add_hline(y=70)

    fig3.add_hline(y=30)

    fig3.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =========================================================
# MACD GRAPH
# =========================================================

if "MACD" in chart_options:

    st.subheader("📊 MACD Indicator")

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD"],
        mode="lines",
        name="MACD"
    ))

    fig4.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Signal_Line"],
        mode="lines",
        name="Signal Line"
    ))

    fig4.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# =========================================================
# VOLUME CHART
# =========================================================

if "Volume" in chart_options:

    st.subheader("📦 Volume Analysis")

    fig5 = go.Figure()

    fig5.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"],
        name="Volume"
    ))

    fig5.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

# =========================================================
# BOLLINGER BANDS
# =========================================================

if "Bollinger Bands" in chart_options:

    st.subheader("📉 Bollinger Bands")

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price"
    ))

    fig6.add_trace(go.Scatter(
        x=df["Date"],
        y=df["BB_High"],
        mode="lines",
        name="Upper Band"
    ))

    fig6.add_trace(go.Scatter(
        x=df["Date"],
        y=df["BB_Low"],
        mode="lines",
        name="Lower Band"
    ))

    fig6.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("📋 Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    columns=["Pred Down", "Pred Up"],
    index=["Actual Down", "Actual Up"]
)

st.dataframe(cm_df)

# =========================================================
# ACTIVE FEATURES
# =========================================================

st.subheader("🧠 Active Features")

st.write(features)

# =========================================================
# FINAL DATASET
# =========================================================

st.subheader("📌 Final Engineered Dataset")

st.dataframe(df.tail())

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
### 🚀 Built for IndiMinds 2026

Domains Covered:
- Quantitative Research
- AI/ML
- Finance
- Predictive Analytics
- Full Stack Development
""")
# =========================================================
# THEORY SECTION
# Add this section after title and before file upload
# =========================================================

st.markdown("---")

st.header("📘 Theory & Concept Explanation")

with st.expander("📊 What is Stock Market Direction Prediction?"):

    st.markdown("""
    Stock Market Direction Prediction means predicting whether a stock price
    will move UP 📈 or DOWN 📉 in the future.

    This project uses:

    - Historical stock market data
    - Technical indicators
    - Machine Learning models

    to analyze market behavior and predict future trends.
    """)

# =========================================================
# TECHNICAL INDICATORS THEORY
# =========================================================

with st.expander("🧠 Engineered Market Features & Technical Indicators"):

    st.markdown("""
    ## 1. RSI (Relative Strength Index)

    RSI measures market momentum and identifies:

    - Overbought conditions
    - Oversold conditions

    ### Formula:

    RSI = 100 - (100 / (1 + RS))

    ### Interpretation:

    - RSI > 70 → Overbought
    - RSI < 30 → Oversold

    -------------------------------------------------------

    ## 2. SMA (Simple Moving Average)

    SMA calculates average stock price over a fixed period.

    It helps identify market trends.

    ### Formula:

    SMA = (P1 + P2 + ... + Pn) / n

    ### Interpretation:

    - Price above SMA → Bullish trend
    - Price below SMA → Bearish trend

    -------------------------------------------------------

    ## 3. EMA (Exponential Moving Average)

    EMA gives more importance to recent prices.

    It reacts faster than SMA.

    ### Use:

    - Detect trend changes
    - Better short-term analysis

    -------------------------------------------------------

    ## 4. MACD (Moving Average Convergence Divergence)

    MACD measures trend strength and momentum.

    ### Formula:

    MACD = EMA(12) - EMA(26)

    Signal Line = 9-day EMA of MACD

    ### Interpretation:

    - MACD above Signal Line → Bullish
    - MACD below Signal Line → Bearish

    -------------------------------------------------------

    ## 5. Daily Return

    Daily Return shows percentage change in stock price.

    ### Formula:

    Daily Return = (Today's Close - Previous Close) / Previous Close

    -------------------------------------------------------

    ## 6. Volatility

    Volatility measures market risk and price fluctuations.

    Higher volatility means larger price movement.

    -------------------------------------------------------

    ## 7. Bollinger Bands

    Bollinger Bands measure price volatility.

    They consist of:

    - Upper Band
    - Lower Band
    - Moving Average

    ### Interpretation:

    - Price near upper band → Overbought
    - Price near lower band → Oversold
    """)

# =========================================================
# MACHINE LEARNING THEORY
# =========================================================

with st.expander("🤖 Machine Learning Models Used"):

    st.markdown("""
    ## 1. Random Forest Classifier

    Random Forest is an ensemble learning algorithm.

    It uses multiple decision trees to make predictions.

    ### Advantages:

    - High accuracy
    - Reduces overfitting
    - Handles complex data

    -------------------------------------------------------

    ## 2. Logistic Regression

    Logistic Regression is used for classification problems.

    In this project:

    - 1 → Market UP
    - 0 → Market DOWN

    It predicts probability using statistical relationships.

    -------------------------------------------------------

    ## 3. Ensemble Prediction

    This project combines:

    - Random Forest
    - Logistic Regression

    to improve prediction accuracy.
    """)

# =========================================================
# PROJECT WORKFLOW THEORY
# =========================================================

with st.expander("⚙ Project Workflow"):

    st.markdown("""
    ## Complete Workflow

    ### Step 1 → Upload Dataset

    User uploads stock market CSV file.

    Required Columns:

    - Date
    - Open
    - High
    - Low
    - Close
    - Volume

    -------------------------------------------------------

    ### Step 2 → Data Cleaning

    System removes:

    - Missing values
    - Invalid values
    - Duplicate records

    -------------------------------------------------------

    ### Step 3 → Feature Engineering

    The system creates technical indicators like:

    - RSI
    - SMA
    - EMA
    - MACD
    - Volatility

    -------------------------------------------------------

    ### Step 4 → Machine Learning Training

    ML models learn historical market patterns.

    -------------------------------------------------------

    ### Step 5 → Prediction

    System predicts:

    - UP 📈
    - DOWN 📉

    -------------------------------------------------------

    ### Step 6 → Buy/Sell Signal

    Based on:

    - Prediction probability
    - RSI value
    - Trend indicators

    -------------------------------------------------------

    ### Step 7 → Visualization

    Interactive charts display:

    - Candlestick chart
    - RSI graph
    - MACD chart
    - Price trends
    """)

# =========================================================
# MODEL PERFORMANCE THEORY
# =========================================================

with st.expander("📊 Accuracy Metrics Explanation"):

    st.markdown("""
    ## Accuracy

    Accuracy shows total correct predictions.

    -------------------------------------------------------

    ## Precision

    Precision measures prediction quality.

    High precision means fewer false predictions.

    -------------------------------------------------------

    ## Recall

    Recall measures how many actual positive cases
    were correctly predicted.

    -------------------------------------------------------

    ## Confusion Matrix

    Confusion Matrix compares:

    - Actual Results
    - Predicted Results

    It helps evaluate ML model performance.
    """)

# =========================================================
# FUTURE SCOPE
# =========================================================

with st.expander("🚀 Future Scope"):

    st.markdown("""
    Future improvements can include:

    - Live stock market API integration
    - Real-time prediction system
    - AI chatbot assistant
    - Portfolio management
    - Deep Learning models
    - News sentiment analysis
    - Cloud deployment
    - Multi-stock comparison
    - Automated trading system
    """)
    
# CORRELATION HEATMAP
# =========================================================

st.subheader("🔥 Correlation Heatmap")

corr_df = df[features].corr()

fig_heat = px.imshow(
    corr_df,
    text_auto=True,
    aspect="auto",
    title="Feature Correlation Heatmap"
)

fig_heat.update_layout(
    template="plotly_dark",
    height=700
)

st.plotly_chart(
    fig_heat,
    use_container_width=True
)
# =========================================================
# RISK ANALYSIS
# =========================================================

st.subheader("⚠ Risk Analysis")

latest_volatility = df["Volatility"].iloc[-1]

latest_rsi = df["RSI"].iloc[-1]

risk_score = 0

# VOLATILITY CHECK

if latest_volatility > 0.05:
    risk_score += 2

elif latest_volatility > 0.02:
    risk_score += 1

# RSI CHECK

if latest_rsi > 75 or latest_rsi < 25:
    risk_score += 2

elif latest_rsi > 65 or latest_rsi < 35:
    risk_score += 1

# FINAL RISK LEVEL

if risk_score >= 3:

    st.error("🔴 HIGH RISK MARKET")

    risk_level = "High Risk"

elif risk_score >= 2:

    st.warning("🟠 MEDIUM RISK MARKET")

    risk_level = "Medium Risk"

else:

    st.success("🟢 LOW RISK MARKET")

    risk_level = "Low Risk"

# RISK DETAILS

risk_df = pd.DataFrame({
    "Metric": [
        "Volatility",
        "RSI",
        "Risk Level"
    ],
    "Value": [
        round(latest_volatility, 4),
        round(latest_rsi, 2),
        risk_level
    ]
})

st.dataframe(risk_df)
# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

st.subheader("📊 Feature Importance Analysis")

try:

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    fig_importance.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    st.dataframe(importance_df)

except Exception as e:

    st.warning(
        "Feature Importance Graph Available Only For Random Forest Model"
    )