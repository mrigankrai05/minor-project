import os
import pandas as pd
import numpy as np
import nltk
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression 
from nltk.corpus import stopwords
import re
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pandas.tseries.offsets import DateOffset
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Initialization and Model Training ---
app = Flask(__name__)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2)) 
model = LinearSVC(random_state=42, dual=True, max_iter=2000) 

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def train_model():
    """Loads data, cleans it, and trains the sentiment model."""
    global vectorizer, model
    try:
        df_train = pd.read_csv('training_data.csv')
        df_train.dropna(subset=['review', 'sentiment'], inplace=True)
        df_train['cleaned_review'] = df_train['review'].apply(clean_text)
        X = vectorizer.fit_transform(df_train['cleaned_review'])
        y = df_train['sentiment']
        model.fit(X, y)
        print("✅ VIVA Version: Model trained successfully.")
    except Exception as e:
        print(f"❌ An error occurred during model training: {e}")

# --- Chart Generation Functions ---
def generate_sentiment_pie_chart(df):
    sentiment_counts = df['predicted_sentiment'].value_counts()
    if sentiment_counts.empty: return None
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    labels = sentiment_counts.index
    color_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#94a3b8'}
    colors = [color_map.get(label, '#6b7280') for label in labels]
    ax.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.axis('equal')
    plt.title('Sentiment Distribution', pad=20, fontweight='bold')
    return fig_to_base64(fig)

def generate_profit_over_time_chart(df):
    df_c = df.copy()
    df_c['purchase_date'] = pd.to_datetime(df_c['purchase_date'])
    df_time = df_c.set_index('purchase_date').resample('ME')['net_profit'].sum()
    if df_time.empty: return None
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot(df_time.index, df_time.values, marker='o', linestyle='-', color='#3b82f6')
    ax.set_title('Historical Profit Over Time', pad=20, fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Net Profit ($)')
    ax.grid(True, linestyle='--', alpha=0.6); plt.xticks(rotation=45)
    return fig_to_base64(fig)

def generate_forecast_chart(df, value_col, title, y_label):
    df_c = df.copy()
    df_c['purchase_date'] = pd.to_datetime(df_c['purchase_date'])
    df_ts = df_c.set_index('purchase_date').resample('ME')[value_col].sum()
    if len(df_ts) < 2: return None, 0
    df_ts = df_ts.reset_index(); df_ts['time'] = np.arange(len(df_ts.index))
    X, y = df_ts[['time']], df_ts[value_col]
    model_lr = LinearRegression().fit(X, y)
    last_time = df_ts['time'].max()
    future_time = np.arange(last_time + 1, last_time + 7).reshape(-1, 1)
    future_pred = model_lr.predict(future_time)
    last_date = df_ts['purchase_date'].max()
    future_dates = pd.to_datetime([last_date + DateOffset(months=i) for i in range(1, 7)])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot(df_ts['purchase_date'], y, label='Historical Data', marker='o', color='#3b82f6')
    ax.plot(future_dates, future_pred, label='Forecast', linestyle='--', marker='o', color='#f97316')
    ax.set_title(title, pad=20, fontweight='bold'); ax.set_xlabel('Month'); ax.set_ylabel(y_label)
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); plt.xticks(rotation=45)
    
    next_month_prediction = future_pred[0] if len(future_pred) > 0 else 0
    return fig_to_base64(fig), next_month_prediction

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/analyze', methods=['POST'])
def analyze_data():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    if not file.filename.endswith('.csv'): return jsonify({'error': 'Invalid file type'}), 400

    try:
        df_user = pd.read_csv(file)
        
        required_cols = ['review', 'purchase_date', 'net_profit', 'product_category', 'region', 'rating', 'units_sold']
        if not all(col in df_user.columns for col in required_cols):
             return jsonify({'error': f'CSV is missing required columns. It must contain: {", ".join(required_cols)}'}), 400

        df_user.dropna(subset=required_cols, inplace=True)
        if df_user.empty: return jsonify({'error': 'CSV has no valid data rows after cleaning.'}), 400

        df_user['cleaned_review'] = df_user['review'].apply(clean_text)
        X_user = vectorizer.transform(df_user['cleaned_review'])
        df_user['predicted_sentiment'] = model.predict(X_user)

        profit_forecast_chart, next_month_profit = generate_forecast_chart(df_user, 'net_profit', 'Profit Forecast (Next 6 Months)', 'Net Profit ($)')

        charts = {
            'sentiment_pie': generate_sentiment_pie_chart(df_user),
            'profit_time': generate_profit_over_time_chart(df_user),
            'profit_forecast': profit_forecast_chart,
        }
        
        # Calculate summary statistics
        sentiment_counts = df_user['predicted_sentiment'].value_counts()
        total_reviews = len(df_user)
        total_profit = df_user['net_profit'].sum()
        df_user['purchase_date'] = pd.to_datetime(df_user['purchase_date'])
        monthly_profit = df_user.set_index('purchase_date').resample('ME')['net_profit'].sum()

        negative_reviews = df_user[df_user['predicted_sentiment'] == 'Negative']['cleaned_review']
        trends = []
        if not negative_reviews.empty and negative_reviews.str.strip().astype(bool).any():
            try:
                trend_vectorizer = TfidfVectorizer(max_features=10, ngram_range=(1,2)).fit(negative_reviews)
                trends = trend_vectorizer.get_feature_names_out().tolist()
            except ValueError:
                trends = []

        category_profit = df_user.groupby('product_category')['net_profit'].sum()
        top_category = category_profit.idxmax() if not category_profit.empty else "N/A"
        top_category_profit = category_profit.max() if not category_profit.empty else 0
        
        region_sales = df_user.groupby('region')['units_sold'].sum()
        top_region = region_sales.idxmax() if not region_sales.empty else "N/A"
        top_region_sales = int(region_sales.max()) if not region_sales.empty else 0
        
        avg_rating = df_user['rating'].mean()

        summary_stats = {
            'positive_count': int(sentiment_counts.get('Positive', 0)),
            'negative_count': int(sentiment_counts.get('Negative', 0)),
            'neutral_count': int(sentiment_counts.get('Neutral', 0)),
            'positive_pct': f"{(sentiment_counts.get('Positive', 0) / total_reviews) * 100:.1f}%" if total_reviews > 0 else "0.0%",
            'negative_pct': f"{(sentiment_counts.get('Negative', 0) / total_reviews) * 100:.1f}%" if total_reviews > 0 else "0.0%",
            'neutral_pct': f"{(sentiment_counts.get('Neutral', 0) / total_reviews) * 100:.1f}%" if total_reviews > 0 else "0.0%",
            'total_profit': f"${total_profit:,.2f}",
            'avg_monthly_profit': f"${monthly_profit.mean():,.2f}" if not monthly_profit.empty else "$0.00",
            'next_month_profit': f"${next_month_profit:,.2f}",
            'top_category': f"{top_category} (${top_category_profit:,.2f})",
            'top_region': f"{top_region} ({top_region_sales:,} units)",
            'avg_rating': f"{avg_rating:.2f} out of 5" if not pd.isna(avg_rating) else "N/A"
        }
        
        return jsonify({'charts': charts, 'summary_stats': summary_stats, 'trends': trends})

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({'error': f"An error occurred on the server: {e}"}), 500

if __name__ == '__main__':
    train_model()
    app.run(port=5001, debug=True)