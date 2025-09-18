import os
import pandas as pd
import numpy as np
import nltk
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import confusion_matrix
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

# Suppress the UserWarning from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Initialization and Model Training ---
app = Flask(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
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
    """Loads data, cleans it, trains the sentiment model, and fits the vectorizer."""
    global vectorizer, model
    try:
        df_train = pd.read_csv('training_data.csv')
        df_train.dropna(subset=['review', 'sentiment'], inplace=True)
        
        if 'sentiment' not in df_train.columns:
            raise ValueError("training_data.csv must have a 'sentiment' column.")
            
        df_train['cleaned_review'] = df_train['review'].apply(clean_text)
        X = vectorizer.fit_transform(df_train['cleaned_review'])
        y = df_train['sentiment']
        model.fit(X, y)
        print("✅ Upgraded model trained successfully on startup.")
    except FileNotFoundError:
        print("❌ CRITICAL ERROR: training_data.csv not found. The app cannot function without it.")
    except Exception as e:
        print(f"❌ An error occurred during model training: {e}")

# --- Chart Generation Functions ---
def generate_sentiment_pie_chart(df):
    sentiment_counts = df['predicted_sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    labels = sentiment_counts.index
    color_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#94a3b8'}
    colors = [color_map.get(label, '#6b7280') for label in labels]
    ax.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.axis('equal')
    plt.title('Predicted Sentiment Distribution', pad=20, fontweight='bold')
    return fig_to_base64(fig)

def generate_profit_over_time_chart(df):
    df_c = df.copy()
    df_c['purchase_date'] = pd.to_datetime(df_c['purchase_date'])
    df_time = df_c.set_index('purchase_date').resample('ME')['net_profit'].sum()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df_time.index, df_time.values, marker='o', linestyle='-', color='#3b82f6')
    ax.set_title('Total Profit Over Time', pad=20, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Net Profit ($)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    return fig_to_base64(fig)

def generate_sales_by_region_chart(df):
    region_sales = df.groupby('region')['units_sold'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    sns.barplot(x=region_sales.index, y=region_sales.values, palette='viridis', ax=ax, hue=region_sales.index, legend=False)
    ax.set_title('Total Units Sold by Region', pad=20, fontweight='bold')
    ax.set_xlabel('Region')
    ax.set_ylabel('Total Units Sold')
    return fig_to_base64(fig)

def generate_rating_sentiment_heatmap(df):
    if df.empty: return ""
    contingency_table = pd.crosstab(df['rating'], df['predicted_sentiment'])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Star Rating vs. Predicted Sentiment', pad=20, fontweight='bold')
    ax.set_xlabel('Predicted Sentiment')
    ax.set_ylabel('Customer Star Rating')
    return fig_to_base64(fig)
    
def generate_category_performance_chart(df):
    category_profit = df.groupby('product_category')['net_profit'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    sns.barplot(x=category_profit.index, y=category_profit.values, palette='plasma', ax=ax, hue=category_profit.index, legend=False)
    ax.set_title('Total Profit by Product Category', pad=20, fontweight='bold')
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Total Net Profit ($)')
    plt.xticks(rotation=45, ha='right')
    return fig_to_base64(fig)

def generate_recommendations(df):
    if df.empty: return {}
    recommendations = {}
    
    # Category Analysis for High Potential & Optimize
    category_analysis = df.groupby('product_category').agg(
        total_profit=('net_profit', 'sum'),
        total_units_sold=('units_sold', 'sum')
    ).reset_index()
    category_analysis = category_analysis[category_analysis['total_units_sold'] > 0]
    
    if not category_analysis.empty:
        category_analysis['profit_per_unit'] = category_analysis['total_profit'] / category_analysis['total_units_sold']
        median_profit = category_analysis['profit_per_unit'].median()
        median_sales = category_analysis['total_units_sold'].median()
        
        high_potential = category_analysis[(category_analysis['profit_per_unit'] > median_profit) & (category_analysis['total_units_sold'] < median_sales)]
        if not high_potential.empty:
            focus_cat = high_potential.sort_values('profit_per_unit', ascending=False).iloc[0]
            recommendations['focus_on'] = f"Focus marketing on '{focus_cat['product_category']}'. It has a high profit margin (${focus_cat['profit_per_unit']:.2f}/unit) but low sales volume."

        cash_cows = category_analysis[(category_analysis['profit_per_unit'] < median_profit) & (category_analysis['total_units_sold'] > median_sales)]
        if not cash_cows.empty:
            optimize_cat = cash_cows.sort_values('total_units_sold', ascending=False).iloc[0]
            recommendations['optimize'] = f"'{optimize_cat['product_category']}' is a 'cash cow' with high sales but low margins (${optimize_cat['profit_per_unit']:.2f}/unit). Explore cost reduction."

    # Product Level Analysis for Promote & Investigate
    product_performance = df.groupby('product_name').agg(
        total_profit=('net_profit', 'sum'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    if not product_performance.empty:
        best_product = product_performance.sort_values('total_profit', ascending=False).iloc[0]
        worst_product = product_performance[product_performance['avg_rating'] < 3].sort_values('total_profit').iloc[0] if not product_performance[product_performance['avg_rating'] < 3].empty else product_performance.sort_values('total_profit').iloc[0]
        
        recommendations['promote'] = f"Promote '{best_product['product_name']}'. It's your most profitable product, generating ${best_product['total_profit']:,.2f}."
        recommendations['investigate'] = f"Investigate '{worst_product['product_name']}'. It has a low average rating ({worst_product['avg_rating']:.1f}/5) and low profit."

    return recommendations

# --- Forecasting Functions ---
def generate_forecast_chart(df, value_col, title, y_label):
    df_c = df.copy()
    df_c['purchase_date'] = pd.to_datetime(df_c['purchase_date'])
    df_ts = df_c.set_index('purchase_date').resample('ME')[value_col].sum()
    if len(df_ts) < 2: return "", 0
    df_ts = df_ts.reset_index(); df_ts['time'] = np.arange(len(df_ts.index))
    X, y = df_ts[['time']], df_ts[value_col]
    model_lr = LinearRegression().fit(X, y)
    last_time = df_ts['time'].max()
    future_time = np.arange(last_time + 1, last_time + 7).reshape(-1, 1)
    future_pred = model_lr.predict(future_time)
    last_date = df_ts['purchase_date'].max()
    future_dates = pd.to_datetime([last_date + DateOffset(months=i) for i in range(1, 7)])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df_ts['purchase_date'], y, label='Historical Data', marker='o', color='#3b82f6')
    ax.plot(future_dates, future_pred, label='Forecast', linestyle='--', marker='o', color='#f97316')
    ax.set_title(title, pad=20, fontweight='bold'); ax.set_xlabel('Month'); ax.set_ylabel(y_label)
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); plt.xticks(rotation=45)
    return fig_to_base64(fig), future_pred[0] if len(future_pred) > 0 else 0

def generate_sentiment_trend_chart(df):
    df_c = df.copy()
    df_c['purchase_date'] = pd.to_datetime(df_c['purchase_date'])
    monthly_sentiment = df_c.set_index('purchase_date').resample('ME')['predicted_sentiment'].apply(lambda x: (x == 'Positive').sum() / len(x) * 100 if len(x) > 0 else 0)
    if len(monthly_sentiment) < 2: return "", 0
    df_ts = monthly_sentiment.reset_index(); df_ts.columns = ['purchase_date', 'positive_pct']; df_ts['time'] = np.arange(len(df_ts.index))
    X, y = df_ts[['time']], df_ts['positive_pct']
    model_lr = LinearRegression().fit(X, y)
    last_time = df_ts['time'].max()
    future_time = np.arange(last_time + 1, last_time + 7).reshape(-1, 1)
    future_pred = model_lr.predict(future_time)
    last_date = df_ts['purchase_date'].max()
    future_dates = pd.to_datetime([last_date + DateOffset(months=i) for i in range(1, 7)])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df_ts['purchase_date'], y, label='Historical Trend', marker='o', color='#22c55e')
    ax.plot(future_dates, future_pred, label='Forecast', linestyle='--', marker='o', color='#f97316')
    ax.set_title('Positive Sentiment Trend Forecast', pad=20, fontweight='bold'); ax.set_xlabel('Month'); ax.set_ylabel('Positive Reviews (%)'); ax.set_ylim(0, 100)
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); plt.xticks(rotation=45)
    return fig_to_base64(fig), future_pred[0] if len(future_pred) > 0 else 0

def generate_top_category_sales_forecast(df):
    if df.empty or df['units_sold'].sum() == 0: return "", "N/A", 0
    top_category = df.groupby('product_category')['units_sold'].sum().idxmax()
    df_category = df[df['product_category'] == top_category].copy()
    chart_b64, next_month_pred = generate_forecast_chart(df_category, 'units_sold', f'Forecast for Top Category: {top_category}', 'Units Sold')
    return chart_b64, top_category, next_month_pred

def predict_top_selling_product(df):
    if df.empty: return "N/A"
    df_c = df.copy()
    df_c['purchase_date'] = pd.to_datetime(df_c['purchase_date'])
    most_recent_date = df_c['purchase_date'].max()
    three_months_ago = most_recent_date - DateOffset(months=3)
    recent_sales = df_c[df_c['purchase_date'] >= three_months_ago]
    if recent_sales.empty: return "Not enough recent data"
    top_product = recent_sales.groupby('product_name')['units_sold'].sum().idxmax()
    return top_product

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file type. Please upload a CSV.'}), 400

    try:
        df_user = pd.read_csv(file)
        required_cols = ['product_name', 'product_category', 'review', 'rating', 'net_profit', 'units_sold', 'purchase_date', 'region']
        if not all(col in df_user.columns for col in required_cols):
             return jsonify({'error': f'CSV is missing required columns. It must contain: {", ".join(required_cols)}'}), 400

        df_user['purchase_date'] = pd.to_datetime(df_user['purchase_date'], errors='coerce', dayfirst=False)
        df_user['rating'] = pd.to_numeric(df_user['rating'], errors='coerce')
        df_user.dropna(subset=['purchase_date', 'rating', 'review'], inplace=True)
        
        if df_user.empty:
            return jsonify({'error': 'No valid data rows found after cleaning. Please check date formats and ensure ratings and reviews are not empty.'}), 400

        df_user['rating'] = df_user['rating'].astype(int)
        df_user['cleaned_review'] = df_user['review'].apply(clean_text)
        X_user = vectorizer.transform(df_user['cleaned_review'])
        df_user['predicted_sentiment'] = model.predict(X_user)
        
        negative_reviews = df_user[df_user['predicted_sentiment'] == 'Negative']['cleaned_review']
        trends = []
        if not negative_reviews.empty and negative_reviews.str.strip().astype(bool).any():
            try:
                trend_vectorizer = TfidfVectorizer(max_features=10, ngram_range=(1,2)).fit(negative_reviews)
                trends = trend_vectorizer.get_feature_names_out().tolist()
            except ValueError:
                trends = [] 
        
        total_reviews = len(df_user)
        sentiment_counts = df_user['predicted_sentiment'].value_counts()
        positive_count = int(sentiment_counts.get('Positive', 0))
        negative_count = int(sentiment_counts.get('Negative', 0))
        neutral_count = int(sentiment_counts.get('Neutral', 0))

        monthly_profit = df_user.set_index('purchase_date').resample('ME')['net_profit'].sum()
        total_profit = df_user['net_profit'].sum()

        region_sales = df_user.groupby('region')['units_sold'].sum()
        top_region = region_sales.idxmax() if not region_sales.empty else "N/A"
        top_region_sales = int(region_sales.max()) if not region_sales.empty else 0

        category_profit = df_user.groupby('product_category')['net_profit'].sum()
        top_category = category_profit.idxmax() if not category_profit.empty else "N/A"
        top_category_profit = category_profit.max() if not category_profit.empty else 0

        avg_rating = df_user['rating'].mean()

        profit_forecast_chart, next_month_profit = generate_forecast_chart(df_user, 'net_profit', 'Overall Profit Forecast', 'Net Profit ($)')
        sales_forecast_chart, next_month_sales = generate_forecast_chart(df_user, 'units_sold', 'Overall Sales Volume Forecast', 'Units Sold')
        top_cat_forecast_chart, top_cat_name, top_cat_next_sales = generate_top_category_sales_forecast(df_user)
        sentiment_trend_chart, next_month_sentiment = generate_sentiment_trend_chart(df_user)
        
        summary_stats = {
            'sentiment': {'positive_pct': f"{(positive_count / total_reviews) * 100:.1f}%" if total_reviews > 0 else "0.0%", 'negative_pct': f"{(negative_count / total_reviews) * 100:.1f}%" if total_reviews > 0 else "0.0%", 'neutral_pct': f"{(neutral_count / total_reviews) * 100:.1f}%" if total_reviews > 0 else "0.0%", 'positive_count': positive_count, 'negative_count': negative_count, 'neutral_count': neutral_count, 'total_reviews': total_reviews },
            'profit': {'total_profit': f"${total_profit:,.2f}", 'avg_monthly_profit': f"${monthly_profit.mean():,.2f}" if not monthly_profit.empty else "$0.00" },
            'region': {'top_region': top_region, 'top_region_sales': f"{top_region_sales:,}" },
            'category': {'top_category': top_category, 'top_category_profit': f"${top_category_profit:,.2f}" },
            'rating': {'avg_rating': f"{avg_rating:.2f} out of 5" if not pd.isna(avg_rating) else "N/A" },
            'trends': {'negative_reviews_count': negative_count },
            'predictions': { 
                'next_month_profit': f"${next_month_profit:,.2f}", 
                'next_month_sales': f"{int(next_month_sales):,}", 
                'next_month_sentiment': f"{next_month_sentiment:.1f}% Positive",
                'top_category_name': top_cat_name,
                'next_month_top_category_sales': f"{int(top_cat_next_sales):,} units",
                'next_month_top_product': predict_top_selling_product(df_user)
            }
        }
        
        charts = {
            'sentiment_pie': generate_sentiment_pie_chart(df_user),
            'profit_time': generate_profit_over_time_chart(df_user),
            'sales_region': generate_sales_by_region_chart(df_user),
            'rating_sentiment': generate_rating_sentiment_heatmap(df_user),
            'category_performance': generate_category_performance_chart(df_user),
            'profit_forecast': profit_forecast_chart,
            'sales_forecast': sales_forecast_chart,
            'sentiment_trend': sentiment_trend_chart,
            'top_category_sales_forecast': top_cat_forecast_chart
        }

        recommendations = generate_recommendations(df_user)
        
        return jsonify({'charts': charts, 'trends': trends, 'summary_stats': summary_stats, 'recommendations': recommendations})

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({'error': 'An error occurred while processing the file.'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    train_model()
    app.run(debug=True)
