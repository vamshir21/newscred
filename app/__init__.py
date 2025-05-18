from flask import Flask
import joblib
import os
import nltk
import pandas as pd # Ensure pandas is imported
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_very_secret_and_random_string_for_flash_123!'

PROJECT_BASE_DIR_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.config['DATABASE_PATH'] = os.path.join(PROJECT_BASE_DIR_APP, 'data', 'newscred_feedback.db')

PROJECT_NLTK_DATA_PATH_APP = os.path.join(PROJECT_BASE_DIR_APP, 'my_nltk_data')
if not os.path.exists(PROJECT_NLTK_DATA_PATH_APP):
    try:
        os.makedirs(PROJECT_NLTK_DATA_PATH_APP)
        print(f"App __init__ created NLTK data directory: {PROJECT_NLTK_DATA_PATH_APP}")
    except Exception as e_mkdir: print(f"App __init__ failed to create NLTK data directory: {e_mkdir}")
nltk.data.path = [PROJECT_NLTK_DATA_PATH_APP]
print(f"App __init__: NLTK data path exclusively set to: {nltk.data.path}")

MODEL_PATH = os.path.join(PROJECT_BASE_DIR_APP, 'models', 'fake_news_classifier.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_BASE_DIR_APP, 'models', 'tfidf_vectorizer.pkl')
try:
    print(f"App __init__: Loading model from: {MODEL_PATH}")
    app.model = joblib.load(MODEL_PATH)
    print("App __init__: Model loaded successfully.")
    print(f"App __init__: Loading vectorizer from: {VECTORIZER_PATH}")
    app.vectorizer = joblib.load(VECTORIZER_PATH)
    print("App __init__: Vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"App __init__ ERROR: Model or Vectorizer file not found. {e}")
    app.model = None; app.vectorizer = None
except Exception as e:
    print(f"App __init__: Error loading model/vectorizer: {e}"); app.model = None; app.vectorizer = None

REPUTATION_DATA_PATH = os.path.join(PROJECT_BASE_DIR_APP, 'data', 'source_reputation.csv')
app.source_reputation_df = None
try:
    print(f"App __init__: Loading source reputation data from: {REPUTATION_DATA_PATH}")
    app.source_reputation_df = pd.read_csv(REPUTATION_DATA_PATH)
    if 'domain' in app.source_reputation_df.columns:
        app.source_reputation_df['domain'] = app.source_reputation_df['domain'].str.lower().str.strip()
    print(f"App __init__: Source reputation data loaded ({len(app.source_reputation_df) if app.source_reputation_df is not None else 0} entries).")
except FileNotFoundError: print(f"App __init__ WARNING: Source reputation file not found at {REPUTATION_DATA_PATH}.")
except pd.errors.EmptyDataError: print(f"App __init__ WARNING: Source reputation file at {REPUTATION_DATA_PATH} is empty.")
except Exception as e: print(f"App __init__ ERROR loading source reputation data: {e}")

def init_db():
    data_dir = os.path.join(PROJECT_BASE_DIR_APP, 'data')
    if not os.path.exists(data_dir):
        try: os.makedirs(data_dir); print(f"App __init__: Created data directory for database: {data_dir}")
        except Exception as e_mkdir_data: print(f"App __init__: ERROR creating data directory for database: {e_mkdir_data}"); return
    try:
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_type TEXT,
                analyzed_input TEXT,
                predicted_label_internal INTEGER,
                user_corrected_label_internal INTEGER,
                feedback_type TEXT,
                user_comment TEXT DEFAULT NULL
            )
        ''')
        conn.commit(); conn.close()
        print(f"App __init__: Feedback database initialized/checked at {app.config['DATABASE_PATH']}")
    except Exception as e_db: print(f"App __init__: ERROR initializing feedback database: {e_db}")
init_db()

from app import routes