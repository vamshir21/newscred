import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- FORCE NLTK DATA PATH ---
PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_NLTK_DATA_PATH = os.path.join(PROJECT_BASE_DIR, 'my_nltk_data')

if not os.path.exists(PROJECT_NLTK_DATA_PATH):
    try:
        os.makedirs(PROJECT_NLTK_DATA_PATH)
        print(f"Created dedicated NLTK data directory: {PROJECT_NLTK_DATA_PATH}")
    except Exception as e:
        print(f"Error creating NLTK data directory {PROJECT_NLTK_DATA_PATH}: {e}")
        exit()
nltk.data.path = [PROJECT_NLTK_DATA_PATH]
# print(f"NLTK data path exclusively set to: {nltk.data.path}") # Can be verbose, uncomment if debugging
# --- END FORCE NLTK DATA PATH ---

# --- Check/Download NLTK resources ---
nltk_data_downloaded = {"stopwords": False, "punkt": False, "vader_lexicon": False}
# print("Checking NLTK resources (within project-specific path)...") # Can be verbose

# Stopwords Check
try:
    stopwords.words('english')
    nltk_data_downloaded["stopwords"] = True
    # print("- 'stopwords' resource found.")
except LookupError:
    print("NLTK resource 'stopwords' not found. Downloading to project path...")
    try:
        nltk.download('stopwords', download_dir=PROJECT_NLTK_DATA_PATH, quiet=True)
        nltk_data_downloaded["stopwords"] = True
        print("'stopwords' downloaded to project path.")
    except Exception as e:
        print(f"ERROR downloading 'stopwords': {e}"); exit()

# Punkt Check
try:
    nltk.word_tokenize("Test punkt functionality.")
    nltk_data_downloaded["punkt"] = True
    # print("- 'punkt' resource is functional.")
except LookupError:
    print("Primary 'punkt' tokenization check failed. Attempting download...")
    try:
        nltk.download('punkt', download_dir=PROJECT_NLTK_DATA_PATH, quiet=True)
        nltk.word_tokenize("Test punkt functionality after download.")
        nltk_data_downloaded["punkt"] = True
        # print("- 'punkt' resource is now functional after download.")
    except Exception as e:
        print(f"CRITICAL ERROR: 'punkt' tokenization failed even after download attempt: {e}"); exit()

# VADER Lexicon Check
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk_data_downloaded["vader_lexicon"] = True
    # print("- 'vader_lexicon' resource found.")
except LookupError:
    print("NLTK resource 'vader_lexicon' not found. Downloading to project path...")
    try:
        nltk.download('vader_lexicon', download_dir=PROJECT_NLTK_DATA_PATH, quiet=True)
        nltk_data_downloaded["vader_lexicon"] = True
        print("'vader_lexicon' downloaded to project path.")
    except Exception as e:
        print(f"ERROR downloading 'vader_lexicon': {e}")
        print("Warning: vader_lexicon download failed, sentiment analysis in app might not work.")

# print(f"NLTK resources status: {nltk_data_downloaded}\n") # Can be verbose

# --- Define paths ---
DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data')
FAKE_NEWS_PATH = os.path.join(DATA_DIR, 'Fake.csv')
TRUE_NEWS_PATH = os.path.join(DATA_DIR, 'True.csv')
MODELS_DIR = os.path.join(PROJECT_BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'fake_news_classifier.pkl')
VECTORIZER_SAVE_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

# --- Text Preprocessing Function ---
stop_words_set = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower(); text = re.sub(r'\W', ' ', text); text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text); tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_set and len(word) > 2]
    return " ".join(tokens)

# --- Main Training Logic ---
if __name__ == '__main__':
    print("Starting model training process...")
    # ... (rest of the training logic is the same as the last working version) ...
    print(f"Loading datasets from: {DATA_DIR}")
    try:
        df_fake = pd.read_csv(FAKE_NEWS_PATH)
        df_true = pd.read_csv(TRUE_NEWS_PATH)
        print("Datasets loaded successfully.")
    except FileNotFoundError as e: print(f"Error loading dataset: {e}\nPlease ensure paths are correct."); exit()
    df_fake['label'] = 1; df_true['label'] = 0
    df_fake['content'] = df_fake['title'] + " " + df_fake['text']
    df_true['content'] = df_true['title'] + " " + df_true['text']
    df_combined = pd.concat([df_fake[['content', 'label']], df_true[['content', 'label']]], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Combined dataset shape: {df_combined.shape}\nValue counts for 'label':\n{df_combined['label'].value_counts()}")
    df_combined.dropna(subset=['content'], inplace=True)
    if df_combined.empty: print("Error: No data after dropping NaNs."); exit()
    print("Preprocessing text data...")
    df_combined['processed_content'] = df_combined['content'].apply(preprocess_text)
    X = df_combined['processed_content']; y = df_combined['label']
    if X.empty: print("Error: No features (X) after preprocessing."); exit()
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Performing TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train); X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("Evaluating model...")
    y_pred_train = model.predict(X_train_tfidf); y_pred_test = model.predict(X_test_tfidf)
    print("\n--- Training Set Evaluation ---\nAccuracy: {:.4f}\n{}".format(accuracy_score(y_train, y_pred_train), classification_report(y_train, y_pred_train)))
    print("\n--- Test Set Evaluation ---\nAccuracy: {:.4f}\n{}".format(accuracy_score(y_test, y_pred_test), classification_report(y_test, y_pred_test)))
    print(f"\nSaving model to {MODEL_SAVE_PATH}"); joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Saving TF-IDF vectorizer to {VECTORIZER_SAVE_PATH}"); joblib.dump(tfidf_vectorizer, VECTORIZER_SAVE_PATH)
    print("\nModel training and saving complete!")