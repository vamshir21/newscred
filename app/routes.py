from flask import render_template, request, current_app, jsonify, redirect, url_for, flash
from app import app # Import the 'app' instance
import nltk
import re
from nltk.corpus import stopwords
import requests
from newspaper import Article, ArticleException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer as SummarizerLSA 
from sumy.nlp.stemmers import Stemmer as SumyStemmer 
from sumy.utils import get_stop_words as sumy_get_stop_words 
from google.cloud import vision
import json 
import os 
import traceback 
from urllib.parse import urlparse
import sqlite3 
import datetime
import pandas as pd

# --- Configuration for External APIs ---
GOOGLE_SEARCH_API_KEY = "AIzaSyD1tomgKaKXtxTdZ9sWUnrIp_c3_XB0cWw"  # !!! REPLACE THIS !!!
CUSTOM_SEARCH_ENGINE_ID = "7133080eca6d546e3"    # !!! REPLACE THIS !!!

# --- Attempt to Load spaCy Model (module level) ---
nlp_spacy_global = None # Use a distinct name to ensure we are referring to the global one
try:
    import spacy 
    print("routes.py: Loading spaCy model (en_core_web_sm)...")
    nlp_spacy_global = spacy.load("en_core_web_sm")
    print("routes.py: spaCy model loaded successfully.")
except ImportError:
    print("routes.py WARNING: spaCy library not installed. Enhanced query generation will be limited. Run: pip install spacy")
except OSError:
    print("routes.py WARNING: spaCy 'en_core_web_sm' model not found. Run: python -m spacy download en_core_web_sm")
    print("Enhanced query generation for web corroboration will be limited.")
except Exception as e_spacy_load:
    print(f"routes.py WARNING: Error loading spaCy model: {e_spacy_load}. Enhanced query generation will be limited.")
    # nlp_spacy_global remains None, which is its default

# --- NLTK Resource Checks ---
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError: nltk.download('stopwords', quiet=True); stop_words_set = set(stopwords.words('english'))
try: nltk.word_tokenize("test")
except LookupError: nltk.download('punkt', quiet=True)
try: nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: nltk.download('vader_lexicon', quiet=True)

# --- Initialize Sentiment Analyzer ---
analyzer = None
try:
    analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Could not initialize SentimentIntensityAnalyzer: {e}. Sentiment analysis will be disabled.")

# --- Helper Functions ---
def preprocess_text_for_prediction(text):
    if not isinstance(text, str): return ""
    text = text.lower(); text = re.sub(r'\W', ' ', text); text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text); tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_set and len(word) > 2]
    return " ".join(tokens)

def get_article_data_from_url(url): # Now returns title as well
    text, title, top_image, all_images, error_msg = None, None, None, [], None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        article_parser = Article(url, request_timeout=20, browser_user_agent=headers['User-Agent'])
        article_parser.download(); article_parser.parse()
        text = article_parser.text
        title = article_parser.title 
        top_image = article_parser.top_image if article_parser.top_image else None
        all_images = list(article_parser.images)
        if not text and article_parser.html: print(f"Warning: Newspaper3k extracted no text for {url}, HTML length: {len(article_parser.html)}")
    except ArticleException as e: error_msg = f"Could not process article (ArticleException): {e}"
    except requests.exceptions.RequestException as e: error_msg = f"Failed to fetch URL content: {e}"
    except Exception as e: error_msg = f"Unexpected error processing URL: {e}"; print(f"Generic URL processing error for {url}: {e}"); traceback.print_exc()
    return text, title, top_image, all_images, error_msg

LANGUAGE_SUMY = "english"
def summarize_text(text, sentences_count=3):
    if not text or not text.strip() or len(text.split()) < 20: 
        return "Text too short or simple to generate a meaningful summary."
    try:
        parser = PlaintextParser.from_string(text, SumyTokenizer(LANGUAGE_SUMY)); stemmer = SumyStemmer(LANGUAGE_SUMY)
        summarizer = SummarizerLSA(stemmer); summarizer.stop_words = sumy_get_stop_words(LANGUAGE_SUMY)
        summary = " ".join([str(s) for s in summarizer(parser.document, sentences_count)])
        return summary if summary.strip() else "Could not generate a meaningful summary."
    except Exception as e: print(f"Error during summarization: {e}"); traceback.print_exc(); return "Error generating summary."

FACT_CHECKING_SITES_KEYWORDS = ["snopes.com", "politifact.com", "factcheck.org", "reuters.com/fact-check", "apnews.com/apf-factcheck", "checkyourfact.com", "boomlive.in", "vishvasnews.com", "altnews.in"]
REPUTABLE_NEWS_KEYWORDS = ["reuters.com", "apnews.com", "bbc.com", "nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com", "npr.org", "pbs.org"]
DEBUNKING_KEYWORDS = ["hoax", "debunked", "false", "scam", "misleading", "not true", "conspiracy", "unsubstantiated", "rumor"]

def get_enhanced_web_corroboration(article_title, article_text_content, num_results=5):
    corroboration_data = {"query_used": "N/A", "fact_checks": [], "reputable_sources": [], "general_results": [], "debunking_flags": [], "error": None}
    
    # Use the global nlp_spacy_global which was loaded at module level
    global nlp_spacy_global 

    if GOOGLE_SEARCH_API_KEY == "YOUR_GOOGLE_API_KEY" or CUSTOM_SEARCH_ENGINE_ID == "YOUR_CSE_ID":
        corroboration_data["error"] = "Web corroboration feature not configured by administrator."; print(corroboration_data["error"]); return corroboration_data
    
    search_query = ""
    if article_title and len(article_title.strip()) > 10: search_query = article_title.strip()
    elif nlp_spacy_global and article_text_content: # Check if spaCy model is loaded
        try:
            doc = nlp_spacy_global(article_text_content[:1000]) 
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"]]
            if entities: search_query = " ".join(list(set(entities))[:5]) 
        except Exception as e_spacy_ner:
            print(f"spaCy NER error during query generation: {e_spacy_ner}")
            # Do not modify nlp_spacy_global here; its state is from module load
    if not search_query and article_text_content: search_query = " ".join(preprocess_text_for_prediction(article_text_content).split()[:7])
    if not search_query: corroboration_data["error"] = "Not enough info for search query."; return corroboration_data
    
    corroboration_data["query_used"] = search_query
    print(f"Enhanced Web Corroboration Query: {search_query}")
    try:
        api_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}&q={search_query}&num={num_results}"
        response = requests.get(api_url, timeout=15); response.raise_for_status()
        search_results_json = response.json()
        if "items" in search_results_json:
            for item in search_results_json["items"]:
                result_item = {"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")}
                item_text_lower = (str(result_item["title"] or "") + " " + str(result_item["snippet"] or "")).lower()
                for dk_word in DEBUNKING_KEYWORDS:
                    if dk_word in item_text_lower: corroboration_data["debunking_flags"].append(f"'{result_item['title']}' (snippet/title) contains '{dk_word}'.")
                is_fact_check = any(fc_site in (result_item["link"] or "").lower() for fc_site in FACT_CHECKING_SITES_KEYWORDS)
                is_reputable = any(rep_site in (result_item["link"] or "").lower() for rep_site in REPUTABLE_NEWS_KEYWORDS)
                if is_fact_check: corroboration_data["fact_checks"].append(result_item)
                elif is_reputable: corroboration_data["reputable_sources"].append(result_item)
                else: corroboration_data["general_results"].append(result_item)
        else: corroboration_data["error"] = "No search results returned by API." if not corroboration_data.get("error") else corroboration_data.get("error")
    except requests.exceptions.HTTPError as http_err:
        error_detail = f"HTTP error in web corroboration: {http_err}. Response: {http_err.response.content.decode(errors='ignore') if http_err.response else 'No response content'}"
        print(error_detail); corroboration_data["error"] = "Search API HTTP error. Check API Key, CSE ID, and quotas."
    except Exception as e: error_detail = f"Unexpected error during web corroboration: {e}"; print(error_detail); traceback.print_exc(); corroboration_data["error"] = "Unexpected error during web corroboration."
    return corroboration_data

def get_reverse_image_analysis(image_url):
    analysis = {"original_url": image_url, "web_entities": [], "full_matching_images": [], "pages_with_matching_images": [], "error": None}
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        analysis["error"] = "Image search not configured (GOOGLE_APPLICATION_CREDENTIALS not set)."; print(analysis["error"]); return analysis
    try:
        client = vision.ImageAnnotatorClient(); image = vision.Image(); image.source.image_uri = image_url
        response = client.web_detection(image=image, max_results=3)
        if response.error.message: raise Exception(f"Vision API Error: {response.error.message}")
        annotations = response.web_detection
        if annotations.web_entities: analysis["web_entities"] = [f"{e.description} (Score: {e.score:.2f})" for e in annotations.web_entities]
        if annotations.full_matching_images: analysis["full_matching_images"] = [img.url for img in annotations.full_matching_images]
        if annotations.pages_with_matching_images: analysis["pages_with_matching_images"] = [{"url": p.url, "title": p.page_title if p.page_title else "N/A"} for p in annotations.pages_with_matching_images]
        if not any([analysis["web_entities"], analysis["full_matching_images"], analysis["pages_with_matching_images"]]):
             analysis["web_entities"].append("No specific web intelligence found by Vision API.")
    except Exception as e: print(f"Error in reverse image search for {image_url}: {e}"); traceback.print_exc(); analysis["error"] = f"API error during image analysis: {str(e)[:150]}..."
    return analysis

def extract_domain(url_string):
    try:
        if not url_string: return None
        if not url_string.startswith(('http://', 'https://')): url_string = 'http://' + url_string
        parsed_uri = urlparse(url_string); domain = '{uri.netloc}'.format(uri=parsed_uri).lower()
        return domain[4:] if domain.startswith('www.') else domain
    except Exception as e: print(f"Error parsing domain from URL {url_string}: {e}"); return None

def get_source_credibility(url_domain):
    source_info = {"domain": url_domain, "score_num": None, "score_str": "-", "bias": "N/A", "factual_reporting": "N/A", "notes": "Source not in our database.", "rating":"Unknown"}
    if not url_domain or current_app.source_reputation_df is None or current_app.source_reputation_df.empty: return source_info
    match = current_app.source_reputation_df[current_app.source_reputation_df['domain'] == url_domain]
    if not match.empty:
        source_info["score_num"] = int(match.iloc[0].get('credibility_score', 50)) if pd.notna(match.iloc[0].get('credibility_score')) else 50
        source_info["bias"] = match.iloc[0].get('mbfc_bias', 'N/A')
        source_info["factual_reporting"] = match.iloc[0].get('mbfc_factual', 'N/A')
        source_info["notes"] = match.iloc[0].get('notes', 'Details from database.')
    else:
        parts = url_domain.split('.');
        if len(parts) > 2:
            base_domain = '.'.join(parts[-2:])
            match = current_app.source_reputation_df[current_app.source_reputation_df['domain'] == base_domain]
            if not match.empty:
                source_info["score_num"] = int(match.iloc[0].get('credibility_score', 50)) if pd.notna(match.iloc[0].get('credibility_score')) else 50
                source_info["bias"] = match.iloc[0].get('mbfc_bias', 'N/A')
                source_info["factual_reporting"] = match.iloc[0].get('mbfc_factual', 'N/A')
                source_info["notes"] = f"Info based on base domain: {base_domain}. " + (match.iloc[0].get('notes', '') or "")
                source_info["domain"] = f"{url_domain} (data for {base_domain})"
    if source_info["score_num"] is not None:
        source_info["score_str"] = str(source_info["score_num"])
        if source_info["score_num"] >= 85: source_info["rating"] = "Very High Credibility"
        elif source_info["score_num"] >= 70: source_info["rating"] = "High Credibility"
        elif source_info["score_num"] >= 50: source_info["rating"] = "Mixed Credibility"
        elif source_info["score_num"] >= 25: source_info["rating"] = "Low Credibility"
        else: source_info["rating"] = "Very Low Credibility / Questionable"
    return source_info

@app.route('/', methods=['GET', 'POST'])
def index():
    view_data = {
        "prediction_text": None, "prediction_label": None, "confidence_score": None,
        "sentiment_label": None, "sentiment_score": None, "article_summary": None,
        "corroboration_data": {}, "top_image_url": None, "all_image_urls": [],
        "image_analysis_results": [], "source_credibility": None, "error_message": None,
        "input_type": request.form.get('input_type', 'text'),
        "article_source_url": None, "analysis_done": False,
        "news_text_value": request.form.get('news_text', ''),
        "news_url_value": request.form.get('news_url', ''),
        "article_title_value": None
    }
    news_text_to_analyze = None; article_title_for_corroboration = None

    if request.method == 'POST':
        view_data["analysis_done"] = True
        if view_data["input_type"] == 'url':
            news_url = request.form.get('news_url', '').strip()
            view_data["article_source_url"] = news_url
            if not news_url: view_data["error_message"] = "Please enter a URL."
            else:
                text_from_url, title_from_url, top_img, all_imgs, scrape_err = get_article_data_from_url(news_url)
                if scrape_err: view_data["error_message"] = scrape_err
                elif not text_from_url or not text_from_url.strip(): view_data["error_message"] = "Could not extract significant text from URL."
                else:
                    news_text_to_analyze = text_from_url
                    article_title_for_corroboration = title_from_url
                    view_data["article_title_value"] = title_from_url
                    view_data["top_image_url"] = top_img
                    view_data["all_image_urls"] = all_imgs
                url_domain = extract_domain(news_url)
                if url_domain: view_data["source_credibility"] = get_source_credibility(url_domain)
                else: view_data["source_credibility"] = {"rating": "Invalid URL", "score_str":"-", "notes": "Could not parse domain."}
        else: 
            news_text_to_analyze = request.form.get('news_text', '').strip()
            view_data["article_source_url"] = "Pasted Text"
            article_title_for_corroboration = news_text_to_analyze[:100] # Use first 100 chars as pseudo-title
            if not news_text_to_analyze: view_data["error_message"] = "Please enter some news text."
            view_data["source_credibility"] = {"rating": "N/A for pasted text", "score_str":"-", "notes": "Source credibility applicable for URLs only."}

        if not view_data["error_message"] and news_text_to_analyze:
            print(f"Analyzing text (length: {len(news_text_to_analyze)} chars)...")
            view_data["article_summary"] = summarize_text(news_text_to_analyze)
            if analyzer:
                try:
                    vs=analyzer.polarity_scores(news_text_to_analyze); compound=vs['compound']
                    if compound >= 0.05: view_data["sentiment_label"]="Positive"
                    elif compound <= -0.05: view_data["sentiment_label"]="Negative"
                    else: view_data["sentiment_label"]="Neutral"
                    view_data["sentiment_score"]=f"{compound:.2f} (Compound)"
                except Exception as e_sent: print(f"Sentiment error: {e_sent}"); view_data["sentiment_label"]="Error"
            else: view_data["sentiment_label"]="Analyzer N/A"
            
            view_data["corroboration_data"] = get_enhanced_web_corroboration(article_title_for_corroboration, news_text_to_analyze)
            
            if view_data["top_image_url"]:
                view_data["image_analysis_results"].append(get_reverse_image_analysis(view_data["top_image_url"]))
            
            if not current_app.model or not current_app.vectorizer:
                view_data["error_message"] = "Core classification model not loaded."
            else:
                try:
                    processed_text = preprocess_text_for_prediction(news_text_to_analyze)
                    if not processed_text or not processed_text.strip():
                        view_data["prediction_text"]="Text empty after preprocessing."; view_data["prediction_label"]=-1
                    else:
                        vectorized=current_app.vectorizer.transform([processed_text])
                        pred=current_app.model.predict(vectorized); proba=current_app.model.predict_proba(vectorized)
                        view_data["prediction_label"]=int(pred[0])
                        view_data["prediction_text"] = f"Prediction: {'REAL' if pred[0]==0 else 'FAKE'} News"
                        view_data["confidence_score"] = f"{proba[0][pred[0]]*100:.2f}"
                except Exception as e_pred: print(f"Pred error: {e_pred}"); traceback.print_exc(); view_data["error_message"]="Err during classification."
        
        if view_data["analysis_done"] and not view_data["error_message"] and not view_data["prediction_text"] and not news_text_to_analyze :
            if view_data["input_type"] == 'url' and (not view_data["news_url_value"] or not view_data["news_url_value"].strip()): view_data["error_message"] = "URL was provided but seems empty or invalid."
            elif view_data["input_type"] == 'text' and not view_data["news_text_value"].strip(): view_data["error_message"] = "Please enter news text."
    return render_template('index.html', **view_data)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        original_input_type = request.form.get('original_input_type')
        # original_news_text = request.form.get('original_news_text') # Not strictly needed for redirect
        # original_news_url = request.form.get('original_news_url')   # Not strictly needed for redirect
        analyzed_input = request.form.get('analyzed_input_value')
        predicted_label_internal_str = request.form.get('predicted_label_internal')
        user_corrected_label_str = request.form.get('user_corrected_label')
        feedback_type = request.form.get('feedback_type')
        print(f"\n--- FEEDBACK RECEIVED ---"); print(f"Input: {analyzed_input[:100]}..."); print(f"Pred Label: {predicted_label_internal_str}, Feedback: {feedback_type}, Corrected: {user_corrected_label_str}")
        feedback_message = "Thank you for your feedback!"; feedback_category = "info"
        try:
            db_path = current_app.config['DATABASE_PATH']
            conn = sqlite3.connect(db_path); cursor = conn.cursor()
            predicted_label_to_save = None
            if predicted_label_internal_str and predicted_label_internal_str != 'None' and predicted_label_internal_str.lstrip('-').isdigit(): # Check if it's a digit, possibly negative
                predicted_label_to_save = int(predicted_label_internal_str)

            user_corrected_label_to_save = None; corrected_text_display = ""
            if user_corrected_label_str and user_corrected_label_str.isdigit():
                user_corrected_label_to_save = int(user_corrected_label_str)
                corrected_text_display = 'REAL' if user_corrected_label_to_save == 0 else 'FAKE'
            
            cursor.execute('''INSERT INTO feedback (input_type, analyzed_input, predicted_label_internal, user_corrected_label_internal, feedback_type)
                              VALUES (?, ?, ?, ?, ?)''', 
                           (original_input_type, analyzed_input, predicted_label_to_save, user_corrected_label_to_save, feedback_type))
            conn.commit(); conn.close()
            if feedback_type == 'correction' and user_corrected_label_to_save is not None:
                feedback_message = f"Thank you! Your correction to '{corrected_text_display}' has been recorded."; feedback_category = "success"
            elif feedback_type == 'accurate':
                feedback_message = "Thank you for confirming the prediction's accuracy!"; feedback_category = "success"
            else: feedback_message = "Feedback noted. If correcting, please select a label."; feedback_category = "warning"
        except Exception as e_db_save: print(f"ERROR saving feedback: {e_db_save}"); traceback.print_exc(); feedback_message = "Could not save feedback."; feedback_category = "danger"
        flash(feedback_message, feedback_category)
    return redirect(url_for('index')) # Corrected endpoint name