from flask import render_template, request, current_app, jsonify, redirect, url_for, flash
from app import app 
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
from bs4 import BeautifulSoup # For URL content type check

# --- Configuration for External APIs ---
GOOGLE_SEARCH_API_KEY = "AIzaSyD1tomgKaKXtxTdZ9sWUnrIp_c3_XB0cWw"
CUSTOM_SEARCH_ENGINE_ID = "7133080eca6d546e3"

# --- Load spaCy Model ---
# ... (after imports and API key configurations)

# --- Constants for Web Corroboration ---
FACT_CHECKING_SITES_KEYWORDS = [
    "snopes.com", "politifact.com", "factcheck.org", 
    "reuters.com/fact-check", "apnews.com/apf-factcheck", "checkyourfact.com", 
    "boomlive.in", "vishvasnews.com", "altnews.in", "fullfact.org" # Added one more
]
REPUTABLE_NEWS_KEYWORDS = [
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com", 
    "washingtonpost.com", "wsj.com", "theguardian.com", "npr.org", 
    "pbs.org", "cnn.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com" # Added a few more common ones
]
DEBUNKING_KEYWORDS = [
    "hoax", "debunked", "false", "scam", "misleading", "not true", 
    "conspiracy", "unsubstantiated", "rumor", "myth", "disinformation", "misinformation" # Added more
]
# --- End Constants for Web Corroboration ---

# --- Load spaCy Model (module level) ---
nlp_spacy_global = None 
# ... (rest of spaCy loading logic) ...

# ... (NLTK checks, Sentiment Analyzer init) ...

# --- Helper Functions ---
# ... (preprocess_text_for_prediction, etc.) ...

def get_enhanced_web_corroboration(article_title, article_text_content, num_results=5):
    # This function will now be able to see the lists defined above
    corroboration_data = {"query_used": "N/A", "fact_checks": []}#... rest of the function
nlp_spacy_global = None 
try:
    import spacy 
    print("routes.py: Loading spaCy model (en_core_web_sm)...")
    nlp_spacy_global = spacy.load("en_core_web_sm")
    print("routes.py: spaCy model loaded successfully.")
except ImportError: print("routes.py WARNING: spaCy library not installed. Run: pip install spacy")
except OSError: print("routes.py WARNING: spaCy model not found. Run: python -m spacy download en_core_web_sm")
except Exception as e_spacy_load: print(f"routes.py WARNING: Error loading spaCy: {e_spacy_load}")

# --- NLTK Resource Checks ---
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError: nltk.download('stopwords', quiet=True); stop_words_set = set(stopwords.words('english'))
try: nltk.word_tokenize("test")
except LookupError: nltk.download('punkt', quiet=True)
try: nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: nltk.download('vader_lexicon', quiet=True)

analyzer = None
try: analyzer = SentimentIntensityAnalyzer()
except Exception as e: print(f"Could not initialize SentimentIntensityAnalyzer: {e}.")

# --- Helper Functions ---
def preprocess_text_for_prediction(text):
    if not isinstance(text, str): return ""
    text = text.lower(); text = re.sub(r'\W', ' ', text); text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text); tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_set and len(word) > 2]
    return " ".join(tokens)

def get_article_data_from_url(url): 
    text, title, top_image, all_images, error_msg = None, None, None, [], None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        # Initial quick check for obviously non-news domains or paths
        parsed_url = urlparse(url)
        domain = extract_domain(url)
        non_news_domains = ["amazon.com", "ebay.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "github.com", "stackoverflow.com"]
        if domain and any(nd in domain for nd in non_news_domains):
            return None, None, None, [], f"The URL ({domain}) does not appear to be a news publication site."
        path_lower = parsed_url.path.lower()
        non_article_paths = ["/product/", "/dp/", "/gp/", "/store/", "/search", "/cart", "/account", "/profile", "/category/", "/tag/", "/user/"]
        if any(nap in path_lower for nap in non_article_paths) and not any(news_path in path_lower for news_path in ["/news/", "/article/"]): # Allow if it has news/article
             return None, None, None, [], f"The URL path suggests it might not be a direct news article."

        # Fetch HTML content for meta tag check
        html_content = ""
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            html_content = response.text
        except requests.exceptions.RequestException as req_e:
            return None, None, None, [], f"Failed to fetch URL content for metadata check: {req_e}"

        is_likely_article_page = False
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            og_type_tag = soup.find('meta', property='og:type')
            if og_type_tag and og_type_tag.get('content', '').lower() == 'article':
                is_likely_article_page = True
            # Add more checks here if needed, e.g., schema.org NewsArticle type

        if not is_likely_article_page:
            # Soft warning, newspaper3k might still work or might fail
            print(f"Warning: URL {url} does not have strong 'article' metadata. Proceeding with newspaper3k parse...")
            # Could return an error here if we want to be stricter:
            # return None, None, None, [], "The page metadata does not strongly indicate it's a news article."

        article_parser = Article(url, request_timeout=15, browser_user_agent=headers['User-Agent'])
        article_parser.set_html(html_content) # Use already fetched HTML
        # article_parser.download() # Not needed if HTML is already set
        article_parser.parse()
        text = article_parser.text; title = article_parser.title 
        top_image = article_parser.top_image if article_parser.top_image else None
        all_images = list(set(article_parser.images))
        
        if not text: 
            error_msg = "Newspaper3k could not extract main text content from the URL."
            if not is_likely_article_page: # Strengthen error if metadata also didn't indicate article
                error_msg += " The page also lacks strong 'article' metadata."

    except ArticleException as e: error_msg = f"Article processing error: {e}"
    except Exception as e: error_msg = f"Unexpected error processing URL: {e}"; print(f"Generic URL processing error for {url}: {e}"); traceback.print_exc()
    return text, title, top_image, all_images, error_msg

# ... (summarize_text, get_enhanced_web_corroboration, get_reverse_image_analysis, extract_domain, get_source_credibility remain as in previous full code for them)
# (For brevity, assuming those functions are correctly pasted from my prior "full code" response for enhanced corroboration)

LANGUAGE_SUMY = "english"
def summarize_text(text, sentences_count=3):
    if not text or not text.strip() or len(text.split()) < 30: return "Text too short or simple to generate a meaningful summary."
    try:
        parser = PlaintextParser.from_string(text, SumyTokenizer(LANGUAGE_SUMY)); stemmer = SumyStemmer(LANGUAGE_SUMY)
        summarizer = SummarizerLSA(stemmer); summarizer.stop_words = sumy_get_stop_words(LANGUAGE_SUMY)
        summary = " ".join([str(s) for s in summarizer(parser.document, sentences_count)])
        return summary if summary.strip() else "Could not generate a meaningful summary."
    except Exception as e: print(f"Error during summarization: {e}"); traceback.print_exc(); return "Error generating summary."

def get_enhanced_web_corroboration(article_title, article_text_content, num_results=5):
    corroboration_data = {"query_used": "N/A", "fact_checks": [], "reputable_sources": [], "general_results": [], "debunking_flags": [], "error": None}
    global nlp_spacy_global 
    if GOOGLE_SEARCH_API_KEY == "YOUR_GOOGLE_API_KEY" or CUSTOM_SEARCH_ENGINE_ID == "YOUR_CSE_ID":
        corroboration_data["error"] = "Web corroboration feature not configured by administrator."; print(corroboration_data["error"]); return corroboration_data
    search_query = ""
    if article_title and len(article_title.strip()) > 10: search_query = article_title.strip()
    elif nlp_spacy_global and article_text_content: 
        try:
            doc = nlp_spacy_global(article_text_content[:1000]) 
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"]]
            if entities: search_query = " ".join(list(set(entities))[:5]) 
        except Exception as e_spacy_ner: print(f"spaCy NER error: {e_spacy_ner}")
    if not search_query and article_text_content: search_query = " ".join(preprocess_text_for_prediction(article_text_content).split()[:7])
    if not search_query: corroboration_data["error"] = "Not enough info for search query."; return corroboration_data
    corroboration_data["query_used"] = search_query; print(f"Corroboration Query: {search_query}")
    try:
        api_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}&q={search_query}&num={num_results}"
        response = requests.get(api_url, timeout=15); response.raise_for_status()
        search_results_json = response.json()
        if "items" in search_results_json:
            for item in search_results_json["items"]:
                res_item = {"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")}
                item_txt = (str(res_item["title"] or "") + " " + str(res_item["snippet"] or "")).lower()
                for dk in DEBUNKING_KEYWORDS:
                    if dk in item_txt: corroboration_data["debunking_flags"].append(f"'{res_item['title']}' contains '{dk}'.")
                if any(fc in (res_item["link"] or "").lower() for fc in FACT_CHECKING_SITES_KEYWORDS): corroboration_data["fact_checks"].append(res_item)
                elif any(rs in (res_item["link"] or "").lower() for rs in REPUTABLE_NEWS_KEYWORDS): corroboration_data["reputable_sources"].append(res_item)
                else: corroboration_data["general_results"].append(res_item)
        else: corroboration_data["error"] = "No search results from API." if not corroboration_data.get("error") else corroboration_data.get("error")
    except requests.exceptions.HTTPError as http_err: err_detail = f"HTTP error: {http_err}. Resp: {http_err.response.content.decode(errors='ignore') if http_err.response else 'N/A'}"; print(err_detail); corroboration_data["error"] = "Search API HTTP error."
    except Exception as e: err_detail = f"Corroboration error: {e}"; print(err_detail); traceback.print_exc(); corroboration_data["error"] = "Unexpected corroboration error."
    return corroboration_data

def get_reverse_image_analysis(image_url):
    analysis = {"original_url": image_url, "web_entities": [], "full_matching_images": [], "pages_with_matching_images": [], "error": None}
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        analysis["error"] = "Image search config error (Credentials)."; print(analysis["error"]); return analysis
    try:
        client = vision.ImageAnnotatorClient(); image_vision = vision.Image(); image_vision.source.image_uri = image_url # Renamed to avoid conflict
        response = client.web_detection(image=image_vision, max_results=3)
        if response.error.message: raise Exception(f"Vision API Error: {response.error.message}")
        annotations = response.web_detection
        if annotations.web_entities: analysis["web_entities"] = [f"{e.description} (Score: {e.score:.2f})" for e in annotations.web_entities]
        if annotations.full_matching_images: analysis["full_matching_images"] = [img.url for img in annotations.full_matching_images]
        if annotations.pages_with_matching_images: analysis["pages_with_matching_images"] = [{"url": p.url, "title": p.page_title if p.page_title else "N/A"} for p in annotations.pages_with_matching_images]
        if not any([analysis["web_entities"], analysis["full_matching_images"], analysis["pages_with_matching_images"]]):
             analysis["web_entities"].append("No specific web intelligence found by Vision API.")
    except Exception as e: print(f"Reverse image search error for {image_url}: {e}"); traceback.print_exc(); analysis["error"] = f"API error: {str(e)[:150]}..."
    return analysis

def extract_domain(url_string):
    try:
        if not url_string: return None
        if not url_string.startswith(('http://', 'https://')): url_string = 'http://' + url_string
        parsed_uri = urlparse(url_string); domain = '{uri.netloc}'.format(uri=parsed_uri).lower()
        return domain[4:] if domain.startswith('www.') else domain
    except Exception as e: print(f"Error parsing domain: {e}"); return None

def get_source_credibility(url_domain):
    # Initialize with all keys we might want to populate, ensuring defaults
    source_info = {
        "domain": url_domain, 
        "score_num": None,         # Will hold the numeric score
        "score_str": "-",          # For display
        "bias": "N/A", 
        "factual_reporting": "N/A", 
        "notes": "Source not in our database.", 
        "rating": "Unknown"
    }

    if not url_domain or current_app.source_reputation_df is None or current_app.source_reputation_df.empty:
        return source_info # Return defaults if no domain or no reputation data

    df = current_app.source_reputation_df
    
    # Attempt to find a match for the full domain or the base domain
    matched_row = None
    match_direct = df[df['domain'] == url_domain]
    if not match_direct.empty:
        matched_row = match_direct.iloc[0]
        source_info["notes"] = matched_row.get('notes', 'Details from database.') # Prioritize specific notes
    else:
        parts = url_domain.split('.')
        if len(parts) > 2:
            base_domain = '.'.join(parts[-2:])
            match_base = df[df['domain'] == base_domain]
            if not match_base.empty:
                matched_row = match_base.iloc[0]
                source_info["notes"] = f"Info based on base domain: {base_domain}. " + (matched_row.get('notes', '') or "")
                source_info["domain"] = f"{url_domain} (data for {base_domain})" # Clarify which domain data is for
    
    if matched_row is not None:
        # Safely get values from the matched row, providing defaults if columns are missing or NaN
        raw_score = matched_row.get('credibility_score')
        if pd.notna(raw_score): # Check if it's not NaN
            try:
                source_info["score_num"] = int(raw_score)
            except ValueError:
                print(f"Warning: Could not convert credibility_score '{raw_score}' to int for domain {url_domain}.")
                source_info["score_num"] = 50 # Default if conversion fails
        else:
            source_info["score_num"] = 50 # Default if credibility_score is missing/NaN in CSV

        source_info["bias"] = matched_row.get('mbfc_bias', "N/A")
        source_info["factual_reporting"] = matched_row.get('mbfc_factual', "N/A")
        # Notes are already handled above to prioritize direct match notes

    # Assign qualitative rating and string score based on score_num
    if source_info["score_num"] is not None:
        source_info["score_str"] = str(source_info["score_num"])
        score = source_info["score_num"]
        if score >= 85: source_info["rating"] = "Very High Credibility"
        elif score >= 70: source_info["rating"] = "High Credibility"
        elif score >= 50: source_info["rating"] = "Mixed Credibility"
        elif score >= 25: source_info["rating"] = "Low Credibility"
        else: source_info["rating"] = "Very Low Credibility / Questionable"
    else: # Should not happen if we default score_num, but as a fallback
        source_info["rating"] = "Unknown"
        source_info["score_str"] = "-"

    return source_info

def generate_explanation_paragraph(view_data):
    explanation_parts = []
    overall_assessment = "Undetermined" 

    # 1. Core ML prediction
    if view_data.get("prediction_label") is not None and view_data["prediction_label"] != -1:
        pred_text = "REAL" if view_data["prediction_label"] == 0 else "FAKE"
        explanation_parts.append(f"The AI model classifies the text as likely **{pred_text}** (Confidence: {view_data.get('confidence_score', 'N/A')}%).")
        overall_assessment = f"Likely {pred_text}"
    else:
        explanation_parts.append("The AI model could not make a fake/real prediction (e.g., text too short or error).")

    # 2. Sentiment
    if view_data.get("sentiment_label") and view_data["sentiment_label"] not in ["Analyzer N/A", "Error"]:
        explanation_parts.append(f"Sentiment: **{view_data['sentiment_label']}**.")
        if view_data["sentiment_label"] == "Negative" and view_data.get("prediction_label") == 1:
             explanation_parts.append("Highly negative sentiment can sometimes align with manipulative content.")


    # 3. Source Credibility
    sc = view_data.get("source_credibility")
    if sc and sc.get("rating") not in ["Unknown", "N/A for pasted text", "Invalid URL for source check"]:
        explanation_parts.append(f"Source ({sc.get('domain','N/A')}) Credibility: **{sc.get('rating')}** (Score: {sc.get('score_str')}/100).")
        if sc.get("score_num", 100) < 50:
            explanation_parts.append("A lower source credibility score warrants increased skepticism.")
            if "Real" in overall_assessment: overall_assessment += ", but from a Questionable Source"


    # 4. Web Corroboration
    cd = view_data.get("corroboration_data", {})
    if cd.get("error"):
        explanation_parts.append(f"Web corroboration failed: {cd['error']}")
    else:
        corroboration_insights = []
        if cd.get("fact_checks"): corroboration_insights.append(f"found **{len(cd['fact_checks'])} relevant fact-check(s)**")
        if cd.get("reputable_sources"): corroboration_insights.append(f"identified **{len(cd['reputable_sources'])} mention(s) from other reputable sources**")
        if cd.get("debunking_flags"): corroboration_insights.append(f"flagged **{len(cd['debunking_flags'])} potential debunking indicators**")
        if corroboration_insights:
            explanation_parts.append(f"Web search {', '.join(corroboration_insights)}. Check these results for context.")
        elif cd.get("query_used") != "N/A":
             explanation_parts.append("Web search did not yield strong immediate signals from fact-checkers or major reputable sources for the query used.")
    
    # 5. Image Analysis
    ir = view_data.get("image_analysis_results", [])
    if ir and ir[0] and not ir[0].get("error"): # Assuming we check the first image result
        first_img_res = ir[0]
        if first_img_res.get("full_matching_images") or first_img_res.get("pages_with_matching_images"):
            explanation_parts.append("The primary image appears on other web pages, indicating it's not unique. Its original context should be verified.")
        elif first_img_res.get("web_entities") and "No specific web intelligence" not in first_img_res["web_entities"][0]:
             explanation_parts.append(f"The primary image is associated by Google with entities like: {'; '.join(first_img_res['web_entities'][:2])}.")

    # Conclusion
    if not explanation_parts: return "Insufficient data for a detailed explanation."
    
    final_paragraph = " ".join(explanation_parts)
    final_paragraph += f"\n\n**NewsCred+ Overall Assessment Hint:** **{overall_assessment}**. Please review all details and use critical judgment."
    return final_paragraph

# --- Main Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    view_data = { # Initialize all keys
        "prediction_text": None, "prediction_label": None, "confidence_score": None,
        "sentiment_label": None, "sentiment_score": None, "article_summary": None,
        "corroboration_data": {}, "top_image_url": None, "all_image_urls": [],
        "image_analysis_results": [], "source_credibility": None, "error_message": None,
        "input_type": request.form.get('input_type', 'text'),
        "article_source_url": None, "analysis_done": False,
        "news_text_value": request.form.get('news_text', ''),
        "news_url_value": request.form.get('news_url', ''),
        "article_title_value": None, "explanation_paragraph": None
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
                elif not text_from_url or not text_from_url.strip(): view_data["error_message"] = "Could not extract significant text from URL. It might not be a news article or is protected."
                else:
                    news_text_to_analyze = text_from_url; article_title_for_corroboration = title_from_url
                    view_data.update({"article_title_value": title_from_url, "top_image_url": top_img, "all_image_urls": all_imgs})
                url_domain = extract_domain(news_url)
                view_data["source_credibility"] = get_source_credibility(url_domain) if url_domain else {"rating": "Invalid URL", "score_str":"-", "notes": "Could not parse domain."}
        else: 
            news_text_to_analyze = request.form.get('news_text', '').strip()
            view_data["article_source_url"] = "Pasted Text"; article_title_for_corroboration = news_text_to_analyze[:100] 
            if not news_text_to_analyze: view_data["error_message"] = "Please enter some news text."
            view_data["source_credibility"] = {"rating": "N/A for pasted text", "score_str":"-", "notes": "Source credibility for URLs only."}

        if not view_data["error_message"] and news_text_to_analyze:
            print(f"Analyzing (len: {len(news_text_to_analyze)})...")
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
            
            images_to_process_for_api = [];
            if view_data["top_image_url"]: images_to_process_for_api.append(view_data["top_image_url"])
            # Add more images based on filter (simplified loop here)
            if view_data["all_image_urls"]:
                for img_url in view_data["all_image_urls"][:2]: # Limit to first 2 additional for now
                    if img_url not in images_to_process_for_api and not any(k in img_url.lower() for k in ["logo", "icon", "gif", "svg"]):
                        images_to_process_for_api.append(img_url)
            for img_url_to_analyze in list(set(images_to_process_for_api))[:2]: # Analyze max 2 unique images
                 if img_url_to_analyze: view_data["image_analysis_results"].append(get_reverse_image_analysis(img_url_to_analyze))

            if not current_app.model or not current_app.vectorizer: view_data["error_message"] = "Core classification model not loaded."
            else:
                try:
                    processed_text = preprocess_text_for_prediction(news_text_to_analyze)
                    if not processed_text or not processed_text.strip(): view_data["prediction_text"]="Text empty after preprocessing."; view_data["prediction_label"]=-1
                    else:
                        vectorized=current_app.vectorizer.transform([processed_text]); pred=current_app.model.predict(vectorized); proba=current_app.model.predict_proba(vectorized)
                        view_data["prediction_label"]=int(pred[0])
                        view_data["prediction_text"] = f"Prediction: {'REAL' if pred[0]==0 else 'FAKE'} News"
                        view_data["confidence_score"] = f"{proba[0][pred[0]]*100:.2f}"
                except Exception as e_pred: print(f"Pred error: {e_pred}"); traceback.print_exc(); view_data["error_message"]="Err during classification."
        
        if not view_data["error_message"]: view_data["explanation_paragraph"] = generate_explanation_paragraph(view_data)
        elif view_data["analysis_done"] and not news_text_to_analyze : # Catch cases where no text was available
            if view_data["input_type"] == 'url' and not view_data.get("news_url_value","").strip() : view_data["error_message"] = "URL empty or invalid."
            elif view_data["input_type"] == 'text' and not view_data.get("news_text_value","").strip(): view_data["error_message"] = "Please enter text."

    return render_template('index.html', **view_data)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        original_input_type = request.form.get('original_input_type')
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
            if predicted_label_internal_str and predicted_label_internal_str != 'None' and predicted_label_internal_str.lstrip('-').isdigit(): 
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
    return redirect(url_for('index'))