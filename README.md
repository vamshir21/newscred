# newscred
# NewsCred+: Cloud-Powered Fake News Detection and Credibility Scoring System

## 🚀 Project Overview

NewsCred+ is an advanced, AI-driven system designed to combat online misinformation. It provides a comprehensive suite of tools to help users:
*   Detect fake news articles based on their textual content.
*   Evaluate the credibility of news sources.
*   Analyze the sentiment expressed in news articles.
*   Perform reverse image searches to identify potentially manipulated or out-of-context media.
*   Get concise summaries of news articles.
*   Corroborate information by checking against other web sources and fact-checking sites.

This project leverages machine learning, natural language processing (NLP), web scraping, reverse image search APIs, and cloud deployment principles to offer a robust solution against the spread of disinformation.

## 🌟 Key Features Implemented

*   ✅ **Fake News Detection:** Utilizes a machine learning classifier (currently Logistic Regression with TF-IDF) trained on a labeled news dataset to classify articles as potentially "Real" or "Fake."
*   ✅ **URL & Text Input:** Users can either paste news article text directly or provide a URL, from which the system will attempt to scrape the main content.
*   ✅ **Sentiment Analysis:** Analyzes the tone (Positive/Negative/Neutral) of the news content using NLTK's VADER sentiment analyzer.
*   ✅ **Article Summarization:** Provides an extractive summary of the article content using the `sumy` library (LSA algorithm).
*   ✅ **Image Extraction & Basic Reverse Image Search:**
    *   Extracts the top image and other images from URL-based articles using `newspaper3k`.
    *   Integrates with Google Cloud Vision API for web detection (reverse image search) on the top extracted image, showing web entities, matching images, and pages where the image appears. *(Requires Google Cloud Credentials)*
*   ✅ **Web Corroboration (Enhanced):**
    *   Generates search queries based on the article title or extracted keywords (using spaCy for Named Entity Recognition if available, otherwise basic keyword extraction).
    *   Uses Google Custom Search API to find related information online. *(Requires Google API Key & CSE ID)*
    *   Categorizes search results into:
        *   Fact-Checking Sites
        *   Reputable News Sources
        *   General Mentions
    *   Flags results containing debunking keywords.
*   ✅ **Source Credibility Scoring (Basic):**
    *   Extracts the domain from a URL.
    *   Checks against a manually curated CSV (`data/source_reputation.csv`) for pre-defined credibility scores, bias, and factual reporting ratings (inspired by MBFC).
*   ✅ **Feedback Collection (Stubbed):**
    *   UI elements allow users to provide feedback on the accuracy of the fake news prediction.
    *   Feedback is saved to a local SQLite database (`data/newscred_feedback.db`) for potential future model retraining.
*   ✅ **Web Application Interface:** Deployed as a local web application using Flask, allowing users to interact with the system through their browser.

## ⚙️ Technical Stack

*   **Languages:** Python, HTML, CSS, JavaScript
*   **Core Libraries/Frameworks:**
    *   **Machine Learning:** Scikit-learn (`LogisticRegression`, `TfidfVectorizer`)
    *   **NLP:** NLTK (tokenization, stopwords, VADER sentiment), spaCy (Named Entity Recognition for corroboration queries)
    *   **Web Scraping/Article Parsing:** `newspaper3k`, `requests`, `BeautifulSoup4` (implicitly by newspaper3k)
    *   **Summarization:** `sumy`
    *   **Web Framework:** Flask
    *   **Data Handling:** Pandas
    *   **Model Persistence:** `joblib`
    *   **Database:** SQLite3 (for feedback)
*   **APIs Used:**
    *   Google Custom Search JSON API (for web corroboration)
    *   Google Cloud Vision API (for reverse image search)
*   **Development Environment:** Python Virtual Environment (`venv`)

## 📊 System Pipeline Overview

1.  **User Input:** User provides news article text or a URL.
2.  **URL Processing (if URL provided):**
    *   Article text, title, and images are scraped using `newspaper3k`.
    *   The domain is extracted for source credibility assessment.
3.  **Source Credibility:** The extracted domain is checked against the `source_reputation.csv` database.
4.  **Text Analysis (on scraped or pasted text):**
    *   **Preprocessing:** Text is cleaned (lowercase, remove punctuation, remove stopwords).
    *   **Fake News Classification:** The preprocessed text is vectorized using the loaded TF-IDF model and classified by the trained Logistic Regression model.
    *   **Sentiment Analysis:** NLTK VADER analyzes the sentiment of the original text.
    *   **Summarization:** `sumy` generates an extractive summary.
5.  **Web Corroboration:**
    *   Keywords/title are used to query the Google Custom Search API.
    *   Results are fetched and categorized (fact-checks, reputable sources, debunking flags).
6.  **Image Analysis (if top image available):**
    *   The top image URL is sent to the Google Cloud Vision API for web detection.
    *   Results (web entities, matching images, pages) are processed.
7.  **Result Aggregation & Display:** All analysis results are compiled and presented to the user on a web page.
8.  **Feedback Submission (Optional):** User can submit feedback on the fake news prediction, which is stored in the SQLite database.

## 🛠️ Setup and Installation

### Prerequisites
*   Python (Recommended: 3.9, 3.10, or 3.11 - spaCy compilation can be problematic with very new Python versions like 3.12+ on Windows without careful setup).
*   `pip` (Python package installer)
*   Git (for cloning the repository)
*   **Microsoft C++ Build Tools:** Required on Windows for installing some dependencies like `spaCy` if pre-compiled wheels are not available. Download "Build Tools for Visual Studio" and install the "Desktop development with C++" workload. **Restart your PC after installation.**

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd newscred-plus # Or your project folder name
    ```

2.  **Create and Activate a Python Virtual Environment:**
    ```bash
    # For Python 3.11 (replace with your Python version if different and compatible)
    python -m venv venv  # Or: py -3.11 -m venv venv
    
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```
    If you encounter issues installing `spacy` on Windows, ensure Microsoft C++ Build Tools are installed and you've restarted. Try installing from a "Developer Command Prompt for VS".

4.  **Download NLTK and spaCy Resources:**
    *   The application attempts to download necessary NLTK resources (`stopwords`, `punkt`, `vader_lexicon`) into a local `my_nltk_data` folder upon first run of `models/train_model.py` or app startup.
    *   Download the spaCy English model:
        ```bash
        python -m spacy download en_core_web_sm
        ```

5.  **API Key Configuration:**
    *   **Google Custom Search API:**
        1.  Enable the "Custom Search API" in your Google Cloud Console project.
        2.  Create an API Key.
        3.  Create a Custom Search Engine (CSE) at [cse.google.com](https://cse.google.com/cse/all) and configure it to "Search the entire web." Get the **Search Engine ID (CX)**.
        4.  Open `app/routes.py` and replace the placeholder values for `GOOGLE_SEARCH_API_KEY` and `CUSTOM_SEARCH_ENGINE_ID` with your actual credentials.
           ```python
           GOOGLE_SEARCH_API_KEY = "YOUR_ACTUAL_GOOGLE_API_KEY"
           CUSTOM_SEARCH_ENGINE_ID = "YOUR_ACTUAL_CSE_ID"
           ```
           **WARNING:** Do not commit your actual API keys to a public repository. For production, use environment variables.
    *   **Google Cloud Vision API:**
        1.  Enable the "Cloud Vision API" in your Google Cloud Console project.
        2.  Create a Service Account, grant it the "Cloud Vision API User" role, and download its JSON key file.
        3.  Set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to the **full path** of this downloaded JSON key file in the terminal where you will run the Flask app.
            ```bash
            # Example for Windows (Command Prompt) for current session:
            set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"
            # Example for Linux/macOS for current session:
            export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
            ```

6.  **Create Source Reputation Data:**
    *   Ensure the file `data/source_reputation.csv` exists. You can start with a few entries like:
      ```csv
      domain,mbfc_bias,mbfc_factual,credibility_score,notes
      nytimes.com,left-center,high,90,"Reputable, established"
      reuters.com,least-biased,very-high,95,"Highly reputable, fact-based"
      ```
    *   The application will create `data/newscred_feedback.db` on first run.

7.  **Train the Model (if not already done or if you want to retrain):**
    *   This script also populates the local `my_nltk_data` directory.
    ```bash
    python models/train_model.py
    ```
    This will create `models/fake_news_classifier.pkl` and `models/tfidf_vectorizer.pkl`.

## 🚀 Running the Application

1.  Ensure all setup steps are complete, especially API key configuration and environment variables.
2.  Make sure your virtual environment is activated.
3.  Navigate to the project root directory (`NEWSCRED`).
4.  Run the Flask application:
    ```bash
    python run.py
    ```
5.  Open your web browser and go to `http://127.0.0.1:5000`.

## 📝 Project Structure
NEWSCRED/
├── app/ # Flask application
│ ├── static/ # Static files (CSS, JS - if any beyond inline)
│ ├── templates/ # HTML templates
│ │ └── index.html
│ ├── init.py # Initializes Flask app, loads models, DB
│ └── routes.py # Flask routes and application logic
├── data/ # Data files
│ ├── source_reputation.csv # Manual source credibility data
│ ├── newscred_feedback.db # SQLite database for user feedback (created on run)
│ ├── Fake.csv # Sample training data (if included)
│ └── True.csv # Sample training data (if included)
├── models/ # Trained ML models and training scripts
│ ├── fake_news_classifier.pkl # Saved fake news model
│ ├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
│ └── train_model.py # Script to train the model
├── my_nltk_data/ # Local NLTK data (created on run)
├── venv/ # Python virtual environment
├── .gitignore # Specifies intentionally untracked files by Git
├── requirements.txt # Python package dependencies
├── run.py # Script to run the Flask development server

## 🔥 Potential Impact & Use Cases

*   Empower social media users to quickly assess the credibility of news encountered online.
*   Assist journalists and researchers in the initial vetting of sources and claims.
*   Provide a multi-faceted, explainable credibility score, moving beyond simple binary classifications.
*   Serve as an educational tool about the complexities of misinformation and the signals used to detect it.

## 🚀 Future Improvements & "Wow" Factors to Develop

*   **Advanced ML Model:** Replace Logistic Regression with a Transformer-based model (e.g., BERT, RoBERTa) fine-tuned on more diverse and recent fake news datasets for improved text classification accuracy.
*   **Official Source Cross-Checking:** For key organizations mentioned in an article, automatically attempt to find their official website and search for corroborating or contradicting information directly from the source.
*   **Dynamic Source Credibility:**
    *   Integrate WHOIS lookups for domain age as an indicator.
    *   Basic scraping of "About Us"/"Contact" pages for transparency signals.
*   **Richer Image Analysis:**
    *   Provide more context for Google Vision API results (e.g., common themes of pages where image appears).
    *   Look for signs of image manipulation if advanced APIs/models are explored.
*   **Feedback Loop for Model Retraining:** Develop a robust pipeline to periodically retrain the fake news classifier using verified user feedback stored in the database.
*   **Meta-Summary of Findings:** Generate a concise paragraph summarizing all the different analyses performed by NewsCred+ for a given article.
*   **Video Deepfake Detection (Advanced):** Incorporate tools and models for analyzing video content.
*   **Browser Extension & Mobile App:** Increase accessibility.
*   **Real-time Monitoring Dashboards:** For tracking system usage and detected misinformation trends.
*   **Cloud Deployment:** Deploy the application on a cloud platform (AWS, Azure, Google Cloud) for global access using Docker and CI/CD pipelines.
*   **Enhanced UI/UX:** Improve the visual presentation of results for better clarity and user experience.

## 🤝 Contribution & Feedback

This is a college project. Feedback, suggestions, and contributions (if applicable to your course structure) are welcome. Please feel free to open an issue or contact 
vamshi
vamshikrishna200421@gmail.com

## 📜 License


 This project is for educational purposes only.

---