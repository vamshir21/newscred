import nltk
import os

# --- FORCE NLTK DATA PATH (Same as in train_model.py) ---
PROJECT_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assuming this script is in NEWSCRED
PROJECT_NLTK_DATA_PATH = os.path.join(PROJECT_BASE_DIR, 'my_nltk_data')

if not os.path.exists(PROJECT_NLTK_DATA_PATH):
    print(f"ERROR: NLTK data directory not found: {PROJECT_NLTK_DATA_PATH}")
    print("Please ensure it exists and is populated, or run train_model.py first to create it.")
    exit()

nltk.data.path = [PROJECT_NLTK_DATA_PATH]
print(f"NLTK data path exclusively set to: {nltk.data.path}")
# --- END FORCE NLTK DATA PATH ---

print("\nAttempting to load English PunktSentenceTokenizer directly...")
try:
    # Try to load the English Punkt model directly
    # NLTK looks for 'tokenizers/punkt/english.pickle' or similar within its data path
    english_sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("SUCCESS: Loaded 'tokenizers/punkt/english.pickle' directly.")

    test_text = "This is a test sentence. This is another one."
    print(f"\nAttempting to tokenize sentences: '{test_text}'")
    sentences = english_sentence_tokenizer.tokenize(test_text)
    print(f"Sentences found: {sentences}")

    print(f"\nAttempting nltk.word_tokenize on: '{test_text}'")
    words = nltk.word_tokenize(test_text)
    print(f"Words found: {words}")
    print("SUCCESS: nltk.word_tokenize also worked.")

except LookupError as e:
    print(f"LOOKUP ERROR during test: {e}")
    print("This suggests the resource alias or internal loading is the primary issue.")
except Exception as e:
    print(f"UNEXPECTED ERROR during test: {e}")
    import traceback
    traceback.print_exc()

print("\n--- NLTK Data Path Contents ---")
punkt_dir_path = os.path.join(PROJECT_NLTK_DATA_PATH, 'tokenizers', 'punkt')
if os.path.exists(punkt_dir_path):
    print(f"Contents of {punkt_dir_path}:")
    for item in os.listdir(punkt_dir_path):
        print(f"  - {item}")
    
    english_pickle_path = os.path.join(punkt_dir_path, 'english.pickle')
    if os.path.exists(english_pickle_path):
        print(f"\n'{english_pickle_path}' exists.")
    else:
        print(f"\n'{english_pickle_path}' DOES NOT exist.")

    py3_english_pickle_path = os.path.join(punkt_dir_path, 'PY3', 'english.pickle')
    if os.path.exists(py3_english_pickle_path):
        print(f"'{py3_english_pickle_path}' exists.")
    else:
        print(f"'{py3_english_pickle_path}' DOES NOT exist.")
else:
    print(f"Directory {punkt_dir_path} DOES NOT exist.")