import nltk
import shutil
import os

print("Attempting to clean and re-download NLTK 'punkt' package...")

# Determine NLTK's default download directory
# This is usually C:\Users\YourName\AppData\Roaming\nltk_data on Windows
try:
    nltk_data_dir = nltk.downloader.Downloader().default_download_dir()
    print(f"NLTK default data directory: {nltk_data_dir}")
except Exception as e:
    print(f"Could not determine NLTK default data directory: {e}")
    # Fallback to a common Windows path if the above fails
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')
    print(f"Using fallback NLTK data directory: {nltk_data_dir}")

# Define paths for punkt resources
punkt_tokenizer_path = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
punkt_zip_path = os.path.join(nltk_data_dir, 'tokenizers', 'punkt.zip')

# --- Attempt to delete existing punkt resources ---
deleted_something = False
if os.path.exists(punkt_tokenizer_path):
    try:
        shutil.rmtree(punkt_tokenizer_path)
        print(f"- Successfully deleted directory: {punkt_tokenizer_path}")
        deleted_something = True
    except Exception as e:
        print(f"- Error deleting directory {punkt_tokenizer_path}: {e}")

if os.path.exists(punkt_zip_path):
    try:
        os.remove(punkt_zip_path)
        print(f"- Successfully deleted file: {punkt_zip_path}")
        deleted_something = True
    except Exception as e:
        print(f"- Error deleting file {punkt_zip_path}: {e}")

if not deleted_something:
    print("- No existing 'punkt' directory or zip file found to delete at expected paths.")

# --- Attempt to re-download 'punkt' forcefully ---
print("\nAttempting to download 'punkt' (force overwrite)...")
try:
    # Using force=True to try and overwrite if NLTK thinks it's up-to-date
    nltk.download('punkt', download_dir=nltk_data_dir, force=True, raise_on_error=True, quiet=False)
    print("'punkt' package download attempt complete.")
except Exception as e:
    print(f"ERROR during 'punkt' download: {e}")
    print("Please ensure you have an internet connection.")
    print("If this continues, there might be a deeper issue with NLTK paths or permissions.")

# --- Verify ---
print("\nVerifying 'punkt' and 'punkt_tab' after download attempt...")
try:
    nltk.word_tokenize("Verify punkt.") # Basic check
    nltk.data.find('tokenizers/punkt_tab/english/punkt.xml') # Specific check
    print("SUCCESS: 'punkt' and 'punkt_tab' seem to be available now!")
except LookupError as e:
    print(f"VERIFICATION FAILED: Still encountering LookupError: {e}")
    print("This is very persistent. Please double-check your NLTK installation and data paths.")
    print(f"NLTK is searching for data in paths including: {nltk.data.path}")

print("\nfix_nltk.py script finished.")