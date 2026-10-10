# AppInsight — Presentation-Ready Financial App Review Analysis

## Files
- `appinsight.py`: NLP preprocessing, built-in Finance app filter, model training, aspect analysis, and summary logic.
- `app.py`: Streamlit user interface.
- `requirements.txt`: Python dependencies.

## Dataset
Download only `googleplaystore_user_reviews.csv` from:
https://www.kaggle.com/datasets/lava18/google-play-store-apps?select=googleplaystore_user_reviews.csv

The app has a built-in list of 39 Finance-category app names from the matching Kaggle dataset. It filters the uploaded review file by the `App` column, so `googleplaystore.csv` metadata does not need to be uploaded separately.

**Scope note:** Google Play's Finance category includes banking, payments, credit, investment, and other finance-related apps. The built-in list is a practical filter for this dataset, not a guarantee that every app is a bank.

## Run in VS Code (Windows)
1. Install Python 3.10 or 3.11 and the VS Code Python extension.
2. Open this folder in VS Code.
3. In Terminal > New Terminal, create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   streamlit run app.py
   ```
6. For automatic loading, place `googleplaystore_user_reviews.csv` in the same folder as `app.py`; otherwise upload it in the browser. The app filters financial apps internally. You do not need `googleplaystore.csv`.

## Current features
- NLTK text cleaning, stopword removal, lemmatization, and optional stemming in the core function.
- TF-IDF + Logistic Regression sentiment classifier with accuracy, precision, recall, and F1.
- Keyword-based aspect/issue detection.
- Common-word frequency table and sentiment chart.
- Offline data-grounded summary with sentiment distribution, common topics, and representative review examples.
- Download filtered reviews as CSV.

Translation and the vectorization demonstration have been removed from the UI. The summary no longer attempts to download a transformer model, so it works without transformer installation or external model access.
