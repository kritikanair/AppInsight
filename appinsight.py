"""
AppInsight - core NLP functions.
The review CSV is uploaded by the user; financial-app filtering uses a built-in
allowlist based on the Finance apps represented in the chosen Kaggle dataset.
"""
import re
from collections import Counter

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Finance-category apps present in the matching Kaggle review/metadata files.
# Matching is case-insensitive; app names are checked against the Review CSV's App column.
FINANCIAL_APPS = [
    "A+ Mobile", "ACE Elite", "Acorns - Invest Spare Change", "Amex Mobile",
    "Associated Credit Union Mobile", "BBVA Compass Banking", "BBVA Spain",
    "BZWBK24 mobile", "Bank of America Mobile Banking", "BankMobile Vibe App",
    "Banorte Movil", "Barclays US for Android", "Betterment", "Bloomberg Professional",
    "Branch", "Bualuang mBanking", "CNBC: Breaking Business News & Live Market Data",
    "Chase Mobile", "Citi Mobile®", "Citibanamex Movil", "Citizens Bank Mobile Banking",
    "Credit Karma", "Credit Sesame", "CreditWise from Capital One",
    "Current debit card and app made for teens", "Digit Save Money Automatically",
    "Discover Mobile", "E*TRADE Mobile", "Ecobank Mobile Banking",
    "Even - organize your money, get paid early", "Expense IQ Money Manager",
    "Experian - Free Credit Report", "Fortune City - A Finance App", "Fox Business",
    "Fresh EBT - Food Stamp Balance", "GCash - Buy Load, Pay Bills, Send Money",
    "GoBank", "Google Pay", "HDFC Bank MobileBanking"
]
FINANCIAL_APP_KEYS = {name.strip().casefold() for name in FINANCIAL_APPS}

# Download small NLTK resources on first run if they are missing.
def ensure_nltk_data():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

ensure_nltk_data()
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text, method="lemmatization"):
    """Basic cleaning, tokenization, stopword removal and stemming/lemmatization."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    tokens = re.findall(r"[a-z']+", text)
    cleaned = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 1:
            if method == "stemming":
                token = STEMMER.stem(token)
            elif method == "lemmatization":
                try:
                    token = LEMMATIZER.lemmatize(token)
                except LookupError:
                    pass
            cleaned.append(token)
    return " ".join(cleaned)


def find_review_column(df):
    for name in ["Translated_Review", "Review", "review", "content", "text"]:
        if name in df.columns:
            return name
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) and sample.str.len().mean() > 15:
                return col
    raise ValueError("Could not find a review text column. Expected Translated_Review or Review.")


def find_sentiment_column(df):
    for name in ["Sentiment", "sentiment", "label", "Label"]:
        if name in df.columns:
            return name
    return None


def prepare_reviews(df, finance_only=True):
    """Standardize the review CSV and optionally retain built-in Finance app names."""
    if df is None or df.empty:
        raise ValueError("The uploaded review CSV is empty.")
    if "App" not in df.columns:
        raise ValueError("The review CSV needs an 'App' column to filter financial apps.")

    review_col = find_review_column(df)
    result = pd.DataFrame()
    result["Review"] = df[review_col].fillna("").astype(str).str.strip()
    result["App"] = df["App"].fillna("").astype(str).str.strip()
    sentiment_col = find_sentiment_column(df)
    result["Sentiment"] = df[sentiment_col].fillna("").astype(str).str.title() if sentiment_col else ""
    result = result[result["Review"].ne("") & result["Review"].str.lower().ne("nan")].copy()

    if finance_only:
        result = result[result["App"].str.casefold().isin(FINANCIAL_APP_KEYS)].copy()
        if result.empty:
            raise ValueError(
                "No reviews matched the built-in Finance app list. Check that this is the Kaggle "
                "googleplaystore_user_reviews.csv file and that app names have not been changed."
            )
        result["Category"] = "FINANCE (built-in app list)"
    else:
        result["Category"] = "All categories"
    result.reset_index(drop=True, inplace=True)
    return result


def train_sentiment_model(df):
    """Train TF-IDF + Logistic Regression using the dataset's existing sentiment labels."""
    data = df.copy()
    data["clean"] = data["Review"].apply(preprocess_text)
    data = data[data["Sentiment"].isin(["Positive", "Negative", "Neutral"])]
    data = data[data["clean"].str.len() > 0]
    if len(data) < 20 or data["Sentiment"].nunique() < 2:
        raise ValueError("At least 20 labelled reviews and 2 sentiment classes are needed to train the model.")
    counts = data["Sentiment"].value_counts()
    stratify = data["Sentiment"] if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        data["clean"], data["Sentiment"], test_size=0.2, random_state=42, stratify=stratify
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (macro)": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1-score (macro)": f1_score(y_test, predictions, average="macro", zero_division=0),
        "Test samples": int(len(y_test)),
    }
    return model, metrics


def predict_reviews(model, reviews):
    clean = [preprocess_text(x) or "empty review" for x in reviews]
    return model.predict(clean)


ASPECT_KEYWORDS = {
    "Payments & transactions": ["payment", "transaction", "transfer", "upi", "refund", "money", "bank transfer", "deposit", "withdraw"],
    "Login & registration": ["login", "log in", "sign in", "password", "otp", "register", "verification", "fingerprint"],
    "App performance": ["slow", "crash", "crashing", "freeze", "loading", "hang", "update", "bug", "open app", "not open"],
    "Customer support": ["support", "customer care", "helpline", "complaint", "response", "service"],
    "User interface": ["interface", "design", "layout", "navigation", "easy to use", "user friendly", "look"],
    "Security & privacy": ["security", "secure", "fraud", "privacy", "blocked", "suspicious", "scam"],
}


def extract_aspects(review):
    text = str(review).lower()
    found = [aspect for aspect, words in ASPECT_KEYWORDS.items() if any(word in text for word in words)]
    return found or ["General feedback"]


def aspect_summary(df, sentiment_col="Predicted Sentiment"):
    rows = []
    for _, row in df.iterrows():
        sentiment = row.get(sentiment_col, row.get("Sentiment", "Unknown"))
        for aspect in extract_aspects(row["Review"]):
            rows.append({"Aspect": aspect, "Sentiment": sentiment})
    if not rows:
        return pd.DataFrame(columns=["Aspect", "Reviews"])
    expanded = pd.DataFrame(rows)
    counts = expanded.groupby("Aspect").size().sort_values(ascending=False)
    return counts.rename("Reviews").reset_index()


def word_frequency(texts, top_n=15):
    combined = " ".join(preprocess_text(t) for t in texts)
    counts = Counter(combined.split())
    return pd.DataFrame(counts.most_common(top_n), columns=["Word", "Frequency"])


def build_review_summary(df, max_examples=3):
    """Build a readable, data-grounded summary without downloading a model."""
    if df is None or df.empty:
        return {"overview": "No reviews are available to summarize.", "aspects": [], "examples": []}

    data = df.copy()
    if "Predicted Sentiment" in data.columns:
        sentiments = data["Predicted Sentiment"].astype(str)
    elif "Sentiment" in data.columns:
        sentiments = data["Sentiment"].astype(str)
    else:
        sentiments = pd.Series(["Unknown"] * len(data), index=data.index)

    counts = sentiments.value_counts()
    total = int(counts.sum()) or 1
    sentiment_bits = [f"{name.lower()}: {count / total:.0%}" for name, count in counts.items() if name != "Unclassified"]
    overview = f"This summary covers {len(data):,} financial-app reviews. "
    if sentiment_bits:
        overview += "Sentiment distribution is " + ", ".join(sentiment_bits) + ". "

    aspect_rows = []
    for review in data["Review"].astype(str):
        for aspect in extract_aspects(review):
            aspect_rows.append(aspect)
    aspect_counts = Counter(aspect_rows)
    top_aspects = aspect_counts.most_common(4)
    if top_aspects:
        overview += "The most frequently detected topics are " + ", ".join(
            f"{name.lower()} ({count})" for name, count in top_aspects if name != "General feedback"
        ) + "." if any(name != "General feedback" for name, _ in top_aspects) else "Many reviews contain general feedback that does not match the current issue keywords."

    examples = []
    for label in ["Negative", "Positive", "Neutral"]:
        subset = data.loc[sentiments.eq(label), "Review"].astype(str)
        # Pick a useful, not-too-short representative review, preferring distinct examples.
        candidates = sorted((s.strip() for s in subset if len(s.strip()) >= 25), key=len)
        if candidates:
            pick = candidates[len(candidates) // 2]
            if pick not in examples:
                examples.append(f"{label}: {pick[:280]}{'…' if len(pick) > 280 else ''}")
        if len(examples) >= max_examples:
            break
    return {"overview": overview.strip(), "aspects": top_aspects, "examples": examples}
