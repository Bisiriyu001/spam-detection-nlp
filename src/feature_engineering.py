import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------
# Create and fit TF-IDF Vectorizer
# -------------------------------

def create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """
    Initialize a TF-IDF Vectorizer.

    Parameters:
    - max_features: Limit vocabulary size (default 5000)
    - ngram_range: Use unigrams + bigrams

    Returns:
        TfidfVectorizer object
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )


def fit_vectorizer(vectorizer, text_data):
    """
    Fit TF-IDF vectorizer on training text.
    """
    return vectorizer.fit(text_data)


def transform_text(vectorizer, text_data):
    """
    Transform text into TF-IDF vectors.
    """
    return vectorizer.transform(text_data)


# -------------------------------
# Save & Load Vectorizer
# -------------------------------

def save_vectorizer(vectorizer, filepath):
    """
    Save TF-IDF vectorizer to a .pkl file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(filepath):
    """
    Load TF-IDF vectorizer from .pkl file.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)
