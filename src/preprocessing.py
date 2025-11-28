import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize global objects once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Basic text cleaning:
    - Lowercase
    - Remove URLs
    - Remove punctuation
    - Remove numbers
    - Remove extra spaces
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # keep letters only
    text = re.sub(r"\s+", " ", text).strip()          # remove extra spaces
    return text


def remove_stopwords(text):
    """
    Remove common English stopwords.
    """
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


def lemmatize_text(text):
    """
    Convert words to their base form (lemma).
    """
    tokens = text.split()
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmas)


def preprocess_text(text):
    """
    Apply full text preprocessing pipeline:
    1. Clean text
    2. Remove stopwords
    3. Lemmatize text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text
