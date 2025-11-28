import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pickle

from src.preprocessing import preprocess_text
from src.feature_engineering import load_vectorizer


# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="",
    layout="centered"
)


# -------------------------------
# LOAD MODELS & VECTORIZER
# -------------------------------
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# SVM MODEL
with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "rb") as f:
    svm_model = pickle.load(f)

# TF-IDF
vectorizer = load_vectorizer(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))


# Initialize session state
if "example" not in st.session_state:
    st.session_state.example = ""


# -------------------------------
# HEADER UI
# -------------------------------
st.markdown(
    """
    <h2 style='text-align:center; color:#4CAF50;'>Spam Detection App</h2>
    <p style='text-align:center;'>
        Classify SMS or email messages as <b>Spam</b> or <b>Ham</b>.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)


# -------------------------------
#  EXAMPLE MESSAGE BUTTON + TEXT INPUT
# -------------------------------

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


def set_example():
    st.session_state.input_text = (
        "Congratulations! You won a free ticket. Reply with your bank details to claim."
    )

# Button that fills the text area
st.button("Use Example Message", on_click=set_example)

# Text area linked to session_state
user_input = st.text_area(
    "Enter a message:",
    key="input_text",
)

# -------------------------------
# CUSTOM DARK INPUT STYLING
# -------------------------------
st.markdown("""
    <style>
        textarea {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #444444 !important;
        }
        input, textarea {
            caret-color: #FFFFFF !important;
        }
        ::placeholder {
            color: #BBBBBB !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = preprocess_text(user_input)
        vec = vectorizer.transform([cleaned])

        pred = svm_model.predict(vec)[0]
        score = svm_model.decision_function(vec)[0]

        # -------------------------------
        # RESULT UI
        # -------------------------------
        if pred == "spam":
            st.markdown(
                f"""
                <div style='padding:15px; border-radius:10px; background-color:#800000;'>
                    <h3 style='color:#ffffff;'>Spam Detected</h3>
                    <p>Model Score: <b>{score:.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='padding:15px; border-radius:10px; background-color:#002b1a;'>
                    <h3 style='color:#ffffff;'>Not Spam</h3>
                    <p>Model Score: <b>{score:.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# -------------------------------
# SHORT EXPLANATION
# -------------------------------
st.markdown(
    """
    <p style='font-size:14px; color:white; margin-top:10px;'>
    The spam detector compares your message with patterns learned from real spam examples.
    It looks for:
    <br>• common scam phrases
    <br>• money or prize claims
    <br>• urgent requests
    <br>• suspicious wording
    <br><br>
    The score shows how similar the message is to known spam. Higher scores suggest higher risk.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:13px; color:white;'>
        Built with using Streamlit & Machine Learning.
    </p>
    """,
    unsafe_allow_html=True
)
