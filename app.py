import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Spam Message Detector", page_icon="📧", layout="wide")
DATA = Path(__file__).parent / "data" / "sample_messages.csv"

@st.cache_resource
def train():
    df = pd.read_csv(DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, df, accuracy_score(y_test, pred), y_test, pred

model, df, accuracy, y_test, pred = train()
st.title("📧 Spam Message Prediction")
st.caption("A machine-learning demonstration using TF-IDF and Multinomial Naive Bayes.")
c1,c2,c3 = st.columns(3)
c1.metric("Training examples", len(df))
c2.metric("Labels", "Spam / Ham")
c3.metric("Hold-out accuracy", f"{accuracy:.1%}")
st.info("Academic note: this starter bundle includes a small illustrative dataset so the app runs immediately. For a meaningful evaluation, replace it with the full UCI SMS Spam Collection and retrain.")
st.subheader("Test a message")
message = st.text_area("Paste an SMS or email message", placeholder="Example: Congratulations! Claim your free reward now...")
if st.button("Analyse message", type="primary", disabled=not message.strip()):
    label = model.predict([message])[0]
    probs = model.predict_proba([message])[0]
    confidence = float(max(probs))
    if label == "spam":
        st.error(f"⚠️ Prediction: SPAM  |  Model confidence: {confidence:.1%}")
    else:
        st.success(f"✅ Prediction: HAM (likely legitimate)  |  Model confidence: {confidence:.1%}")
    st.caption("This is a model prediction, not a guarantee. Do not click suspicious links or share sensitive information.")
with st.expander("Model evaluation on the small demonstration hold-out split"):
    st.write(f"Accuracy: {accuracy:.1%}")
    st.text(classification_report(y_test, pred, zero_division=0))
    st.write("Confusion matrix (rows = actual, columns = predicted; label order: ham, spam)")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, pred, labels=["ham","spam"]),
                              index=["Actual ham","Actual spam"], columns=["Predicted ham","Predicted spam"]))
st.subheader("How it works")
st.markdown("1. Text is converted into TF-IDF numerical features.  2. Multinomial Naive Bayes estimates the message class.  3. The app displays the predicted label and model confidence.")
st.caption("Dataset: illustrative examples only. See README for how to download and cite the UCI dataset.")
