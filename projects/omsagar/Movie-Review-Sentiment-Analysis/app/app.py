import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

lime_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in cleaned.split()])
    vec = tfidf.transform([lemmatized])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return ('Positive' if pred == 1 else 'Negative'), max(prob) * 100

def predict_proba_for_lime(texts):
    cleaned = [' '.join([lemmatizer.lemmatize(w) for w in clean_text(t).split()]) for t in texts]
    vecs = tfidf.transform(cleaned)
    return model.predict_proba(vecs)

st.set_page_config(page_title="Sentiment AI", page_icon="🎬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Page load animation ---- */
.stApp {
    background: linear-gradient(180deg, #fbfbfe 0%, #f4f3fb 100%);
    animation: pageFadeIn 0.6s ease;
}

@keyframes pageFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.block-container {
    animation: pageSlideUp 0.5s cubic-bezier(.16,1,.3,1);
}

@keyframes pageSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Logo (SVG icon, not emoji) ---- */
.logo-wrap {
    text-align: center;
    margin-top: 10px;
    margin-bottom: 4px;
}

.logo-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 24px;
    border-radius: 999px;
    background: #4338ca;
}

.logo-badge svg { display: block; }

.logo-badge span {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: white;
    letter-spacing: -0.02em;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    font-weight: 500;
    margin-top: 14px;
    margin-bottom: 2rem;
}

/* ---- Text area ---- */
div[data-testid="stTextArea"] textarea {
    border-radius: 16px;
    border: 1.5px solid #e5e3f7;
    padding: 18px;
    font-size: 1.02rem;
    background: #ffffff;
    transition: border-color 0.2s ease;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #4338ca;
    box-shadow: 0 0 0 4px rgba(67,56,202,0.12);
}

/* ---- Button with click/hover motion ---- */
.stButton > button {
    background: #4338ca;
    color: white;
    border-radius: 999px;
    padding: 14px 32px;
    font-weight: 600;
    font-size: 1.02rem;
    border: none;
    width: 100%;
    transition: transform 0.15s cubic-bezier(.4,0,.2,1), background 0.2s ease;
}

.stButton > button:hover {
    background: #3730a3;
    transform: translateY(-2px);
}

.stButton > button:active {
    transform: translateY(0px) scale(0.97);
}

/* ---- Result card ---- */
.result-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 28px;
    margin-top: 22px;
    border: 1px solid #edecf9;
    animation: cardIn 0.4s cubic-bezier(.16,1,.3,1);
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(14px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.sentiment-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.sentiment-positive {
    color: #059669;
    font-size: 1.7rem;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
}

.sentiment-negative {
    color: #dc2626;
    font-size: 1.7rem;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
}

.confidence-label {
    color: #9ca3af;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.confidence-value {
    color: #111827;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
}

/* ---- Word contribution bars ---- */
.word-row {
    display: flex;
    align-items: center;
    margin: 8px 0;
    gap: 10px;
}

.word-label {
    width: 90px;
    font-size: 0.88rem;
    color: #374151;
    font-weight: 500;
}

.word-bar {
    height: 10px;
    border-radius: 6px;
    animation: barGrow 0.5s cubic-bezier(.16,1,.3,1);
    transform-origin: left;
}

@keyframes barGrow {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

/* ---- Custom loading animation ---- */
.custom-loader {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 0;
    color: #4338ca;
    font-size: 0.92rem;
    font-weight: 500;
}

.spinner {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 3px solid #e5e3f7;
    border-top-color: #4338ca;
    animation: spin 0.7s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #fbfbfe;
    border-right: 1px solid #ece9fd;
}

[data-testid="stSidebar"] h3 {
    color: #4338ca;
    font-family: 'Poppins', sans-serif;
}

/* ---- Tabs ---- */
div[data-testid="stTabs"] button {
    font-weight: 600;
    color: #9ca3af;
    font-family: 'Poppins', sans-serif;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4338ca;
}

/* ---- File uploader ---- */
section[data-testid="stFileUploaderDropzone"] {
    border-radius: 16px;
    border: 1.5px dashed #c7c2f0;
    background: #ffffff;
}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 16px;
    border: 1px solid #edecf9;
}
</style>
""", unsafe_allow_html=True)

FILM_ICON = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="1"/><path d="M7 6v12M17 6v12M2 10h5M17 10h5M2 14h5M17 14h5"/></svg>"""
CHECK_ICON = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4"/></svg>"""
X_ICON = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>"""
SEARCH_ICON = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4338ca" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>"""

st.markdown(f"""
<div class="logo-wrap">
    <div class="logo-badge">
        {FILM_ICON}
        <span>Sentiment AI</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Understand movie reviews at a glance</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])

with tab1:
    review = st.text_area("", placeholder="Paste a movie review here...", height=130, label_visibility="collapsed")
    predict_clicked = st.button("Analyze sentiment")

    if predict_clicked:
        if review.strip():
            sentiment, confidence = predict_sentiment(review)
            css_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
            icon = CHECK_ICON if sentiment == "Positive" else X_ICON

            st.markdown(f"""
            <div class="result-card">
                <div class="sentiment-row">
                    {icon}
                    <span class="{css_class}">{sentiment}</span>
                </div>
                <br>
                <span class="confidence-label">Confidence</span><br>
                <span class="confidence-value">{confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            loader = st.empty()
            loader.markdown("""
            <div class="custom-loader">
                <div class="spinner"></div>
                <span>Analyzing word contributions...</span>
            </div>
            """, unsafe_allow_html=True)

            lime_exp = lime_explainer.explain_instance(review, predict_proba_for_lime, num_features=8)
            word_scores = lime_exp.as_list()
            loader.empty()

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; margin-top:8px;">
                {SEARCH_ICON}
                <span style="font-weight:600; font-size:1.05rem; color:#111827; font-family:'Poppins',sans-serif;">Why this prediction?</span>
            </div>
            """, unsafe_allow_html=True)

            max_score = max(abs(s) for _, s in word_scores) if word_scores else 1
            for word, score in sorted(word_scores, key=lambda x: abs(x[1]), reverse=True):
                bar_color = "#059669" if score > 0 else "#dc2626"
                bar_width = min((abs(score) / max_score) * 100, 100)
                st.markdown(f"""
                <div class="word-row">
                    <span class="word-label">{word}</span>
                    <div class="word-bar" style="background:{bar_color}; width:{bar_width}%;"></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a review first.")

with tab2:
    uploaded = st.file_uploader("Upload a CSV with a 'review' column", type=['csv'])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if 'review' in batch_df.columns:
            batch_df['prediction'], batch_df['confidence'] = zip(*batch_df['review'].apply(
                lambda x: predict_sentiment(str(x))
            ))
            st.dataframe(batch_df[['review', 'prediction', 'confidence']], use_container_width=True)

            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#059669' if v == 'Positive' else '#dc2626' for v in batch_df['prediction'].value_counts().index]
            batch_df['prediction'].value_counts().plot(kind='bar', ax=ax, color=colors)
            ax.set_title("Sentiment distribution", fontsize=12, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig)
        else:
            st.error("CSV must have a 'review' column")

st.sidebar.markdown("### About")
st.sidebar.write("TF-IDF (unigrams + bigrams) + tuned Logistic Regression trained on 50,000 IMDB movie reviews. Predictions explained using LIME.")
st.sidebar.markdown("---")
st.sidebar.caption("Built as part of AI/ML Internship — Month 1 Project")