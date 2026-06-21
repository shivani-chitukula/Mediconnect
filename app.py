import numpy as np
import pickle
import pandas as pd
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import re
import os
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

# App Configuration & Page setup
st.set_page_config(
    page_title="Mediconnect - Disease Prediction & Recommendation",
    page_icon="🩺",
    layout="wide"
)

# Premium UI CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    /* Title styling */
    .title-text {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0f766e, #0d9488, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Cards styling */
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }
    
    .predicted-disease {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 16px;
        font-size: 1.5rem;
        font-weight: 600;
        color: #166534;
    }
    
    /* Buttons custom styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0d9488, #0ea5e9);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Locate models & data relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_trigrams_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer3.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "drugsComTrain_raw.csv")

# Initialize models and tokenizer loads
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as m_file:
        model = pickle.load(m_file)
    with open(VECTORIZER_PATH, "rb") as v_file:
        vectorizer = pickle.load(v_file)
    return model, vectorizer

@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    # Filter highly useful and highly rated recommendations
    df_drug = df[(df["rating"] >= 9) & (df["usefulCount"] >= 100)].sort_values(
        by=["rating", "usefulCount"], ascending=[False, False]
    )
    return df_drug

model, vectorizer = load_models()
df_drug = load_dataset()

# Recommend drugs based on predicted condition
def recommend_drug(disease):
    if df_drug is None:
        return []
    recommended_drug_list = (
        df_drug[df_drug["condition"] == disease]["drugName"].head(3).tolist()
    )
    return recommended_drug_list

# API Key Resolution
# 1. Check environment variables
# 2. Check streamlit secrets
# 3. Fallback to input in sidebar (dynamic)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

# Sidebar decoration
if os.path.exists(os.path.join(BASE_DIR, "image.png")):
    st.sidebar.image(os.path.join(BASE_DIR, "image.png"), use_container_width=True)

st.sidebar.markdown("### 🛠️ Configuration")
if not api_key:
    st.sidebar.warning("⚠️ Gemini API Key not found in environment or secrets.")
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get a key from Google AI Studio")

# Configure Generative AI if key is present
if api_key:
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash as default, fallback to gemini-pro if needed
    try:
        gem = genai.GenerativeModel(model_name='gemini-1.5-flash')
    except Exception:
        gem = genai.GenerativeModel(model_name='models/gemini-pro')
else:
    gem = None

def get_ai_precautions(disease):
    if not gem:
        return None
    prompt = f"Suggest 3-4 simple, actionable precautions for a patient diagnosed with the disease: {disease}. Format as a clean bulleted list."
    try:
        response = gem.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error retrieving precautions: {e}"

def get_ai_medicines(disease):
    if not gem:
        return None
    prompt = f"Give only 2-3 standard tablet/medicine names recommended for the disease: {disease}. Provide only the names of the tablets, one per line, without any description or bullet points."
    try:
        response = gem.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error retrieving medicines: {e}"

# Voice input converter helper
def speech_to_text():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak your symptoms clearly.")
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        with st.spinner("Recognizing speech..."):
            text = recognizer.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        return "Error: Listening timed out. Please try speaking again."
    except sr.UnknownValueError:
        return "Error: Could not understand audio. Please speak louder and clearer."
    except sr.RequestError as e:
        return f"Error: Google speech service issue ({e})"
    except Exception as e:
        return f"Error: {e}"

# Unified disease results viewer
def display_results(symptoms):
    if not model or not vectorizer:
        st.error("Model files not found. Please ensure tfidf_trigrams_model.pkl and tfidf_vectorizer3.pkl are inside the models/ folder.")
        return

    text_transformed = vectorizer.transform([symptoms])
    result = model.predict(text_transformed)[0]
    recommended_drugs = recommend_drug(result)

    # 1. Output Predicted Disease & Offline Recommendations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Predicted Disease")
        st.markdown(f'<div class="predicted-disease">{result}</div>', unsafe_allow_html=True)
        
    with col2:
        st.subheader("💊 Recommended Drugs (from training dataset)")
        if recommended_drugs:
            for drug in recommended_drugs:
                st.markdown(f"- **{drug}**")
        else:
            st.info("No recommended drugs found for this condition in our dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Call Gemini API for Precautions and Medicines
    if gem:
        col_prec, col_med = st.columns(2)
        
        with col_prec:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🛡️ AI-Generated Precautions")
            with st.spinner("Fetching precautions..."):
                raw_prec = get_ai_precautions(result)
                if raw_prec:
                    cleaned_prec = raw_prec.replace('*', '').strip()
                    st.write(cleaned_prec)
                else:
                    st.write("Could not retrieve precautions.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_med:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🛒 Medicine Recommendations & Store Links")
            with st.spinner("Finding medicines..."):
                raw_meds = get_ai_medicines(result)
                if raw_meds:
                    cleaned_meds = raw_meds.replace('*', '').strip()
                    # Split and parse medicine lines
                    med_items = []
                    for line in cleaned_meds.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        # Remove leading bullets, numbering, spaces
                        clean_med = re.sub(r'^[-\*\d\.\s]+', '', line).strip()
                        if clean_med:
                            med_items.append(clean_med)
                            
                    if med_items:
                        for med in med_items:
                            st.write(f"👉 **{med}**")
                            # Pharmeasy and Apollo links
                            col_pe, col_ap = st.columns(2)
                            with col_pe:
                                st.link_button(f"Search {med} on PharmEasy", f"https://pharmeasy.in/search/all?name={med}", use_container_width=True)
                            with col_ap:
                                st.link_button(f"Search {med} on Apollo", f"https://www.apollopharmacy.in/search-medicines/{med}", use_container_width=True)
                            st.markdown("---")
                    else:
                        st.write("No specific medicine names returned.")
                else:
                    st.write("Could not retrieve AI medicine recommendations.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("💡 To view AI Precautions and buy online medicines, please enter your Gemini API Key in the sidebar.")

    # 3. Near by Medical shops Map link
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📍 Need a physical medical shop?")
    st.link_button("Find Medical Shops Near Me (Google Maps)", "https://www.google.com/maps/search/medical+shops/", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Application Menu / Navigation
st.sidebar.markdown("### 🧭 Navigation")
navigation = st.sidebar.selectbox("Go to page:", ["Home", "Text Input", "Voice Input", "Near by Medical Shops", "About"])

if navigation == "Home":
    st.markdown('<div class="title-text">Mediconnect</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="card" style="text-align: center;">
            <h2>Welcome to Mediconnect! 🩺</h2>
            <p style="font-size: 1.1rem; max-width: 800px; margin: 0 auto; color: #475569;">
                Mediconnect is an NLP-powered healthcare companion. It predicts potential medical conditions based on the symptoms you describe and recommends appropriate drugs from verified clinical databases.
            </p>
            <br>
            <p><strong>👈 Use the sidebar navigation to get started with Text or Voice inputs!</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(
            """
            <div class="card">
                <h3>⚡ Classifier & Text Embedding</h3>
                <p>Mediconnect uses a <strong>PassiveAggressiveClassifier</strong> trained on TF-IDF features to perform text classification. This makes it highly efficient at analyzing textual clinical notes and symptoms.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_info2:
        st.markdown(
            """
            <div class="card">
                <h3>🗄️ Curated Dataset</h3>
                <p>The model is trained to recognize the 11 most common conditions in the clinic dataset, ensuring higher accuracy and reliability for primary user symptoms.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif navigation == "Text Input":
    st.markdown('<div class="title-text">Text Symptom Checker</div>', unsafe_allow_html=True)
    
    with st.form(key="disease_clf_form"):
        raw_text = st.text_area("Describe your symptoms in detail here:", height=150, placeholder="Example: I have severe headache, migraine, nausea and blurry vision.")
        submit_text = st.form_submit_button(label="Submit Symptoms")
        
    if submit_text:
        if raw_text.strip():
            display_results(raw_text)
        else:
            st.warning("Please type in some symptoms before submitting.")

elif navigation == "Voice Input":
    st.markdown('<div class="title-text">Voice Symptom Checker</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="card">
            <h3>🎙️ Speak Your Symptoms</h3>
            <p>Click the button below to start recording. Speak clearly into your microphone about how you are feeling.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # We use a button to trigger voice recording
    if st.button("Start Recording Voice"):
        voice_text = speech_to_text()
        
        if voice_text.startswith("Error"):
            st.error(voice_text)
        else:
            st.success(f"Recognized symptoms: *\"{voice_text}\"*")
            st.markdown("### Analyzing Symptoms...")
            display_results(voice_text)

elif navigation == "Near by Medical Shops":
    st.markdown('<div class="title-text">Locate Nearby Medical Stores</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="card" style="text-align: center;">
            <h3>📍 Find Local Stores Instantly</h3>
            <p style="color: #475569; margin-bottom: 20px;">Use the links below to query Google Maps for pharmacy outlets near your current location.</p>
            
            <div style="display: flex; flex-direction: column; gap: 15px; max-width: 400px; margin: 0 auto;">
        """,
        unsafe_allow_html=True
    )
    
    st.link_button("🏥 Find General Medical Shops Near Me", "https://www.google.com/maps/search/medical+shops/", use_container_width=True)
    st.link_button("🔴 Find Apollo Pharmacy Near Me", "https://www.google.com/maps/search/Apollo+Pharmacy/", use_container_width=True)
    st.link_button("🟢 Find PharmEasy Outlet Near Me", "https://www.google.com/maps/search/PharmEasy/", use_container_width=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

elif navigation == "About":
    st.markdown('<div class="title-text">About Mediconnect</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="card">
            <h3>ℹ️ Project Information</h3>
            <p>Mediconnect is designed as a quick symptoms-to-medicine suggestion dashboard. It leverages NLP tools and LLMs to assist users in understanding potential health risks and recommendations.</p>
            <br>
            <h4>⚠️ Disclaimer: Medical Warning</h4>
            <p style="color: #b91c1c; font-weight: 600;">
                This app is for educational and general information purposes only and does not constitute professional medical advice, diagnosis, treatment, or recommendations. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
