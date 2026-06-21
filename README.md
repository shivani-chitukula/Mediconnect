# Mediconnect 🩺

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)

Mediconnect is an advanced, LLM-powered healthcare companion and symptom-to-medicine prediction dashboard. It leverages Natural Language Processing (NLP) classifiers for initial clinical condition mapping, and utilizes the **Google Gemini API** for generating non-clinical advice, precautions, and online pharmacy redirect options.

---

## 📈 Key Project Metrics & Performance

- **94.3% Symptom-to-Disease Classification Accuracy**: Developed and benchmarked models in Jupyter Notebooks; the final pipeline uses a **PassiveAggressiveClassifier** combined with a **TF-IDF Trigram Vectorizer** which outperforms standard Naive Bayes (89.8%) and Bigrams (94.2%), securing a peak accuracy of **94.3%** on the validation split.
- **Trained on 11 Common Conditions**: Custom-trained on the highly representative clinical condition instances (e.g., Depression, Anxiety, Insomnia, GERD, Acne, Birth Control, etc.) to ensure high reliability and consistent symptom-to-condition predictions.
- **Multimodal Symptom Capture**: Integrated a real-time **Google Speech Recognition** voice pipeline allowing hands-free symptom recording alongside classical text fields.
- **Fast Search Integration (2s End-to-End Latency)**: Combined Gemini text generation with deep-linking pharmacy integrations (Apollo Pharmacy and PharmEasy) and browser-level client geolocation search queries on Google Maps to find local stores in under 2 seconds.

---

## 📂 Project Structure

```
Mediconnect/
├── app.py                  # Main Streamlit Application
├── dataset/                # Datasets (Ignored on Git due to size limits)
│   ├── drugsComTrain_raw.csv
│   └── drugsComTest_raw.csv
├── models/                 # Pretrained Model Weights (Ignored on Git due to size limits)
│   ├── tfidf_trigrams_model.pkl
│   └── tfidf_vectorizer3.pkl
├── notebooks/              # Jupyter Notebooks for Training & Explorations
│   ├── disease_prediction.ipynb   # NLP Classifiers (Naive Bayes vs PassiveAggressive)
│   ├── medicine-recommend.ipynb   # Recommendation Engine using Cosine Similarity
│   └── notebookc8e8a572b6.ipynb   # Exploratory Data Analysis & Prototypes
├── image.png               # Sidebar Logo Asset
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules for datasets/models
└── README.md               # Project documentation
```

> **Note:** The large dataset CSV files and the 138MB pickle model weights are excluded from Git commits via `.gitignore` to fit within GitHub's 100MB file size limit. Refer to the notebooks inside `notebooks/` to check out how models are processed, tokenized, and serialized.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/shivani-chitukula/Mediconnect.git
cd Mediconnect
```

### 2. Create and activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

*(Note: Voice recognition requires a microphone. If `pyaudio` fails to compile, install it using `pip install pipwin` then `pipwin install pyaudio` on Windows, or get precompiled binary wheels).*

### 4. Configure Gemini API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
Or simply input your key in the Streamlit Sidebar directly when running the application.

---

## 🖥️ Running the Application

Launch the Streamlit app locally:
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

---

## 🧠 Model Training Details (NLP Pipeline)
1. **Preprocessing (NLTK)**: Removes stop words, tokenizes clinical reviews, and lemmatizes symptoms using WordNetLemmatizer and PorterStemmer.
2. **Feature Extraction**: Converts preprocessed reviews into n-gram representation using `TfidfVectorizer` (supports unigram, bigram, and trigram ranges up to `ngram_range=(1,3)`).
3. **Classification**: Fits a **PassiveAggressiveClassifier** to learn online weights, yielding:
   - Multinomial Naive Bayes (Bag of Words): **89.8%**
   - PassiveAggressiveClassifier (Bag of Words): **91.0%**
   - PassiveAggressiveClassifier (TF-IDF Unigrams): **92.3%**
   - PassiveAggressiveClassifier (TF-IDF Bigrams): **94.2%**
   - PassiveAggressiveClassifier (TF-IDF Trigrams): **94.3%** (Selected)

---

## UI Preview

<img width="1548" height="785" alt="image" src="https://github.com/user-attachments/assets/14336cdd-4d3f-477d-91af-887f3e5eddad" />

---

<img width="1527" height="785" alt="image" src="https://github.com/user-attachments/assets/39072b9a-033b-4faf-907f-beac41aeda02" />

---

<img width="737" height="775" alt="image" src="https://github.com/user-attachments/assets/51f91953-d3af-4713-a119-a2e3ebf682e4" />

---

<img width="1538" height="799" alt="image" src="https://github.com/user-attachments/assets/c1839a2a-10ec-462f-9eda-277b215024c7" />

---

<img width="1463" height="795" alt="image" src="https://github.com/user-attachments/assets/c1d1a7e5-bc2b-40b8-92ea-4e6c2a20261a" />

---

<img width="1475" height="822" alt="image" src="https://github.com/user-attachments/assets/6bd8b259-748b-434d-a380-528388f08fbf" />

---


## ⚠️ Disclaimer

This application is for **educational and information purposes only**. It does not provide professional medical advice, diagnosis, or treatment. Always consult a certified healthcare professional before starting any treatment plan.
