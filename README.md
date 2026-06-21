# Mediconnect 🩺

Mediconnect is an NLP-powered medical symptom checker and recommendation dashboard built with **Streamlit** and integrated with **Google Gemini**. It predicts potential clinical conditions based on symptoms described through text or voice, recommends relevant drugs from clinical databases, suggests health precautions, and provides pharmacy links.

---

## 🚀 Features

- **Text Symptom Checker**: Describe symptoms in plain text to predict the underlying disease.
- **Voice Symptom Checker**: Record and translate spoken symptoms directly into the classifier using Google Speech Recognition.
- **AI Precautions & Medicines**: Integrates the **Google Gemini API** to suggest 3-4 health precautions and recommend 2-3 specific tablet names.
- **Direct purchase integrations**: Generates direct search links for Apollo Pharmacy and PharmEasy for each recommended tablet.
- **Interactive Store Map Locator**: Direct links to map out General Pharmacies, Apollo Pharmacy, or PharmEasy outlets near the user.
- **Premium Responsive UI**: Curated theme, custom clean cards, styled inputs, and sleek animations.

---

## 📂 Project Structure

```
Mediconnect/
├── app.py                  # Main Streamlit Application
├── dataset/                # Datasets (Ignored on Git due to size)
│   ├── drugsComTrain_raw.csv
│   └── drugsComTest_raw.csv
├── models/                 # Pretrained Model Weights (Ignored on Git due to size)
│   ├── tfidf_trigrams_model.pkl
│   └── tfidf_vectorizer3.pkl
├── notebooks/              # Jupyter notebooks for model training
│   ├── disease_prediction.ipynb
│   ├── medicine-recommend.ipynb
│   └── notebookc8e8a572b6.ipynb
├── image.png               # Sidebar Logo Asset
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules for datasets/models
└── README.md               # Project documentation
```

> **Note:** The datasets and trained `.pkl` models are excluded from this Git repository to stay within GitHub's 100MB file limit. Refer to the Notebooks folder to see how models are trained and how datasets are retrieved.

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

*(Note: Voice recognition requires a microphone. If `pyaudio` fails to install on Windows, you can install it using `pip install pipwin` then `pipwin install pyaudio`, or install the precompiled wheels from Gohlke's page.)*

### 4. Configure Gemini API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
Alternatively, you can input your key in the Streamlit Sidebar directly when the application is running.

---

## 🖥️ Running the Application

Start the Streamlit development server locally:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ⚠️ Disclaimer

This application is for **educational and information purposes only**. It does not provide professional medical advice, diagnosis, or treatment. Always consult a certified healthcare professional before starting any treatment plan.
