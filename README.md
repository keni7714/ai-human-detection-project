# 🕵🏽‍♀️ TextSleuth - AI vs Human Text Detection

This Streamlit web app detects whether a given piece of text is AI-generated or human-written using machine learning models. It was built for the course *Introduction to Large Language Models / Intro to AI Agents*.

---

## 📊 Features

* Support Vector Machine (SVM), Decision Tree, and AdaBoost classifiers
* TF-IDF vectorizer for feature extraction
* Preprocessing pipeline with stopword removal
* Interactive UI with single prediction, batch processing, and model comparison
* Visual confidence charts and prediction summaries

---

## ⚙️ Setup Instructions

### Prerequisites

* Python 3.8+
* Streamlit

### Installation

```bash
git clone https://github.com/keni7714/ai-human-detection-project.git
cd ai-human-detection-project
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn nltk
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
ai_human_detection_project/  
├── app.py                  # Main Streamlit App  
├── models/                 # Trained Models  
│   ├── tfidf_vectorizer.pkl  
│   ├── svm_model.pkl  
│   ├── decision_tree_model.pkl  
│   └── adaboost_model.pkl  
├── sample_data/           # Optional sample inputs  
│   ├── sample_texts.txt  
│   └── sample_data.csv  
├── notebooks/             # Jupyter notebooks (training, EDA, testing)  
│   └── Keni_Omorojie_assignment2.ipynb  
├── README                  
└── requirements.txt       # Dependencies  
```

---

## 👤 Author

**Keni Omorojie**
📧 [keniomorojie@gmail.com](mailto:keniomorojie@gmail.com)
🐙 [github.com/keni7714](https://github.com/keni7714)

---

## 💬 Notes
This app was developed as part of a course assignment. Future updates may include additional models and improved UI/UX.