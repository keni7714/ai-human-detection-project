# STREAMLIT PROJECT 1 - AI vs HUMAN TEXT DETECTION
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string

# Page Configuration
st.set_page_config(
    page_title="TextSleuth - AI Detection Lab",
    page_icon="🕵🏽‍♂️💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    html, body, [class*="css"]  {
        background-color: #fefefe;
        color: #2e2e2e;
        font-family: 'Segoe UI', sans-serif;
    }

    .main-header {
        font-size: 2.8rem;
        color: #55aa99;
        text-align: center;
        margin-bottom: 2rem;
    }

    .stButton > button {
        background-color: #88e1c1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #55aa99;
    }

    .stTextArea textarea {
        background-color: #f5fff9;
        border: 1px solid #b7e4c7;
        border-radius: 8px;
        padding: 0.75rem;
    }

    .stMetric {
        background-color: #e0f7f1;
        border-left: 5px solid #55aa99;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .ai-prediction {
        background-color: #fff0f3;
        border-left: 5px solid #ff6b81;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .human-prediction {
        background-color: #ebfbee;
        border-left: 5px solid #69db7c;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# === Text Cleaning Components ===

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None): return self

    def transform(self, X):
        return [' '.join([token for token in re.sub(r'[^\w\s]', ' ', text.lower()).split()
                          if len(token) >= 2 and token.isalpha() and token not in self.stop_words])
                for text in X]

class WordNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, method='lemmatization'):
        self.normalizer = PorterStemmer() if method == 'stemming' else WordNetLemmatizer()
        self.method = method

    def fit(self, X, y=None): return self

    def transform(self, X):
        return [' '.join([self.normalizer.stem(token) if self.method == 'stemming'
                          else self.normalizer.lemmatize(token) for token in text.split()])
                for text in X]

# === Model Loading ===
@st.cache_resource
def load_models():
    models = {}
    try:
        models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
        st.success("✅ Loaded tfidf_vectorizer.pkl")

        models['svm'] = joblib.load('models/svm_model.pkl')
        st.success("✅ Loaded svm_model.pkl")

        models['decision_tree'] = joblib.load('models/decision_tree_model.pkl')
        st.success("✅ Loaded decision_tree_model.pkl")

        models['adaboost'] = joblib.load('models/adaboost_model.pkl')
        st.success("✅ Loaded adaboost_model.pkl")

        print("Loaded models:", list(models.keys()))  # Optional logging

        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_choice, models):
    if models is None:
        return None, None

    try:
        # DO NOT CLEAN MANUALLY — just send raw input
        raw_input = [text]

        # Predict
        if model_choice == "svm":
            prediction = models['svm'].predict(raw_input)[0]
            probabilities = models['svm'].predict_proba(raw_input)[0]
        elif model_choice == "decision_tree":
            prediction = models['decision_tree'].predict(raw_input)[0]
            probabilities = models['decision_tree'].predict_proba(raw_input)[0]
        elif model_choice == "adaboost":
            prediction = models['adaboost'].predict(raw_input)[0]
            probabilities = models['adaboost'].predict_proba(raw_input)[0]
        else:
            return None, None

        class_names = ['Human', 'AI']
        prediction_label = class_names[prediction]
        return prediction_label, probabilities

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None


def get_available_models(models):
    """Get list of available models for selection"""
    available = []

    if models is None:
        return available

    if 'svm' in models:
        available.append(("svm", "🧠 Support Vector Machine"))
    if 'decision_tree' in models:
        available.append(("decision_tree", "🌳 Decision Tree"))
    if 'adaboost' in models:
        available.append(("adaboost", "⚡ AdaBoost"))

    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🗺️Navigation🧭")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction🪄", "📚 Batch Processing", "⚖️ Model Comparison 📈", "📊 Model Info 🧪", "🆘 Help"]

)

# Load models
models = load_models()

# ============================================================================  
# HOME PAGE  
# ============================================================================  

if page == "🏠 Home":  
    st.markdown('<h1 class="main-header">🕵🏽‍♀️ TextSleuth - AI Detection Lab 💻</h1>', unsafe_allow_html=True)  

    st.markdown("""  
    Welcome to **TextSleuth**, your interactive web app for detecting whether a piece of writing was created by a human or generated by artificial intelligence.  
    This application uses machine learning models to analyze text and provide predictions along with confidence scores.  
    """)  

    # App overview  
    col1, col2, col3 = st.columns(3)  

    with col1:  
        st.markdown("""  
        ### 🔮 Single Prediction 🪄  
        - Manually enter or paste text  
        - Choose a model  
        - Get instant AI vs Human prediction  
        - View prediction probabilities  
        """)  

    with col2:  
        st.markdown("""  
        ### 📚 Batch Processing  
        - Upload multiple text files (TXT, PDF, DOCX)  
        - Process and predict in bulk  
        - Export results to CSV  
        """)  

    with col3:  
        st.markdown("""  
        ### ⚖️ Model Comparison 📈  
        - Run multiple models on the same input  
        - See where models agree/disagree  
        - Explore performance insights  
        """)  

    # Model status  
    st.subheader("📋 Model Status")  
    if models:  
        st.success("✅ Models loaded successfully!")  

        col1, col2, col3 = st.columns(3)  

        with col1:  
            if models.get('svm'):  
                st.info("🧠 **SVM Model**\n✅ Available")  
            else:  
                st.warning("🧠 **SVM Model**\n❌ Not Available")  

        with col2:  
            if models.get('decision_tree'):  
                st.info("🌳 **Decision Tree**\n✅ Available")  
            else:  
                st.warning("🌳 **Decision Tree**\n❌ Not Available")  

        with col3:  
            if models.get('adaboost'):  
                st.info("⚡ **AdaBoost**\n✅ Available")  
            else:  
                st.warning("⚡ **AdaBoost**\n❌ Not Available")  

        st.markdown("---")  
        if models.get('vectorizer'):  
            st.info("🔤 **TF-IDF Vectorizer**\n✅ Available")  
        else:  
            st.warning("🔤 **TF-IDF Vectorizer**\n❌ Not Available")  

    else:  
        st.error("❌ Models not loaded. Please check model files.")



# ============================================================================
# 🔮 SINGLE PREDICTION PAGE 🪄
# ============================================================================

if page == "🔮 Single Prediction🪄":
    st.header("🔮 AI vs Human Prediction 🪄")
    st.markdown("Enter text below and select a model to detect whether it's AI-generated or human-written.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )

            # Example texts
            with st.expander("💬 Try example inputs"):
                examples = [
                    "The philosophical implications of consciousness continue to challenge modern neuroscience.",
                    "Large language models demonstrate remarkable generalization across varied NLP tasks.",
                    "I had a great time at the concert last night — it was electrifying!",
                    "The model exhibits limitations in maintaining factual consistency across multi-hop questions.",
                    "Sometimes I just sit and write poetry to make sense of my emotions."
                ]

                col1, col2 = st.columns(2)
                for i, example in enumerate(examples):
                    with col1 if i % 2 == 0 else col2:
                        if st.button(f"Example {i+1}", key=f"example_{i}"):
                            st.session_state.user_input = example

            # Use example input if available
            default_text = st.session_state.get("user_input", "")

            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                value=default_text,
                placeholder="Paste or write your paragraph here...",
                height=150
            )

            # Update session if user types something new
            st.session_state.user_input = user_input

            # Character count
            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")

            # Predict
            if st.button("🧠 Run Detection", type="primary"):
                if user_input.strip():
                    with st.spinner('Analyzing text...'):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)

                        if prediction and probabilities is not None:
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                if prediction.lower() == "ai":
                                    st.warning(f"🤖 Prediction: **AI-Generated Text**")
                                else:
                                    st.success(f"🧑‍💼 Prediction: **Human-Written Text**")

                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")

                            # Probability summary
                            st.subheader("📊 Prediction Confidence")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🧑 Human", f"{probabilities[0]:.1%}")
                            with col2:
                                st.metric("🤖 AI", f"{probabilities[1]:.1%}")

                            # Horizontal bar chart
                            prob_df = pd.DataFrame({
                                'Class': ['Human', 'AI'],
                                'Probability': probabilities
                            })

                            import altair as alt
                            chart = alt.Chart(prob_df).mark_bar().encode(
                                x=alt.X('Probability:Q', axis=alt.Axis(format='%')),
                                y=alt.Y('Class:N', sort='-x'),
                                color=alt.Color('Class:N', legend=None)
                            ).properties(height=120)

                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.error("Prediction failed. Please check your model setup.")
                else:
                    st.warning("Please enter some text to classify!")
        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")


# ============================================================================  
# 📚 BATCH PROCESSING PAGE  
# ============================================================================  

elif page == "📚 Batch Processing":  
    st.header("📚 Batch Processing Mode")  
    st.markdown("Upload a text or CSV file to detect whether each entry is **AI-generated** or **human-written**.")  

    if models:  
        available_models = get_available_models(models)  

        if available_models:  
            uploaded_file = st.file_uploader(  
                "📤 Upload File:",  
                type=['txt', 'csv'],  
                help="Upload .txt (one entry per line) or .csv (text in first column)"  
            )  

            if uploaded_file:  
                model_choice = st.selectbox(  
                    "📌 Select Model:",  
                    options=[model[0] for model in available_models],  
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)  
                )  

                if st.button("📊 Run Batch Detection"):  
                    try:  
                        if uploaded_file.type == "text/plain":  
                            content = uploaded_file.read().decode("utf-8")  
                            texts = [line.strip() for line in content.split('\n') if line.strip()]  
                        else:  
                            df = pd.read_csv(uploaded_file)  
                            texts = df.iloc[:, 0].astype(str).tolist()  

                        if not texts:  
                            st.error("No valid text entries found in the uploaded file.")  
                        else:  
                            st.info(f"Processing {len(texts)} entries...")  
                            results = []  
                            progress = st.progress(0)  

                            for i, text in enumerate(texts):  
                                prediction, probabilities = make_prediction(text, model_choice, models)  
                                if prediction:  
                                    results.append({  
                                        'Short Text': text[:100] + "..." if len(text) > 100 else text,  
                                        'Prediction': prediction,  
                                        'Confidence': f"{max(probabilities):.1%}",  
                                        'Human %': f"{probabilities[0]:.1%}",  
                                        'AI %': f"{probabilities[1]:.1%}"  
                                    })  
                                progress.progress((i + 1) / len(texts))  

                            if results:  
                                st.success("✅ Batch processing complete!")  
                                result_df = pd.DataFrame(results)  

                                st.subheader("📊 Summary")  
                                col1, col2, col3 = st.columns(3)  
                                ai_count = sum(r['Prediction'] == 'AI' for r in results)  
                                human_count = len(results) - ai_count  
                                avg_conf = np.mean([float(r['Confidence'].strip('%')) for r in results])  

                                col1.metric("🧾 Total", len(results))  
                                col2.metric("🧑 Human", human_count)  
                                col3.metric("🤖 AI", ai_count)  
                                st.metric("📈 Avg Confidence", f"{avg_conf:.1f}%")  

                                st.dataframe(result_df, use_container_width=True)  

                                st.download_button(  
                                    label="📥 Download Results",  
                                    data=result_df.to_csv(index=False).encode('utf-8'),  
                                    file_name="ai_human_predictions.csv",  
                                    mime='text/csv'  
                                )  
                            else:  
                                st.error("No predictions were generated.")  
                    except Exception as e:  
                        st.error(f"❌ Error processing file: {e}")  
            else:  
                st.info("Upload a file above to begin batch processing.")  

                with st.expander("📄 Supported Formats"):  
                    st.markdown("""  
                    **Text File (.txt)**  
                    ```
                    The sky is blue today.
                    AI models are trained using large datasets.
                    ```  

                    **CSV File (.csv)**  
                    ```
                    text
                    The sun rises in the east.
                    Reinforcement learning requires feedback.
                    ```  
                    """)  
        else:  
            st.error("❌ No models available for batch prediction.")  
    else:  
        st.warning("Models not loaded. Please check the model files.")  


# ============================================================================
# ⚖️ MODEL COMPARISON PAGE 📈
# ============================================================================

elif page == "⚖️ Model Comparison 📈":
    st.header("⚖️ Compare Models Side-by-Side 📈")
    st.markdown("See how different models classify the **same text** and view their prediction confidence.")

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            input_text = st.text_area("✏️ Enter text for comparison:", placeholder="Type or paste a paragraph here...", height=120)

            if st.button("📊 Compare Models"):
                if input_text.strip():
                    results = []

                    for key, name in available_models:
                        try:
                            # Use pipeline to predict (no manual vectorization)
                            pred = models[key].predict([input_text])[0]
                            probs = models[key].predict_proba([input_text])[0]

                            label = "Human" if pred == 0 else "AI"

                            results.append({
                                "Model": name,
                                "Prediction": label,
                                "Confidence": f"{max(probs):.1%}",
                                "Human": probs[0],
                                "AI": probs[1]
                            })

                        except Exception as e:
                            st.error(f"❌ {name} failed: {e}")

                    if results:
                        df = pd.DataFrame(results)
                        st.subheader("📋 Model Predictions")
                        st.dataframe(df[['Model', 'Prediction', 'Confidence']], use_container_width=True)

                        st.subheader("📈 Probability Distribution")
                        chart_cols = st.columns(len(results))
                        for i, res in enumerate(results):
                            with chart_cols[i]:
                                st.markdown(f"**{res['Model']}**")
                                chart_data = pd.DataFrame({
                                    'Class': ['Human', 'AI'],
                                    'Probability': [res['Human'], res['AI']]
                                })
                                st.bar_chart(chart_data.set_index("Class"))

                        # Agreement Check
                        predictions = set(r["Prediction"] for r in results)
                        if len(predictions) == 1:
                            st.success(f"✅ All models agree: **{predictions.pop()}**")
                        else:
                            st.warning("⚠️ Models gave **different predictions**")
                            for res in results:
                                st.write(f"- **{res['Model']}**: {res['Prediction']}")
                    else:
                        st.error("No predictions returned.")
                else:
                    st.warning("⚠️ Please enter some text to compare.")
        elif len(available_models) == 1:
            st.info("You need at least two models for comparison. Only one is loaded.")
        else:
            st.error("No models available.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================  
# 📦 MODEL INFO PAGE 🧪  
# ============================================================================  

elif page == "📊 Model Info 🧪":  
    st.header("📦 Model Overview 🧪")  

    if models:  
        st.success("✅ Models are loaded and ready!")  

        # Model Details  
        st.subheader("🧠 Available Models")  
        col1, col2 = st.columns(2)  

        with col1:  
            st.markdown("""  
            ### 🧠 Support Vector Machine  
            **Type:** Margin-Based Classifier  
            **Kernel:** Linear  
            **Features:** TF-IDF Vectorized Text  

            **Strengths:**  
            - High accuracy with text  
            - Effective in high-dimensional spaces  
            - Works well with small/medium datasets  
            """)  

        with col2:  
            st.markdown("""  
            ### 🌳 Decision Tree  
            **Type:** Rule-Based Tree Model  
            **Splitting Criterion:** Gini Impurity  
            **Features:** TF-IDF Vectorized Text  

            **Strengths:**  
            - Interpretable model structure  
            - Captures non-linear patterns  
            - Fast prediction time  
            """)  

        st.markdown("""  
        ### ⚡ AdaBoost  
        **Type:** Ensemble Learning (Boosting)  
        **Base Estimator:** Decision Stumps  
        **Features:** TF-IDF Vectorized Text  

        **Strengths:**  
        - Combines weak learners into a strong one  
        - Good generalization  
        - Handles bias-variance tradeoff  
        """)  

        # Feature Engineering Info  
        st.subheader("🔤 Feature Engineering")  
        st.markdown("""  
        **Text Representation:** TF-IDF (Term Frequency-Inverse Document Frequency)  
        - **Max Features:** Top 5,000 terms  
        - **N-grams:** Unigrams and Bigrams  
        - **Stop Words Removed:** Yes (English)  
        - **Minimum Document Frequency:** 2  
        """)  

        # File Status  
        st.subheader("📁 Model Files Status")  
        file_status = []  

        files_to_check = [  
            ("models/tfidf_vectorizer.pkl", "TF-IDF Vectorizer", 'vectorizer' in models),  
            ("models/svm_model.pkl", "SVM Classifier", 'svm' in models),  
            ("models/decision_tree_model.pkl", "Decision Tree Classifier", 'decision_tree' in models),  
            ("models/adaboost_model.pkl", "AdaBoost Classifier", 'adaboost' in models)  
        ]  

        for filename, description, is_loaded in files_to_check:  
            file_status.append({  
                "File": filename,  
                "Description": description,  
                "Status": "✅ Loaded" if is_loaded else "❌ Not Found"  
            })  

        st.table(pd.DataFrame(file_status))  

        # Training Information  
        st.subheader("📚 Model Training Info")  
        st.markdown("""  
        **Dataset:** AI vs Human Text Classification  
        - **Classes:** Human-Written (0) vs AI-Generated (1)  
        - **Preprocessing:** Lowercasing, punctuation removal, stop word removal, normalization  
        - **Models Trained Separately:** All models trained using same features for fair evaluation  
        """)  
    else:  
        st.warning("⚠️ Models not loaded. Please check the 'models/' directory and restart the app.")  


# ============================================================================  
# 🆘 HELP PAGE  
# ============================================================================  

elif page == "🆘 Help":  
    st.header("🆘 Help & Instructions")  

    with st.expander("🔮 Single Prediction 🪄"):  
        st.write("""  
        1. **Select a model** (SVM, Decision Tree, or AdaBoost)  
        2. **Paste your text** into the input box  
        3. **Click 'Run Detection'**  
        4. **View prediction**, confidence level, and class breakdown  
        5. Use sample inputs to test various examples  
        """)  

    with st.expander("📚 Batch Processing"):  
        st.write("""  
        1. Prepare a `.txt` (one text per line) or `.csv` (text in first column)  
        2. **Upload the file**  
        3. **Choose a model**  
        4. **Click 'Process File'**  
        5. **Download the results** with predictions and confidence levels  
        """)  

    with st.expander("⚖️ Model Comparison 📈"):  
        st.write("""  
        1. Enter one text sample  
        2. Click **Compare Models**  
        3. See predictions from each model and confidence %  
        4. View probability breakdowns in side-by-side charts  
        """)  

    with st.expander("🔧 Troubleshooting"):  
        st.write("""  
        **Common Issues:**  

        - **Models not loading:**  
          - Ensure all `.pkl` files are inside the `models/` directory  

        - **Prediction failed:**  
          - Check that your input text is not empty  
          - Make sure your vectorizer and models are trained with the same preprocessing  

        - **File upload issues:**  
          - Use `.txt` or `.csv` formats only  
          - Ensure text is in the first column if using `.csv`  
        """)  

   # Project Info  
st.subheader("📁 Project Structure Overview")  
st.code("""  
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
       
""")


# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**AI vs Human Text Detection App**  
Built with Streamlit

**Models:**  
- 🧠 Support Vector Machine  
- 🌳 Decision Tree  
- ⚡ AdaBoost  

**Framework:** scikit-learn  
**Deployment:** Streamlit Local / Cloud Ready  
**Version:** v1.0  
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; font-size: 0.95rem;'>
    Built with ❤️ using <strong>Streamlit</strong> | AI vs Human Text Detection App<br>
    <strong>By Keni Omorojie</strong><br>
    Part of the course series: <em>Introduction to Large Language Models & AI Agents</em><br>
    <br>
    <small>✅ Certified Project Submission for Project 1 | Summer 2025</small><br>
    <small>🔗 <a href="https://github.com/keni7714/ai-human-detection-project.git" target="_blank">View Source on GitHub</a></small>

</div>
""", unsafe_allow_html=True)
