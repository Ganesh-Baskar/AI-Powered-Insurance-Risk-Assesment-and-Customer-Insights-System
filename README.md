# AI-Powered-Insurance-Risk-Assesment-and-Customer-Insights-System

A dynamic, intelligent dashboard built with Streamlit, seamlessly integrating machine learning, natural language processing (NLP), and data visualization to convert raw insurance data into strategic business intelligence.

📌 Executive Overview

This end-to-end AI solution empowers insurance providers by automating critical decisions and enhancing risk visibility. It covers everything from risk profiling and claim forecasting to fraud detection, sentiment analysis, and document simplification — delivering actionable insights at every step of the customer lifecycle.

📊 Exploratory Data Analysis Highlights

Claim Amount Distribution: Right-skewed, indicating few high-value outliers.

Fraud Analysis: Strong class imbalance observed, requiring corrective techniques.

Policy Trends: "Comprehensive" emerges as the dominant policy type.

Gender Distribution: Slight male skew but relatively balanced.

Income vs. Claim Amount: Weak linear relationship.

Correlation Matrix: Notable link between Claim_Amount and Premium_Amount.

🧪 Model Training & Evaluation

1️⃣ Risk Score Classification

Algorithms: Logistic Regression, Random Forest, XGBoost

Top Accuracy: 89.4% using Random Forest

Skillset: Multi-class classification, feature engineering, cross-validation

2️⃣ Claim Amount Regression

Algorithms: Linear Regression, Random Forest, XGBoost, LightGBM

Best RMSE: ~150 (Random Forest)

Skillset: Regression analysis, metric tuning, data transformation

3️⃣ Fraud Detection

Algorithms: Random Forest, Logistic Regression, XGBoost, LightGBM

Best Accuracy: 93.5% using LightGBM

Skillset: SMOTE balancing, ensemble techniques, diagnostic matrices

4️⃣ Sentiment Analysis

Tools Used: TextBlob, NLTK

Model: Naive Bayes — 90% Accuracy

Skillset: Text cleansing, polarity analysis, customer feedback mining

5️⃣ Document Translation & Summarization

Tools: Google Translate API, rule-based summarizer

Output: Multilingual policy PDFs

Skillset: PDF parsing, multilingual NLP workflows, summarization logic

6️⃣ Multilingual FAQ Chatbot

Techniques: Rule-based with language detection

Capabilities: Supports Indian regional languages

Skillset: Intent recognition, language handling, conversational design

Sure! Here's a cleaner and more structured rewrite of your project overview that emphasizes clarity and professionalism, while preserving all the rich detail you've packed in:

---

# 🛡️ Insurance AI Project Overview

## ⚙️ Modules, Challenges & Solutions

| Module              | Challenge                             | Solution                                                                 |
|---------------------|----------------------------------------|--------------------------------------------------------------------------|
| **Risk Scoring**     | Class overlap, imbalance              | Stratified cross-validation, class weighting                             |
| **Claim Prediction** | Outliers, skewed data                 | Log transformation, Random Forest to model non-linear relationships      |
| **Fraud Detection**  | Minority class detection              | SMOTE + ensemble methods (LightGBM, XGBoost)                              |
| **Sentiment Analysis** | Sarcasm, multilingual content       | Rule-based polarity, text cleaning, fallback language detection          |
| **Translation**      | PDF formatting issues                 | PyMuPDF + ReportLab for custom rendering                                 |
| **Chatbot**          | Rule-based limitations, edge cases    | Intent mapping, multilingual fallback logic                              |

---

## 🔍 Customer Segmentation

- **Clustering Technique:** DBSCAN  
- **Input Features:** Age, Annual Income, Policy Type, Claim Amount  
- **Objective:** Personalized marketing & premium pricing strategy  
- **Output:** Grouped customers by similar risk behavior profiles  

---

## 💡 Future Enhancements

- Deep learning for advanced sentiment & fraud detection  
- Voice interaction with chatbot via speech-to-text  
- Real-time dashboards and analytics refresh  
- User authentication and report archiving  
- Cloud integration (AWS/GCP)

---

## 🗂️ Project Structure

```
Insurance_AI_Project
├── 📂 dataset           # Raw & processed datasets
├── 📂 notebooks       # EDA, ML, NLP experiments
├── 📂 models          # Saved models (Pickle, ONNX, TensorFlow, PyTorch)
├── 📂 scripts         # Data preprocessing & model training
├── 📂 deployment      # APIs (Flask/FastAPI), Docker, Streamlit UI
├── 📂 reports         # Documentation & analysis reports
├── 📜 README.md       # Setup guide & project summary
├── 📜 requirements.txt# Dependencies used
└── 📜 app.py          # Deployment entry point
```

---

## 🔧 Skills Demonstrated

- Machine Learning: Classification, Regression, Clustering  
- NLP: Sentiment Analysis, Translation, Conversational AI  
- Visualization: EDA, interpretability, reporting  
- Model Tuning & Evaluation  
- Streamlit App Development  
- PDF rendering & file handling

---

## 📎 Appendices

### ✨ Sample Features

| Feature          | Type       | Description                          |
|------------------|------------|--------------------------------------|
| Age              | Numeric    | Age of policyholder                  |
| Gender           | Categorical| Male / Female                        |
| Annual Income    | Numeric    | Reported yearly income               |
| Policy_Type      | Categorical| Type of insurance policy             |
| Claim_Amount     | Numeric    | Claim value in currency              |
| Fraudulent_Claim | Binary     | True/False indicator of fraud        |

### 💻 Model Snippet

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

---

## 📚 References

- [Scikit-learn](https://scikit-learn.org/)
- [Analytics Vidhya](https://www.analyticsvidhya.com/)
- [Google Translate API](https://cloud.google.com/translate/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [TextBlob](https://textblob.readthedocs.io/)

