import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Credit Rating Application",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
.main { background-color: #f4f6f9; }

h1 { text-align: center; color: #4B0082; }

.metric-box {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 16px;
}

.accuracy {background-color: #1abc9c;}
.precision {background-color: #3498db;}
.recall {background-color: #9b59b6;}
.f1 {background-color: #e67e22;}
.auc {background-color: #2ecc71;}
.mcc {background-color: #e74c3c;}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("<h1>💳 Credit Rating Demonstration Application</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
upld_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if upld_file is not None:

    df = pd.read_csv(upld_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("---")

    # Map target: good=0, bad=1
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    
    # ---------------------------
    # CHECK IF TARGET EXISTS
    # ---------------------------
    if "class" in df.columns:
        X = df.drop("class", axis=1)
        y_true = df["class"]
        has_target = True
    else:
        X = df.copy()
        has_target = False

    # ---------------------------
    # LOAD MODELS
    # ---------------------------
    log_model = joblib.load("model/logistic_regression.joblib")
    dt_model = joblib.load("model/decision_tree.joblib")
    rf_model = joblib.load("model/random_forest.joblib")
    knn_model = joblib.load("model/knn.joblib")
    nb_model = joblib.load("model/naive_bayes.joblib")
    xg_model = joblib.load("model/xgboost.joblib")
    preprocessor = joblib.load("model/preprocessor.joblib")

    # ---------------------------
    # MODEL SELECTION
    # ---------------------------
    st.subheader("🤖 Select Model")

    model_name = st.selectbox(
        "Choose a classification model",
        ("Logistic Regression", "Decision Tree", "Random Forest",
         "kNN", "XGBoost", "Naive Bayes")
    )

    model_dict = {
        "Logistic Regression": log_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "kNN": knn_model,
        "XGBoost": xg_model,
        "Naive Bayes": nb_model
    }

    model = model_dict[model_name]

    # ---------------------------
    # PREPROCESS + PREDICT
    # ---------------------------
    X_processed = preprocessor.transform(X)
    y_pred = model.predict(X_processed)
    y_prob = model.predict_proba(X_processed)[:, 1]  
    #df["Prediction"] = y_pred

    #st.subheader("🔮 Model Predictions")
    #st.dataframe(df, use_container_width=True)

  
    #st.markdown("---")
    #st.subheader("📊 Stored Model Performance")

    #with open("model_metrics.json", "r") as f:
    #    metrics = json.load(f)

    #model_metrics = metrics[model_name]

    #accuracy = model_metrics["accuracy"]
    #recall = model_metrics["recall"]
    #precision = model_metrics["precision"]
    #f1 = model_metrics["f1"]
    #auc = model_metrics["auc"]
    #mcc = model_metrics["mcc"]

   
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.markdown(f'<div class="metric-box accuracy">Accuracy<br>{accuracy:.3f}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-box precision">Precision<br>{precision:.3f}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-box recall">Recall<br>{recall:.3f}</div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-box f1">F1 Score<br>{f1:.3f}</div>', unsafe_allow_html=True)
    col5.markdown(f'<div class="metric-box auc">AUC<br>{auc:.3f}</div>', unsafe_allow_html=True)
    col6.markdown(f'<div class="metric-box mcc">MCC<br>{mcc:.3f}</div>', unsafe_allow_html=True)

    # ---------------------------
    # CONFUSION MATRIX (Only if target exists)
    # ---------------------------
    if has_target:
        st.markdown("---")
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(3,3))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap="Purples", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # ---------------------------
        # CLASSIFICATION REPORT
        # ---------------------------
        #st.markdown("---")
        #st.subheader("Classification Report")
        #report = classification_report(y_true, y_pred)
        #st.code(report)

else:
    st.info("⬆️ Please upload a CSV file to start the application.")