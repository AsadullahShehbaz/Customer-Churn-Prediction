# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ================================
# App Config
# ================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ================================
# Custom CSS for Styling
# ================================
st.markdown("""
    <style>
        /* Main background adapts to theme */
        .main {
            background-color: transparent;
        }

        /* Title styling */
        .title {
            font-size: 38px; 
            font-weight: 700; 
            color: #2ecc71; /* Bright green for both modes */
            text-align: center; 
            margin-bottom: 20px;
        }

        /* Subtitle styling */
        .subtitle {
            font-size: 22px; 
            color: #3498db; /* Bright blue for contrast */
            text-align: center; 
            margin-bottom: 30px;
        }

        /* Author box styling */
        .author-box {
            background: linear-gradient(90deg, #ffe066, #fab1a0);
            padding: 15px; 
            border-radius: 15px;
            text-align: center; 
            font-size: 18px;
            font-weight: 500; 
            margin-bottom: 25px;
            color: #2d3436; /* dark text inside */
        }
    </style>
""", unsafe_allow_html=True)


# ================================
# Generate Mock Data
# ================================
@st.cache_data
def load_data(n=10000):
    np.random.seed(42)
    data = pd.DataFrame({
        'CustomerID': np.arange(n),
        'Gender': np.random.choice(['Male', 'Female'], size=n),
        'SeniorCitizen': np.random.choice([0, 1], size=n),
        'Tenure': np.random.randint(1, 72, size=n),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, size=n), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n),
        'Churn': np.random.choice([0, 1], size=n, p=[0.73, 0.27])
    })
    data['TotalCharges'] = (data['Tenure'] * data['MonthlyCharges']).round(2)
    return data

data = load_data()

# ================================
# Sidebar Navigation
# ================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Dashboard", "Model Training", "Prediction","Final Report","About Me"])

# ================================
# 1. Introduction
# ================================
if page == "Introduction":
    st.markdown("<div class='title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Telecom Company — Machine Learning Project</div>", unsafe_allow_html=True)
    st.markdown("<div class='author-box'>👨‍💻 Author: <b>Ali Asadullah Shehbaz</b></div>", unsafe_allow_html=True)

    st.write("""
    ### Problem Overview
    Customer churn is when a customer stops using a company's service.  
    In telecom, high churn directly impacts revenue and growth.  

    ### Project Objective
    - Perform **EDA (Exploratory Data Analysis)**
    - Engineer **features**  
    - Train models (**Logistic Regression, XGBoost**)  
    - Evaluate using **Confusion Matrix, AUC-ROC**  
    - Provide insights & predictions
    """)

    st.success("Use the sidebar to navigate to different sections of the app ✅")
    
# ================================
# 2. EDA
# ================================
elif page == "Dashboard":
    st.header("📊 Customer Churn Analytics Dashboard")

    # ================================
    # KPIs (Key Metrics)
    # ================================
    churn_rate = round(data["Churn"].mean() * 100, 2)
    avg_tenure = round(data["Tenure"].mean(), 1)
    avg_charges = round(data["MonthlyCharges"].mean(), 2)
    total_customers = len(data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate}%")
    col3.metric("Avg Tenure", f"{avg_tenure} months")
    col4.metric("Avg Monthly Charges", f"${avg_charges}")

    # ================================
    # Univariate Analysis
    # ================================
    st.subheader("📈 Univariate Analysis")

    import plotly.express as px

    # Churn Distribution
    fig = px.pie(data, names="Churn", title="Churn Distribution",
                 color="Churn", color_discrete_map={0:"green", 1:"red"})
    st.plotly_chart(fig, use_container_width=True)

    # Gender Distribution
    fig = px.histogram(data, x="Gender", color="Gender",
                       title="Gender Distribution", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # Tenure Distribution
    fig = px.histogram(data, x="Tenure", nbins=40, title="Tenure Distribution",
                       marginal="box", color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig, use_container_width=True)

    # Monthly Charges Distribution
    fig = px.histogram(data, x="MonthlyCharges", nbins=40, title="Monthly Charges Distribution",
                       marginal="violin", color_discrete_sequence=["#ff7f0e"])
    st.plotly_chart(fig, use_container_width=True)

    # ================================
    # Bivariate Analysis
    # ================================
    st.subheader("🔗 Bivariate Analysis")

    # Contract vs Churn
    fig = px.histogram(data, x="Contract", color="Churn", barmode="group",
                       title="Contract Type vs Churn",
                       color_discrete_map={0:"green", 1:"red"})
    st.plotly_chart(fig, use_container_width=True)

    # Payment Method vs Churn
    fig = px.histogram(data, x="PaymentMethod", color="Churn", barmode="group",
                       title="Payment Method vs Churn",
                       color_discrete_map={0:"green", 1:"red"})
    st.plotly_chart(fig, use_container_width=True)

    # Tenure vs Monthly Charges (by Churn)
    fig = px.scatter(data, x="Tenure", y="MonthlyCharges", color="Churn",
                     title="Tenure vs Monthly Charges (by Churn)",
                     color_discrete_map={0:"green", 1:"red"}, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly Charges by Contract Type
    fig = px.box(data, x="Contract", y="MonthlyCharges", color="Contract",
                 title="Monthly Charges by Contract Type")
    st.plotly_chart(fig, use_container_width=True)

    # ================================
    # Correlation Heatmap
    # ================================
    st.subheader("🔍 Correlation Heatmap")

    corr = data[["SeniorCitizen","Tenure","MonthlyCharges","TotalCharges","Churn"]].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu",
                    title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ================================
# 3. Model Training
# ================================
# ================================
# 3. Model Training
# ================================
elif page == "Model Training":
    st.header("⚡ Model Training & Evaluation")

    # Features & Target
    X = pd.get_dummies(data.drop(["CustomerID", "Churn"], axis=1), drop_first=True)
    y = data["Churn"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ================================
    # Available Models
    # ================================
    model_choice = st.selectbox("Select Model", [
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Support Vector Machine (SVM)",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "Naive Bayes",
        "Extra Trees",
        "MLP (Neural Network)"
    ])

    if st.button("Train Model"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:,1]

        elif model_choice == "K-Nearest Neighbors":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:,1]

        elif model_choice == "Support Vector Machine (SVM)":
            from sklearn.svm import SVC
            model = SVC(probability=True)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:,1]

        elif model_choice == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

        elif model_choice == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

        elif model_choice == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

        elif model_choice == "Naive Bayes":
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:,1]

        elif model_choice == "Extra Trees":
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

        elif model_choice == "MLP (Neural Network)":
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:,1]

        # ================================
        # Evaluation
        # ================================
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("ROC Curve & AUC")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax.plot([0,1], [0,1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)


# ================================
# 4. Prediction
# ================================
elif page == "Prediction":
    st.header("🧑‍💻 Predict Customer Churn")

    st.write("Enter customer details below to predict churn:")

    # User Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    monthly = st.slider("Monthly Charges", 20.0, 120.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    # Prepare input
    input_df = pd.DataFrame([{
        "Gender": gender, "SeniorCitizen": senior, "Tenure": tenure,
        "MonthlyCharges": monthly, "Contract": contract, "PaymentMethod": payment
    }])
    input_df["TotalCharges"] = tenure * monthly

    # Match training features
    X = pd.get_dummies(data.drop(["CustomerID", "Churn"], axis=1), drop_first=True)
    scaler = StandardScaler().fit(X)

    input_enc = pd.get_dummies(input_df, drop_first=True).reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_enc)

    # Train a Logistic Regression model (default for prediction)
    model = LogisticRegression()
    y = data["Churn"]
    model.fit(scaler.transform(X), y)

    # Prediction
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if st.button("Predict Churn"):
        if pred == 1:
            st.error(f"⚠️ This customer is likely to churn. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ This customer is likely to stay. (Probability: {prob:.2f})")
# ================================
# 5. Final Report Dashboard
# ================================
elif page == "Final Report":
    st.markdown("<h1 style='text-align:center; color:#2ecc71;'>📊 Customer Churn Prediction – Final Report</h1>", unsafe_allow_html=True)

    # --- Task Summary ---
    with st.container():
        st.subheader("📝 Project Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Project ID", "1")
        col2.metric("Duration", "5 Days")
        col3.metric("Objective", "Predict Churn")

        best_model = "SVM"
        roc_auc = 0.52
        churn_rate = "26.7%"

        # --- KPI Metrics Section ---
        st.markdown("### 📊 Key Performance Indicators")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="🏆 Best Model", value=best_model)

        with col2:
            st.metric(label="📈 ROC AUC", value=f"{roc_auc:.2f}")

        with col3:
            st.metric(label="📉 Churn Rate", value=churn_rate)
        st.info("**Goal:** Build ML models to predict customer churn using telecom data and provide insights.")
    # --- Deliverables Section ---
    with st.expander("📦 Deliverables", expanded=True):
        st.markdown("""
        - 📊 **EDA** – Univariate & Bivariate insights with visualizations  
        - 🛠 **Feature Engineering** – Encoding, scaling, and feature selection  
        - 🔀 **Model Training** – Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost, LightGBM  
        - 📈 **Evaluation** – Confusion Matrix, ROC-AUC, Model Comparison  
        - 📑 **Final Report** – Results & Recommendations  
        """)

    # --- Dataset Description ---
    st.subheader("📂 Dataset Overview")
    st.write("**10,000 customer records with demographics, billing info, contracts, and churn label.**")

    dataset_info = pd.DataFrame({
        "Feature": ["CustomerID", "Gender", "SeniorCitizen", "Tenure", "MonthlyCharges",
                    "TotalCharges", "Contract", "PaymentMethod", "Churn"],
        "Description": [
            "Unique customer identifier",
            "Male / Female",
            "Senior citizen status (1=Yes, 0=No)",
            "Months with company",
            "Monthly bill amount",
            "Total bill since joining",
            "Contract type (Month-to-month / 1yr / 2yr)",
            "Billing method",
            "Target variable (1=Churn, 0=Stay)"
        ]
    })
    st.table(dataset_info)

    # --- EDA Tabs ---
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate", "Correlation"])

    with tab1:
        st.image("https://raw.githubusercontent.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/countplot_gender_distribution.png", caption="Gender Distribution")
        st.image("https://raw.github.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/countplot_seniorcitizen_distribution.png", caption="Senior Citizen Status")
        st.image("https://raw.githubusercontent.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/histplot_tenure_distribution.png", caption="Tenure Distribution")

    with tab2:
        st.image("https://raw.githubusercontent.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/gender_vs_churn.png", caption="Gender vs Churn")
        st.image("https://raw.githubusercontent.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/churn_vs_senior_citizen.png", caption="Senior Citizen vs Churn")
        st.image("https://raw.githubusercontent.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/churn_rate_vs_contract.png", caption="Contract vs Churn")

    with tab3:
        st.image("https://raw.githubusercontent.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/correlation_matrix.png", caption="Correlation Heatmap")

    # --- Feature Engineering ---
    st.subheader("🛠 Feature Engineering Steps")
    st.markdown("""
    1. **Train/Test Split** (stratified to preserve churn ratio)  
    2. **Data Cleaning** (duplicates, leakage check)  
    3. **Derived Features** – Tenure groups, high monthly flag, contract-payment combo  
    4. **Encoding & Scaling** – One-hot encoding, StandardScaler  
    5. **Multicollinearity Removal** – Dropped `TotalCharges`, redundant features  
    6. **Feature Selection** – Contract, Tenure, MonthlyCharges ranked top predictors  
    """)

    # --- Model Comparison ---
    st.subheader("🤖 Model Performance")
    st.image("https://raw.githubusercontent.com/AsadullahShehbaz/Bank-Churn-Prediction-and-Analysis/main/images/barplot%20of%20models.png", caption="Model Comparison (ROC AUC)")

    model_results = pd.DataFrame({
        "Model": ["SVM", "XGBoost", "AdaBoost", "Decision Tree", "Random Forest", "Logistic Regression", "LightGBM"],
        "ROC AUC": [0.5186, 0.5089, 0.5078, 0.5059, 0.5020, 0.4853, 0.4956]
    })
    st.dataframe(model_results.style.highlight_max(axis=0, color="lightgreen"))

    # --- Key Observations ---
    st.subheader("🔍 Key Observations & Next Steps")
    st.warning("""
    - All models performed **close to random guess (~0.5 ROC AUC)**  
    - **SVM** achieved best performance (**0.5186**) but not deployable yet  
    - Dataset lacks strong predictive features  
    - Next: Collect behavioral data, SMOTE/imbalance handling, advanced feature engineering  
    """)

    # --- Conclusion ---
    st.success("✅ **Conclusion:** Current dataset insufficient for production-grade churn prediction. More domain-driven features required.")


elif page == "About Me":
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1>👋 Hi, I'm <span style="color:#4CAF50;">Asadullah Shehbaz</span></h1>
            <h3>🚀 Data Scientist | 🤖 ML Engineer | 📊 AI Enthusiast</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- About Me Section ---
    st.markdown(
        """
        ### 🧠 About Me  
        - 🔭 Current Intern at **[SkilledScore](https://skilledscore.com)**  
        - 🌱 Exploring **LLMs** & **GenAI**  
        - 💬 Skills : `Python`,`EDA` , `Machine Learning`, `Deep Learning`
        - 📫 Reach me at: **asadullahcreative@gmail.com**  
        """,
        unsafe_allow_html=True
    )

    # --- Connect With Me Section ---
    st.markdown("### 🔗 Connect With Me")

    st.markdown(
        """
        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
            <a href="https://www.linkedin.com/in/asadullah-shehbaz-18172a2bb/" target="_blank">
                <img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white">
            </a>
            <a href="https://github.com/AsadullahShehbaz" target="_blank">
                <img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
            </a>
            <a href="https://www.kaggle.com/asadullahcreative" target="_blank">
                <img src="https://img.shields.io/badge/-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white">
            </a>
            <a href="https://web.facebook.com/profile.php?id=61576230402114" target="_blank">
                <img src="https://img.shields.io/badge/-Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white">
            </a>
            <a href="https://www.instagram.com/asad_ullahshehbaz/" target="_blank">
                <img src="https://img.shields.io/badge/-Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

   