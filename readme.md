# 📊 Internship Task Report  

**Task Title:** Customer Churn Prediction for a Telecom Company  
**Internship Duration:** 5 Days  
**Intern:** Data Science  
**Date:** 11 August 2025  

---

# 1. **Task Overview**
The goal of this task was to predict **customer churn** for a telecom company using historical customer data.  
Identifying customers likely to churn enables proactive retention strategies, improving both **customer loyalty** and **company revenue**.

---

# 2. **Dataset Description**
The dataset contains **10,000 customer records** with the following features:

| Feature         | Description |
|-----------------|-------------|
| `CustomerID`    | Unique identifier for each customer |
| `Gender`        | Male or Female |
| `SeniorCitizen` | Senior citizen status (1 = Yes, 0 = No) |
| `Tenure`        | Months the customer has stayed with the company |
| `MonthlyCharges`| Amount charged per month |
| `TotalCharges`  | Total amount charged over the tenure |
| `Contract`      | Contract type: Month-to-month, One year, Two year |
| `PaymentMethod` | Payment method used by the customer |
| `Churn`         | Target variable (1 = Churned, 0 = Stayed) |

---

# 3. **Exploratory Data Analysis (EDA)**

### 3.1 **Univariate Analysis**
#### **Gender Distribution**
- Male: **5,013** customers (50.13%)  
- Female: **4,987** customers (49.87%)  
**Observation:** The dataset is **gender balanced**, minimizing potential gender bias in model training.

#### **Senior Citizen Distribution**
- Non-Senior (0): **5,007** customers (50.07%)  
- Senior (1): **4,993** customers (49.93%)  
**Observation:** The dataset is balanced by age group, with minimal age-related bias.

#### **Tenure Summary**
| Metric  | Value |
|---------|-------|
| Count   | 10,000 |
| Mean    | 35.96 months |
| Std Dev | 20.50 months |
| Min     | 1 month |
| Q1      | 18 months |
| Median  | 36 months |
| Q3      | 54 months |
| Max     | 71 months |
**Observation:** Tenure is widely spread, indicating steady acquisition and churn patterns.

#### **Monthly Charges Summary**
| Metric  | Value |
|---------|-------|
| Count   | 10,000 |
| Mean    | $70.45 |
| Std Dev | $28.94 |
| Min     | $20.00 |
| Q1      | $45.53 |
| Median  | $70.59 |
| Q3      | $95.61 |
| Max     | $120.00 |
**Observation:** Charges are evenly distributed, with most customers paying around **$70/month**.

#### **Contract Distribution**
- Month-to-month: **32.19%**  
- One year: **34.55%**  
- Two year: **33.26%**  
**Observation:** Contract types are **evenly distributed**, useful for unbiased contract-based churn analysis.

#### **Payment Method Distribution**
- Electronic check: **25.16%**  
- Bank transfer: **25.08%**  
- Credit card: **24.92%**  
- Mailed check: **24.84%**  
**Observation:** Payment preferences are **balanced**, so differences in churn are likely service-related.

#### **Churn Distribution**
- Non-Churn (0): **73.30%**  
- Churn (1): **26.70%**  
**Observation:** The dataset is **imbalanced**, requiring resampling or class weighting in modeling.

#### **Total Charges Summary**
| Metric  | Value |
|---------|-------|
| Count   | 10,000 |
| Mean    | $2,541.81 |
| Std Dev | $1,879.65 |
| Min     | $21.20 |
| Q1      | $1,035.06 |
| Median  | $2,117.14 |
| Q3      | $3,717.35 |
| Max     | $8,384.39 |
**Observation:** As expected, **Total Charges** are highly dependent on **Tenure** and **Monthly Charges**.

---

### 3.2 **Bivariate Analysis**

#### **Gender vs Churn**
| Gender | Non-Churn (0) | Churn (1) |
|--------|--------------:|----------:|
| Female | 3,657 | 1,330 |
| Male   | 3,673 | 1,340 |
**Observation:** Gender has **no significant impact** on churn.

#### **Senior Citizen vs Churn**
| Senior Citizen | Non-Churn (0) | Churn (1) |
|----------------|--------------:|----------:|
| 0 | 3,680 | 1,327 |
| 1 | 3,650 | 1,343 |
**Observation:** Senior status does not strongly influence churn.

#### **Contract Type vs Churn**
| Contract Type  | Non-Churn (0) | Churn (1) |
|----------------|--------------:|----------:|
| Month-to-month | 2,373 | 846 |
| One year       | 2,517 | 938 |
| Two year       | 2,440 | 886 |
**Observation:** Churn rates are similar across contracts, with **One-year contracts** having slightly higher churn.

#### **Payment Method vs Churn**
| Payment Method   | Non-Churn (0) | Churn (1) |
|------------------|--------------:|----------:|
| Bank transfer    | 1,836 | 672 |
| Credit card      | 1,817 | 675 |
| Electronic check | 1,845 | 671 |
| Mailed check     | 1,832 | 652 |
**Observation:** Churn distribution is uniform across payment methods.

---

### 3.3 **Monthly Charges vs Churn**
- Churn exists across **all charge ranges ($20–$120)**.  
- No clear trend in raw values, suggesting binning into **Low / Medium / High** ranges may reveal patterns.  

---

### 3.4 **Contract & Payment Method vs Churn Rate**
| Contract Type    | Payment Method   | Churn % |
|------------------|------------------|--------:|
| Month-to-month   | Bank transfer    | 27.78% |
| Month-to-month   | Credit card      | 27.18% |
| Month-to-month   | Electronic check | 24.64% |
| Month-to-month   | Mailed check     | 25.43% |
| One year         | Bank transfer    | 28.26% |
| One year         | Credit card      | 27.34% |
| One year         | Electronic check | 26.21% |
| One year         | Mailed check     | 26.80% |
| Two year         | Bank transfer    | 24.27% |
| Two year         | Credit card      | 26.74% |
| Two year         | Electronic check | 28.95% |
| Two year         | Mailed check     | 26.49% |
**Observation:**  
- Churn ranges **24%–29%** across groups.  
- Slightly higher churn in **month-to-month & one-year** contracts compared to **two-year** contracts.  
- Payment method impact is **minimal**.

---

## 4. **Correlation Insights**
- **Total Charges** ↔ **Tenure**: **77% correlation** (longer tenure → higher total charges).  
- **Total Charges** ↔ **Monthly Charges**: **56% correlation** (expected due to formula).  

---

# 3.Feature Engineering 

## **Objective**
The goal of the feature engineering phase was to transform raw telecom customer data into a clean, informative, and model-ready format while preventing data leakage and preserving predictive signal.

---

## **Steps Performed**

### **1. Train/Test Split**
- **Method:** Stratified split on the target variable (`Churn`) to maintain the churn ratio in both training and test sets.
- **Reason:** Prevents leakage during feature engineering and ensures fair evaluation.

---

### **2. Data Cleaning & Validation**
- Removed duplicate records (none found).
- Verified and corrected data types where necessary.
- Ensured no leakage-prone features (e.g., post-churn calculated metrics).

---

### **3. Creation of Derived Features**
- **`TenureGroup`** — Binned customers into ranges (0–12, 13–24, etc.) to capture non-linear churn patterns.
- **`HighMonthlyFlag`** — Binary indicator for monthly charges above the median.
- **`AutoPay`** — Indicator for automated payment methods (`Bank transfer` / `Credit card`).
- **`Contract_Payment`** — Combined feature from contract type and payment method.

---

### **4. Encoding of Categorical Variables**
- **One-Hot Encoding:** Applied to nominal categorical features (`Contract`, `PaymentMethod`, `Gender`,`SeniorCitizen`, `HighMonthlyFlag`, `AutoPay`).
- **K-Fold Target Encoding:** Applied to `Contract_Payment` **within training data only** to avoid leakage.

---

### **5. Scaling of Numeric Variables**
- Applied `StandardScaler` to `Tenure`, `MonthlyCharges`, and other numeric columns for normalization, benefiting algorithms like Logistic Regression.

---

### **6. Multicollinearity Check (VIF)**
- Found **high VIF (>5)** for:
  - `Tenure` → VIF ≈ 6.99
  - `MonthlyCharges` → ∞
  - `TotalCharges` → 10.17
  - `AvgChargesPerMonth` → ∞
- **Decision:** Kept `Tenure` and `MonthlyCharges` (core features), dropped `TotalCharges` and `AvgChargesPerMonth` due to redundancy.

---

### **7. Feature Selection**
- Used `RandomForest` + `SelectFromModel` to assess importance of engineered features.
- Confirmed `Contract`, `Tenure`, and `MonthlyCharges` as top predictors.

---

## **Key Outcomes**
- Prepared **model-ready dataset** with no leakage and reduced redundancy.
- **Multicollinearity addressed** by removing overlapping variables.
- Standardized **categorical encoding** for model input.
- Applied **scaling to numeric variables** for better model performance.
- Engineered features likely to boost predictive accuracy.

---

# 4. **Model Creation and Evaluation**

## **Objective**
The goal of this phase was to train multiple machine learning models, optimize their hyperparameters, and evaluate their ability to predict customer churn. The primary evaluation metric chosen was **ROC AUC**, as it measures the model’s discriminatory power across thresholds.

---

## **4.1 Models Considered**
The following models were evaluated:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost
- LightGBM

---

## **4.2 Training and Evaluation Setup**
- **Train/Test Split:** 80/20 (stratified to preserve churn ratio).
- **Hyperparameter Tuning:** Performed using **GridSearchCV** with **3-fold cross-validation**.
- **Evaluation Metric:** ROC AUC (higher is better, 0.5 ≈ random guessing).
- **Preprocessing:** One-hot encoding for categorical variables, scaling for numeric features.
- **Feature Selection:** `SelectFromModel` with Random Forest was used before final training.

---

## **4.3 Model Performance Summary**
| Model              | Best Hyperparameters | ROC AUC |
|--------------------|----------------------|---------|
| Logistic Regression | `C=10, penalty='l2'` | 0.4853 |
| SVM                | `C=1, kernel='linear', gamma='scale'` | 0.5186 |
| Decision Tree      | `max_depth=3, min_samples_split=2` | 0.5059 |
| Random Forest      | `max_depth=5, n_estimators=200` | 0.5020 |
| AdaBoost           | `n_estimators=100, learning_rate=1` | 0.5078 |
| XGBoost            | `n_estimators=200, max_depth=7, learning_rate=0.1` | 0.5089 |
| LightGBM           | `n_estimators=100, max_depth=5, learning_rate=0.01, num_leaves=31` | 0.4956 |

---

## **4.4 Observations**
- All models scored **close to 0.5 ROC AUC**, indicating **no significant predictive power** beyond random chance.
- **SVM** achieved the highest score (**0.5186**), but still not acceptable for practical use.
- Even **ensemble methods (XGBoost, LightGBM, Random Forest)** underperformed, suggesting the issue lies in:
  - **Feature quality** (features may not strongly correlate with churn).
  - **Data imbalance or noise** (though churn imbalance was handled, other quality issues may remain).
  - **Feature engineering** might need more domain-specific insights.

---

## **4.5 Key Insights & Next Steps**
- The current features are **not providing strong signal** for churn prediction.
- Further steps:
  - Perform **advanced feature engineering** (interaction terms, customer usage patterns, payment history trends).
  - Explore **non-linear transformations** or **binning strategies**.
  - Try **SMOTE or class weighting** if imbalance impacts learning.
  - Validate data quality (missing patterns, outliers).
- Consider gathering **additional behavioral data** to improve model performance.

---

## ✅ **Conclusion**
At this stage, no model achieved satisfactory predictive capability. Before deployment, additional **feature engineering and data enrichment** are necessary to improve the model’s ability to distinguish churners from non-churners.
