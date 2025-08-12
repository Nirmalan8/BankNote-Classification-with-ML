#  Bank Note Authentication – Classification Project

##  Project Overview
This project focuses on **classifying banknotes as authentic or forged** using machine learning.  
We train and evaluate classification models based on features extracted from images of banknotes using **wavelet transforms**.

The dataset contains statistical features such as:
- **Variance**
- **Skewness**
- **Curtosis**
- **Entropy**

Our goal: **Build a model that accurately predicts whether a banknote is genuine or counterfeit.**

---

##  Dataset
**Source:** (https://archive.ics.uci.edu/ml/datasets/banknote+authentication)  

**Features:**
| Column Name | Description |
|-------------|-------------|
| variance    | Variance of the Wavelet Transformed image |
| skewness    | Skewness of the Wavelet Transformed image |
| curtosis    | Curtosis of the Wavelet Transformed image |
| entropy     | Entropy of the image |
| class       | Target variable (0 = Authentic, 1 = Forged) |

---

##  Tech Stack
- **Language:** Python 3.x  
- **Libraries:**  
  - `pandas`, `numpy` – Data handling
  - `matplotlib`, `seaborn` – Visualization
  - `scikit-learn` – ML models, metrics
  - `xgboost` – Gradient boosting classifier

---

##  Workflow
1. **Load & Explore Data**
   - Check dataset shape, data types, missing values
   - Visualize class distribution

2. **Data Preprocessing**
   - Standardize numerical features
   - Train-test split

3. **Model Training**
   - Logistic Regression
   - Decision Tree
   - XGBoost Classifier

4. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

5. **Feature Importance**
   - Visualize which features impact predictions most

---

##  Results
- **Best Performing Model:** XGBoost Classifier  
- **Accuracy:** ~99% on test data  
- **Key Insight:** Wavelet-transformed variance and skewness are the most important indicators.

---
## Future Improvements
Add deep learning approach using neural networks

Deploy model as a web app with Flask or Streamlit

Collect a larger dataset for better generalization

---
