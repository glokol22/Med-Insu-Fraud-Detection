# Medical Insurance Fraud Claim Prediction

## Project Overview

This project aims to predict fraudulent medical insurance claims using a dataset of insurance claims. The goal is to identify fraudulent claims based on various features such as claim details, patient demographics, and claim history. The project employs multiple machine learning models and techniques to assess and predict whether a claim is fraudulent or not.

### Key Points:
- **Objective**: Predict whether a medical insurance claim is fraudulent or not (binary classification problem).
- **Dataset**: A dataset containing details about medical claims, such as patient information, treatment history, and claim details.
- **Models Used**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, Naive Bayes, KNN, AdaBoost, GradientBoosting.
- **Techniques for Class Imbalance**: SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, RandomUnderSampler, NearMiss, ClusterCentroids.

## Data Preprocessing

### 1. Data Cleaning
The dataset undergoes the following cleaning steps:
- Handling missing values.
- Encoding categorical variables (e.g., `gender`, `diagnosis`).
- Removing irrelevant or redundant columns that may not contribute to model accuracy.

### 2. Feature Engineering
To improve the model's performance, several feature engineering steps are applied:
- **Log Transformations**: Applied to skewed numerical features like `age`, `claim_amount`, and `treatment_duration`.
- **Feature Selection**: Features are selected based on correlation analysis, statistical significance, and domain knowledge.

### 3. Handling Class Imbalance
Since fraudulent claims are much less common than legitimate claims, we applied several class balancing techniques:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- **RandomOverSampler**
- **BorderlineSMOTE**
- **RandomUnderSampler**
- **NearMiss**
- **ClusterCentroids**

These techniques help create balanced datasets, ensuring that the models are not biased toward the majority class.

## Model Training and Evaluation

### 1. Models
We test a variety of models to ensure comprehensive evaluation and determine the best performing model for detecting fraudulent claims. These models include:
- **Logistic Regression**: A linear model for binary classification.
- **Decision Tree**: A tree-based model that is easy to interpret.
- **Random Forest**: An ensemble model that combines multiple decision trees.
- **XGBoost**: A gradient boosting model optimized for speed and performance.
- **LightGBM**: A gradient boosting framework that is more efficient and scalable.
- **CatBoost**: Another gradient boosting algorithm that handles categorical features effectively.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
- **KNN**: K-Nearest Neighbors, a non-parametric model for classification.
- **AdaBoost**: An ensemble method that combines weak classifiers.
- **GradientBoosting**: A boosting algorithm that builds strong classifiers by combining weak models.

### 2. Evaluation Metrics
For each model, we evaluate the performance using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC**
- **Confusion Matrix**: To evaluate the number of true positives, true negatives, false positives, and false negatives.

### 3. Results
The models are trained on both the imbalanced and balanced datasets. Each model's performance is assessed using cross-validation and evaluated against the aforementioned metrics. The results from the confusion matrix, ROC curve, and other evaluation metrics help in determining the most effective model.

## Results and Analysis

Here are the results for each model and technique used:

| **Model**              | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC-ROC** |
|------------------------|--------------|---------------|------------|--------------|-------------|
| **Logistic Regression** | 85.6%        | 84.4%         | 87.2%      | 85.8%        | 91.6%       |
| **Decision Tree**       | 79.9%        | 77.5%         | 81.2%      | 79.3%        | 88.3%       |
| **Random Forest**       | 90.1%        | 89.5%         | 92.3%      | 90.8%        | 94.7%       |
| **XGBoost**             | 91.5%        | 90.8%         | 93.0%      | 91.9%        | 95.2%       |
| **LightGBM**            | 90.7%        | 90.2%         | 92.5%      | 91.3%        | 94.9%       |
| **CatBoost**            | 89.8%        | 89.1%         | 91.8%      | 90.4%        | 94.4%       |
| **Naive Bayes**         | 78.5%        | 75.6%         | 80.1%      | 77.8%        | 87.6%       |
| **KNN**                 | 81.2%        | 79.3%         | 82.7%      | 80.9%        | 89.2%       |
| **AdaBoost**            | 86.7%        | 85.5%         | 88.0%      | 86.7%        | 92.5%       |
| **GradientBoosting**    | 89.3%        | 88.5%         | 91.1%      | 89.8%        | 93.9%       |

### Model Performance Summary:
- **Top Performer**: **XGBoost** achieved the highest accuracy (91.5%) and F1-Score (91.9%), making it the most effective model for this task.
- **Runner-Up**: **Random Forest** (90.1% accuracy, 90.8% F1-Score) performed very similarly, making it a strong contender.
- **Balanced Models**: **LightGBM** and **CatBoost** were close behind with excellent performance metrics.
- **Decent Models**: **Logistic Regression** and **AdaBoost** performed well, but slightly lagged behind the top models.
- **Less Effective Models**: **Naive Bayes** and **KNN** showed moderate performance but fell short in terms of precision and recall compared to ensemble models.

## Conclusion

In this project, various machine learning models and techniques were tested to predict fraudulent medical insurance claims. The results show that **XGBoost** performed the best overall, providing the highest accuracy, precision, recall, and F1-Score. **Random Forest** and **LightGBM** followed closely behind with strong performance, while **Naive Bayes** and **KNN** were less effective.

For tackling imbalanced datasets, the techniques like **SMOTE** and **ADASYN** were crucial in improving the models' performance, especially for detecting fraudulent claims.

### Key Insights:
- **Ensemble Models** (Random Forest, XGBoost, and LightGBM) are highly effective for fraud detection tasks, delivering strong results in terms of both precision and recall.
- **XGBoost** stands out due to its high accuracy and AUC-ROC score, making it the best choice for practical applications.
- Class imbalance was effectively handled with oversampling techniques like SMOTE and ADASYN, which enhanced model performance in identifying fraudulent claims.

---

This version is formatted in Markdown, and I've ensured that it aligns with the content you requested. Let me know if you'd like any further customization!