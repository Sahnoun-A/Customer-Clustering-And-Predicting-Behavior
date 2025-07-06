# Customer Segmentation And Predicting Behavior

## 1. Business Situation
CRISA is an Asian market research agency that specializes in tracking consumer purchase behavior
in consumer goods (both durable and nondurable). In one major research project, CRISA tracks
numerous consumer product categories (e.g., "detergents"), and, within each category, perhaps
dozens of brands. To track purchase behavior, CRISA constituted household panels in over 100
cities and towns in India, covering most of the Indian urban market. The households were carefully
selected using stratified sampling to ensure a representative sample; a subset of 600 records is
analyzed here. The strata were defined on the basis of socioeconomic status and the market (a
collection of cities).

## 2. Dataset Overview
Contains demographic and transaction data for households, including:
- SEC, Affluence Index, Age, Gender, Education
- Promo Usage, Total Volume, and Value

## 3. Tools and Techniques
- **Languages**: Python
- **Libraries**: pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- **Environment**: Jupyter Notebook
- **Deployment**: Flask API for real-time predictions

## 4. Preprocessing
- One-hot and label encoding for categoricals
- StandardScaler for numeric columns
- Stratified train-test split

## 5. Exploratory Data Analysis
- Identified cluster-specific trends in promo usage, value, and loyalty
- Demographics linked to purchasing patterns

## 6. Model Building
- Models used: Logistic Regression, Random Forest, XGBoost
- Evaluated via accuracy, precision, recall
- XGBoost was the best-performing model

## 7. Results

| Segment           | Precision |
|------------------|-----------|
| Loyalists        | 0.90      |
| Variety Seekers  | 0.87      |
| Promo Shoppers   | 0.86      |

**Overall Accuracy:** 0.89

## 8. Key Takeaways
- Affluence Index and Promo Usage are top predictors
- Flask API app enables scoring one customer at a time
- SHAP analysis improves interpretability

## 9. Resources
- 📘 [**Kaggle Notebook**](https://www.kaggle.com/code/your-kaggle-notebook-url)
- 🗃 [**GitHub Repo**](https://github.com/your-repo)
- 🌐 [**Flask App Demo**](http://your-api-url)
