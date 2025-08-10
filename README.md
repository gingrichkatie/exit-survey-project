# Exit Survey Analysis — CIS 9660 Project #2 (Q1)

## Goal
Predict the **primary reason for leaving** from exit survey responses to help HR identify attrition trends and target retention strategies.

## Dataset
- **Source:** [Kaggle — Employee Exit Survey (DETE/TAFE)](https://www.kaggle.com/datasets/abiyyuhrusin/employee-exit-survey)
- **Format:** CSV with structured fields (Age, Gender, Department, Length_of_Service, etc.)
- **Placement:** Save as `employee_exit_survey.csv` in the project root.

## Pipeline
1. Normalize headers, convert Age ranges to numeric midpoints.
2. Derive `primary_reason` from separation fields using keyword grouping.
3. Select features:
   - Numeric: `['Age']`
   - Categorical: `['Gender', 'Department']`
   - Text: None
4. Train/test split (70/30) + 5-fold cross-validation.
5. Models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Linear SVM, KNN.
6. Save evaluation plots and model artifacts for deployment.

## Results
- **Best Model:** Logistic Regression
- **Test Accuracy:** 0.704
- **Weighted F1:** 0.644
- **Class Distribution:**
  - Career Growth: 352
  - Other: 339
  - Relocation: 131
- **Figures:**
  - Class counts: `figures/class_counts.png`
  - CV accuracy: `figures/cv_accuracy.png`
  - Test metrics: `figures/test_metrics.png`
  - K-Means elbow: `figures/kmeans_elbow.png`

## Running the Notebook
Run all code blocks in order (`.ipynb` provided). Outputs will be saved in:
- `figures/` — visualizations for appendix
- `artifacts/` — model and preprocessing objects for deployment

## Streamlit App
To run locally:
```bash
pip install -r requirements.txt
streamlit run exit-survey-streamlit.py
