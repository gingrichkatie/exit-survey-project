Exit Survey Analysis — CIS 9660 Project #2 (Q1)
Goal
Predict the primary reason for leaving from exit survey responses to help HR identify attrition trends and target retention strategies.
Dataset
Source: Kaggle — Employee Exit Survey (DETE/TAFE)
Format: CSV with structured fields (Age, Gender, Department, Length_of_Service, etc.)
Placement: Save as employee_exit_survey.csv in the project root.
Pipeline
Normalize headers, convert Age ranges to numeric midpoints.
Derive primary_reason from separation fields using keyword grouping.
Select features:
Numeric: ['Age']
Categorical: ['Gender', 'Department']
Text: None
Train/test split (70/30) + 5-fold cross-validation.
Models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Linear SVM, KNN.
Save evaluation plots and model artifacts for deployment.
Results
Best Model: Logistic Regression
Test Accuracy: 0.704
Weighted F1: 0.644
Class distribution:
Career Growth: 352
Other: 339
Relocation: 131
Figures:
Class counts: figures/class_counts.png
CV accuracy: figures/cv_accuracy.png
Test metrics: figures/test_metrics.png
K-Means elbow: figures/kmeans_elbow.png
Running the Notebook
Run all code blocks in order (.ipynb provided). Outputs will be saved in:
figures/ — visualizations for appendix
artifacts/ — model and preprocessing objects for deployment
Streamlit App
To run locally:
pip install -r requirements.txt
streamlit run streamlit_app.py
Upload CSV: Predict reasons for leaving in batch.
Manual Form: Enter a single employee's info and get an instant prediction.
Deployment
Push the following to GitHub and connect to Streamlit Community Cloud:
streamlit_app.py
artifacts/ folder
requirements.txt
(Optional) sample CSV for uploads
Artifacts
artifacts/best_model.pkl — trained model
artifacts/preprocess.pkl — numeric/categorical preprocessing pipeline
artifacts/text_vectorizer.pkl — TF-IDF vectorizer (None in this run)
artifacts/column_config.pkl — feature/target configuration
Notes
No usable free-text comments in this dataset, so TF-IDF was not applied.
Career Growth is the top attrition driver; recommend targeted career pathing and promotion strategies.
Response bias and class imbalance remain; future work could include oversampling and interpretability tools (e.g., SHAP).
License & Citation
(https://www.kaggle.com/datasets/abiyyuhrusin/employee-exit-survey)
