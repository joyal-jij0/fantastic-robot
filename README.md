# Fantastic Robot - Machine Learning Pipeline

This project implements a complete machine learning pipeline using a Jupyter Notebook. It covers data loading, exploratory data analysis (EDA), data cleaning, preprocessing, model training, ensemble methods, and submission generation.

## 1. Data Loading
- The training and testing datasets are loaded from CSV files (`train.csv` and `test.csv`).
- Early checks include printing shapes and head of data to understand the dataset structure.

## 2. Exploratory Data Analysis (EDA)
- Summary statistics and data types are inspected using methods like `.head()`, `.info()`, and `.describe()`.
- Visualizations such as missing value heatmaps, histograms, and boxplots are generated.  
  *Why?* To detect anomalies, understand distribution, and identify outliers or missing values that require attention.

## 3. Data Cleaning & Outlier Handling
- **Missing Values:**  
  - Categorical features: Missing values are replaced with the mode.
  - Numerical features: Missing values are imputed using a KNN imputer.
- **Outlier Handling:**  
  - Outliers are capped using quantile-based thresholds (typically at 1% and 99%), mitigating their adverse effects on model training.

## 4. Preprocessing Pipeline
- A `Pipeline` is constructed using `ColumnTransformer` for consistency:
  - **Numerical Features:** Standard scaled using `StandardScaler`.
  - **Categorical Features:** One-hot encoded with `OneHotEncoder`.
- This modular approach guarantees the same transformations are applied consistently during training and inference.

## 5. Model Training & Evaluation
- Several models are trained to leverage diverse algorithmic strengths:
  - Logistic Regression (configured for multinomial classification).
  - Random Forest, Gradient Boosting, SVM, and XGBoost.
- Each model is evaluated using accuracy on a validation set.  
  *Why?* Different models capture different patterns and an ensemble can significantly boost performance.

## 6. Ensemble Methods
- **Voting Classifier:**  
  - Combines predictions via soft voting where probabilities from all classifiers are averaged.
- **Stacking Classifier:**  
  - Uses base models to generate predictions that are then combined by a meta-estimator (first Logistic Regression, then an SVM variant).
  - This meta-learning approach aims to improve prediction accuracy by aggregating model outputs.

## 7. Final Model & Submission
- The final stacking classifier is retrained on the full training dataset.
- Test data undergoes the same preprocessing steps to guarantee alignment.
- Predictions from the final model are mapped to descriptive class labels.
- A submission file (`submission.csv`) is created containing `Planet_ID` and `Predicted_Class` for further evaluation.

## 8. How & Why
- **Preprocessing:** Ensures data consistency, handles missing values, and reduces outlier influence.
- **Multiple Models:** Different algorithms exploit various data aspects; ensemble methods leverage their individual strengths.
- **Stacking:** Offers robust predictions by combining diverse model outputs, reducing individual model biases.
- **Visualization:** Critical to understand data behavior and validate cleaning steps.

## 9. Usage
1. Install required libraries: pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib, etc.
2. Place `train.csv` and `test.csv` in the project directory.
3. Open and run the provided Jupyter Notebook to execute each processing step.
4. Generate the `submission.csv` file containing the final predictions.

## 10. Conclusion
This notebook demonstrates the integration of data cleaning, preprocessing, and advanced ensemble techniques into a single pipeline. Each component is carefully structured to contribute to an accurate and robust predictive model.
