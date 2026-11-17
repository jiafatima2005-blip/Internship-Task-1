# Internship-Task-1
Multi-Class Benchmark Classification Pipeline --- California Housing
Project Overview
This project implements an end-to-end multi-class classification
pipeline using the supplied California Housing dataset and Scikit-Learn.

The original median_house_value column contains continuous house
prices. To make the problem suitable for multi-class classification, it
is converted into three project-defined categories:

  Category                      House value
  Low                       Below $150,000
Medium       $150,000 to below $300,000
High                  $300,000 and above

The notebook compares Logistic Regression and Decision Tree
classifiers using accuracy, precision, recall, F1-score, classification
reports, and confusion matrices.
Project Files
- California_Housing_Multi_Class_Classification_Pipeline.ipynb ---
  main Jupyter Notebook.
- california_housing.csv --- dataset used by the notebook.
- README.md --- project documentation.
Keep the notebook and CSV file in the same folder.
Objectives
- Load and explore the dataset using Pandas.
- Inspect missing values and descriptive statistics.
- Visualize feature relationships with a Seaborn pairplot.
- Convert continuous house values into Low, Medium, and High classes.
- Split data into training and testing sets using an 80/20 split.
- Impute missing values, standardize numeric features, and encode
  categorical features.
- Train Logistic Regression and Decision Tree classifiers.
- Compare classification metrics.
- Plot confusion matrix heatmaps.
- Visualize model decision regions using PCA.
Technologies and Libraries
- Python 3
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
Dataset Features
The dataset includes:
- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- median_house_value
- ocean_proximity
median_house_value is used to create the classification target. The
original continuous target column is excluded from model input features
to prevent target leakage.
Methodology
1. Data Loading and Exploration
The CSV file is loaded with Pandas. The notebook displays sample rows,
dataset information, descriptive statistics, missing-value counts, class
counts, and a pairplot based on a reproducible sample of up to 1,000
rows.
2. Target Engineering
The continuous median_house_value target is grouped into three
classes:
- Low: less than $150,000
- Medium: $150,000 to less than $300,000
- High: $300,000 or more
These thresholds are introduced for this classification task; they are
not original labels supplied by the dataset.
3. Train/Test Split
The data is divided into 80% training data and 20% testing data. A
stratified split preserves class proportions. The random state is set to
42 for reproducibility.
4. Preprocessing
- Numeric columns: median imputation followed by StandardScaler.
- Categorical columns: most-frequent imputation followed by
  OneHotEncoder.
- Preprocessing is fitted on training data only and then applied to
  test data.
5. Model Training
Two supervised classification algorithms are trained:
- Logistic Regression
- Decision Tree Classifier
6. Model Evaluation
The notebook reports accuracy, weighted precision, weighted recall, and
weighted F1-score. It also prints per-class precision, recall, F1-score,
and support through classification_report.

Confusion matrix heatmaps are plotted for both models.
7. Decision Region Visualization
PCA projects the processed feature space onto two dimensions for
visualization. The classifiers are trained on the full processed feature
set; PCA is used to display their decision regions, not to replace the
original training features.
