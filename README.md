# Logistic Regression for Titanic Survival Prediction

This Jupyter Notebook contains a Python script for building and evaluating logistic regression models to predict passenger survival on the Titanic. The analysis uses the "titanic-training-data.csv" dataset and covers data preprocessing, model building, and performance evaluation.

## Dataset

The dataset includes information about Titanic passengers, such as their age, sex, class, family relationships, fare, and more. The primary goal is to predict whether a passenger survived the disaster.

### Data Preprocessing

- Handling Missing Values: The notebook addresses missing values in columns like "Age," "Cabin," and "Embarked" using appropriate strategies such as removal and imputation.
- Encoding Categorical Variables: Categorical variables such as "Pclass," "Sex," and "Embarked" are transformed into binary columns using one-hot encoding.
- Exploratory Data Analysis (EDA): Visualizations are used to understand relationships between variables, including survival rates based on sex and passenger class.

### Model Building

- Logistic Regression Model: A logistic regression model is created to predict passenger survival.
- Train-Test Split: The dataset is split into training and testing sets for model evaluation.
- Model Evaluation: The notebook calculates and displays the accuracy score, classification report, and confusion matrix for the logistic regression model.

### Additional Models

- Decision Tree Classifier: A decision tree classifier with limited depth is built and evaluated.
- Bagging Classifier: Bagging is applied to the decision tree classifier to improve performance.
- AdaBoost Classifier: AdaBoost is used to boost the decision tree classifier's performance.
- Random Forest Classifier: A random forest classifier is created and assessed.

## Usage

You can run this Jupyter Notebook in a Python environment (e.g., Jupyter Notebook or JupyterLab) to analyze the Titanic dataset, preprocess the data, build logistic regression models, and compare their performance. Ensure you have the required libraries installed, as mentioned in the notebook.

## Dependencies

This notebook relies on the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
