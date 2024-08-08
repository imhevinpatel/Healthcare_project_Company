# Healthcare_project_Company
health care kidney chronic

# Health Care Project - Chronic Kidney Disease Prediction

This project is a machine learning-based approach to predict chronic kidney disease (CKD) based on various health parameters. The project involves data preprocessing, exploratory data analysis, model building, and model evaluation. The trained models are saved for future predictions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Model Building](#model-building)
   - [Logistic Regression](#logistic-regression)
   - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
5. [Model Evaluation](#model-evaluation)
6. [Feature Importance](#feature-importance)
7. [Saving the Model](#saving-the-model)
8. [Installation and Setup](#installation-and-setup)
9. [Usage](#usage)
10. [Contributing](#contributing)


## Project Overview

The goal of this project is to predict the presence of chronic kidney disease (CKD) using various medical parameters. The dataset consists of 400 samples with 25 features each, including age, blood pressure, blood sugar levels, and more.

## Data Preprocessing

In this step, the dataset is cleaned and prepared for modeling:
- **Null Values Handling**: Null values are analyzed, and a decision is made whether to drop or retain them based on their impact on the dataset size and model performance.
- **Categorical Data Encoding**: Categorical features are converted into numerical values for compatibility with machine learning algorithms.
- **Data Balancing**: The dataset is balanced to avoid bias in the model, especially for KNN, which performs poorly on unbalanced data.

## Exploratory Data Analysis (EDA)

EDA is performed to understand the distribution and relationships of the features:
- **Correlation Matrix**: A heatmap is generated to visualize correlations between features.
- **Target Variable Distribution**: The distribution of the target variable (CKD or not) is analyzed.
- **Feature Distributions**: Histograms and box plots are used to understand the distribution of features like age and blood pressure.

## Model Building

### Logistic Regression

A logistic regression model is built and trained on the dataset:
- **Training Accuracy**: The model achieved a training accuracy of 100%.
- **Testing Accuracy**: The model achieved a testing accuracy of 97.5%.

### K-Nearest Neighbors (KNN)

A KNN model is also built and fine-tuned:
- **Hyperparameter Tuning**: GridSearchCV is used to find the best parameters for KNN.
- **Accuracy**: The tuned model achieved an accuracy of 100% on the test set.

## Model Evaluation

The models are evaluated using metrics like accuracy and confusion matrix:
- **Confusion Matrix**: Confusion matrices are plotted to visualize the model performance, highlighting true positives, false positives, true negatives, and false negatives.
- **Accuracy Score**: Both models showed high accuracy, with logistic regression slightly lower than KNN.

## Feature Importance

The importance of each feature is analyzed based on the coefficients from the logistic regression model:
- **Visualization**: Feature importance is visualized using bar plots and histograms.

## Saving the Model

The trained logistic regression model is saved using Python's `pickle` library for future use:
- **Model Saving**: The model is serialized and saved.
- **Model Loading**: The saved model can be loaded and used for predictions.

## Installation and Setup

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/health-care-kidney.git
   cd health-care-kidney
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Run the notebook or script to preprocess the data, train the models, and save the trained model.

## Usage

You can use the trained model to predict CKD on new data:
1. Load the model:
   ```python
   import pickle
   model = pickle.load(open('model.pkl', 'rb'))
   ```

2. Make predictions:
   ```python
   predictions = model.predict(new_data)
   ```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.



---

