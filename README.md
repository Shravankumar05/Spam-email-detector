# Spam Email Detection

This project implements a spam email detector using machine learning techniques. It classifies emails as either **spam** or **ham** (not spam) using various models such as Logistic Regression, Support Vector Machines, Naive Bayes, and Random Forest.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)

## Project Overview

The project focuses on detecting spam emails by analyzing their content and extracting meaningful features. The approach involves text preprocessing, feature extraction using TF-IDF, and training multiple classifiers to evaluate their performance.

## Dataset

The dataset used in this project is `spam_ham_dataset.csv`, which contains labeled email data. Each email is labeled as spam or ham, with the `label_num` column indicating `1` for spam and `0` for ham.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.6 or later
- Required Python libraries (see below)

### Install Required Libraries

You can install the necessary Python libraries using pip. Run the following command in your terminal:

pip install numpy pandas matplotlib seaborn scikit-learn

## Usage
### 1. Clone the repository
Clone the repository to your local machine:
git clone [https://github.com/Shravan/spam-email-detection.git](https://github.com/Shravankumar05/Spam-email-detector)
cd spam-email-detection

### 2.Install required libraries
You can do this with:
pip install numpy pandas matplotlib seaborn scikit-learn

### 3. Run the Jupyter Notebook
jupyter notebook

### 4. Spam detection function
def spam_detection(mail):
    mail_features = feature_extraction.transform(mail)
    prediction = LR.predict(mail_features)
    
    if prediction[0] == 1:
        return "Spam mail"
    else:
        return "Ham mail"

Example usage:
result = spam_detection(["Your email content here"])
print(result)

## Models Used
### 1. Logistic regression
- A simple linear model used for binary classification tasks.
- Effective for large datasets with a linear decision boundary.

### 2. Support vector machines
- Uses a hyperplane to separate classes in feature space.
- Suitable for both linear and non-linear classification with kernel trick.
- Implemented with a radial basis function (RBF) kernel in this project.

### 3. Naive Bayes
- A probabilistic model based on Bayes' theorem.
- Particularly effective for text classification.
- Assumes features are independent to simplify calculations.

### 4. Random forest
- An ensemble method that combines multiple decision trees.
- Robust to overfitting and effective for classification tasks.
- Uses the average prediction of multiple trees for final classification
