# Spam Email Detection

This project implements a spam email detector using machine learning techniques. It classifies emails as either **spam** or **ham** (not spam) using various models such as Logistic Regression, Support Vector Machines, Naive Bayes, and Random Forest.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

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

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
