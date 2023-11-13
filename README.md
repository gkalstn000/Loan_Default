## Loan Data Analysis and Prediction

### Description

This project focuses on analyzing loan data to predict certain outcomes. It involves preprocessing the data, visualizing key features, and applying logistic regression for prediction.

### Script Overview

The script performs the following tasks:

1. **Data Loading**: Reads a CSV file containing loan data, removing thousands separators in numerical values.
2. **Data Preprocessing**:
   - Separates features (`X_loan`) and labels (`y_loan`).
   - Splits the data into training (70%) and testing (30%) sets.
   - Processes categorical variables (`First_home`, `Status`, `State`) and converts them into numerical formats.
   - Applies min-max normalization to various numerical features.
3. **Data Visualization**: Includes a histogram plot for the `Status` feature after preprocessing.
4. **Model Training and Evaluation**:
   - Trains a logistic regression model on the processed data.
   - Evaluates the model's performance on both the training and test datasets.
   - Generates a classification report to assess the model's predictive accuracy.

### Requirements

This script requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

### Usage

To run this script, ensure that you have the required libraries installed, and execute it in a Python 3 environment. The data file `data.csv` should be in the specified directory or updated to the correct path.

### Additional Notes

- The script assumes a specific structure for the input CSV file. Ensure that the data format matches the expected schema, particularly for columns used in the script.
- The performance and accuracy of the logistic regression model may vary based on the quality and characteristics of the input data.
