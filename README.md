# House Price Prediction

Predicting house prices from property features with linear regression, including feature selection and residual analysis. College coursework project.

## Dataset

A housing-prices CSV (`Housing.csv`) with 545 listings and 13 columns: price, area, bedrooms, bathrooms, stories, parking, and yes/no flags for main road, guest room, basement, hot water heating, air conditioning and preferred area, plus furnishing status. No missing values. **The CSV is not included in this repo**, so download the dataset and point the notebook's `pd.read_csv(...)` path at it.

## What the notebook does

1. **Cleaning:** checks for nulls and removes outliers in price and area
2. **EDA:** distributions and correlations (area is the feature most correlated with price)
3. **Preparation:** converts yes/no columns to numbers, creates dummy variables for furnishing status, splits train/test, rescales features
4. **Modelling:** Recursive Feature Elimination (RFE) down to 6 features, then a linear regression fitted with statsmodels OLS
5. **Evaluation:** residual analysis on the training data, then R² on the test set

## Result

Test R² = **0.58**.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
jupyter notebook HousePricePrediction.ipynb
```
