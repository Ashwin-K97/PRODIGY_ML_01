import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import graphviz
from sklearn.tree import export_graphviz

from collections import Counter

from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
from Data import *

# Load the dataset
df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')

# Pre process
def pre_process(df):
    columns_to_drop = ['MiscFeature', 'PoolQC', 'Fence', 'Alley']
    df.drop(columns=columns_to_drop, inplace=True)
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    numeric_col = numeric_df.columns
    for col in numeric_col:
        df[col] = df[col].fillna(df[col].mean())
        
    str_df = df.select_dtypes(include='object')
    str_col = str_df.columns
    label_encoders = {}
    
    for col in str_col:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df['SaleCondition'].value_counts().idxmax())
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str)) 
        
    return df


df_train_pp = pre_process(df_train)
df_test_pp = pre_process(df_test)

# Plot Heat map
plt.figure(figsize=(60, 40))
sns.heatmap(df_train_pp.corr(), annot=True, fmt='.2f', cmap='twilight_shifted')
plt.show()


high_corr_columns = ['1stFlrSF', 'FullBath', 'Fireplaces', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'MasVnrArea', 'OverallQual', 'SalePrice', 'TotRmsAbvGrd', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd']
df_high_corr = df_train_pp[high_corr_columns]

plt.figure(figsize=(18, 12))
sns.heatmap(df_high_corr.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.show()


# Train and Test

X_train = df_train_pp.drop(['Id', 'SalePrice'], axis=1)
y_train = df_train_pp['SalePrice']

X_test = df_test_pp.drop(['Id'], axis=1)

results = []

regressors = [
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    MLPRegressor(random_state=42, max_iter=1000),
    AdaBoostRegressor(random_state=42),
    CatBoostRegressor(silent=True, random_state=42),
    LGBMRegressor(),
    KNeighborsRegressor(),
    LinearRegression(),
    Ridge(),
    Lasso(),
    SVR(),
]

regressor_names = [
    'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network', 
    'AdaBoost', 'CatBoost', 'LGBM', 'KNN', 
    'Linear', 'Ridge', 'Lasso', 'SVR',
]

r2_scores = []
mse_scores = []
mae_scores = []

for regressor in regressors:
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_train)
    
    r2_scores.append(r2_score(y_train, y_pred))
    mse_scores.append(mean_squared_error(y_train, y_pred))
    mae_scores.append(mean_absolute_error(y_train, y_pred))

result_df = pd.DataFrame({
    'Regressor': regressor_names,
    'R2': r2_scores,
    'MSE': mse_scores,
    'MAE': mae_scores
})

results.append(result_df)
result_df


lgbm_model = LGBMRegressor()
lgbm_model.fit(X_train, y_train)

y_pred_train = lgbm_model.predict(X_train)

r2_train = r2_score(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

print("R2:", r2_train)
print("MSE:", mse_train)
print("MAE:", mae_train)


lgbm_pred = lgbm_model.predict(X_test)

lgbm_end_pred = pd.DataFrame(lgbm_pred, index=df_test['Id'])
lgbm_end_pred.columns = ['SalePrice']
# lgbm_end_pred.to_csv('submission.csv', sep=',')
lgbm_end_pred.head(5)


catboost_model = CatBoostRegressor(silent=True, random_state=42)
catboost_model.fit(X_train, y_train)

y_pred_train = catboost_model.predict(X_train)

r2_train = r2_score(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

print("R2:", r2_train)
print("MSE:", mse_train)
print("MAE:", mae_train)

catboost_pred = catboost_model.predict(X_test)

catboost_end_pred = pd.DataFrame(catboost_pred, index=df_test['Id'])
catboost_end_pred.columns = ['SalePrice']
catboost_end_pred.to_csv('submission.csv', sep=',')
catboost_end_pred.head(5)


feature_importances = catboost_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

important_features = importance_df[importance_df['Importance'] > 1]
important_features