import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR

# Load your dataset
data = pd.read_excel('Players August 2024 - Copy.xlsx')

# Define the feature and target columns
features = [
    'Height', 'Country', 'CAPS', 'CAPS GOALS', 'Foot', 'Age', 'Position', 'Position Role', 'MarketValue', 
    'DaysLeftofContract', 'TEAM RATING', 'TEAM RANKING', 'LEAGUE RATING', 'HighestMarketValue', 
    'AgeAtHighestMarketValue', 'NumberOfMarketValueChanges', 'LatestMarketValue', 'MeanMarketValue', 
    'MedianMarketValue', 'MarketValueStdDeviation', 'MarketValueTrend', 'TotalIncrease', 'TotalDecrease', 
    'CurrentToMaxRatio', 'DurationAtMaxValue'
]
target = 'OverallStats'

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_features = ['Height', 'CAPS', 'CAPS GOALS', 'Age', 'MarketValue', 'DaysLeftofContract', 
                    'TEAM RATING', 'TEAM RANKING', 'LEAGUE RATING', 'HighestMarketValue', 
                    'AgeAtHighestMarketValue', 'NumberOfMarketValueChanges', 'LatestMarketValue', 
                    'MeanMarketValue', 'MedianMarketValue', 'MarketValueStdDeviation', 
                    'TotalIncrease', 'TotalDecrease', 'CurrentToMaxRatio', 'DurationAtMaxValue']
categorical_features = ['Country', 'Foot', 'Position', 'Position Role', 'MarketValueTrend']

# Preprocessing for numeric data: impute missing values, create interaction features, scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Adding interaction features
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define individual models with pipelines and hyperparameter tuning
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor())
])

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor())
])

lgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('lgb', LGBMRegressor())
])

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('gb', GradientBoostingRegressor())
])

svr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('svr', SVR())
])

# Hyperparameter tuning for each pipeline using GridSearchCV
rf_params = {'rf__n_estimators': [100, 500], 'rf__max_depth': [10, 30]}
xgb_params = {'xgb__n_estimators': [300, 849], 'xgb__max_depth': [6, 11], 'xgb__learning_rate': [0.01, 0.05]}
lgb_params = {'lgb__n_estimators': [300, 849], 'lgb__max_depth': [10, 15], 'lgb__learning_rate': [0.01, 0.05]}
gb_params = {'gb__n_estimators': [100, 500], 'gb__max_depth': [3, 5], 'gb__learning_rate': [0.01, 0.05]}
svr_params = {'svr__C': [0.1, 1], 'svr__gamma': ['scale', 'auto'], 'svr__kernel': ['rbf', 'linear']}

# Creating GridSearchCV objects
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lgb_grid = GridSearchCV(lgb_pipeline, lgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gb_grid = GridSearchCV(gb_pipeline, gb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid = GridSearchCV(svr_pipeline, svr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fitting each grid search
rf_grid.fit(X_train, y_train)
xgb_grid.fit(X_train, y_train)
lgb_grid.fit(X_train, y_train)
gb_grid.fit(X_train, y_train)
svr_grid.fit(X_train, y_train)

# Defining the stacking regressor with the best models from grid search
stacking_regressor = StackingRegressor(
    estimators=[
        ('rf', rf_grid.best_estimator_),
        ('xgb', xgb_grid.best_estimator_),
        ('lgb', lgb_grid.best_estimator_),
        ('gb', gb_grid.best_estimator_),
        ('svr', svr_grid.best_estimator_)
    ],
    final_estimator=RidgeCV(alphas=np.logspace(-6, 6, 13))
)

# Fit the stacking regressor
stacking_regressor.fit(X_train, y_train)

# Evaluate the stacking regressor on the test set
y_pred_stacking = stacking_regressor.predict(X_test)
stacking_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stacking))
stacking_mse = mean_squared_error(y_test, y_pred_stacking)
stacking_r2 = r2_score(y_test, y_pred_stacking)
stacking_mae = mean_absolute_error(y_test, y_pred_stacking)

print(f'Stacking Regressor Test RMSE: {stacking_rmse}')
print(f'Stacking Regressor Test MSE: {stacking_mse}')
print(f'R-squared (R²) Score: {stacking_r2}')
print(f'Mean Absolute Error (MAE): {stacking_mae}')

# Save the model
model_filename = 'stacking_regressor_model_model_August_Test.pkl'
joblib.dump(stacking_regressor, model_filename)
