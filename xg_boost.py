
"""
    Simple XGBoost model to visualize weight importance.

    One simple way of doing this involves counting the number 
    of times each feature is split on across all boosting rounds 
    (trees) in the model, and then visualizing the result as a 
    bar graph, with the features ordered according to how many 
    times they appear. XGBoost has a plot_importance() function 
    that allows you to do exactly this, and you'll get a chance 
    to use it in this exercise!
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


# Import dataset, sample shape: (10459, 35)
df = pd.read_csv('RiskData.csv')


# Clean dataset (filled with NaN values to impute).
df.replace('.', np.nan, inplace=True)


# Create arrays for the features and the response variable
y = df['Risk_Flag'].values
X = df.drop('Risk_Flag', axis=1).values
# print('x: ', X, len(X), 'y: ', y,  len(y))


# Create the DMatrix: risk_dmatrix
risk_dmatrix =  xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective": "reg:squarederror", "max_depth": 4}

# Train the model: xg_reg
xg_reg = xgb.train(dtrain=risk_dmatrix, params=params, num_boost_round=10)

# Plot the feature importances
xgb.plot_importance(booster=xg_reg)
plt.show()