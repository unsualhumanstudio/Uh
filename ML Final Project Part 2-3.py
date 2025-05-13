'''
Diana Valdes
IDSN 542, Fall 2024
valdesco@usc.edu
Final Project Part 1-3
'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#The imports below are new for 4/7 lecture
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
#imnports for 4/9 lecture
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import customtransformerindex as indxcf
import safetyscorecustomtransformer as sccf
from sklearn.model_selection import GridSearchCV

#Part 2 - Preparing Training data

#Open up the file housing.csv and converting the data into a pandas DataFrame object
def load_traffic_data():

    return pd.read_csv("Traffic Data.csv")

#Creating a new attribute density_cat to see if it can help identify areas of high risk for pedestrians
def do_the_cut(traffic):
    # Fill missing values in Traffic_Density with its median
    traffic["Traffic_Density"] = traffic["Traffic_Density"].fillna(traffic["Traffic_Density"].median())
    bins = [-0.1, 0.5, 1, 1.5, 2, np.inf]
    labels = [1, 2, 3, 4, 5]
    traffic['density_cat'] = pd.cut(traffic["Traffic_Density"], bins=bins, labels=labels)
    #traffic["density_cat"].hist()
    #plt.show() #You have to delete the popup window to continue the program

traffic = load_traffic_data()
do_the_cut(traffic)

#Creating reliable train/test sets
if traffic["density_cat"].isnull().sum() > 0:
    raise ValueError("density_cat still contains NaN values. Check bin definitions.")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(traffic, traffic["density_cat"]):
    strat_train_set = traffic.loc[train_index]
    strat_test_set = traffic.loc[test_index]

#making a copy to not make changes to the file data
traffic = strat_train_set.copy()

#Removing Accident so that the model can predict it not learn it
traffic = strat_train_set.drop("Accident", axis=1)
traffic_labels = strat_train_set["Accident"].copy()
# Handling missing values in traffic_labels
traffic_labels = traffic_labels.fillna(traffic_labels.median())
# Identifying numeric and categorical columns
numeric_columns = traffic.select_dtypes(include=[np.number]).columns
categorical_columns = traffic.select_dtypes(exclude=[np.number]).columns

# Converting object columns to category
for column in categorical_columns:
    traffic[column] = traffic[column].astype('category')
    traffic[column] = traffic[column].cat.codes

#if loop to check spelling and make sure it's a column label before removing the text attribute
#so that we can use num_pipeline on the numeric data and OneHotEncoder on non-numeric data
if "weather" in traffic.columns:
    weather_cat = traffic.drop("weather", axis=1)
else:
    print("Column 'weather' not found, skipping drop.")
    weather_cat = traffic.copy()

if "road_type" in traffic.columns:
    road_type_num = traffic.drop("road_type", axis=1)
else:
    print("Column 'road_type' not found.")
    road_type_num = traffic.copy()

if "time_of_day" in traffic.columns:
    time_of_day_num = traffic.drop("time_of_day", axis=1)
else:
    print("Column 'time_of_day' not found, skipping drop.")
    time_of_day_num = traffic.copy()

if "accident_severity" in traffic.columns:
    accident_severity_num = traffic.drop("accident_severity", axis=1)
else:
    print("Column 'accident_severity' not found, skipping drop.")
    accident_severity_num = traffic.copy()

if "road_condition" in traffic.columns:
    road_condition_num = traffic.drop("road_condition", axis=1)
else:
    print("Column 'road_condition' not found, skipping drop.")
    road_condition_num = traffic.copy()

if "vehicle_type" in traffic.columns:
    vehicle_type_num = traffic.drop("vehicle_type", axis=1)
else:
    print("Column 'vehicle_type' not found, skipping drop.")
    vehicle_type_num = traffic.copy()

if "road_light_condition" in traffic.columns:
    road_light_condition_num = traffic.drop("road_light_condition", axis=1)
else:
    print("Column 'road_light_condition' not found, skipping drop.")
    road_light_condition_num = traffic.copy()

print(weather_cat.info())
print(road_type_num.info())
print(time_of_day_num.info())
print(accident_severity_num.info())
print(road_condition_num.info())
print(vehicle_type_num.info())
print(road_light_condition_num.info())

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('risk_index_adder', indxcf.RiskIndexAdder()),
    ('std_scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),  # For categorical data
    ('onehot', OneHotEncoder(handle_unknown="ignore")),])


#applying transformations
weather_cat_tr = cat_pipeline.fit_transform(weather_cat)
road_type_num_tr = num_pipeline.fit_transform(road_type_num)
time_of_day_num_tr = num_pipeline.fit_transform(time_of_day_num)
accident_severity_num_tr = num_pipeline.fit_transform(accident_severity_num)
road_condition_num_tr = num_pipeline.fit_transform(road_condition_num)
vehicle_type_num_tr = num_pipeline.fit_transform(vehicle_type_num)
road_light_condition_num_tr = num_pipeline.fit_transform(road_light_condition_num)

#We separate the attributes into the numeric ones and the text ones | didn't work for this data set
#num_attribs = list(traffic)
#cat_attribs = ['weather', 'road_type', 'time_of_day', 'accident_severity', 'road_condition','vehicle_type', 'road_light_condition']

#combining numeric and categorical processing of attributes
full_pipeline = ColumnTransformer([("num", num_pipeline, numeric_columns),
                                   ("cat", cat_pipeline, categorical_columns),])
# Applying the pipeline
traffic_prepared = full_pipeline.fit_transform(traffic)
#print(traffic_prepared)
#print(traffic)


#Training and evaluating on the training set
lin_reg = LinearRegression()
lin_reg.fit(traffic_prepared, traffic_labels)


#Done. Now we have a working Linear Regression model. Let's try it out on a few instances from the training set
some_data = traffic.iloc[:5]
some_labels = traffic_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))



traffic_predictions = lin_reg.predict(traffic_prepared)
lin_mse = mean_squared_error(traffic_labels, traffic_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)



tree_reg = DecisionTreeRegressor()
tree_reg.fit(traffic_prepared, traffic_labels)
housing_predictions = tree_reg.predict(traffic_prepared)
tree_mse = mean_squared_error(traffic_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


scores = cross_val_score(tree_reg, traffic_prepared, traffic_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

'''
#display the results
print("DT_Scores:", scores)
print("DT_Mean:", scores.mean())
print("DT_Standard deviation:", scores.std())
'''

#Let's do cross validation on the linear regression model
lin_scores = cross_val_score(lin_reg, traffic_prepared, traffic_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("LE_Scores:", lin_rmse_scores)
print("LE_Mean:", lin_rmse_scores.mean())
print("LE_Standard deviation:", lin_rmse_scores.std())


forest_reg = RandomForestRegressor()
forest_reg.fit(traffic_prepared, traffic_labels)
housing_predictions = forest_reg.predict(traffic_prepared)
forest_mse = mean_squared_error(traffic_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

forest_scores = cross_val_score(forest_reg, traffic_prepared, traffic_labels,
                             scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("Forest Scores:", forest_rmse_scores)
print("Forest Mean:", forest_rmse_scores.mean())
print("Forest Standard deviation:", forest_rmse_scores.std())


#-----------------------------------
#Fine Tune your model
param_grid = [
#    {'n_estimators': [80, 160], 'max_features': [8, 15]},
    {'bootstrap': [False], 'n_estimators': [80], 'max_features': [8]},]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV( forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit( traffic_prepared, traffic_labels)

print(grid_search.best_params_)


#print(grid_search.best_estimator_)


cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score), params)


feature_importances = grid_search.best_estimator_.feature_importances_
#print(feature_importances)


extra_attribs = ["Risk_Index", "Safety_Score"]
cat_encoder = full_pipeline.named_transformers_["cat"].named_steps["onehot"]
cat_one_hot_attribs = []
for cat in cat_encoder.categories_:
    cat_one_hot_attribs.extend(list(cat))
attributes = list(numeric_columns) + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))



final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("Accident", axis=1)
y_test = strat_test_set["Accident"].copy()
if y_test.isnull().sum() > 0:
    print("Missing values detected in y_test. Filling missing values with median.")
    y_test = y_test.fillna(y_test.median())

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

