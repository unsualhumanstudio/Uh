'''
Diana Valdes
IDSN 542, Fall 2024
valdesco@usc.edu
Final Project Part 1
'''

'''
Final project Part 1:
You are to write some Python code to identify your dataset attributes, their data type, and any missing
values. You can also use the info() function to print out the attributes and the number of values each
attribute has. Also, use the corr() function to find out if you have any linear correlations on the numeric
attributes.
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy
import sklearn

#P1: Data Exploration

#loading data set
def load_DataSet():
     return pd.read_csv('Traffic.csv')

dataSet = load_DataSet()

# Printing the attributes and the number of values each attribute has
print("Dataset Attributes and Values:")
dataSet.info()

print(dataSet["Road_Type"].value_counts())
print(dataSet.describe())

dataSet.hist( bins=50, figsize=(20,15))
plt.show()

# Checking for missing values
print("\nMissing Values:")
for column in dataSet.columns: #for loop to loop through each column and calculate missing values
    missingCount = len(dataSet) - dataSet[column].count()
    print(f"{column}: {missingCount} missing values")

# find out if you have any linear correlations on the numeric attributes.
print("\nCorrelation Matrix:")
print(dataSet.corr(numeric_only=True))

