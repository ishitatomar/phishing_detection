import pandas as pd
import numpy as np

#Loading the dataset
df=pd.read_csv("dataset.csv")

#Printing initial shape of DataFrame
print("Initial shape of DataFrame is ",df.shape)

#Cleaning the Data

#Handling the missing values (if found then filled withe the mode)
for c in df.columns:
	if df[c].isnull().any():
		df[c].fillna(df[c].mode()[0], inplace=True)

#Removing the duplicate rows if any
df.drop_duplicates(inplace=True)

#Printing shape of DataFrame after removing any duplicate rows
print("\nShape of DataFrame after removing duplicates is ", df.shape)

#Handling outliers using the Interquartile Range method (IQR)
numerical_cols = df.select_dtypes(include=np.number).columns
for c in numerical_cols:
	Q1=df[c].quantile(0.25)
	Q3=df[c].quantile(0.75)
	IQR=Q3-Q1
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	df = df[(df[c] >= lower_bound) & (df[c] <= upper_bound)]

#Printing shape of DataFrame after handling the outliers
print("\nShape of DataFrame after handling  outliers is ", df.shape)
