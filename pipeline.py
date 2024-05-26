import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

## 1) Join housing data, GDP, mortage rates, create boolean column for election year

# Importing data, selecting subset based on dates of housing data

WA = pd.read_csv("data\Washington.csv")
GDP = pd.read_csv("data\GDP.csv")
MORT = pd.read_csv("data\MORTGAGE30US.csv")

WA["Date"] = pd.to_datetime(WA["Date"])
GDP["DATE"] = pd.to_datetime(GDP["DATE"])
MORT["DATE"] = pd.to_datetime(MORT["DATE"])

start_date = WA["Date"].min()
GDP = GDP[GDP.DATE.dt.year >= start_date.year]
MORT = MORT[MORT.DATE.dt.year >= start_date.year]

# Interpolating GDP and Mortgage data

GDP.set_index('DATE', inplace=True)
GDP_day = GDP.resample('D').interpolate(method='spline', order=3)

MORT.set_index('DATE', inplace=True)
MORT_day = MORT.resample('D').interpolate(method='spline', order=3)  # can use ffill instead: MORT_day = MORT.resample('D').ffill() 

# Joining data together

df = pd.merge(WA, GDP_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, MORT_day, left_on="Date", right_on="DATE", how='left')

# adding election year boolean column

df['is_election'] = df.Date.dt.year % 4 == 0

df.rename(columns={'MORTGAGE30US': 'Mortgage (30Yr)'}, inplace=True)


## 2) Cleaning data: check null, negative values

# removes rows of any data points that contain null values
df = df.dropna()

# checks and removes any rows that contain unusual values outside the dataset (negative values)
col_check = df.select_dtypes(include=[np.number]).columns
df = df[(df[col_check] >= 0).all(axis = 1)]

# converting boolean to int
df['is_election'] = df['is_election'].astype(int)

print(df)

## 3) LASSO regression model




## 4) Analysis, data viz, r-squared for other states (running LASSO regression for each state, comparing how well model works)




## 5) Creating presentation, 


