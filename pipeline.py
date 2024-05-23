import numpy as np
import pandas as pd

## 1) Join housing data, GDP, mortage rates, create boolean column for election year

WA = pd.read_csv("data\Washington.csv")
GDP = pd.read_csv("data\GDP.csv")
MORT = pd.read_csv("data\MORTGAGE30US.csv")

WA["Date"] = pd.to_datetime(WA["Date"])
GDP["DATE"] = pd.to_datetime(GDP["DATE"])
MORT["DATE"] = pd.to_datetime(MORT["DATE"])

start_date = WA["Date"].min()
GDP = GDP[GDP.DATE.dt.year >= start_date.year]
MORT = MORT[MORT.DATE >= start_date]

GDP.set_index('DATE', inplace=True)
GDP_day = GDP.resample('D').ffill()

MORT.set_index('DATE', inplace=True)
MORT_day = MORT.resample('D').ffill()

df = pd.merge(WA, GDP_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, MORT_day, left_on="Date", right_on="DATE", how='left')

## 2) Cleaning data: check null, negative values




## 3) LASSO regression model




## 4) Analysis, data viz, r-squared for other states (running LASSO regression for each state, comparing how well model works)




## 5) Creating presentation, 


