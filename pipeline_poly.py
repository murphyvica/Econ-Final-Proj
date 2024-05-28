import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


## 1) Join housing data, GDP, mortage rates, create boolean column for election year

# Importing data, selecting subset based on dates of housing data

WA = pd.read_csv("data\Washington.csv")
GDP = pd.read_csv("data\WA_GDP.csv")
MORT = pd.read_csv("data\MORTGAGE30US.csv")
CPI = pd.read_csv("data\CPIAUCSL.csv")
FFUNDS = pd.read_csv("data\FEDFUNDS.csv")

WA["Date"] = pd.to_datetime(WA["Date"])
GDP["DATE"] = pd.to_datetime(GDP["DATE"])
MORT["DATE"] = pd.to_datetime(MORT["DATE"])
CPI["DATE"] = pd.to_datetime(CPI["DATE"])
FFUNDS["DATE"] = pd.to_datetime(FFUNDS["DATE"])

start_date = WA["Date"].min()
GDP = GDP[GDP.DATE.dt.year >= start_date.year]
MORT = MORT[MORT.DATE.dt.year >= start_date.year]
CPI = CPI[CPI.DATE.dt.year >= start_date.year]
FFUNDS = FFUNDS[FFUNDS.DATE.dt.year >= start_date.year]

# Interpolating GDP and Mortgage data

GDP.set_index('DATE', inplace=True)
GDP_day = GDP.resample('D').interpolate(method='spline', order=3)

MORT.set_index('DATE', inplace=True)
MORT_day = MORT.resample('D').interpolate(method='spline', order=3)  # can use ffill instead: MORT_day = MORT.resample('D').ffill() 

CPI.set_index('DATE', inplace=True)
CPI_day = CPI.resample('D').interpolate(method='spline', order=3)

FFUNDS.set_index('DATE', inplace=True)
FFUNDS_day = FFUNDS.resample('D').interpolate(method='spline', order=3)

# Joining data together

df = pd.merge(WA, GDP_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, MORT_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, CPI_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, FFUNDS_day, left_on="Date", right_on="DATE", how='left')

# adding election year boolean column

df['is_election'] = df.Date.dt.year % 4 == 0

df.rename(columns={'MORTGAGE30US': 'Mortgage (30Yr)'}, inplace=True)
df.rename(columns={'WANGSP': 'GDP'}, inplace=True)
df.rename(columns={'CPIAUCSL': 'CPI'}, inplace=True)

## 2) Cleaning data: check null, negative values

# removes rows of any data points that contain null values
df = df.dropna()

# checks and removes any rows that contain unusual values outside the dataset (negative values)
col_check = df.select_dtypes(include=[np.number]).columns
df = df[(df[col_check] >= 0).all(axis = 1)]

# converting boolean to int
df['is_election'] = df['is_election'].astype(int)

print(df)

# Define LASSO model
X = df[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS']]
y = df['Price']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# create train and test sets
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X_poly, y, df['Date'],test_size = 0.2)

# standardizing
scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)

# Train the model
poly_reg = LinearRegression()
poly_reg.fit(X_train, y_train)

# Predictions
y_pred_train = poly_reg.predict(X_train)
y_pred_test = poly_reg.predict(X_test)

print(poly_reg.score(X_test, y_test))


## 4) Analysis, data viz, r-squared for other states (running LASSO regression for each state, comparing how well model works)

# Create a dataframe to align predictions with dates
train_results = pd.DataFrame({'Date': dates_train, 'Actual': y_train, 'Predicted': y_pred_train})
test_results = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_pred_test})

# Concatenate results
all_results = pd.concat([train_results, test_results]).sort_values('Date')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Actual')
plt.plot(all_results['Date'], all_results['Predicted'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

## 5) Creating presentation, 


