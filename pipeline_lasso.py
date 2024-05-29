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
GDP = pd.read_csv("data\WA_GDP.csv")
MORT = pd.read_csv("data\MORTGAGE30US.csv")
CPI = pd.read_csv("data\CPIAUCSL.csv")
FFUNDS = pd.read_csv("data\FEDFUNDS.csv")
POP = pd.read_csv("data\WA_pop.csv")

WA["Date"] = pd.to_datetime(WA["Date"])
GDP["DATE"] = pd.to_datetime(GDP["DATE"])
MORT["DATE"] = pd.to_datetime(MORT["DATE"])
CPI["DATE"] = pd.to_datetime(CPI["DATE"])
FFUNDS["DATE"] = pd.to_datetime(FFUNDS["DATE"])
POP['DATE'] = pd.to_datetime(POP['DATE'], format='%Y')

start_date = WA["Date"].min()
GDP = GDP[GDP.DATE.dt.year >= start_date.year]
MORT = MORT[MORT.DATE.dt.year >= start_date.year]
CPI = CPI[CPI.DATE.dt.year >= start_date.year]
FFUNDS = FFUNDS[FFUNDS.DATE.dt.year >= start_date.year]
POP = POP[POP.DATE.dt.year >= start_date.year]

# Interpolating GDP and Mortgage data

GDP.set_index('DATE', inplace=True)
GDP_day = GDP.resample('D').interpolate(method='spline', order=3)

MORT.set_index('DATE', inplace=True)
MORT_day = MORT.resample('D').interpolate(method='spline', order=3)  # can use ffill instead: MORT_day = MORT.resample('D').ffill() 

CPI.set_index('DATE', inplace=True)
CPI_day = CPI.resample('D').interpolate(method='spline', order=3)

FFUNDS.set_index('DATE', inplace=True)
FFUNDS_day = FFUNDS.resample('D').interpolate(method='spline', order=3)

POP.set_index('DATE', inplace=True)
POP_day = POP.resample('D').interpolate(method='spline', order=3)

# Joining data together

df = pd.merge(WA, GDP_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, MORT_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, CPI_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, FFUNDS_day, left_on="Date", right_on="DATE", how='left')
df = pd.merge(df, POP_day, left_on="Date", right_on="DATE", how='left')

# adding election year boolean column

df['is_election'] = df.Date.dt.year % 4 == 0

df.rename(columns={'MORTGAGE30US': 'Mortgage (30Yr)'}, inplace=True)
df.rename(columns={'WANGSP': 'GDP'}, inplace=True)
df.rename(columns={'CPIAUCSL': 'CPI'}, inplace=True)

## 2) Cleaning data: check null, negative values

# removes rows of any data points that contain null values
df = df.dropna()

# checks and removes any rows that contain unusual values outside the dataset (negative values)
col_check = ['Price']
condition = (df[col_check] >= 0).all(axis=1)
df = df[condition]

# converting boolean to int
df['is_election'] = df['is_election'].astype(int)

print(df)

## 3) LASSO regression model

# Define LASSO model
X = df[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS', '% Pop Growth']]
y = df['Price']

# create train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

split_date_1 = '2015-03-01'
split_date_2 = '2020-03-01'

train = df[(df['Date'] < split_date_1) | (df['Date'] > split_date_2)]

test = df[(df['Date'] >= split_date_1) & (df['Date'] <= split_date_2)]


# Split the data into training and test sets

X_train = train[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS', '% Pop Growth']]
y_train = train['Price']
dates_train = train['Date']

X_test = test[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS', '% Pop Growth']]
y_test = test['Price']
dates_test = test['Date']

# standardizing
# scale = StandardScaler()
# X_train_scale = scale.fit_transform(X_train)
# X_test_scale = scale.transform(X_test)

# train model
lasso = Lasso(alpha=6000)
lasso.fit(X_train, y_train)

# prediction model
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# print(y_train_pred)
# print(y_test_pred)

## 4) Analysis, data viz, r-squared for other states (running LASSO regression for each state, comparing how well model works)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_results = pd.DataFrame({'Date': dates_train, 'Actual': y_train, 'Predicted': y_train_pred})
test_results = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_test_pred})

train_seg_1 = train_results[train_results['Date'] <= split_date_1]
train_seg_2 = train_results[train_results['Date'] >= split_date_2]

#print('alpha: ', lasso.alpha_)
print('rmse: ', np.sqrt(train_mse))
print('score: ', lasso.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Price'], label='Actual Prices')
plt.plot(train_seg_1['Date'], train_seg_1['Predicted'], label='Predicted Train', color = 'Orange')
plt.plot(test_results['Date'], test_results['Predicted'], label='Predicted Test', color = 'Green')
plt.plot(train_seg_2['Date'], train_seg_2['Predicted'], color = 'Orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Housing Prices: Actual vs Predicted')
plt.show()

## 5) Creating presentation, 


