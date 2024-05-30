import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly
import us

## 1) Join housing data, GDP, mortage rates, create boolean column for election year

# Importing data, selecting subset based on dates of housing data

WA = pd.read_csv("data\washington\Washington.csv")
GDP = pd.read_csv("data\washington\WANGSP.csv")
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

# Polynomial Regression Model

# Choosing variables and determining test and train period
X = df[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS', '% Pop Growth']]
y = df['Price']
dates = df['Date']

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

# Setting up polynomial regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train) # Transforming training data
X_test_poly = poly.transform(X_test) # Transforming test data

# Scaling data
scaler = StandardScaler() 
X_train_poly = scaler.fit_transform(X_train_poly) 
X_test_poly = scaler.transform(X_test_poly) 

# Train the model
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Predictions
y_pred_train = poly_reg.predict(X_train_poly)
y_pred_test = poly_reg.predict(X_test_poly)

print(poly_reg.score(X_test_poly, y_test))

# Create a dataframe to align predictions with dates
train_results = pd.DataFrame({'Date': dates_train, 'Actual': y_train, 'Predicted': y_pred_train})
test_results = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_pred_test})

# Concatenate results
train_seg_1 = train_results[train_results['Date'] <= split_date_1]
train_seg_2 = train_results[train_results['Date'] >= split_date_2]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Actual')
plt.plot(train_seg_1['Date'], train_seg_1['Predicted'], label='Predicted Train', color = 'Orange')
plt.plot(test_results['Date'], test_results['Predicted'], label='Predicted Test', color = 'Green')
plt.plot(train_seg_2['Date'], train_seg_2['Predicted'], color = 'Orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

# LASSO Model

# Choosing variables and determining test and train period
X = df[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS', '% Pop Growth']]
y = df['Price']

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

# train model
lasso = Lasso(alpha=5500)
lasso.fit(X_train, y_train)

# prediction model
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Create a dataframe to align predictions with dates
train_results = pd.DataFrame({'Date': dates_train, 'Actual': y_train, 'Predicted': y_train_pred})
test_results = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_test_pred})

# Concatenate results
train_seg_1 = train_results[train_results['Date'] <= split_date_1]
train_seg_2 = train_results[train_results['Date'] >= split_date_2]

print('rmse: ', np.sqrt(train_mse))
print('score: ', lasso.score(X_test, y_test))

# Plot the results
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

# Not included in presentation but helpful in understanding how prices move for various states
# PLOTS data for all states allowing for user to choose which state to see

# reads state data
state_data = pd.read_csv("data/State.csv")
state = pd.DataFrame(state_data)
# list to filter out dates for plotting
exclude = ['RegionID', 'SizeRank','RegionName', 'RegionType', 'StateName']

# identifies dates from column names and arranges data for plotting
date_columns = [col for col in state.columns if col not in exclude]
melted = state.melt(id_vars='RegionName', value_vars=date_columns, var_name='Date', value_name='Price')

# converts to datetime
melted['Date'] = pd.to_datetime(melted['Date'], format='%Y-%m-%d')

# creates interactive plot
fig = px.line(melted, x='Date', y='Price', color='RegionName', title='Housing Prices Over Time')

# widget to plot
fig.update_layout(
    updatemenus=[
        dict(
            type="dropdown",
            buttons=[
                dict(
                    label="All",
                    method="update",
                    args=[{"visible": [True] * len(melted['RegionName'].unique())},
                          {"title": "All States"}]),
            ] + [
                dict(
                    label=state,
                    method="update",
                    args=[{"visible": [region == state for region in melted['RegionName']]},
                          {"title": f"State: {state}"}])
                for state in melted['RegionName'].unique()
            ],
            direction="down"
        )
    ]
)

fig.show()

# map data to determine our model onto other states
abbreviation_to_name = {"AL": "Alabama","AR": "Arkansas", "AZ": "Arizona","CA": "California","CO": "Colorado",
    "CT": "Connecticut","DE": "Delaware","FL": "Florida","GA": "Georgia","IA": "Iowa","ID": "Idaho","IL": "Illinois",
    "IN": "Indiana","KS": "Kansas","KY": "Kentucky","LA": "Louisiana","MA": "Massachusetts","MD": "Maryland","ME": "Maine",
    "MI": "Michigan","MN": "Minnesota","MO": "Missouri","MS": "Mississippi","MT": "Montana","NC": "North Carolina","ND": "North Dakota",
    "NE": "Nebraska","NH": "New Hampshire","NJ": "New Jersey","NM": "New Mexico","NV": "Nevada","NY": "New York","OH": "Ohio",
    "OK": "Oklahoma","OR": "Oregon","PA": "Pennsylvania","RI": "Rhode Island","SC": "South Carolina","SD": "South Dakota",
    "TN": "Tennessee","TX": "Texas","UT": "Utah","VA": "Virginia","VT": "Vermont","WA": "Washington","WI": "Wisconsin",
    "WV": "West Virginia","WY": "Wyoming"}

name_to_abbreviation = {v: k for k, v in abbreviation_to_name.items()}

for state, abv in name_to_abbreviation.items():
    state_lower = state.lower()

    # reads specific state data
    pr = pd.read_csv(f"data/{state_lower}/{state}.csv")
    GDP = pd.read_csv(f"data/{state_lower}/{abv}NGSP.csv")
    MORT = pd.read_csv("data/MORTGAGE30US.csv")
    CPI = pd.read_csv("data/CPIAUCSL.csv")
    FFUNDS = pd.read_csv("data/FEDFUNDS.csv")

    # subsetting based on dates and organizing dataframe
    pr["Date"] = pd.to_datetime(pr["Date"])
    GDP["DATE"] = pd.to_datetime(GDP["DATE"])
    MORT["DATE"] = pd.to_datetime(MORT["DATE"])
    CPI["DATE"] = pd.to_datetime(CPI["DATE"])
    FFUNDS["DATE"] = pd.to_datetime(FFUNDS["DATE"])

    start_date = pr["Date"].min()
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

    df = pd.merge(pr, GDP_day, left_on="Date", right_on="DATE", how='left')
    df = pd.merge(df, MORT_day, left_on="Date", right_on="DATE", how='left')
    df = pd.merge(df, CPI_day, left_on="Date", right_on="DATE", how='left')
    df = pd.merge(df, FFUNDS_day, left_on="Date", right_on="DATE", how='left')

    # adding election year boolean column
    df['is_election'] = df.Date.dt.year % 4 == 0

    df.rename(columns={'MORTGAGE30US': 'Mortgage (30Yr)'}, inplace=True)
    df.rename(columns={f'{abv}NGSP': 'GDP'}, inplace=True)
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

    # Polynomial Regression Model
    X = df[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS']]
    y = df['Price']
    dates = df['Date']

    split_date_1 = '2015-03-01'
    split_date_2 = '2020-03-01'

    train = df[(df['Date'] < split_date_1) | (df['Date'] > split_date_2)]

    test = df[(df['Date'] >= split_date_1) & (df['Date'] <= split_date_2)]


    # Split the data into training and test sets

    X_train = train[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS']]
    y_train = train['Price']
    dates_train = train['Date']

    X_test = test[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS']]
    y_test = test['Price']
    dates_test = test['Date']

    # Setting up polynomial regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train) # Transforming training data
    X_test_poly = poly.transform(X_test) # Transforming test data

    # Scaling data
    scaler = StandardScaler() 
    X_train_poly = scaler.fit_transform(X_train_poly) 
    X_test_poly = scaler.transform(X_test_poly) 

    # Train the model
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)

    # Predictions
    y_pred_train = poly_reg.predict(X_train_poly)
    y_pred_test = poly_reg.predict(X_test_poly)
    
    print(poly_reg.score(X_test_poly, y_test))

    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print("rmse: ", np.sqrt(mse))
    print("r2: ", r2)

    # organizes and sends results to metrics csv file
    metrics = {'State' : [f'{state}'], 'mse' : [mse], 'r2' : [r2]}
    out = pd.DataFrame(metrics)
    out.to_csv('output_data\metrics.csv', mode='a', header=False)