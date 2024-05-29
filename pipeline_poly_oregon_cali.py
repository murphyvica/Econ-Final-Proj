import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly
import us


## 1) Join housing data, GDP, mortage rates, create boolean column for election year

# Importing data, selecting subset based on dates of housing data

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

    pr = pd.read_csv(f"data/{state_lower}/{state}.csv")
    GDP = pd.read_csv(f"data/{state_lower}/{abv}NGSP.csv")
    MORT = pd.read_csv("data\MORTGAGE30US.csv")
    CPI = pd.read_csv("data\CPIAUCSL.csv")
    FFUNDS = pd.read_csv("data\FEDFUNDS.csv")

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

    # Define LASSO model
    X = df[['GDP', 'Mortgage (30Yr)', 'is_election', 'CPI', 'FEDFUNDS']]
    y = df['Price']
    dates = df['Date']

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    # create train and test sets
    #X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X_poly, y, df['Date'],test_size = 0.2)

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

    # Train the model
    poly_reg = LinearRegression()
    poly_reg.fit(X_train, y_train)

    # Predictions
    y_pred_train = poly_reg.predict(X_train)
    y_pred_test = poly_reg.predict(X_test)

    print(poly_reg.score(X_test, y_test))

    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print("rmse: ", np.sqrt(mse))
    print("r2: ", r2)

    # metrics = {'State' : [f'{state}'], 'mse' : [mse], 'r2' : [r2]}
    # out = pd.DataFrame(metrics)
    # out.to_csv('metrics.csv')

    metrics = {'State' : [f'{state}'], 'mse' : [mse], 'r2' : [r2]}
    out = pd.DataFrame(metrics)
    out.to_csv('metrics.csv', mode='a', header=False)


## 4) Analysis, data viz, r-squared for other states (running LASSO regression for each state, comparing how well model works)

# Create a dataframe to align predictions with dates
train_results = pd.DataFrame({'Date': dates_train, 'Actual': y_train, 'Predicted': y_pred_train})
test_results = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_pred_test})

# Concatenate results
#all_results = pd.concat([train_results, test_results]).sort_values('Date')

train_seg_1 = train_results[train_results['Date'] <= split_date_1]
train_seg_2 = train_results[train_results['Date'] >= split_date_2]

# Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(df['Date'], df['Price'], label='Actual')
# plt.plot(train_seg_1['Date'], train_seg_1['Predicted'], label='Predicted Train', color = 'Orange')
# plt.plot(test_results['Date'], test_results['Predicted'], label='Predicted Test', color = 'Green')
# plt.plot(train_seg_2['Date'], train_seg_2['Predicted'], color = 'Orange')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('Oregon Polynomial Regression')
# plt.legend()
# plt.show()