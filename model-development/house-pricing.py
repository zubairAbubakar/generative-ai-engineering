import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

file_name= "house.csv"
df = pd.read_csv(file_name)

print(df.head())

df.dtypes

df.describe()

df.drop(columns=['id', 'Unnamed: 0'], inplace=True)
df.describe()

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# Use the method value_counts to count the number of houses with unique floor values,
# use the method .to_frame() to convert it to a data frame.
floor_counts = df['floors'].value_counts().to_frame()
print(floor_counts)

# Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or
# without a waterfront view have more price outliers.
sns.boxplot(x='waterfront', y='price', data=df)
plt.title('Price Distribution by Waterfront View')
plt.xlabel('Waterfront View (0 = No, 1 = Yes)')
plt.ylabel('Price')
plt.show()

# Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.
sns.regplot(x='sqft_above', y='price', data=df, line_kws={"color": "red"})
plt.title('Price vs. Square Feet Above')
plt.xlabel('Square Feet Above')
plt.ylabel('Price')
plt.show()

df_numeric = df.select_dtypes(include=[np.number])
df_numeric.corr()['price'].sort_values()

X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2.
X = df[['sqft_living']]
Y = df['price']
lm.fit(X, Y)
r_squared = lm.score(X, Y)
print(f'R^2 for sqft_living: {r_squared}')


# Fit a linear regression model to predict the 'price' using the list of features:
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X = df[features]
Y = df['price']
lm.fit(X, Y)
r_squared = lm.score(X, Y)
print(f'R^2 for multiple features: {r_squared}')

# Create a list of tuples, the first element in the tuple contains the name of the estimator:
# 'scale'
# 'polynomial'
# 'model'
# The second element in the tuple contains the model constructor
# StandardScaler()
# PolynomialFeatures(include_bias=False)
# LinearRegression()

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

# Use the list to create a pipeline object to predict the 'price',
# fit the object using the features in the list features, and calculate the R^2.
pipeline = Pipeline(Input)
X = df[features]
Y = df['price']
pipeline.fit(X, Y)
r_squared = pipeline.score(X, Y)
print(f'R^2 for pipeline: {r_squared}')


X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1,
# and calculate the R^2 using the test data.
ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
r_squared = ridge.score(x_test, y_test)
print(f'R^2 for Ridge regression: {r_squared}')

# Perform a second order polynomial transform on both the training data and testing data.
# Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1,
# and calculate the R^2 utilising the test data provided.
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
ridge.fit(x_train_poly, y_train)
r_squared_poly = ridge.score(x_test_poly, y_test)
print(f'R^2 for Ridge regression with polynomial features: {r_squared_poly}')