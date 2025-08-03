import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

file_name= "insurance.csv"
df = pd.read_csv(file_name)

print(df.head())

lm = LinearRegression()
lm

X = df[['bmi']]
Y = df['charges']

lm.fit(X, Y)

lm.coef_
lm.intercept_

# lm.predict(x)

# smoker is a categorical attribute, replace with most frequent entry
is_smoker = df['smoker'].value_counts().idxmax()
df["smoker"].replace(np.nan, is_smoker, inplace=True)

# age is a continuous variable, replace with mean age
mean_age = df['age'].astype('float').mean(axis=0)
df["age"].replace(np.nan, mean_age, inplace=True)

# Update data types
df[["age","smoker"]] = df[["age","smoker"]].astype("int")

print(df.info())

df[["charges"]] = np.round(df[["charges"]],2)
print(df.head())

# Implement the regression plot for charges with respect to bmi.
sns.regplot(x="bmi", y="charges", data=df, line_kws={"color": "red"})
plt.ylim(0,)

# Implement the box plot for charges with respect to smoker.
sns.boxplot(x="smoker", y="charges", data=df)

# Print the correlation matrix for the dataset.
corr_matrix = df.corr()
print(corr_matrix)

# Fit a linear regression model that may be used to predict the charges value, just by using the smoker attribute of the dataset. Print the
#  score of this model.
X = df[['smoker']]
Y = df['charges']
lm.fit(X, Y)
print("Coefficient:", lm.coef_)
print("Intercept:", lm.intercept_)
print("R^2 Score:", lm.score(X, Y))

# Fit a linear regression model that may be used to predict the charges value, just by using all other attributes of the dataset. Print the
#  score of this model. You should see an improvement in the performance.
X = df.drop(columns=['charges'])
Y = df['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
lm.fit(X_train, Y_train)
print("Coefficient:", lm.coef_)
print("Intercept:", lm.intercept_)
print("R^2 Score:", lm.score(X_test, Y_test))

# Initialize a Ridge regressor that used hyperparameter alpha=0.1.
# Fit the model using training data subset. Print the R^2
#  score for the testing data.
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, Y_train)
print("Ridge Coefficient:", ridge.coef_)
print("Ridge Intercept:", ridge.intercept_)
print("Ridge R^2 Score:", ridge.score(X_test, Y_test))
yhat = ridge.predict(X_test)
print("Ridge R^2 Score:", r2_score(Y_test, yhat))
print("Ridge Mean Squared Error:", mean_squared_error(Y_test, yhat))

# Apply polynomial transformation to the training parameters with degree=2. Use this transformed feature set to fit the same regression model,
# as above, using the training subset. Print the
#  score for the testing subset.
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
lm.fit(X_poly, Y_train)
X_test_poly = poly.transform(X_test)
ridge.fit(X_poly, Y_train)
y_hat_poly = ridge.predict(X_test_poly)
print("Polynomial Mean Squared Error:", mean_squared_error(Y_test, y_hat_poly))
print("Polynomial Coefficient:", lm.coef_)
print("Polynomial Intercept:", lm.intercept_)
print("Polynomial R^2 Score:", lm.score(X_test_poly, Y_test))