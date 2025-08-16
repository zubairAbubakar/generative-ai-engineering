import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.preprocessing import normalize

file_name= "yellow_tripdata.csv"
raw_data = pd.read_csv(file_name)

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8,
                               random_state=35)

# fit the model to the training data
# Note: The fit method is used to train the model on the training data.
# It takes the feature matrix X_train and the target variable y_train as inputs.
# The model learns the relationships between the features and the target variable during this process.
dt_reg.fit(X_train, y_train)

# run inference using the sklearn model
y_pred = dt_reg.predict(X_test)

# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

# Q2. Identify the top 3 features with the most effect on the tip_amount.
feature_importances = dt_reg.feature_importances_
top_features_indices = feature_importances.argsort()[-3:][::-1]
top_features = proc_data.columns[top_features_indices]
print("Top 3 features affecting tip_amount:")
for feature in top_features:
    print(feature)