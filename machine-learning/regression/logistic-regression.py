import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, f1_score, log_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

file_name= "ChurnData.csv"
churn_df = pd.read_csv(file_name)

# For this lab, we can use a subset of the fields available to develop out model.
# Let us assume that the fields we use are 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip' and of course 'churn'.

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Create a logistic regression model and fit it to the training data
LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat[:10]

yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]

# Since the purpose here is to predict the 1 class more acccurately, you can also examine what role each input feature has to play in the prediction of the 1 class.
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# Evaluate the model's performance
print("Jaccard Score: ", jaccard_score(y_test, yhat))
print("F1 Score: ", f1_score(y_test, yhat, average='weighted'))
print("Log Loss: ", log_loss(y_test, yhat_prob))

# Let us assume we add the feature 'callcard' to the original set of input features.
# What will the value of log loss be in this case?
churn_df = pd.read_csv(file_name)
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard']])
y = np.asarray(churn_df['churn'])
X_norm = StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X_train, y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

# Evaluate the model's performance with the new feature
print("Jaccard Score with callcard: ", jaccard_score(y_test, yhat))
print("F1 Score with callcard: ", f1_score(y_test, yhat, average='weighted'))
print("Log Loss with callcard: ", log_loss(y_test, yhat_prob))

