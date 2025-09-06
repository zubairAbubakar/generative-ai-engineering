import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Load the rainfall dataset from a URL
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

df.count()

df = df.dropna()
df.info()

df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()

# Create a function to map dates to seasons
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Exercise 1: Map the dates to seasons and drop the Date column
df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns=['Date'])
df.info()

# Exercise 2. Define the feature and target dataframes
X = df.drop(columns=['RainToday'], axis=1)
y = df['RainToday']

# Exercise 3. How balanced are the classes?
class_counts = y.value_counts()
print(class_counts)
# RainToday
# No     5766
# Yes    1791
# Name: count, dtype: int64

# Exercise 4. What can you conclude from these counts?
# How often does it rain annually in the Melbourne area?

# Exercise 5. Split data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define preprocessing transformers for numerical and categorical features¶
# Exercise 6. Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features
# Complete the followng code:
#
# numeric_features = X_train.select_dtypes(include=['...']).columns.tolist()
# categorical_features = X_train.select_dtypes(include=['...', 'category']).columns.tolist()
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Exercise 7. Combine the transformers into a single preprocessing column transformer
# Complete the followng code:
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, ...),
#         ('cat', categorical_transformer, ...)
#     ]
# )
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# Exercise 8. Create a pipeline by combining the preprocessing with a Random Forest classifier
# Complete the following code:
#
# pipeline = Pipeline(steps=[
#     ('preprocessor', ...),
#     ('...', RandomForestClassifier(random_state=42))
# ])
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True)


# Exercise 9. Instantiate and fit GridSearchCV to the pipeline
# Complete the followng code:
#
# grid_search = GridSearchCV(..., param_grid, cv=..., scoring='accuracy', verbose=2)
# grid_search.fit(..., ...)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Exercise 10. Display your model's estimated score¶
# Complete the followng code:
#
# test_score = grid_search.score(..., ...)
# print("Test set score: {:.2f}".format(test_score))
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# Exercise 11. Get the model predictions from the grid search estimator on the unseen data
# Complete the followng code:
#
# y_pred = grid_search.predict(...)
y_pred = grid_search.predict(X_test)


# Exercise 12. Print the classification report
# Complete the followng code:
#
# print("\nClassification Report:")
# print(...(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Exercise 13. Plot the confusion matrix
# Complete the followng code:
#
# conf_matrix = ...(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=...)
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# Exercise 14. Extract the feature importances¶
# Complete the followng code:
#
# feature_importances = grid_search.best_estimator_['classifier']. ...
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# ## Exercise 15. Update the pipeline and the parameter grid
# Let's update the pipeline and the parameter grid and train a Logistic Regression model and compare the performance of the two models. You'll need to replace the clasifier with LogisticRegression. We have supplied the parameter grid for you.
#
# Complete the following code:
# ```python
# # Replace RandomForestClassifier with LogisticRegression
# pipeline.set_params(...=LogisticRegression(random_state=42))
#
# # update the model's estimator to use the new pipeline
# grid_search.estimator = ...
#
# # Define a new grid with Logistic Regression parameters
# param_grid = {
#     # 'classifier__n_estimators': [50, 100],
#     # 'classifier__max_depth': [None, 10, 20],
#     # 'classifier__min_samples_split': [2, 5],
#     'classifier__solver' : ['liblinear'],
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__class_weight' : [None, 'balanced']
# }
#
# grid_search.param_grid = ...
#
# # Fit the updated pipeline with LogisticRegression
# model.fit(..., ...)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)

print(classification_report(y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

#               precision    recall  f1-score   support
#
#           No       0.86      0.93      0.89      1154
#          Yes       0.68      0.51      0.58       358
#
#     accuracy                           0.83      1512
#    macro avg       0.77      0.72      0.74      1512
# weighted avg       0.82      0.83      0.82      1512