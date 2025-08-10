import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sns

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, f1_score, log_loss, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

file_name= "Obesity_level_prediction_dataset.csv"
data = pd.read_csv(file_name)

print(data.isnull().sum())
print(data.info())
print(data.describe())

sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()

# Scale the numerical features to standardize their ranges for better model performance.
# Standardizing continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])

# Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()

# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

# Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3)
# and observe the impact on model performance.
test_sizes = [0.1, 0.2, 0.3]
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Train OvA model
    model_ova.fit(X_train, y_train)
    y_pred_ova = model_ova.predict(X_test)
    accuracy_ova = accuracy_score(y_test, y_pred_ova)

    # Train OvO model
    model_ovo.fit(X_train, y_train)
    y_pred_ovo = model_ovo.predict(X_test)
    accuracy_ovo = accuracy_score(y_test, y_pred_ovo)

    print(f"Test Size: {test_size}")
    print(f"One-vs-All Accuracy: {np.round(100*accuracy_ova,2)}%")
    print(f"One-vs-One Accuracy: {np.round(100*accuracy_ovo,2)}%")


# Plot a bar chart of feature importance using the coefficients from the One vs All logistic regression model.
# Also try for the One vs One model.
coefficients_ova = model_ova.coef_[0]
feature_names = X.columns
importance_df_ova = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients_va})
importance_df_ova = importance_df_ova.sort_values(by='Coefficient', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(importance_df_ova['Feature'], importance_df_ova['Coefficient'])
plt.title('Feature Importance (One-vs-All)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.show()


#  Write a function obesity_risk_pipeline to automate the entire pipeline:
#
# Loading and preprocessing the data
# Training the model
# Evaluating the model
# The function should accept the file path and test set size as the input arguments.
def obesity_risk_pipeline(file_path, test_size=0.2):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Scale the numerical features
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

    # Combine with the original dataset
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # One-hot encode categorical features
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Exclude target column
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Encode the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    # Prepare final dataset
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Train OvA model
    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)

    # Predictions and evaluation for OvA
    y_pred_ova = model_ova.predict(X_test)
    accuracy_ova = accuracy_score(y_test, y_pred_ova)

    print("One-vs-All (OvA) Strategy")
    print(f"Accuracy: {np.round(100*accuracy_ova,2)}%")

    return model_ova
