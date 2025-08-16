import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sns

from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, f1_score, log_loss, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

file_name= "drug200.csv"
data = pd.read_csv(file_name)

data.info()

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['BP'] = label_encoder.fit_transform(data['BP'])
data['Cholesterol'] = label_encoder.fit_transform(data['Cholesterol'])

data.isnull().sum()

data.drop('Drug',axis=1).corr()['Drug_num']

category_counts = data['Drug'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()


y = data['Drug']
X = data.drop(['Drug','Drug_num'], axis=1)

# Split the dataset into training and testing sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

tree_predictions = drugTree.predict(X_testset)

print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

plot_tree(drugTree)
plt.show()

# If the max depth of the tree is reduced to 3, how would the performance of the model be affected?
drugTree_reduced = DecisionTreeClassifier(criterion="entropy", max_depth=3)
drugTree_reduced.fit(X_trainset, y_trainset)
tree_predictions_reduced = drugTree_reduced.predict(X_testset)
print("Reduced Decision Tree's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions_reduced))


