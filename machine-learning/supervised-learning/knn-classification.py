import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.preprocessing import normalize
from sklearn.utils import compute_sample_weight

file_name= "teleCust1000t.csv"
df = pd.read_csv(file_name)

df['custcat'].value_counts()


