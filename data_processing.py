# import
git add data_processing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
y_train = pd.read_csv('data/train_labels.csv')
X_train = pd.read_csv('data/train_values.csv')
X_test = pd.read_csv('data/test_values.csv')#data processing
#whatever we do on the training also do on the test set
#1.drop secondery use columns
# drop columns that include 'secondary_use' in their name
cols_to_drop_train = [c for c in X_train.columns if 'secondary_use' in c]
if cols_to_drop_train:
	X_train = X_train.drop(columns=cols_to_drop_train)
cols_to_drop_test = [c for c in X_test.columns if 'secondary_use' in c]
if cols_to_drop_test:
	X_test = X_test.drop(columns=cols_to_drop_test)
#2.drop count_floors_pre_eq that is highly correlated with other features
X_train = X_train.drop(columns=['count_floors_pre_eq'])
X_test = X_test.drop(columns=['count_floors_pre_eq'])
#3.outliers
#for continous variables we calculate 95%percentile 

#x and y dataframes as outputs

