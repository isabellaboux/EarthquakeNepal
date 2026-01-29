# def main():
#     print("Hello from challenge1!")


# if __name__ == "__main__":
#     main()

# import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load data
y_train = pd.read_csv('data/train_labels.csv')
X_train = pd.read_csv('data/train_values.csv')
X_test = pd.read_csv('data/test_values.csv')

# explore
X_train.head()

# plot 
X_test.hist()
sns.heatmap(X_test.select_dtypes(include='number').corr())

print(X_train.describe())
print(X_train.columns)
