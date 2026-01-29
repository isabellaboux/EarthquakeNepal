
import pandas as pd
import seaborn as sns
from modelling import train_data, test_data
from data_processing import processing_dataset


#
train_X = pd.read_csv("/Users/carricarte/Documents/DSR/data_challenge/EarthquakeNepal/data/train_values.csv")
train_y = pd.read_csv("/Users/carricarte/Documents/DSR/data_challenge/EarthquakeNepal/data/train_labels.csv")
test_X = pd.read_csv("/Users/carricarte/Documents/DSR/data_challenge/EarthquakeNepal/data/test_values.csv")
test_y = pd.read_csv("/Users/carricarte/Documents/DSR/data_challenge/EarthquakeNepal/data/test_labels.csv")
#
myTrain_X, myTrain_y, myTest_X, myTest_X = processing_dataset(train_X, test_X, train_y, test_y)
#
#
F1_score, F1_std = train_data(train_X, train_y)