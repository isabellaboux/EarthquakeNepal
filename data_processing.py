# import

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load data
y_train = pd.read_csv('data/train_labels.csv')
X_train = pd.read_csv('data/train_values.csv')
X_test = pd.read_csv('data/test_values.csv')

y_train_copy = y_train.copy(deep=True)
X_train_copy = X_train.copy(deep=True)
X_test_copy = X_test.copy(deep=True)

	

def winsorize_dataframe(dataframe: pd.DataFrame, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> pd.DataFrame:

	num_cols = [
		col for col in dataframe.select_dtypes(include=[np.number]).columns
		if dataframe[col].nunique() > 2
	]

	winsor_limits = {}
	for col in num_cols:
		p5 = dataframe[col].quantile(0.05)
		p95 = dataframe[col].quantile(0.95)
		winsor_limits[col] = (p5, p95)
		dataframe[col] = dataframe[col].clip(p5, p95)

	for col, (p5, p95) in winsor_limits.items():
		if col in X_test_copy.columns:
			dataframe[col] = dataframe[col].clip(p5, p95)

	return dataframe

def drop_unnecessary_columns(dataframe: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
	"""
	Docstring for drop_unnecessary_columns
	
	:param dataframe: Description
	:type dataframe: pd.DataFrame
	:param columns_to_drop: Description
	:type columns_to_drop: list
	:return: Description
	:rtype: DataFrame
	"""
	return dataframe.drop(columns=columns_to_drop)


def turn_geolevel_into_categorical(dataframe: pd.DataFrame) -> pd.DataFrame:
	cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
	for col in cols:
		dataframe[col] = dataframe[col].astype("category")
	return dataframe

columns_to_drop = ['plan_configuration', "legal_ownership_status", "count_floors_pre_eq"]

#X_train_dropped = drop_unnecessary_columns(dataframe=X_train, columns_to_drop=columns_to_drop)
#X_test_dropped = drop_unnecessary_columns(dataframe=X_test, columns_to_drop=columns_to_drop)
