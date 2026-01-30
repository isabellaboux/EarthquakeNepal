# import
from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def winsorize_dataframe(dataframe: pd.DataFrame, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> pd.DataFrame:

	num_cols = [

		col for col in dataframe.select_dtypes(include=[np.number]).columns
		if dataframe[col].nunique() > 2
	]

	winsor_limits = {}

	description = dataframe.describe()

	for col in description.columns:

		mean_ = description[col].iloc[1]
		std_ = description[col].iloc[2]

		if dataframe[col].max() > mean_ + 3*(std_):

			p5 = dataframe[col].quantile(0.05)
			dataframe[col] = dataframe[col].clip(lower=p5)

		if dataframe[col].max() < mean_ - 3*(std_):

			p95 = dataframe[col].quantile(0.95)
			dataframe[col] = dataframe[col].clip(upper=p95)

	return dataframe

def drop_columns(dataframe: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:

	"""
	Docstring for drop_unnecessary_columns

	:param dataframe: Description
	:type dataframe: pd.DataFrame
	:param columns_to_drop: Description
	:type columns_to_drop: list
	:return: Description
	:rtype: DataFrame
	"""

	# columns_name = dataframe.columns()

	# columns_to_drop = columns_name[29::]

	return dataframe.drop(columns=columns_to_drop)



# def target_encode_feature(feature_to_encode, labels):
#
# 	my_encoder = TargetEncoder(smooth="Auto")
# 	return my_encoder.fit_transform(feature_to_encode, labels)


def turn_geolevel_into_categorical(dataframe: pd.DataFrame) -> pd.DataFrame:
	cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
	for col in cols:
		dataframe[col] = dataframe[col].astype(str)
	return dataframe


# def onehot_encoder(feature_to_encode):
# 	my_encoder = OneHotEncoder(handle_unknown="ignore")
# 	return my_encoder.fit_transform(feature_to_encode)


def encode_features(dataframe: pd.DataFrame, y, dataframe_test) -> tuple[Any, Any]:
	cols1 = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
	cols2 = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position"]

	preprocessor = ColumnTransformer(transformers=[
		("cat1", OneHotEncoder(handle_unknown="ignore"), cols1),
		("cat2", TargetEncoder(smooth="auto"), cols2)], remainder="passthrough")
	x_train = preprocessor.fit_transform(dataframe, y)
	x_test = preprocessor.transform(dataframe_test)
	return x_train, x_test



def encode_labels(labels):
	return labels.to_numpy() - 1


