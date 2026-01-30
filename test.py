
import pandas as pd
import seaborn as sns
from modelling import train_cv_model
from sklearn.model_selection import train_test_split
from data_processing import winsorize_dataframe, drop_columns, encode_labels, encode_features
from sklearn.metrics import f1_score, accuracy_score

train_X = pd.read_csv("/Users/carricarte/Documents/DSR/data_challenge/EarthquakeNepal/data/train_values.csv")
train_y = pd.read_csv("/Users/carricarte/Documents/DSR/data_challenge/EarthquakeNepal/data/train_labels.csv")

X_train, X_val, y_train, y_val = train_test_split(
    train_X,
    train_y,
    test_size=0.1,
    random_state=42)

encoded_train_y = encode_labels(train_y["damage_grade"])

def processing_dataset(df, y, df_test):

    columns_to_drop = ["count_floors_pre_eq", "building_id", 'plan_configuration',
                       'legal_ownership_status', 'has_secondary_use',
                       'has_secondary_use_agriculture', 'has_secondary_use_hotel',
                       'has_secondary_use_rental', 'has_secondary_use_institution',
                       'has_secondary_use_school', 'has_secondary_use_industry',
                       'has_secondary_use_health_post', 'has_secondary_use_gov_office',
                       'has_secondary_use_use_police', 'has_secondary_use_other']

    df = drop_columns(df, columns_to_drop)
    df = winsorize_dataframe(df)

    return encode_features(df, y, df_test)


train_X_processed, test_X_processed = processing_dataset(train_X, encoded_train_y, X_val)
test_predictions = train_cv_model(train_X_processed, encoded_train_y, test_X_processed)

test_predictions = test_predictions + 1

F1_score = f1_score(y_val["damage_grade"].to_numpy(), test_predictions, average="micro")
print(F1_score)
