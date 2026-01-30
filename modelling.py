import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, KFold

def train_cv_model(X_train, y, X_test):

    # model: XGboost
    model=XGBClassifier(max_depth=5)
    model.fit(X_train, y)

    # Define CV strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate model
    scores = cross_val_score(model, X_train, y, cv=kf, scoring='f1_macro')
    # print(f"CV Scores: {scores}")
    # print(f"Average F1-Macro: {scores.mean():.3f}")
    # print(f"Standard Deviation: {scores.std():.3f}")


    # # plot the first tree
    # plot_tree(model, num_trees=0)
    # plt.show()

    # # predict
    # predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    # # metrics
    # F1_train = f1_score(y, predictions_train, average='micro')
    #  = f1_score(y, predictions_test, average='micro')
    # print(f'The F1 at training is {F1_train}')
    # print(f'The F1 at test is {F1_test}')
    return predictions_test
    # return scores.mean, scores.std
