import pandas as pd
import numpy as np
from xgboost import XGBClassifier,plot_tree
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, KFold

# # create dummy dataframes
# X_train = pd.DataFrame({
#     #'A' : ['spam', 'eggs', 'spam', 'eggs'] * 6,
#     #'B' : ['alpha', 'beta', 'gamma'] * 8,
#     'D' : np.random.randn(24),
#     'E' : np.random.randint(2,10,24),
#     #'F' : [np.random.choice(['rand_1', 'rand_2', 'rand_4', 'rand_6']) for i in range(24)],
# })

# X_test = pd.DataFrame({
#     #'A' : ['spam', 'eggs', 'spam', 'eggs'] * 6,
#     #'B' : ['alpha', 'beta', 'gamma'] * 8,
#     'D' : np.random.randn(24),
#     'E' : np.random.randint(2,10,24),
#     #'F' : [np.random.choice(['rand_1', 'rand_2', 'rand_4', 'rand_6']) for i in range(24)],
# })

# y = pd.DataFrame({
#     'E' : np.random.randint(0,2,24)
# })

def train_cv_model(X_train, y):

    # model: XGboost
    model=XGBClassifier(max_depth=5)
    model.fit(X_train, y)

    # Define CV strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate model
    scores = cross_val_score(model, X_train, y, cv=kf, scoring='f1_macro')
    print(f"CV Scores: {scores}")
    print(f"Average F1-Macro: {scores.mean():.3f}")
    print(f"Standard Deviation: {scores.std():.3f}")


    # # plot the first tree
    # plot_tree(model, num_trees=0)
    # plt.show()

    # # predict
    # predictions_train = model.predict(X_train)
    # predictions_test = model.predict(X_test)

    # # metrics
    # F1_train = f1_score(y, predictions_train)
    # F1_test = f1_score(y, predictions_test)
    # print(f'The F1 at training is {F1_train}')
    # print(f'The F1 at test is {F1_test}')

    return scores.mean, scores.std
