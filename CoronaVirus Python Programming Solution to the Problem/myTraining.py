import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[: test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    # Reading Data
    df = pd.read_csv('data.csv')
    # Train Test Spliting
    train , test = data_split(df, 0.2)

    X_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    X_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(1848 ,)
    Y_test = test[['infectionProb']].to_numpy().reshape(461 ,)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # open a file where you want to store the data
    file = open("model.pkl", "wb")

    # dump information to the file
    pickle.dump(clf, file)
    file.close()



