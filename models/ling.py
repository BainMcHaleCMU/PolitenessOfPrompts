from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def train_svm(train_features):
    X_train = train_features.drop('label', axis=1)
    y_train = train_features['label']
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, test_features):
    X_test = test_features.drop('label', axis=1)
    y_test = test_features['label']
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


