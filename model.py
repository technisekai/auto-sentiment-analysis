import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as m

# classifiers algorithm
classifiers = {
    'Logistic Regression':LogisticRegression(),
    'K Nearest Neighbour':KNeighborsClassifier(8),
    'Decision Tree':DecisionTreeClassifier(),
    'Naive Bayes':GaussianNB(),
    'Support Vector Machine':SVC(),
    'Random Forest': RandomForestClassifier()
}

def build_model(choosed_classifier, X_train, X_test, y_train, y_test):
    # make dataframe for save results of training
    log_cols = ["Classifier", "Accuracy", "Precision Score", "Recall Score", "F1-Score"]
    log = pd.DataFrame(columns=log_cols)
    # itterative modeling
    for choosed in choosed_classifier:
        cls = classifiers[choosed]
        cls = cls.fit(X_train, y_train)
        y_out = cls.predict(X_test)
        # calculate matrics
        accuracy = m.accuracy_score(y_test, y_out)
        precision = m.precision_score(y_test, y_out, average='macro')
        recall = m.recall_score(y_test, y_out, average='macro')
        f1_score = m.f1_score(y_test, y_out, average='macro')
        # append to dataframe
        log_entry = pd.DataFrame([[choosed, accuracy, precision, recall, f1_score]], columns = log_cols)
        log = log.append(log_entry, ignore_index=True)

    return log