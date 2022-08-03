from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# split data 
def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# vectorization
def vectorization(X_train, X_test):
    tf_idf = TfidfVectorizer()
    tf_idf.fit(X_train)
    # ubah teks ke vektor
    X_train = tf_idf.transform(X_train).toarray()
    X_test = tf_idf.transform(X_test).toarray()
    return X_train, X_test