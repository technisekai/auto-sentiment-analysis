from numpy import vectorize
import pandas as pd
import os
from model import *
from preprocessing import *
from splitAndVectorization import *
from visualization import *
from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# file upload configuration
UPLOAD_FOLDER = 'datasets/'
ALLOWED_EXTENSION = 'csv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# secure dataset
def secure(dataset):
    filename_validation = secure_filename(dataset)
    path = os.path.join(UPLOAD_FOLDER, filename_validation)
    return path

# classifiers algorithm
classifier = [
    'Logistic Regression',
    'K Nearest Neighbour',
    'Decision Tree',
    'Naive Bayes',
    'Support Vector Machine',
    'Random Forest'
]

# validation for file uploaded
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/')
def home():
    return render_template('index.html', title="Auto Sentiment Analysis")

@app.route('/upload_dataset/', methods=['POST', 'GET'])
def upload_dataset():
    dataset, df = None, None
    # request data
    if request.method == 'POST':
        dataset = request.files['dataset']
    if dataset and allowed_file(dataset.filename):
        # secure filename and save dataset
        PATH_DATASET = secure(dataset.filename)
        dataset.save(PATH_DATASET)
        # read dataset
        df = pd.read_csv(PATH_DATASET)
    return render_template('index.html', df=dict(df.head(5)), path_df=dataset.filename, classifier=classifier, title="Auto Sentiment Analysis")

@app.route('/conf/<path_df>', methods=['POST', 'GET'])
def conf(path_df):
    df, feature_data, class_data, choosed_classifier = None, None, None, None
    # request data
    if request.method == 'POST':
        # split data into feature and class
        feature_data = request.form['feature_data']
        class_data = request.form['class_data']
        # choosed algorithm
        choosed_classifier = request.form.getlist('classifier')
        # read dataset
        PATH_DATASET = secure(path_df)
        df = pd.read_csv(PATH_DATASET)
        # preprocessing
        df['result_of_preprocessing'] = df[feature_data].apply(preprocess)
        # if type of class data is string
        if df[class_data].dtype != 'int64':
            df[class_data] = df[class_data].apply(num_label)
        # split feature and class
        X = df['result_of_preprocessing'] #feature data
        y = df[class_data].values #class data
        X_train, X_test, y_train, y_test = split(X, y)
        # tf idf
        X_train, X_test = vectorization(X_train, X_test)
        # classification
        results = build_model(choosed_classifier, X_train, X_test, y_train, y_test)
    return render_template('index.html', df=dict(df.head(5)), classifier=classifier, results=dict(results), title="Auto Sentiment Analysis")

if __name__ == '__main__':
    app.run(debug=True)