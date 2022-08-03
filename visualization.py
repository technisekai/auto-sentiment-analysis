import pandas as pd
import matplotlib.pyplot as plt

def counts_class_bar(df, class_data):
    df[class_data].value_counts().plot(kind='bar')
    plt.savefig('static/visualization/'+class_data+'.png')
    return class_data+'.png'

#def wordcloud(df, feature_data):
#    text_to_string = pd.Series(df[feature_data]).str.cat(sep=' ')