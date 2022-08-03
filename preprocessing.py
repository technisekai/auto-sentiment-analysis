import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# preparing var to stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# preparing normalize word
alay_dict = pd.read_csv('preprocessing_file/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 
                                      1: 'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
# preparing remove stopword
id_stopword_dict = pd.read_csv('preprocessing_file/stopwordbahasa.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

# lowercase
def lowercase(text):
    return text.lower()

# remove uncertain word in sentence
def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('@[^\s]+',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text

# remove non-alphanumeruic   
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    return text

# stemming
def stemming(text):
    return stemmer.stem(text)

# normalization
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

# remove stopword
def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

# preprocessing using function defined above
def preprocess(text):
    text = lowercase(text) # 1
    text = remove_unnecessary_char(text) # 2
    text = remove_nonaplhanumeric(text) # 3
    text = normalize_alay(text) # 4
    text = remove_stopword(text) # 6
    text = stemming(text) # 5
    return text

# class data to num
def num_label(text):
    if text[0] == 'n':
        return -1
    elif text[0] == 'p':
        return 1
    else:
        return 0