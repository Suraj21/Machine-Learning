import os
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer # Lemmatisation process

data_folder = "E:/Class/Data/bbc/"
model_folder = "E:/Class/Data/bbc_models/"
folders = ["business","entertainment","politics","sport","tech"] # Class labels Or Target names

os.chdir(data_folder) # For working directory 

x = []  # input text documents
y = []  # Output 


# This for loop is to read data in a systematic way to have final x and y Refer documentation for os package
for i in folders:
    files = os.listdir(i)
    for text_file in files:
        file_path = i + "/" +text_file
        print ("reading file:", file_path)
        with open(file_path) as f:
            data = f.readlines()
        data = ' '.join(data)
        x.append(data)
        y.append(i)

# Data in DataFrame  and store the data (optional)   
data = {'news': x, 'type': y}       
data = pd.DataFrame(data)
# df = pd.DataFrame(data)
print ('writing csv file ...')
#df.to_csv(data_folder +'dataset.csv', index=False)


"""    
Model preparation
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


# Clean the text using regrex library (re)
def clean_str(string): 
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#data = pd.read_csv(data_folder + 'dataset.csv') # Read the concrete data tha was created from above
x = data['news'].tolist()
y = data['type'].tolist()


wordnet = WordNetLemmatizer() # Define Lemma object

"""
Below for loop will clean every data point, split it in words, apply lemma and rejoin the words to form statement 
All these is done at line 83
"""
for index,value in enumerate(x):
    print ("processing data:",index)
    x[index] = ' '.join([wordnet.lemmatize(word) for word in clean_str(value).split()]) 

vect = TfidfVectorizer(stop_words='english',min_df=2) # define tf*idf vector creation function

X = vect.fit_transform(x) # Create tf*idf vector by fit function 
X.dtype

joblib.dump(vect, model_folder + 'vectorizer.pkl') # Save your tf*idf of your training data
Y = np.array(y)


print ("no of features extracted:",X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print ("train size:", X_train.shape)
print ("test size:", X_test.shape)

model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
model.fit(X_train, y_train)
joblib.dump(model, model_folder + 'news_classifier.pkl') # Save your trained model

y_pred = model.predict(X_test)
c_mat = confusion_matrix(y_test,y_pred)
kappa = cohen_kappa_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print ("Confusion Matrix:\n", c_mat)
print( "\nKappa: ",kappa)
print("\nAccuracy: ",acc)





"""
Create new .py  file for below code use it separately
"""


from sklearn.externals import joblib
#from model import clean_str
#from textblob import Word

vect = joblib.load(model_folder + 'vectorizer.pkl')   #load trained tfidf vectorizer 

model = joblib.load(model_folder + 'news_classifier.pkl') #load your trained classifier model

wordnet = WordNetLemmatizer()
def check_news_type(news_article):  # function to accept raw string and to provide news type(class)
    news_article = [' '.join([wordnet.lemmatize(word) for word in clean_str(news_article).split()])]
    features = vect.transform(news_article)
    return str(model.predict(features)[0])


article = """Ad sales boost Time Warner profit

Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (?600m) for the three months to December, from $639m year-earlier.

The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.

Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding.

Time Warner's fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn. "Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility," chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.

TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake."""


# Prediction
print(check_news_type(article))


