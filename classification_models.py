import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

df = pd.read_csv("bloggers_and_shooters_final_final_final.csv", encoding = 'latin1')
del df['Unnamed: 0']
df = df.dropna()

shooters_only = df.loc[(df.shooter == 1)]
bloggers_only = df.loc[(df.shooter == 0)]
print("Percentage of words that are written by shooters: ", len(shooters_only.text) / len(df.text) * 100, "%")
print("Percentage of words written by bloggers: ", len(bloggers_only.text)/len(df.text) * 100, "%")

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk import pos_tag

df.shooter = df.shooter.astype(str)

def LiteCleaning(t):
    
    # Lower case text
    t = t.lower()
    
    # Removing Whitespace
    def remove_whitespace(t):
        return " ".join(t.split())
    t = remove_whitespace(t)
    
    # Removing punctuations
    punctuations = string.punctuation
    punctuations = punctuations + string.digits + "’" + '“”' + '—' + "‘"
    table_ = str.maketrans(" ", " ", punctuations)
    t = t.translate(table_)
    
    tokenize = word_tokenize(t)
    
    def to_list(string):
        li = []
        li[:0] = string
        return li
    tokenize = to_list(tokenize)
    
    def lemmatizer(tokenize):
        wordnet = WordNetLemmatizer()
        lemWords = [wordnet.lemmatize(tokenized) for tokenized in tokenize]
        return lemWords    
    lemmed = lemmatizer(tokenize)
    
       # Removing stop words
    stop = 'em', 'male','www.schoolshooters.info', 'peter', 'langman', 'phd', 'version', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october','november', 'december', '2013', '2012', '2011', '2014', '2017', '2016', '2018', '2019', '2020', '2010', '2009','2008', '2007', '2006', '2005', '2004', '2003', '2002', '2001', '2000', '1999', 'bla', 'u', 'yo', 'youre', 'aint', 'ive', 'female', 'im', 'didnt', 'like', 'dont', 'see', 'isnt', 'whenever', 'dont', 'cant', 'way', 'want', 'around', 'everything', 'could', 'become', 'show', 'others', 'see', 'something', 'else', 'make', 'fall', 'often', 'get', 'go', 'take', 'may', 'much', 'anyone', 'ever', 'let', 'try', 'tell', 'give', 'get', 'me-by', 'me-if', 'act','i',  'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    
    final = [l for l in lemmed if not l in stop]
    
    tag_noun = [l for l, pos in pos_tag(final) if pos.startswith("N")]
    tag_verb = [l for l, pos in pos_tag(final) if pos.startswith("V")]
    tag_adj = [l for l, pos in pos_tag(final) if pos.startswith("J")]
    
    words = tag_noun + tag_verb + tag_adj
   
    
    return words

df["text"] = df["text"].apply(LiteCleaning)

train, test = train_test_split(df, test_size = 0.30, random_state = 0)

train_tagged = train.apply(lambda r: 
                           TaggedDocument(words = (r["text"]), 
                                          tags = [r.shooter]), axis = 1)

test_tagged = test.apply(lambda r:
                         TaggedDocument(words = (r["text"]), 
                                        tags = [r.shooter]), axis = 1)

import multiprocessing
cores = multiprocessing.cpu_count()
# Vocabulary

model_dbow = Doc2Vec(dm= 0, vector_size =300, 
                     negative = 5, hs = 0, 
                     min_count = 2, sample = 0,
                     workers = cores)

model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

## Doc2VecModel
# %%time
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                     total_examples = len(train_tagged.values),
                     epochs =1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
    
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score,accuracy_score, plot_confusion_matrix,classification_report
from collections import Counter

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0],
                                 model.infer_vector(doc.words))
                                for doc in sents])
    
    return targets, regressors

y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import make_pipeline

X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)

from sklearn.metrics import make_scorer

## LOGIT
logReg = LogisticRegression(n_jobs = 5, C = 1e5, random_state = 0)
logReg.fit(X_resampled, y_resampled)
y_pred_logit = logReg.predict(X_test)

logi_report = classification_report(y_test, y_pred_logit)
logi_conf_mat = confusion_matrix(y_test, y_pred_logit)

print(f"Logistic Regression Testing Results DBOW:\n", logi_report)
print(f"Confusion Matrix:\n", logi_conf_mat)

## POLY SVM
from sklearn.svm import SVC
svc_gauss = SVC(kernel = 'poly', random_state = 0)
svc_gauss.fit(X_resampled, y_resampled)
y_pred = svc_gauss.predict(X_test)

svm_report = classification_report(y_test, y_pred)
svm_conf_mat = confusion_matrix(y_test, y_pred)

print(f"SVM Testing Results DBOW: \n", svm_report)
print(f"Confusion Matrix \n", svm_conf_mat)


## GRAD BOOST
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.5,
                                max_depth =2, random_state = 0)
gbc.fit(X_resampled, y_resampled)

y_pred = gbc.predict(X_test)

gbc_report = classification_report(y_test, y_pred)
gbc_conf_mat = confusion_matrix(y_test, y_pred)

print(f"Gradient Boosting Report DBOW: \n",gbc_report)
print(f"Confusion Matrix: \n", gbc_conf_mat)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(X_resampled, y_resampled)
y_pred = classifier.predict(X_test)
DTC_report = classification_report(y_test, y_pred)
DTC_conf_mat = confusion_matrix(y_test, y_pred)
print(f"Decision Tree Classification Report DBOW: \n", DTC_report)
print(f"Decision Tree Confusion Matrix: \n", DTC_conf_mat)

# MLPNNET
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=60000, random_state = 0)
mlp.fit(X_resampled, y_resampled)

y_pred = mlp.predict(X_test)

MLP_report = classification_report(y_test, y_pred)
MLP_conf_mat = confusion_matrix(y_test, y_pred)

print(f"Multi-Layer Perceptron Classification Report DBOW: \n", MLP_report)
print(f"Multi-Layer Perceptron Classification Confusion Matrix: \n", MLP_conf_mat)


## DIST MEM
model_dmm = Doc2Vec(dm=1, dm_mean = 1, vector_size = 300, window = 10, negative  =5,
                    min_count = 1, workers = 5, alpha = 0.065, min_alpha = 0.065)

model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

# %%time

for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                    total_examples = len(train_tagged.values), epochs = 1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha
    
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec 

new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

def get_vectors(model, tagged_docs):
    sents = tagged_docs.values    
    targets, regressors = zip(*[(doc.tags[0], 
                                 model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors


y_train, X_train = get_vectors(new_model, train_tagged)
y_test, X_test = get_vectors(new_model, test_tagged)

# Incoporporating upsampling
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)


# Logistic Regression
# Training and Testing
logReg = LogisticRegression(n_jobs = 5, C = 1e5, random_state = 0)
logReg.fit(X_resampled, y_resampled)
y_pred = logReg.predict(X_test)

logi_report = classification_report(y_test, y_pred)
logi_conf_mat = confusion_matrix(y_test, y_pred)

print(f"Logistic Regression Testing Results DBOW + DM:\n", logi_report)
print(f"Confusion Matrix:\n", logi_conf_mat)


## SVM LINEAR
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)

svc_gauss = SVC(kernel = 'linear', random_state = 0)
svc_gauss.fit(X_resampled, y_resampled)
y_pred = svc_gauss.predict(X_test)

svm_report = classification_report(y_test, y_pred)
svm_conf_mat = confusion_matrix(y_test, y_pred)

print(f"SVM Testing Results DBOW + DM: \n", svm_report)
print(f"Confusion Matrix \n", svm_conf_mat)


## GRAD BOOST
from sklearn.ensemble import GradientBoostingClassifier
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)

gbc = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.5,
                                max_depth =2, random_state = 0)
gbc.fit(X_resampled, y_resampled)


y_pred = gbc.predict(X_test)

gbc_report = classification_report(y_test, y_pred)
gbc_conf_mat = confusion_matrix(y_test, y_pred)

print(f"Gradient Boosting Report DBOW + DM: \n",gbc_report)
print(f"Confusion Matrix: \n", gbc_conf_mat)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)

classifier = DecisionTreeClassifier()
classifier.fit(X_resampled, y_resampled)
y_pred = classifier.predict(X_test)

DTC_report = classification_report(y_test, y_pred)
DTC_conf_mat = confusion_matrix(y_test, y_pred)

print(f"Naive Bayes Classification Report DBOW + DM: \n", DTC_report)
print(f"Naive Bayes Confusion Matrix: \n", DTC_conf_mat)


# Nueral Network: Multilayer Perceptron Classifier
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 100), max_iter=6000, random_state = 0)
mlp.fit(X_resampled, y_resampled)

y_pred = mlp.predict(X_test)

MLP_report = classification_report(y_test, y_pred)
MLP_conf_mat = confusion_matrix(y_test, y_pred)

print(f"Multi-Layer Perceptron Classification Report DBOW + DM: \n", MLP_report)
print(f"Multi-Layer Perceptron Classification Confusion Matrix: \n", MLP_conf_mat)
