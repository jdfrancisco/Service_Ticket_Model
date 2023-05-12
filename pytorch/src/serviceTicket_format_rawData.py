import json 
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
nlp = en_core_web_sm.load()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import NMF
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import plot_roc_curve
from pprint import pprint

# Setting max rows and columns
# pd.set_option('max_columns', 50)
# pd.set_option('max_rows', 50)

# Import Textblob for extracting noun phrases
from textblob import TextBlob

# Import pickle to save and load the model
import pickle

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# # Opening JSON file 
# f = open('../Input/complaints-2021-05-14_08_16_.json') 
  
# # returns JSON object as a dictionary 
# data = json.load(f)
# df=pd.json_normalize(data)

# Open a csv file into a dataframe
df = pd.read_csv('../Input/raw_ITtickets.csv', encoding="latin-1")

# Inspect the dataframe to understand the given data.
df.info()

# Print top 5 rows of the data
df.head()

#print the column names
pprint(df.columns)

#Assign new column names
# df.columns = ['index', 'type', 'id', 'score', 'tags', 'zip_code','complaint_id', 'issue', 'date_received',
#        'state', 'consumer_disputed', 'product','company_response', 'company', 'submitted_via',
#        'date_sent_to_company', 'company_public_response','sub_product', 'timely',
#        'complaint_what_happened', 'sub_issue','consumer_consent_provided']

df.columns = ['Index', 'Title', 'Resolution', 'class']

#Assign nan in place of blanks in the complaints column
df[df.loc[:, 'Title'] == ''] = np.nan

# Check if blank values still exist
df[df.loc[:, 'Title'] == '']

pprint(df.shape)

#Remove all rows where complaints column is nan
df = df[~df['Title'].isnull()]

pprint(df.shape)

# Convert Title column to string for performing text operations
df['Title'] = df['Title'].astype(str)

# Write your function here to clean the text and remove all the unnecessary elements.
def clean_text(sent):
    sent = sent.lower() # Text to lowercase
    pattern = '[^\w\s]' # Removing punctuation
    sent = re.sub(pattern, '', sent) 
    pattern = '\w*\d\w*' # Removing words with numbers in between
    sent = re.sub(pattern, '', sent) 
    return sent

df_clean = pd.DataFrame(df['Title'].apply(clean_text))
# df_clean.columns = ['Title']

df_clean

#Write your function to Lemmatize the texts
def lemmmatize_text(text):
    sent = []
    doc = nlp(text)
    for token in doc:
        sent.append(token.lemma_)
    return " ".join(sent)

#Create a dataframe('df_clean') that will have only the complaints and the lemmatized complaints 
df_clean['complaint_lemmatized'] = df_clean['Title'].apply(lemmmatize_text)

df_clean

#Write your function to extract the POS tags 
def get_POS_tags(text):
    sent = []
    blob = TextBlob(text)
    sent = [word for (word,tag) in blob.tags if tag=='NN']
    return " ".join(sent)

# Extract Complaint after removing POS tags
df_clean['complaint_POS_removed'] = df_clean['complaint_lemmatized'].apply(get_POS_tags)

#The clean dataframe should now contain the raw complaint, lemmatized complaint and the complaint after removing POS tags.
df_clean

#Removing -PRON- from the text corpus
df_clean['Complaint_clean'] = df_clean['complaint_POS_removed'].str.replace('-PRON-', '')

# Creating a function to extract top ngrams(unigram/bigram/trigram) based on the function inputs
def get_top_ngrams(text, n=None, ngram=(1,1)):
  vec = CountVectorizer(stop_words='english', ngram_range=ngram).fit(text)
  bagofwords = vec.transform(text)
  sum_words = bagofwords.sum(axis=0)
  words_frequency = [(word, sum_words[0, index]) for word, index in vec.vocabulary_.items()]
  words_frequency = sorted(words_frequency, key = lambda x: x[1], reverse=True)
  return words_frequency[:n]

top_30words = get_top_ngrams(df_clean['Complaint_clean'].values.astype('U'), n=30, ngram=(1,1))
df_unigram = pd.DataFrame(top_30words, columns=['unigram', 'count'])
df_unigram

#Print the top 10 words in the unigram frequency
pprint(df_unigram.head(10))

#Write your code here to find the top 30 bigram frequency among the complaints in the cleaned datafram(df_clean). 
top_30words = get_top_ngrams(df_clean['Complaint_clean'].values.astype('U'), n=30, ngram=(2,2))
df_bigram = pd.DataFrame(top_30words, columns=['bigram', 'count'])
df_bigram

#Print the top 10 words in the bigram frequency
pprint(df_bigram.head(10))

#Write your code here to find the top 30 trigram frequency among the complaints in the cleaned datafram(df_clean). 
top_30words = get_top_ngrams(df_clean['Complaint_clean'].values.astype('U'), n=30, ngram=(3,3))
df_trigram = pd.DataFrame(top_30words, columns=['trigram', 'count'])
df_trigram

#Print the top 10 words in the trigram frequency
pprint(df_trigram.head(10))

df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.replace('xxxx','')

#All masked texts has been removed
df_clean

#Write your code here to initialise the TfidfVectorizer 
tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')

#Write your code here to create the Document Term Matrix by transforming the complaints column present in df_clean.
dtm = tfidf.fit_transform(df_clean['Complaint_clean'])

print(dtm)

#Load your nmf_model with the n_components i.e 5
num_topics = 5

#keep the random_state =40
nmf_model = NMF(n_components=num_topics, random_state=40)

nmf_model.fit(dtm)
print(len(tfidf.get_feature_names_out()))

H = nmf_model.components_       # Topic-term matrix

#Print the Top15 words for each of the topics
words = np.array(tfidf.get_feature_names_out())
topic_words = pd.DataFrame(np.zeros((num_topics, 15)), index=[f'Topic {i + 1}' for i in range(num_topics)],
                           columns=[f'Word {i + 1}' for i in range(15)]).astype(str)
for i in range(num_topics):
    ix = H[i].argsort()[::-1][:15]
    topic_words.iloc[i] = words[ix]

print(topic_words)

# Observation Looking at the topics above, for each topic, we can give a label based on their products/services:

# Example:
# Topic 1 = Computer issues / crashes / freezes / viruses
# Topic 2 = Network drive and email issues
# Topic 3 = Account settings
# Topic 4 = Internet connectivity
# Topic 5 = Other

#Create the best topic for each complaint in terms of integer value 0,1,2,3 & 4
topic_results = nmf_model.transform(dtm)

#Assign the best topic to each of the cmplaints in Topic Column
df_clean['Topic'] = topic_results.argmax(axis=1)

pprint(df_clean.head())

#Print the first 5 Complaint for each of the Topics
df_clean_5=df_clean.groupby('Topic').head(5)
df_clean_5.sort_values('Topic')

#Create the dictionary of Topic names and Topics
Topic_names = { 0:"Computer issues / crashes / freezes / viruses", 1:"Network drive and email issues", 2:"Account settings",
               3:"Internet connectivity", 4:"Other" }
#Replace Topics with Topic Names
df_clean['Topic'] = df_clean['Topic'].map(Topic_names)

df_clean.shape

df_clean.head()

# #Create the dictionary again of Topic names and Topics
# Topic_names = { "Bank account services":0, "Credit card / Prepaid card":1, "Others":2,
#                "Theft/Dispute reporting":3, "Mortgages/loans":4 }
# #Replace Topics with Topic Names
# df_clean['Topic'] = df_clean['Topic'].map(Topic_names)

# df_clean.shape

df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.slice(0,127)

#Keep the columns"Complaint_clean" & "Topic" only in the new dataframe --> training_data
training_data = df_clean[['Complaint_clean', 'Topic']]
training_data.index.name = "Complaint #"
print(training_data)

training_data.to_csv('../Input/input_training_data.csv')
