#Olist product review analysis
#Based on Kaggle user: Nilo K.
#Last modified in 2024-04-18 by ML group 6

#%% import libraries

# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image
import random

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

import textdistance

from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import unicodedata

import warnings
warnings.filterwarnings('ignore')

import os

from transformers import MarianMTModel, MarianTokenizer


#%% translation model initail setup and test

model_name = "Helsinki-NLP/opus-mt-roa-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

src_text = ["Ola mundo!"] #Hello World!

model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]\

print(tgt_text[0])


#%% loading datasets and merge

#Merging order data with review data
order_data = pd.read_csv("olist_order_items_dataset.csv")
review_data = pd.read_csv("olist_order_reviews_dataset.csv")
merged = review_data.merge(order_data, 'left', on='order_id')

#Merging review data with product category
category_data = pd.read_csv('olist_products_dataset.csv')
merged2 = merged.merge(category_data, 'left', on='product_id')

#Final dataset with review_id, order_id, review scores/comments and product categories (both languages)
translation_data = pd.read_csv('product_category_name_translation.csv')
product_review_data = merged2.merge(translation_data, 'left', on='product_category_name')
product_review_data


#%%translating comments in different runs since it takes really long

#translating comments
previous_run = pd.read_csv("translated_list.csv")
translated_comment = previous_run["0"].to_list()
#translated_comment = []
for i in range(len(translated_comment),len(product_review_data["review_comment_message"])):
    if i % 1000 == 0:
        df = pd.DataFrame(translated_comment)
        df.to_csv("translated_list.csv") #save per 1000 comment to prevent crashs
    print("iteration: " + str(i), end='\r')
    if pd.isnull(product_review_data["review_comment_message"][i]):
        translated_comment.append(product_review_data["review_comment_message"][i])
        continue
    else:
        src_text = product_review_data["review_comment_message"][i]
        translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translated_comment.append([tgt_text[0]])
        
# df = pd.DataFrame(translated_comment)
# df.to_csv("translated_list.csv") #export full list

product_review_full = product_review_data.head(len(translated_comment))
product_review_full['review_comment_message'] = translated_comment
product_review_full['review_comment_message'] = product_review_full['review_comment_message'][product_review_full['review_comment_message'].notna()].astype(str)
product_review_full.to_csv("translated_reviews.csv") #export full dataset to prevent re-translating


#%% data frame for analysis

df = pd.read_csv("translated_reviews.csv")
df[['review_score', 'review_comment_title', 'review_comment_message', 'product_category_name_english']].describe(include='all')

print('Comment titles')
print(df['review_comment_title'].value_counts().head())

print('\nScore distribution')
print(df['review_score'].value_counts())

# data cleaning
df['review_comment_title'] = df['review_comment_title'].str.strip().str.lower()
df['product_category_name'] = df['product_category_name'].str.replace('_', ' ').str.lower()

print('\nScore distribution stratisfied split')
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['review_score']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
print(strat_train_set['review_score'].value_counts() / len(strat_train_set['review_score']))

#%% score plot
print('\nScore distribution stratisfied split')
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['review_score']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
print(strat_train_set['review_score'].value_counts() / len(strat_train_set['review_score']))

scores = df['review_score'].value_counts().reset_index()
scores.columns = ['review_scores', 'count']
plt.figure(figsize=(6, 4))
sns.barplot(x='review_scores', y='count', data=scores)
plt.title('Distribution of Review Scores')
plt.xlabel('Review Scores')
plt.ylabel('Count')
plt.show()

#%% comment rate plot

df['review_length'] = df['review_comment_message'].str.len()
df[['review_score', 'review_length', 'review_comment_message']].head()

def comment_rate(df):
    return df['review_length'].count() / len(df)

comment_rates = df.groupby('review_score').apply(comment_rate).reset_index()

comment_rates.columns = ['review_score', 'comment_rate']

plt.figure(figsize=(6, 4))
sns.barplot(x='review_score', y='comment_rate', data=comment_rates)
plt.xlabel('Review Score')
plt.ylabel('Comment Rate')
plt.title('Comment Rate by Review Score')
plt.show()

g = sns.FacetGrid(data=df, col='review_score', hue='review_score', xlim = (0, 500), ylim = (0, 5000))
g.map(plt.hist, 'review_length', bins=100)
g.set_xlabels('Comment Length')
g.set_ylabels('Number of Reviews')
plt.gcf().set_size_inches(12, 5)


#%%tokenization of comments
def remove_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('utf-8')

STOP_WORDS = set(remove_accents(w) for w in nltk.corpus.stopwords.words('english'))
stop_list = ["good", "hey", "yeah", "beautiful", "great", "like", "love", "yes", "perfect","right","mt","kind","super","jewels","fine","oh","xxx","top","blah","dk","sorry","show","words","bad","store","okay","site","service","wrong","order", "xxxx","lights","opinion","products","product","ooo"]
for i in stop_list:
    STOP_WORDS.add(i) # This word is key to understand delivery problems later


def comments_to_words(comment):
    lowered = comment.lower()
    normalized = remove_accents(lowered)
    tokens = nltk.tokenize.word_tokenize(normalized)
    words = tuple(t for t in tokens if t not in STOP_WORDS and t.isalpha())
    return words

def words_to_ngrams(words):
    unigrams, bigrams, trigrams = [], [], []
    for comment_words in words:
        unigrams.extend(comment_words)
        bigrams.extend(' '.join(bigram) for bigram in nltk.bigrams(comment_words))
        trigrams.extend(' '.join(trigram) for trigram in nltk.trigrams(comment_words))
    
    return unigrams, bigrams, trigrams

def plot_freq(tokens, color):
    plt.figure(figsize=(12, 5))
    nltk.FreqDist(tokens).plot(25, cumulative=False, color=color, )

def rgb_float_to_int(rgb):
    return tuple(int(255 * c) for c in rgb)

WORDCLOUD_1S_PALETTE = [rgb_float_to_int(rgb) for rgb in sns.color_palette('Reds', n_colors=9)[2:]]
WORDCLOUD_5S_PALETTE = [rgb_float_to_int(rgb) for rgb in sns.color_palette('Blues', n_colors=9)[2:]]

def get_1s_color(*args, **kwargs):
    return random.choice(WORDCLOUD_1S_PALETTE)

def get_5s_color(*args, **kwargs):
    return random.choice(WORDCLOUD_5S_PALETTE)

def plot_wordcloud(words, style):
    if style == '1s':
        color_function = get_1s_color
        mask_fn = 'dislike.png'
        
    elif style == '5s':
        color_function = get_5s_color
        mask_fn = 'like.png'
        
    mask = np.array(Image.open(f'{mask_fn}'))[:, :, 3]
    mask_icon = mask == 0
    mask_bg = mask > 0
    mask[mask_icon] = 255
    mask[mask_bg] = 0
        
    wordcloud = WordCloud(background_color='white', mask=mask)
    wordcloud.generate(' '.join(words))
    wordcloud.recolor(color_func=color_function)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.gcf().set_size_inches(16, 10)

def n_gram_per_category(df):
    category_ngrams = {}

    for i in df['product_category_name_english'].unique():
        reviews_5s = df[(df['review_score'] == 5) & (df['product_category_name_english'] == i)]
        reviews_1s = df[(df['review_score'] == 1) & (df['product_category_name_english'] == i)]

        unigrams_5s, bigrams_5s, trigrams_5s = words_to_ngrams(reviews_5s['review_comment_words'])
        unigrams_1s, bigrams_1s, trigrams_1s = words_to_ngrams(reviews_1s['review_comment_words'])
        category_ngrams[str(i)+'_5s'] = unigrams_5s, bigrams_5s, trigrams_5s
        category_ngrams[str(i)+'_1s'] = unigrams_1s, bigrams_1s, trigrams_1s

    return category_ngrams

commented_reviews = df[df['review_comment_message'].notnull()].copy()
commented_reviews['review_comment_words'] = commented_reviews['review_comment_message'].apply(comments_to_words)

reviews_5s = commented_reviews[commented_reviews['review_score'] == 5]
reviews_1s = commented_reviews[commented_reviews['review_score'] == 1]

unigrams_5s, bigrams_5s, trigrams_5s = words_to_ngrams(reviews_5s['review_comment_words'])
unigrams_1s, bigrams_1s, trigrams_1s = words_to_ngrams(reviews_1s['review_comment_words'])


#%% plots positive
plot_freq(unigrams_5s, color='green')
plot_freq(bigrams_5s, color='green')
plot_freq(trigrams_5s, color='green')
plot_wordcloud(trigrams_5s, '5s')


#%% plots negative
plot_freq(unigrams_1s, color='red')
plot_freq(bigrams_1s, color='red')
plot_freq(trigrams_1s, color='red')
plot_wordcloud(trigrams_1s, '1s')


#%% plots category positive
plot_freq(n_gram_per_category(commented_reviews)['bed_bath_table_5s'][0], color='green')
plot_freq(n_gram_per_category(commented_reviews)['bed_bath_table_5s'][1], color='green')
plot_freq(n_gram_per_category(commented_reviews)['bed_bath_table_5s'][2], color='green')


#%% plots category negative
plot_freq(n_gram_per_category(commented_reviews)['bed_bath_table_1s'][0], color='red')
plot_freq(n_gram_per_category(commented_reviews)['bed_bath_table_1s'][1], color='red')
plot_freq(n_gram_per_category(commented_reviews)['bed_bath_table_1s'][2], color='red')
# %%
