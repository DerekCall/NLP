import streamlit as st
import streamlit.components.v1 as components

import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pyLDAvis
import pyLDAvis.sklearn
# pyLDAvis.enable_notebook()

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#spacy
import spacy
from nltk.corpus import stopwords

#visualizations
import pyLDAvis
# import pyLDAvis.gensim
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvis


st.write("""
# LDA Topic Modeling Visualization
""")

st.title("pyLDAvis")

# st.markdown( "THIS SHOWS I AM A FANCY PERSON" )
st.sidebar.title("""
Topic Names:

1. Mystery/Thriller
2. Contemporary Fiction
3. Classic Literature/Fantasy
4. Self-Improvement
5. Science/Science Fiction
6. Cooking
7. Historical
8. French
9. How-To
10. German
11. Buzzwords
12. Nazi/Horror
""")

HtmlFile = open("lda.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
# print(source_code)
components.html(source_code, height=800)


# html_string = lda.html
# st.html(html_string)


# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

books_df = pd.read_csv('./books_with_blurbs.csv')

corpus = [x for x in books_df.Blurb]
practice_corpus = corpus[:400]

#appending some common words
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(['orson', 'scott', 'card','short','stories'])

# Change max_df, min_df and corpus to run with all documents
vectorizer = CountVectorizer(stop_words=stop_words, min_df=65, max_df=.05)
doc_word = vectorizer.fit_transform(corpus)

tfidf = TfidfVectorizer(**vectorizer.get_params()) 
new_tfidf = tfidf.fit_transform(corpus) 

lsa = TruncatedSVD(8)
doc_topic = lsa.fit_transform(new_tfidf)

lsa = TruncatedSVD(12)
doc_topic = lsa.fit_transform(doc_word)

lda_tf = LatentDirichletAllocation(n_components=12, random_state=0)
lda_tf.fit(new_tfidf)

