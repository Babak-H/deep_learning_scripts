# goal : create word vectors from game of thrones dataset

from __future__ import absolute_import, division, print_function
import codecs # for word encoding
import glob # for regex
import multiprocessing
import os
import re
import nltk
import gensim.models.word2vec as w2v # main word2vec library
import sklearn.manifold
import numpy as np

# process data
nltk.download("punkt")
nltk.download("stopwords")

# book_filenames = sorted(glob.glob("./*.txt"))
# print(len(book_filenames))

books = ["got1.txt", "got2.txt", "got3.txt", "got4.txt", "got5.txt"]
corpus_raw = u""
for book in books:
    with codecs.open(book, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print(len(corpus_raw))

# download the trained tokenizer model
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
raw_sentences = tokenizer.tokenize(corpus_raw)

def sentences_to_wordlist(raw):
    clean = re.sub('[^a-zA-Z]', ' ', raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentences_to_wordlist(raw_sentence))


print(raw_sentences[5])
print(sentences_to_wordlist(raw_sentences[5]))

len(sentences)
token_count = sum([len(sentence) for sentence in sentences])
print(token_count)

# train word2vec model

# dimensions of the result vectors
num_features = 300
# Ignores all words with total frequency lower than this.
min_word_count = 3
# multi proccessing part
num_workers = multiprocessing.cpu_count()
# how many words to be considered around the current word for the skip-gram model
context_size = 7
# downsampling for frequent words
downsampling = 1e-3
seed = 1

# word2vec model
throne2vec = w2v.Word2Vec(sg=1, seed=seed, workers=num_workers, size=num_features, min_count=min_word_count, window=context_size, sample=downsampling)

# create dictionary of unique words based on our corpus of text
throne2vec.build_vocab(sentences)
print(len(throne2vec.wv.vocab))

throne2vec.train(sentences, total_examples=throne2vec.corpus_count, epochs=20)

# save the trained word2vec model
if not os.path.exists("trained"):
        os.makedirs("trained")
throne2vec.save(os.path.join("trained", "throne2ve.w2v"))


# find semantic similarities between words
# this uses cosine similarity formula
throne2vec.most_similar("Stark")
throne2vec.most_similar("direwolf")
throne2vec.most_similar("Danny")
throne2vec.most_similar("dragon")

#make relationship between two pair of words
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = throne2vec.wv.most_similar_cosmul(positive=[end2, start1],negative=[end1])
    start2 = similarities[0][0]
    print(start1, end1, start2, end2)
    
nearest_similarity_cosmul("Jaime", "sword", "wine")
    
    
    
    