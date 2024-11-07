import re
from collections import Counter
import math
from numpy import dot
from numpy.linalg import norm
from cont_dict import contractions_dict


def decontract(text):
    words = text.split()
    expanded_words = [contractions_dict.get(word, word) for word in words]
    return ' '.join(expanded_words)


def preprocess(text, exclude_punctuation=False): #rhis function removes punctuation, makes the text in lowercase and returns a list of the words in the text
    text= decontract(text)
    text = text.translate(text.maketrans('', '', r",;:!?./)([]{}+-\\'"))
    if exclude_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    tokens = text.lower().split()
    #print(tokens)
    return tokens


def word_frequency(token_list): #takes the list of words, returns a dictionary with the tokens sorted by their frequency in descending order
    counts = Counter(token_list)
    my_dict= dict(counts)
    vocab = {key: value for key, value in sorted(my_dict.items(), key=lambda my_dict: my_dict[1], reverse=True)}
    #print("vocab111= ",vocab)
    return vocab


def create_mappings(vocab): #gets the dictionary of sorted tokens
    #word_to_int: A dictionary that maps each word to a unique integer which is its index (enumerate), int_to_word is its inverse (inverse as in key and value not order).
    word_to_int = {word: idx for idx, word in enumerate(vocab)}
    int_to_word = {idx: word for word, idx in word_to_int.items()}
    #print("word to int= ",word_to_int)
    #print("int to word= ",int_to_word)
    return word_to_int, int_to_word


def compute_tf(text, word_to_int):
    v=word_frequency(preprocess(text))
    tf = Counter(v)
    #print(v)
    #print("counter= ", tf)
    x={}
    for word, count in tf.items():
        if word in word_to_int:
            x[word_to_int[word]] = count
    #print(x)
    return x

def list_of_sentences(text):
    ls = re.split(r'(?<=[.!?]) +', text)
    return ls

def compute_idf(text, word_to_int):
    ls=list_of_sentences(text)
    n_sentences=len(ls)
    #print("ls= ",ls)
    idf={}
    for word, idx in word_to_int.items():
        n_sentences_w_word = 0
        for s in ls:
            words = preprocess(s)
            if word in words:
                n_sentences_w_word += 1
        if (n_sentences_w_word != 0):
            idf[idx] = math.log((n_sentences / (n_sentences_w_word)))
    #print("IDF= ", idf)
    return idf


def compute_tfidf(tf, idf):
    tfidf = {}
    #print("tf=", tf)
    #print("ifd= ", idf)
    for word, tf_val in tf.items():
        if word in idf:
            tfidf[word] = tf_val * idf[word]
    #print("tfidf= ", tfidf)
    return tfidf

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

