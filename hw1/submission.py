import json
import collections
import argparse
import random
import numpy as np

from util import *
import nltk
from nltk import ngrams

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the  
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    words = ex['sentence1'] + ex['sentence2']
    print(words)
    bow_feature = collections.Counter(words)
    return dict(bow_feature)
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    def bigram_feature(sentence):
        bigram_list = []
        for i in range(len(sentence)-1):
            bigram = (sentence[i], sentence[i+1])
            bigram_list.append(bigram)
        return collections.Counter(bigram_list)

    
    
    # # Initialize feature dictionary
    # features = defaultdict(float)
    
    # # Preprocess the sentences
    # premise = ex['sentence1']
    # hypothesis = ex['sentence2']
    # premise = [word.lower() for word in premise if word.lower()]
    # hypothesis = [word.lower() for word in hypothesis if word.lower()]
    
    # # Extract Unigram Features
    # unigram_features = Counter(ngrams(premise + hypothesis, 1))
    # for unigram, count in unigram_features.items():
    #     features[unigram] += count
        
    # # Extract Bigram Features
    # bigram_features = Counter(ngrams(premise + hypothesis, 2))
    # for bigram, count in bigram_features.items():
    #     features[bigram] += count
    
    
    # return features
    #unigram feature extractor
    bow_feature = extract_unigram_features(ex)
    #update to bigram feature extractor
    bow_feature.update(bigram_feature(ex['sentence1']))
    bow_feature.update(bigram_feature(ex['sentence2']))
 
    return bow_feature
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    weights = collections.defaultdict(float)
    for epoch in range(num_epochs):
        # print(f"Starting epoch {epoch + 1}...")
        random.shuffle(train_data)
        for bow in train_data:
            label = bow['gold_label']   #y_i
            bow_feature = feature_extractor(bow)
            probability = predict(weights,bow_feature)    #~=f_w(x)

            gradient = {f: (probability - label) * bow_feature[f] for f in bow_feature}
            increment(weights, gradient, -learning_rate)
    return dict(weights)
    # END_YOUR_CODE
    

def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the dictionary word2ind.
    """
    # BEGIN_YOUR_CODE
    word2ind = {word: idx for idx, word in enumerate(sorted(set(tokens)))}
    
    # Initialize the co-occurrence matrix with zeros
    co_mat = np.zeros((len(word2ind), len(word2ind)), dtype=np.int32)
    
    # Length of the token list
    n_tokens = len(tokens)
    
    # Build the co-occurrence matrix
    for i, token in enumerate(tokens):
        # Current word index
        center_idx = word2ind[token]
        
        # Calculate the bounds for the window around the current word
        left_bound = max(0, i - window_size)
        right_bound = min(n_tokens, i + window_size + 1)
        
        # Iterate over the window around the current word
        for j in range(left_bound, right_bound):
            if i != j:  # Don't count the word co-occurring with itself
                neighbor_idx = word2ind[tokens[j]]
                co_mat[center_idx, neighbor_idx] += 1
                
    return word2ind, co_mat


    # END_YOUR_CODE

def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    # BEGIN_YOUR_CODE
    U, s, Vt = np.linalg.svd(co_mat)
    
    U_k = U[:, :embed_size]
    s_k = s[:embed_size]
    
    S_k_diag = np.diag(np.sqrt(s_k))
    embeddings = np.dot(U_k, S_k_diag)
    
    return embeddings
    # END_YOUR_CODE

def top_k_similar(word_index, embeddings, word2ind, k=10, metric='dot'):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk-words : [str]
    """
    # BEGIN_YOUR_CODE
    if metric == 'dot':
        similarities = np.dot(embeddings, embeddings[word_index])
        
        top_k_indices = np.argsort(similarities)[::-1][1:k+1]
        
        ind2word = {index: word for word, index in word2ind.items()}
        top_k_words = [ind2word[idx] for idx in top_k_indices]
        
        return top_k_words
    elif metric == 'cosine':
        target_vector = embeddings[word_index]
        
        dot_products = np.dot(embeddings, target_vector)
        
        norms = np.linalg.norm(embeddings, axis=1)
        
        similarities = dot_products / (norms * np.linalg.norm(target_vector))
    
        top_k_indices = np.argsort(similarities)[::-1][1:k+1]  # Skip the word itself
        
        ind2word = {index: word for word, index in word2ind.items()}
        top_k_words = [ind2word[idx] for idx in top_k_indices]
        
    return top_k_words

    # END_YOUR_CODE
