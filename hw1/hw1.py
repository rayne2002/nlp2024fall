#%% question 2.2
from collections import defaultdict

def extract_unigram_features(premise, hypothesis):
    """
    This function extracts unigram features (BoW) from the premise and hypothesis.
    
    Args:
        premise (str): The premise sentence.
        hypothesis (str): The hypothesis sentence.
        
    Returns:
        features (dict): A dictionary where keys are unigrams and values are their frequencies in the premise and hypothesis.
    """
    
    # Initialize a defaultdict to store the word counts
    features = defaultdict(int)
    
    # Split premise and hypothesis into words (unigrams)
    premise_words = premise.split()
    hypothesis_words = hypothesis.split()
    
    # Count words in the premise
    for word in premise_words:
        features[word] += 1
    
    # Count words in the hypothesis
    for word in hypothesis_words:
        features[word] += 1
    
    return features



#%% question 2.4
import numpy as np
from collections import defaultdict

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(weights, feature_vector):
    """
    Predict the probability that the label is 1.
    """
    z = np.dot(weights, feature_vector)
    return sigmoid(z)

def learn_predictor(train_examples, test_examples, feature_extractor, num_epochs=100, learning_rate=0.01):
    """
    Train a logistic regression model using gradient descent.
    
    Args:
        train_examples: A list of (sentence, label) pairs for training.
        test_examples: A list of (sentence, label) pairs for testing.
        feature_extractor: A function that extracts features from the sentence pairs.
        num_epochs: The number of epochs to run gradient descent.
        learning_rate: The learning rate for gradient descent.
    
    Returns:
        weights: The learned weight vector.
    """
    # Initialize weights (set to zero initially)
    weights = defaultdict(float)
    
    # Extract feature vector size from the training examples
    for epoch in range(num_epochs):
        for sentence_pair, label in train_examples:
            # Extract features for the sentence pair
            features = feature_extractor(sentence_pair[0], sentence_pair[1])
            
            # Compute the dot product between weights and features
            feature_vector = np.array([features[key] for key in sorted(features.keys())])
            
            # Compute the prediction (probability of label being 1)
            prob = predict(weights, feature_vector)
            
            # Compute the gradient (f_w(x) - y) * phi(x)
            error = prob - label
            for key in features:
                weights[key] -= learning_rate * error * features[key]
        
        # Check error rate on the training set
        train_error = evaluate_error_rate(train_examples, weights, feature_extractor)
        test_error = evaluate_error_rate(test_examples, weights, feature_extractor)
        
        # Print epoch stats
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Error: {train_error}, Test Error: {test_error}")
        
        # Early stopping if error rates meet the condition
        if train_error < 0.3 and test_error < 0.4:
            print("Stopping early, error rate condition met.")
            break
    
    return weights

def evaluate_error_rate(examples, weights, feature_extractor):
    """
    Compute the error rate (1 - accuracy) for a set of examples.
    
    Args:
        examples: A list of (sentence, label) pairs.
        weights: The learned weight vector.
        feature_extractor: A function that extracts features from the sentence pairs.
    
    Returns:
        error_rate: The fraction of examples misclassified.
    """
    num_errors = 0
    for sentence_pair, label in examples:
        # Extract features for the sentence pair
        features = feature_extractor(sentence_pair[0], sentence_pair[1])
        feature_vector = np.array([features[key] for key in sorted(features.keys())])
        
        # Compute prediction
        prediction = predict(weights, feature_vector)
        
        # Classify based on threshold 0.5
        predicted_label = 1 if prediction >= 0.5 else 0
        if predicted_label != label:
            num_errors += 1
    
    return num_errors / len(examples)
