import loglinear as ll
import random
import numpy as np

from utils import *

STUDENT={'name': 'Harel Moshayof',
         'ID': '315073510'}

def feats_to_vec(text):
    # YOUR CODE HERE.
    vec = np.zeros(len(F2I))  
    for i in range(len(text)):
        bigram= text[i:i+2]
        if bigram in F2I.keys():  
            vec[F2I[bigram]] += 1  
    return vec



def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        x = feats_to_vec(features)
        y = L2I[label]  
        pred = ll.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad+=1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        
        for label, features in train_data:
            x = feats_to_vec(features) 
            y = L2I[label]                
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for j in range(len(params)):
                params[j] -= learning_rate * grads[j]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params




if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = read_data("train")
    dev_data = read_data("dev")
    num_iterations = 50
    learning_rate= 0.001
    in_dim = len(F2I)
    out_dim = len(L2I)
    # ...
   
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

