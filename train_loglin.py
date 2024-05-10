import loglinear as ll
import random
import numpy as np

STUDENT={'name': 'Harel Moshayof',
         'ID': '315073510'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    print(features)
    # vec = np.zeros(len(utils.vocab))
    # for bigram in features:
    #     idx = utils.bigram_index[bigram]
    #     vec[idx] += 1
    return None


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        pass
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
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def _generate_bigrams(seq):
    """
    Generate bigrams from a list of words.
    """
    bigrams = []
    for i in range(len(seq) - 1):
        bigram = (seq[i:i+2], seq[i + 2:i+4])
        bigrams.append(bigram)
    return bigrams

def _read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            label, text = line.strip().split('\t')
            data.append((label, _generate_bigrams(text)))
    return data


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = _read_data('train')
    print(train_data[0])
    dev_data = _read_data('dev')
    num_iterations=10
    learning_rate= 1e-5
    in_dim=10
    out_dim=10
    # ...
   
    # params = ll.create_classifier(in_dim, out_dim)
    # trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

