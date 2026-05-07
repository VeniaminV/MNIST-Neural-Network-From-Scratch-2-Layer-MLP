# MNIST Neural Network 2 layer
# CS 445 programming assignment 1

import numpy as np
# pandas = used to easily read CSV files
import pandas as pd
# matplotlib is used for plotting graphs
import matplotlib.pyplot as plt
# sklearn is used for confusion matrix
from sklearn.metrics import confusion_matrix



# Sigmoid activation function
# maps values to range (0,1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
# used in backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)


# Load MNIST CSV file
# first column = label (0–9)
# rest = 784 pixel values
def load_data(file_path):
    data = pd.read_csv(file_path)

    # .iloc is used to select data by integer position, instead of column names
    y = data.iloc[:, 0].values          # labels
    X = data.iloc[:, 1:].values         # pixel data

    # normalize pixel values to [0,1]
    # this helps with training stability
    X = X / 255.0

    return X, y

# This is added for Experiment 3
# Will create a balanced subset of the data by taking an equal number of samples from each class
# 0.5 means we take half of the data, 0.25 means we take a quarter of the data
def balanced_subset(X, y, fraction):
    num_classes = 10

    # how many samples per class we want
    per_class_limit = int((len(X) / num_classes) * fraction)

    X_new = []
    y_new = []

    # track how many samples we took per digit class (0–9)
    class_counts = [0] * num_classes

    # loop through dataset
    for x, label in zip(X, y):

        # only take sample if we still need more from this class
        if class_counts[label] < per_class_limit:
            X_new.append(x)
            y_new.append(label)
            class_counts[label] += 1

    return np.array(X_new), np.array(y_new)



# Convert labels into target vectors for training
# Each label is turned into a 10-element output vector
# correct class is set to 0.9 and all others to 0.1
def make_targets(y):
    # initialize all targets to 0.1
    #avoids zero gradients which helps with training stability
    targets = np.full((len(y), 10), 0.1)

    #set correct class to 0.9
    for i in range(len(y)):
        targets[i][y[i]] = 0.9

    return targets



# Forward pass prediction
# Goes through the data and returns the predicted class
def predict(X, W1, W2):

    #activation for the hidden layer
    hidden = sigmoid(np.dot(X, W1.T))

    #activation for the output layer
    output = sigmoid(np.dot(hidden, W2.T))

    #return the index of the highest output value as the predicted class
    return np.argmax(output, axis=1)



# Compute accuracy
# compares predicted labels to true labels and returns the percentage correct
def compute_accuracy(X, y, W1, W2):

    preds = predict(X, W1, W2)

    # fraction of correct predictions
    return np.mean(preds == y)


# TRAINING FUNCTION
def train_network(X_train, y_train, X_test, y_test,
                  hidden_size, epochs=50, lr=0.1, momentum=0.9):

    input_size = 784
    output_size = 10

    # initialize weights randomly in small value
    # .uniform will give every value between -0.05 and 0.05 an equal chance of being chosen
    W1 = np.random.uniform(-0.05, 0.05, (hidden_size, input_size))
    W2 = np.random.uniform(-0.05, 0.05, (output_size, hidden_size))

    # store previous updates for momentum
    dW1_prev = np.zeros_like(W1)
    dW2_prev = np.zeros_like(W2)

    # convert labels to target vectors
    T = make_targets(y_train)

    # storing the label values that i need for my plots
    train_acc_list = []
    test_acc_list = []


    # TRAINING LOOP
    for epoch in range(epochs):
        # looping through each training example one by one (stochastic gradient descent)
        for i in range(len(X_train)):
            # input sample
            x = X_train[i]
            #target output
            t = T[i]

            #forward pass
            # ------------------------------------------------------------------

            # compute hidden layer input
            # applying acrivation funciton 
            hidden_input = np.dot(W1, x)
            hidden_output = sigmoid(hidden_input)

            # compute output layer input
            # final output after activation function
            output_input = np.dot(W2, hidden_output)
            output_output = sigmoid(output_input)

            # backpropagation
            #--------------------------------------------------------------

            #error at output layer
            error_output = (t - output_output) * sigmoid_derivative(output_output)
            #error is the difference between target and actual output, multiplied by the derivative of the activation function
            error_hidden = np.dot(W2.T, error_output) * sigmoid_derivative(hidden_output)

            # update weights
            # -------------------------------------------------------------------

            # update hidden to output weights
            dW2 = lr * np.outer(error_output, hidden_output) + momentum * dW2_prev
            # update input to hidden weights
            dW1 = lr * np.outer(error_hidden, x) + momentum * dW1_prev

            # doing the updates
            W2 += dW2
            W1 += dW1

            dW2_prev = dW2
            dW1_prev = dW1

        #evaluate after each epoch
        train_acc = compute_accuracy(X_train, y_train, W1, W2)
        test_acc = compute_accuracy(X_test, y_test, W1, W2)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch {epoch}: Train={train_acc:.4f} Test={test_acc:.4f}")

    # return final weights and accuracy lists for plotting
    return W1, W2, train_acc_list, test_acc_list


# MAIN PROGRAM

# make sure these files are in the same folder
X_train, y_train = load_data("mnist_train.csv")

#for experiment 3, 50% of data , uncomment the line below
#X_train, y_train = balanced_subset(X_train, y_train, 0.5)

#for experiment 3, 25% of data , uncomment the line below
#X_train, y_train = balanced_subset(X_train, y_train, 0.25)

X_test, y_test = load_data("mnist_test.csv")

# number of neurons in hidden layer
# in the assignment this will be the n , hidden size neurons
hidden_size = 100

# train model
W1, W2, train_acc, test_acc = train_network(
    X_train, y_train,
    X_test, y_test,
    hidden_size,
    epochs=50,
    lr=0.1,
    momentum=0.9
)


# Plot accuracy
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Neural Network Accuracy")
plt.legend()
#plt.pause(15)
#plt.show()
plt.savefig("accuracy.png")
plt.close()

# Confusion Matrix (TEST SET)
preds = predict(X_test, W1, W2)
cm = confusion_matrix(y_test, preds)

print("\nConfusion Matrix:")
print(cm)