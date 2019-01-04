import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics
from matplotlib.pyplot import specgram

def create(tr_features,tr_labels,ts_features,ts_labels):
    """
    define neural network through TensorFlow
    :tr_features: input, features of training set, np array
    :tr_labels: input, labels of training set (one-hot encoded), np array
    :ts_features: input, features of test set, np array
    :ts_labels: input, labels of test set (one-hot encoded), np array

    """

    #set parameters
    training_epochs = 50
    n_dim = tr_features.shape[1]
    n_classes = 10
    n_hidden_units_one = 280 
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.01

    #set placeholders for inputs and outputs
    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,n_classes])

    #set weights and biases of layer 1
    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
    #set activation function of layer 1 (sigmoid)
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

    #set weights and biases of layer 2
    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], 
    mean = 0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
    #set activation function of layer 2 (sigmoid)
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

    #set weights and biases of final layer
    W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
    #set activation function of final layer (softmax), y_ is final output
    y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

    init = tf.initialize_all_variables()

    #set cost function
    cost_function = -1*tf.reduce_sum(Y * tf.log(y_))
    #use gradient descent as optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    #set correct prediction variable
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    #set accuracy variable
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost_history = np.empty(shape=[1],dtype=float)
    y_true, y_pred = None, None
    #run network
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):            
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
            cost_history = np.append(cost_history,cost)
        
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
        y_true = sess.run(tf.argmax(ts_labels,1))
        print("Test accuracy: ",round(session.run(accuracy, 
            feed_dict={X: ts_features,Y: ts_labels}),3))

    fig = plt.figure(figsize=(10,8))
    plt.plot(cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()

    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average="micro")
    print("F-Score:", round(f,3))
    return p,r,f,s