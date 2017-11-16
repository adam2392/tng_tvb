import time

import numpy as np
np.random.seed(1234)
from functools import reduce
import math as m

import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
# from utils import augment_EEG, cart2sph, pol2cart

import tensorflow as tf
import keras

# for CNN
from keras.layers import Conv2D, MaxPooling2D

# for RNN
from keras.layers import LSTM

# for utility functionality
from keras.layers import Dense, Dropout, Flatten


# imports tensorflow
from keras import backend as K

class IEEGdnn():
    def __init__(self, imsize=32, n_colors=3)
        # initialize class elements
        self.imsize = imsize
        self.n_colors = n_colors

    def build_cnn(self, input_var=None, w_init=None, n_layers=(4, 2, 1), poolsize=(2,2), n_filters_first=32):
        '''
        Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
        Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
        the number in previous stack.
        input_var: Theano variable for input to the network
        outputs: pointer to the output of the last layer of network (softmax)

        :param input_var: theano variable as input to the network
        :param w_init: Initial weight values
        :param n_layers: number of layers in each stack. An array of integers with each
                        value corresponding to the number of layers in each stack.
                        (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
        :param n_filters_first: number of filters in the first layer
        :param imSize: Size of the image
        :param n_colors: Number of color channels (depth)
        :return: a pointer to the output of last layer
        '''
        # keep weights of all layers
        weights = []

        # initialize counter
        count=0

        # check for weight initialization -> apply Glorotuniform
        if w_init is None:
            w_init = [keras.initializers.glorot_uniform()] * sum(n_layers)

        # set up input layer of CNN
        # network = 

        # add the rest of the hidden layers
        for idx, n_layer in enumerate(n_layers):
            for ilay in range(n_layer):
                network = Conv2D(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3),
                              W=w_init[count], pad='same')
                count+=1

                # add weights 
                weights.append(network.W)
            # create a network at the end with a max pooling
            network = MaxPooling2D(network, pool_size=poolsize)

        return network, weights

    def build_cnn_lstm(self, input_vars, nb_classes,grad_clip=110, imsize=32, n_colors=3, n_timewin=3):
        '''
        Builds the complete network with LSTM layer to integrate time from sequences of EEG images.

        :param input_vars: list of EEG images (one image per time window)
        :param nb_classes: number of classes
        :param grad_clip:  the gradient messages are clipped to the given value during
                            the backward pass.
        :param imsize: size of the input image (assumes a square input)
        :param n_colors: number of color channels in the image
        :param n_timewin: number of time windows in the snippet
        :return: a pointer to the output of last layer
        '''
        # create a list of convnets in time
        convnets = []

        # initialize weights with nothing
        w_init = None

        # build parallel CNNs with shared weights across time
        for i in range(n_timewin):
            if i == 0:
                convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
            else:
                convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
            convnets.append(FlattenLayer(convnet))

        # at this point convnets shape is [numTimeWin][n_samples, features]
        # we want the shape to be [n_samples, features, numTimeWin]
        # convpool = ConcatLayer(convnets)
        # convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
        # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
        convpool = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh)
        # We only need the final prediction, we isolate that quantity and feed it
        # to the next layer.
        convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
        # A fully-connected layer of 256 units with 50% dropout on its inputs:
        convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
        # And, finally, the output layer with 50% dropout on its inputs:
        convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
        return convpool


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim == 4:
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]


def train(images, labels, fold, model_type, batch_size=32, num_epochs=5):
    """
    A sample training function which loops over the training set and evaluates the network
    on the validation set after each epoch. Evaluates the network on the training set
    whenever the
    :param images: input images
    :param labels: target labels
    :param fold: tuple of (train, test) index numbers
    :param model_type: model type ('cnn', '1dconv', 'maxpool', 'lstm', 'mix')
    :param batch_size: batch size for training
    :param num_epochs: number of epochs of dataset to go over for training
    :return: none
    """
    num_classes = len(np.unique(labels))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)
    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    # Prepare Theano variables for inputs and targets
    input_var = T.TensorType('floatX', ((False,) * 5))()
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    # Finally, launch the training loop.
    print("Starting training...")
    best_validation_accu = 0
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        av_train_err = train_err / train_batches
        av_val_err = val_err / val_batches
        av_val_acc = val_acc / val_batches
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(av_train_err))
        print("  validation loss:\t\t{:.6f}".format(av_val_err))
        print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))
        if av_val_acc > best_validation_accu:
            best_validation_accu = av_val_acc
            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            av_test_err = test_err / test_batches
            av_test_acc = test_acc / test_batches
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(av_test_err))
            print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
            # Dump the network weights to a file like this:
            np.savez('weights_lasg_{0}'.format(model_type), *lasagne.layers.get_all_param_values(network))
    print('-'*50)
    print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
    print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))


