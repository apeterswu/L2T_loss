"""
This tutorial introduces the multilayer perceptron using Theano with dynamic loss function.
"""

from __future__ import print_function

import os
import sys
import timeit

import numpy
import argparse
import theano
import theano.tensor as T
import six.moves.cPickle as pickle

from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_raw_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist.pkl.gz',
                    help='dataset for training.')
parser.add_argument('--reload_pretrain', action='store_true', default=False,
                    help='whether to reload the pretrained model.')
parser.add_argument('--reload_pretrain_model', default=None,
                    help='pretrained whole model to reload.')
parser.add_argument('--pretrain_save_to', default='model.pkl',
                    help='file to save the pretrained model.')
parser.add_argument('--reload_model', default=None,
                    help='jointly trained whole model to reload.')
parser.add_argument('--save_to', default='model.pkl',
                    help='file to save the whole model.')
parser.add_argument('--momentum_update', action='store_true', default=False,
                    help='whether to update the loss function through momentum optimizer instead of sgd.')
parser.add_argument('--only_train_model', action='store_true', default=False,
                    help='whether to only train the model (student model).')
parser.add_argument('--no_bias', action='store_true', default=False,
                    help='whether there is no bias vector of loss function.')
parser.add_argument('--cross_valid_measure', action='store_true', default=False,
                    help='whether to use cross validation methods to calculate the valid measure.')
parser.add_argument('--approx_k', type=float, default=50.,
                    help='if validation measure is "approx", set the "k".')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning_rate of model update.')
parser.add_argument('--loss_lr', type=float, default=0.01,
                    help='learning rate of loss function update.')
parser.add_argument('--model_pretrain_iterations', type=int, default=1000,
                    help='iterations of pretraining the model.')
parser.add_argument('--loss_pretrain_iterations', type=int, default=500,
                    help='iterationns of pretraining the loss function.')
parser.add_argument('--model_iterations', type=int, default=200,
                    help='iterations of model training w.r.t loss function.')
parser.add_argument('--meta_iterations', type=int, default=100,
                    help='iterations of meta training w.r.t reverse mode.')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size of data feeding.')
parser.add_argument('--n_in', type=int, default=28*28,
                    help='inpuit dimension w.r.t image or other inputs.')
parser.add_argument('--n_hidden', type=int, default=500,
                    help='hidden dimension in MLP.')
parser.add_argument('--n_out', type=int, default=10,
                    help='output class number.')
parser.add_argument('--save_by_whole', action='store_true', default=False,
                    help='whether to save the loss function model by the whole train/valid accuracy.')
parser.add_argument('--combine_loss', action='store_true', default=False,
                    help='whether to train the student model by MLE and loss function.')
parser.add_argument('--loss_alpha', type=float, default=0.5,
                    help='alpha to control the loss weight of MLE and loss function.')
parser.add_argument('--dynamic', action='store_true', default=False,
                    help='whether to update loss function dynamically.')
args = parser.parse_args()


def save_params(params, save_to):
    all_params = []
    for p in params:
        all_params.append(p.get_value(borrow=True))
    with open(save_to, 'wb') as f:
        pickle.dump(all_params, f)


def load_params(params, load_path):
    with open(load_path, 'rb') as f:
        all_params = pickle.load(f)
    for i, p in enumerate(params):
        p.set_value(all_params[i], borrow=True)


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        Hidden unit activation is given by: tanh(dot(input,W) + b)
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :param n_in: dimensionality of input
        :param n_out: number of hidden units

        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.init_params_values = [W_values, b_values]
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def reset_params(self):
        for i, p in enumerate(self.params):
            p.set_value(self.init_params_values[i], borrow=True)


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        :param n_hidden: number of hidden units
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input

    def reset_params(self):
        self.hiddenLayer.reset_params()
        self.logRegressionLayer.reset_params()

    def measure(self, y, k=args.approx_k):
        if args.measure == "approx":
            # minimize 2*sigmoid(k*(max p_i - p_y)) - 1
            delta = T.max(self.logRegressionLayer.p_y_given_x, axis=1) \
                    - self.logRegressionLayer.p_y_given_x[T.arange(y.shape[0]), y]
            output = 2 * T.nnet.sigmoid(k * delta) - 1
            output_mean = T.mean(output)
            return output_mean
        elif args.measure == "expectation":
            # minimize \sum_i p_i(measure(i,y)), measure(i,y) = 1 if i == y, else 0
            sum_value = T.sum(self.logRegressionLayer.p_y_given_x)
            sum_p_target = T.sum(self.logRegressionLayer.p_y_given_x[T.arange(y.shape[0]), y])
            output = sum_value - sum_p_target
            return output


class LossMLP(object):
    def __init__(self, rng, input, target, n_in, n_hidden, n_out, sigma=0.0001):
        # define (student) model
        self.MLP = MLP(
            rng=rng,
            input=input,
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out
        )

        # define loss function parameters
        W = numpy.eye(n_out, dtype=theano.config.floatX)
        self.W = theano.shared(W, name='w', borrow=True)
        b = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = b
        if not args.no_bias:
            self.b = theano.shared(b, name='b', borrow=True)

        # define the loss computation
        if args.loss_w_out:
            self.pw = T.dot(T.log(self.MLP.logRegressionLayer.p_y_given_x), self.W) + self.b
            self.pwy = self.pw[T.arange(target.shape[0]), target]
            self.loss_cost = -T.mean(T.nnet.sigmoid(self.pwy))  # -sigmoid[(logp)^T W + b]_y
        else:
            self.pw = T.dot(self.MLP.logRegressionLayer.p_y_given_x, self.W) + self.b
            self.pwy = self.pw[T.arange(target.shape[0]), target]
            if args.loss_cost_type == 'sigmoid':
                self.loss_cost = -T.mean(T.log(T.nnet.sigmoid(self.pwy)))  # -log(sigmoid(p^T w) + b)_y
            elif args.loss_cost_type == 'max':
                sigma_ones = T.ones(self.pwy.shape, dtype=theano.config.floatX) * sigma
                self.loss_cost = -T.mean(T.log(T.max([sigma_ones, self.pwy], axis=0)))  # -log(max, (p^T W)_y)
            else:
                print('The loss cost type is not defined. Please specify "sigmoid" or "max" type.')
                exit(1)

        self.MLP_cost = self.MLP.negative_log_likelihood
        self.MLP_errors = self.MLP.errors

        self.MLP_params = self.MLP.params
        self.Loss_params = [self.W]
        if not args.no_bias:
            self.Loss_params += [self.b]
        self.params = self.MLP_params + self.Loss_params


def test_lenet(learning_rate=0.01, loss_learning_rate=0.01,
               model_pretrain_iterations=1000,
               model_iterations=500, meta_iterations=100,
               dataset='mnist.pkl.gz', batch_size=500, n_in=28*28, n_hidden=500, n_out=10,
               nkerns=[20, 50]):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron
 
    This is demonstrated on MNIST.
    :param learning_rate: learning rate used (factor for the stochastic gradient
    :param L1_reg: L1-norm's weight when added to the cost (see regularization)
    :param L2_reg: L2-norm's weight when added to the cost (see regularization)
    :param n_epochs: maximal number of epochs to run the optimizer
    :param dataset: the path of the MNIST dataset file from
                    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

   """
    rng = numpy.random.RandomState(23455)

    datasets = load_raw_data(dataset)

    # dataset are now numpy arrays
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    if args.cross_valid_measure:
        train_valid_set_x = numpy.concatenate((train_set_x, valid_set_x), axis=0)
        train_valid_set_y = numpy.concatenate((train_set_y, valid_set_y), axis=0)
        new_idx = range(len(train_valid_set_x))
        numpy.random.shuffle(new_idx)
        new_train_valid_set_x = train_valid_set_x[new_idx]
        new_train_valid_set_y = train_valid_set_y[new_idx]
        # split the data into 6 folds
        k_fold = 6
        each_fold = 10000
        train_valid_set_train_list = []
        train_valid_set_valid_list = []
        for i in range(k_fold):
            valid_x = new_train_valid_set_x[i*each_fold:(i+1)*each_fold]
            valid_y = new_train_valid_set_y[i*each_fold:(i+1)*each_fold]
            train_valid_set_valid_list.append((valid_x, valid_y))
            train_x = numpy.concatenate((new_train_valid_set_x[:i*each_fold], new_train_valid_set_x[(i+1)*each_fold:]), axis=0)
            train_y = numpy.concatenate((new_train_valid_set_y[:i*each_fold], new_train_valid_set_y[(i+1)*each_fold:]), axis=0)
            train_valid_set_train_list.append((train_x, train_y))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size
    n_test_batches = test_set_x.shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    # define the student model here
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=n_hidden,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=n_out)

    # define the pretrain cost
    model_cost = layer3.negative_log_likelihood(y)

    # define the error computation on valida/test/train dataset w.r.t to model
    model_error = theano.function(
        inputs=[x, y],
        outputs=layer3.errors(y)
    )

    # define the measure of student model
    delta = T.max(layer3.p_y_given_x, axis=1) - layer3.p_y_given_x[T.arange(y.shape[0]), y]
    output = 2 * T.nnet.sigmoid(args.approx_k * delta) - 1
    model_measure = T.mean(output)

    # define the loss parameters and loss computation graph
    #######################################################
    loss_W = numpy.eye(n_out, dtype=theano.config.floatX)
    loss_W = theano.shared(loss_W, name='w', borrow=True)
    loss_b = numpy.zeros((n_out,), dtype=theano.config.floatX)
    if not args.no_bias:
        loss_b = theano.shared(loss_b, name='b', borrow=True)

    if args.dynamic:
        loss_w2 = theano.shared(numpy.zeros((n_out, n_out), dtype=theano.config.floatX), name='loss_w2')
        loss_theta = theano.shared(numpy.zeros((2, ), dtype=theano.config.floatX), name='loss_theta')
        loss_b2 = theano.shared(numpy.zeros((1, ), dtype=theano.config.floatX), name='loss_b2')

    iter_var = T.fscalar('iter_var')
    acc_var = T.fscalar('acc_var')
    # define the loss cost
    if args.dynamic:
        hand_feature = 2 * T.nnet.sigmoid(loss_theta[0] * iter_var + loss_theta[1] * acc_var + loss_b2) - 1
        hand_feature = T.reshape(hand_feature, [1])
        hand_feature_w = loss_w2 * hand_feature
        prob_w = T.dot(T.log(layer3.p_y_given_x), loss_W + hand_feature_w) + loss_b
        prob_wy = prob_w[T.arange(y.shape[0]), y]
        loss_cost = -T.mean(T.nnet.sigmoid(prob_wy))
    else:
        prob_w = T.dot(T.log(layer3.p_y_given_x), loss_W) + loss_b
        prob_wy = prob_w[T.arange(y.shape[0]), y]
        loss_cost = -T.mean(T.nnet.sigmoid(prob_wy))  # -sigmoid[(logp)^T W + b]_y

    #######################################################
    # parameters for student model and teacher model
    student_params = layer3.params + layer2.params + layer1.params + layer0.params
    loss_params = [loss_W]
    if not args.no_bias:
        loss_params += [loss_b]
    if args.dynamic:
        loss_params += [loss_w2, loss_theta, loss_b2]
    all_params = student_params + loss_params

    # try to reload the model
    success_reload_pretrain = False
    if args.reload_pretrain:
        # reload and pretrained model exists
        if os.path.exists(args.reload_pretrain_model):
            print('Reloading the pretrained model.')
            load_params(all_params, args.reload_pretrain_model)
            success_reload_pretrain = True
        else:
            print('The pretrained model does not exist!')

    # define the updates of pre-train the model according to negative log likelihood
    model_pretrain_gparams = [T.grad(model_cost, param) for param in student_params]
    model_pretrain_updates = [(param, param - learning_rate * gparam)
                              for param, gparam in zip(student_params, model_pretrain_gparams)]
    f_pretrain_model = theano.function(
        inputs=[x, y],
        outputs=model_cost,
        updates=model_pretrain_updates
    )

    #######################################################
    # define the updates of train model according to the loss function (jointly training model update)
    # model_gparams = [T.grad(loss_cost, param, disconnected_inputs='ignore') for param in model_params]
    model_gparams = [T.grad(loss_cost, param) for param in student_params]
    model_updates = [(param, param - learning_rate * gparam)
                     for param, gparam in zip(student_params, model_gparams)]
    if args.dynamic:
        f_train_model = theano.function(
            inputs=[x, y, iter_var, acc_var],
            outputs=[loss_cost, model_gparams[0]],
            updates=model_updates
        )
    else:
        f_train_model = theano.function(
            inputs=[x, y],
            outputs=[loss_cost, model_gparams[0]],
            updates=model_updates
        )

    #######################################################
    # define the only_train_model loss and updates
    only_train_loss = args.loss_alpha * model_cost + (1 - args.loss_alpha) * loss_cost
    only_train_gparams = [T.grad(only_train_loss, param) for param in student_params]
    only_train_updates = [(param, param - learning_rate * gparam)
                          for param, gparam in zip(student_params, only_train_gparams)]
    if args.dynamic:
        f_only_train_model = theano.function(
            inputs=[x, y, iter_var, acc_var],
            outputs=only_train_loss,
            updates=only_train_updates
        )
    else:
        f_only_train_model = theano.function(
            inputs=[x, y],
            outputs=only_train_loss,
            updates=only_train_updates
        )
    if args.combine_loss:
        print('Combine mle and loss function, the mle weight alpha is %f' % args.loss_alpha)

    #######################################################
    # define the reverse mode training
    valid_measure = model_measure
    dw_gparams = [T.grad(valid_measure, param) for param in student_params]
    dw_gparams_shared = [theano.shared(p.get_value(borrow=True) * 0.) for p in student_params]  # can access the value
    dw_gparams_shared_valid_update = [(dw_gparam_shared, dw_gparam_shared + dw_gparam)
                                      for dw_gparam_shared, dw_gparam in zip(dw_gparams_shared, dw_gparams)]
    f_measure_dw_gparams = theano.function(
        inputs=[x, y],
        outputs=valid_measure,
        updates=dw_gparams_shared_valid_update
    )

    if args.momentum_update:
        gamma = 0.9
        dl_gparams_accum_shared = [theano.shared(p.get_value(borrow=True) * 0.) for p in loss_params]
    dl_gparams_shared = [theano.shared(p.get_value(borrow=True) * 0.) for p in loss_params]
    dw_gparams_sum = T.sum([T.sum(dw_gparam * model_gparam)
                            for dw_gparam, model_gparam in zip(dw_gparams_shared, model_gparams)])
    dw_ggparams = [T.grad(dw_gparams_sum, param) for param in student_params]
    dl_ggparams = [T.grad(dw_gparams_sum, param) for param in loss_params]
    dw_gparams_updates = [(gparam, gparam - learning_rate * ggparam)
                          for gparam, ggparam in zip(dw_gparams_shared, dw_ggparams)]
    dl_gparams_updates = [(gparam, gparam - learning_rate * ggparam)
                          for gparam, ggparam in zip(dl_gparams_shared, dl_ggparams)]
    if args.dynamic:
        f_dw_dl_updates = theano.function(
            inputs=[x, y, iter_var, acc_var],
            outputs=[],
            updates=dw_gparams_updates + dl_gparams_updates
        )
    else:
        f_dw_dl_updates = theano.function(
            inputs=[x, y],
            outputs=[],
            updates=dw_gparams_updates+dl_gparams_updates
        )

    # """For debugging, print the gradient of dw_gparams"""
    # f_dw_gparams = theano.function(
    #     inputs=[x, y],
    #     outputs=dw_gparams
    # )
    # """For debugging, print the gradient of dw_ggparam and dl_ggparam"""
    # f_dw_dl_ggparams = theano.function(
    #     inputs=[x, y],
    #     outputs=dw_ggparams + dl_ggparams
    # )

    def calcualte_batch_error(data_x, data_y, iter):
        errors = model_error(data_x[iter*batch_size:(iter+1)*batch_size],
                             data_y[iter * batch_size:(iter + 1) * batch_size])
        return errors * 10

    #######################################################
    # define the valid error calculate function
    def calculate_model_error(data_x, data_y, n_batches):
        errors = [model_error(data_x[i*batch_size:(i+1)*batch_size],
                                    data_y[i*batch_size:(i+1)*batch_size])
                        for i in range(n_batches)]
        error_mean = numpy.mean(errors)
        return error_mean

    #######################################################
    # only train the student model according to the fixed loss function.
    if args.only_train_model:
        if os.path.exists(args.reload_model):
            # reload model exists
            print('Reloading the model.')
            load_params(all_params, args.reload_model)
            valid_error_mean = calculate_model_error(valid_set_x, valid_set_y, n_valid_batches)
            test_error_mean = calculate_model_error(test_set_x, test_set_y, n_test_batches)
            print('Reload model valid error: %f, test error: %f' % (valid_error_mean * 100., test_error_mean* 100.))
        else:
            print('The reload model does not exist!')

        best_valid_error = valid_error_mean
        best_test_error = test_error_mean
        best_valid_index = 0
        best_test_index = 0
        validation_frequency = n_train_batches
        train_loss_sum = 0.
        for iteration in range(model_iterations):
            cur_train_x, cur_train_y = train_set_x, train_set_y
            cur_valid_x, cur_valid_y = valid_set_x, valid_set_y
            minibatch_index = iteration % n_train_batches
            if args.dynamic:
                batch_error = calcualte_batch_error(cur_train_x, cur_train_y, minibatch_index)
                iter_input = (iteration / 10000)
            if args.combine_loss:
                if args.dynamic:
                    minibatch_avg_cost = f_only_train_model(
                        cur_train_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                        cur_train_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                        numpy.float32(iter_input), numpy.float32(batch_error))
                else:
                    minibatch_avg_cost = f_only_train_model(
                        cur_train_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                        cur_train_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size])
            else:
                if args.dynamic:
                    minibatch_avg_cost, model_gradients = f_train_model(
                        cur_train_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                        cur_train_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                        numpy.float32(iter_input), numpy.float32(batch_error))
                else:
                    minibatch_avg_cost, model_gradients = f_train_model(
                        cur_train_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                        cur_train_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size])
            train_loss_sum += minibatch_avg_cost
            if iteration % 20 == 0:
                avg_train_loss = train_loss_sum / 20.
                train_loss_sum = 0.
                print('minibatch %i, loss w.r.t loss function: %f, average loss: %f'
                      % (minibatch_index, minibatch_avg_cost, avg_train_loss))
            # evaluate on validation/test dataset after the model update
            if (iteration + 1) % validation_frequency == 0:
                valid_error_mean = calculate_model_error(cur_valid_x, cur_valid_y, n_valid_batches)
                print('%i iteration, Validation error w.r.t to model: %f' % (iteration + 1, valid_error_mean * 100.))
                test_error_mean = calculate_model_error(test_set_x, test_set_y, n_test_batches)
                print('%i iteration, test error w.r.t to model: %f' % (iteration + 1, test_error_mean * 100.))
                if valid_error_mean < best_valid_error:
                    best_valid_error = valid_error_mean
                    best_valid_index = iteration
                if test_error_mean < best_test_error:
                    best_test_error = test_error_mean
                    best_test_index = iteration
        print('Best valid error is at iteration %i, error is %f.' % (best_valid_index, best_valid_error * 100.))
        print('Best test error is at iteration %i, error is %f.' % (best_test_index, best_test_error * 100.))
        return

    print('###n train batches: ', n_train_batches)
    print('###n valid batches: ', n_valid_batches)

    #######################################################
    # not reload the model, then train all the modules from scratch
    if not success_reload_pretrain:
        # the pre-train process of model
        validation_frequency = n_train_batches
        print('Model pretraining start.')
        train_loss_sum = 0.
        for iter in range(model_pretrain_iterations):
            minibatch_index = iter % n_train_batches
            minibatch_avg_cost = f_pretrain_model(
                train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                train_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size])
            train_loss_sum += minibatch_avg_cost

            if iter % 20 == 0:
                avg_train_loss = train_loss_sum / 20.
                train_loss_sum = 0.
                print('model pretrain: minibatch %i, loss %f, average loss: %f'
                      % (minibatch_index, minibatch_avg_cost, avg_train_loss))
            if (iter + 1) % validation_frequency == 0:
                valid_error_mean = calculate_model_error(valid_set_x, valid_set_y, n_valid_batches)
                print('Validation error w.r.t to model: %f' % (valid_error_mean * 100.))
        print('Model pretraining finished.')

        # save the pretrined model
        if args.pretrain_save_to:
            save_params(all_params, args.pretrain_save_to)

    #######################################################
    # Valida/Test/Train error after reload/pretrain the model.
    valid_error_mean = calculate_model_error(valid_set_x, valid_set_y, n_valid_batches)
    print('The pretrained validation error w.r.t to model: %f' % (valid_error_mean * 100.))
    test_error_mean = calculate_model_error(test_set_x, test_set_y, n_test_batches)
    print('The pretrained test error w.r.t to model: %f' % (test_error_mean * 100.))
    train_error_mean = calculate_model_error(train_set_x, train_set_y, n_train_batches)
    print('The pretrained train error w.r.t to model: %f' % (train_error_mean * 100.))

    #######################################################
    # the reverse mode training update phrase
    best_valid_error = valid_error_mean
    best_valid_meta_index = 0
    best_test_error = test_error_mean
    best_test_meta_index = 0
    validation_frequency = n_train_batches
    for miter in range(meta_iterations):
        if args.cross_valid_measure:
            cross_k = numpy.random.randint(k_fold)
            print('k fold is %i' % cross_k)
            cur_train_x, cur_train_y = train_valid_set_train_list[cross_k]
            cur_valid_x, cur_valid_y = train_valid_set_valid_list[cross_k]
        else:
            cur_train_x, cur_train_y = train_set_x, train_set_y
            cur_valid_x, cur_valid_y = valid_set_x, valid_set_y
        print('Meta iteration: %i' % miter)

        train_loss_sum = 0.
        model_params_list = list()  # save the model parameters after each update step
        iter_input_list = list()
        batch_acc_list = list()
        print('Update the model parameters w.r.t loss function. [param, param - learning_rate * gparam]')
        for iter in range(model_iterations):
            # save the parameters into list before update
            model_params_list.append([p.get_value(borrow=True) for p in student_params])

            minibatch_index = iter % n_train_batches
            if args.dynamic:
                batch_error = calcualte_batch_error(cur_train_x, cur_train_y, minibatch_index)
                iter_input = iter / 25000
                iter_input_list.append(iter_input)
                batch_acc_list.append(batch_error)
                minibatch_avg_cost, model_gradients = f_train_model(
                    cur_train_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                    cur_train_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                    numpy.float32(iter_input), numpy.float32(batch_error))
            else:
                minibatch_avg_cost, model_gradients = f_train_model(
                    cur_train_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                    cur_train_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size])
            train_loss_sum += minibatch_avg_cost

            if iter % 20 == 0:
                avg_train_loss = train_loss_sum / 20.
                train_loss_sum = 0.
                print('minibatch %i, loss w.r.t loss function: %f, average loss: %f'
                      % (minibatch_index, minibatch_avg_cost, avg_train_loss))

            # evaluate on validation dataset after the model update
            if (iter + 1) % validation_frequency == 0:
                valid_error_mean = calculate_model_error(cur_valid_x, cur_valid_y, n_valid_batches)
                print('%i iteration, Validation error w.r.t to model: %f' % (iter + 1, valid_error_mean * 100.))

        # update the initialized parameters of dw according to validation measure
        print('Calculate validation measures.')
        valid_measure_value = [f_measure_dw_gparams(cur_valid_x[i*batch_size:(i+1)*batch_size],
                                                    cur_valid_y[i*batch_size:(i+1)*batch_size])
                               for i in range(n_valid_batches)]
        for p in dw_gparams_shared:
            p.set_value(p.get_value(borrow=True) / n_valid_batches, borrow=True)
            # dw_gparams_shared div the batch number of validation data
        valid_measure_mean = numpy.mean(valid_measure_value)
        print('Meta iteration %i, Validation measure: %f' % (miter, valid_measure_mean))

        # dw_gparams_shared now is the first dw according to validation measure
        print('Calculate validation errors.')
        valid_error_mean = calculate_model_error(cur_valid_x, cur_valid_y, n_valid_batches)
        print('Meta iteration %i, Validation error: %f' % (miter, valid_error_mean * 100.))
        if args.save_by_whole:
            train_error_mean = calculate_model_error(cur_train_x, cur_train_y, n_train_batches)
            train_valid_error = (train_error_mean * n_train_batches + valid_error_mean * n_valid_batches) / (n_train_batches + n_valid_batches)
            print('Meta iteration %i, train+valid error: %f' % (miter, train_valid_error * 100.))
            if train_valid_error < best_valid_error:
                best_valid_error = train_valid_error
                best_valid_meta_index = miter
                print('Current meta iteration %i is the best w.r.t train+valid error.' % miter)
                if args.save_to:
                    print('Save loss function (teacher) and model (student) to %s' % args.save_to)
                    save_params(all_params, args.save_to + '.best')
        else:
            if valid_error_mean < best_valid_error:
                best_valid_error = valid_error_mean
                best_valid_meta_index = miter
                print('Current meta iteration %i is the best w.r.t validation error.' % miter)
                if args.save_to:
                    print('Save loss function (teacher) and model (student) to %s' % args.save_to)
                    save_params(all_params, args.save_to+'.best')

        print('Calculate test errors.')
        test_error_mean = calculate_model_error(test_set_x, test_set_y, n_test_batches)
        print('Meta iteration %i, Test error: %f' % (miter, test_error_mean * 100.))
        if test_error_mean < best_test_error:
            best_test_error = test_error_mean
            best_test_meta_index = miter
            print('Current meta iteration %i is the best w.r.t test error.' % miter)

        print('Calculate train model errors.')
        train_error_mean = calculate_model_error(cur_train_x, cur_train_y, n_train_batches)
        print('Meta iteration %i, Training data error: %f' % (miter, train_error_mean * 100.))

        # update the gradients and dw and dl parameters according to the training data form T-1: 0
        print('Reverse model update and calculate gradients.')
        for iter in range(model_iterations)[::-1]:
            model_params_saved = model_params_list[iter]
            for p, p_save in zip(student_params, model_params_saved):
                p.set_value(p_save, borrow=True)  # reset the model parameters w to be the saved old w
            minibatch_index = iter % n_train_batches
            if args.dynamic:
                batch_error = batch_acc_list.pop()
                iter_input = iter_input_list.pop()
                f_dw_dl_updates(cur_train_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                                cur_train_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                                numpy.float32(iter_input), numpy.float32(batch_error))
            else:
                f_dw_dl_updates(cur_train_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                                cur_train_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size])

        # update the parameters of loss function
        if args.momentum_update:
            print('Momentum update the loss function parameters. [param, param - learning_rate * gparam_accum]')
            for dloss_param, dl_gparams_accum in zip(dl_gparams_shared, dl_gparams_accum_shared):
                dl_gparams_accum.set_value(gamma * dl_gparams_accum.get_value(borrow=True) +
                                           (1-gamma) * dloss_param.get_value(borrow=True), borrow=True)
            for loss_param, dl_gparams_accum in zip(loss_params, dl_gparams_accum_shared):
                loss_param.set_value(loss_param.get_value(borrow=True) -
                                     loss_learning_rate * dl_gparams_accum.get_value(borrow=True), borrow=True)
        else:
            print('SGD update the loss function parameters. [param, param - learning_rate * gparam]')
            for loss_param, dloss_param in zip(loss_params, dl_gparams_shared):
                loss_param.set_value(loss_param.get_value(borrow=True) -
                                     loss_learning_rate * dloss_param.get_value(borrow=True), borrow=True)

        print('Set the dw and dl saved value to be 0.')
        for p in dw_gparams_shared:
            p.set_value(p.get_value(borrow=True) * 0., borrow=True)
        for p in dl_gparams_shared:
            p.set_value(p.get_value(borrow=True) * 0., borrow=True)
    if args.save_by_whole:
        print(
            'Best train+valid error is at iteration %i, error is %f.' % (best_valid_meta_index, best_valid_error * 100.))
    else:
        print('Best valid error is at iteration %i, error is %f.' % (best_valid_meta_index, best_valid_error*100.))
    print('Best test error is at iteration %i, error is %f.' % (best_test_meta_index, best_test_error*100.))


if __name__ == '__main__':
    test_lenet(dataset=args.dataset, learning_rate=args.lr, loss_learning_rate=args.loss_lr,
               model_pretrain_iterations=args.model_pretrain_iterations,
               model_iterations=args.model_iterations, meta_iterations=args.meta_iterations,
               batch_size=args.batch_size, n_in=args.n_in,
               n_hidden=args.n_hidden, n_out=args.n_out
               )
