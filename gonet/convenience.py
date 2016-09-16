"""
Code for simplifying generically used functions.
"""
import tensorflow as tf
import math

initial_weight_deviation = 0.01
leaky_relu_leakiness = 0.001


def weight_variable(shape, stddev=initial_weight_deviation):
    """
    Create a generic weight variable.

    :param shape: The shape of the weight variable.
    :type shape: list[int]
    :param stddev: The standard deviation to initialize the weights to.
    :type stddev: float
    :return: The weight variable.
    :rtype: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, constant=initial_weight_deviation):
    """
    Create a generic bias variable.

    :param shape: The shape of the bias variable.
    :type shape: list[int]
    :param constant: The initial value of the biases.
    :type constant: float
    :return: The bias variable.
    :rtype: tf.Variable
    """
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)


def conv2d(images, weights, strides=None):
    """
    Create a generic convolutional operation.

    :param images: The images to prefrom the convolution on.
    :type images: tf.Tensor
    :param weights: The weight variable to be applied.
    :type weights: tf.Variable
    :param strides: The strides to perform the convolution on.
    :type strides: (int, int, int, int)
    :return: The convolutional operation.
    :rtype: tf.Tensor
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    return tf.nn.conv2d(images, weights, strides=strides, padding='SAME')


def leaky_relu(x):
    """
    A basic implementation of a leaky ReLU.

    :param x: The input of the ReLU activation.
    :type x: tf.Tensor
    :return: The tensor filtering on the leaky activation.
    :rtype: tf.Tensor
    """
    return tf.maximum(tf.mul(leaky_relu_leakiness, x), x)


def conv_layer(scope_name, input_tensor, depth_in, depth_out, conv_height=3, conv_width=3, strides=(1, 1, 1, 1),
               histogram_summary=False):
    """
    Adds a convolutional layer with all the toppings.

    :param scope_name: The name of the scope.
    :type scope_name: str
    :param input_tensor: The tensor input to the layer.
    :type input_tensor: tf.Tensor
    :param depth_in: The input depth.
    :type depth_in: int
    :param depth_out: The output_tensor depth.
    :type depth_out: int
    :param conv_height: The height of the convolutions.
    :type conv_height: int
    :param conv_width: The width of the convolutions.
    :type conv_width: int
    :param strides: The striding of the convolutions. Defaults to all 1.
    :type strides: (int, int, int, int)
    :param histogram_summary: Whether or not to show histogram summaries in Tensorboard. Defaults to False.
    :type histogram_summary: bool
    :return: The output/activation tensor.
    :rtype: tf.Tensor
    """
    with tf.name_scope(scope_name):
        weights = weight_variable([conv_height, conv_width, depth_in, depth_out])
        biases = bias_variable([depth_out])
        output_tensor = leaky_relu(conv2d(input_tensor, weights, strides=strides) + biases)
        if histogram_summary:
            tf.histogram_summary(scope_name + '_weights', weights)
            tf.histogram_summary(scope_name + '_activations', output_tensor)
        return output_tensor, weights


def size_from_stride_two(size, iterations=1):
    """
    Provides the appropriate size that will be output with a stride two filter.

    :param size: The original size.
    :type size: int
    :param iterations: The number of times the stride two iteration was preformed.
    :type iterations: int
    :return: The filter output size.
    :rtype: int
    """
    if iterations == 1:
        return math.ceil(size / 2)
    else:
        return math.ceil(size_from_stride_two(size, iterations=iterations - 1) / 2)


def random_boolean_tensor():
    """
    Generates a single element boolean tensor with a random value.

    :return:  The random boolean tensor.
    :rtype: tf.Tensor
    """
    uniform_random = tf.random_uniform([], 0, 1.0)
    return tf.less(uniform_random, 0.5)
