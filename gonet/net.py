"""
Code related to the GoNet class.
"""
import datetime
import multiprocessing
import os
import time
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, summarize_weights, max_pool2d, dropout

from gonet.data import Data
from gonet.interface import Interface
from gonet.convenience import weight_variable, bias_variable, leaky_relu
from gonet.settings import Settings


class Net(multiprocessing.Process):
    """
    The class to build and interact with the GoNet TensorFlow graph.
    """

    def __init__(self, message_queue=None, settings=None):
        super().__init__()
        if settings:
            self.settings = settings
        else:
            self.settings = Settings()

        # Common variables.
        self.data = Data(settings=settings)
        self.dropout_keep_probability = 0.5

        # Logging.
        self.summary_step_period = 1
        self.validation_step_period = 10
        self.step_summary_name = "Loss per pixel"
        self.image_summary_on = True

        # Internal setup.
        self.stop_signal = False
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.saver = None
        self.session = None
        self.dataset_selector_tensor = tf.placeholder(dtype=tf.string)
        self.dropout_keep_probability_tensor = tf.placeholder(tf.float32)
        self.learning_rate_tensor = tf.train.exponential_decay(self.settings.initial_learning_rate,
                                                               self.global_step,
                                                               self.settings.learning_rate_decay_steps,
                                                               self.settings.learning_rate_decay_rate)
        self.queue = message_queue
        self.predicted_test_labels = None
        self.test_step = 0

        os.nice(10)

    @property
    def default_feed_dictionary(self):
        """The default feed dictionary"""
        return {
            self.dropout_keep_probability_tensor: self.dropout_keep_probability,
            self.dataset_selector_tensor: 'train',
        }

    def train(self):
        """
        Adds the training operations and runs the training loop.
        """
        # Prepare session.
        self.session = tf.Session()

        print('Preparing data...')
        # Setup the inputs.
        with tf.variable_scope('Input'):
            images_tensor, labels_tensor = self.create_input_tensors()

        print('Building graph...')
        # Add the forward pass operations to the graph.
        predicted_labels_tensor = self.create_inference_op(images_tensor)

        # Add the loss operations to the graph.
        with tf.variable_scope('loss'):
            loss_tensor = self.create_loss_tensor(predicted_labels_tensor, labels_tensor)
            reduce_mean_loss_tensor = tf.reduce_mean(loss_tensor)
            tf.scalar_summary(self.step_summary_name, reduce_mean_loss_tensor)

        if self.image_summary_on:
            with tf.variable_scope('comparison_summary'):
                self.image_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor, loss_tensor)

        # Add the training operations to the graph.
        training_op = self.create_training_op(value_to_minimize=reduce_mean_loss_tensor)

        # Prepare the summary operations.
        summaries_op = tf.merge_all_summaries()
        summary_path = os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                    datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))
        self.log_source_files(summary_path + '_source')
        train_writer = tf.train.SummaryWriter(summary_path + '_train', self.session.graph)
        validation_writer = tf.train.SummaryWriter(summary_path + '_validation', self.session.graph)

        # The op for initializing the variables.
        initialize_op = tf.global_variables_initializer()

        # Prepare saver.
        self.saver = tf.train.Saver(max_to_keep=self.settings.number_of_models_to_keep)

        print('Initializing graph...')
        # Initialize the variables.
        self.session.run(initialize_op)

        # Restore from saved model if passed.
        if self.settings.restore_model_file_name:
            self.model_restore()

        # Start input enqueue threads.
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coordinator)

        print('Starting training...')
        # Preform the training loop.
        try:
            while not coordinator.should_stop() and not self.stop_signal:
                # Regular training step.
                start_time = time.time()
                _, loss, summaries, step = self.session.run(
                    [training_op, reduce_mean_loss_tensor, summaries_op, self.global_step],
                    feed_dict=self.default_feed_dictionary
                )
                duration = time.time() - start_time

                # Information print step.
                if step % self.settings.print_step_period == 0:
                    print('Step %d: %s = %.5f (%.3f sec / step)' % (
                        step, self.step_summary_name, loss, duration))

                # Summary write step.
                if step % self.settings.summary_step_period == 0:
                    train_writer.add_summary(summaries, step)

                # Validation step.
                if step % self.settings.validation_step_period == 0:
                    start_time = time.time()
                    loss, summaries = self.session.run(
                        [reduce_mean_loss_tensor, summaries_op],
                        feed_dict={**self.default_feed_dictionary,
                                   self.dropout_keep_probability_tensor: 1.0,
                                   self.dataset_selector_tensor: 'validation'}
                    )
                    duration = time.time() - start_time
                    validation_writer.add_summary(summaries, step)
                    print('Validation step %d: %s = %.5g (%.3f sec / step)' % (step, self.step_summary_name,
                                                                               loss, duration))

                if step % self.settings.model_auto_save_step_period == 0 and step != 0:
                    self.save_model()

                # Handle interface messages from the user.
                self.interface_handler()
        except tf.errors.OutOfRangeError as error:
            if self.global_step == 0:
                print('Data not found.')
            else:
                raise error
        finally:
            # When done, ask the threads to stop.
            coordinator.request_stop()

        # Wait for threads to finish.
        coordinator.join(threads)
        self.session.close()

    def model_restore(self):
        """
        Restores a saved model.
        """
        print('Restoring model from %s...' % self.settings.restore_model_file_name)
        if self.settings.restore_mode == 'continue':
            variables_to_restore = None  # All variables.
        elif self.settings.restore_mode == 'transfer':
            # Only restore trainable variabales (i.e. don't restore global step, decayed learning rate, and the like).
            variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            raise ValueError('\'{}\' is not a valid restore mode.'.format(self.settings.restore_mode))
        restorer = tf.train.Saver(var_list=variables_to_restore)
        restorer.restore(self.session, self.settings.restore_model_file_name)

    def create_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images.

        Using `getattr` is typically a bad enough choice, that an explanation is warranted here. During experiments
        inference functions are quickly added, removed, and changed for testing. It's inconvenient to require a case
        switch in this case. This code is only to be run by the researcher (myself) so there are no security issues.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        try:
            inference_op = getattr(self, self.settings.inference_op_name)
        except AttributeError:
            inference_op = getattr(self, 'create_{}_inference_op'.format(self.settings.inference_op_name))
        return tf.identity(inference_op(images), name='inference_op')

    def mercury_module(self, variable_scope, input_tensor, aisle_convolution_depth, spatial_convolution_depth,
                       max_pool_depth, dropout_on=False, normalization_function=None,
                       activation_function=leaky_relu, strided_max_pool_on=False):
        """
        This module has 4 parts. A simple 1x1 dimensionality shift (the aisle convolution), a 1x3 convolution, a 3x1
        convolution, and a 2x2 max pooling with dimensionality shift. All have stride of 1. The outputs of each part are
        concatenated to form an output tensor.

        :param variable_scope: What to name the module scope in the graph.
        :type variable_scope: str
        :param input_tensor: The input tensor to work on.
        :type input_tensor: tf.Tensor
        :param aisle_convolution_depth: The output depth of the 1x1 convolution.
        :type aisle_convolution_depth: int
        :param spatial_convolution_depth: The output depth of the 1x3 and 3x1 convolutions (each).
        :type spatial_convolution_depth: int
        :param max_pool_depth: The output depth of the (dimensional shifted) max pool.
        :type max_pool_depth: int
        :param dropout_on: A boolean to choose whether or not dropout should be applied.
        :type dropout_on: bool
        :param normalization_function: A normalization to be applied before activations. Defaults to batch_norm.
        :type normalization_function: tf.Tensor -> tf.Tensor
        :param activation_function: The activation function to be applied.
        :type activation_function: tf.Tensor -> tf.Tensor
        :param strided_max_pool_on: Whether to include a strided max pool at the end of the module.
        :type strided_max_pool_on: bool
        :return: The output activation tensor.
        :rtype: tf.Tensor
        """
        with tf.variable_scope(variable_scope):
            part1 = convolution2d(input_tensor, aisle_convolution_depth, [1, 1], activation_fn=activation_function,
                                  normalizer_fn=normalization_function)
            part2 = convolution2d(input_tensor, spatial_convolution_depth, [3, 1], activation_fn=activation_function,
                                  normalizer_fn=normalization_function)
            part3 = convolution2d(input_tensor, spatial_convolution_depth, [1, 3], activation_fn=activation_function,
                                  normalizer_fn=normalization_function)
            max_pool_output = max_pool2d(input_tensor, kernel_size=2, stride=1, padding='SAME')
            part4 = convolution2d(max_pool_output, max_pool_depth, [1, 1], activation_fn=activation_function,
                                  normalizer_fn=normalization_function)
            output_tensor = tf.concat(3, [part1, part2, part3, part4])
            output_tensor = self.general_module_end_operations(output_tensor, dropout_on, strided_max_pool_on)
            return output_tensor

    def terra_module(self, variable_scope, input_tensor, convolution_output_depth, kernel_size=3, dropout_on=False,
                     normalization_function=None, activation_function=leaky_relu, strided_max_pool_on=False):
        """
        A basic square 2D convolution layer followed by optional batch norm and dropout.

        :param variable_scope: What to name the module scope in the graph.
        :type variable_scope: str
        :param input_tensor: The input tensor to work on.
        :type input_tensor: tf.Tensor
        :param convolution_output_depth: The output depth of the convolution.
        :type convolution_output_depth: int
        :param kernel_size: The size of the square convolutional kernel.
        :type kernel_size: int
        :param dropout_on: A boolean to choose whether or not dropout should be applied.
        :type dropout_on: bool
        :param normalization_function: A normalization to be applied before activations. Defaults to batch_norm.
        :type normalization_function: tf.Tensor -> tf.Tensor
        :param activation_function: The activation function to be applied.
        :type activation_function: tf.Tensor -> tf.Tensor\
        :param strided_max_pool_on: Whether to include a strided max pool at the end of the module.
        :type strided_max_pool_on: bool
        :return: The output activation tensor.
        :rtype: tf.Tensor
        """
        with tf.variable_scope(variable_scope):
            output_tensor = convolution2d(input_tensor, convolution_output_depth, [kernel_size, kernel_size],
                                          activation_fn=activation_function, normalizer_fn=normalization_function)
            output_tensor = self.general_module_end_operations(output_tensor, dropout_on, strided_max_pool_on)
            return output_tensor

    def general_module_end_operations(self, tensor, dropout_on, strided_max_pool_on):
        """
        Common end of module operations.

        :param tensor: The tensor being processed.
        :type tensor: tf.Tensor
        :param dropout_on: Whether to include dropout or not.
        :type dropout_on: bool
        :param strided_max_pool_on: Whether to include a strided max pool at the end of the module.
        :type strided_max_pool_on: bool
        :return: The processed tensor.
        :rtype: tf.Tensor
        """
        if strided_max_pool_on:
            tensor = max_pool2d(tensor, kernel_size=3, stride=2, padding='VALID')
        if dropout_on:
            tensor = dropout(tensor, self.dropout_keep_probability)
        return tensor

    def create_shallow_net_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a (shallow) deep convolution net.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        module1_output = self.mercury_module('module1', images, 3, 8, 3)
        module2_output = self.mercury_module('module2', module1_output, 8, 16, 16)
        module3_output = self.mercury_module('module3', module2_output, 16, 32, 32)

        predicted_labels = convolution2d(module3_output, 1, kernel_size=1)

        summarize_weights()

        return predicted_labels

    def create_linear_classifier_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using only a linear classifier.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        pixel_count = self.settings.image_height * self.settings.image_width
        flat_images = tf.reshape(images, [-1, pixel_count * self.settings.image_depth])
        weights = weight_variable([pixel_count * self.settings.image_depth, pixel_count], stddev=0.001)
        biases = bias_variable([pixel_count], constant=0.001)

        flat_predicted_labels = tf.matmul(flat_images, weights) + biases
        predicted_labels = tf.reshape(flat_predicted_labels,
                                      [-1, self.settings.image_height, self.settings.image_width, 1])
        return predicted_labels

    def create_loss_tensor(self, predicted_labels, labels):
        """
        Create the loss op and add it to the graph.

        :param predicted_labels: The labels predicted by the graph.
        :type predicted_labels: tf.Tensor
        :param labels: The ground truth labels.
        :type labels: tf.Tensor
        :return: The loss tensor.
        :rtype: tf.Tensor
        """
        return self.create_relative_differences_tensor(predicted_labels, labels)

    @staticmethod
    def create_relative_differences_tensor(predicted_labels, labels):
        """
        Determines the L1 relative differences between two label maps.

        :param predicted_labels: The first label map tensor (usually the predicted labels).
        :type predicted_labels: tf.Tensor
        :param labels: The second label map tensor (usually the actual labels).
        :type labels: tf.Tensor
        :return: The difference tensor.
        :rtype: tf.Tensor
        """
        difference = tf.abs(predicted_labels - labels)
        return difference / labels

    @staticmethod
    def create_absolute_differences_tensor(predicted_labels, labels):
        """
        Determines the L1 absolute differences between two label maps.

        :param predicted_labels: The first label map tensor (usually the predicted labels).
        :type predicted_labels: tf.Tensor
        :param labels: The second label map tensor (usually the actual labels).
        :type labels: tf.Tensor
        :return: The difference tensor.
        :rtype: tf.Tensor
        """
        return tf.abs(predicted_labels - labels)

    def create_training_op(self, value_to_minimize):
        """
        Create and add the training op to the graph.

        :param value_to_minimize: The value to train on.
        :type value_to_minimize: tf.Tensor
        :return: The training op.
        :rtype: tf.Operation
        """
        tf.scalar_summary('Learning rate', self.learning_rate_tensor)
        variables_to_train = self.attain_variables_to_train()
        return tf.train.AdamOptimizer(self.learning_rate_tensor).minimize(value_to_minimize,
                                                                          global_step=self.global_step,
                                                                          var_list=variables_to_train)

    def attain_variables_to_train(self):
        """
        Gets the list of variables to train based on the scopes to train list.

        :return: The list of variables to train.
        :rtype: list[tf.Variable]
        """
        if self.settings.scopes_to_train:
            return [variable for scope in self.settings.scopes_to_train for variable in
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)]
        else:
            return None

    @staticmethod
    def convert_to_heat_map_rgb(tensor):
        """
        Convert a tensor to a heat map.

        :param tensor: The tensor values to be converted.
        :type tensor: tf.Tensor
        :return: The heat map image tensor.
        :rtype: tf.Tensor
        """
        maximum = tf.reduce_max(tensor, reduction_indices=[1, 2, 3], keep_dims=True)
        minimum = tf.reduce_min(tensor, reduction_indices=[1, 2, 3], keep_dims=True)
        ratio = 2 * (tensor - minimum) / (maximum - minimum)
        b = tf.maximum(0.0, (1 - ratio))
        r = tf.maximum(0.0, (ratio - 1))
        g = 1 - b - r
        return (tf.concat(3, [r, g, b]) * 2) - 1

    def image_comparison_summary(self, images, labels, predicted_labels, label_differences):
        """
        Combines the image, label, and difference tensors together into a presentable image. Then adds the
        image summary op to the graph.

        :param images: The original image.
        :type images: tf.Tensor
        :param labels: The tensor containing the actual label values.
        :type labels: tf.Tensor
        :param predicted_labels: The tensor containing the predicted labels.
        :type predicted_labels: tf.Tensor
        :param label_differences: The tensor containing the difference between the actual and predicted labels.
        :type label_differences: tf.Tensor
        """
        concatenated_labels = tf.concat(1, [labels, predicted_labels, label_differences])
        concatenated_heat_maps = self.convert_to_heat_map_rgb(concatenated_labels)
        display_images = tf.div(images, tf.reduce_max(tf.abs(images)))
        comparison_image = tf.concat(1, [display_images, concatenated_heat_maps])
        tf.image_summary('comparison', comparison_image)

    def interface_handler(self):
        """
        Handle input from the user using the interface.
        """
        if self.queue:
            if not self.queue.empty():
                message = self.queue.get(block=False)
                if message == 'save':
                    self.save_model()
                elif message == 'quit':
                    self.stop_signal = True

    def save_model(self):
        """
        Saves the current graph model.
        """
        save_path = self.saver.save(self.session,
                                    os.path.join(self.settings.models_directory, self.settings.network_name + '.ckpt'),
                                    global_step=self.global_step)
        tf.train.write_graph(self.session.graph_def, self.settings.models_directory, self.settings.network_name + '.pb')
        print('Model saved in file: %s' % save_path)

    def create_feed_selectable_input_tensors(self, dataset_dictionary):
        """
        Creates images and label tensors which are placed within a cond statement to allow switching between datasets.
        A feed input into the network execution is added to allow for passing the name of the dataset to be used in a
        particular step.

        :param dataset_dictionary: A dictionary containing as keys the names of the datasets and as values a pair with
                                   containing the images and labels of that dataset.
        :type dataset_dictionary: dict[str, (tf.Tensor, tf.Tensor)]
        :return: The general images and labels tensor produced by the case statement, as well as the selector tensor.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        images_tensor, labels_tensor = tf.cond(tf.equal(self.dataset_selector_tensor, 'validation'),
                                               lambda: dataset_dictionary['validation'],
                                               lambda: dataset_dictionary['train'])
        return images_tensor, labels_tensor

    def create_input_tensors(self):
        """
        Create the image and label tensors for each dataset and produces a selector tensor to choose between datasets
        during runtime.

        :return: The general images and labels tensors which are conditional on a selector tensor.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        training_images_tensor, training_labels_tensor = self.data.create_input_tensors_for_dataset(
            data_type='train',
            batch_size=self.settings.batch_size
        )
        validation_images_tensor, validation_labels_tensor = self.data.create_input_tensors_for_dataset(
            data_type='validation',
            batch_size=self.settings.batch_size
        )
        images_tensor, labels_tensor = self.create_feed_selectable_input_tensors(
            {
                'train': (training_images_tensor, training_labels_tensor),
                'validation': (validation_images_tensor, validation_labels_tensor)
            }
        )
        return images_tensor, labels_tensor

    def create_test_dataset_input_tensors(self):
        """
        Creates the images input tensor for the test dataset.

        :return: The images and labels tensors for the test dataset.
        :rtype: tf.Tensor, tf.Tensor
        """
        images_tensor, labels_tensor = self.data.create_input_tensors_for_dataset(data_type='test',
                                                                                  batch_size=self.settings.batch_size)
        # Attach names to the tensors.
        images_tensor = tf.identity(images_tensor, name='images_input_op')
        labels_tensor = tf.identity(labels_tensor, name='labels_input_op')
        return images_tensor, labels_tensor

    def run(self):
        """
        Allow for training the network from a multiprocessing standpoint.
        """
        self.train()

    def test(self):
        """
        Use a trained model to predict labels for a test set of images.
        """
        if self.settings.restore_model_file_name is None:
            self.settings.restore_model_file_name = self.attain_latest_model_path()
            if not self.settings.restore_model_file_name:
                print('No model to restore from found.')
                return

        print('Preparing data...')
        # Setup the inputs.
        images_tensor, labels_tensor = self.create_test_dataset_input_tensors()

        print('Building graph...')
        # Add the forward pass operations to the graph.
        self.create_inference_op(images_tensor)

        # The op for initializing the variables.
        initialize_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())

        # Prepare the saver.
        saver = tf.train.Saver()

        # Create a session for running operations in the Graph.
        self.session = tf.Session()

        print('Running prediction...')
        # Initialize the variables.
        self.session.run(initialize_op)

        # Load model.
        print('Restoring model from {model_file_path}...'.format(model_file_path=self.settings.restore_model_file_name))
        saver.restore(self.session, self.settings.restore_model_file_name)

        # Start input enqueue threads.
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coordinator)

        self.test_run_preloop()

        # Preform the prediction loop.
        try:
            while not coordinator.should_stop() and not self.stop_signal:
                self.test_run_loop_step()
                self.test_step += 1
        except tf.errors.OutOfRangeError:
            if self.test_step == 0:
                print('Data not found.')
            else:
                print('Done predicting after %d steps.' % self.test_step)
        finally:
            # When done, ask the threads to stop.
            coordinator.request_stop()

        self.test_run_postloop()

        # Wait for threads to finish.
        coordinator.join(threads)
        self.session.close()

    def test_run_preloop(self):
        """
        The code run before the test loop. Mostly for setting up things that will be used within the loop.
        """
        self.predicted_test_labels = np.ndarray(shape=[0] + list(self.settings.label_shape), dtype=np.float32)

    def test_run_loop_step(self):
        """
        The code that will be used during the each iteration of the test loop (excluding the step incrementation).
        """
        predicted_labels_tensor = self.session.graph.get_tensor_by_name('inference_op:0')
        predicted_labels_batch = self.session.run(
            predicted_labels_tensor,
            feed_dict={**self.default_feed_dictionary, self.dropout_keep_probability_tensor: 1.0}
        )
        self.predicted_test_labels = np.concatenate((self.predicted_test_labels, predicted_labels_batch))
        print('{image_count} images processed.'.format(image_count=(self.test_step + 1) * self.settings.batch_size))

    def test_run_postloop(self):
        """
        The code that will be run once the inference test loop is finished. Mostly for saving data or statistics.
        """
        predicted_labels_save_path = os.path.join(self.settings.data_directory, 'predicted_labels')
        print('Saving labels to {}.npy...'.format(predicted_labels_save_path))
        np.save(predicted_labels_save_path, self.predicted_test_labels)

    def attain_latest_model_path(self):
        """
        Determines the model path for the model which matches the network name and has the highest step label.

        :return: The model path.
        :rtype: str
        """
        latest_model_name = None
        latest_model_step = -1
        for file_name in os.listdir(self.settings.models_directory):
            if self.settings.network_name + '.ckpt' in file_name and 'meta' not in file_name:
                number_start_index = file_name.index('ckpt-') + len('ckpt-')
                model_step = int(file_name[number_start_index:])
                if model_step > latest_model_step:
                    latest_model_step = model_step
                    latest_model_name = file_name
        if not latest_model_name:
            return
        return os.path.join(self.settings.models_directory, latest_model_name)

    def log_source_files(self, output_file_name):
        """
        Takes all the Python and txt files in the working directory and compresses them into a zip file.

        :param output_file_name: The name of the output file.
        :type output_file_name: str
        """
        file_names_to_zip = []
        for file_name in os.listdir('.'):
            if file_name.endswith(".py") or file_name.endswith('.txt'):
                file_names_to_zip.append(file_name)
        with ZipFile(output_file_name + '.zip', 'w') as zip_file:
            for file_name in file_names_to_zip:
                zip_file.write(file_name)
            data_tree_string = self.generate_directory_tree_string(self.settings.data_directory)
            zip_file.writestr('data_tree.txt', data_tree_string)

    @staticmethod
    def generate_directory_tree_string(directory):
        """
        Creates a string to display the file tree of a given directory.

        :param directory: The directory (path) to show the tree of.
        :type directory: str
        :return: The string containing a tree display of the contained files.
        :rtype: str
        """
        tree_string = ''
        for root, _, files in os.walk(directory):
            level = root.replace(directory, '').count(os.sep)
            indent = ' ' * 4 * level
            tree_string += '{}{}/\n'.format(indent, os.path.basename(root))
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                tree_string += '{}{}\n'.format(sub_indent, f)
        return tree_string


if __name__ == '__main__':
    interface = Interface(network_class=Net)
    interface.run()
