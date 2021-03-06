"""
Code for managing the TFRecord data.
"""
import glob
import json
import os
import re
import h5py
import numpy as np
import tensorflow as tf

from gonet.convenience import random_boolean_tensor
from gonet.settings import Settings
from gonet.tfrecords_processor import TFRecordsProcessor


class Data:
    """
    A class for managing the TFRecord data.
    """

    def __init__(self, settings=None):
        if settings:
            self.settings = settings
        else:
            self.settings = Settings()

        self.images = None
        self.labels = None
        self.data_name = None

        # Internal attributes.
        self.dataset_type = None

        os.nice(10)

    @property
    def data_path(self):
        """
        Gives the path to the data file.

        :return: The path to the data file.
        :rtype: str
        """
        return os.path.join(self.settings.data_directory, self.data_name)

    @staticmethod
    def read_and_decode_single_example_from_tfrecords(file_name_queue, data_type=None):
        """
        A definition of how TF should read a single example proto from the file record.

        :param file_name_queue: The file name queue to be read.
        :type file_name_queue: tf.QueueBase
        :param data_type: The dataset type being used in.
        :type data_type: str
        :return: The read file data including the image data and label data.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        go_tfrecords_reader = TFRecordsProcessor()
        image, label = go_tfrecords_reader.create_image_and_label_inputs_from_file_name_queue(file_name_queue,
                                                                                              data_type=data_type)
        image = tf.cast(image, tf.float32)

        return image, label

    def preaugmentation_preprocess(self, image, label):
        """
        Preprocesses the image and label to be in the correct format for training.

        :param image: The image to be processed.
        :type image: tf.Tensor
        :param label: The label to be processed.
        :type label: tf.Tensor
        :return: The processed image and label.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        image = tf.image.resize_images(image, [self.settings.image_height, self.settings.image_width])
        label = tf.image.resize_images(label, [self.settings.image_height, self.settings.image_width])
        return image, label

    @staticmethod
    def postaugmentation_preprocess(image, label):
        """
        Preprocesses the image and label to be in the correct format for training.

        :param image: The image to be processed.
        :type image: tf.Tensor
        :param label: The label to be processed.
        :type label: tf.Tensor
        :return: The processed image and label.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        image = tf.image.per_image_standardization(image)
        return image, label

    def randomly_flip_horizontally(self, image, label):
        """
        Simultaneously and randomly flips the image and label horizontally, such that they still match after flipping.

        :param image: The image to be flipped (maybe).
        :type image: tf.Tensor
        :param label: The label to be flipped (maybe).
        :type label: tf.Tensor
        :return: The image and label which may be flipped.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        image = tf.image.random_flip_left_right(image, seed=0)
        label = tf.image.random_flip_left_right(label, seed=0)
        return image, label

    def augment(self, image, label):
        """
        Augments the data in various ways.

        :param image: The image to be augmented.
        :type image: tf.Tensor
        :param label: The label to be augmented
        :type label: tf.Tensor
        :return: The augmented image and label
        :rtype: (tf.Tensor, tf.Tensor)
        """
        # Add Gaussian noise.
        image = image + tf.random_normal(image.get_shape(), mean=0, stddev=8)

        image, label = self.randomly_flip_horizontally(image, label)

        return image, label

    def create_input_tensors_for_dataset(self, data_type, batch_size):
        """
        Prepares the data inputs.

        :param data_type: The type of data file (usually train, validation, or test).
        :type data_type: str
        :param batch_size: The size of the batches
        :type batch_size: int
        :return: The images and depths inputs.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        file_name_queue = self.attain_file_name_queue(data_type)
        image, label = self.read_and_decode_single_example_from_tfrecords(file_name_queue, data_type=data_type)
        image, label = self.preaugmentation_preprocess(image, label)
        if data_type == 'train':
            image, label = self.augment(image, label)
        image, label = self.postaugmentation_preprocess(image, label)

        if data_type in ['test', 'deploy']:
            images, labels = tf.train.batch(
                [image, label], batch_size=batch_size, num_threads=1, capacity=500 + 3 * batch_size
            )
        else:
            images, labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=4,
                capacity=500 + 3 * batch_size, min_after_dequeue=500
            )

        return images, labels

    def attain_file_name_queue(self, data_type):
        """
        Creates the file name queue for the specified data set.

        :param data_type: The type of dataset being created.
        :type data_type: str
        :return: The file name queue.
        :rtype: tf.QueueBase
        """
        if data_type in ['test', 'deploy']:
            num_epochs = 1
            shuffle = False
        else:
            num_epochs = None
            shuffle = True
        if self.settings.datasets_json:
            file_paths = self.file_names_from_json(data_type)
        else:
            file_paths = self.file_names_for_patterns(data_type)
        file_name_queue = tf.train.string_input_producer(file_paths, num_epochs=num_epochs, shuffle=shuffle)
        return file_name_queue

    def file_names_for_patterns(self, data_type):
        """
        Creates the files names list for the dataset from patterns.

        :param data_type: The type of dataset being created.
        :type data_type: str
        :return: The file names.
        :rtype: list[str]
        """
        if data_type == 'train':
            pattern = self.settings.train_pattern
        elif data_type == 'validation':
            pattern = self.settings.validation_pattern
        elif data_type == 'test':
            pattern = self.settings.test_pattern
        elif data_type == 'deploy':
            pattern = self.settings.deploy_pattern
        else:
            raise ValueError('{} is not a valid data type.'.format(data_type))
        all_file_paths = glob.glob(os.path.join(self.settings.data_directory, '**', '*.tfrecords'),
                                   recursive=True)
        file_paths = [file_path for file_path in all_file_paths if re.search(pattern, os.path.basename(file_path))]
        return file_paths

    def file_names_from_json(self, data_type):
        """
        Creates the files names list for the dataset from JSON.

        :param data_type: The type of dataset being created.
        :type data_type: str
        :return: The file names.
        :rtype: list[str]
        """
        with open(self.settings.datasets_json) as json_file:
            datasets_dictionary = json.load(json_file)
        file_basenames = datasets_dictionary[data_type]
        file_paths = [os.path.join(self.settings.data_directory, basename) for basename in file_basenames]
        return file_paths

    def convert_mat_file_to_numpy_file(self, mat_file_path, number_of_samples=None):
        """
        Generate image and depth numpy files from the passed mat file path.

        :param mat_file_path: The path to the mat file.
        :type mat_file_path: str
        :param number_of_samples: The number of samples to extract.
        :type number_of_samples: int
        """
        mat_data = h5py.File(mat_file_path, 'r')
        images = self.convert_mat_data_to_numpy_array(mat_data, 'images', number_of_samples=number_of_samples)
        images = self.crop_data(images)
        depths = self.convert_mat_data_to_numpy_array(mat_data, 'depths', number_of_samples=number_of_samples)
        depths = self.crop_data(depths)
        basename = os.path.basename(os.path.splitext(mat_file_path)[0])
        data_directory = os.path.dirname(mat_file_path)
        np.save(os.path.join(data_directory, 'images_' + basename) + '.npy', images)
        np.save(os.path.join(data_directory, 'depths_' + basename) + '.npy', depths)

    @staticmethod
    def convert_mat_data_to_numpy_array(mat_data, variable_name_in_mat_data, number_of_samples=None):
        """
        Converts a mat data variable to a numpy array.

        :param mat_data: The mat data containing the variable to be converted.
        :type mat_data: h5py.File
        :param variable_name_in_mat_data: The name of the variable to extract.
        :type variable_name_in_mat_data: str
        :param number_of_samples: The number of samples to extract.
        :type number_of_samples: int
        :return: The numpy array.
        :rtype: np.ndarray
        """
        mat_variable = mat_data.get(variable_name_in_mat_data)
        reversed_array = np.array(mat_variable)
        array = reversed_array.transpose()
        if variable_name_in_mat_data in ('images', 'depths'):
            array = np.rollaxis(array, -1)
        return array[:number_of_samples]

    @staticmethod
    def crop_data(array):
        """
        Crop the NYU data to remove dataless borders.

        :param array: The numpy array to crop
        :type array: np.ndarray
        :return: The cropped data.
        :rtype: np.ndarray
        """
        return array[:, 8:-8, 8:-8]

    def convert_numpy_to_tfrecords(self, images, labels=None):
        """
        Converts numpy arrays to a TFRecords.
        """
        number_of_examples = images.shape[0]
        if labels is not None:
            if labels.shape[0] != number_of_examples:
                raise ValueError("Images count %d does not match label count %d." %
                                 (labels.shape[0], number_of_examples))
            label_shape = labels.shape[1:]
        else:
            label_shape = []
        image_shape = images.shape[1:]
        file_name = os.path.join(self.settings.data_directory, self.data_name + '.tfrecords')
        print('Writing', file_name)
        TFRecordsProcessor().write_from_numpy(file_name, image_shape, images, label_shape, labels)

    def import_mat_file(self, mat_path):
        """
        Imports a Matlab mat file into the data images and labels (concatenating the arrays if they already exists).

        :param mat_path: The path to the mat file to import.
        :type mat_path: str
        """
        with h5py.File(mat_path, 'r') as mat_data:
            uncropped_images = self.convert_mat_data_to_numpy_array(mat_data, 'images')
            images = self.crop_data(uncropped_images)
            uncropped_labels = self.convert_mat_data_to_numpy_array(mat_data, 'depths')
            labels = self.crop_data(uncropped_labels)
            self.images = images
            self.labels = labels

    def import_file(self, file_path):
        """
        Import the data.
        Should be overwritten by subclasses.

        :param file_path: The file path of the file to be imported.
        :type file_path: str
        """
        self.import_mat_file(file_path)

    def convert_to_tfrecords(self):
        """
        Converts the data to a TFRecords file.
        """
        self.convert_numpy_to_tfrecords(self.images, self.labels)

    def generate_all_tfrecords(self):
        """
        Creates the TFRecords for the data.
        """
        import_file_paths = self.attain_import_file_paths()
        if not import_file_paths:
            print('No data found in %s.' % os.path.join(self.settings.data_directory, 'import'))
        for import_file_path in import_file_paths:
            print('Converting %s...' % str(import_file_path))
            self.import_file(import_file_path)
            self.obtain_export_name(import_file_path)
            self.convert_to_tfrecords()

    def obtain_export_name(self, import_file_path):
        """
        Extracts the name to be used for the export file.

        :param import_file_path: The import path.
        :type import_file_path: str | (str, str)
        :return: The name of the export file.
        :rtype: str
        """
        self.data_name = os.path.splitext(os.path.basename(import_file_path))[0]

    def attain_import_file_paths(self):
        """
        Gets a list of all the file paths for files to be imported.

        :return: The list of the file paths to be imported.
        :rtype: list[str]
        """
        import_file_paths = []
        for file_directory, _, file_names in os.walk(self.settings.import_directory):
            mat_names = [file_name for file_name in file_names if file_name.endswith('.mat')]
            for mat_name in mat_names:
                mat_path = os.path.abspath(os.path.join(file_directory, mat_name))
                import_file_paths.append(mat_path)
        return import_file_paths


if __name__ == '__main__':
    data = Data()
    data.generate_all_tfrecords()
