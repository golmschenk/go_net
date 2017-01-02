"""
Code for dealing with reading and interacting with TFRecords outside of the main network.
"""

import numpy as np
import tensorflow as tf


class TFRecordsReader:
    """
    A class for dealing with reading and interacting with TFRecords outside of the main network.
    """
    def convert_to_numpy(self, file_name, data_type=None):
        """
        Reads entire TFRecords file as NumPy.

        :param file_name: The TFRecords file name to read.
        :type file_name: str
        :param data_type: Data type of data. Used if that data type doesn't include things like labels.
        :type data_type: str
        :return: The images and labels NumPy
        :rtype: (np.ndarray, np.ndarray)
        """
        feature_types = self.attain_feature_types(data_type)
        image_tensors = []
        label_tensors = []
        for tfrecord in tf.python_io.tf_record_iterator(file_name):
            features = tf.parse_single_example(tfrecord, features=feature_types)
            image_shape, label_shape = self.extract_shapes_from_tfrecords_features(features, data_type)

            flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
            image_tensor = tf.reshape(flat_image, image_shape)
            image_tensors.append(tf.squeeze(image_tensor))

            if data_type != 'deploy':
                flat_label = tf.decode_raw(features['label_raw'], tf.float32)
                label_tensor = tf.reshape(flat_label, label_shape)
                label_tensors.append(tf.squeeze(label_tensor))
        with tf.Session() as session:
            initialize_op = tf.global_variables_initializer()
            session.run(initialize_op)
            images, labels = session.run([image_tensors, label_tensors])
        return np.stack(images), np.stack(labels)

    @staticmethod
    def attain_feature_types(data_type):
        """
        Get the needed features type dictionary to read the TFRecords.

        :param data_type: The type of data being process. Determines whether to look for labels.
        :type data_type: str
        :return: The feature type dictionary.
        :rtype: dict[str, tf.FixedLenFeature]
        """
        feature_types = {
            'image_height': tf.FixedLenFeature([], tf.int64),
            'image_width': tf.FixedLenFeature([], tf.int64),
            'image_depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
        if data_type != 'deploy':
            feature_types.update({
                'label_height': tf.FixedLenFeature([], tf.int64),
                'label_width': tf.FixedLenFeature([], tf.int64),
                'label_depth': tf.FixedLenFeature([], tf.int64),
                'label_raw': tf.FixedLenFeature([], tf.string)
            })
        return feature_types

    def create_image_and_label_inputs_from_file_name_queue(self, file_name_queue, data_type=None):
        """
        Creates the inputs for the image and label for a given file name queue.

        :param file_name_queue: The file name queue to be used.
        :type file_name_queue: tf.Queue
        :param data_type: The type of data (train, validation, test, deploy, etc) to determine how to process.
        :type data_type: str
        :return: The image and label inputs.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_name_queue)
        feature_types = self.attain_feature_types(data_type)
        features = tf.parse_single_example(serialized_example, features=feature_types)

        image_shape, label_shape = self.extract_shapes_from_tfrecords_features(features, data_type)

        flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(flat_image, image_shape)

        if data_type != 'deploy':
            flat_label = tf.decode_raw(features['label_raw'], tf.float32)
            label = tf.reshape(flat_label, label_shape)
        else:
            # Makes a fake label tensor for preprocessing to work on.
            label = tf.constant(-1.0, dtype=tf.float32, shape=[1, 1, 1])
        return image, label

    @staticmethod
    def extract_shapes_from_tfrecords_features(features, data_type):
        """
        Extracts the image and label shapes from the TFRecords' features. Uses a short TF session to do so.

        :param features: The recovered TFRecords' protobuf features.
        :type features: dict[str, tf.Tensor]
        :param data_type: The type of data (train, validation, test, deploy, etc) to determine how to process.
        :type data_type: str
        :return: The image and label shape tuples.
        :rtype: (int, int, int), (int, int, int)
        """
        image_height_tensor = tf.cast(features['image_height'], tf.int64)
        image_width_tensor = tf.cast(features['image_width'], tf.int64)
        image_depth_tensor = tf.cast(features['image_depth'], tf.int64)
        if data_type == 'deploy':
            label_height_tensor, label_width_tensor, label_depth_tensor = None, None, None  # Line to quiet inspections
        else:
            label_height_tensor = tf.cast(features['label_height'], tf.int64)
            label_width_tensor = tf.cast(features['label_width'], tf.int64)
            label_depth_tensor = tf.cast(features['label_depth'], tf.int64)
        # To read the TFRecords file, we need to start a TF session (including queues to read the file name).
        with tf.Session() as session:
            initialize_op = tf.global_variables_initializer()
            session.run(initialize_op)
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator)
            if data_type == 'deploy':
                image_height, image_width, image_depth = session.run(
                    [image_height_tensor, image_width_tensor, image_depth_tensor])
                label_shape = None
            else:
                image_height, image_width, image_depth, label_height, label_width, label_depth = session.run(
                    [image_height_tensor, image_width_tensor, image_depth_tensor, label_height_tensor,
                     label_width_tensor, label_depth_tensor])
                label_shape = (label_height, label_width, label_depth)
            coordinator.request_stop()
            coordinator.join(threads)
        image_shape = (image_height, image_width, image_depth)
        return image_shape, label_shape
