"""
Code for the settings of the network.
"""
import subprocess


class Settings:
    """
    A class for the settings of the network.
    """
    def __init__(self):
        self.network_name = 'go_net'

        # Common settings
        self.batch_size = 3
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_rate = 0.1
        self.learning_rate_decay_steps = 100000

        # Logging
        self.print_step_period = 1
        self.summary_step_period = 100
        self.validation_step_period = 100
        self.model_auto_save_step_period = 10000
        self.number_of_models_to_keep = 1

        # Data sizes
        # Note, these are sizes *within* the network. Data in TFRecords will be resized to this size.
        self.image_height = 464 // 8
        self.image_width = 624 // 8
        self.image_depth = 3

        # Paths
        self.data_directory = 'data'
        self.logs_directory = 'logs'
        self.models_directory = 'models'
        self.import_directory = 'import'
        # Note, these use glob matching, not regex.
        self.train_patterns = None
        self.validation_patterns = None
        self.test_patterns = None
        self.deploy_patterns = None

        # AWS specific overriding.
        if self.is_azure_instance():
            self.azure_overrides()

        # Internal attributes.
        self._label_height = None
        self._label_width = None
        self._label_depth = None

    @property
    def label_height(self):
        """
        The height of the label data. Defaults to the height of the image.

        :return: Label height.
        :rtype: int
        """
        if self._label_height is None:
            return self.image_height
        return self._label_height

    @label_height.setter
    def label_height(self, value):
        self._label_height = value

    @property
    def label_width(self):
        """
        The width of the label data. Defaults to the width of the image.

        :return: Label width.
        :rtype: int
        """
        if self._label_width is None:
            return self.image_width
        return self._label_width

    @label_width.setter
    def label_width(self, value):
        self._label_width = value

    @property
    def label_depth(self):
        """
        The depth of the label data. Defaults to 1.

        :return: Label depth.
        :rtype: int
        """
        if self._label_depth is None:
            return 1
        return self._label_depth

    @label_depth.setter
    def label_depth(self, value):
        self._label_depth = value

    @property
    def image_shape(self):
        """
        The tuple shape of the image.

        :return: Image shape.
        :rtype: (int, int, int)
        """
        return self.image_height, self.image_width, self.image_depth

    @image_shape.setter
    def image_shape(self, shape):
        self.image_height, self.image_width, self.image_depth = shape

    @property
    def label_shape(self):
        """
        The tuple shape of the label.

        :return: Label shape.
        :rtype: (int, int, int)
        """
        return self.label_height, self.label_width, self.label_depth

    @label_shape.setter
    def label_shape(self, shape):
        self.label_height, self.label_width, self.label_depth = shape

    @staticmethod
    def is_azure_instance():
        """
        Checks if the network is being run on an AWS instance.

        :return: True if on AWS, false otherwise.
        :rtype: bool
        """
        completed_process = subprocess.run(['grep', '-q', 'unknown-245', '/var/lib/dhcp/dhclient.eth0.leases'],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return completed_process.returncode == 0

    def azure_overrides(self):
        """
        Updates the settings for running in a AWS instance.
        """
        self.data_directory = '/home/golmschenk/storage/data'
        self.logs_directory = '/home/golmschenk/storage/logs'
        self.models_directory = '/home/golmschenk/storage/models'
