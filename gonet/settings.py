"""
Code for the settings of the network.
"""
import os
import subprocess
import getpass


class Settings:
    """
    A class for the settings of the network.
    """
    def __init__(self):
        self.network_name = 'go_net'

        # Common settings
        self.inference_op_name = 'shallow_net'
        self.batch_size = 3
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_rate = 0.1
        self.learning_rate_decay_steps = 100000
        self.run_mode = 'train'
        self.norbu = False

        # Advanced settings
        self.restore_model_file_name = None
        self.restore_mode = 'continue'  # Should be 'continue' or 'transfer' (for transfer learning).
        self.scopes_to_train = None  # The list of scopes or None for all scopes.
        self.restore_scopes_to_exclude = None

        # Logging and saving
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
        self.import_directory = 'data/import'
        self.restore_checkpoint_directory = None
        # Note, these use regex.
        self.train_pattern = '.*'
        self.validation_pattern = '.*'
        self.test_pattern = None
        self.deploy_pattern = None
        self.datasets_json = None

        # AWS specific overriding.
        if self.is_azure_instance():
            self.azure_overrides()

        # Setup.
        self.create_needed_paths()

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

    @property
    def restore_model_file_path(self):
        """
        The path to the restore model.

        :return: The path to the restore model.
        :rtype: str
        """
        if self.restore_model_file_name:
            return os.path.join(self.models_directory, self.restore_model_file_name)
        else:
            return None

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
        if getpass.getuser() == 'ntsering':
            self.import_directory = '/home/ntsering/storage/import'
            self.data_directory = '/home/ntsering/storage/data'
            self.models_directory = '/home/ntsering/storage/n_models'
            self.logs_directory = '/home/ntsering/storage/n_logs'
        else:
            self.data_directory = '/home/golmschenk/storage/data'
            self.logs_directory = '/home/golmschenk/storage/logs'
            self.models_directory = '/home/golmschenk/storage/models'
            self.import_directory = '/home/golmschenk/storage/import'

    def create_needed_paths(self):
        """
        Creates the logging and model storage paths if they don't exist
        """
        os.makedirs(self.logs_directory, exist_ok=True)
        os.makedirs(self.models_directory, exist_ok=True)
        os.makedirs(self.data_directory, exist_ok=True)

    def __setattr__(self, name, value):
        """
        Altering this to warn about setting attributes not used elsewhere (i.e. warn of setting typo).
        """
        if self.__class__ is not Settings and name not in dir(Settings()):
            print('Warning: \'{}\' is not defined in the base settings class. Perhaps there is a typo?'.format(name))
            input('Click enter to continue anyway.')
        super().__setattr__(name, value)
