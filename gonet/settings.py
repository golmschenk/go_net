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

        # Paths
        self.data_directory = 'data'
        self.logs_directory = 'logs'
        self.models_directory = 'models'

        # AWS specific overriding.
        if self.is_aws_instance():
            self.aws_overrides()

    @staticmethod
    def is_aws_instance():
        """
        Checks if the network is being run on an AWS instance.

        :return: True if on AWS, false otherwise.
        :rtype: bool
        """
        completed_process = subprocess.run(["which", "ec2metadata"])
        return completed_process.returncode == 0

    def aws_overrides(self):
        """
        Updates the settings for running in a AWS instance.
        """
        self.data_directory = '/home/ubuntu/efs/data'
        self.logs_directory = '/home/ubuntu/efs/logs'
        self.models_directory = '/home/ubuntu/efs/models'
