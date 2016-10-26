"""
Code for the settings of the network.
"""

class Settings:
    """
    A class for the settings of the network.
    """
    def __init__(self):
        self.network_name = 'go_net'

        # Common settings
        self.batch_size = 3
        self.initial_learning_rate = 0.00001
        self.learning_rate_decay_rate = 0.96
        self.learning_rate_decay_steps = 10000

        # Logging
        self.print_step_period = 1
        self.summary_step_period = 1
        self.validation_step_period = 10
