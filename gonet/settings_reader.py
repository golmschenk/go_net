"""
Code for reading settings from the project at the default.
"""
import os
from collections import namedtuple
import yaml

class SettingsReader:
    """
    A class for reading settings.
    """
    @staticmethod
    def attain_settings():
        """
        Returns the combined settings for the project and the default settings.

        :return: Returns a named tuple of the settings.
        :rtype: NamedTuple
        """
        default_settings_directory = os.path.dirname(os.path.realpath(__file__))
        default_settings_yaml_path = os.path.join(default_settings_directory, 'default_settings.yaml')
        settings_dict = yaml.safe_load(open(default_settings_yaml_path))

        project_settings_yaml_path = os.path.join(os.getcwd(), 'settings.yaml')
        if os.path.isfile(project_settings_yaml_path):
            project_settings = yaml.safe_load(open(project_settings_yaml_path))
            settings_dict = settings_dict.update(project_settings)
        
        settings = namedtuple('Settings', settings_dict.keys())(**settings_dict)
        return settings
