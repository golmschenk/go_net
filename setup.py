"""
The installer file.
"""

from setuptools import setup

setup(
    name='gonet',
    version='0.1.0',
    description='The Go Net.',
    url='https://github.com/golmschenk/go_net',
    license='MIT',
    packages=['gonet'],
    entry_points='''
        [console_scripts]
        goconverter=gonet.converter:command_line_interface
    ''',
)
