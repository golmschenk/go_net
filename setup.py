"""
The installer file.
"""
import os

from setuptools import setup

if os.path.isdir("/usr/local/cuda"):
    tensorflow_postfix = '-gpu'
else:
    tensorflow_postfix = ''

setup(
    name='gonet',
    version='0.1.7',
    description='The Go Net.',
    url='https://github.com/golmschenk/go_net',
    license='MIT',
    packages=['gonet'],
    entry_points='''
        [console_scripts]
        goconverter=gonet.converter:command_line_interface
    ''',
    install_requires=[
        'tensorflow{}'.format(tensorflow_postfix),
        'h5py'
    ]
)
