"""
The installer file.
"""

from setuptools import setup
from subprocess import run

completed_process = run(['nvcc', '--version'])
if completed_process.returncode:
    tensorflow_postfix = ''
else:
    tensorflow_postfix = '-gpu'

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
