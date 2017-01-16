"""
Code for simple conversions between file types.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import numpy as np
import re
from PIL import Image

from gonet.tfrecords_processor import TFRecordsProcessor


class Converter:
    """
    A class for simple conversions of data.
    """

    @staticmethod
    def convert_tfrecords_to_numpy(input_tfrecords_path, output_numpy_directory):
        """
        Converts a tfrecords file into image and label numpy files.
        """
        os.makedirs(output_numpy_directory, exist_ok=True)
        images_numpy, labels_numpy = TFRecordsProcessor().read_to_numpy(input_tfrecords_path)
        data_name = os.path.splitext(os.path.basename(input_tfrecords_path))[0]
        output_base_name = os.path.join(output_numpy_directory, data_name)
        np.save('{}_images.npy'.format(output_base_name), images_numpy)
        np.save('{}_labels.npy'.format(output_base_name), labels_numpy)

    @staticmethod
    def convert_video_to_images(input_video_path, output_frames_directory, frames_per_second=30):
        """
        Converts a video to images.

        :param input_video_path: The path of the video to convert.
        :type input_video_path: str
        :param output_frames_directory: The path of the directory to ouput the frames. Does not need to exist.
        :type output_frames_directory: str
        :param frames_per_second: The number of images per second of video to generate.
        :type frames_per_second: int
        """
        if not os.path.isdir(output_frames_directory):
            os.mkdir(output_frames_directory)
        output_frames_path = os.path.join(output_frames_directory, r'image_%d.jpg')
        command = 'ffmpeg -i {ivp} -qscale:v 2 -vf fps={fps} {ofp}'.format(ivp=input_video_path,
                                                                           fps=frames_per_second,
                                                                           ofp=output_frames_path)
        print('Running command: %s' % command)
        argument_list = shlex.split(command)
        subprocess.run(argument_list)

    @staticmethod
    def convert_images_to_video(input_frames_directory, output_video_path, frames_per_second=30):
        """
        Converts images to a video.

        :param input_frames_directory: The directory of the images to convert.
        :type input_frames_directory: str
        :param output_video_path: The path to output the video to.
        :type output_video_path: str
        :param frames_per_second: The number of images per second of video.
        :type frames_per_second: int
        """
        input_frames_path = os.path.join(input_frames_directory, r'image_%d.jpg')
        command = 'ffmpeg -i {ifp} -vf fps={fps} {ovp}'.format(ifp=input_frames_path,
                                                               fps=frames_per_second,
                                                               ovp=output_video_path)
        print('Running command: %s' % command)
        argument_list = shlex.split(command)
        subprocess.run(argument_list)

    def convert_images_to_numpy(self, input_images_directory, output_numpy_path):
        """
        Converts a directory of images into a NumPy array.

        :param input_images_directory: The images directory.
        :type input_images_directory: str
        :param output_numpy_path: The path to output the NumPy array to.
        :type output_numpy_path: str
        """
        image_types = ('.jpg', '.jpeg', '.png')
        file_name_list = os.listdir(input_images_directory)
        file_name_list = sorted(file_name_list, key=self.natural_sort_key)
        image_list = []
        for file_name in file_name_list:
            if file_name.endswith(image_types):
                image_file = Image.open(os.path.join(input_images_directory, file_name))
                image_file.load()
                image = np.asarray(image_file, dtype="uint8")
                image_list.append(image)
        images = np.stack(image_list)
        np.save(output_numpy_path, images)

    @staticmethod
    def natural_sort_key(element, _natural_sort_regex=re.compile('([0-9]+)')):
        """
        A key to allow for natural sorting.
        Taken from: http://stackoverflow.com/a/16090640/1191087

        :param element: The element under consideration.
        :type element: str
        :param _natural_sort_regex: The regex of the natural sort.
        :type _natural_sort_regex: type(re.compile(''))
        :return: The natural sort key.
        :rtype: list[int]
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(_natural_sort_regex, element)]

    def convert_video_to_numpy(self, input_video_path, output_numpy_path, frames_per_second=30):
        """
        Converts a video into a NumPy array.

        :param input_video_path: The path to the video.
        :type input_video_path: str
        :param output_numpy_path: The path where the NumPy array should be saved.
        :type output_numpy_path: str
        :param frames_per_second: The frames per second to export.
        :type frames_per_second: int
        """
        output_directory = os.path.dirname(output_numpy_path)
        temporary_frames_directory = os.path.join(output_directory, 'temporary_frames_directory')
        os.mkdir(temporary_frames_directory)
        self.convert_video_to_images(input_video_path, temporary_frames_directory, frames_per_second=frames_per_second)
        self.convert_images_to_numpy(temporary_frames_directory, output_numpy_path)
        shutil.rmtree(temporary_frames_directory)

    @staticmethod
    def convert_bgr_numpy_to_rgb_numpy(input_numpy_path, output_numpy_path):
        """
        Converts images in NumPy from BGR to RGB (and vica versa).

        :param input_numpy_path: The original numpy file path.
        :type input_numpy_path: str
        :param output_numpy_path: The location to save the converted data.
        :type output_numpy_path: str
        """
        images = np.load(input_numpy_path)
        if len(images.shape) == 3:
            new_images = images[:, :, [2, 1, 0]]
        else:
            new_images = images[:, :, :, [2, 1, 0]]
        np.save(output_numpy_path, new_images)

    @staticmethod
    def stack_numpy_directory(input_directory, output_numpy_path, substring=None):
        """
        Stacks the numpy files in a directory into a single file.

        :param input_directory: The directory to stack.
        :type input_directory: str
        :param output_numpy_path: Where to output the stacked file.
        :type output_numpy_path: str
        :param substring: Optional requirement that the file includes this parameter in its name.
        :type substring: str
        """
        numpy_arrays = []
        for file_name in os.listdir(input_directory):
            if file_name.endswith('.npy'):
                if not substring or substring in file_name:
                    numpy_arrays.append(np.load(os.path.join(input_directory, file_name)))
        numpy_stack = np.stack(numpy_arrays)
        np.save(output_numpy_path, numpy_stack)

    @staticmethod
    def stack_image_directory_as_numpy(input_directory, output_numpy_path):
        """
        Convert the images in a directory to a stacked numpy file.

        :param input_directory: The directory containing the images.
        :type input_directory: str
        :param output_numpy_path: Where to output the numpy file to.
        :type output_numpy_path:  str
        """
        images = []
        for file_name in os.listdir(input_directory):
            image_types = ('.jpg', '.jpeg', '.png')
            if file_name.endswith(image_types):
                image = np.asarray(Image.open(os.path.join(input_directory, file_name)), dtype="uint8")
                images.append(image)
        images = np.stack(images)
        np.save(output_numpy_path, images)

    def nt_to_standard(self, input_directory=None, output_directory=None):
        """
        Convert from nt format to the standard GoNet input format. nt format being single numpy files, each containing
        one image, and single label files with matching names. Each image has the string "image" in the name and each
        label has the word "density" in the name.

        :param input_directory: The directory to work on. Defaults to the current directory.
        :type input_directory: str
        :param output_directory: The directory to output to. Defaults to the current directory.
        :type output_directory: str
        """
        if not input_directory:
            input_directory = os.getcwd()
        if not output_directory:
            output_directory = os.getcwd()
        dataset_name = os.path.basename(os.path.normpath(input_directory))
        output_name = os.path.join(output_directory, dataset_name)
        self.stack_numpy_directory(input_directory, output_name + '_images.npy', substring='image')
        self.stack_numpy_directory(input_directory, output_name + '_labels.npy', substring='density')


def command_line_interface():
    """
    A command line interface for the converter code.
    """
    parser = argparse.ArgumentParser(
        description='A data converter for various purposes.'
    )
    # Subparsers.
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'command'

    # Video load subparser specific arguments.
    nt_to_standard_title = 'nt2standard'
    nt_to_standard_parser = subparsers.add_parser(nt_to_standard_title, help='Converts from nt to standard format.')
    nt_to_standard_parser.add_argument('-i', '--input_directory', type=str, help='The directory to convert.')
    nt_to_standard_parser.add_argument('-o', '--output_directory', type=str, help='The directory to output to.')
    nt_to_standard_parser.set_defaults(command=nt_to_standard_title)

    args = parser.parse_args()

    converter = Converter()
    if args.command == nt_to_standard_title:
        converter.nt_to_standard(args.input_directory, args.output_directory)


if __name__ == '__main__':
    command_line_interface()