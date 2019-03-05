#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian 16 input and output files preparation.
@author: Yu Che
"""
import os
import shutil
import re
from datetime import datetime


# noinspection PyBroadException
class GaussianInout:
    """
    Gaussian 16 input and output files preparation function.
    """
    def __init__(self, method, mol, seq, path='../../Documents'):
        """
        :param method: DFT method
        :param mol: Molecule name
        :param seq: Sequence type
        :param path: The root path for input and output files
        :type method: str
        :type mol: str
        :type seq: str
        """
        self.gauss_method = method
        self.mol_group = '{}_{}'.format(mol, seq)
        # Gaussian16 header and barkla bash template
        self.header = './header_{}'.format(method)
        self.bash = './bash_template'
        # Molecular coordinators
        self.mol_folder = '{}/{}/{}'.format(path, mol, seq)
        # Gaussian16 input and output folder
        self.input_folder = '{}/input_{}'.format(path, method)
        self.output_folder = '{}/output_{}'.format(path, method)
        # Error and negative frequency folder
        self.check_target_folder = ('{}/result'.format(self.output_folder))
        # Normal terminated results
        self.normal_folder = self.output_folder + '/' + self.mol_group
        self.chk_path = ('%Chk=/users/psyche/volatile/gaussian'
                         '/chk/{}/{}/'.format(method, mol))

    def setup_target_folder(self, folder):
        """
        Setting targeted folder path.

        :param folder: The path for targeted folder.
        :return: None
        """
        if folder.endswith('/'):
            folder = folder[:-1]
        self.check_target_folder = folder

    def setup_chk_path(self, chk_path):
        """
        Editing default checkpoint information for input files.

        :param chk_path: The chk line in gaussian header
        :return: None
        """
        self.chk_path = chk_path

    def info(self, info):
        """
        Print variables for different function.

        :param info: One of 'all', 'input', 'neg', 'error' and 'error_input'.
        :type info: str
        :return: None
        """
        if info in ['all']:
            print('Gaussian16 {}'.format(self.gauss_method))
        if info in ['all', 'input']:
            print(
                'Header template: {}\n'
                'Checkpoint line: {}\n'
                'Molecular:       {}\n'
                'Gaussian input:  {}/{}'.format(
                    self.header,
                    self.chk_path,
                    self.mol_folder,
                    self.input_folder,
                    self.mol_group
                )
            )
        if info in ['all', 'neg', 'error']:
            print(
                'Targeted folder: {}\n'
                'Normal output:   {}'.format(
                    self.check_target_folder,
                    self.normal_folder
                )
            )
        if info in ['all', 'error_input']:
            print(
                'Error input folder: {}\n'
                'Error check folder: {}'.format(
                    (self.input_folder + '/' +
                     self.check_target_folder.split('/')[-1]),
                    self.check_target_folder
                )
            )
        if info not in ['all', 'input', 'neg', 'error', 'error_input']:
            print(
                'The info variable must belong to: all, input, neg, error,'
                'error_input.'
            )

    def prep_input(self):
        """
        Generate gaussian input files.\n
        All chemical files must be stored under self.check_target_folder
        folder.\n
        Header information read from self.header file.\n
        Checkpoint file path got from self.chk_path variable and named
        as same as the molecule file.

        :return: None
        """
        print('Processing...')
        start = datetime.now()
        # Create folders for origin Gaussian input files
        input_origin_folder = self.input_folder + '/' + self.mol_group
        if not os.path.exists(self.input_folder):
            os.mkdir(self.input_folder)
        if not os.path.exists(input_origin_folder):
                os.mkdir(input_origin_folder)
        with open(self.header, 'r') as header:
            template = tuple(header.readlines())
        # Generate Gaussian input data for all molecules
        for file in os.listdir(self.mol_folder):
            input_data = list(template)
            # Get the molecular name
            name = file.split('.')[0]
            # Edit the checkpoint line
            input_data[2] = self.chk_path + name + '.chk\n'
            # Read molecule file
            molecular_file = self.mol_folder + '/' + file
            with open(molecular_file, 'r') as data:
                # Get all atoms coordinates
                # For mol format
                if file.endswith('.mol'):
                    for line in data:
                        try:
                            if line[35] == '0':
                                coordinates = '{}   {}\n'.format(
                                    line[31:33],
                                    line[1:30]
                                )
                                input_data.append(coordinates)
                        except:
                            pass
                # For xyz format
                elif file.endswith('.xyz'):
                    for line in data:
                        try:
                            if line[0].isalpha():
                                input_data.append(line)
                        except:
                            pass
                else:
                    print('Waring!\n{} is not a MOL or XYZ format file!'.format(
                        file))
                    break
            # Adding terminate line
            input_data.append('\n')
            # Writing data into a gjf file
            input_path = '{}/{}.gjf'.format(input_origin_folder, name)
            with open(input_path, 'w') as input_file:
                input_file.writelines(input_data)
        print('Finished. Total time:{}'.format(datetime.now() - start))

    def check_freq(self):
        """
        Checking the frequency information and distributing negative frequency
        output files into 'neg_freq' folder.

        :return: None
        """
        print('Targeted folder: {}'.format(self.check_target_folder))
        start = datetime.now()
        for file in os.listdir(self.check_target_folder):
            if not file.endswith('.out'):
                print('Error!\n{} is not a Gaussian out file!'.format(file))
                break
            path = self.check_target_folder + '/' + file
            with open(path, 'r') as gauss_out:
                for line in gauss_out:
                    # Checking the frequencies
                    if line.startswith(' Frequencies'):
                        data = re.split(r'\s+', line)
                        # Normal terminated jobs
                        if float(data[3]) > 0:
                            if not os.path.exists(self.normal_folder):
                                os.mkdir(self.normal_folder)
                                print('The final out files: {}'.format(
                                    self.normal_folder))
                            # Move to the final folder
                            shutil.move(path, self.normal_folder)
                            break
                        # Negative frequencies
                        elif float(data[3]) < 0:
                            check_out_folder = self.output_folder + '/neg_freq'
                            if not os.path.exists(check_out_folder):
                                os.mkdir(check_out_folder)
                                print(check_out_folder)
                            # Move to negative frequencies folder
                            shutil.move(path, check_out_folder)
                            break
        print('Finished. Total time:{}'.format(datetime.now() - start))

    def check_error(self):
        """
        Checking the output files and distributing error files into different
        folder.\n
        Error folders are automatically created and named by the error number.

        :return: None
        """
        # Checking the error for output files
        print('Targeted folder: {}'.format(self.check_target_folder))
        start = datetime.now()
        for file in os.listdir(self.check_target_folder):
            if not file.endswith('.out'):
                print('Error!\n{} is not a Gaussian out file!'.format(file))
                break
            path = self.check_target_folder + '/' + file
            with open(path, 'r') as gauss_out:
                lines = gauss_out.readlines()
                # Checking the ending line
                if not re.match(r' File| Normal', lines[-1]):
                    unfinished = self.check_target_folder + '/../unfinished'
                    if not os.path.exists(unfinished):
                        os.mkdir(unfinished)
                    shutil.move(path, unfinished)
                else:
                    error_line = lines[-4:-3][0]
                # Checking the error indicator
                    if error_line.startswith(' Error termination'):
                        error = re.split(r'[/.]', error_line)[-3][1:]
                        # Creating a new folder for different
                        # categories of error
                        new_path = self.output_folder + '/error_' + error
                        if not os.path.exists(new_path):
                            os.mkdir(new_path)
                            print(new_path)
                        shutil.move(path, new_path)
        print('Finished. Total time:{}'.format(datetime.now() - start))

    def error_or_freq_input(self):
        """
        Generating input files for error and negative frequency results.\n
        Using prepared header information.

        :return: None
        """
        print('Targeted folder: {}'.format(self.check_target_folder))
        start = datetime.now()
        error_list = os.listdir(self.check_target_folder)
        with open(self.header, 'r') as header:
            template = tuple(header.readlines())
        for file in error_list:
            input_data = list(template)
            name = file.split('.')[0]
            # Edit checkpoint line
            input_data[2] = self.chk_path + name + '.chk\n'
            # Creating folder and writing the input files
            input_folder_error = (self.input_folder + '/' +
                                  self.check_target_folder.split('/')[-1])
            if not os.path.exists(input_folder_error):
                os.mkdir(input_folder_error)
            input_file_path = input_folder_error + '/{}.gjf'.format(name)
            with open(input_file_path, 'w') as input_file:
                input_file.writelines(input_data)
        print('Finished. Total time:{}'.format(datetime.now() - start))

    def distributed_files(self, path, number):
        """
        Distributed files into sub folders that can be applied for array jobs on
        barkla.

        :param path: The root folder
        :param number: The number of files in each sub-folders
        :type path: str
        :type number: int
        :return: None
        """
        print('Starting...')
        n, i = 0, 1
        if not path.endswith('/'):
            path = path + '/'
        j = len(os.listdir(path)) % number
        for file in os.listdir(path):
            root_path = path + file
            sub_array_path = path + str(i)
            if not os.path.exists(sub_array_path):
                os.mkdir(sub_array_path)
            shutil.move(root_path, sub_array_path)
            n += 1
            if (n == number and i != 1) or (n == (number + j) and i == 1):
                n = 0
                i += 1
        print('Finished!\n'
              'Distributed into {} folders.'.format((i - 1)))

    def undistributed_files(self, path):
        """
        Collecting all sub-folders files to their root folder.

        :param path: The root folder
        :type path: str
        :return:
        """
        if not path.endswith('/'):
            path = path + '/'
        for folder in os.listdir(path):
            file_list = os.listdir(path + str(folder))
            for file in file_list:
                target_path = '{}/{}/{}'.format(path, folder, file)
                shutil.move(target_path, path)
            os.removedirs(path + str(folder))
        print('Finished!')


if __name__ == '__main__':
    gauss_function = GaussianInout(method='PM7', mol='dyes', seq='dimer')
    gauss_function.info('all')
    gauss_function.undistributed_files('../../Documents/test')
    gauss_function.distributed_files('../../Documents/test', 10)
