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


class GaussianInout:
    """
    Gaussian 16 input and output files preparation function.
    """
    def __init__(self, method, mol, seq, root_path='../../Documents'):
        """
        :param method: DFT method
        :param mol: Molecule name
        :param seq: Sequence type
        :param root_path: The root path for input and output files
        :type method: str
        :type mol: str
        :type seq: str
        """
        self.gauss_method = method
        self.elements = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P',
                         16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        self.mol_name = '{}_{}'.format(mol, seq)
        # Gaussian16 header and barkla bash template
        self.header = './header_{}'.format(method)
        self.bash = './bash_template'
        # Molecular coordinators
        self.mol_origin = '{}/{}/{}'.format(root_path, mol, seq)
        # Gaussian16 input and output folder
        self.input_folder = '{}/input_{}/{}'.format(
            root_path, method, self.mol_name)
        self.output_folder = '{}/output_{}/{}'.format(
            root_path, method, self.mol_name)
        # Error and negative frequency folder
        self.origin_result_folder = ('{}/result'.format(self.output_folder))
        # Normal terminated results
        self.normal_result_folder = (
                self.output_folder + '/{}_out'.format(self.mol_name)
        )
        self.mol_result = self.output_folder + '/{}_xyz'.format(self.mol_name)
        self.chk_path = (
            '%Chk=/users/psyche/volatile/gaussian/chk/{}/'.format(method)
        )
        if not os.path.exists(self.input_folder.split('/{}'.format(mol))[0]):
            os.mkdir(self.input_folder.split('/{}'.format(mol))[0])
            if not os.path.exists(self.input_folder):
                os.mkdir(self.input_folder)
        if not os.path.exists(self.output_folder.split('/{}'.format(mol))[0]):
            os.mkdir(self.output_folder.split('/{}'.format(mol))[0])
            if not os.path.exists(self.output_folder):
                os.mkdir(self.output_folder)

    def setup_result_folder(self, folder):
        """
        Setting targeted folder path.

        :param folder: The path for targeted folder.
        :return: None
        """
        if folder.endswith('/'):
            folder = folder[:-1]
        self.origin_result_folder = folder

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
                'Header template:    {}\n'
                'Checkpoint line:    {}\n'
                'Molecular folder:   {}\n'
                'Gaussian input:     {}/{}'.format(
                    self.header,
                    self.chk_path,
                    self.mol_origin,
                    self.input_folder,
                    self.mol_name
                )
            )
        if info in ['all', 'neg', 'error']:
            print(
                'HPC results folder: {}\n'
                'Normal output:      {}'.format(
                    self.origin_result_folder,
                    self.normal_result_folder
                )
            )
        if info in ['all', 'error_input']:
            print(
                'Error input folder: {}\n'
                'Error check folder: {}'.format(
                    (self.input_folder + '/' +
                     self.origin_result_folder.split('/')[-1]),
                    self.origin_result_folder
                )
            )
        if info not in ['all', 'input', 'neg', 'error', 'error_input']:
            print(
                'The info variable must belong to: all, input, neg, error,'
                'error_input.'
            )

    def prep_input(self, geom):
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
        input_origin_folder = self.input_folder + '/{}'.format(self.mol_name)
        if not os.path.exists(self.input_folder):
            os.mkdir(self.input_folder)
        if not os.path.exists(input_origin_folder):
            os.mkdir(input_origin_folder)
        with open(self.header, 'r') as header:
            template = header.readlines()
        # Generate Gaussian input data for all molecules
        for file in os.listdir(self.mol_origin):
            input_data = template.copy()
            # Get the molecular name
            name = file.split('.')[0]
            # Edit the checkpoint line
            for i in range(len(input_data)):
                if input_data[i].startswith('%Chk'):
                    input_data[i] = self.chk_path + '{}.chk\n'.format(name)
            # Read molecule file
            if geom == 'local':
                molecular_file = self.mol_origin + '/' + file
                with open(molecular_file, 'r') as data:
                    # Get all atoms coordinates
                    # For mol format
                    if file.endswith('.mol'):
                        for line in data:
                            segments = re.split(r'\s+', line)
                            try:
                                if segments[4] in self.elements.values():
                                    xyz_line = '{}{:>10}{:>10}{:>10}\n'.format(
                                        segments[4],
                                        segments[1],
                                        segments[2],
                                        segments[3]
                                    )
                                    input_data.append(xyz_line)
                            except IndexError:
                                pass
                    # For xyz format
                    elif file.endswith('.xyz'):
                        for line in data:
                            try:
                                if line[0].isalpha():
                                    input_data.append(line)
                            except IndexError:
                                pass
                    else:
                        print('Waring!\n'
                              '{} is not MOL or XYZ format!'.format(file))
                        break
                # Adding terminate line
                input_data.append('\n')
            elif geom == 'chk':
                geom_line = False
                for line in input_data:
                    if line.startswith('# Geom'):
                        geom_line = True
                if not geom_line:
                    input_data.insert(4, '# Geom=Checkpoint Guess=Read\n')
            else:
                print('Error! Geom must be local or chk.')
            # Writing data into a gjf file
            input_path = '{}/{}.gjf'.format(input_origin_folder, name)
            with open(input_path, 'w') as input_file:
                input_file.writelines(input_data)
        print('Finished. Total time:{}'.format(datetime.now() - start))

    def error_screening(self):
        """
        Checking the output files and distributing error files into different
        folder.\n
        Error folders are automatically created and named by the error number.

        :return: None
        """
        # Checking the error for output files
        print('Targeted folder: {}'.format(self.origin_result_folder))
        start = datetime.now()
        i, j = 0, 0
        for file in os.listdir(self.origin_result_folder):
            if not file.endswith('.out'):
                print('Error!\n{} is not a Gaussian out file!'.format(file))
                break
            path = self.origin_result_folder + '/' + file
            with open(path, 'r') as gauss_out:
                lines = gauss_out.readlines()
                # Checking the ending line
                if not re.match(r' File| Normal', lines[-1]):
                    unfinished = self.origin_result_folder + '/../unfinished'
                    if not os.path.exists(unfinished):
                        os.mkdir(unfinished)
                    shutil.move(path, unfinished)
                    i += 1
                else:
                    error_line = lines[-4:-3][0]
                # Checking the error indicator
                    if error_line.startswith(' Error termination'):
                        error = re.split(r'[/.]', error_line)[-3][1:]
                        # Creating a new folder for different
                        # categories of error
                        error_folder = self.output_folder + '/error_' + error
                        if not os.path.exists(error_folder):
                            os.mkdir(error_folder)
                            print(error_folder)
                        shutil.move(path, error_folder)
                        j += 1
        print(
            'Finished.\n'
            'Normal terminated:         {}\n'
            'Error result:              {}\n'
            'Total time:{}'.format(i, j, (datetime.now() - start))
        )

    def neg_freq_screening(self):
        """
        Checking the frequency information and distributing negative frequency
        output files into 'neg_freq' folder.

        :return: None
        """
        print('Targeted folder: {}'.format(self.origin_result_folder))
        start = datetime.now()
        i, j = 0, 0
        for file in os.listdir(self.origin_result_folder):
            if not file.endswith('.out'):
                print('Error!\n{} is not a Gaussian out file!'.format(file))
                break
            path = self.origin_result_folder + '/' + file
            with open(path, 'r') as gauss_out:
                for line in gauss_out:
                    # Checking the frequencies
                    if line.startswith(' Frequencies'):
                        data = re.split(r'\s+', line)
                        # Normal terminated jobs
                        if float(data[3]) > 0:
                            if not os.path.exists(self.normal_result_folder):
                                os.mkdir(self.normal_result_folder)
                            shutil.move(path, self.normal_result_folder)
                            i += 1
                            break
                        # Negative frequencies
                        elif float(data[3]) < 0:
                            error_folder = self.output_folder + '/neg_freq'
                            if not os.path.exists(error_folder):
                                os.mkdir(error_folder)
                            shutil.move(path, error_folder)
                            j += 1
                            break
        print(
            'Finished.\n'
            'Normal results:            {}\n'
            'Negative frequency result: {}\n'
            'Total time:{}'.format(i, j, (datetime.now() - start))
        )

    def prep_error_input(self, error):
        """
        Generating input files for error and negative frequency results.
        Using prepared header information.

        :return: None
        """
        error_folder = self.output_folder + '/{}'.format(error)
        print('Targeted folder: {}'.format(error_folder))
        start = datetime.now()
        error_list = os.listdir(error_folder)
        with open(self.header, 'r') as header:
            template = header.readlines()
        for file in error_list:
            input_data = template.copy()
            name = file.split('.')[0]
            geom_line = False
            # Edit checkpoint line
            for i in range(len(input_data)):
                if input_data[i].startswith('%Chk'):
                    input_data[i] = self.chk_path + '{}.chk\n'.format(name)
                elif input_data[i].startswith('# Geom'):
                    geom_line = True
            if not geom_line:
                input_data.insert(4, '# Geom=Checkpoint Guess=Read\n')
            # Creating folder and writing the input files
            error_input_folder = (self.input_folder + '/{}'.format(error))
            if not os.path.exists(error_input_folder):
                os.mkdir(error_input_folder)
            error_input_file = error_input_folder + '/{}.gjf'.format(name)
            with open(error_input_file, 'w') as input_file:
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

    def reading_opt_out_files(self):
        for out_file in os.listdir(self.normal_result_folder):
            final_step_line, energy_line = 0, 0
            first_atom_line, last_atom_line = 0, 0
            out_file_path = self.normal_result_folder + '/' + out_file
            with open(out_file_path, 'r') as file:
                lines = file.readlines()
            # Finding the converged step position
            for i in range(len(lines)):
                if lines[i].startswith(' Optimization completed'):
                    final_step_line = i
                    break
            # Finding the energy and coordinates position
            for j in range(final_step_line - 1, -1, -1):
                if 'SCF Done' in lines[j]:
                    energy_line = j
                if 'Coordinates (Angstroms)' in lines[j]:
                    first_atom_line = j + 3
                    for k in range(first_atom_line, final_step_line):
                        if lines[k].startswith(' -'):
                            last_atom_line = k - 1
                            break
                    break
            # Reading the energy data
            energy_str = re.split(r'\s+', lines[energy_line])[5]
            if 'E' in energy_str:
                energy_e = energy_str.split('E')
                energy = float(energy_e[0]) * 10 ** int(energy_e[1])
            else:
                energy = energy_str
            # Reading all coordinates data
            atom_numbers = re.split(r'\s+', lines[last_atom_line])[1]
            coordinate_lines = []
            for n in range(first_atom_line, last_atom_line+1):
                segments = re.split(r'\s+', lines[n])
                coordinate_line = '{}{:>12}{:>12}{:>12}\n'.format(
                    self.elements[int(segments[2])],
                    segments[4],
                    segments[5],
                    segments[6]
                )
                coordinate_lines.append(coordinate_line)
            # xyz format lines list
            xyz_format_lines = ['{}\n'.format(atom_numbers),
                                'Energy: {} A.U.\n'.format(energy)]
            xyz_format_lines = xyz_format_lines + coordinate_lines
            if not os.path.exists(self.mol_result):
                os.mkdir(self.mol_result)
            path = self.mol_result + '/{}.xyz'.format(out_file.split('.')[0])
            with open(path, 'w') as mol_file:
                mol_file.writelines(xyz_format_lines)


if __name__ == '__main__':
    gauss_function = GaussianInout(method='PM7_opt', mol='dyes', seq='monomer')
    gauss_function.info('all')
    gauss_function.undistributed_files('../../Documents/test')
    gauss_function.distributed_files('../../Documents/test', 10)
    gauss_function.reading_opt_out_files()
