#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for managing SMILES dataframe.
@author: Yu Che
"""
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from rdkit.Chem import AllChem as Chem


def load_df(path):
    df = pd.read_csv(path)
    return df


def read_characters(path):
    """
    Reading the character-level one-hot json file.

    :param path: The path for saved character.json
    :type path: str
    :return: A list of characters
    :rtype: list
    """
    characters = yaml.safe_load(open(path))
    print('The length of characters:    {}'.format(len(characters)))
    return characters


def generate_canon_smi(df, origin_smile_column):
    """
    Converting origin SMILES into canonical SMILES and saving them into
    the 'canonical_smi' column.

    :param df: The input dataframe
    :param origin_smile_column: The origin SMILES column number
    :type df: pandas.core.frame.DataFrame
    :type origin_smile_column: int
    """
    print('Processing...')
    start = datetime.now()
    if 'canonical_smi' not in df.columns:
        df['canonical_smi'] = None
    else:
        raise KeyError('The dataframe have an canonical SMILES column.')
    for i in range(0, len(df)):
        smile = df.iloc[i, origin_smile_column]
        try:
            canonical_smi = Chem.MolToSmiles(
                Chem.MolFromSmiles(smile), canonical=True
            )
        except Exception:
            raise ValueError(
                'Please check the incorrect SMILES: {}'.format(smile)
            )
        df.loc[i, 'canonical_smi'] = canonical_smi
    print('Finished. Total time:{}'.format(datetime.now() - start))


def padding_smile(smile, max_len, padding='all'):
    """

    :param smile:
    :param max_len:
    :param padding:
    :return:
    """
    if len(smile) <= max_len:
        if padding == 'all':
            return smile + ' ' * (max_len - len(smile))
        elif padding == 'last':
            return smile + ' '
        elif padding == 'none':
            return smile
        else:
            raise ValueError


def smiles_to_one_hot(df, characters):
    """
    Character level one-hot embedding

    :param df: The input dataframe
    :param characters: The list of characters
    :type df: pandas.core.frame.DataFrame
    :type characters: list
    :return: The embedded matrix
    :rtype: numpy.ndarray
    """
    if 'canonical_smi' not in df.columns:
        raise KeyError('Dataframe does not have canonical SMILES.')
    max_len = 0
    for i in range(0, len(df)):
        if max_len < len(df.loc[i, 'canonical_smi']):
            max_len = len(df.loc[i, 'canonical_smi'])
    token_index = dict((c, i) for i, c in enumerate(characters))
    characters_len = len(characters)
    print('The maximal length of SMILES: {}'.format(max_len))
    smiles = df.loc[:, 'canonical_smi'].tolist()
    smiles_matrix = np.zeros((len(smiles), max_len, characters_len),
                             dtype=np.float32)
    for i, smile in enumerate(smiles):
        for j, character in enumerate(smile):
            try:
                smiles_matrix[i, j, token_index[character]] = 1.
            except KeyError as error:
                print('Unrecorded characters founded./n'
                      'Index:  {}\n'
                      'SMILES: {}'.format(i, smile))
                raise error
    return smiles_matrix


# noinspection PyTypeChecker
def one_hot_to_smiles(smiles_matrix, characters):
    """
    Converting one-hot embedding matrix to SMILES list.

    :param smiles_matrix:
    :param characters:
    :type smiles_matrix: np.ndarray
    :type characters: list
    :return: SMILES list
    :rtype: list
    """
    smiles_list = []
    for one_hot_matrix in smiles_matrix:
        smile_temp = ''
        for one_hot_vector in one_hot_matrix:
            index = np.argmax(one_hot_vector)
            smile_temp += characters[index]
        smiles_list.append(smile_temp)
    return smiles_list
