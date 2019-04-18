#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian 16 input and output files preparation.
@author: Yu Che
"""
import os, shutil, re
from gaussian import GaussianInout


class FilesIO(GaussianInout):
    def __init__(self, method, mol, seq):
        super().__init__(method, mol, seq)

    def read_files(self):
        pass
