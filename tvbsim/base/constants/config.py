# coding=utf-8

import os
import numpy as np
import tvbsim
from datetime import datetime

class GenericConfig(object):
    _module_path = os.path.dirname(fragility.__file__)

class InputConfig(object):
    _base_input = os.getcwd()

    @property
    def RAW_DATA_FOLDER(self):
        if self._raw_data is not None:
            return self._raw_data

        # Expecting to run in the top of stats GIT repo, with the dummy head
        return os.path.join(self._base_input, "data", "raw")

    def __init__(self, raw_folder=None):
        self._raw_data = raw_folder

class OutputConfig(object):
    subfolder = None

    def __init__(self, out_base=None, separate_by_run=False):
        """
        :param work_folder: Base folder where logs/figures/results should be kept
        :param separate_by_run: Set TRUE, when you want logs/results/figures to be in different files / each run
        """
        self._out_base = out_base or os.path.join(os.getcwd(), "tvbsim_out")
        self._separate_by_run = separate_by_run

    @property
    def FOLDER_LOGS(self):
        folder = os.path.join(self._out_base, "logs")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        return folder

    @property
    def FOLDER_RES(self):
        folder = os.path.join(self._out_base, "res")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        if self.subfolder is not None:
            os.path.join(folder, self.subfolder)
        return folder

    @property
    def FOLDER_FIGURES(self):
        folder = os.path.join(self._out_base, "figs")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        if self.subfolder is not None:
            os.path.join(folder, self.subfolder)
        return folder

    @property
    def FOLDER_TEMP(self):
        return os.path.join(self._out_base, "temp")

class FiguresConfig(object):
    VERY_LARGE_SIZE = (40, 20)
    VERY_LARGE_PORTRAIT = (30, 50)
    SUPER_LARGE_SIZE = (80, 40)
    LARGE_SIZE = (20, 15)
    SMALL_SIZE = (15, 10)
    FIG_FORMAT = 'pdf' # 'eps' 'pdf' 'svg'
    SAVE_FLAG = True
    SHOW_FLAG = True              # interactive mode and show?
    MOUSE_HOOVER = False
    MATPLOTLIB_BACKEND = "Qt4Agg" # , "Agg", "qt5"

class CalculusConfig(object):
    SYMBOLIC_CALCULATIONS_FLAG = False

    # Normalization configuration
    WEIGHTS_NORM_PERCENT = 95

    MIN_SINGLE_VALUE = np.finfo("single").min
    MAX_SINGLE_VALUE = np.finfo("single").max
    MAX_INT_VALUE = np.iinfo(np.int64).max
    MIN_INT_VALUE = np.iinfo(np.int64).max
# class HypothesisConfig(object):
#     def __init__(self, head_folder=None):
        # self.head_folder = head_folder

class Config(object):
    generic = GenericConfig()
    figures = FiguresConfig()
    calcul = CalculusConfig()

    def __init__(self, 
                raw_data_folder=None,
                output_base=None, 
                separate_by_run=False):
        self.input = InputConfig(raw_data_folder)
        self.out = OutputConfig(output_base, separate_by_run)
        # self.hypothesis = HypothesisConfig(head_folder)
