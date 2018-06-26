from tvbsim.base.constants.config import FiguresConfig
import matplotlib
# matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy
from collections import OrderedDict
from tvbsim.visualize.baseplotter import BasePlotter
from tvbsim.base.computations.math_utils import compute_in_degree
import tvbsim.base.constants.model_constants as constants

class PlotterFreq(BasePlotter):
    def __init__(self, config=None):
        super(PlotterFreq, self).__init__(config)

        self.HighlightingDataCursor = lambda *args, **kwargs: None

        if matplotlib.get_backend(
        ) in matplotlib.rcsetup.interactive_bk and self.config.figures.MOUSE_HOOVER:
            try:
                from mpldatacursor import HighlightingDataCursor
                self.HighlightingDataCursor = HighlightingDataCursor
            except ImportError:
                self.config.figures.MOUSE_HOOVER = False
                self.logger.warning(
                    "Importing mpldatacursor failed! No highlighting functionality in plots!")
        else:
            self.logger.warning(
                "Noninteractive matplotlib backend! No highlighting functionality in plots!")
            self.config.figures.MOUSE_HOOVER = False

    def plot_psd(self, psd, figure_name='PSD '):
        pyplot.figure(figure_name + str(heatmap.number_of_regions),
                      self.config.figures.SUPER_LARGE_PORTRAIT)

        # ax = self.plot_heatmap_overtime(heatmap.tvbsim,
        #                            heatmap.labels, heatmap.timepoints, 111,
        #                            "Fragility measure", caxlabel="Fragility Metric")
        # for onset in onsettimes:
        #     ax = self.plotvertlines(ax, onset, 'red')
        # for offset in offsettimes:
        #     ax = self.plotvertlines(ax, offset, 'black')

        self._save_figure(None,
            figure_name.replace(" ","_").replace("\t","_"))
        self._check_show()


    def plot_tfr(self, freqmap, freqband=constants.GAMMA,
                      figure_name='Multitaper FFT at Gamma Band '):
        pyplot.figure(figure_name + str(winmat.number_of_regions),
                      self.config.figures.VERY_LARGE_SIZE)

        # apply transformatino to the frequency map
        # freqmap.binfreqvalues(freqband=freqband)
        # self.plot_heatmap_overtime(heatmap.power_band,
        #                            heatmap.labels, 111,
        #                            "Absolute power")
        self._save_figure(None,
            figure_name.replace(" ","_").replace("\t","_"))
        self._check_show()
