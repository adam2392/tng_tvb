from tvbsim.base.constants.config import FiguresConfig
import matplotlib
# matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy
from collections import OrderedDict
from tvbsim.visualize.baseplotter import BasePlotter

# import computations
from tvbsim.base.utils.data_structures_utils import generate_region_labels, ensure_list, ensure_string, \
    isequal_string, sort_dict, linspace_broadcast, \
    list_of_dicts_to_dicts_of_ndarrays, \
    dicts_of_lists_to_lists_of_dicts, \
    extract_dict_stringkeys
from tvbsim.base.utils.log_error import raise_value_error


class PlotterSim(BasePlotter):
    def __init__(self, config=None):
        super(PlotterSim, self).__init__(config)

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

    '''
    Top level function

    '''

    def plot_simulated_timeseries(self, timeseries, model, seizure_indices, seeg_list=[],
                                  spectral_raster_plot=False, title_prefix="", **kwargs):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ", " + model._ui_name + ": "
        region_labels = timeseries.space_labels
        state_variables = timeseries.dimension_labels[TimeseriesDimensions.VARIABLES.value]
        source_ts = timeseries.get_source()
        start_plot = int(numpy.round(0.01 * source_ts.data.shape[0]))
        self.plot_raster({'source(t)': source_ts.squeezed[start_plot:, :]},
                         timeseries.time_line.flatten()[start_plot:],
                         time_units=timeseries.time_unit, special_idx=seizure_indices,
                         title=title_prefix + "Simulated source rasterplot", offset=2.0,
                         labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

        if isinstance(model, EpileptorDP2D):
            # We assume that at least x1 and z are available in res
            sv_dict = {
                'x1(t)': timeseries.x1.squeezed,
                'z(t)': timeseries.z.squeezed}

            self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                 special_idx=seizure_indices, title=title_prefix + "Simulated TAVG",
                                 labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

            self.plot_trajectories(sv_dict, special_idx=seizure_indices,
                                   title=title_prefix + 'State space trajectories', labels=region_labels,
                                   figsize=FiguresConfig.LARGE_SIZE)
        else:
            # We assume that at least source and z are available in res
            sv_dict = {
                'source(t)': source_ts.squeezed,
                'z(t)': timeseries.z.squeezed}

            self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                 special_idx=seizure_indices, title=title_prefix + "Simulated source-z",
                                 labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

            if PossibleVariables.X1.value in state_variables and PossibleVariables.Y1.value in state_variables:
                sv_dict = {
                    'x1(t)': timeseries.x1.squeezed,
                    'y1(t)': timeseries.y1.squeezed}

                self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                     special_idx=seizure_indices, title=title_prefix + "Simulated pop1",
                                     labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)
            if PossibleVariables.X2.value in state_variables and PossibleVariables.Y2.value in state_variables and \
                    PossibleVariables.G.value in state_variables:
                sv_dict = {'x2(t)': timeseries.x2.squeezed, 'y2(t)': timeseries.y2.squeezed,
                           'g(t)': timeseries.g.squeezed}

                self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                     special_idx=seizure_indices, title=title_prefix + "Simulated pop2-g",
                                     labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

            if spectral_raster_plot:
                self.plot_spectral_analysis_raster(timeseries.time_line, source_ts.squeezed,
                                                   time_units=timeseries.time_unit, freq=None,
                                                   special_idx=seizure_indices,
                                                   title=title_prefix + "Spectral Analysis",
                                                   labels=region_labels, figsize=FiguresConfig.LARGE_SIZE, **kwargs)

            if isinstance(model, EpileptorDPrealistic):
                if PossibleVariables.SLOPE_T.value in state_variables and \
                        PossibleVariables.IEXT2_T.value in state_variables:
                    sv_dict = {'1/(1+exp(-10(z-3.03))': 1 / (1 + numpy.exp(-10 * (timeseries.z.squeezed - 3.03))),
                               'slope': timeseries.slope_t.squeezed, 'Iext2': timeseries.Iext2_t.squeezed}
                    title = model._ui_name + ": Simulated controlled parameters"

                    self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                         special_idx=seizure_indices, title=title_prefix + title, labels=region_labels,
                                         figsize=FiguresConfig.VERY_LARGE_SIZE)
                if PossibleVariables.X0_T.value in state_variables and PossibleVariables.IEXT1_T.value in state_variables \
                        and PossibleVariables.K_T.value:
                    sv_dict = {'x0_values': timeseries.x0_t.squeezed, 'Iext1': timeseries.Iext1_t.squeezed,
                               'K': timeseries.K_t.squeezed}

                    self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                         special_idx=seizure_indices,
                                         title=title_prefix + "Simulated parameters",
                                         labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

        self.plot_simulated_seeg_timeseries(
            seeg_list, title_prefix=title_prefix)

    def plot_trajectories(self, data_dict, subtitles=None, special_idx=[], title='State space trajectories',
                          figure_name=None, labels=[], figsize=FiguresConfig.LARGE_SIZE):
        return self.plot_timeseries(data_dict, [],
                                    mode="traj",
                                    special_idx=special_idx,
                                    title=title, subtitles=subtitles,
                                    figure_name=figure_name,
                                    labels=labels, figsize=figsize)
    # works!

    def plot_seeg_timeseries(
            self, seeg_list, title_prefix="Ep", special_idx=[]):
        for seeg in ensure_list(seeg_list):
            title = title_prefix + "SEEG N=" + \
                str(len(seeg.space_labels)) + " raster plot"
            self.plot_raster({'SEEG': seeg.squeezed_data}, seeg.time_line,
                             time_units=seeg.time_unit, special_idx=special_idx,
                             title=title,
                             offset=0.1, labels=seeg.space_labels,
                             figsize=FiguresConfig.VERY_LARGE_SIZE)

    def plot_raster(self, data_dict, time,
                    time_units="ms", special_idx=[],
                    title='Raster plot', subtitles=[],
                    offset=1.0, figure_name=None,
                    labels=[], figsize=FiguresConfig.VERY_LARGE_SIZE):
        return self.plot_timeseries(data_dict, time,
                                    mode="raster", subplots=None,
                                    time_units=time_units, special_idx=special_idx,
                                    title=title, subtitles=subtitles,
                                    offset=offset, figure_name=figure_name,
                                    labels=labels, figsize=figsize)

    def plot_timeseries(self, data_dict, time=None,
                        mode="ts", subplots=None,
                        time_units="ms", special_idx=[],
                        title='Time series', subtitles=[],
                        offset=1.0, figure_name=None,
                        labels=[], figsize=FiguresConfig.LARGE_SIZE):
        # get the number of variables to plot
        n_vars = len(data_dict)
        # get the variable keys and their values
        var_keys = list(reversed(data_dict.keys()))
        data = list(reversed(data_dict.values()))

        # loop through data and preprocess to get the plotting limits
        data_lims = []
        for id, d in enumerate(data):
            if isequal_string(mode, "raster"):
                drange = numpy.percentile(
                    d.flatten(), 95) - numpy.percentile(d.flatten(), 5)
                data[id] = d / drange  # zscore(d, axis=None)
            data_lims.append([d.min(), d.max()])

        # get shape of data to be plotted
        data_shape = data[0].shape
        n_times, nTS = data_shape[:2]
        if len(data_shape) > 2:
            nSamples = data_shape[2]
        else:
            nSamples = 1
        if len(subtitles) == 0:
            subtitles = var_keys
        # generate ylabels for the plot
        labels = generate_region_labels(nTS, labels)

        self.logger.debug("Plotting in mode: {}".format(mode))
        if isequal_string(mode, "traj"):
            data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits = self._trajectories_plot(
                    n_vars, nTS, nSamples, subplots)
        elif isequal_string(mode, "raster"):
            data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                self._timeseries_plot(
                    time,
                    n_vars,
                    nTS,
                    n_times,
                    time_units,
                    0,
                    offset,
                    data_lims)
        elif isequal_string(mode, "ts"):
            data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                self._timeseries_plot(
                    time,
                    n_vars,
                    nTS,
                    n_times,
                    time_units,
                    ensure_list(subplots)[0])

        # set plotting parameters: alpha_ratio, colors, alphas
        alpha_ratio = 1.0 / nSamples
        colors = numpy.array(['k'] * nTS)
        alphas = numpy.maximum(
            numpy.array(
                [def_alpha] *
                nTS) *
            alpha_ratio,
            0.1)
        colors[special_idx] = 'r'
        alphas[special_idx] = numpy.maximum(alpha_ratio, 0.1)
        lines = []

        # initialize figure
        pyplot.figure(title, figsize=figsize)
        pyplot.hold(True)
        axes = []

        # print each column of subplots
        for icol in range(n_cols):
            if n_rows == 1:
                # If there are no more rows, create axis, and set its limits,
                # labels and possible subtitle
                axes += ensure_list(pyplot.subplot(n_rows,
                                                   n_cols, icol + 1, projection=projection))
                axlimits(data_lims, time, n_vars, icol)
                axlabels(labels, var_keys, n_vars, n_rows, 1, 0)
                pyplot.gca().set_title(subtitles[icol])
            for iTS in loopfun(nTS, n_rows, icol):
                if n_rows > 1:
                    # If there are more rows, create axes, and set their
                    # limits, labels and possible subtitles
                    axes += ensure_list(pyplot.subplot(n_rows,
                                                       n_cols, iTS + 1, projection=projection))
                    subtitle(labels, iTS)
                    axlimits(data_lims, time, n_vars, icol)
                    axlabels(
                        labels,
                        var_keys,
                        n_vars,
                        n_rows,
                        (iTS % n_rows) + 1,
                        iTS)
                lines += ensure_list(plot_lines(data_fun(data,
                                                         time, icol), iTS, colors, alphas, labels))
            if isequal_string(
                    mode, "raster"):  # set yticks as labels if this is a raster plot
                axYticks(labels, nTS, icol)
                pyplot.gca().invert_yaxis()

        if self.config.figures.MOUSE_HOOVER:
            for line in lines:
                self.HighlightingDataCursor(line, formatter='{label}'.format, bbox=dict(fc='white'),
                                            arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        # run saving of the figure
        pyplot.gca().invert_yaxis()
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return pyplot.gcf(), axes, lines

    def _timeseries_plot(self, time, n_vars, nTS, n_times,
                         time_units, subplots, offset=0.0, data_lims=[]):
        """
        Function helper for timeseries related plots
        """
        # create vector of the time ticks
        def_time = range(n_times)
        try:
            time = numpy.array(time).flatten()
            if len(time) != n_times:
                self.logger.warning(
                    "In numpy time doesn't match data! Setting a default time step vector!")
                time = def_time
        except BaseException:
            self.logger.warning(
                "Setting a default time step vector manually! Inumpyut time: " +
                str(time))
            time = def_time

        # set the units of the time
        time_units = ensure_string(time_units)
        # create lambda function for accessing the data

        def data_fun(data, time, icol): return (data[icol], time, icol)

        def plot_ts_raster(x, iTS, colors, alphas, labels, offset):
            x, time, ivar = x
            try:
                return pyplot.plot(time, -x[:, iTS] + (offset * iTS + x[:, iTS].mean()), colors[iTS], label=labels[iTS],
                                   alpha=alphas[iTS])
            except BaseException:
                self.logger.warning(
                    "Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, -x[:, iTS] + offset * iTS, colors[iTS], label=str(iTS),
                                   alpha=alphas[iTS])

        def plot_ts(x, iTS, colors, alphas, labels):
            x, time, ivar = x
            try:
                return pyplot.plot(
                    time, x[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except BaseException:
                self.logger.warning(
                    "Cannot convert labels' strings for line labels!")
                return pyplot.plot(
                    time, x[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def axlabels_ts(labels, n_rows, irow, iTS):
            if irow == n_rows:
                pyplot.gca().set_xlabel("Time (" + time_units + ")")
            if n_rows > 1:
                try:
                    pyplot.gca().set_ylabel(str(iTS) + "." + labels[iTS])
                except BaseException:
                    self.logger.warning(
                        "Cannot convert labels' strings for y axis labels!")
                    pyplot.gca().set_ylabel(str(iTS))

        def axlimits_ts(data_lims, time, icol):
            pyplot.gca().set_xlim([time[0], time[-1]])
            if n_rows > 1:
                pyplot.gca().set_ylim([data_lims[icol][0], data_lims[icol][1]])
            else:
                pyplot.autoscale(enable=True, axis='y', tight=True)

        def axYticks(labels, nTS, ivar, offsets=offset):
            pyplot.gca().set_yticks(
                (offset * numpy.array([range(nTS)]).flatten()).tolist())
            try:
                pyplot.gca().set_yticklabels(labels.flatten().tolist())
            except BaseException:
                labels = generate_region_labels(nTS, [], "")
                self.logger.warning(
                    "Cannot convert region labels' strings for y axis ticks!")

        if offset > 0.0:
            def plot_lines(
                x,
                iTS,
                colors,
                alphas,
                labels): return plot_ts_raster(
                x,
                iTS,
                colors,
                alphas,
                labels,
                offset)
        else:
            def plot_lines(
                x, iTS, colors, alphas, labels): return plot_ts(
                x, iTS, colors, alphas, labels)

        def this_axYticks(
            labels, nTS, ivar): return axYticks(
            labels, nTS, offset)
        if subplots:
            n_rows = nTS
            def_alpha = 1.0
        else:
            n_rows = 1
            def_alpha = 0.5

        def subtitle_col(subtitle): return pyplot.gca().set_title(subtitle)

        def subtitle(iTS, labels): return None
        projection = None

        def axlabels(
            labels,
            vars,
            n_vars,
            n_rows,
            irow,
            iTS): return axlabels_ts(
            labels,
            n_rows,
            irow,
            iTS)

        def axlimits(
            data_lims,
            time,
            n_vars,
            icol): return axlimits_ts(
            data_lims,
            time,
            icol)

        def loopfun(nTS, n_rows, icol): return range(nTS)
        return data_fun, time, plot_lines, projection, n_rows, n_vars, def_alpha, loopfun, \
            subtitle, subtitle_col, axlabels, axlimits, this_axYticks

    def _trajectories_plot(self, n_dims, nTS, nSamples, subplots):
        def data_fun(data, time, icol): return data

        def plot_traj_2D(x, iTS, colors, alphas, labels):
            x, y = x
            try:
                return pyplot.plot(
                    x[:, iTS], y[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except BaseException:
                self.logger.warning(
                    "Cannot convert labels' strings for line labels!")
                return pyplot.plot(
                    x[:, iTS], y[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def plot_traj_3D(x, iTS, colors, alphas, labels):
            x, y, z = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS],
                                   colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except BaseException:
                self.logger.warning(
                    "Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], colors[iTS], label=str(
                    iTS), alpha=alphas[iTS])

        def subtitle_traj(labels, iTS):
            try:
                pyplot.gca().set_title(str(iTS) + "." + labels[iTS])
            except BaseException:
                self.logger.warning(
                    "Cannot convert labels' strings for subplot titles!")
                pyplot.gca().set_title(str(iTS))

        def axlabels_traj(vars, n_vars):
            pyplot.gca().set_xlabel(vars[0])
            pyplot.gca().set_ylabel(vars[1])
            if n_vars > 2:
                pyplot.gca().set_zlabel(vars[2])

        def axlimits_traj(data_lims, n_vars):
            pyplot.gca().set_xlim([data_lims[0][0], data_lims[0][1]])
            pyplot.gca().set_ylim([data_lims[1][0], data_lims[1][1]])
            if n_vars > 2:
                pyplot.gca().set_zlim([data_lims[2][0], data_lims[2][1]])

        if n_dims == 2:
            def plot_lines(
                x,
                iTS,
                colors,
                labels,
                alphas): return plot_traj_2D(
                x,
                iTS,
                colors,
                labels,
                alphas)
            projection = None
        elif n_dims == 3:
            def plot_lines(
                x,
                iTS,
                colors,
                labels,
                alphas): return plot_traj_3D(
                x,
                iTS,
                colors,
                labels,
                alphas)
            projection = '3d'
        else:
            raise_value_error(
                "Data dimensions are neigher 2D nor 3D!, but " +
                str(n_dims) +
                "D!")
        n_rows = 1
        n_cols = 1
        if subplots is None:
            if nSamples > 1:
                n_rows = int(numpy.floor(numpy.sqrt(nTS)))
                n_cols = int(numpy.ceil((1.0 * nTS) / n_rows))
        elif isinstance(subplots, (list, tuple)):
            n_rows = subplots[0]
            n_cols = subplots[1]
            if n_rows * n_cols < nTS:
                raise_value_error("Not enough subplots for all time series:"
                                  "\nn_rows * n_cols = product(subplots) = product(" + str(
                                      subplots) + " = "
                                  + str(n_rows * n_cols) + "!")
        if n_rows * n_cols > 1:
            def_alpha = 0.5

            def subtitle(labels, iTS): return subtitle_traj(labels, iTS)

            def subtitle_col(subtitles, icol): return None
        else:
            def_alpha = 1.0

            def subtitle(): return None

            def subtitle_col(
                subtitles,
                icol): return pyplot.gca().set_title(
                pyplot.gcf().title)

        def axlabels(
            labels,
            vars,
            n_vars,
            n_rows,
            irow,
            iTS): return axlabels_traj(
            vars,
            n_vars)

        def axlimits(
            data_lims,
            time,
            n_vars,
            icol): return axlimits_traj(
            data_lims,
            n_vars)

        def loopfun(nTS, n_rows, icol): return range(icol, nTS, n_rows)
        return data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
            subtitle, subtitle_col, axlabels, axlimits
