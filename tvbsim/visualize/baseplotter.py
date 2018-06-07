import numpy
import os
import seaborn as sns
import warnings
from natsort import natsorted

import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fragility.base.constants.config import Config, FiguresConfig
from fragility.base.utils.log_error import initialize_logger


class BasePlotter(object):
    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

    def _check_show(self):
        if self.config.figures.SHOW_FLAG:
            # mp.use('TkAgg')
            pyplot.ion()
            pyplot.show()
        else:
            # mp.use('Agg')
            pyplot.ioff()
            pyplot.close()

    @staticmethod
    def _figure_filename(fig=None, figure_name=None):
        if fig is None:
            fig = pyplot.gcf()
        if figure_name is None:
            figure_name = fig.get_label()
        # replace all unnecessary characters
        figure_name = figure_name.replace(
            ": ",
            "_").replace(
            " ",
            "_").replace(
            "\t",
            "_").replace(
                ",",
            "")
        return figure_name

    def _save_figure(self, fig, figure_name):
        if self.config.figures.SAVE_FLAG:
            # get figure name and set it with the set format
            figure_name = self._figure_filename(fig, figure_name)
            figure_name = figure_name[:numpy.min(
                [100, len(figure_name)])] + '.' + FiguresConfig.FIG_FORMAT
            figure_dir = self.config.out.FOLDER_FIGURES
            if not (os.path.isdir(figure_dir)):
                os.mkdir(figure_dir)
            pyplot.savefig(os.path.join(figure_dir, figure_name))

    @staticmethod
    def plotvertlines(ax, time, color='k'):
        '''
        '''
        # plot vertical lines of 'predicted' onset/offset
        ax.axvline(
            time,
            color=color,
            linestyle='dashed',
            linewidth=5)
        return ax

    def setfonts(self, title_font=None, axis_font=None):
        self.title_font = title_font
        self.axis_font = axis_font
        if not title_font:
            # Set the font dictionaries (for plot title and axis titles)
            self.title_font = {'fontname': 'Arial',
                               'size': '34',
                               'color': 'black',
                               'weight': 'normal',
                               'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
        # if isinstance(axis_font, list):
        if not axis_font:
            self.axis_font = {'family': 'Arial',
                              'size': '30'}

    def setfigsize(self, figsize):
        self.figsize = figsize

    # Custom function to draw the diff bars=
    @staticmethod
    def label_diff(ax, i, j, text, X, Y):
        x = (X[i] + X[j]) / 2
        y = 0.5 * max(Y[i], Y[j])
        dx = abs(X[i] - X[j])

        props = {'connectionstyle': 'bar', 'arrowstyle': '-',
                 'shrinkA': 20, 'shrinkB': 20, 'linewidth': 2}
        ax.annotate(text, xy=(X[i] + 0.45, y * 1.75), zorder=20, fontsize=40)
        ax.annotate('', xy=(X[i] + 0.1, y),
                    xytext=(X[j] - 0.1, y), arrowprops=props)
        return ax

    @staticmethod
    def plot_bars(distrib_list, labels, subplot, title, ylabel=None,
                  show_x_labels=False, indices_red=None, sharex=None, showpval=False):
        ax = pyplot.subplot(subplot, sharex=sharex)
        pyplot.title(title)

        # set the number of distributions to plot
        n_vector = labels.shape[0]
        x_ticks = numpy.array(range(n_vector), dtype=numpy.int32)
        color = 'b'
        colors = numpy.repeat([color], n_vector)
        coldif = False
        if indices_red is not None:
            colors[indices_red] = 'r'
            coldif = True

        if len(distrib_list) == 1:
            distrib = numpy.array(distrib_list).ravel()
            var = numpy.var(distrib)
            mean = numpy.mean(distrib)
            ax.bar(x_ticks, mean, yerr=var, ecolor='red', capsize=5,
                   color=colors, align='center', alpha=0.4)
        else:
            for i in range(len(distrib_list)):
                distrib = numpy.array(distrib_list[i]).ravel()
                var = numpy.var(distrib)
                mean = numpy.mean(distrib)
                ax.bar(x_ticks[i], mean, yerr=var, ecolor='red', capsize=5,
                       color=colors, align='center', alpha=0.4)

        ax.grid(True, color='grey')
        ax.set_xticks(x_ticks)
        if show_x_labels:
            region_labels = numpy.array(
                ["%d. %s" % l for l in zip(range(n_vector), labels)])
            ax.set_xticklabels(region_labels)
            if coldif:
                labels = ax.xaxis.get_ticklabels()
                for ids in indices_red:
                    labels[ids].set_color('r')
                ax.xaxis.set_ticklabels(labels)
        else:
            ax.set_xticklabels([])
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # should we show pvalue scores?
        if showpval:
            import scipy
            var = [numpy.var(distrib) for distrib in distrib_list]
            mean = [numpy.mean(distrib) for distrib in distrib_list]
            for i in range(len(distrib_list) - 1):
                stat, pval = scipy.stats.ttest_ind(distrib_list[i],
                                                   distrib_list[i + 1],
                                                   axis=0,
                                                   equal_var=False,
                                                   nan_policy='propagate')
                # Call the function
                if pval < 0.005:
                    text = "p<0.005"
                else:
                    text = "p={0:.4f}".format(pval)
                ax = BasePlotter.label_diff(ax, i, i + 1, text, x_ticks, mean)

        # ax.autoscale(tight=True)
        return ax

    @staticmethod
    def plot_vector(vector, labels, subplot, title, show_y_labels=True,
                    indices_red=None, sharey=None, fontsize=FiguresConfig.VERY_LARGE_FONT_SIZE):
        ax = pyplot.subplot(subplot, sharey=sharey)
        pyplot.title(title, fontsize=fontsize)
        n_vector = labels.shape[0]
        y_ticks = numpy.array(range(0, n_vector * 3, 3), dtype=numpy.int32)
        color = 'k'
        colors = numpy.repeat([color], n_vector)
        coldif = False
        if indices_red is not None:
            colors[indices_red] = 'r'
            coldif = True
        if len(vector.shape) == 1:
            ax.barh(y_ticks, vector, height=0.85, color=colors, align='center')
        else:
            ax.barh(y_ticks, vector[0, :], height=0.85,
                    color=colors, align='center')
        # ax.invert_yaxis()
        ax.grid(True, color='grey')
        ax.set_yticks(y_ticks)
        if show_y_labels:
            region_labels = numpy.array(
                ["%d. %s" % l for l in zip(range(n_vector), labels)])
            ax.set_yticklabels(region_labels)
            if coldif:
                labels = ax.yaxis.get_ticklabels()
                for ids in indices_red:
                    labels[ids].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        xticks = ax.get_yticks()
        ax.set_xticklabels(xticks, fontsize=fontsize)
        ax.set_xlabel('In-Degree Weight', fontsize=fontsize)
        ax.set_ylabel("Regions", fontsize=fontsize)
        ax.autoscale(tight=True)
        if sharey is None:
            ax.invert_yaxis()
        return ax

    @staticmethod
    def plot_heatmap_overtime(adj, labels, timepoints, subplot, title, show_y_labels=True, show_x_labels=True,
                              indices_red_y=None, sharey=None, fontsize=FiguresConfig.LARGE_FONT_SIZE,
                              caxlabel=None):
        ax = pyplot.subplot(subplot, sharey=sharey)     # initialize ax
        # set title
        pyplot.title(title, fontsize=fontsize)
        y_color = 'k'

        adj_size = adj.shape[0]
        time_size = adj.shape[1]  # numpy.arange(time_size), dtype=numpy.int32)

        # set the xticks & color
        x_ticks = numpy.array(
            numpy.arange(0, time_size, time_size / 10),
            dtype=numpy.int32)
        x_color = 'k'

        # set time ticks
        timepoints = numpy.array(timepoints)
        time_ticks = numpy.ceil(timepoints[x_ticks, 0]/1000) # convert to seconds

        # set the yticks & color
        y_ticks = numpy.array(range(adj_size), dtype=numpy.int32)
        if indices_red_y is None:
            indices_red_y = y_ticks
            y_ticks = indices_red_y
            y_color = y_color
        else:
            y_color = 'r'
            y_ticks = range(len(indices_red_y))

        # get the region labels
        region_labels = numpy.array(["%d. %s" %
                                     l for l in zip(range(adj_size), labels)])

        # plot the heatmap
        cmap = pyplot.set_cmap('jet')
        img = ax.imshow(adj[indices_red_y, :],
                        cmap=cmap, aspect='auto',
                        interpolation='nearest')
        ax.set_xticks(x_ticks)
        ax.grid(True, color='grey')
        if show_y_labels:
            region_labels = numpy.array(
                ["%d. %s" % l for l in zip(range(adj_size), labels)])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(region_labels, fontsize=fontsize/1.25)
            if not (x_color == y_color):
                labels = ax.yaxis.get_ticklabels()
                for idx in indices_red_y:
                    labels[idx].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        if show_x_labels:
            ax.set_xticklabels(time_ticks,
                               rotation=270,
                               color=x_color)
        else:
            ax.set_xticklabels([])
        ax.autoscale(tight=True)
        # make a color bar
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)
        if caxlabel is not None:
            cax1.set_ylabel(caxlabel, rotation=270,
                            fontsize=fontsize, labelpad=60)
            cax1.tick_params(labelsize=fontsize)

        ax.set_xlabel('Time (sec)', fontsize=fontsize)
        ax.set_ylabel("Channels", fontsize=fontsize)
        return ax

    @staticmethod
    def plot_regions2regions(adj, labels, subplot, title, show_y_labels=True, show_x_labels=True,
                             indices_red_x=None, sharey=None, caxlabel=None, fontsize=FiguresConfig.LARGE_FONT_SIZE):
        ax = pyplot.subplot(subplot, sharey=sharey)
        pyplot.title(title, fontsize=fontsize)
        y_color = 'k'
        adj_size = adj.shape[0]
        y_ticks = numpy.array(range(adj_size), dtype=numpy.int32)
        if indices_red_x is None:
            indices_red_x = y_ticks
            x_ticks = indices_red_x
            x_color = y_color
        else:
            x_color = 'r'
            x_ticks = range(len(indices_red_x))
        region_labels = numpy.array(["%d. %s" %
                                     l for l in zip(range(adj_size), labels)])
        cmap = pyplot.set_cmap('autumn_r')  # 'jet')
        img = ax.imshow(adj[indices_red_x, :].T,
                        cmap=cmap, interpolation=None)
        ax.set_xticks(x_ticks)
        ax.grid(True, color='grey')
        if show_y_labels:
            region_labels = numpy.array(
                ["%d. %s" % l for l in zip(range(adj_size), labels)])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(region_labels)
            if not (x_color == y_color):
                labels = ax.yaxis.get_ticklabels()
                for idx in indices_red_x:
                    labels[idx].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        if show_x_labels:
            ax.set_xticklabels(region_labels[indices_red_x],
                               rotation=270,
                               ha='center',
                               color=x_color)
            # ax.xaxis.tick_left()
        else:
            ax.set_xticklabels([])
        # ax.autoscale(tight=True)
        # make a color bar
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        pyplot.colorbar(img, cax=cax1)
        if caxlabel is not None:
            cax1.set_ylabel(caxlabel, rotation=270,
                            fontsize=fontsize, labelpad=60)
            cax1.tick_params(labelsize=fontsize)
        return ax

    @staticmethod
    def _set_axis_labels(fig, sub, n_regions, region_labels,
                         indices2emphasize, color='k', position='left'):
        y_ticks = range(n_regions)
        region_labels = numpy.array(["%d. %s" %
                                     l for l in zip(y_ticks, region_labels)])
        big_ax = fig.add_subplot(sub, frameon=False)
        if position == 'right':
            big_ax.yaxis.tick_right()
            big_ax.yaxis.set_label_position("right")
        big_ax.set_yticks(y_ticks)
        big_ax.set_yticklabels(region_labels, color='k')
        if not (color == 'k'):
            labels = big_ax.yaxis.get_ticklabels()
            for idx in indices2emphasize:
                labels[idx].set_color(color)
            big_ax.yaxis.set_ticklabels(labels)
        big_ax.invert_yaxis()
        big_ax.axes.get_xaxis().set_visible(False)
        # TODO: find out what is the next line about and why it fails...
        # big_ax.axes.set_facecolor('none')

    def plot_in_columns(self, data_dict_list, labels, width_ratios=[], left_ax_focus_indices=[],
                        right_ax_focus_indices=[], description="", title="", figure_name=None,
                        figsize=FiguresConfig.VERY_LARGE_SIZE, **kwargs):
        fig = pyplot.figure(title, frameon=False, figsize=figsize)
        fig.suptitle(description)
        n_subplots = len(data_dict_list)
        if not width_ratios:
            width_ratios = numpy.ones((n_subplots,)).tolist()
        matplotlib.gridspec.GridSpec(1, n_subplots, width_ratios=width_ratios)
        if 10 > n_subplots > 0:
            subplot_ind0 = 100 + 10 * n_subplots
        else:
            raise ValueError(
                "\nSubplots' number " +
                str(n_subplots) +
                "is not between 1 and 9!")
        n_regions = len(labels)
        subplot_ind = subplot_ind0
        ax = None
        ax0 = None
        for iS, data_dict in enumerate(data_dict_list):
            subplot_ind += 1
            data = data_dict["data"]
            focus_indices = data_dict.get("focus_indices")
            if subplot_ind == 0:
                if not left_ax_focus_indices:
                    left_ax_focus_indices = focus_indices
            else:
                ax0 = ax
            if data_dict.get("plot_type") == "vector_violin":
                ax = self.plot_vector_violin(data_dict.get("data_samples", []), data, [],
                                             labels, subplot_ind, data_dict["name"],
                                             colormap=kwargs.get("colormap", "YlOrRd"), show_y_labels=False,
                                             indices_red=focus_indices, sharey=ax0)
            elif data_dict.get("plot_type") == "regions2regions":
                ax = self.plot_regions2regions(data, labels, subplot_ind, data_dict["name"], show_y_labels=False,
                                               show_x_labels=True, indices_red_x=focus_indices, sharey=ax0)
            else:
                ax = self.plot_vector(data, labels, subplot_ind, data_dict["name"], show_y_labels=False,
                                      indices_red=focus_indices, sharey=ax0)
        if right_ax_focus_indices == []:
            right_ax_focus_indices = focus_indices
        self._set_axis_labels(
            fig,
            121,
            n_regions,
            labels,
            left_ax_focus_indices,
            'r')
        self._set_axis_labels(
            fig,
            122,
            n_regions,
            labels,
            right_ax_focus_indices,
            'r',
            'right')
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return fig

    def plots(self, data_dict, shape=None, transpose=False, skip=0, xlabels={}, xscales={}, yscales={}, title='Plots',
              figure_name=None, figsize=FiguresConfig.VERY_LARGE_SIZE):
        if shape is None:
            shape = (1, len(data_dict))
        fig, axes = pyplot.subplots(shape[0], shape[1], figsize=figsize)
        fig.set_label(title)
        for i, key in enumerate(data_dict.keys()):
            ind = numpy.unravel_index(i, shape)
            if transpose:
                axes[ind].plot(data_dict[key].T[skip:])
            else:
                axes[ind].plot(data_dict[key][skip:])
            axes[ind].set_xscale(xscales.get(key, "linear"))
            axes[ind].set_yscale(yscales.get(key, "linear"))
            axes[ind].set_xlabel(xlabels.get(key, ""))
            axes[ind].set_ylabel(key)
        fig.tight_layout()
        self._save_figure(fig, figure_name)
        self._check_show()
        return fig, axes

    def pair_plots(self, data, keys, diagonal_plots={}, transpose=False, skip=0,
                   title='Pair plots', subtitles=None, figure_name=None,
                   figsize=FiguresConfig.VERY_LARGE_SIZE):

        def confirm_y_coordinate(data, ymax):
            data = list(data)
            data.append(ymax)
            return tuple(data)

        if subtitles is None:
            subtitles = keys
        data = ensure_list(data)
        n = len(keys)
        fig, axes = pyplot.subplots(n, n, figsize=figsize)
        fig.set_label(title)
        colorcycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys):
                for datai in data:
                    if transpose:
                        di = datai[key_i].T[skip:]
                    else:
                        di = datai[key_i][skip:]
                    if i == j:
                        hist_data = axes[i, j].hist(
                            di, int(numpy.round(numpy.sqrt(len(di)))), log=True)[0]
                        if i == 0 and len(di.shape) > 1 and di.shape[1] > 1:
                            axes[i, j].legend(["chain " + str(ichain + 1)
                                               for ichain in range(di.shape[1])])
                        hist_max = numpy.array(hist_data).max()
                        # The mean line
                        axes[i, j].vlines(di.mean(axis=0), 0, hist_max, color=colorcycle, linestyle='dashed',
                                          linewidth=1)
                        # Plot a line (or marker) in the same axis as hist
                        diag_line_plot = ensure_list(
                            diagonal_plots.get(key_i, ((), ()))[0])
                        if len(diag_line_plot) in [1, 2]:
                            if len(diag_line_plot) == 1:
                                diag_line_plot = confirm_y_coordinate(
                                    diag_line_plot, hist_max)
                            else:
                                diag_line_plot[1] = diag_line_plot[1] / \
                                    numpy.max(diag_line_plot[1]) * hist_max
                            if len(diag_line_plot[0]) == 1:
                                axes[i, j].plot(
                                    diag_line_plot[0], diag_line_plot[1], "o", color='k', markersize=10)
                            else:
                                axes[i, j].plot(diag_line_plot[0], diag_line_plot[1], color='k',
                                                linestyle="dashed", linewidth=1)
                        # Plot a marker in the same axis as hist
                        diag_marker_plot = ensure_list(
                            diagonal_plots.get(key_i, ((), ()))[1])
                        if len(diag_marker_plot) in [1, 2]:
                            if len(diag_marker_plot) == 1:
                                diag_marker_plot = confirm_y_coordinate(
                                    diag_marker_plot, hist_max)
                            axes[i, j].plot(
                                diag_marker_plot[0], diag_marker_plot[1], "*", color='k', markersize=10)

                    else:
                        if transpose:
                            dj = datai[key_j].T[skip:]
                        else:
                            dj = datai[key_j][skip:]
                        axes[i, j].plot(dj, di, '.')
                if i == 0:
                    axes[i, j].set_title(subtitles[j])
                if j == 0:
                    axes[i, j].set_ylabel(key_i)
        fig.tight_layout()
        self._save_figure(fig, figure_name)
        self._check_show()
        return fig, axes
