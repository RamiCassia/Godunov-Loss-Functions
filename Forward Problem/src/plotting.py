import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

from src.utilities import Utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Plots():

    def post_process_train(output, true, fig_save_path, epoch, loss, n_cols):

        error, error_r, error_u, error_v, error_E = Utils.calculate_errors(output, true)

        fig, ax = plt.subplots(nrows=2, ncols= n_cols, figsize=(3.5*n_cols,7))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i in range(0,2):
            for j in range(0, n_cols):
                if n_cols != 1:
                    if i == 0:
                        ax[i,j].imshow(np.flipud(output[j,-1,0,:,:]))
                        ax[i,j].set_title(str(np.round(Utils.frobenius_norm(true[j]-output[j]) / Utils.frobenius_norm(true[j]),4)))
                    elif i == 1:
                        ax[i,j].imshow(np.flipud(true[j,-1,0,:,:]))

                    ax[i,j].tick_params(axis='both', which='both', bottom=False,  top=False, left = False, right = False, labelbottom=False, labeltop = False, labelleft = False, labelright = False)
                else:
                    if i == 0:
                        ax[i].imshow(np.flipud(output[j,-1,0,:,:]))
                        ax[i].set_title('Error = ' + str(np.round(Utils.frobenius_norm(true[j]-output[j]) / Utils.frobenius_norm(true[j]),4)), size = 'large')
                    elif i == 1:
                        ax[i].imshow(np.flipud(true[j,-1,0,:,:]))

                    ax[i].tick_params(axis='both', which='both', bottom=False,  top=False, left = False, right = False, labelbottom=False, labeltop = False, labelleft = False, labelright = False)

        if n_cols != 1:
            ax[0,0].set_ylabel('Prediction', size = 'x-large')
            ax[1,0].set_ylabel('Reference',size = 'x-large')

        else:
            ax[0].set_ylabel('Prediction',size = 'x-large')
            ax[1].set_ylabel('Reference',size = 'x-large')

        my_suptitle = fig.suptitle('Epoch = ' + str(epoch) + ' | ' + 'Loss = ' + str(np.round(loss,3)) + ' | '  + 'Error = ' + str(np.round(error,4)), size = 'xx-large', fontweight = 'bold', horizontalalignment = 'center', y=1.03)

        if epoch >= 0:
            plt.savefig(fig_save_path + 'comparison_' + str(epoch) + '.png', bbox_inches='tight',bbox_extra_artists=[my_suptitle])

        return error, error_r, error_u, error_v, error_E

    def post_process_inference(output, true, fig_save_path, n_cols, t0_index, n_infer, time_steps, configuration):

        error, error_r, error_u, error_v, error_E = Utils.calculate_errors(output, true)
        error_seq, error_seq_r, error_seq_u, error_seq_v, error_seq_E = Utils.calculate_errors_seq(output, true)

        fig, ax = plt.subplots(nrows=2, ncols= n_cols, figsize=(3.5*n_cols,7))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i in range(0,2):
            for j in range(0, n_cols):
                if n_cols != 1:
                    if i == 0:
                        ax[i,j].imshow(np.flipud(output[0,int(j/(n_cols-1)*time_steps),0,:,:]))
                        ax[i,j].set_title(str(np.round(Utils.frobenius_norm(true[:,int(j/(n_cols-1)*time_steps),:,:,:]-output[:,int(j/(n_cols-1)*time_steps),:,:,:]) / Utils.frobenius_norm(true[:,int(j/(n_cols-1)*time_steps),:,:,:]),4)))
                    elif i == 1:
                        ax[i,j].imshow(np.flipud(true[0,int(j/(n_cols-1)*time_steps),0,:,:]))

                    ax[i,j].tick_params(axis='both', which='both', bottom=False,  top=False, left = False, right = False, labelbottom=False, labeltop = False, labelleft = False, labelright = False)

                else:
                    if i == 0:
                        ax[i].imshow(np.flipud(output[j,-1,0,:,:]))
                        ax[i].set_title('Error = ' + str(np.round(Utils.frobenius_norm(true[j]-output[j]) / Utils.frobenius_norm(true[j]),4)), size = 'large')
                    elif i == 1:
                        ax[i].imshow(np.flipud(true[j,-1,0,:,:]))

                    ax[i].tick_params(axis='both', which='both', bottom=False,  top=False, left = False, right = False, labelbottom=False, labeltop = False, labelleft = False, labelright = False)

        if n_cols != 1:
            ax[0,0].set_ylabel('Prediction', size = 'x-large')
            ax[1,0].set_ylabel('Reference',size = 'x-large')

        else:
            ax[0].set_ylabel('Prediction',size = 'x-large')
            ax[1].set_ylabel('Reference',size = 'x-large')

        my_suptitle = fig.suptitle(' | ' + 'Time = ' + '(' + str(t0_index) + ' - ' + str(t0_index + (n_infer*time_steps)) + ')' + r'$\Delta t$' +  ' | ' + 'Average Error = ' + str(np.round(error,4)) + ' | ', size = 'xx-large', fontweight = 'bold', horizontalalignment = 'center', y=1.05)

        plt.savefig(fig_save_path + 'comparison_' + str(configuration) + '_' + str(t0_index) + '_' +  str(time_steps) + '.png', bbox_inches='tight',bbox_extra_artists=[my_suptitle])

        #plt.show()

        return error_seq, error_seq_r, error_seq_u, error_seq_v, error_seq_E


    def plot_dict_list(data_dict, learning_rates, title, x_label, y_label, log_y, color_background, include, print_interval, fig_save_path, fig_name, leg_loc = "upper right"):

        size = None
        count = 0

        for key, value in data_dict.items():
            if isinstance(value, list):
                count += 1
                if count == 2:
                    size = len(value)
                    break

        epoch_numbers = np.arange(0, size)*print_interval

        if len(epoch_numbers) == 1:
            return

        plt.figure(figsize=(12, 6))
        grid = plt.GridSpec(1, 2, width_ratios=[25, 1])
        ax = plt.subplot(grid[0])

        i = 0

        linestyles =  ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
        colors = ['b', 'g', 'r', 'darkorange', 'm']
        for key, values in data_dict.items():
            if i in include:
                ax.plot(epoch_numbers,values, label=key, linestyle = linestyles[i], color = colors[i], linewidth = 2.5)
                i += 1
            else:
                i += 1
                continue

        if log_y:
            plt.yscale('log')

        plt.title(title, fontsize = 13, pad = 20)
        plt.xlabel(x_label, fontsize = 13, labelpad = 10)
        plt.ylabel(y_label, fontsize = 13, labelpad = 10)
        plt.locator_params(axis="x", integer=True)
        plt.xlim(min(epoch_numbers), max(epoch_numbers))
        plt.legend(loc=leg_loc, handlelength=4)

        if color_background:
            cmap = plt.get_cmap("binary_r")
            norm = LogNorm(vmin=min(learning_rates), vmax=max(learning_rates))
            cax = plt.subplot(grid[1])
            cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
            cax.set_title("LR", fontsize = 12, pad=15)
            for i in range(len(epoch_numbers) - 1):
                ax.axvspan(epoch_numbers[i], epoch_numbers[i + 1], facecolor=cmap(norm(learning_rates[print_interval*i])), edgecolor="none")

        plt.tight_layout()

        plt.savefig(fig_save_path + fig_name + '.png', bbox_inches='tight')


    def inf_plot_dict_of_lists(data_dict, time_dict, train_regions, xlabel, ylabel, title, fig_save_path, fig_name, include, legend_loc='upper right'):

        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
        line_colors = ['b', 'g', 'r', 'darkorange', 'm']

        fig, ax = plt.subplots(figsize=(12, 6))
        i = 0
        n = 0
        for key, sublists in data_dict.items():
            if i in include:
                n = 0
                for sublist in sublists:
                    x_values = np.arange(len(sublist))
                    line, = ax.plot(time_dict[n], sublist, label=key, color=line_colors[i], linestyle = line_styles[i])
                    n += 1
                i += 1
            else:
                i += 1
                continue


        if train_regions:
            for start, end in train_regions:
                ax.axvspan(start, end, alpha=0.2, color='gray')

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        custom_legend = ax.legend([handles[labels.index(label)] for label in unique_labels],
                                unique_labels, loc=legend_loc)

        plt.xlabel(xlabel, fontsize = 13, labelpad = 10)
        plt.ylabel(ylabel, fontsize = 13, labelpad = 10)
        plt.title(title, fontsize = 13, pad = 20)

        ax.grid(which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--', linewidth='0.5', color='gray')
        ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray', alpha = 0.4)

        plt.tight_layout
        plt.savefig(fig_save_path + fig_name + '.png', bbox_inches='tight')