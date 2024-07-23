import numpy as np
import matplotlib.pyplot as plt

class Plots():

    def plot_outputs(inputs, outputs, targets, bilinear, bicubic, error_l, error_bilinear_l, error_bicubic_l, epoch, path, iter):

        n = int(len(error_l)*5)
        num_rows = (n + 4) // 5
        fig_width = 10
        fig_height = fig_width * num_rows / 5

        fig, axs = plt.subplots(num_rows, 5, figsize=(fig_width, fig_height))

        if n % 5 != 0:
            for i in range(n % 5, 5):
                axs[-1, i].remove()

        for ax in axs.flat:
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(len(error_l)):
            for j in range(5):
                ax = axs[j] if len(error_l) == 1 else axs[i,j]

                if j == 0:
                    ax.imshow(np.flipud(inputs[i, -1, 0]))
                    ax.set_title('Input')
                    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                if j == 1:
                    ax.imshow(np.flipud(targets[i, -1, 0]))
                    ax.set_title('Target')
                    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                if j == 2:
                    ax.imshow(np.flipud(outputs[i, -1, 0]))
                    ax.set_title('Recon (' + str(np.round(error_l[i], 4)) + ')')
                    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                if j == 3:
                    ax.imshow(np.flipud(bilinear[i, -1, 0]))
                    ax.set_title('Bilinear (' + str(np.round(error_bilinear_l[i], 4)) + ')')
                    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                if j == 4:
                    ax.imshow(np.flipud(bicubic[i, -1, 0]))
                    ax.set_title('Bicubic (' + str(np.round(error_bicubic_l[i], 4)) + ')')
                    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        plt.tight_layout()


        plt.savefig(path + str(epoch) + '_' + str(iter) + '.png', bbox_inches = 'tight')

        plt.show()