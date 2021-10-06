from libraries import *
from vizz.utils import *
from paths import *

def masked_heatmap(corr_matrix):
    '''Plot the (diagonal) correlation matrix
    :param corr_matrix: Correlation matrix
    '''
    plt.figure(figsize=(12, 7))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap='YlOrBr', vmax=1.0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    save_fig("features_correlation", images_dir)
    plt.show()

