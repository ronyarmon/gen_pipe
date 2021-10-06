import sys
import os
from libraries import *

def save_fig(fig_name, results_dir, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(results_dir, fig_name + "." + fig_extension)
    print("Saving figure", fig_name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
