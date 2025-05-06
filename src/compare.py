import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from adjustText import adjust_text

import numpy as np

import pickle
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dump_flax', default='dump_250_1024_momentum', help='Dump dir of Jax')
parser.add_argument('--dump_tf', default='old_repository/dump_tf_250_1024', help='Dump dir of Tensorflow')

FLAGS = parser.parse_args()

LOG_DIR1 = FLAGS.dump_flax
LOG_DIR2 = FLAGS.dump_tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 


def _get_model_file(model_path, model_name):
    """
    Returns the file path for saving or loading the model parameters.

    This function combines the provided model path and model name to
    generate a full file path for the model file. The file extension used
    is `.tar`, which is commonly used for storing model parameters in 
    a compressed archive format.

    Args:
        model_path (str): The directory path where the model file should be saved or loaded from.
        model_name (str): The name of the model file (without extension) to be stored or loaded.

    Returns:
        str: The full path to the model file, which combines the model_path and model_name with a `.tar` extension.
    """
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


if __name__ == "__main__":
    with open(_get_model_file(LOG_DIR1,'Flax_Accuracy'),"rb") as f:
            flax_acc = pickle.load(f)

    with open(_get_model_file(LOG_DIR2,'Tf_Accuracy'),"rb") as f:
            tf_acc = pickle.load(f)

    fig, ax = plt.subplots()

    x = np.arange(0.0,1.1,0.1)

    line = mlines.Line2D([0, 1], [0, 1], color='red')

    scatter = ax.scatter(flax_acc,tf_acc,vmin = 0.0, vmax = 1.0)
    scatter = ax.set_title('Tensorflow vs Jax&Flax Accuracies')
    scatter = ax.set_xlabel('Jax&Flax Accuracies')
    scatter = ax.set_ylabel('Tensorflow Accuracies')
    scatter = ax.grid(visible = True)
    scatter = ax.add_line(line)
    scatter = ax.xaxis.set_ticks(x)
    scatter = ax.yaxis.set_ticks(x)
    names = [SHAPE_NAMES[i] for i in range(len(SHAPE_NAMES))]
    TEXT = []
    for i in range(len(SHAPE_NAMES)):
        #   ax.text(flax_acc[i],tf_acc[i],names[i])
          TEXT.append(ax.text(flax_acc[i],tf_acc[i],names[i]))

    adjust_text(TEXT, x=flax_acc, y=tf_acc,
            arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))
    
    plt.savefig(LOG_DIR1+'/Compared Accuracies')
    # plt.show()

    sys.modules[__name__].__dict__.clear()