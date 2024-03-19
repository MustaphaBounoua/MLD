from shutil import copyfile
import torch
import torch.nn.functional as F
from src.dataLoaders.MHD.utils.pytrajkin_utils import *
import torchvision
import random
import numpy as np

import uuid
import os

def get_greyscale_image(path):

    # convert this pil image to the cv one
    cv_img = cv2.imread(path)
    cv_img_gs = cv2.cvtColor(np.array(cv_img), cv2.COLOR_BGR2GRAY)
    cv_img_gs_inv = 255 - cv_img_gs
    roi, _ = get_char_img_thumbnail_helper(cv_img_gs_inv)
    return roi

def plot_single_stroke_digit_evaluation(data):

    #data is a single stroke with the last entry as the time scale...
    fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    data_len = int(len(data)/2)
    ax.plot(data[:data_len], -data[data_len:], 'k', linewidth=random.randint(12, 28))

    # Range of possible linestrokes
    # Min = 12
    # Max = 28

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    return fig, ax


def plot_single_stroke_digit(data):

    #data is a single stroke with the last entry as the time scale...
    fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    data_len = int(len(data[:-1])/2)
    ax.plot(data[:data_len], -data[data_len:-1], 'k', linewidth=random.randint(12, 28))

    # Range of possible linestrokes
    # Min = 12
    # Max = 28

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    return fig, ax


def plot_single_stroke_digit(data):

    #data is a single stroke with the last entry as the time scale...
    fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    data_len = int(len(data[:-1])/2)
    ax.plot(data[:data_len], -data[data_len:-1], 'k', linewidth=random.randint(12, 28))

    # Range of possible linestrokes
    # Min = 12
    # Max = 28

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    return fig, ax

def generate_image_from_trajectory(traj, tmp_path, img_path, save_image=False):


    # Plot Trajectory in color and save image
    fig, ax = plot_single_stroke_digit(traj)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    filename = os.path.join(tmp_path, str(str(uuid.uuid4())+"tmp.png"))
    fig.savefig(filename, bbox_inches=extent, dpi=100)
    plt.close(fig)

    # Get image in greyscale
    g_img = get_greyscale_image(filename)
    if save_image:
        cv2.imwrite(img_path, g_img)
  
    os.remove(filename) 
    # Process image
    np_img = np.asarray(g_img)
    return torch.from_numpy(np_img).unsqueeze(0).float()/float(255.), g_img  # Normalize data!
