"""
module of utilities
"""
import pickle as cp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os.path
import time
from collections import defaultdict
from scipy import interpolate
import cv2
import copy
from collections import defaultdict
import scipy
from scipy import interpolate
import scipy.ndimage as spi

import src.dataLoaders.MHD.utils.pytrakin_rxzero  as pytk_rz
from tqdm import tqdm

def parse_data_set(file):
    """
    parse data set to read letters
    """
    uji_data = dict()
    f = open(file, 'r')

    # extract coordinates
    start_idx = 3  # the first 3 rows are comments
    for i in range(start_idx):
        dumps = f.readline()
    # for each speciman
    while 1:
        line = f.readline()
        if not line:
            break
        # check if this is the start of another session
        title = line.split()
        if title[0] == '//' and len(title) == 1:
            # print 'start a new session'
            # read another three lines
            for j in range(2):
                dumps = f.readline()
        else:
            # ignore current one, as it is a comment
            for j in range(2):
                # two duplicates for each subject && letter
                line = f.readline()
                input_text = line.split(' ')
                # check if this is a letter among 'a' ~ 'z' or 'A' ~ 'Z'
                # letter_code = ord(input_text[1])
                # valid_letter = False
                # if letter_code <= ord('z') and letter code >= ord('a'):
                #   valid_letter = True
                # elif letter_code <= ord('Z') and letter_code >= ord('A'):
                #   valid_letter = True
                # else:
                #   valid_letter = False
                # check if this is a new record
                letter = input_text[1]
                if letter not in uji_data:
                    uji_data[letter] = []

                # read this letter, we should do this for
                # line indicating strokes
                line = f.readline()
                input_text = line.split()
                num_strokes = int(input_text[1])
                letter_coords = []
                for k in range(num_strokes):
                    line = f.readline()
                    input_text = line.split()
                    # the 2nd indicates number of data points
                    num_pnts = int(input_text[1])
                    # prepare
                    stroke_traj = np.zeros([num_pnts, 2])
                    for l in range(num_pnts):
                        stroke_traj[l, 0] = int(input_text[3 + l * 2])
                        stroke_traj[l, 1] = int(input_text[3 + l * 2 + 1])
                    letter_coords.append(stroke_traj)
                uji_data[letter].append(letter_coords)

    f.close()
    return uji_data


def normalize_data(data_set):
    """
    normalize data:
    1. center all characters
    2. scale the size
    """
    normed_data = copy.deepcopy(data_set)
    for key in normed_data:
        for l in normed_data[key]:
            # merge strokes to find center & size
            tmp_traj = np.concatenate(l, axis=0)
            # print tmp_traj
            traj_center = np.mean(tmp_traj, axis=0)
            tmp_traj -= traj_center
            traj_max = np.amax(tmp_traj, axis=0)
            scale = np.amax(traj_max)

            for s in l:
                # for each stroke
                s -= traj_center
                s /= scale
    return normed_data


def interp_data(data_set):
    """
    interpolate data
    """
    interp_data = dict()
    for key in data_set:
        interp_data[key] = []
        for l in data_set[key]:
            interp_letter = []
            for s in l:
                time_len = s.shape[0]
                if time_len > 3:
                    # interpolate each dim, cubic
                    t = np.linspace(0, 1, time_len)
                    spl_x = interpolate.splrep(t, s[:, 0])
                    spl_y = interpolate.splrep(t, s[:, 1])
                    # resample, 4 times more, vel is also scaled...
                    t_spl = np.linspace(0, 1, 4 * len(t))
                    x_interp = interpolate.splev(t_spl, spl_x, der=0)
                    y_interp = interpolate.splev(t_spl, spl_y, der=0)
                    # #construct new stroke
                    interp_letter.append(np.concatenate([[x_interp], [y_interp]], axis=0).transpose())
                else:
                    # direct copy if no sufficient number of points
                    interp_letter.append(s)
            interp_data[key].append(interp_letter)
    return interp_data


def interp_data_fixed_num(data_set, num=100):
    """
    interpolate data with fixed number of points
    """
    interp_data = dict()
    for key in data_set:
        interp_data[key] = []
        for l in data_set[key]:
            interp_letter = []
            for s in l:
                time_len = s.shape[0]
                if time_len > 3:
                    # interpolate each dim, cubic
                    t = np.linspace(0, 1, time_len)
                    spl_x = interpolate.splrep(t, s[:, 0])
                    spl_y = interpolate.splrep(t, s[:, 1])
                    # resample, 4 times more, vel is also scaled...
                    t_spl = np.linspace(0, 1, num)
                    x_interp = interpolate.splev(t_spl, spl_x, der=0)
                    y_interp = interpolate.splev(t_spl, spl_y, der=0)
                    # #construct new stroke
                    data = np.concatenate([x_interp, y_interp])
                    dt = float(time_len) / num
                    interp_letter.append(np.concatenate([data, [dt]]))
                else:
                    # direct copy if no sufficient number of points
                    interp_letter.append(s)
            interp_data[key].append(interp_letter)
    return interp_data


def smooth_data(data_set):
    """
    smooth data
    """
    smoothed_data = dict()
    for key in data_set:
        smoothed_data[key] = []
        for l in data_set[key]:
            smoothed_letter = []
            for s in l:
                time_len = s.shape[0]
                if time_len > 3:
                    # smooth the data, gaussian filter
                    filtered_stroke = np.array([spi.gaussian_filter(dim, 3) for dim in s.transpose()]).transpose()
                    smoothed_letter.append(filtered_stroke)
                else:
                    # direct copy if no sufficient number of points
                    smoothed_letter.append(s)
            smoothed_data[key].append(smoothed_letter)
    return smoothed_data

def plot_single_stroke_char_or_digit(data):

    #data is a single stroke with the last entry as the time scale...
    fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    data_len = int(len(data[:-1])/2)
    ax.plot(data[:data_len], -data[data_len:-1], 'k', linewidth=12.0)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    return fig, ax

def generate_images_for_chars_and_digits(data, path, overwrite=False, grayscale=True, thumbnail_size=(28, 28)):
    data_dic = defaultdict(list)

    func_path = path
    folder = 'images'
    gs_folder = 'grayscale'

    output_path = os.path.join(func_path, folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gs_output_path = os.path.join(func_path, gs_folder)
    if not os.path.exists(gs_output_path):
        os.makedirs(gs_output_path)
    for dict_key in data.keys():
        print('Processing character or digit {0}'.format(dict_key))
        char_folder = 'char_{0}_{1}'.format(ord(dict_key), dict_key)
        output_path_char = os.path.join(output_path, char_folder)
        if not os.path.exists(output_path_char):
            os.makedirs(output_path_char)

        gs_output_path_char = os.path.join(gs_output_path, char_folder)
        if not os.path.exists(gs_output_path_char):
            os.makedirs(gs_output_path_char)

        for d_idx, d in enumerate(tqdm(data[dict_key])):
            # print 'Generating images for the {0}-th demonstrations...'.format(d_idx)
            tmp_fname = 'ascii_{0}_{1:04d}.png'.format(ord(dict_key), d_idx)
            tmp_fpath = os.path.join(output_path_char, tmp_fname)
            fig = None
            if not os.path.exists(tmp_fpath) or overwrite:
                # print(tmp_fpath)
                fig, ax = plot_single_stroke_char_or_digit(d)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(tmp_fpath, bbox_inches=extent, dpi=100)

            if grayscale:
                tmp_fname_grayscale = 'ascii_{0}_{1:04d}_grayscale_thumbnail.png'.format(ord(dict_key), d_idx)
                tmp_fpath_grayscale = os.path.join(gs_output_path_char, tmp_fname_grayscale)
                if not os.path.exists(tmp_fpath_grayscale) or overwrite:
                    thumbnail = get_char_img_thumbnail(tmp_fpath, tmp_fpath_grayscale)
                    # print('Generating grayscale image {0}'.format(tmp_fname_grayscale))
                else:
                    image = Image.open(tmp_fpath_grayscale)
                    thumbnail = np.asarray(image)

                #get the np array data for this image
                data_dic[dict_key].append([d, np.asarray(thumbnail),  tmp_fpath_grayscale])
            if fig is not None:
                plt.close(fig)
            # time.sleep(0.5)
    return data_dic


#utilities for computing convenience

def expand_traj_dim_with_derivative(data, dt=0.01):
    augmented_trajs = []
    for traj in data:
        time_len = len(traj)
        t = np.linspace(0, time_len*dt, time_len)
        if time_len > 3:
            if len(traj.shape) == 1:
                """
                mono-dimension trajectory, row as the entire trajectory...
                """
                spl = interpolate.splrep(t, traj)
                traj_der = interpolate.splev(t, spl, der=1)
                tmp_augmented_traj = np.array([traj, traj_der]).T
            else:
                """
                multi-dimensional trajectory, row as the state variable...
                """
                tmp_traj_der = []
                for traj_dof in traj.T:
                    spl_dof = interpolate.splrep(t, traj_dof)
                    traj_dof_der = interpolate.splev(t, spl_dof, der=1)
                    tmp_traj_der.append(traj_dof_der)
                tmp_augmented_traj = np.vstack([traj.T, np.array(tmp_traj_der)]).T

            augmented_trajs.append(tmp_augmented_traj)

    return augmented_trajs


def extract_images(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    def extract_image_helper(d):
        #flatten the image and scale them
        return d.flatten().astype(dtype) * 1./255.
    images = []
    if data_dict is not None:
        for char in sorted(data_dict.keys(), key=lambda k:k[-1]):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                images += [extract_image_helper(d) for d in data_dict[char]]
    return np.array(images)

def extract_jnt_trajs(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    def extract_jnt_trajs_helper(d):
        #flatten the image and scale them, is it necessary for joint trajectory, say within pi radians?
        return d.flatten().astype(dtype)
    jnt_trajs = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                jnt_trajs += [extract_jnt_trajs_helper(d) for d in data_dict[char]]
    return np.array(jnt_trajs)

def extract_jnt_fa_parms(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    fa_parms = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                fa_parms += [d for d in data_dict[char]]
    fa_parms = np.array(fa_parms)
    #Gaussian statistics for potential normalization
    fa_mean = np.mean(fa_parms, axis=0)
    fa_std = np.std(fa_parms, axis=0)
    return fa_parms, fa_mean, fa_std


'''
utility to threshold the character image
'''
def threshold_char_image(img):
    #do nothing for now
    return img
'''
utility to segment character contour and get the rectangular bounding box
'''
def segment_char_contour_bounding_box(img):
    cv2_version = cv2.__version__.split('.')
    # print cv2_version
    # CHANGED THIS FROM HANG
    if int(cv2_version[0]) < 3:
        # for opencv below 3.0.0
        ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # for opencv from 3.0.0
        ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    #for blank image
    if len(rects) == 0:
        return [0, 0, img.shape[1], img.shape[0]]
    #rect length-4 array, (rect[0], rect[1]) - lower left corner point, (rect[2], rect[3]) - width, height
    corner_pnts = []
    for rect in rects:
        corner_pnts.append([rect[0], rect[1]])
        corner_pnts.append([rect[0]+rect[2], rect[1]+rect[3]])
    corner_pnts = np.array(corner_pnts)
    l_corner_pnt = np.amin(corner_pnts, axis=0)
    u_corner_pnt = np.amax(corner_pnts, axis=0)
    return [l_corner_pnt[0], l_corner_pnt[1], u_corner_pnt[0]-l_corner_pnt[0], u_corner_pnt[1]-l_corner_pnt[1]]
'''
utility to resize
'''
def get_char_img_thumbnail_helper(img_data):
    #first threshold the img
    thres_img = threshold_char_image(img_data)
    #then figure out the contour bouding box
    bound_rect = segment_char_contour_bounding_box(thres_img)
    center = [bound_rect[0] + bound_rect[2]/2., bound_rect[1] + bound_rect[3]/2.]
    #crop the interested part
    leng = max([int(bound_rect[2]), int(bound_rect[3])])
    border = int(0.6*leng)
    pt1 = int(center[1] -bound_rect[3] // 2)
    pt2 = int(center[0] -bound_rect[2] // 2)
    cv_img_bckgrnd = np.zeros((border+leng, border+leng))
    # print cv_img_bckgrnd.shape
    # print bound_rect
    # print center
    # print border
    # print (pt1+border//2),(pt1+bound_rect[3]+border//2), (pt2+border//2),(pt2+bound_rect[2]+border//2)
    # print cv_img_bckgrnd[(border//2):(bound_rect[3]+border//2), (border//2):(bound_rect[2]+border//2)].shape

    cv_img_bckgrnd[ (border//2+(leng-bound_rect[3])//2):(bound_rect[3]+border//2+(leng-bound_rect[3])//2),
                    (border//2+(leng-bound_rect[2])//2):(bound_rect[2]+border//2+(leng-bound_rect[2])//2)] = img_data[pt1:(pt1+bound_rect[3]), pt2:(pt2+bound_rect[2])]
    # roi = cv_img_gs_inv[pt1:(pt1+border*2+leng), pt2:(pt2+border*2+leng)]
    # Resize the image
    roi = cv2.resize(cv_img_bckgrnd, (28, 28), interpolation=cv2.INTER_AREA)
    return roi, bound_rect

def get_char_img_thumbnail(img_fname, gs_fname):

    # convert this pil image to the cv one
    cv_img = cv2.imread(img_fname)
    cv_img_gs = cv2.cvtColor(np.array(cv_img), cv2.COLOR_BGR2GRAY)
    cv_img_gs_inv = 255 - cv_img_gs
    roi, _ = get_char_img_thumbnail_helper(cv_img_gs_inv)
    # roi = cv2.dilate(roi, (3, 3))

    # write this image
    cv2.imwrite(gs_fname, roi)
    return roi

def extend_data_with_lognormal_sampling(data_dict, sample_per_char=100, shift_mean=True):

    # function to diversify the ujichar data
    # the motivation is that, the single stroke letters are not even thus the trained model
    # tend to not perform well for less observed samples. The idea is to generate locally perturbed
    # letters based on human handwriting, with the developed sampling scheme based upon lognormal model
    # note it is desired to balance the number of samples...

    res_data = defaultdict(list)

    # for shifting the data to center it
    for char in sorted(data_dict.keys()):
        n_samples = int(sample_per_char/(len(data_dict[char]) + 1))
        if n_samples == 0:
            # just get the record and continue
            res_data[char] = data_dict[char]
        else:
            # lets first estimate the lognormal parameters for the letter and then perturb them...
            for traj in tqdm(data_dict[char]):
                res_data[char] += [traj]
                res_data[char] += extend_data_with_lognormal_sampling_helper(traj, n_samples, shift_mean)
    return res_data

def extend_data_with_lognormal_sampling_helper(char_traj, n_samples, shift_mean):

    # the input char_traj is flattened with the last entry as the time, get the 2D form
    data_len = int((len(char_traj) - 1)/2)
    t_idx = np.linspace(0, 1.0, data_len)

    # is it necessary to also have noise on this?
    x0 = char_traj[0]
    y0 = char_traj[data_len]
    pos_traj = np.array([char_traj[:data_len], char_traj[data_len:-1]]).T

    # estimate the lognormal parms
    lognorm_parms = np.array(pytk_rz.rxzero_train(pos_traj))
    if np.any(np.isinf(lognorm_parms)):
        print('Unable to extract lognormal parameters. Only use the original trajectory.')
        return []
    n_comps = len(lognorm_parms)

    # generate noise for each components, considering amplitude (+-20%), start angle(+-20 deg) and straightness(+-10% difference)
    ang_difference = lognorm_parms[:, 5] - lognorm_parms[:, 4]
    noises = np.random.randn(n_samples, n_comps, 3) / 3      #white noise to ensure 99.7% samples are within the specified range...
    parm_noises = np.array([ np.array([noise[:, 0]*.2*lognorm_parms[:, 0], np.zeros(n_comps), np.zeros(n_comps), np.zeros(n_comps),
        noise[:, 1]*np.pi/9, noise[:, 1]*np.pi/9 + noise[:, 2]*.1*ang_difference]).T for noise in noises])
    perturbed_parms = np.array([lognorm_parms + parm_noise for parm_noise in parm_noises])

    # apply the noise, remember to flatten and put back the phase scale...
    res_char_trajs = [np.concatenate([pytk_rz.rxzero_traj_eval(perturbed_parm, t_idx, x0, y0)[0].T.flatten(), [char_traj[-1]]]) for perturbed_parm in perturbed_parms]
    if shift_mean:
        mean_coords =  [np.mean(np.reshape(traj[:-1], (2, -1)).T, axis=0) for traj in res_char_trajs]
        for d_idx in range(len(res_char_trajs)):
            data_len = int((len(res_char_trajs[d_idx]) - 1)/2)
            res_char_trajs[d_idx][0:data_len] -= mean_coords[d_idx][0]
            res_char_trajs[d_idx][data_len:-1] -= mean_coords[d_idx][1]
    return res_char_trajs