"""
A module to implement Robust XZERO algorithm to extract Sig-normal model parameters
See Ref. O'Reilly and Plamondon, Development of a Sigma-Lognormal representation for on-line signatures

Note this is neither a complete nor exact implementation of the referenced work. It might contain lots of conjectures
and modifications according to my own understanding as some aspects are not totally clear...
"""

import numpy as np
from scipy.special import erf
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import copy
import itertools

#constant...
sqrt_2 = np.sqrt(2)


def get_vel_profile(stroke):
    """
    get the velocity profile for a stroke
    input is an array of position coordinates
    """
    vel = np.diff(stroke, axis=0)
    vel_prf = np.sum(vel**2, axis=-1)**(1./2)
    return vel_prf

def get_continuous_ang(stroke):
    """
    get continous angle profile
    see Continous-Angle-Time-Curve
    """
    vel_vec = np.diff(stroke, axis=0)
    ang = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])
    #compute the change of ang
    ang_diff = np.diff(ang)
    #x and y:
    #x = cos(ang_diff)
    #y = sin(ang_diff)
    x = np.cos(ang_diff)
    y = np.sin(ang_diff)
    thres = 1e-14
    #update delta diff
    for idx in range(len(ang_diff)):
        if x[idx] > thres:
            ang_diff[idx] = np.arctan2(y[idx], x[idx])
        elif x[idx] < -thres and y[idx] > 0:
            ang_diff[idx] = np.arctan2(y[idx], x[idx]) + np.pi
        elif x[idx] < -thres and y[idx] < 0:
            ang_diff[idx] = np.arctan2(y[idx], x[idx]) - np.pi
        elif np.abs(x[idx]) < thres and y[idx] > 0:
            ang_diff[idx] = np.pi/2
        elif np.abs(x[idx]) < thres and y[idx] < 0:
            ang_diff[idx] = -np.pi/2
    cont_ang_prf = ang[0] + np.cumsum(ang_diff)
    return np.concatenate([[ang[0]], cont_ang_prf])


def vel_profile_registration(vel_profile, vel_max_lst=[]):
    """
    function to generate a sequence of five characteristic points to for stroke vel_profile_registration
    for heuristics to generate initial guess of siglognormal parameters
    See Ref. Section 3.2 - Stroke identification
    """
    #i guess here the inflexion point is specified with respect to velocity profile
    #otherwise the inflexion point of position profile should just be derivative of velocity
    #so crossing zero means the local maximum/minimum of velocity
    #however, this is not for sure, need to check the comprehensive one
    #the above conjecture is confirmed by examing another paper for XZERO
    acc_profile = np.diff(np.append([0], vel_profile))
    acc_profile = np.diff(np.append([0], acc_profile))
    #search local maximum
    #the profile should be well smoothed to prevent identify strokes from noisy profile
    width = 5
    t3_array = []
    #sliding over the profile...
    for start_idx in range(len(vel_profile) - width + 1):
        sub_prf = vel_profile[start_idx:(start_idx+width-1)]
        if np.argmax(sub_prf) == (width-1)/2:   #center point is larger than adjacent ones
            t3_array.append(int(start_idx + (width-1)/2))
        else:
            #<hyin/Feb-17th-2015>or acc is smallest?
            # sub_acc_prf = acc_profile[start_idx:(start_idx+width-1)]
            # if np.argmin(sub_acc_prf) == (width-1)/2 and (sub_prf[-1]-sub_prf[(width-1)/2])*(sub_prf[0]-sub_prf[(width-1)/2])<0:
            #     #saddle point
            #     t3_array.append(start_idx + (width-1)/2)
            pass
    if not t3_array:
        #no local maximum found, invalid profile
        print('Invalid profile - velocity profile must have at least one local maximum...')
        return None, None
    #for each t3, search t_1, t_2, t_4, t_5 (See Ref.):
    #t_1 - first point preceding p3_n which is a local minimum or has a magnitude or less than 1% of v3_n
    #t_2 - first inflexion point before t_3 dot{v} = 0
    #t_4 - first inflexion point after t_3 dot{v} = 0
    #t_5 - first point following p3_n which is a local minimum or has a magnitude of less than 1% of v3_n
    t1_array = []
    t2_array = []
    t4_array = []
    t5_array = []

    #find the maximum v_t3, TO CONFIRM: should this count v_t3 extracted before?
    #let's try...
    if not vel_max_lst:
        vel_max_lst.append(np.max(vel_profile[t3_array]))
        vel_max=vel_max_lst[0]
    else:
        vel_max=vel_max_lst[0]
    t3_array = [t3 for t3 in t3_array if vel_profile[t3] * 12 >= vel_max]

    for t3 in t3_array:
        #the scan is not totally robust, need further investigation. Currently just use this...
        #from t3 search along the two directions...
        t1_found, t2_found, t4_found, t5_found = False, False, False, False
        #for t1 and t2
        for i in range(t3):
            if t3-i < (width - 1)/2:
                break
            sub_prf = vel_profile[int((t3-i-(width-1)/2)):int((t3-i+(width-1)/2))]
            if not t1_found:
                if np.argmin(sub_prf) == int((width-1)/2) or vel_profile[t3-i] < 0.01 * vel_profile[t3]:
                    t1_array.append(t3-i)
                    t1_found = True
            if not t2_found:
                if acc_profile[t3-i+1] * acc_profile[t3-i-1] < 0:
                    t2_array.append(t3-i)
                    t2_found = True
            if t1_found and t2_found:
                break
        if not t1_found:
            #it is unlikely to happen, but for the first stroke, probably we need start of stroke as time index 0
            t1_array.append(0)
        if not t2_found:
            t2_array.append(int((t1_array[-1]+t3)/2.))
        # elif float(t2_array[-1] - t3) / (t3 - t1_array[-1]) > 0.9:
        #   t2_array[-1] = int((t1_array[-1]+t3)/2.)    #too close to t3
        # elif float(t2_array[-1] - t3) / (t3 - t1_array[-1]) < 0.1:
        #   t2_array[-1] = int((t1_array[-1]+t3)/2.)    #too close to t1
        # else:
        #   pass

        #for t4 and t5
        for i in range(len(vel_profile) - t3):
            i_in_array = i+t3+1
            if i_in_array+(width-1)/2 >= len(vel_profile):
                break
            sub_prf = vel_profile[int((i_in_array-(width-1)/2)):int((i_in_array+(width-1)/2))]
            if not t4_found:
                if acc_profile[i_in_array+1] * acc_profile[i_in_array-1] < 0:
                    t4_array.append(i_in_array)
                    t4_found = True
            if not t5_found:
                if np.argmin(sub_prf) == int((width-1)/2) or vel_profile[i_in_array] < 0.01 * vel_profile[t3]:
                    t5_array.append(i_in_array)
                    t5_found = True

            if t4_found and t5_found:
                break
        if not t5_found:
            t5 = len(vel_profile)-1
            t5_array.append(t5)
        else:
            t5 = t5_array[-1]

        if not t4_found:
            t4_array.append(int((t3+t5)/2.))
        #heuristics for inflection points, they might suffer from noise as they need second-order derivatives
        # elif float(t4_array[-1] - t3) / (t5_array[-1] - t3) > 0.9:
        #   t4_array[-1] = int((t3+t5)/2.)  #too close to t5
        # elif float(t4_array[-1] - t3) / (t5_array[-1] - t3) < 0.1:
        #   t4_array[-1] = int((t3+t5)/2.)  #too close to t3
        # else:
        #   pass
    #criteria 1: the area under the curve delimited by p1 and p5 must be greater than the mean minux one standard deviation
    #of the area under the curve of all computed series
    #statistics of area of all computed series...
    # this might be too restrictive if components are not that many
    areas = [sum(vel_profile[t1_array[i]:t5_array[i]+1]) for i in range(len(t3_array))]
    if len(areas) == 0:
        print('No further velocity profile is identified...')
        res = None
    else:
        areas_mean = np.average(areas)
        areas_std = np.std(areas)
        if len(t3_array) == 1:
            #only one candidate, just retain this
            res = np.array([[t1_array[0], t2_array[0], t3_array[0], t4_array[0], t5_array[0]]])
        else:
            #<hyin/Mar-10th-2015> add a threshold to prevent zero std, though this seems to be really rare
            res = np.array([[t1_array[i], t2_array[i], t3_array[i], t4_array[i], t5_array[i]] for i in range(len(t3_array))
                if areas[i] > areas_mean - 2*areas_std - 1e-6])
    #res = np.array([[t1_array[i], t2_array[i], t3_array[i], t4_array[i], t5_array[i]] for i in range(len(t3_array))])
    return res, areas

def vel_profile_reg_test(vel_profile):
    #first plot this vel profile
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(range(len(vel_profile)), vel_profile)
    plt.ion()
    #ready for get registration points...
    vel_max_lst = []
    reg_pnts = vel_profile_registration(vel_profile, vel_max_lst)
    #for each registration points...
    colors = ['r', 'y', 'k', 'g', 'w']
    for reg_pnt_idx in range(5):
        ax.plot(reg_pnts[:, reg_pnt_idx].transpose(),
            vel_profile[reg_pnts[:, reg_pnt_idx].transpose()],
            colors[reg_pnt_idx]+'o')
    plt.ioff()
    #also plot acc profile
    acc_profile = np.diff(np.append([0], vel_profile))
    acc_profile = np.diff(np.append([0], acc_profile))
    ax_acc = fig.add_subplot(212)
    ax_acc.plot(range(len(acc_profile)), acc_profile)
    plt.show()
    return

def rxzero_lognormal_grad(parms, t_idx):
    """
    L(parm) = D / (sqrt(2*pi) * sigma * (t - t0)) * e ^{ -(ln(t-t0) - mu)^2 / (2*sigma^2)}
    parm = D, t0, mu, sigma
    """
    D, t0, mu, sigma = parms
    t_minus_t0 = t_idx - t0
    thres = 1e-6
    sigma = thres if sigma < thres else sigma

    res = np.zeros((len(t_idx), len(parms)))
    dLdD = 1. / (sigma * np.sqrt(2 * np.pi) * (t_minus_t0[t_minus_t0 >= thres]+1e-5)) * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2))
    dLdt0 = D / (sigma * np.sqrt(2 * np.pi) * (t_minus_t0[t_minus_t0 >= thres]+1e-5)**2) * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2)) + \
            D / (sigma * np.sqrt(2 * np.pi) * (t_minus_t0[t_minus_t0 >= thres]+1e-5)) * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2)) * \
            (np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)/(sigma**2) / (t_minus_t0[t_minus_t0 >= thres]+1e-5)
    dLdmu = D / (sigma * np.sqrt(2 * np.pi) * (t_minus_t0[t_minus_t0 >= thres]+1e-5)) * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2)) * \
            (np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)/(sigma**2)
    dLdsig = - D / (sigma**2 * np.sqrt(2 * np.pi) * (t_minus_t0[t_minus_t0 >= thres]+1e-5)) * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2)) + \
            D / (sigma * np.sqrt(2 * np.pi) * (t_minus_t0[t_minus_t0 >= thres]+1e-5)) * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2)) * \
            (np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2 / sigma**3

    #an array of gradient entries for an array of time indices
    #the time indices are associated to the row dimension
    res[t_minus_t0 >= thres, :] = np.array([dLdD, dLdt0, dLdmu, dLdsig]).T
    return res

def rxzero_lognormal_grad_test():
    """
    Unit test for the core lognormal grad function
    """
    D = np.random.rand() * 1.5
    t0 = np.random.rand() * 0.3
    mu = np.random.rand() * 0.5
    sigma = np.random.rand() * 0.5 + 0.1
    test_parm = [D, t0, mu, sigma]

    t_array = np.linspace(0.0, 1.0, 100)
    analytical_grad = rxzero_lognormal_grad(test_parm, t_array)
    #finite difference
    EPS = 1e-5
    num_grad = []
    for idx, parm in enumerate(test_parm):
        parm_plus = copy.copy(test_parm)
        parm_plus[idx] += EPS
        parm_neg = copy.copy(test_parm)
        parm_neg[idx] -= EPS
        eval_plus = rxzero_vel_amp_eval(parm_plus, t_array)
        eval_neg = rxzero_vel_amp_eval(parm_neg, t_array)
        tmp_grad = (eval_plus - eval_neg)/(2.0 * EPS)
        num_grad.append(tmp_grad)

    num_grad = np.array(num_grad).T
    grad_diff = np.mean(np.sum((analytical_grad - num_grad)**2, axis=1))
    return grad_diff

def rxzero_vel_amp_eval(parm, t_idx):
    """
    siglognormal velocity amplitude evaluation
    """
    if len(parm) == 6:
        D, t0, mu, sigma, theta_s, theta_e = parm
    elif len(parm) == 4:
        D, t0, mu, sigma = parm
    else:
        print('Invalid length of parm...')
        return None
    #argument for the sig log normal function
    t_minus_t0 = t_idx - t0
    #truncation to keep time larger than 0
    thres = 1e-6
    #regularize sig
    sigma = thres if sigma < thres else sigma
    #t_minus_t0[t_minus_t0 < thres] = thres
    res = np.zeros(len(t_idx))
    res[t_minus_t0 < thres] = 0.0
    res[t_minus_t0 >= thres] = D / (sigma * np.sqrt(2 * np.pi) * (t_minus_t0[t_minus_t0 >= thres]+1e-5)) * np.exp((np.log(t_minus_t0[t_minus_t0 >= thres]) - mu)**2/(-2*sigma**2))

    return res

def rxzero_vel_amp_eval_grad(parm, t_idx):
    if len(parm) == 6:
        D, t0, mu, sigma, theta_s, theta_e = parm
    elif len(parm) == 4:
        D, t0, mu, sigma = parm
    else:
        print('Invalid length of parm...')
        return None
    res = rxzero_lognormal_grad(parm, t_idx)
    return res

def rxzero_normal_Phi_eval(parm, t_idx):
    """
    Phi function for sig-log-normal model
    parm is a tuple of (D, t0, mu, sigma, theta_s, theta_e), though D is not used here
    """
    D, t0, mu, sigma, theta_s, theta_e = parm
    #argument for the erf function
    t_minus_t0 = t_idx - t0
    #truncation to keep time larger than 0
    thres = 1e-6
    t_minus_t0[t_minus_t0 < thres] = thres
    sigma = thres if sigma < thres else sigma
    #take log, mu and sigma to get the input argument
    z = (np.log(t_minus_t0) - mu) / sigma
    #evaluate erf function
    res = theta_s + (theta_e - theta_s) / 2 * (1 + erf(z))
    return res

def rxzero_normal_Phi_eval_grad(parm, t_idx):
    #see wikipedia for the derivative of erf function
    #derf(z)/dz = 2/sqrt(pi) * e^(-z^2)
    D, t0, mu, sigma, theta_s, theta_e = parm
    #argument for the erf function
    t_minus_t0 = t_idx - t0
    #truncation to keep time larger than 0
    thres = 1e-6
    t_minus_t0[t_minus_t0 < thres] = thres
    sigma = thres if sigma < thres else sigma
    #take log, mu and sigma to get the input argument
    z = (np.log(t_minus_t0) - mu) / sigma

    dLdz = 2.0 / np.sqrt(np.pi) * np.exp(-z**2)
    dLdD = np.zeros(len(t_idx)) * dLdz
    dLdt0 = - 1./(sigma * t_minus_t0) * dLdz
    dLdt0[t_minus_t0 < thres] = 0.0
    dLdmu = - 1./sigma  * dLdz
    dLdsigma = -z/sigma * dLdz

    res = np.zeros((len(t_idx), len(parm)))
    res[:, 0:4] = (theta_e - theta_s) / 2 * np.array([dLdD, dLdt0, dLdmu, dLdsigma]).T

    dLdte = 1./2 * (1 + erf(z))
    dLdts = 1. - dLdte
    res[:, 4:6] = np.array([dLdts, dLdte]).T
    return res

def rxzero_normal_Phi_eval_grad_test():
    """
    Unit test for Phi evaluation gradient function
    """
    D = np.random.rand() * 1.5
    t0 = np.random.rand() * 0.3
    mu = np.random.rand() * 0.5
    sigma = np.random.rand() * 0.5 + 0.1

    theta_s = np.random.rand() * np.pi
    theta_e = np.random.rand() * np.pi

    test_parm = [D, t0, mu, sigma, theta_s, theta_e]

    t_array = np.linspace(0.0, 1.0, 100)
    analytical_grad = rxzero_normal_Phi_eval_grad(test_parm, t_array)
    #finite difference
    EPS = 1e-5
    num_grad = []
    for idx, parm in enumerate(test_parm):
        parm_plus = copy.copy(test_parm)
        parm_plus[idx] += EPS
        parm_neg = copy.copy(test_parm)
        parm_neg[idx] -= EPS
        eval_plus = rxzero_normal_Phi_eval(parm_plus, t_array)
        eval_neg = rxzero_normal_Phi_eval(parm_neg, t_array)
        tmp_grad = (eval_plus - eval_neg)/(2.0 * EPS)
        num_grad.append(tmp_grad)

    num_grad = np.array(num_grad).T
    grad_diff = np.mean(np.sum((analytical_grad - num_grad)**2, axis=1))
    return grad_diff

def rxzero_traj_eval(parms, t_idx, x0, y0):
    """
    evaluate position trajectory given parameters and time index series...
    Note that this is compatible for evaluting combined strokes so the parameters
    are actualy an array of parm tuple
    """
    v_amp_array = np.array([rxzero_vel_amp_eval(parm, t_idx) for parm in parms])
    phi_array = np.array([rxzero_normal_Phi_eval(parm, t_idx) for parm in parms])
    dt = t_idx[1] - t_idx[0]
    v_x = np.sum(np.abs(v_amp_array) * np.cos(phi_array), axis=0)
    v_y = np.sum(np.abs(v_amp_array) * np.sin(phi_array), axis=0)
    v_vec = np.concatenate([[v_x], [v_y]], axis=0).transpose()
    #more consideration is needed for this dt...
    res_pos = np.array([x0, y0]) + np.cumsum(v_vec, axis=0) * dt
    return res_pos, v_vec

def rxzero_traj_eval_grad(parms, t_idx):
    """
    Analytical gradient for evaluated trajectory with respect to the log-normal parameters
    It is expected to boost the optimization performance when the parameters are high-dimensional...
    """
    v_amp_array = np.array([rxzero_vel_amp_eval(parm, t_idx) for parm in parms])
    phi_array = np.array([rxzero_normal_Phi_eval(parm, t_idx) for parm in parms])
    v_amp_grad_array = np.array([np.vstack([rxzero_vel_amp_eval_grad(parm[0:4], t_idx).T, np.zeros((2, len(t_idx)))]).T for parm in parms])
    phi_grad_array = np.array([rxzero_normal_Phi_eval_grad(parm, t_idx) for parm in parms])

    v_x_grad = np.concatenate([(v_amp_grad_array[parm_idx].T * np.cos(phi_array[parm_idx]) - v_amp_array[parm_idx] * np.sin(phi_array[parm_idx]) * phi_grad_array[parm_idx].T).T for parm_idx in range(len(parms))], axis=1)
    v_y_grad = np.concatenate([(v_amp_grad_array[parm_idx].T * np.sin(phi_array[parm_idx]) + v_amp_array[parm_idx] * np.cos(phi_array[parm_idx]) * phi_grad_array[parm_idx].T).T for parm_idx in range(len(parms))], axis=1)

    dt = t_idx[1] - t_idx[0]
    pos_x_grad = np.cumsum(v_x_grad, axis=0) * dt
    pos_y_grad = np.cumsum(v_y_grad, axis=0) * dt
    return np.array([pos_x_grad, pos_y_grad]), np.array([v_x_grad, v_y_grad])

def rxzero_traj_eval_grad_test():
    num_comps = 5

    def prepare_parm():
        D = np.random.rand() * 1.5
        t0 = np.random.rand() * 0.3
        mu = np.random.rand() * 0.5
        sigma = np.random.rand() * 0.5 + 0.1

        theta_s = np.random.rand() * np.pi
        theta_e = np.random.rand() * np.pi
        return np.array([D, t0, mu, sigma, theta_s, theta_e])

    test_parms = np.array([prepare_parm() for i in range(num_comps)])

    t_array = np.linspace(0.0, 1.0, 100)
    analytical_grad, _ = rxzero_traj_eval_grad(test_parms, t_array)
    #finite difference
    EPS = 1e-5
    num_grad_x = []
    num_grad_y = []
    for idx, test_parm in enumerate(test_parms.flatten()):
        parm_plus = copy.copy(test_parms.flatten())
        parm_plus[idx] += EPS
        parm_neg = copy.copy(test_parms.flatten())
        parm_neg[idx] -= EPS
        parm_plus_conc = np.reshape(parm_plus, (-1, 6))
        parm_neg_conc = np.reshape(parm_neg, (-1, 6))
        eval_plus, _ = rxzero_traj_eval(parm_plus_conc, t_array, 0, 0)
        eval_neg, _ = rxzero_traj_eval(parm_neg_conc, t_array, 0, 0)
        tmp_grad = (eval_plus - eval_neg)/(2.0 * EPS)
        num_grad_x.append(tmp_grad[:, 0])
        num_grad_y.append(tmp_grad[:, 1])
    num_grad_x = np.array(num_grad_x).T
    num_grad_y = np.array(num_grad_y).T

    grad_diff = np.mean(np.sum((analytical_grad[0] - num_grad_x)**2, axis=1)) + np.mean(np.sum((analytical_grad[1] - num_grad_y)**2, axis=1))
    return grad_diff

def rxzero_estimation(reg_pnts, vel_profile, dt=0.01):
    """
    robust rxzero estimation given characteristic points and velocity profile...
    """
    pnt_idx_array = [1, 2, 3]
    sigma_array = []
    #three combinations of characteristic velocities for estimating sigma
    vel_idx_itr = itertools.combinations(pnt_idx_array, 2) #pick two points from p2, p3, p4, note the array must be sorted...
    #record this... it seems the iter can only be used once... really?
    vel_idx_lst = []
    for itr in vel_idx_itr:
        vel_idx_lst.append(itr)
    for idx1, idx2 in vel_idx_lst:
        if idx1==1 and idx2==2:
            beta_23 = vel_profile[reg_pnts[1]] / vel_profile[reg_pnts[2]]
            sigma_array.append(np.sqrt(-2-2*np.log(beta_23) - 1/(2*np.log(beta_23))))
        elif idx1==1 and idx2==3:
            beta_42 = vel_profile[reg_pnts[3]] / vel_profile[reg_pnts[1]]
            sigma_array.append(np.sqrt(2*np.sqrt(1 + np.log(beta_42)**2) - 2))
        elif idx1==2 and idx2==3:
            beta_43 = vel_profile[reg_pnts[3]] / vel_profile[reg_pnts[2]]
            sigma_array.append(-2 - 2*np.log(beta_43) - 1/(2*np.log(beta_43)))
        else:
            print('This should not happen, check the code...')
    candidate_parms = []

    for sigma in sigma_array:
        #a
        a_array = np.array([3*sigma, 1.5*sigma**2+sigma*np.sqrt(0.25*sigma**2+1),
                sigma**2, 1.5*sigma**2-sigma*np.sqrt(0.25*sigma**2+1), -3*sigma])
        #pick two time points t_i & t_j to evaluate other parameters
        for idx1, idx2 in vel_idx_lst:
            #mu
            mu = np.log((reg_pnts[idx1] - reg_pnts[idx2]) * dt / (np.exp(-a_array[idx1]) - np.exp(-a_array[idx2])))
            #t0
            t0 = reg_pnts[idx1] * dt - np.exp(mu - a_array[idx1])
            #D
            D = np.sqrt(2*np.pi) * vel_profile[reg_pnts[idx1]] * sigma * np.exp(a_array[idx1]**2/(2*sigma**2)
                -a_array[idx1]+mu)
            candidate_parms.append((D, t0, mu, sigma))
    #find the best candidate
    #print candidate_parms
    vel_eval_array = np.array([rxzero_vel_amp_eval(parm, np.arange(len(vel_profile))*dt) for parm in candidate_parms])
    rec_err_array = np.sum((vel_eval_array[:, reg_pnts[0]:reg_pnts[4]] - vel_profile[reg_pnts[0]:reg_pnts[4]])**2, axis=1)
    return  candidate_parms[np.argmin(rec_err_array)]

def rxzero_sig_reg_pnts_ang(pos_traj, reg_pnts):
    vel_vec = np.diff(pos_traj, axis=0)
    #only reg_pnts 1, 2, 3 matters...
    phi_p1 = np.arctan2(vel_vec[reg_pnts[1], 1], vel_vec[reg_pnts[1], 0])
    phi_p2 = np.arctan2(vel_vec[reg_pnts[2], 1], vel_vec[reg_pnts[2], 0])
    phi_p3 = np.arctan2(vel_vec[reg_pnts[3], 1], vel_vec[reg_pnts[3], 0])

    reg_pnts_phi = [None, phi_p1, phi_p2, phi_p3, None]
    return reg_pnts_phi

def rxzero_sig_ang_estimation(vel_parm, reg_pnts_phi):
    D, t0, mu, sigma = vel_parm
    a_array = np.array([3*sigma, 1.5*sigma**2+sigma*np.sqrt(0.25*sigma**2+1),
                sigma**2, 1.5*sigma**2-sigma*np.sqrt(0.25*sigma**2+1), -3*sigma])
    l_array = np.array([0,
    D/2*(1+erf(-a_array[1]/(sigma*sqrt_2))),
    D/2*(1+erf(-a_array[2]/(sigma*sqrt_2))),
    D/2*(1+erf(-a_array[3]/(sigma*sqrt_2))),
    D])
    #only extract velocity vector at the interested registration points, the end points are ont reliable

    #<hyin/Feb-11-2015> guess need to have further adjustment because of the domain of angular position
    #e.g., phi_p1 = -2.5, phi_p3 = 2.5, it should be an arc on the below side of circle
    #however, since arctan2 return a value between [-pi, pi], vanilla difference of phi_p3 and phi_p1
    #will cause another circle
    ang_diff = reg_pnts_phi[3] - reg_pnts_phi[1]
    if ang_diff > np.pi:
        ang_diff -= 2*np.pi
    elif ang_diff < -np.pi:
        ang_diff += 2*np.pi
    else:
        pass

    delta_phi = ang_diff /(l_array[3] - l_array[1])
    theta_s = reg_pnts_phi[2] - delta_phi*(l_array[2]-l_array[0])
    theta_e = reg_pnts_phi[2] + delta_phi*(l_array[4]-l_array[2])
    # print 'l_array:', l_array
    # print 'angle guess (reg_pnts):', phi_array[reg_pnts[0]], phi_array[reg_pnts[2]], phi_array[reg_pnts[4]]
    # print 'angle guess:', theta_s, theta_e
    return theta_s, theta_e

def rxzero_sig_local_optimization(parm, vel_traj, vel_profile, reg_pnts, dt=0.01):
    """
    a subroutine to local optimization for each extracted stroke parameters
    According to the Ref., mu, sigma, theta_s and theta_e will be optimized. D and t_0 will
    be inferred by making the estimated lognormal maximum point equal to the numerical one
    <hyin/Jan-12-2015>
    However, it is not quite clear what specific objective it attempts to optimize. I guess
    it should be the position trajectory, as theta_s & theta_e are included. Velocity profile
    should be independent to these as I understand
    <hyin/Jan-20-2015>
    try velocity vector for this time
    """
    t_array = np.arange(len(vel_traj))*dt
    D, t0, mu, sigma, theta_s, theta_e = parm
    def obj_func(x, *args):
        vel_traj, D, t0 = args
        #objective function
        #x: mu, sigma, theta_s, theta_e
        _, res_vel_vec = rxzero_traj_eval([(D, t0, x[0], x[1], x[2], x[3])], t_array, 0, 0)
        """
        not sure if the full length traj should be compared or only the section between characteristic points are required
        """
        return np.sum(np.sum((res_vel_vec[reg_pnts[0]:reg_pnts[4], :] - vel_traj[reg_pnts[0]:reg_pnts[4], :])**2, axis=1))

    def obj_func_grad(x, *args):
        vel_traj, D, t0 = args
        _, res_vel_vec = rxzero_traj_eval([(D, t0, x[0], x[1], x[2], x[3])], t_array, 0, 0)
        _, vel_vec_grad = rxzero_traj_eval_grad([(D, t0, x[0], x[1], x[2], x[3])], t_array)
        #slice the effective dimensions
        grad = np.sum(2 * ((res_vel_vec[reg_pnts[0]:reg_pnts[4], 0] - vel_traj[reg_pnts[0]:reg_pnts[4], 0]) * vel_vec_grad[0][reg_pnts[0]:reg_pnts[4], 2:].T +
                    (res_vel_vec[reg_pnts[0]:reg_pnts[4], 1] - vel_traj[reg_pnts[0]:reg_pnts[4], 1]) * vel_vec_grad[1][reg_pnts[0]:reg_pnts[4], 2:].T), axis=1)
        return grad

    theta_scale = 0.1
    bounds_array = [
        #here note the bound of sig, it must be larger than zero...
        (mu-sigma, mu+sigma), (0.5*sigma, 1.5*sigma),       #mu, sig
        #(theta_s-np.pi*theta_scale, theta_s+np.pi*theta_scale), (theta_e-np.pi*theta_scale, theta_e+np.pi*theta_scale)]                                    #theta_s, theta_e
        (None, None), (None, None)]                                 #theta_s, theta_e

    opt_res = sciopt.minimize(  fun=obj_func,
                                jac=obj_func_grad,
                                x0=[mu, sigma, theta_s, theta_e], args=(vel_traj, D, t0),
                                bounds=bounds_array,
                                tol=1e-8,
                                options={'disp': False})

    #'xtol': 1e-8,
    # print('xopt:' + str(opt_res.x))
    # print('opt_func:' + str(obj_func(opt_res.x, vel_traj, D, t0)))

    _, res_vel_vec_init = rxzero_traj_eval([parm], t_array, 0, 0)
    _, res_vel_vec_opt = rxzero_traj_eval([(D, t0, opt_res.x[0], opt_res.x[1], opt_res.x[2], opt_res.x[3])], t_array, 0, 0)
    # print np.sum(np.sum((vel_traj[reg_pnts[0]:reg_pnts[4], :] - res_vel_vec_init[reg_pnts[0]:reg_pnts[4], :])**2, axis=1))
    # print np.sum(np.sum((vel_traj[reg_pnts[0]:reg_pnts[4], :] - res_vel_vec_opt[reg_pnts[0]:reg_pnts[4], :])**2, axis=1))
    #infer D and t0
    t0 = reg_pnts[2]*dt - np.exp(opt_res.x[0]-opt_res.x[1]**2)
    D = np.sqrt(2*np.pi) * vel_profile[reg_pnts[2]] * opt_res.x[1] * np.exp(opt_res.x[0]-opt_res.x[1]**2/2)
    return (D, t0, opt_res.x[0], opt_res.x[1], opt_res.x[2], opt_res.x[3])

#NOTE: There should be something wrong with the paper (either unconscientious or conscientious)
#I guess the estimation of the angular positions should be based on the position trajectory without add/substraction
#otherwise, the initial guess might be serverely distorted...
#so I guess there are some implicit rules for different parameters:
#characteristic points: vel_profile - work on add/sub trajectories
#D, t0, mu, sig:        vel_profile - work on add/sub trajectories
#theta_s, theta_e:      vel_vec     - work on original trajectories
#local optimization:    vel_vec     - work on original trajectories (subsection)
#global optimizaiton:   position/vel_vec iteratively
#let's test these ideas...<hyin/Jan-20-2015>
def rxzero_sig_extract(pos_traj, bFirstMode=True, dt=0.01, premodes=None, global_pos_traj=None, vel_max_lst=[]):
    """
    a subroutine to extract a stroke
    """
    #get velocity profile
    vel_profile = get_vel_profile(pos_traj)/dt
    #get vel vec
    vel_vec = np.diff(pos_traj, axis=0)/dt
    #get phi_array - ang position trajectory: same length with vel_profile?
    if global_pos_traj is None:
        phi_array = get_continuous_ang(pos_traj)
    else:
        # phi_array = get_continuous_ang(global_pos_traj)
        # vel_vec = np.diff(global_pos_traj, axis=0)/dt
        phi_array = get_continuous_ang(pos_traj)

    #get a series of registration points
    reg_pnts, areas = vel_profile_registration(vel_profile, vel_max_lst)
    #locate the mode
    if bFirstMode:
        if reg_pnts is None:
            #terminate first mode estimation...
            return None, None
        if premodes is None or not premodes:
            #no previously extracted modes, just choose the first one
            mode_reg_pnts = reg_pnts[0]
        else:
            #we must find the reg points with t3 greater than previous ones...
            #this might also be applied to most important mode, but let's first ignore that...
            #note, inferring
            mode_reg_pnts = None
            for candidate_reg_pnts in reg_pnts:
                if all(candidate_reg_pnts[2] > premode for premode in premodes):
                    mode_reg_pnts = candidate_reg_pnts
                    break
            if mode_reg_pnts is None:
                #no validate reg points, cannot conduct extraction...
                return None, None
    else:
        #get the most important mode
        mode_reg_pnts = reg_pnts[np.argmax(areas)]

    #rxzero estimation
    D, t0, mu, sigma = rxzero_estimation(mode_reg_pnts, vel_profile, dt)
    #angle estimation
    #note the phi_array might not be a robust estimation from the full section
    #try subsection limits by registration points...
    reg_pnts_phi = rxzero_sig_reg_pnts_ang(pos_traj, mode_reg_pnts)
    theta_s, theta_e = rxzero_sig_ang_estimation((D, t0, mu, sigma), reg_pnts_phi)
    # print 'before local optimization:'
    # print (D, t0, mu, sigma, theta_s, theta_e)
    #opt_parm = (D, t0, mu, sigma, theta_s, theta_e)
    #local optimization
    opt_parm = rxzero_sig_local_optimization((D, t0, mu, sigma, theta_s, theta_e), vel_vec, vel_profile, mode_reg_pnts, dt)
    return opt_parm, mode_reg_pnts

def fit_parm_component_with_global_optimization(pos_traj, parms, free_comp_idx, dt=0.01, maxIters=1, solver_max_iters=200):
    """
    global optimization for adjusting free parms to fit given position trajectory
    this is similar to what we did in the rxzero global optimization
    but now we want to see how each component can fit the overall letter profile
    this is an extended version - it might be better to merge it with the existing rxzero, but let's have it here for now - Done <hyin/Mar-29th-2016>
    """
    #following the ref, it is something like a variational inference procedure
    #1. optimize velocity with respect to D, t0, mu, sig
    #2. optimize position with respect to theta_s, theta_e
    #3. full optimization with respect to D, t0, mu, sig, theta_s, theta_e
    t_array = np.arange(len(pos_traj))*dt
    t_vel_array = np.arange(len(pos_traj)-1)*dt
    vel_vec_traj = np.diff(pos_traj, axis=0) / dt
    def obj_func_step_1(x, *args):
        curr_parms, vel_traj, free_comp_idx = args
        #construct parameters
        vel_parms = np.reshape(x, (-1, 4))
        # conc_parms = np.concatenate([vel_parms, theta_parms], axis=1)
        conc_parms = np.array(curr_parms, copy=True)
        conc_parms[free_comp_idx, [0,1,2,3]] = vel_parms
        _, eval_vel = rxzero_traj_eval(conc_parms, t_vel_array, 0, 0)
        _, vel_vec_grad = rxzero_traj_eval_grad(conc_parms, t_vel_array)
        #slice the effective dimensions, only dimensions of free vel parms are needed
        effective_dims = np.concatenate([range(idx * curr_parms.shape[1], idx * curr_parms.shape[1] + 4) for idx in np.concatenate(free_comp_idx)])
        grad = np.sum(2 * ((eval_vel[:, 0] - vel_traj[:, 0]) * vel_vec_grad[0][:, effective_dims].T +
                    (eval_vel[:, 1] - vel_traj[:, 1]) * vel_vec_grad[1][:, effective_dims].T), axis=1)
        return np.sum(np.sum(np.abs((eval_vel - vel_traj)), axis=1)), grad

    def obj_func_step_2(x, *args):
        curr_parms, pos_traj, free_comp_idx = args
        theta_parms = np.reshape(x, (-1, 2))
        conc_parms = np.array(curr_parms, copy=True)
        conc_parms[free_comp_idx, [4, 5]] = theta_parms
        eval_pos, _ = rxzero_traj_eval(conc_parms, t_array, pos_traj[0, 0], pos_traj[0, 1])
        pos_vec_grad, _ = rxzero_traj_eval_grad(conc_parms, t_array)
        #slice the effective dimensions
        effective_dims = np.concatenate([range(comp_idx*6+4, comp_idx*6+6) for comp_idx in np.concatenate(free_comp_idx)])
        grad = np.sum(2 * ((eval_pos[:, 0] - pos_traj[:, 0]) * pos_vec_grad[0][:, effective_dims].T +
                    (eval_pos[:, 1] - pos_traj[:, 1]) * pos_vec_grad[1][:, effective_dims].T), axis=1)
        return np.sum(np.sum(np.abs((eval_pos - pos_traj)), axis=1)), grad

    def obj_func_step_3(x, *args):
        curr_parms, vel_traj, free_comp_idx = args
        full_parms = np.reshape(x, (-1, 6))
        conc_parms = np.array(curr_parms, copy=True)
        conc_parms[free_comp_idx, range(6)] = full_parms
        _, eval_vel = rxzero_traj_eval(conc_parms, t_vel_array, 0, 0)
        _, vel_vec_grad = rxzero_traj_eval_grad(conc_parms, t_vel_array)
        #slice the effective dimensions, only dimensions of free vel parms are needed
        effective_dims = np.concatenate([range(idx * curr_parms.shape[1], idx * curr_parms.shape[1] + curr_parms.shape[1]) for idx in np.concatenate(free_comp_idx)])
        grad = np.sum(2 * ((eval_vel[:, 0] - vel_traj[:, 0]) * vel_vec_grad[0][:, effective_dims].T +
                    (eval_vel[:, 1] - vel_traj[:, 1]) * vel_vec_grad[1][:, effective_dims].T), axis=1)

        return np.sum(np.sum(np.abs((eval_vel - vel_traj)), axis=1)), grad

    curr_parms = np.array(parms)
    opt_dict = {'maxiter':solver_max_iters}
    #<hyin/Feb-20th-2015> fixed a bug that leads to no solution
    #actually, skipping step 1 and step 3 seems okay for sake of speed (buggy version)
    #better to add an option to choose precise verions/approximate version
    #or derive closed form gradient/do offline training on a workstation...
    for i in range(maxIters):
        #step 1
        x_init_1 = curr_parms[free_comp_idx, range(4)].flatten()
        #construct bounds...
        bounds_step_1 = []
        for parm in curr_parms[free_comp_idx, range(curr_parms.shape[1])]:
            parm_bounds = [ (0.5*parm[0], 3.5*parm[0]), #D
                            (parm[1] - 0.5*np.abs(parm[1]), parm[1] + 0.5*np.abs(parm[1])), #t0
                            (None, None),               #mu
                            (0.00, 3)                 #sigma
                            ]
            bounds_step_1 = bounds_step_1 + parm_bounds
        opt_res = sciopt.minimize(obj_func_step_1, x_init_1, args=(curr_parms, vel_vec_traj, free_comp_idx), jac=True, bounds=bounds_step_1, options=opt_dict)
        curr_parms[free_comp_idx, [0, 1, 2, 3]] = np.reshape(opt_res.x, (-1, 4))

        #step 2
        x_init_2 = curr_parms[free_comp_idx, range(4, 6)].flatten()
        bounds_step_2 = []
        for parm in curr_parms[free_comp_idx, range(curr_parms.shape[1])]:
            parm_bounds = [ (-np.pi, np.pi),               #theta_s
                            (-np.pi, np.pi)                #theta_e
                            ]
            bounds_step_2 = bounds_step_2 + parm_bounds
        opt_res = sciopt.minimize(obj_func_step_2, x_init_2, args=(curr_parms, pos_traj, free_comp_idx), jac=True, bounds=bounds_step_2, options=opt_dict)
        curr_parms[free_comp_idx, [4, 5]] = np.reshape(opt_res.x, (-1, 2))

        #step 3
        x_init_3 = curr_parms[free_comp_idx, range(curr_parms.shape[1])].flatten()
        #construct bounds...
        bounds_step_3 = []
        for parm in curr_parms[free_comp_idx, range(curr_parms.shape[1])]:
            parm_bounds = [ (0.5*parm[0], 3.5*parm[0]), #D
                            (parm[1] - 0.5*np.abs(parm[1]), parm[1] + 0.5*np.abs(parm[1])), #t0
                            (None, None),               #mu
                            (0.0, 3.0),               #sigma
                            (-np.pi, np.pi),               #theta_s
                            (-np.pi, np.pi)                #theta_e
                            ]
            bounds_step_3 = bounds_step_3 + parm_bounds

        opt_res = sciopt.minimize(obj_func_step_3, x_init_3, args=(curr_parms, vel_vec_traj, free_comp_idx), jac=True, bounds=bounds_step_3, options=opt_dict)
        curr_parms[free_comp_idx, range(6)] = np.reshape(opt_res.x, (-1, 6))[:, range(6)]

    #evaluate position reconstruction error
    eval_pos, eval_vel = rxzero_traj_eval(curr_parms, t_array, pos_traj[0, 0], pos_traj[0, 1])
    # recons_err = np.sum(np.sum((eval_pos - pos_traj)**2, axis=1))
    #evaluate reconstruction error from velocity profile?
    recons_err = np.sum(np.sum((eval_vel[:-1, :] - vel_vec_traj)**2, axis=1))
    return curr_parms, recons_err

def rxzero_global_optimization(pos_traj, parms, dt=0.01, maxIters=1, solver_max_iters=200):
    """
    global optimization for all parms to fit given position trajectory
    <hyin/Mar-29th-2016> unify routines to allow optimization constrained on given components
    """
    free_comp_idx = np.array([range(len(parms))]).T
    curr_parms, _ = fit_parm_component_with_global_optimization(pos_traj, parms, free_comp_idx, dt, maxIters, solver_max_iters)

    return curr_parms

def rxzero_add_stroke(pos_traj, logsig_parms, dt=0.01, reg_pnts=None):
    """
    funciton to add a stroke to position trajectory from logsig parameters
    """
    t_array = np.arange(len(pos_traj))*dt
    #probably we should use the actual phi_array, as the angular estimation might not be accurate
    stroke, _ = rxzero_traj_eval([logsig_parms], t_array, 0, 0)
    if reg_pnts is None:
        new_stroke = pos_traj + stroke
    else:
        #only deal with p1-p5 section
        new_stroke = copy.copy(pos_traj)
        new_stroke[reg_pnts[0]:reg_pnts[4], :] += stroke[reg_pnts[0]:reg_pnts[4], :]
    return new_stroke

def rxzero_subtract_stroke(pos_traj, logsig_parms, dt=0.01, reg_pnts=None):
    """
    function to subtract a stroke to position trajectory from logsig parameters
    """
    t_array = np.arange(len(pos_traj))*dt
    stroke, _ = rxzero_traj_eval([logsig_parms], t_array, 0, 0)
    if reg_pnts is None:
        new_stroke = pos_traj - stroke
    else:
        #only deal with p1-p5 section
        new_stroke = copy.copy(pos_traj)
        # print reg_pnts
        # print new_stroke[reg_pnts[0]:reg_pnts[4], :]
        # print stroke[reg_pnts[0]:reg_pnts[4], :]
        new_stroke[reg_pnts[0]:reg_pnts[4], :] -= stroke[reg_pnts[0]:reg_pnts[4], :]
    return new_stroke

def rxzero_calc_SNR(vel_vec, logsig_parms, reg_pnts, dt=0.01):
    """
    helper function to calculate the fitness of reconstruction of stroke
    """
    t_array = np.arange(len(vel_vec))*dt
    v_n = vel_vec
    _, v_a = rxzero_traj_eval([logsig_parms], t_array, 0, 0)
    #find two inflection points...
    D, t0, mu, sig, theta_s, theta_e = logsig_parms
    t2_idx = reg_pnts[1]
    t4_idx = reg_pnts[3]
    v_n = v_n[t2_idx:(t4_idx+1), :]
    v_a = v_a[t2_idx:(t4_idx+1), :]
    SNR_vxy = 20 * np.log( np.sum(np.sum(v_n**2, axis=1)) / np.sum(np.sum((v_n - v_a)**2, axis=1)) )
    return SNR_vxy

def rxzero_calc_SNR_full(vel_vec, logsig_parms, dt=0.01):
    t_array = np.arange(len(vel_vec))*dt
    v_n = vel_vec
    _, v_a = rxzero_traj_eval(logsig_parms, t_array, 0, 0)
    SNR_full = 20 * np.log( np.sum(np.sum(v_n**2, axis=1)) / np.sum(np.sum((v_n - v_a)**2, axis=1)) )
    return SNR_full

def rxzero_train(pos_traj, dt = 0.01, global_opt_itrs=1, verbose=False, getregpnts=False):

    #the first mode
    SNRmin = 20  #original: 30
    IterMax = 3
    I = 0
    J = 0
    bFirstMode = True
    maxStrokes = 15

    vel_vec = np.diff(pos_traj, axis=0)/dt
    phi_array = get_continuous_ang(pos_traj)
    curr_traj = copy.copy(pos_traj)
    parms = []
    reg_pnts_lst = []
    premodes = []
    vel_max_lst = []
    #initialize, the first P
    p0, reg_pnts = rxzero_sig_extract(pos_traj, bFirstMode, dt, global_pos_traj=pos_traj, vel_max_lst=vel_max_lst)
    premodes.append(reg_pnts[2])
    if p0 is None:
        return None

    parms.append(p0)
    reg_pnts_lst.append(reg_pnts)

    axes = None

    while I < IterMax:

        #check current estimation...
        if verbose:
            axes = rxzero_plot_helper(curr_traj, [parms[J]], vel_vec, reg_pnts, dt, axes)
            input('Current Guess. Press ENTER to continue...')

        #substract stroke
        curr_traj = rxzero_subtract_stroke(curr_traj, parms[J], dt, reg_pnts=None)

        if verbose:
            axes = rxzero_plot_helper(curr_traj, [parms[J]], vel_vec, reg_pnts, dt, axes)
            input('Cut current guess. Press ENTER to continue...')

        #extract J+1
        p_j_plus_1, reg_pnts_plus_1 = rxzero_sig_extract(curr_traj, bFirstMode, dt, premodes[0:(J+1)], global_pos_traj=pos_traj, vel_max_lst=vel_max_lst)
        if p_j_plus_1 is None:
            #it seems to be no next stroke
            #estimate based on current traj
            pass
        elif any([not (float('-inf') < float(num) < float('inf')) for num in p_j_plus_1]):
            #also no valid next stroke if the extracted parameters are invalid
            #<hyin/Feb-11-2015> this sometimes happens when the end stroke does not converge to zero velocity
            #see case ['D'][106]
            pass
        else:
            if I == 0:
                parms.append(p_j_plus_1)    #add new parameters for the first iteration
                reg_pnts_lst.append(reg_pnts_plus_1)
                premodes.append(reg_pnts_plus_1[2])
            else:
                parms[J+1] = p_j_plus_1
                reg_pnts_lst[J+1] = reg_pnts_plus_1
                premodes[J+1] = reg_pnts_plus_1[2]

            if verbose:
                axes = rxzero_plot_helper(curr_traj, [parms[J+1]], vel_vec, reg_pnts_plus_1, dt, axes)
                input('Guess For J+1. Press ENTER to continue...')

        #eliminate the impact the p_j_plus_1
        #first add back p_j
        curr_traj = rxzero_add_stroke(curr_traj, parms[J], dt, reg_pnts=None)

        if verbose:
            axes = rxzero_plot_helper(curr_traj, [parms[J]], vel_vec, reg_pnts, dt, axes)
            input('Add back initial guess. Press ENTER to continue...')

        if p_j_plus_1 is None:
            pass
        elif any([not (float('-inf') < float(num) < float('inf')) for num in p_j_plus_1]):
            #also no valid next stroke if the extracted parameters are invalid
            #<hyin/Feb-11-2015> this sometimes happens when the end stroke does not converge to zero velocity
            #see case ['D'][106]
            pass
        else:
            curr_traj = rxzero_subtract_stroke(curr_traj, parms[J+1], dt, reg_pnts=None)

            if verbose:
                axes = rxzero_plot_helper(curr_traj, [parms[J+1]], reg_pnts=None, dt=dt, axes=axes)
                input('Substract J+1 to make an isolated stroke. Press ENTER to continue...')

            #formal extraction, note ignore current guess for J & J+1
            extract_parm, reg_pnts_curr = rxzero_sig_extract(curr_traj, bFirstMode, dt, premodes[0:-2], global_pos_traj=pos_traj, vel_max_lst=vel_max_lst)
            if extract_parm is not None:
                #check if values are valid
                if any(np.isnan(extract_parm)) or any(np.isinf(extract_parm)):

                    #<hyin/Feb-17th-2015> there seems to be a possible issue, subtracting J+1 stroke
                    #makes the isolated stroke severaly distorted which gives invalid parameters
                    #in this case, let's try to ignore the current stroke and directly go for the next one

                    parms[J] = parms[J+1]
                    premodes[J] = premodes[J+1]
                    print('----------------------------')
                    print('Invalid parameters extracted after subtraction J+1 stroke, use J+1 as J')
                    print('----------------------------')
                    curr_traj = rxzero_add_stroke(curr_traj, parms[J], dt, reg_pnts=None)
                    continue
                else:
                    parms[J] = extract_parm
            if reg_pnts_curr is None:
                pass
                #keep initial guess unchanged if formal extraction is failed
                #what happens here?
            else:
                reg_pnts_lst[J] = reg_pnts_curr
                premodes[J] = reg_pnts_curr[2]

        reg_pnts_curr = reg_pnts_lst[J]

        if verbose:
            axes = rxzero_plot_helper(curr_traj, [parms[J]], vel_vec, reg_pnts_curr, dt, axes)
            input('Isolated one. Press ENTER to continue...')
        #check SNR
        SNR = rxzero_calc_SNR(vel_vec, parms[J], reg_pnts_curr, dt)
        #<hyin/Jan-31-2015>note, i don't understand the meaning of this SNR
        #from the Ref., it's trying to compare against the original trajectories
        #but there seems to be some example that the substracted profile deviates from the target
        #making the repeated extraction meaning less...
        #temporarily, fixed this as True. Meaning we trust the first mode extraction and always continue
        if SNR > SNRmin or I > IterMax or True:
            #find a good reconstruct
            #only extract the next one when it exists
            if len(parms) == J+2:
                #we are satisfied with this extraction of stroke, move on to the next one...
                curr_traj = rxzero_add_stroke(curr_traj, parms[J+1], dt, reg_pnts=None)
                curr_traj = rxzero_subtract_stroke(curr_traj, parms[J], dt, reg_pnts=None)
                J = J + 1
                I = 0
                #new initial guess for next stroke extraction
                p_j, reg_pnts = rxzero_sig_extract(curr_traj, bFirstMode, dt, premodes[0:-1], global_pos_traj=pos_traj, vel_max_lst=vel_max_lst)
                if p_j is not None and not (any(np.isnan(p_j)) or any(np.isinf(p_j))):
                    parms[J] = p_j
                    reg_pnts_lst[J] = reg_pnts
                    premodes[J] = reg_pnts[2]
                else:
                    #no more stroke to extract...
                    break
            else:
                break
        else:
            #i'm really not sure if we should add back the P+1 now...
            I = I + 1
            #curr_traj = rxzero_add_stroke(curr_traj, parms[J+1])
    #before global optimization
    if verbose:
        rxzero_plot_helper(pos_traj, parms, vel_vec_n=vel_vec, reg_pnts=None, dt=dt, axes=None)
        input('Initial guess before global optimization...')

    #do global optimization
    parms = rxzero_global_optimization(pos_traj, parms, dt, maxIters=global_opt_itrs)

    if getregpnts:
        return parms, reg_pnts_lst
    else:
        return parms

def rxzero_plot_helper(pos_traj, parms, vel_vec_n=None, reg_pnts=None, dt=0.01, axes=None):
    gs = plt.GridSpec(3, 2)
    if axes is None:
        fig = plt.figure()
        ax_pos = fig.add_subplot(gs[0, 0:2])
        ax_vel = fig.add_subplot(gs[1, 0])
        ax_vel_x = fig.add_subplot(gs[2, 0])
        ax_vel_y = fig.add_subplot(gs[2, 1])
        ax_ang = fig.add_subplot(gs[1, 1])
        plt.ion()
        plt.show()
    else:
        fig, ax_pos, ax_vel, ax_vel_x, ax_vel_y, ax_ang = axes

    t_array = np.arange(len(pos_traj))*dt
    vel_profile = get_vel_profile(pos_traj)/dt
    vel_t_array = np.arange(len(vel_profile))*dt
    phi_array = get_continuous_ang(pos_traj)

    ax_pos.plot(pos_traj[:, 0], -pos_traj[:, 1])
    # ax_pos.set_aspect('equal')
    ax_vel.plot(vel_t_array, vel_profile)

    vel_vec = np.diff(pos_traj, axis=0)/dt
    ax_vel_x.plot(vel_t_array, vel_vec[:, 0], 'b')
    ax_vel_y.plot(vel_t_array, vel_vec[:, 1], 'b')
    ax_ang.plot(vel_t_array, phi_array, 'b')

    ax_vel_x.hold(True)
    ax_vel_y.hold(True)
    ax_ang.hold(True)
    ax_pos.hold(True)
    ax_vel.hold(True)
    #show
    rec_traj, rec_vel_vec = rxzero_traj_eval(parms, t_array, 0, 0)
    rec_vel_amp = np.sum(rec_vel_vec**2, axis=1)**(1./2)
    rec_phi_array = get_continuous_ang(rec_traj)
    if reg_pnts is not None:
        ax_pos.plot(rec_traj[:, 0] + pos_traj[reg_pnts[0], 0], -rec_traj[:, 1] - pos_traj[reg_pnts[0], 1], 'g')
        #show start/end point if reg_pnts are there
        ax_pos.plot(pos_traj[reg_pnts[0], 0], -pos_traj[reg_pnts[0], 1], 'ro')
        ax_pos.plot(pos_traj[reg_pnts[4], 0], -pos_traj[reg_pnts[4], 1], 'wo')
        ax_vel.plot(t_array, rec_vel_amp, 'r')
    else:
        ax_pos.plot(rec_traj[:, 0] + pos_traj[0, 0], -rec_traj[:, 1] - pos_traj[0, 1], 'g')
        ax_vel.plot(t_array, rec_vel_amp, 'r')

    #for each parms
    for parm in parms:
        #reconstruct
        tmp_rec_traj, _ = rxzero_traj_eval([parm], t_array, 0, 0)
        tmp_rec_vel_profile = get_vel_profile(tmp_rec_traj)/dt
        ax_vel.plot(vel_t_array, tmp_rec_vel_profile, 'g')
        #phi
        tmp_phi_array = rxzero_normal_Phi_eval(parm, t_array)
        ax_ang.plot(t_array, tmp_phi_array, 'g')
    #show original trajectory from vel_vec_n
    if vel_vec_n is not None and reg_pnts is not None:
        pos_n = np.cumsum(vel_vec_n[reg_pnts[0]:reg_pnts[4], :], axis=0)*dt + pos_traj[reg_pnts[0], :]
        ax_pos.plot(pos_n[:, 0], -pos_n[:, 1], 'r')

    colors = ['r', 'y', 'k', 'g', 'w']
    if reg_pnts is not None:
        for reg_pnt_idx in range(5):
                ax_vel.plot(reg_pnts[reg_pnt_idx]*dt,
                    vel_profile[reg_pnts[reg_pnt_idx]],
                    colors[reg_pnt_idx]+'o')
                ax_vel_x.plot(  reg_pnts[reg_pnt_idx]*dt,
                                vel_vec[reg_pnts[reg_pnt_idx], 0],
                                colors[reg_pnt_idx]+'o')
                ax_vel_y.plot(  reg_pnts[reg_pnt_idx]*dt,
                                vel_vec[reg_pnts[reg_pnt_idx], 1],
                                colors[reg_pnt_idx]+'o')
                ax_ang.plot(    reg_pnts[reg_pnt_idx]*dt,
                                phi_array[reg_pnts[reg_pnt_idx]],
                                colors[reg_pnt_idx]+'o')


    ax_vel_x.plot(t_array, rec_vel_vec[:, 0], 'r')
    ax_vel_y.plot(t_array, rec_vel_vec[:, 1], 'r')
    ax_ang.plot(vel_t_array, rec_phi_array, 'r')

    ax_pos.set_title('Letter Profile')
    ax_vel.set_title('Velocity magnitude')
    ax_vel_x.set_title('Velocity X')
    ax_vel_y.set_title('Velocity Y')
    ax_ang.set_title('Angular Position')

    ax_pos.set_xlabel('X Coord')
    ax_pos.set_ylabel('Y Coord')
    ax_vel.set_xlabel('Time (sec)')
    ax_vel.set_ylabel('Velocity')
    ax_vel_x.set_xlabel('Time (sec)')
    ax_vel_x.set_ylabel('Velocity')
    ax_vel_y.set_xlabel('Time (sec)')
    ax_vel_y.set_ylabel('Velocity')
    ax_ang.set_xlabel('Time (sec)')
    ax_ang.set_ylabel('Angular Pos (rad)')

    plt.tight_layout()
    plt.draw()
    ax_pos.hold(False)
    ax_vel.hold(False)
    ax_vel_x.hold(False)
    ax_vel_y.hold(False)
    ax_ang.hold(False)
    return (fig, ax_pos, ax_vel, ax_vel_x, ax_vel_y, ax_ang)

def rxzero_train_test(pos_traj):
    """
    a test function to check combined routines...
    pos_traj: letter trajectory to train (for one section of pen up & down)
    """
    #original position & vel profile...
    plt.ion()
    dt = 0.01
    t_array = np.arange(len(pos_traj))*dt
    vel_profile = get_vel_profile(pos_traj)/dt
    vel_t_array = np.arange(len(vel_profile))*dt

    #train
    parms = rxzero_train(pos_traj, verbose=True)
    #show it
    fig, ax_pos, ax_vel, ax_vel_x, ax_vel_y, ax_ang = rxzero_plot_helper(pos_traj, parms)

    return

def rxzero_train_batch_test(pos_traj_lst):
    """
    train a batch of position trajectories, a set of one-stroke letters...
    """
    parms_list = []
    i = 0
    for pos_traj in pos_traj_lst:
        parms_list.append(rxzero_train(pos_traj[0]))
        print('Finished extracting letter with index:' + str(i))
        i+=1

    return parms_list

def rxzero_stat_test(pos_traj_lst, parms_lst, dt=0.01):
    """
    do some statistics for extracted parms...
    """
    stat = dict()
    stat['nStrokes'] = []
    stat['SNR'] = []
    stat['grouped_parms'] = dict()

    i = 0
    for parms in parms_lst:
        vel_vec = np.diff(pos_traj_lst[i][0], axis=0)/dt
        tmp_SNR = rxzero_calc_SNR_full(vel_vec, parms)
        if tmp_SNR > 25:
            #good extraction, take it
            stat['nStrokes'].append(len(parms))
            stat['SNR'].append(tmp_SNR)
            #create new one if not exists
            if not str(len(parms)) in stat['grouped_parms']:
                stat['grouped_parms'][str(len(parms))] = []
                stat['grouped_parms'][str(len(parms))].append(parms)
            else:
                stat['grouped_parms'][str(len(parms))].append(parms)
        i+=1

    #show something...
    fig_nstrokes = plt.figure()
    #number of strokes
    ax_nstrokes = fig_nstrokes.add_subplot(111)

    n, bins, patches = ax_nstrokes.hist(np.array(stat['nStrokes']), 20, normed=0, facecolor='green', alpha=0.75)
    ax_nstrokes.set_xlabel('Num of strokes')
    ax_nstrokes.set_ylabel('Occurrence')
    ax_nstrokes.set_title('Distribution of Number of Strokes')

    #for a specific number of stroke, see distribution of the parameters D and sig
    fig_ddist = plt.figure()
    ax_ddist = fig_ddist.add_subplot(111)
    nCheckStroke = 5
    #print np.array(stat['grouped_parms'][str(nCheckStroke)]).shape
    D_parm = np.array(stat['grouped_parms'][str(nCheckStroke)])[:, 0, 0]
    n, bins, patches = ax_ddist.hist(D_parm, 20, normed=1, facecolor='green', alpha=0.75)
    ax_ddist.set_xlabel('Values')
    ax_ddist.set_ylabel('Ratio')
    ax_ddist.set_title('Histogram for D1 of {0}-stroke Features'.format(nCheckStroke))

    #
    fig_sigdist = plt.figure()
    ax_sigdist = fig_sigdist.add_subplot(111)
    sig_parm = np.array(stat['grouped_parms'][str(nCheckStroke)])[:, 1, 3]
    n, bins, patches = ax_sigdist.hist(sig_parm, 20, normed=1, facecolor='green', alpha=0.75)
    ax_sigdist.set_xlabel('Values')
    ax_sigdist.set_ylabel('Ratio')
    ax_sigdist.set_title('Histogram for Sig2 of {0}-stroke Features'.format(nCheckStroke))

    #synthesize a letter
    avg_idx = 0
    avg_parms = [np.mean(np.array(stat['grouped_parms'][str(nCheckStroke)])[20:50, i, :], axis=0) for i in range(nCheckStroke)]
    #just select one as base
    avg_parms = np.array(stat['grouped_parms'][str(nCheckStroke)])[avg_idx, :, :]
    std_parms = [np.std(np.array(stat['grouped_parms'][str(nCheckStroke)])[20:50, i, :], axis=0) for i in range(nCheckStroke)]

    #apply noise
    avg_parms[:, 0] = avg_parms[:, 0] + np.random.randn(nCheckStroke)*(np.array(std_parms)[:, 0])*0.0
    #use avg_parms to draw a letter
    t_array = np.arange(0, 2.5, 0.01)

    avg_traj, _ = rxzero_traj_eval(avg_parms, t_array, 0, 0)

    #print avg_traj
    fig_prf = plt.figure()
    ax_prf = fig_prf.add_subplot(111)
    ax_prf.plot(avg_traj[:, 0]+pos_traj_lst[avg_idx][0][0, 0], -avg_traj[:, 1]-pos_traj_lst[avg_idx][0][0, 1], linewidth=2.0)
    ax_prf.set_xlabel('X Coord')
    ax_prf.set_ylabel('Y Coord')
    ax_prf.set_title('Synthesis of Letter Profile')
    #fig, ax_pos, ax_vel, ax_vel_x, ax_vel_y, ax_ang = rxzero_plot_helper(pos_traj_lst[4][0], avg_parms)
    ax_prf.hold(True)

    ax_prf.plot(pos_traj_lst[avg_idx][0][:, 0], -pos_traj_lst[avg_idx][0][:, 1], 'r')
    plt.show()

    return stat

def fit_parm_scale_ang_component_with_global_optimization(pos_traj, parms, free_comp_idx, dt=0.01, maxIters=1):
    #this is a more restrictive one compared with the standard global optimization routine
    #we just want to try what if only scale and angular positions are allowed to be optimized...
    #similarly, first go for scale then angular position and then all of them
    t_array = np.arange(len(pos_traj))*dt
    t_vel_array = np.arange(len(pos_traj)-1)*dt
    vel_vec_traj = np.diff(pos_traj, axis=0) / dt
    def obj_func_step_1(x, *args):
        curr_parms, vel_traj, free_comp_idx = args
        #construct parameters
        vel_parms = np.reshape(x, (-1, 1))
        # conc_parms = np.concatenate([vel_parms, theta_parms], axis=1)
        conc_parms = np.array(curr_parms, copy=True)
        conc_parms[free_comp_idx, 0] = vel_parms
        _, eval_vel = rxzero_traj_eval(conc_parms, t_vel_array, 0, 0)
        return np.sum(np.sum(np.abs((eval_vel - vel_traj)), axis=1))

    def obj_func_step_2(x, *args):
        curr_parms, pos_traj, free_comp_idx = args
        theta_parms = np.reshape(x, (-1, 2))
        conc_parms = np.array(curr_parms, copy=True)

        conc_parms[free_comp_idx, [4, 5]] = theta_parms
        eval_pos, _ = rxzero_traj_eval(conc_parms, t_array, pos_traj[0, 0], pos_traj[0, 1])
        return np.sum(np.sum(np.abs((eval_pos - pos_traj)), axis=1))

    def obj_func_step_3(x, *args):
        curr_parms, vel_traj, free_comp_idx = args
        full_parms = np.reshape(x, (-1, 3))
        conc_parms = np.array(curr_parms, copy=True)
        conc_parms[free_comp_idx, 0]    = full_parms[:, [0]]
        conc_parms[free_comp_idx, [4, 5]]  = full_parms[:, [1, 2]]
        _, eval_vel = rxzero_traj_eval(conc_parms, t_vel_array, 0, 0)
        return np.sum(np.sum(np.abs((eval_vel - vel_traj)), axis=1))

    curr_parms = np.array(parms)
    #<hyin/Feb-20th-2015> fixed a bug that leads to no solution
    #actually, skipping step 1 and step 3 seems okay for sake of speed (buggy version)
    #better to add an option to choose precise verions/approximate version
    #or derive closed form gradient/do offline training on a workstation...
    for i in range(maxIters):
        #step 1
        x_init_1 = curr_parms[free_comp_idx, 0].flatten()
        #construct bounds...
        bounds_step_1 = []
        for parm in curr_parms[free_comp_idx, :]:
            parm_bounds = [ (0.5*parm[0][0], 3.5*parm[0][0]) #D
                            ]
            bounds_step_1 = bounds_step_1 + parm_bounds
        opt_res = sciopt.minimize(obj_func_step_1, x_init_1, args=(curr_parms, vel_vec_traj, free_comp_idx), bounds=bounds_step_1)
        curr_parms[free_comp_idx, 0] = np.reshape(opt_res.x, (-1, 1))

        #step 2
        x_init_2 = curr_parms[free_comp_idx, 4:6].flatten()
        bounds_step_2 = []
        for parm in curr_parms[free_comp_idx, :]:
            parm_bounds = [ (-np.pi, np.pi),               #theta_s
                            (-np.pi, np.pi)                #theta_e
                            ]
            bounds_step_2 = bounds_step_2 + parm_bounds
        opt_res = sciopt.minimize(obj_func_step_2, x_init_2, args=(curr_parms, pos_traj, free_comp_idx), bounds=bounds_step_2)
        curr_parms[free_comp_idx, [4, 5]] = np.reshape(opt_res.x, (-1, 2))

        #step 3
        x_init_3 = curr_parms[free_comp_idx, [0, 4, 5]].flatten()
        #construct bounds...
        bounds_step_3 = []
        for parm in curr_parms[free_comp_idx, :]:
            parm_bounds = [ (0.5*parm[0][0], 3.5*parm[0][0]), #D
                            (-np.pi, np.pi),               #theta_s
                            (-np.pi, np.pi)                #theta_e
                            ]
            bounds_step_3 = bounds_step_3 + parm_bounds

        opt_res = sciopt.minimize(obj_func_step_3, x_init_3, args=(curr_parms, vel_vec_traj, free_comp_idx), bounds=bounds_step_3)
        curr_parms[free_comp_idx, 0] = np.reshape(opt_res.x, (-1, 3))[:, [0]]
        curr_parms[free_comp_idx, [4, 5]] = np.reshape(opt_res.x, (-1, 3))[:, [1,2]]

    #evaluate position reconstruction error
    eval_pos, eval_vel = rxzero_traj_eval(curr_parms, t_array, pos_traj[0, 0], pos_traj[0, 1])
    # recons_err = np.sum(np.sum((eval_pos - pos_traj)**2, axis=1))
    #evaluate reconstruction error from velocity profile?
    recons_err = np.sum(np.sum((eval_vel[:-1, :] - vel_vec_traj)**2, axis=1))

    return curr_parms, recons_err