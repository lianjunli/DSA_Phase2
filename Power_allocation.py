import numpy as np
import cvxopt as cvx
import copy
from HelperFunc_MCS import *

def PA_GP_MCS_minRate(channel_alloc, channel_gain, B, noise_vec, priority, SU_power, minRate, SNR_gap, QAM_cap, objective_list):

    n_su = channel_alloc.shape[0]
    n_channel = channel_alloc.shape[1]

    channel_gain = copy.deepcopy(channel_gain)
    Noise = copy.deepcopy(noise_vec)
    channel_gain = channel_gain * (10 ** 8)
    Noise = Noise * (10 ** 8)

    # non-zero power list
    channel_alloc_total = np.sum(channel_alloc, dtype=np.int32)

    def A(m, n, j, t):
        if (t == n):
            return -1
        else:
            if (t == j):
                return 1
            else:
                return 0

    def C(m, n, j):
        if (j == n):
            return (Noise[n]) / (channel_gain[n, n] / SNR_gap[n])
        else:
            return channel_gain[n, j] / (channel_gain[n, n] / SNR_gap[n])

    # Objective function
    F_obj = np.zeros(channel_alloc_total * 2)
    i = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                F_obj[channel_alloc_total + i] = priority[n] * B / (10 ** 6)
                i = i + 1

    g_obj = np.zeros(1)

    # Constraints of slack variables (m => n => j)
    n_F_slack = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                for j in range(n_su):
                    if (channel_alloc[j, m] == 1):
                        n_F_slack = n_F_slack + 1

    F_slack = np.zeros((n_F_slack, channel_alloc_total * 2))

    index0 = 0
    index1 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                for j in range(n_su):
                    if (channel_alloc[j, m] == 1):
                        F_slack[index0, channel_alloc_total + index1] = -1

                        index3 = 0
                        for t in range(n_su):
                            if (channel_alloc[t, m] == 1):
                                F_slack[index0, int(np.sum(channel_alloc[:, 0:m])) + index3] = A(m, n, j, t)
                                index3 = index3 + 1
                        index0 = index0 + 1
                index1 = index1 + 1

    g_slack = np.zeros(n_F_slack)
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                for j in range(n_su):
                    if (channel_alloc[j, m] == 1):
                        g_slack[index0] = np.log(C(m, n, j))
                        index0 = index0 + 1

    # Constraints of maximum power
    F_power = np.zeros((channel_alloc_total, channel_alloc_total * 2))
    index0 = 0
    for n in range(n_su):
        for m in range(n_channel):
            if (channel_alloc[n, m] == 1):
                F_power[index0, int(np.sum(channel_alloc[:, 0:m])) + int(np.sum(channel_alloc[:(n+1), m])) - 1] = 1
                index0 = index0 + 1

    g_power = np.zeros(channel_alloc_total)
    index0 = 0
    for n in range(n_su):
        for m in range(n_channel):
            if (channel_alloc[n, m] == 1):
                g_power[index0] = np.log(1/SU_power[n])
                index0 = index0 + 1

    # Constraints of minimum data rate constraint
    F_minRate = np.zeros((n_su, channel_alloc_total * 2))
    index0 = 0
    for n in range(n_su):
        for m in range(n_channel):
            if (channel_alloc[n, m] == 1):
                F_minRate[n, channel_alloc_total + index0] = B / (10 ** 6)
                index0 = index0 + 1

    g_minRate = np.zeros(n_su)
    for n in range(n_su):
        g_minRate[n] = np.log(2 ** (minRate[n] / (10 ** 6)))

    # Constraints of QAM capability
    F_QAM = np.zeros((channel_alloc_total, channel_alloc_total * 2))
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                F_QAM[index0, channel_alloc_total + index0] = 1
                index0 = index0 + 1

    g_QAM = np.zeros(channel_alloc_total)
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                g_QAM[index0] = np.log(2 ** -(QAM_cap[n] + 1))
                index0 = index0 + 1


    K_obj = np.ones(1)

    K_slack = n_su * np.ones(channel_alloc_total)
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                K_slack[index0] = np.sum(channel_alloc[:, m])
                index0 = index0 + 1

    K_power = n_channel * np.ones(n_su)
    for n in range(n_su):
        K_power[n] = np.sum(channel_alloc[n, :])

    K_minRate = np.ones(n_su)

    K_QAM = np.ones(channel_alloc_total)

    K_arr = np.concatenate((K_obj, K_slack, K_power, K_minRate, K_QAM))
    K_arr = K_arr.astype(int)

    F_arr = np.vstack((F_obj, F_slack, F_power, F_minRate, F_QAM))

    g_arr = np.concatenate((g_obj, g_slack, g_power, g_minRate, g_QAM))

    K = K_arr.tolist()
    F_arr = F_arr.T
    F = cvx.matrix(F_arr.tolist())
    g_arr = g_arr.T
    g = cvx.matrix(g_arr.tolist())

    cvx.solvers.options['show_progress'] = False

    sol = cvx.exp(cvx.solvers.gp(K, F, g, quiet=True)['x'])

    power_alloc = np.zeros((n_su, n_channel))
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                power_alloc[n, m] = sol[index0]
                index0 = index0 + 1

    objective_list['total'].append(objective_value(channel_alloc, power_alloc, priority, channel_gain / (10 ** 8), B, noise_vec, SNR_gap))


    return power_alloc
