import numpy as np
import cvxopt as cvx
import copy
from HelperFunc_MCS import *


def PA_IWF(power_alloc, channel_alloc, channel_gain, env, priority, SU_power, objective_list, update_order):

    n_su = power_alloc.shape[0]

    while (True):
        previous_power_alloc = copy.deepcopy(power_alloc)

        for n in update_order:
            power_index = np.where(channel_alloc[n, :] == 1)
            power_index = power_index[0]
            power_dim = power_index.size

            A = np.zeros(power_dim)
            B = np.zeros(power_dim)
            for m in range(power_index.size):
                A[m] = priority[n] * channel_alloc[n, power_index[m]] * env.B / np.log(2)

                for j in range(n_su):
                    if (j != n):
                        B[m] = B[m] + channel_alloc[j, power_index[m]] * channel_gain[n, j] * power_alloc[j, power_index[m]]
                B[m] = B[m] + env.B * env.Noise
                B[m] = B[m] / channel_gain[n, n]

            index_sort = np.argsort(B/A)
            for k in range(index_sort.size):
                water_level = np.sum(A[index_sort[:k+1]]) / (SU_power[n] + np.sum(B[index_sort[:k+1]]))

                if (k == index_sort.size-1 and water_level <= A[index_sort[k]] / B[index_sort[k]]):
                    break

                elif (water_level <= A[index_sort[k]] / B[index_sort[k]] and
                    water_level > A[index_sort[k+1]] / B[index_sort[k+1]]):
                    break

            k = 0
            for m in power_index:
                power_alloc[n, m] = max(0, A[k]/water_level - B[k])
                k = k + 1

            # Record the objective value
            objective_list['total'].append(objective_value(channel_alloc, power_alloc, priority, channel_gain, env))

        if (np.amax(np.absolute(power_alloc - previous_power_alloc)) < 1):
            print('power allocation is updated')
            break

    return power_alloc


def PA_GP(channel_alloc, channel_gain, env, priority, SU_power, objective_list):

    n_su = channel_alloc.shape[0]
    n_channel = channel_alloc.shape[1]

    channel_gain = copy.deepcopy(channel_gain)
    Noise = copy.deepcopy(env.Noise)
    channel_gain = channel_gain * (10 ** 8)
    Noise = Noise * (10 ** 8)


    L = np.ones(n_channel)
    for m in range(n_channel):
        for j in range(n_su):
            L[m] = L[m] + channel_alloc[j, m]

    def A(m, n, j, t):
        if (j == n):
            return -channel_alloc[t, m] / L[m]
        else:
            if (t == j):
                return 1 - channel_alloc[t, m] / L[m]
            else:
                return -channel_alloc[t, m] / L[m]

    def C(m, n, j):
        if (j == n):
            tmp = ((env.B * Noise) ** (1 - 1/L[m])) / L[m]
            for t in range(n_su):
                tmp = tmp * ((channel_gain[n, t]) ** (- channel_alloc[t, m] / L[m]))
            return tmp
        else:
            tmp = ((env.B * Noise) ** (- 1 / L[m])) / L[m] * channel_alloc[j, m] * channel_gain[n, j]
            for t in range(n_su):
                tmp = tmp * ((channel_gain[n, t]) ** (- channel_alloc[t, m] / L[m]))
            return tmp


    # non-zero power list
    channel_alloc_total = np.sum(channel_alloc, dtype=np.int32)

    # Objective function
    F_obj = np.zeros(channel_alloc_total * 2)
    i = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                F_obj[channel_alloc_total + i] = priority[n] * env.B / (10 ** 6)
                i = i + 1

    g_obj = np.zeros(1)

    # Constraints of slack variables (m => n => j)
    F_slack = np.zeros((np.sum(np.sum(channel_alloc, axis=0)**2, dtype=np.int32), channel_alloc_total * 2))

    index0 = 0
    index1 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                for j in range(n_su):
                    if (channel_alloc[j, m] == 1):
                        F_slack[index0, channel_alloc_total + index1] = -1
                        index2 = 0
                        for t in range(n_su):
                            if (channel_alloc[t, m] == 1):
                                F_slack[index0, np.sum(channel_alloc[:, :m], dtype=np.int32) + index2] = A(m, n, j, t)
                                index2 = index2 + 1
                        index0 = index0 + 1
                index1 = index1 + 1


    g_slack = np.zeros(np.sum(np.sum(channel_alloc, axis=0)**2, dtype=np.int32))
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
            if (channel_alloc[n, m]==1):
                F_power[index0, np.sum(channel_alloc[:, :m], dtype=np.int32) + np.sum(channel_alloc[:n, m], dtype=np.int32)] = 1
                index0 = index0 + 1

    g_power = np.zeros(channel_alloc_total)
    index0 = 0
    for n in range(n_su):
        for m in range(n_channel):
            if (channel_alloc[n, m] == 1):
                g_power[index0] = np.log(1/SU_power[n])
                index0 = index0 + 1


    K_obj = np.array([1])

    K_slack = np.zeros(channel_alloc_total)
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                K_slack[index0] = np.sum(channel_alloc[:, m])
                index0 = index0 + 1

    su_alloc_total = np.sum((np.sum(channel_alloc, axis=1))>0)
    K_power = np.zeros(su_alloc_total)
    index0 = 0
    for n in range(n_su):
        if (np.sum(channel_alloc[n, :]) > 0):
            K_power[index0] = np.sum(channel_alloc[n, :])
            index0 = index0 + 1

    K_arr = np.concatenate((K_obj, K_slack, K_power))
    K_arr = K_arr.astype(int)

    F_arr = np.vstack((F_obj, F_slack, F_power))

    g_arr = np.concatenate((g_obj, g_slack, g_power))

    K = K_arr.tolist()
    F_arr = F_arr.T
    F = cvx.matrix(F_arr.tolist())
    g_arr = g_arr.T
    g = cvx.matrix(g_arr.tolist())

    sol = cvx.exp(cvx.solvers.gp(K, F, g)['x'])

    power_alloc = np.zeros((n_su, n_channel))
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                power_alloc[n, m] = sol[index0]
                index0 = index0 + 1

    objective_list['total'].append(objective_value(channel_alloc, power_alloc, priority, channel_gain / (10 ** 8), env))


    return power_alloc


def PA_IWF_minRate(power_alloc, channel_alloc, channel_gain, env, priority, SU_power, minRate, SNR_gap, objective_list, update_order):

    n_su = power_alloc.shape[0]

    SU_max_power = copy.deepcopy(1*SU_power)

    while (True):

        while (True):
            previous_power_alloc = copy.deepcopy(power_alloc)

            for n in update_order:
                power_index = np.where(channel_alloc[n, :] == 1)
                power_index = power_index[0]
                power_dim = power_index.size

                A = np.zeros(power_dim)
                B = np.zeros(power_dim)
                for m in range(power_index.size):
                    A[m] = priority[n] * channel_alloc[n, power_index[m]] * env.B / np.log(2)

                    for j in range(n_su):
                        if (j != n):
                            B[m] = B[m] + channel_alloc[j, power_index[m]] * channel_gain[n, j] * power_alloc[j, power_index[m]]
                    B[m] = B[m] + env.B * env.Noise
                    B[m] = B[m] / (channel_gain[n, n] / SNR_gap[n])

                index_sort = np.argsort(B/A)
                for k in range(index_sort.size):
                    water_level = np.sum(A[index_sort[:k+1]]) / (SU_max_power[n] + np.sum(B[index_sort[:k+1]]))

                    if (k == index_sort.size-1 and water_level <= A[index_sort[k]] / B[index_sort[k]]):
                        break

                    elif (water_level <= A[index_sort[k]] / B[index_sort[k]] and
                        water_level > A[index_sort[k+1]] / B[index_sort[k+1]]):
                        break

                k = 0
                for m in power_index:
                    power_alloc[n, m] = max(0, A[k]/water_level - B[k])
                    k = k + 1

                # Record the objective value
                objective_list['total'].append(objective_value(channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap))

            if (np.amax(np.absolute(power_alloc - previous_power_alloc)) < 1):
                break

        # Adjust the power to satisfy the minimum data rate
        capacity = np.zeros(n_su)
        for n in update_order:
            capacity[n] = capacity_SU(power_alloc[n, :], n, channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap)

            if (capacity[n] * (10 ** 6) < minRate[n]):
                SU_max_power[n] = min(SU_power[n], SU_max_power[n] + 2)

            elif (capacity[n] * (10 ** 6) > 1.2*minRate[n]):
                SU_max_power[n] = max(0, SU_max_power[n] - 2)

        # Check whether every user satisfy the minimum data rate
        count = 0
        for n in update_order:
            if (capacity[n] * (10 ** 6) > minRate[n]):
                count = count + 1
        if (count == n_su):
            return power_alloc


def PA_GP_minRate(channel_alloc, channel_gain, env, priority, SU_power, minRate, objective_list):

    n_su = channel_alloc.shape[0]
    n_channel = channel_alloc.shape[1]

    channel_gain = copy.deepcopy(channel_gain)
    Noise = copy.deepcopy(env.Noise)
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
            return (env.B * Noise) / channel_gain[n, n]
        else:
            return channel_gain[n, j] / channel_gain[n, n]

    # Objective function
    F_obj = np.zeros(channel_alloc_total * 2)
    i = 0
    for m in range(n_channel):
        for n in range(n_su):
            F_obj[channel_alloc_total + i] = priority[n] * env.B / (10 ** 6)
            i = i + 1

    g_obj = np.zeros(1)

    # Constraints of slack variables (m => n => j)
    F_slack = np.zeros((n_su * channel_alloc_total, channel_alloc_total * 2))

    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            for j in range(n_su):
                F_slack[index0, channel_alloc_total + n_su * m + n] = -1

                for t in range(n_su):
                    F_slack[index0, n_su*m + t] = A(m, n, j, t)
                index0 = index0 + 1

    g_slack = np.zeros(n_su * channel_alloc_total)
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            for j in range(n_su):
                g_slack[index0] = np.log(C(m, n, j))
                index0 = index0 + 1

    # Constraints of maximum power
    F_power = np.zeros((channel_alloc_total, channel_alloc_total * 2))
    index0 = 0
    for n in range(n_su):
        for m in range(n_channel):
            F_power[index0, n_su*m + n] = 1
            index0 = index0 + 1

    g_power = np.zeros(channel_alloc_total)
    index0 = 0
    for n in range(n_su):
        for m in range(n_channel):
            g_power[index0] = np.log(1/SU_power[n])
            index0 = index0 + 1

    # Constraints of minimum data rate constraint
    F_minRate = np.zeros((n_su, channel_alloc_total * 2))
    for n in range(n_su):
        for m in range(n_channel):
            F_minRate[n, channel_alloc_total + n_su*m + n] = env.B / (10 ** 6)

    g_minRate = np.zeros(n_su)
    for n in range(n_su):
        g_minRate[n] = np.log(2 ** (minRate[n] / (10 ** 6)))


    K_obj = np.ones(1)

    K_slack = n_su * np.ones(channel_alloc_total)

    K_power = n_channel * np.ones(n_su)

    K_minRate = np.ones(n_su)

    K_arr = np.concatenate((K_obj, K_slack, K_power, K_minRate))
    #K_arr = np.concatenate((K_obj, K_slack, K_power))
    K_arr = K_arr.astype(int)

    F_arr = np.vstack((F_obj, F_slack, F_power, F_minRate))
    #F_arr = np.vstack((F_obj, F_slack, F_power))

    g_arr = np.concatenate((g_obj, g_slack, g_power, g_minRate))
    #g_arr = np.concatenate((g_obj, g_slack, g_power))

    K = K_arr.tolist()
    F_arr = F_arr.T
    F = cvx.matrix(F_arr.tolist())
    g_arr = g_arr.T
    g = cvx.matrix(g_arr.tolist())

    sol = cvx.exp(cvx.solvers.gp(K, F, g)['x'])

    power_alloc = np.zeros((n_su, n_channel))
    index0 = 0
    for m in range(n_channel):
        for n in range(n_su):
            power_alloc[n, m] = sol[index0]
            index0 = index0 + 1

    objective_list['total'].append(objective_value(channel_alloc, power_alloc, priority, channel_gain / (10 ** 8), env))


    return power_alloc


def PA_GP_MCS_minRate(channel_alloc, channel_gain, env, priority, SU_power, minRate, SNR_gap, QAM_cap, objective_list):

    n_su = channel_alloc.shape[0]
    n_channel = channel_alloc.shape[1]

    channel_gain = copy.deepcopy(channel_gain)
    Noise = copy.deepcopy(env.Noise)
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
            return (env.B * Noise) / (channel_gain[n, n] / SNR_gap[n])
        else:
            return channel_gain[n, j] / (channel_gain[n, n] / SNR_gap[n])

    # Objective function
    F_obj = np.zeros(channel_alloc_total * 2)
    i = 0
    for m in range(n_channel):
        for n in range(n_su):
            if (channel_alloc[n, m] == 1):
                F_obj[channel_alloc_total + i] = priority[n] * env.B / (10 ** 6)
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
                F_minRate[n, channel_alloc_total + index0] = env.B / (10 ** 6)
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

    objective_list['total'].append(objective_value(channel_alloc, power_alloc, priority, channel_gain / (10 ** 8), env, SNR_gap))


    return power_alloc
