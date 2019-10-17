import numpy as np
from scipy.stats import norm
import copy

'''
For multiple users without cluster share multiple channels
'''

def channel_capacity_print(channel_alloc, power_alloc, channel_gain, QoS, env, SNR_gap):
    # print channel capacity of each SU
    n_su = channel_alloc.shape[0]
    n_channel = channel_alloc.shape[1]
    for n in range(n_su):
        capacity_sum = 0
        for m in range(n_channel):
            if (channel_alloc[n, m] == 1):
                inter_sum = 0
                for j in range(n_su):
                    if (j != n):
                        inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
                capacity_sum = capacity_sum + env.B * np.log2(
                    1 + channel_gain[n, n] / SNR_gap[n] * power_alloc[n, m] / (inter_sum + env.B * env.Noise))
        print('SU %d:' % n)
        print('channel capacity: achieved = %.2f, required = %.2f, difference = %.2f' % (capacity_sum, QoS[n], capacity_sum-QoS[n]))

def objective_print(channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap):
    # print objective function
    n_su = channel_alloc.shape[0]
    n_channel = channel_alloc.shape[1]
    capacity_sum = 0
    for n in range(n_su):
        for m in range(n_channel):
            if (channel_alloc[n, m] == 1):
                inter_sum = 0
                for j in range(n_su):
                    if (j != n):
                        inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
                capacity_sum = capacity_sum + \
                               priority[n] * env.B \
                               * np.log2(1 + channel_gain[n, n] / SNR_gap[n] * power_alloc[n, m] / (inter_sum + env.B * env.Noise))
    print('objective function = %.2f' % (capacity_sum))

def objective_value(channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap):
    # get objective function
    capacity_sum = 0
    n_channel = power_alloc.shape[1]
    n_su = power_alloc.shape[0]
    for n in range(n_su):
        for m in range(n_channel):
            if (channel_alloc[n, m] == 1):
                inter_sum = 0
                for j in range(n_su):
                    if (j != n):
                        inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
                capacity_sum = capacity_sum + \
                               priority[n] * env.B/(10**6) * \
                               np.log2(1 + channel_gain[n, n] / SNR_gap[n] * power_alloc[n, m] / (inter_sum + env.NoisePower[n]))
    return capacity_sum


def objective_value_SU(p, SU_index, channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap):
    # get objective function of a specific user
    n_channel = power_alloc.shape[1]
    n_su = power_alloc.shape[0]
    B = env.B
    # Noise = env.Noise
    n = SU_index

    capacity_sum = 0
    for m in range(n_channel):
        if (channel_alloc[n, m] == 1):
            inter_sum = 0
            for j in range(n_su):
                if (j != n):
                    inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
            capacity_sum = capacity_sum + priority[n] * B / (10 ** 6) * \
                           np.log2(1 + channel_gain[n, n] / SNR_gap[n] * p[m] / (inter_sum + env.NoisePower[n]))

    for k in range(n_su):
        if (k != n):
            for m in range(n_channel):
                if (channel_alloc[k, m] == 1):
                    inter_sum = 0
                    for j in range(n_su):
                        if (j != n and j != k):
                            inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[k, j] * power_alloc[j, m]
                    capacity_sum = capacity_sum + priority[k] * B / (10 ** 6) * \
                                   np.log2(1 + channel_gain[k, k] / SNR_gap[k] * power_alloc[k, m]
                                   / (inter_sum + env.NoisePower[k] + channel_alloc[n, m] * channel_gain[k, n] * p[m]))
    return capacity_sum

def capacity_SU(p, SU_index, channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap):
    # get channel capacity of a specific user
    n_channel = power_alloc.shape[1]
    n_su = power_alloc.shape[0]
    B = env.B
    # Noise = env.Noise
    n = SU_index

    capacity_sum = 0
    for m in range(n_channel):
        if (channel_alloc[n, m] == 1):
            inter_sum = 0
            for j in range(n_su):
                if (j != n):
                    inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
            capacity_sum = capacity_sum + priority[n] * B / (10 ** 6) * \
                           np.log2(1 + channel_gain[n, n] / SNR_gap[n] * p[m] / (inter_sum + env.NoisePower[n]))
    return capacity_sum

def capacity_SU_channel(p, SU_index, channel_index, channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap):
    # get channel capacity of a specific user on a specfic channel
    n_channel = power_alloc.shape[1]
    n_su = power_alloc.shape[0]
    B = env.B
    # Noise = env.Noise
    n = SU_index
    m = channel_index

    capacity_sum = 0

    if (channel_alloc[n, m] == 1):
        inter_sum = 0
        for j in range(n_su):
            if (j != n):
                inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
        capacity_sum = capacity_sum + priority[n] * B / (10 ** 6) * \
                       np.log2(1 + channel_gain[n, n] / SNR_gap[n] * p[m] / (inter_sum + env.NoisePower[n]))
    return capacity_sum

def rate_SU(group, tx, rx, power_alloc, channel_gain, priority, env, SNR_gap):
    # get rate for (group, tx, rx)
    n_su_group = env.n_user_cluster
    n_group = int(channel_gain.shape[0] / n_su_group)

    def A_fun(group, rx, power_alloc, channel_gain, env):

        rx_index = group * n_su_group + rx

        sum = 0
        for g in range(n_group):
            if (g == group):
                continue
            for i in range(n_su_group):
                inter_tx_index = g * n_su_group + i
                sum = sum + power_alloc[g, 0] * channel_gain[rx_index, inter_tx_index]
        sum = sum / n_su_group
        sum = sum + env.NoisePower[group]

        return sum

    tx_index = group * n_su_group + tx
    rx_index = group * n_su_group + rx

    A = np.zeros(n_su_group - 1)
    for j in range(n_su_group - 1):
        rx = (tx + j + 1) % n_su_group
        A[j] = A_fun(group, rx, power_alloc, channel_gain, env)

    data_rate = priority[group] * env.B / (10 ** 6) \
                * np.log2(1 + power_alloc[group, 0] * channel_gain[rx_index, tx_index] / SNR_gap[group] /
                          A_fun(group, rx, power_alloc, channel_gain, env))

    return data_rate

def cal_CP_simulate(cluster, channel_gain, power_alloc, SNR_gap, sigma, R_min, env):
    # Calculate the simulated coverage probability

    n_cluster = power_alloc.shape[0]
    n_user_cluster = env.n_user_cluster
    N_iteration = 200 # 500
    if (sigma == 0):
        N_iteration = 1

    Total_number_of_test = 0
    Network_capacity_sum = 0
    Success_count = 0


    for iter in range(N_iteration):
        for tx in range(n_user_cluster):
            for rx in range(n_user_cluster):
                if rx == tx: continue

                PLdB = - 10 * np.log10(channel_gain[cluster * n_user_cluster + tx,
                                                    cluster * n_user_cluster + rx])
                if (sigma > 0):
                    PLdB = PLdB + np.random.normal(0, sigma, 1)
                desired_gain = np.float_power(10, -PLdB / 10)


                interference = 0
                for interfered_cluster in range(n_cluster):
                    if interfered_cluster == cluster: continue
                    for interfered_tx in range(n_user_cluster):
                        PLdB = - 10 * np.log10(channel_gain[interfered_cluster * n_user_cluster + interfered_tx,
                                                            cluster * n_user_cluster + rx])
                        if (sigma > 0):
                            PLdB = PLdB + np.random.normal(0, sigma, 1)
                        interfered_gain = np.float_power(10, -PLdB / 10)
                        interference += 1 / n_user_cluster * power_alloc[interfered_cluster] * interfered_gain

                SINR = power_alloc[cluster] * desired_gain / (interference + env.B * env.Noise)
                Data_rate = env.B * np.log2(1 + SINR / SNR_gap)
                Network_capacity_sum += Data_rate / (10**6)
                if (Data_rate >= R_min):
                    Success_count = Success_count + 1
                Total_number_of_test += 1

    coverage_prob = Success_count / Total_number_of_test
    network_capacity = Network_capacity_sum / Total_number_of_test
    return coverage_prob, network_capacity


def cal_CP(cluster, channel_gain, power_alloc, SNR_gap, sigma, R_min, env):
    # Calculate the analytical coverage probability
    n_cluster = env.n_cluster
    n_user_cluster = env.n_user_cluster
    Prob = np.zeros(n_user_cluster)
    for tx in range(n_user_cluster):
        for rx in range(n_user_cluster):
            if rx == tx: continue
            interference = 0
            for interfered_cluster in range(n_cluster):
                if interfered_cluster == cluster: continue
                for interfered_tx in range(n_user_cluster):
                    interference += 1 / n_user_cluster * power_alloc[interfered_cluster] * \
                                    channel_gain[interfered_cluster * n_user_cluster + interfered_tx,
                                                 cluster * n_user_cluster + rx]

            h_min = (2**(R_min/env.B) - 1) * (interference + env.B * env.Noise) / power_alloc[cluster] * SNR_gap
            h_min_dB = 10 * np.log10(h_min)
            h_mean_dB = 10 * np.log10(channel_gain[cluster * n_user_cluster + tx, cluster * n_user_cluster + rx])
            Prob[tx] += 1 / (n_user_cluster - 1) * norm.sf((h_min_dB - h_mean_dB)/sigma)
    return np.mean(Prob)

def cal_CP_simulate(cluster, channel_gain, power_alloc, SNR_gap, sigma, R_min, env):
    # Calculate the simulated coverage probability

    n_cluster = power_alloc.shape[0]
    n_user_cluster = env.n_user_cluster
    N_iteration = 200 # 500
    if (sigma == 0):
        N_iteration = 1

    Total_number_of_test = 0
    Network_capacity_sum = 0
    Success_count = 0


    for iter in range(N_iteration):
        for tx in range(n_user_cluster):
            for rx in range(n_user_cluster):
                if rx == tx: continue

                PLdB = - 10 * np.log10(channel_gain[cluster * n_user_cluster + tx,
                                                    cluster * n_user_cluster + rx])
                if (sigma > 0):
                    PLdB = PLdB + np.random.normal(0, sigma, 1)
                desired_gain = np.float_power(10, -PLdB / 10)


                interference = 0
                for interfered_cluster in range(n_cluster):
                    if interfered_cluster == cluster: continue
                    for interfered_tx in range(n_user_cluster):
                        PLdB = - 10 * np.log10(channel_gain[interfered_cluster * n_user_cluster + interfered_tx,
                                                            cluster * n_user_cluster + rx])
                        if (sigma > 0):
                            PLdB = PLdB + np.random.normal(0, sigma, 1)
                        interfered_gain = np.float_power(10, -PLdB / 10)
                        interference += 1 / n_user_cluster * power_alloc[interfered_cluster] * interfered_gain

                SINR = power_alloc[cluster] * desired_gain / (interference + env.NoisePower[cluster])
                Data_rate = env.B * np.log2(1 + SINR / SNR_gap)
                Network_capacity_sum += Data_rate / (10**6)
                if (Data_rate >= R_min):
                    Success_count = Success_count + 1
                Total_number_of_test += 1

    coverage_prob = Success_count / Total_number_of_test
    network_capacity = Network_capacity_sum / Total_number_of_test
    return coverage_prob, network_capacity

def check_feasibility(h, gamma, bandwidth, env, power, cluster_index):
    n_UE = len(h[0, :])

    B = np.zeros((n_UE, n_UE))
    for n in range(n_UE):
        for j in range(n_UE):
            if n == j:
                B[n, j] = 0
            else:
                B[n, j] = gamma[n] * h[j, n] / h[n, n]

    u = np.zeros(n_UE)
    for n in range(n_UE):
        u[n] = gamma[n] / h[n, n] * env.NoisePower[cluster_index[n]]

    spec_radius = np.max(np.abs(np.linalg.eigvals(B)))
    # print("The spectral radius is {0}".format(spec_radius))

    if spec_radius < 1:
        # print("Feasible minRate constraints.")

        p_min = np.linalg.inv(np.identity(n_UE) - B) @ u
        # print("The minimum power vector is {0}".format(p_min))

        if np.max(p_min) < power:
            # print("Feasible power constraints.")
            is_feasible = True
        else:
            # print("Infeasible power constraints.")
            is_feasible = False
    else:
        # print("Infeasible minRate constraints.")

        is_feasible = False

    G = np.zeros((n_UE, n_UE))
    for n in range(n_UE):
        for j in range(n_UE):
            G[n, j] = h[j, n] / h[n, n]

    lam = np.max(np.abs(np.linalg.eigvals(G)))

    gamma_max = 1 / (lam - 1)

    return is_feasible, gamma_max


def estimate_throughput(i_UE, channel_cluster0, env, p_max, ch_gains):
    channel_cluster = copy.deepcopy(channel_cluster0)
    channel_cluster.append(i_UE)

    B = env.B
    N_0 = env.Noise
    SNR_gap = env.SNR_gap[i_UE]

    ch_gains = ch_gains[channel_cluster, :]
    ch_gains = ch_gains[:, channel_cluster]

    gamma = (2 ** (env.QoS / B) - 1) * SNR_gap * np.ones(len(channel_cluster))

    is_feasible, _ = check_feasibility(ch_gains, gamma, B, env.Noise, p_max)

    sum_of_ph = p_max * np.sum(ch_gains[:-1, -1])

    estT = B * np.log2(1 + ch_gains[-1, -1] * p_max / SNR_gap / (sum_of_ph + B * N_0))

    return is_feasible, estT

def diff_in_throughput(i_UE, channel_cluster0, env, p_max, ch_gains):
    B = env.B
    # N_0 = env.Noise

    ch_gains0 = copy.deepcopy(ch_gains)
    ch_gains0 = ch_gains0[channel_cluster0, :]
    ch_gains0 = ch_gains0[:, channel_cluster0]
    ch_gains0 = np.asarray(ch_gains0)

    num_UE_0 = len(channel_cluster0)

    sum_rate0 = 0
    for n in range(num_UE_0):
        sum_of_ph = p_max * (np.sum(ch_gains0[:, n]) - ch_gains0[n, n])
        sum_rate0 += B * np.log2(1 + ch_gains0[n, n] * p_max / env.SNR_gap[channel_cluster0[n]] / (sum_of_ph + env.NoisePower[channel_cluster0[n]]))

    channel_cluster = copy.deepcopy(channel_cluster0)
    channel_cluster.append(i_UE)

    num_UE = len(channel_cluster)

    SNR_gap = env.SNR_gap[i_UE]

    ch_gains = ch_gains[channel_cluster, :]
    ch_gains = ch_gains[:, channel_cluster]

    gamma = (2 ** (env.QoS / B) - 1) * SNR_gap * np.ones(len(channel_cluster))

    is_feasible, _ = check_feasibility(ch_gains, gamma, B, env, p_max, channel_cluster)

    sum_rate = 0
    for n in range(num_UE):
        sum_of_ph = p_max * (np.sum(ch_gains[:, n]) - ch_gains[n, n])
        sum_rate += B * np.log2(1 + ch_gains[n, n] * p_max / env.SNR_gap[channel_cluster[n]] / (sum_of_ph + env.NoisePower[channel_cluster[n]]))

    return is_feasible, sum_rate - sum_rate0