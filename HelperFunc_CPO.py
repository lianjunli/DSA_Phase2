import numpy as np
import copy

def objective_value(channel_alloc, power_alloc, priority, channel_gain, B, noise_vec, SNR_gap):
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
                               priority[n] * B/(10**6) * \
                               np.log2(1 + channel_gain[n, n] / SNR_gap[n] * power_alloc[n, m] / (inter_sum + noise_vec[n]))
    return capacity_sum

def objective_value_SU(p, SU_index, channel_alloc, power_alloc, priority, channel_gain, B, noise_vec, SNR_gap):
    # get objective function of a specific user
    n_channel = power_alloc.shape[1]
    n_su = power_alloc.shape[0]
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
                           np.log2(1 + channel_gain[n, n] / SNR_gap[n] * p[m] / (inter_sum + noise_vec[n]))

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
                                   / (inter_sum + noise_vec[k] + channel_alloc[n, m] * channel_gain[k, n] * p[m]))
    return capacity_sum

def capacity_SU(p, SU_index, channel_alloc, power_alloc, priority, channel_gain, B, noise_vec, SNR_gap):
    # get channel capacity of a specific user
    n_channel = power_alloc.shape[1]
    n_su = power_alloc.shape[0]
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
                           np.log2(1 + channel_gain[n, n] / SNR_gap[n] * p[m] / (inter_sum + noise_vec[n]))
    return capacity_sum

def check_feasibility(h, gamma, noise_mat, cg, power, channel_cluster, i_CG):
    n_UE = len(channel_cluster)

    B = np.zeros((n_UE, n_UE))
    for n in range(n_UE):
        for j in range(n_UE):
            if n == j:
                B[n, j] = 0
            else:
                B[n, j] = gamma[n] * h[j, n] / h[n, n]

    u = np.zeros(n_UE)
    for n in range(n_UE):
        Noise = 0
        for i in range(len(cg.channel_groups[i_CG])):
            Noise += noise_mat[channel_cluster[n], np.where(np.asarray(cg.channel_IDs) == cg.channel_groups[i_CG][i])[0][0]]
        u[n] = gamma[n] / h[n, n] * Noise

    spec_radius = np.max(np.abs(np.linalg.eigvals(B)))
    # print("The spectral radius is {0}".format(spec_radius))

    p_min = None
    p_min_scaled = None

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

        p_min_scaled = power / np.max(p_min) * p_min

    else:
        # print("Infeasible minRate constraints.")

        is_feasible = False

    G = np.zeros((n_UE, n_UE))
    for n in range(n_UE):
        for j in range(n_UE):
            G[n, j] = h[j, n] / h[n, n]

    lam = np.max(np.abs(np.linalg.eigvals(G)))

    gamma_max = 1 / (lam - 1 + np.finfo(float).eps)

    return is_feasible, gamma_max, p_min, p_min_scaled

def diff_in_throughput(i_UE, i_CG, channel_cluster0, noise_mat, SNR_gap, QoS, cg, p_max, ch_gains, shadow_Fading_Margin):
    B = cg.bandwidth_CGs[i_CG]

    ch_gains0 = copy.deepcopy(ch_gains)
    ch_gains0 = ch_gains0[channel_cluster0, :]
    ch_gains0 = ch_gains0[:, channel_cluster0]
    ch_gains0 = np.asarray(ch_gains0)

    num_UE_0 = len(channel_cluster0)

    sum_rate0 = 0
    for n in range(num_UE_0):
        Noise = 0
        for i in range(len(cg.channel_groups[i_CG])):
            Noise += noise_mat[channel_cluster0[n], np.where(np.asarray(cg.channel_IDs) == cg.channel_groups[i_CG][i])[0][0]]
        sum_of_ph = p_max * (np.sum(ch_gains0[:, n]) - ch_gains0[n, n])
        sum_rate0 += B * np.log2(1 + ch_gains0[n, n] * p_max / SNR_gap[channel_cluster0[n]] / (sum_of_ph + Noise))

    channel_cluster = copy.deepcopy(channel_cluster0)
    channel_cluster.append(i_UE)

    num_UE = len(channel_cluster)

    # SNR_gap = SNR_gap[i_UE]

    ch_gains = ch_gains[channel_cluster, :]
    ch_gains = ch_gains[:, channel_cluster]

    gamma = (2 ** (QoS / B) - 1) * SNR_gap[i_UE] * (10 ** (shadow_Fading_Margin / 10)) * np.ones(len(channel_cluster))

    is_feasible, _, p_min, p_min_scaled = check_feasibility(ch_gains, gamma, noise_mat, cg, p_max, channel_cluster, i_CG)

    sum_rate = 0
    for n in range(num_UE):
        Noise = 0
        for i in range(len(cg.channel_groups[i_CG])):
            Noise += noise_mat[channel_cluster[n], np.where(np.asarray(cg.channel_IDs) == cg.channel_groups[i_CG][i])[0][0]]
        sum_of_ph = p_max * (np.sum(ch_gains[:, n]) - ch_gains[n, n])
        sum_rate += B * np.log2(1 + ch_gains[n, n] * p_max / SNR_gap[channel_cluster[n]] / (sum_of_ph + Noise))

    if is_feasible:
        p_min = p_min[:, np.newaxis]
        p_min_scaled = p_min_scaled[:, np.newaxis]
    else:
        p_min = None
        p_min_scaled = None

    return is_feasible, sum_rate - sum_rate0, p_min, p_min_scaled

def translate_to_std_output(channel_allocation, channel_groups, cluster_ID, power_allocation, sort_output = False):
    out_cluster_IDs=[]
    out_1st_channel_idx=[]
    out_num_channels=[]
    out_power=[]
    for i in range(len(channel_allocation)):
        for j in range(len(channel_allocation[i])):
            cluster = channel_allocation[i][j][0]
            out_cluster_IDs.append(cluster_ID[cluster])
            out_1st_channel_idx.append(channel_groups[i][0])
            out_num_channels.append(len(channel_groups[i]))
            out_power.append(power_allocation[cluster])

    if sort_output:
        idx = sorted(range(len(out_cluster_IDs)), key=lambda k: out_cluster_IDs[k])
        out_cluster_IDs = [out_cluster_IDs[i] for i in idx]
        out_1st_channel_idx = [out_1st_channel_idx[i] for i in idx]
        out_num_channels = [out_num_channels[i] for i in idx]
        out_power = [out_power[i] for i in idx]
    return out_cluster_IDs, out_1st_channel_idx, out_num_channels, out_power

