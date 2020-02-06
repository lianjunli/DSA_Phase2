import numpy as np
import copy
import matplotlib.pyplot as plt
from HelperFunc_CPO import diff_in_throughput


def CA_type1(n_cluster, h_mean, h_min, cg, minRate_inter_gain_type, maxPower, noise_mat, SNR_gap, QoS, shadow_Fading_Margin):

    IS_SUCC = True

    n_CGs = cg.n_CGs

    n_cluster = n_cluster
    unassigned_user_list = [int(i) for i in range(n_cluster)]

    if minRate_inter_gain_type == 2:
        ch_gains = h_min
    else:
        ch_gains = h_mean

    CG_cluster = [[] for _ in range(n_CGs)]

    cluster_wo_feasible_ch = []

    interf2all = np.zeros(n_cluster)
    for i_UE in range(n_cluster):
        interf2all[i_UE] = np.sum(ch_gains[i_UE, ch_gains[i_UE, :] != ch_gains[i_UE, i_UE]])

    i_ascending = np.argsort(interf2all)
    i_descending = i_ascending[::-1]
    unassigned_user_list = np.asarray(unassigned_user_list)
    unassigned_user_list = unassigned_user_list[i_descending].tolist()


    # For the remaining unassigned clusters, check their feasibility in each channel group and allocate channel based
    # on estimated throughput
    p_min = [None for _ in range(n_CGs)]
    p_min_scaled = [None for _ in range(n_CGs)]
    p_min_temp = [[] for _ in range(n_CGs)]
    p_min_scaled_temp = [[] for _ in range(n_CGs)]
    for i_user in unassigned_user_list:
        maxThrougput = float("-inf")
        exist_feasible_ch = False
        for i_CG in range(n_CGs):
            is_feasible, estThroughput, p_min_temp[i_CG], p_min_scaled_temp[i_CG] = diff_in_throughput(i_user, i_CG, CG_cluster[i_CG], noise_mat, SNR_gap, QoS, cg, maxPower, ch_gains, shadow_Fading_Margin)
            if estThroughput > maxThrougput and is_feasible:
                idx = i_CG
                maxThrougput = estThroughput
                exist_feasible_ch = True
        if exist_feasible_ch is False and cg.n_large_CGs > 0:
            return None, None, False, p_min, p_min_scaled
        elif exist_feasible_ch is False and cg.n_large_CGs == 0:
            cluster_wo_feasible_ch.append(i_user)
            IS_SUCC = False
            continue
        else:
            CG_cluster[idx].append(i_user)
            p_min[idx] = p_min_temp[idx]
            p_min_scaled[idx] = p_min_scaled_temp[idx]

    temp = [[] for i in range(n_CGs)]
    for i_ch in range(n_CGs):
        for elem in CG_cluster[i_ch]:
            temp[i_ch].append([elem])

    channel_cluster = temp

    # plot_SU_location_CA(channel_cluster, center_x, center_y, user_x, user_y, area)

    return channel_cluster, cluster_wo_feasible_ch, IS_SUCC, p_min, p_min_scaled


def plot_SU_location_CA(channel_cluster, group_x, group_y, SU_x, SU_y, area):

    n_group = group_x.shape[0]

    plt.figure(figsize=(8, 8))
    plt.ylim(0, area)
    plt.xlim(0, area)

    for n in range(n_group):
        labelstr = "Cluster%d" % n
        plt.annotate(
            labelstr,
            xy=(group_x[n], group_y[n]), xytext=(0, 16),
            textcoords='offset points', ha='center', va='bottom',
        )
    plt.ylabel('y', fontsize=14)
    plt.xlabel('x', fontsize=14)


    for index in range(len(channel_cluster)):
        if (channel_cluster[index] != None):
            for cluster in channel_cluster[index]:
                for n in cluster:
                    if (index == 0):
                        plt.plot(SU_x[n], SU_y[n], 'bo')
                    elif (index == 1):
                        plt.plot(SU_x[n], SU_y[n], 'go')
                    elif (index == 2):
                        plt.plot(SU_x[n], SU_y[n], 'ro')
                    elif (index == 3):
                        plt.plot(SU_x[n], SU_y[n], 'mo')
    plt.show()

