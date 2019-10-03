import numpy as np
import copy
import matplotlib.pyplot as plt
from HelperFunc_MCS import diff_in_throughput

def CA_type1(env, minRate_inter_gain_type):

    n_channel = env.n_channel
    n_user_cluster = env.n_user_cluster

    center_x = env.center_x
    center_y = env.center_y
    user_x = env.user_x
    user_y = env.user_y

    area = env.area

    n_cluster = env.n_cluster
    unassigned_user_list = [int(i) for i in range(n_cluster)]

    if minRate_inter_gain_type == 'min':
        _, ch_gains, _, _, _ = env.channel_gain(list(range(n_cluster)))
    else:
        ch_gains, _, _, _, _ = env.channel_gain(list(range(n_cluster)))

    channel_cluster = [[] for i in range(n_channel)]

    # find the two clusters with smallest channel gain and assign them to cluster 0
    channel_cluster[0].append(int(np.where(np.abs(np.triu(ch_gains, 1) - np.min(ch_gains)) < 10**(-18))[0]))
    channel_cluster[0].append(int(np.where(np.abs(np.triu(ch_gains, 1) - np.min(ch_gains)) < 10**(-18))[1]))
    unassigned_user_list.remove(int(np.where(np.abs(np.triu(ch_gains, 1) - np.min(ch_gains)) < 10**(-18))[0]))
    unassigned_user_list.remove(int(np.where(np.abs(np.triu(ch_gains, 1) - np.min(ch_gains)) < 10**(-18))[1]))

    # Find the cluster with the largest interference channel gain to the channel group 1.
    # Assign this cluster to channel group 2.
    maxGain = 0
    for i_user in unassigned_user_list:
        for i2 in channel_cluster[0]:
            if ch_gains[i_user][i2] > maxGain:
                idx = i_user
                maxGain = ch_gains[i_user][i2]
    channel_cluster[1].append(idx)
    unassigned_user_list.remove(idx)

    # For the remaining channels, assign each one with a cluster
    for i_ch in range(2, n_channel):
        maxGain = 0
        for i_user in unassigned_user_list:
            for i2 in range(i_ch):
                gain2cluster = np.max(ch_gains[i_user][channel_cluster[i2]])
                if gain2cluster > maxGain:
                    idx = i_user
                    maxGain = gain2cluster
        channel_cluster[i_ch].append(idx)
        unassigned_user_list.remove(idx)

    # For the remaining unassigned clusters, check their feasibility in each channel group and allocate channel based
    # on estimated throughput
    for i_user in unassigned_user_list:
        maxThrougput = 0
        exist_feasible_ch = False
        for i_ch in range(n_channel):
            is_feasible, estThroughput = diff_in_throughput(i_user, channel_cluster[i_ch], env, 5000, ch_gains)
            if estThroughput > maxThrougput and is_feasible:
                idx = i_ch
                maxThrougput = estThroughput
                exist_feasible_ch = True
        if exist_feasible_ch == False:
            return None
        channel_cluster[idx].append(i_user)

    temp = [[] for i in range(n_channel)]
    for i_ch in range(n_channel):
        for elem in channel_cluster[i_ch]:
            temp[i_ch].append([elem])

    channel_cluster = temp

    # plot_SU_location_CA(channel_cluster, center_x, center_y, user_x, user_y, area)

    return channel_cluster


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

