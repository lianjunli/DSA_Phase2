import numpy as np

def cal_CP_simulate(cluster, channel_gain, power_alloc, SNR_gap, sigma, R_min, n_user_cluster, noise_vec, B):
    # Calculate the simulated coverage probability

    n_cluster = power_alloc.shape[0]
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

                SINR = power_alloc[cluster] * desired_gain / (interference + noise_vec[cluster])
                Data_rate = B * np.log2(1 + SINR / SNR_gap)
                Network_capacity_sum += Data_rate / (10**6)
                if (Data_rate >= R_min):
                    Success_count = Success_count + 1
                Total_number_of_test += 1

    coverage_prob = Success_count / Total_number_of_test
    network_capacity = Network_capacity_sum / Total_number_of_test
    return coverage_prob, network_capacity

def translate_from_std_output(out_cluster_IDs, out_1st_channel_idx, out_num_channels, out_power, cluster_ID):
    total_num_cluster = len(cluster_ID)
    power_allocation = np.zeros(total_num_cluster)
    channel_allocation=[]
    channel_groups=[]
    first_channel_set=[]
    cluster_index = [cluster_ID.index(element) for element in out_cluster_IDs]
    for i in range(len(cluster_index)):
        first_channel = out_1st_channel_idx[i]
        if not first_channel in first_channel_set:
            first_channel_set.append(first_channel)
            channel_groups.append(list(range(first_channel,first_channel+out_num_channels[i])))
            channel_allocation.append([])
        channel_idx = first_channel_set.index(first_channel)
        channel_allocation[channel_idx].append([cluster_index[i]])
    power_allocation[cluster_index] = out_power
    return channel_allocation, channel_groups, power_allocation