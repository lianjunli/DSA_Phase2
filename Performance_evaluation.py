from HelperFunc_Eva import cal_CP_simulate, translate_from_std_output
import os
import numpy as np

def performance_evaluation(cluster_IDs, first_channel_idx, num_channels, power, cluster_ID,
                           h_all, noise_vec, Log_Normal_sigma, n_cluster, n_user_cluster, SNR_gap_dB, minRate, unit_bandwidth):
    channel_allocation, channel_groups, power_allocation = \
        translate_from_std_output(cluster_IDs, first_channel_idx, num_channels, power, cluster_ID)

    # Initialize the parameters
    CP_simulation = np.zeros(n_cluster)
    capacity_simulation = np.zeros(n_cluster)
    SNR_gap = 10 ** (np.asarray(SNR_gap_dB) / 10)
    QoS_initial = np.asarray(minRate) * (10 ** 6)

    idx_ch = 0
    for c1 in channel_allocation:

        print('\n* Channel {0} with clusters: {1}'.format(idx_ch + 1, c1))

        B = len(channel_groups[idx_ch]) *unit_bandwidth*10**6

        # Retrieve the clusters that share one channel
        if (c1 == None):
            continue
        cluster_list = []
        for c2 in c1:
            for c3 in c2:
                cluster_list.append(c3)
        cluster_list.sort()

        # Initialize the parameters
        n_cluster_optimizer = len(cluster_list)
        SNR_gap_optimizer = SNR_gap[cluster_list]
        QoS_optimizer = QoS_initial[cluster_list]
        power_allocation_cg = power_allocation[cluster_list]
        noise_vec_cg = noise_vec[cluster_list]
        user_idx=[]
        for i in cluster_list:
            user_idx.extend(list(range(i*n_user_cluster,(i+1)*n_user_cluster)))
        h_all_cg = h_all[user_idx][:,user_idx]
        '''
        Coverage probability and Network Capacity calculation
        '''
        for n in range(n_cluster_optimizer):
            # Simulation
            CP_simulation[cluster_list[n]], capacity_simulation[cluster_list[n]] = \
                cal_CP_simulate(n, h_all_cg, power_allocation_cg, SNR_gap_optimizer[n],
                                Log_Normal_sigma, QoS_optimizer[n], n_user_cluster, noise_vec_cg, B)
            print("Cluster %d: log normal sigma: %d dB; Coverage Probability = %.4f; Network capacity = %.2f Mbps."
                  % (
                      cluster_list[n], Log_Normal_sigma, CP_simulation[cluster_list[n]], capacity_simulation[cluster_list[n]]))

        idx_ch += 1

    file_path = '.\\saved_results\\'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    np.save(file_path + 'CP_simulation', CP_simulation)
    np.save(file_path + 'capacity_simulation', capacity_simulation)