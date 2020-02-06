from Environment_simulation import environment_simulation
from CPO import CPO
from Performance_evaluation import performance_evaluation
import numpy as np
from tqdm import tqdm

# The total number of clusters
# n_cluster_set = [10, 20, 30, 40, 50]
n_cluster_set = [10]

n_round = 5

result = np.zeros((len(n_cluster_set), n_round, 6))

for i in range(len(n_cluster_set)):
    for j in range(n_round):
        ###----------------------SYSTEM INPUTS----------------------###
        print('cluster number: ', n_cluster_set[i], ' round: ', j+1)
        random_seed = i+j

        # Minimum data rate scaling up factor in dB
        min_Rate_Margin = 0
        # Shadow fading margin in dB (for environment simulation)
        Log_Normal_sigma = 0

        # Shadow fading margin in dB (for CPO)
        shadow_Fading_Margin = 0

        # Simulation area (square meter)
        area = 1200

        # Intra cluster channel gain type used in minimum data constraint: min: minimum channel gain; average: geometric average channel gain
        minRate_intra_gain_type = 1  # min = 2, average = 1

        # Intra cluster channel gain type used in DC objective function: min: minimum channel gain; average: geometric average channel gain
        DC_intra_gain_type = 1 # min = 2, average = 1

        # The total number of clusters
        n_cluster = n_cluster_set[i]

        # Radio cluster ID
        cluster_ID = list(range(2001, 2001 + n_cluster))

        # The number of users per cluster
        n_user_cluster = 8

        # SNR gap in dB
        SNR_gap_dB = [3.9] * n_cluster

        # Radio cluster priority
        priority = [1] * n_cluster

        # Minimum data rate constraint (Mbps)
        minRate = [0.1] * n_cluster

        # Maximum power constraint (dBm)
        maxPower = [30] * n_cluster

        # Channel IDs
        # channel_IDs = [2, 11, 12, 13, 14, 15, 16]
        # channel_IDs = [11, 12, 13, 14, 15]
        # channel_IDs = [11, 12, 13]
        channel_IDs = [11]

        # Unit channel bandwidth (MHz)
        unit_bandwidth = 1.25

        # NoisePower
        noise_mat=[]


        ###----------------------SYSTEM INPUTS END----------------------###
        # simulate environment
        h_mean, h_min, h_std_dB, h_all, h_all_dB, noise_mat \
            = environment_simulation(channel_IDs, n_cluster, n_user_cluster, area, Log_Normal_sigma, random_seed)

        # channel allocation and power allocation
        cluster_IDs, first_channel_idx, num_channels, power, SUCCESS_INDICATOR, noise_vec, cluster_infeasible_IDs, power_GA, \
        GA_time, GA_converge_time, DC_time, total_rate_GA, total_rate_after_DC, GA_converge_i\
            = CPO(min_Rate_Margin, h_mean, h_min, h_std_dB, shadow_Fading_Margin, minRate_intra_gain_type, DC_intra_gain_type,
                  SNR_gap_dB, priority, minRate, maxPower, cluster_ID, channel_IDs, noise_mat, unit_bandwidth)

        result[i,j,:] = [GA_time, GA_converge_time, DC_time, total_rate_GA, total_rate_after_DC, GA_converge_i]

        # print('\n** CPO Outputs:')
        # print('Clusters cannot be assigned:', cluster_infeasible_IDs)
        # print('Cluster ID', cluster_IDs)
        # print('Assigned first channel index', first_channel_idx)
        # print('Number of channels assigned', num_channels)
        # print('Transmit power DC', [round(elem, 2) for elem in power])
        # print('Transmit power GA', [round(elem, 2) for elem in power_GA])
        # print('Success indicator ', SUCCESS_INDICATOR)

        # performance evaluation
        # print('\n** Evaluation with DC power allocation')
        # performance_evaluation(cluster_IDs, first_channel_idx, num_channels, power, cluster_ID, h_all, noise_vec, 0,
        #                        n_cluster, n_user_cluster, SNR_gap_dB, minRate, unit_bandwidth)
        #
        # print('\n** Evaluation with GA power allocation')
        # performance_evaluation(cluster_IDs, first_channel_idx, num_channels, power_GA, cluster_ID, h_all, noise_vec, 0,
        #                        n_cluster, n_user_cluster, SNR_gap_dB, minRate, unit_bandwidth)
np.save('.\\saved_results\\GA_power_results.npy', result)
print('done')