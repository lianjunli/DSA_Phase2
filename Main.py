from Channel_model import Winner2_Optimizer
from Power_allocation import PA_GP_MCS_minRate
from DC_MCS_minRate_channelCap import PA_DC_MCS_minRate_channelCap
from HelperFunc_MCS import *
from Channel_allocation import CA_type1
import os

import numpy as np
import copy


if __name__ == "__main__":

#----------------------  System Parameters  ----------------------

    # Minimum data rate constraint (Mbps)
    minRate = 1

    # Maximum power constraint (milliwatt)
    maxPower = 5000

    # Minimum data rate scaling up factor
    alpha_list = [1, 2]

    # Fading margin used in method 2 (dB)
    Log_Normal_sigma = 0

    # Simulation area (square meter)
    area = 700

    # Simulation method. 1: without fading margin; 2: with fading margin
    method = 1

    # Intra cluster channel gain type used in minmum data constraint: min: minumu channel gain; average: geometric average channel gain
    minRate_intra_gain_type = 'average' # min, average

    # Intra cluster channel gain type used in DC objective function: min: minumu channel gain; average: geometric average channel gain
    DC_intra_gain_type = 'average' # min, average

    # The total number of clusters
    n_cluster = 10

    # The number of users per cluster
    n_user_cluster = 8

    # SNR gap in dB
    SNR_gap_dB = 3.9

#----------------------  End System Parameters  ----------------------

    # The total number of channels
    n_channel = 2

    # The bandwidth of each channel (Hz)
    bandwidth_list = [5 * (10 ** 6), 5 * (10 ** 6)]

    # Initialize the environment
    env = Winner2_Optimizer(n_channel, n_cluster, n_user_cluster, area)
    env.B = bandwidth_list[0]
    env.B_list = bandwidth_list

    # Initialize the maximum power of each user (mW)
    SU_power = maxPower * np.ones(n_cluster)

    # Initialize priorities among clusters
    priority = np.ones(n_cluster)

    # SNR Gap
    SNR_gap = np.ones(n_cluster) * (10**(SNR_gap_dB / 10))
    env.SNR_gap = SNR_gap

    # QAM capability
    QAM_cap = np.ones(n_cluster) * 10

    # multichannel capability
    channel_cap = np.ones(n_cluster, dtype=int) * 1

    SUCCESS_INDICATOR = False

    '''
    Power Allocation
    '''

    Case = str(method)

    CP_simulation_1 = np.zeros((len(alpha_list), n_cluster))
    CP_simulation_2 = np.zeros((len(alpha_list), n_cluster))
    capacity_simulation_1 = np.zeros((len(alpha_list), n_cluster))
    capacity_simulation_2 = np.zeros((len(alpha_list), n_cluster))
    capacity_optimizer_1 = np.zeros((len(alpha_list), n_cluster))
    capacity_optimizer_2 = np.zeros((len(alpha_list), n_cluster))
    power_alloc_DC_record_1 = np.zeros((len(alpha_list), n_cluster))
    power_alloc_DC_record_2 = np.zeros((len(alpha_list), n_cluster))

    idx = 0
    for alpha in alpha_list:
        print('\n** alpha equals to {}'.format(alpha))

        # Initialize QoS requirement of SUs
        QoS_initial = minRate * (10 ** 6) # 1 Mbps
        QoS = QoS_initial * np.ones(n_cluster)
        env.QoS = QoS_initial * alpha

        # Channel Allocation
        env.n_channel = 2
        env.B_list = [5 * (10 ** 6), 5 * (10 ** 6)]
        channel_alloc_type1 = CA_type1(env, minRate_intra_gain_type)

        if channel_alloc_type1:
            print('\nUse CG configuration 1')
            SUCCESS_INDICATOR = True

        else:
            env.n_channel = 3
            env.B_list = [5 * (10 ** 6), 2.5 * (10 ** 6), 2.5 * (10 ** 6)]
            channel_alloc_type1 = CA_type1(env, minRate_intra_gain_type)

            if channel_alloc_type1:
                print('\nUse CG configuration 2')
                SUCCESS_INDICATOR = True

            else:
                env.n_channel = 4
                env.B_list = [2.5 * (10 ** 6), 2.5 * (10 ** 6), 2.5 * (10 ** 6), 2.5 * (10 ** 6)]
                channel_alloc_type1 = CA_type1(env, minRate_intra_gain_type)

                if channel_alloc_type1:
                    print('\nUse CG configuration 3')
                    SUCCESS_INDICATOR = True

                else:
                    print('No feasible solution under all three CG configurations.')
                    SUCCESS_INDICATOR = False
                    quit()

        # Power allocation
        idx_ch = 0
        for c1 in channel_alloc_type1:

            print('\n* Channel {0} with clusters: {1}'.format(idx_ch + 1, c1))

            env.B = env.B_list[idx_ch]

            # Retrieve the clusters that share one channel
            if (c1 == None):
                continue
            cluster_list = []
            for c2 in c1:
                for c3 in c2:
                    cluster_list.append(c3)
            cluster_list.sort()

            # Initialize the parameters using in the optimizer
            n_cluster_optimizer = len(cluster_list)
            SU_power_optimizer = SU_power[cluster_list]
            priority_optimizer = priority[cluster_list]
            SNR_gap_optimizer = SNR_gap[cluster_list]
            QAM_cap_optimizer = QAM_cap[cluster_list]
            channel_cap_optimizer = channel_cap[cluster_list]
            QoS_optimizer = QoS[cluster_list]

            # Initialize the channel gain
            h_mean_1, h_min_1, h_std_dB, h_all, h_all_dB, = env.channel_gain(cluster_list)
            h_mean_2 = copy.deepcopy(h_mean_1)
            h_min_2 = copy.deepcopy(h_min_1)
            for n in range(n_cluster_optimizer):
                h_mean_2[n, n] = 10 ** (-np.sqrt(Log_Normal_sigma ** 2) / 10) * h_mean_2[n, n]
                h_min_2[n, n] = 10 ** (-np.sqrt(Log_Normal_sigma ** 2) / 10) * h_min_2[n, n]

            if minRate_intra_gain_type == 'min':
                h_minRate_1 = h_min_1
                h_minRate_2 = h_min_2
            else:
                h_minRate_1 = h_mean_1
                h_minRate_2 = h_mean_2

            if DC_intra_gain_type == 'min':
                h_DC_1 = h_min_1
                h_DC_2 = h_min_2
            else:
                h_DC_1 = h_mean_1
                h_DC_2 = h_mean_2

            # Initialize the channel allocation
            channel_alloc_init = np.ones((len(cluster_list), 1))

            # Initialize the update order
            update_priority_optimizer = np.zeros(len(cluster_list))
            for n in range(n_cluster_optimizer):
                update_priority_optimizer[n] = priority_optimizer[n] * h_mean_1[n, n] * (10 ** 6)
            update_order_optimizer = np.argsort(-update_priority_optimizer)

            '''
            Iteration (Geometric programming)
            '''
            # Use the initialized channel allocation
            channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)

            if (Case == '1'):  # with r_min_scale
                objective_list_GP_1 = {'total': []}
                power_alloc_GP_1 = PA_GP_MCS_minRate(channel_alloc_optimizer, h_minRate_1, env,
                                                     priority_optimizer,
                                                     SU_power_optimizer, QoS_optimizer * alpha,
                                                     SNR_gap_optimizer,
                                                     QAM_cap_optimizer, objective_list_GP_1)

            # Record the sum weighted data rate
            if (Case == '2'):  # with r_min_scale and fading margin
                objective_list_GP_2 = {'total': []}
                power_alloc_GP_2 = PA_GP_MCS_minRate(channel_alloc_optimizer, h_minRate_2, env,
                                                     priority_optimizer,
                                                     SU_power_optimizer, QoS_optimizer * alpha,
                                                     SNR_gap_optimizer,
                                                     QAM_cap_optimizer, objective_list_GP_2)

            '''
            Iteration (DC programming)
            '''
            # Use the initialized channel allocation
            channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)

            if (Case == '1'):
                # Record the sum weighted data rate
                objective_list_DC_1 = {'total': []}

                # Use GP as initial power allocation
                power_alloc_DC_1 = copy.deepcopy(power_alloc_GP_1)

                # Record the objective value
                objective_list_DC_1['total'].append(
                    objective_value(channel_alloc_optimizer, power_alloc_DC_1,
                                    priority_optimizer, h_DC_1, env,
                                    SNR_gap_optimizer))
                # power allocation update
                power_engine = PA_DC_MCS_minRate_channelCap(power_alloc_DC_1, channel_alloc_optimizer,
                                                            h_DC_1,
                                                            env,
                                                            priority_optimizer, SU_power_optimizer,
                                                            QoS_optimizer * alpha,
                                                            SNR_gap_optimizer, QAM_cap_optimizer,
                                                            channel_cap_optimizer,
                                                            objective_list_DC_1, update_order_optimizer,
                                                            h_minRate_1)

                power_alloc_DC_1 = power_engine.power_alloc
                power_alloc_DC_record_1[idx, cluster_list] = power_alloc_DC_1.reshape(n_cluster_optimizer)

            if (Case == '2'):
                # Record the sum weighted data rate
                objective_list_DC_2 = {'total': []}

                # Use GP as initial power allocation
                power_alloc_DC_2 = copy.deepcopy(power_alloc_GP_2)

                # Record the objective value
                objective_list_DC_2['total'].append(
                    objective_value(channel_alloc_optimizer, power_alloc_DC_2,
                                    priority_optimizer, h_DC_2, env,
                                    SNR_gap_optimizer))
                # power allocation update
                power_engine = PA_DC_MCS_minRate_channelCap(power_alloc_DC_2, channel_alloc_optimizer,
                                                            h_DC_2,
                                                            env,
                                                            priority_optimizer, SU_power_optimizer,
                                                            QoS_optimizer * alpha,
                                                            SNR_gap_optimizer, QAM_cap_optimizer,
                                                            channel_cap_optimizer,
                                                            objective_list_DC_2, update_order_optimizer,
                                                            h_minRate_2)

                power_alloc_DC_2 = power_engine.power_alloc
                power_alloc_DC_record_2[idx, cluster_list] = power_alloc_DC_2.reshape(n_cluster_optimizer)

            '''
            print individual user's rate
            '''
            print('\n Clusters:')
            print(cluster_list)

            if (Case == '1'):
                user_rate_DC_1 = [[] for i in range(n_cluster_optimizer)]
                print('The data rate (DC):')
                for i in range(n_cluster_optimizer):
                    print('Cluster %d:' % cluster_list[i])
                    for tx in range(n_user_cluster):
                        for rx in range(n_user_cluster):
                            if (rx == tx):
                                continue
                            rate = rate_SU(i, tx, rx, power_alloc_DC_1, h_all, priority_optimizer, env,
                                           SNR_gap_optimizer)
                            user_rate_DC_1[i].append(np.round(rate, decimals=2))
                    print("%.2f Mbps" % (sum(user_rate_DC_1[i]) / len(user_rate_DC_1[i])))
                print('System throughput 1 = %.2f Mbps' % (np.sum(user_rate_DC_1) / len(user_rate_DC_1[0])))

            if (Case == '2'):
                user_rate_DC_2 = [[] for i in range(n_cluster_optimizer)]
                print('The data rate (DC):')
                for i in range(n_cluster_optimizer):
                    print('Cluster %d:' % cluster_list[i])
                    for tx in range(n_user_cluster):
                        for rx in range(n_user_cluster):
                            if (rx == tx):
                                continue
                            rate = rate_SU(i, tx, rx, power_alloc_DC_2, h_all, priority_optimizer, env,
                                           SNR_gap_optimizer)
                            user_rate_DC_2[i].append(np.round(rate, decimals=2))
                    print("%.2f Mbps" % (sum(user_rate_DC_2[i]) / len(user_rate_DC_2[i])))
                print('System throughput 2 = %.2f Mbps' % (np.sum(user_rate_DC_2) / len(user_rate_DC_2[0])))

            '''
            print individual user's power
            '''
            print('\n')
            if (Case == '1'):
                user_rate_DC_1 = [[] for i in range(n_cluster_optimizer)]
                print('Power allocation (DC):')
                for i in range(n_cluster_optimizer):
                    print('Cluster %d:' % cluster_list[i])
                    for tx in range(n_user_cluster):
                        for rx in range(n_user_cluster):
                            if (rx == tx):
                                continue
                    print('{} mW'.format(power_alloc_DC_1[i][0]))

            if (Case == '2'):
                user_rate_DC_2 = [[] for i in range(n_cluster_optimizer)]
                print('The data rate (DC):')
                for i in range(n_cluster_optimizer):
                    print('Cluster %d:' % cluster_list[i])
                    for tx in range(n_user_cluster):
                        for rx in range(n_user_cluster):
                            if (rx == tx):
                                continue
                    print('{} mW'.format(power_alloc_DC_2[i][0]))

            '''
            Coverage probability and Network Capacity calculation
            '''
            print('\n')
            for n in range(n_cluster_optimizer):

                # Simulation
                if (Case == '1'):
                    CP_simulation_1[idx, cluster_list[n]], capacity_simulation_1[idx, cluster_list[n]] = \
                        cal_CP_simulate(n, h_all, power_alloc_DC_1, SNR_gap_optimizer[n],
                                        Log_Normal_sigma, QoS_optimizer[n], env)
                    print("alpha = %.1f, Cluster %d : Coverage Probability (Simulation 1, %d dB) = %.4f"
                          % (
                          alpha, cluster_list[n], Log_Normal_sigma, CP_simulation_1[idx, cluster_list[n]]))

                if (Case == '2'):
                    CP_simulation_2[idx, cluster_list[n]], capacity_simulation_2[idx, cluster_list[n]] = \
                        cal_CP_simulate(n, h_all, power_alloc_DC_2, SNR_gap_optimizer[n],
                                        Log_Normal_sigma, QoS_optimizer[n], env)
                    print("alpha = %.1f, Cluster %d : Coverage Probability (Simulation 2, %d dB) = %.4f"
                          % (
                          alpha, cluster_list[n], Log_Normal_sigma, CP_simulation_2[idx, cluster_list[n]]))

            idx_ch += 1

        idx += 1
        file_path = '.\\saved_results\\'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if (Case == '1'):
            np.save(file_path + 'power_alloc_DC_1', power_alloc_DC_record_1)
            np.save(file_path + 'CP_simulation_1', CP_simulation_1)
            np.save(file_path + 'capacity_simulation_1', capacity_simulation_1)
            np.save(file_path + 'alpha_list_1', alpha_list)
        if (Case == '2'):
            np.save(file_path + 'power_alloc_DC_2', power_alloc_DC_record_2)
            np.save(file_path + 'CP_simulation_2', CP_simulation_2)
            np.save(file_path + 'capacity_simulation_2', capacity_simulation_2)
            np.save(file_path + 'alpha_list_2', alpha_list)
        np.save(file_path + 'channel_alloc.npy', channel_alloc_type1)

