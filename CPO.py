from Channel_Allocation_Phase2 import CA_type1
from Channel_Grouping import ChannelGrouping
from Power_allocation import PA_GP_MCS_minRate
from DC_MCS_minRate_channelCap import PA_DC_MCS_minRate_channelCap
from HelperFunc_MCS import *
import os

def CPO(minRateMargin, h_mean, h_min, h_std_dB, shadow_Fading_Margin, minRate_intra_gain_type, DC_intra_gain_type, SNR_gap_dB, priority, minRate, maxPower, cluster_ID, channel_IDs, noise_mat, n_user_cluster):
    alpha = 10 ** (minRateMargin / 10)
    SU_power = 10 ** (np.asarray(maxPower) / 10)
    priority = np.asarray(priority)
    SNR_gap = 10 ** (np.asarray(SNR_gap_dB) / 10)
    n_cluster = len(maxPower)
    # QAM capability
    QAM_cap = np.ones(n_cluster) * 10
    # multichannel capability
    channel_cap = np.ones(n_cluster, dtype=int) * 1
    # add the shadow fading margin in channel coefficients
    for n in range(n_cluster):
        h_mean[n, n] = 10 ** (-np.sqrt(shadow_Fading_Margin ** 2) / 10) * h_mean[n, n]
        h_min[n, n] = 10 ** (-np.sqrt(shadow_Fading_Margin ** 2) / 10) * h_min[n, n]

    SUCCESS_INDICATOR_CA = False
    SUCCESS_INDICATOR_DC = False
    SUCCESS_INDICATOR = False
    power_alloc_DC_record = np.zeros(n_cluster)


    print('\n** alpha equals to {}'.format(alpha))

    # Initialize QoS requirement of SUs
    QoS_initial = np.asarray(minRate) * (10 ** 6)
    QoS = QoS_initial * alpha

    # Initialize channel grouping
    cg = ChannelGrouping(channel_IDs)

    # Channel Allocation
    channel_alloc_type1 = None
    while cg.n_large_CGs > 0:
        channel_alloc_type1 = CA_type1(n_cluster, h_mean, h_min, cg, minRate_intra_gain_type, SU_power[0], noise_mat, SNR_gap, QoS[0])
        if channel_alloc_type1:
            break
        else:
            cg.split()
    channel_alloc_type1 = CA_type1(n_cluster, h_mean, h_min, cg, minRate_intra_gain_type, SU_power[0], noise_mat, SNR_gap, QoS[0])

    if channel_alloc_type1:
        print('Channel allocation succeed.')
        SUCCESS_INDICATOR_CA = True
        print('\nCG configuration: {}'.format(cg.channel_groups))

    else:
        print('Channel allocation fail, no feasible solution.')
        quit()

    # power allocation
    noise_vec = np.zeros(n_cluster)
    idx_ch = 0
    for c1 in channel_alloc_type1:

        print('\n* Channel {0} with clusters: {1}'.format(idx_ch + 1, c1))

        B = cg.bandwidth_CGs[idx_ch]

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
        h_mean_cg = h_mean[cluster_list][:,cluster_list]
        h_min_cg = h_min[cluster_list][:,cluster_list]
        h_std_dB_cg = h_std_dB[cluster_list]


        if minRate_intra_gain_type == 2:
            h_minRate = h_min_cg
        else:
            h_minRate = h_mean_cg

        if DC_intra_gain_type == 2:
            h_DC = h_min_cg
        else:
            h_DC = h_mean_cg

        # Initialize the channel allocation
        channel_alloc_init = np.ones((len(cluster_list), 1))

        # Initialize the update order
        update_priority_optimizer = np.zeros(len(cluster_list))
        for n in range(n_cluster_optimizer):
            update_priority_optimizer[n] = priority_optimizer[n] * h_mean[n, n] * (10 ** 6)
        update_order_optimizer = np.argsort(-update_priority_optimizer)

        # Noise power calculation
        channel_cg_IDs = cg.channel_groups[idx_ch] #1.25M channel IDs in this channel group, IDs can be inconsecutive number
        channel_cg_Idx = [] #Index are consecutive number starting from 0
        for e in channel_cg_IDs:
            channel_Idx = cg.channel_IDs.index(e)
            channel_cg_Idx.append(channel_Idx)
        noise_cg_mat = noise_mat[cluster_list][:,channel_cg_Idx]
        noise_vec_cg = np.sum(noise_cg_mat,axis=1)
        noise_vec[cluster_list] = noise_vec_cg

        '''
        Iteration (Geometric programming)
        '''
        # Use the initialized channel allocation
        channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)

        objective_list_GP = {'total': []}
        power_alloc_GP = PA_GP_MCS_minRate(channel_alloc_optimizer, h_minRate, B, noise_vec_cg,
                                             priority_optimizer,
                                             SU_power_optimizer, QoS_optimizer,
                                             SNR_gap_optimizer,
                                             QAM_cap_optimizer, objective_list_GP)

        '''
        Iteration (DC programming)
        '''
        # Use the initialized channel allocation
        channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)

        # Record the sum weighted data rate
        objective_list_DC = {'total': []}

        # Use GP as initial power allocation
        power_alloc_DC = copy.deepcopy(power_alloc_GP)

        # Record the objective value
        objective_list_DC['total'].append(
            objective_value(channel_alloc_optimizer, power_alloc_DC,
                            priority_optimizer, h_DC, B, noise_vec_cg,
                            SNR_gap_optimizer))
        # power allocation update
        power_engine = PA_DC_MCS_minRate_channelCap(power_alloc_DC, channel_alloc_optimizer,
                                                    h_DC,
                                                    B,
                                                    noise_vec_cg,
                                                    priority_optimizer, SU_power_optimizer,
                                                    QoS_optimizer,
                                                    SNR_gap_optimizer, QAM_cap_optimizer,
                                                    channel_cap_optimizer,
                                                    objective_list_DC, update_order_optimizer,
                                                    h_minRate)

        power_alloc_DC = power_engine.power_alloc
        SUCCESS_INDICATOR_DC = power_engine.Succeed
        if not SUCCESS_INDICATOR_DC:
            print('Power allocation fail!')
            break
        power_alloc_DC_record[cluster_list] = power_alloc_DC.reshape(n_cluster_optimizer)

        '''
        print individual user's power
        '''
        print('Power allocation (DC):')
        for i in range(n_cluster_optimizer):
            print('Cluster %d: Power = %.2f mW' % (cluster_list[i],power_alloc_DC[i][0]))
            # print('{} mW'.format(power_alloc_DC[i][0]))

        idx_ch += 1
    SUCCESS_INDICATOR = SUCCESS_INDICATOR_CA and SUCCESS_INDICATOR_DC

    if SUCCESS_INDICATOR:
        print('Power allocation succeed.')
        file_path = '.\\saved_results\\'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        np.save(file_path + 'power_alloc_DC', power_alloc_DC_record)
        np.save(file_path + 'channel_alloc.npy', channel_alloc_type1)

    return channel_alloc_type1, cg.bandwidth_CGs, power_alloc_DC_record, noise_vec, SUCCESS_INDICATOR