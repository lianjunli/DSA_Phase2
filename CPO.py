import numpy as np
import copy
from Channel_Allocation import CA_type1
from Channel_Grouping import ChannelGrouping
from Power_allocation import PA_GP_MCS_minRate
from DC_MCS_minRate_channelCap import PA_DC_MCS_minRate_channelCap
from HelperFunc_CPO import objective_value, translate_to_std_output, capacity_SU
import os
from PowerAllocation_GA import PA_GA
from PowerAllocation_AGA_new import PA_AGA
from PowerAllocation_GA_new import PA_GA_new
from PowerAllocation_1dOpt import PA_1dOpt
import time

def CPO(minRateMargin, h_mean, h_min_diag, h_std_dB, shadow_Fading_Margin, minRate_intra_gain_type, DC_intra_gain_type, SNR_gap_dB, priority, minRate, maxPower, cluster_ID, channel_IDs, noise_mat, unit_bandwidth):
    GA = True
    AGA = True
    GA_new = False
    DC_1d = True
    silent = True

    alpha = 10 ** (minRateMargin / 10)
    SU_power = 10 ** (np.asarray(maxPower) / 10)
    priority = np.asarray(priority)
    SNR_gap = 10 ** (np.asarray(SNR_gap_dB) / 10)
    n_cluster = len(maxPower)
    h_min = copy.deepcopy(h_mean)
    # QAM capability
    QAM_cap = np.ones(n_cluster) * 10
    # multichannel capability
    channel_cap = np.ones(n_cluster, dtype=int) * 1
    # add the shadow fading margin in channel coefficients
    for n in range(n_cluster):
        h_mean[n, n] = 10 ** (-np.sqrt(shadow_Fading_Margin ** 2) / 10) * h_mean[n, n]
        h_min[n, n] = 10 ** (-np.sqrt(shadow_Fading_Margin ** 2) / 10) * h_min_diag[n]

    power_alloc_DC_record = np.zeros(n_cluster)
    power_alloc_GA_record = np.zeros(n_cluster)
    power_alloc_AGA_record = np.zeros(n_cluster)
    power_alloc_GA_new_record = np.zeros(n_cluster)
    power_alloc_1dOpt_record = np.zeros(n_cluster)

    # print('\n** alpha equals to {}'.format(alpha))

    # Initialize QoS requirement of SUs
    QoS_initial = np.asarray(minRate) * (10 ** 6)
    QoS = QoS_initial * alpha

    # Initialize channel grouping
    cg = ChannelGrouping(channel_IDs, unit_bandwidth)

    # Channel Allocation
    channel_alloc_type1 = None
    while cg.n_large_CGs > 0:
        channel_alloc_type1, _, _, _, _ = CA_type1(n_cluster, h_mean, h_min, cg, minRate_intra_gain_type, SU_power[0], noise_mat, SNR_gap, QoS[0], shadow_Fading_Margin)
        if channel_alloc_type1:
            break
        else:
            cg.split()
    channel_alloc_type1, cluster_wo_feasible_ch, SUCCESS_INDICATOR_CA, p_min, p_min_scaled = CA_type1(n_cluster, h_mean, h_min, cg, minRate_intra_gain_type, SU_power[0], noise_mat, SNR_gap, QoS[0], shadow_Fading_Margin)
    cluster_infeasible_IDs = [cluster_ID[element] for element in cluster_wo_feasible_ch]

    if SUCCESS_INDICATOR_CA == False:
        print('Channel allocation failed to accommodate all clusters.')
        # print('Clusters cannot be assigned: {}'.format(cluster_wo_feasible_ch))
        # print('CG configuration: {}'.format(cg.channel_groups))
        # print('Channel allocation result',channel_alloc_type1)
    else:
        # print('Channel allocation succeed.')
        # print('CG configuration: {}'.format(cg.channel_groups))
        # print('Channel allocation result', channel_alloc_type1)
        pass

    # else:
    #     print('Channel allocation fail, no feasible solution.')
    #     quit()

    # power allocation
    noise_vec = np.zeros(n_cluster)
    idx_ch = 0
    SUCCESS_INDICATOR_DC = True
    for c1 in channel_alloc_type1:

        # print('\n* Channel {0} with clusters: {1}'.format(idx_ch + 1, c1))

        B = cg.bandwidth_CGs[idx_ch]

        # Retrieve the clusters that share one channel
        if (c1 == None):
            continue
        cluster_list = []
        for c2 in c1:
            for c3 in c2:
                cluster_list.append(c3)
        cluster_list.sort()

        SUCCESS_INDICATOR_temp = False
        while (not SUCCESS_INDICATOR_temp and len(cluster_list)>0):
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
            Genetic Algorithm
            '''
            # Use the initialized channel allocation
            start = time.time()
            channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)

            if not GA:
                power_alloc_GA = p_min[idx_ch][:len(cluster_list)]
                converge_i = 1
                total_i = 1
            else:
                power_alloc_GA, converge_i, total_i = PA_GA(h_DC, B, noise_vec_cg, SU_power_optimizer, QoS_optimizer, SNR_gap_optimizer, channel_alloc_optimizer,
                      priority_optimizer)
                power_alloc_GA = np.asarray(power_alloc_GA)
                power_alloc_GA_record[cluster_list] = power_alloc_GA

            power_alloc_GA = power_alloc_GA.reshape((-1,1))
            GA_time = time.time() - start
            GA_converge_time = GA_time * converge_i/total_i
            GA_feasible_number = 0
            print('GA time cost:', GA_time, 'GA converge iteration:', converge_i)

            if not silent:
                print('Power allocation (GA):')
            if len(cluster_list) > 0:
                for i in range(n_cluster_optimizer):
                    rate = capacity_SU(power_alloc_GA[i], i, channel_alloc_optimizer, power_alloc_GA,
                                    priority_optimizer, h_DC, B, noise_vec_cg,
                                    SNR_gap_optimizer)
                    if rate*10**6 >= QoS_optimizer[i]:
                        GA_feasible_number+=1
                    if not silent:
                        print('Cluster %d: Power = %.2f mW. Rate = %.2f Mbps' % (cluster_list[i],power_alloc_GA[i][0],rate))

            total_rate_GA = objective_value(channel_alloc_optimizer, power_alloc_GA,
                                                priority_optimizer, h_DC, B, noise_vec_cg,
                                                SNR_gap_optimizer)
            print('Total rate with GA', total_rate_GA)

            '''
            Aternative Genetic Algorithm
            '''
            # Use the initialized channel allocation
            start = time.time()
            channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)
            if not AGA:
                power_alloc_AGA = p_min[idx_ch][:len(cluster_list)]
                AGA_converge_i=1
                AGA_total_i =1
            else:
                power_alloc_AGA,AGA_converge_i, AGA_total_i = PA_AGA(h_DC, B, noise_vec_cg, SU_power_optimizer, QoS_optimizer, SNR_gap_optimizer,
                                         channel_alloc_optimizer,
                                         priority_optimizer, np.array(p_min)[0])
                power_alloc_AGA = np.asarray(power_alloc_AGA)
                power_alloc_AGA_record[cluster_list] = power_alloc_AGA

            power_alloc_AGA = power_alloc_AGA.reshape((-1, 1))
            AGA_time = time.time() - start
            AGA_converge_time = AGA_time * AGA_converge_i / AGA_total_i
            print('AGA time cost:', AGA_time, 'AGA converge iteration:', AGA_converge_i)
            AGA_feasible_number=0
            if not silent:
                print('Power allocation (AGA):')
            if len(cluster_list) > 0:
                for i in range(n_cluster_optimizer):
                    rate = capacity_SU(power_alloc_AGA[i], i, channel_alloc_optimizer, power_alloc_AGA,
                                       priority_optimizer, h_DC, B, noise_vec_cg,
                                       SNR_gap_optimizer)
                    if rate*10**6 >= QoS_optimizer[i]:
                        AGA_feasible_number+=1
                    if not silent:
                        print('Cluster %d: Power = %.2f mW. Rate = %.2f Mbps' % (cluster_list[i], power_alloc_AGA[i][0], rate))

            total_rate_AGA = objective_value(channel_alloc_optimizer, power_alloc_AGA,
                                             priority_optimizer, h_DC, B, noise_vec_cg,
                                             SNR_gap_optimizer)
            print('Total rate with AGA', total_rate_AGA)

            '''
            New Genetic Algorithm
            '''
            # Use the initialized channel allocation
            start = time.time()
            channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)
            if not GA_new:
                power_alloc_GA_new = p_min[idx_ch][:len(cluster_list)]
                GAnew_converge_i=1
                GA_new_total_i =1
                min_rate_adjusted_GA_new=np.zeros(len(cluster_list))
                n_min_rate_adjusted=0
            else:
                power_alloc_GA_new, min_rate_adjusted_GA_new, GA_new_total_i, GAnew_converge_i, n_min_rate_adjusted = PA_GA_new(h_DC, B, noise_vec_cg, SU_power_optimizer, QoS_optimizer, SNR_gap_optimizer,
                                               channel_alloc_optimizer,
                                               priority_optimizer, np.array(p_min)[0])
                power_alloc_GA_new = np.asarray(power_alloc_GA_new)
                power_alloc_GA_new_record[cluster_list] = power_alloc_GA_new

            power_alloc_GA_new = power_alloc_GA_new.reshape((-1, 1))
            GA_new_time = time.time() - start
            GAnew_converge_time = GA_new_time * GAnew_converge_i / GA_new_total_i
            print('New GA time cost:', GA_new_time, 'GA_new converge iteration:', GAnew_converge_i)

            n_fsb_clusters_GA_new = 0
            rate_nfsb_clusters_newGA = []

            if not silent:
                print('Power allocation (New GA):')
            if len(cluster_list) > 0:
                for i in range(n_cluster_optimizer):
                    rate = capacity_SU(power_alloc_GA_new[i], i, channel_alloc_optimizer, power_alloc_GA_new,
                                       priority_optimizer, h_DC, B, noise_vec_cg,
                                       SNR_gap_optimizer)
                    if rate > QoS[0] / (10 ** 6):
                        n_fsb_clusters_GA_new += 1
                    else:
                        rate_nfsb_clusters_newGA.append((i, rate))
                    if not silent:
                        print('Cluster %d: Power = %.2f mW. Rate = %.2f Mbps' % (cluster_list[i], power_alloc_GA_new[i][0], rate))

            total_rate_GA_new = objective_value(channel_alloc_optimizer, power_alloc_GA_new,
                                                priority_optimizer, h_DC, B, noise_vec_cg,
                                                SNR_gap_optimizer)
            print('Total rate with new GA', total_rate_GA_new)
            print('Number of feasible clusters in newGA:', n_fsb_clusters_GA_new)

            '''
            1d Optimized Power
            '''
            # Use the initialized channel allocation
            start = time.time()
            channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)
            power_alloc_1dOpt = PA_1dOpt(h_DC, B, noise_vec_cg, SU_power_optimizer, QoS_optimizer, SNR_gap_optimizer,
                                         channel_alloc_optimizer,
                                         priority_optimizer, np.array(p_min)[0])
            power_alloc_1dOpt = np.asarray(power_alloc_1dOpt)
            power_alloc_1dOpt_record[cluster_list] = power_alloc_1dOpt
            power_alloc_1dOpt = power_alloc_1dOpt.reshape((-1, 1))
            power_alloc_1dOpt_round = np.round(power_alloc_1dOpt, decimals=-1)
            oneD_time = time.time() - start
            print('1d optimizer time cost:', oneD_time)

            n_fsb_clusters_1dOpt = 0
            rate_nfsb_clusters_1dOpt = []
            if not silent:
                print('Power allocation (1d):')
            if len(cluster_list) > 0:
                for i in range(n_cluster_optimizer):
                    rate = capacity_SU(power_alloc_1dOpt[i], i, channel_alloc_optimizer, power_alloc_1dOpt_round,
                                       priority_optimizer, h_DC, B, noise_vec_cg,
                                       SNR_gap_optimizer)
                    if rate > (QoS[0] - 0.001) / (10 ** 6):
                        n_fsb_clusters_1dOpt += 1
                    else:
                        rate_nfsb_clusters_1dOpt.append((i, rate))
                    if not silent:
                        print( 'Cluster %d: Power = %.2f mW. Rate = %.2f Mbps' % (cluster_list[i], power_alloc_1dOpt_round[i][0], rate))

            total_rate_1d = objective_value(channel_alloc_optimizer, power_alloc_1dOpt_round,
                                                priority_optimizer, h_DC, B, noise_vec_cg,
                                                SNR_gap_optimizer)
            print('Total rate with 1d Optimized power', total_rate_1d)
            print('Number of feasible clusters in 1d Optimized power:', n_fsb_clusters_1dOpt)


            '''
            Iteration (DC programming)
            '''
            # Use the initialized channel allocation
            channel_alloc_optimizer = copy.deepcopy(channel_alloc_init)

            # Record the sum weighted data rate
            objective_list_DC = {'total': []}

            # Use GP as initial power allocation
            # power_alloc_DC = copy.deepcopy(power_alloc_GP)

            # Use feasibility check result as initial power allocation
            # power_alloc_DC = p_min[idx_ch][:len(cluster_list)]
            power_alloc_DC = power_alloc_1dOpt_round

            # Zero initial power allocation
            # power_alloc_DC = np.zeros((len(cluster_list), 1))

            # Record the objective value
            total_rate_before = objective_value(channel_alloc_optimizer, power_alloc_DC,
                                priority_optimizer, h_DC, B, noise_vec_cg,
                                SNR_gap_optimizer)
            objective_list_DC['total'].append(total_rate_before)
            print('total rate before DC: ', total_rate_before)

            # power allocation update
            start = time.time()
            if DC_1d:
                power_engine = PA_DC_MCS_minRate_channelCap(power_alloc_DC, channel_alloc_optimizer,
                                                            h_DC,
                                                            B,
                                                            noise_vec_cg,
                                                            priority_optimizer, SU_power_optimizer,
                                                            QoS_optimizer - 0.0001 * 10**6,
                                                            SNR_gap_optimizer, QAM_cap_optimizer,
                                                            channel_cap_optimizer,
                                                            objective_list_DC, update_order_optimizer,
                                                            h_minRate)

                power_alloc_DC = power_engine.power_alloc
            DC_time = time.time() - start
            print('DC time cost:', DC_time)
            # SUCCESS_INDICATOR_temp = power_engine.Succeed
            SUCCESS_INDICATOR_temp = True

            power_alloc_DC_round=np.round(power_alloc_DC, decimals=-1)

            if SUCCESS_INDICATOR_temp:
                power_alloc_DC_record[cluster_list] = power_alloc_DC.reshape(n_cluster_optimizer)
                total_rate_after_DC = objective_value(channel_alloc_optimizer, power_alloc_DC,
                                priority_optimizer, h_DC, B, noise_vec_cg,
                                SNR_gap_optimizer)
                total_rate_after_DC_round = objective_value(channel_alloc_optimizer, power_alloc_DC_round,
                                priority_optimizer, h_DC, B, noise_vec_cg,
                                SNR_gap_optimizer)
                print('total rate after DC:', total_rate_after_DC)
            else:
                # print('Power allocation fail, remove the last cluster ', cluster_list[-1],  ' from current channel.')
                cluster_list = cluster_list[:-1]
                SUCCESS_INDICATOR_DC = False
                continue

        n_fsb_clusters_DC=0

        '''
        print individual user's power and rate
        '''
        if not silent:
            print('Power allocation (DC):')
        if len(cluster_list) > 0:
            for i in range(n_cluster_optimizer):
                rate = capacity_SU(power_alloc_DC[i], i, channel_alloc_optimizer, power_alloc_DC,
                                priority_optimizer, h_DC, B, noise_vec_cg,
                                SNR_gap_optimizer)
                if rate >= QoS[0] / (10 ** 6):
                    n_fsb_clusters_DC += 1
                if not silent:
                    print('Cluster %d: Power = %.2f mW. Rate = %.2f Mbps' % (cluster_list[i],power_alloc_DC[i][0],rate))

        idx_ch += 1



    if SUCCESS_INDICATOR_DC:
        # print('Power allocation succeed.')
        pass

    file_path = '.\\saved_results\\'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    np.save(file_path + 'power_alloc_DC', power_alloc_DC_record)
    np.save(file_path + 'channel_alloc.npy', channel_alloc_type1)

    SUCCESS_INDICATOR = SUCCESS_INDICATOR_CA and SUCCESS_INDICATOR_DC
    out_cluster_IDs, out_1st_channel_idx, out_num_channels, out_power = translate_to_std_output(channel_alloc_type1,
                                                                                                cg.channel_groups,
                                                                                                cluster_ID,
                                                                                                power_alloc_DC_record,
                                                                                                sort_output=True)
    _, _, _, out_power_GA = translate_to_std_output(channel_alloc_type1,
                                                    cg.channel_groups,
                                                    cluster_ID,
                                                    power_alloc_GA_record,
                                                    sort_output=True)

    return out_cluster_IDs, out_1st_channel_idx, out_num_channels, out_power, SUCCESS_INDICATOR, noise_vec, cluster_infeasible_IDs, out_power_GA,\
    converge_i, GA_time, GA_converge_time, DC_time, total_rate_GA, total_rate_after_DC, total_rate_after_DC_round, GA_feasible_number,\
           AGA_feasible_number, AGA_time, total_rate_AGA, AGA_converge_time, AGA_converge_i, GA_new_time, GAnew_converge_time, total_rate_GA_new,\
           n_fsb_clusters_GA_new,min_rate_adjusted_GA_new,rate_nfsb_clusters_newGA, n_min_rate_adjusted, n_fsb_clusters_DC, total_rate_1d


