from HelperFunc_CPO import capacity_SU
import numpy as np

def PA_1dOpt(h, B, noise_vec_cg, SU_power_optimizer, QoS_optimizer,SNR_gap_optimizer, channel_alloc_optimizer, priority_optimizer, fsb_power):
    n_cluster = h.shape[0]
    max_power = SU_power_optimizer
    min_rate = QoS_optimizer
    fsb_power = fsb_power.T
    fsb_power = fsb_power[0]
    scaled_fsb_power = np.min(max_power / fsb_power) * fsb_power

    n_power_level = 100
    unit_power = max_power / n_power_level

    power_vec = scaled_fsb_power
    # power_vec = power_vec.reshape((-1, 1))
    rate = []
    for j in range(n_cluster):
        rate_cluster = capacity_SU(power_vec.reshape((-1, 1))[j, :], j, channel_alloc_optimizer, power_vec.reshape((-1, 1)),
                                    h, B, noise_vec_cg,
                                   SNR_gap_optimizer)
        rate.append(rate_cluster)
    best_n_succ = np.sum(np.array(rate) >= min_rate / (10**6) - 0.001)
    best_throughput = np.sum(rate)
    while True:
        power_vec = power_vec + unit_power
        indices_rounddown = ((power_vec - max_power) > 0)
        power_vec[indices_rounddown] = power_vec[indices_rounddown] - unit_power[indices_rounddown]

        rate = []
        for j in range(n_cluster):
            rate_cluster = capacity_SU(power_vec.reshape((-1, 1))[j, :], j, channel_alloc_optimizer, power_vec.reshape((-1, 1)),
                                     h, B, noise_vec_cg,
                                       SNR_gap_optimizer)
            rate.append(rate_cluster)
        n_succ = np.sum(np.array(rate) >= min_rate / (10**6) - 0.001)
        throughput = np.sum(rate)

        if n_succ < best_n_succ or throughput < best_throughput or np.sum(indices_rounddown) == n_cluster:
            break
        else:
            best_n_succ = n_succ
            best_throughput = throughput

    OneDim_opt_power = power_vec

    return list(OneDim_opt_power)