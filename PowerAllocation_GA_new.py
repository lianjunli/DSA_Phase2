from GA_new import *
from HelperFunc_CPO import capacity_SU

def PA_GA_new(h, B, noise_vec_cg, SU_power_optimizer, QoS_optimizer,SNR_gap_optimizer, channel_alloc_optimizer, priority_optimizer, fsb_power):
    n_cluster = h.shape[0]
    max_power = SU_power_optimizer[0]
    min_rate = QoS_optimizer[0]
    fsb_power = fsb_power.T
    fsb_power = fsb_power[0]
    scaled_fsb_power = max_power / np.max(fsb_power) * fsb_power

    n_power_level= 100
    unit_power = max_power / n_power_level
    popSize = 100
    generations = 100
    eliteSize = 20
    mutationRate = 0.2

    max_pl = np.round(max_power / unit_power) * np.ones(n_cluster, dtype=np.int32)
    fsb_pl = np.floor(fsb_power / unit_power)
    scaled_fsb_pl = np.ceil((scaled_fsb_power / unit_power))
    scaled_fsb_pl[scaled_fsb_pl > max_pl] = max_pl[scaled_fsb_pl > max_pl]

    min_rate_adjusted = []
    power_all = np.asarray(scaled_fsb_pl)
    power_all = power_all.reshape((-1, 1)) * unit_power
    for j in range(n_cluster):
        rate_cluster = capacity_SU(power_all[j, :], j, channel_alloc_optimizer, power_all,
                                   priority_optimizer, h, B, noise_vec_cg,
                                   SNR_gap_optimizer)
        min_rate_adjusted.append(rate_cluster)
    min_rate_adjusted = np.array(min_rate_adjusted)
    min_rate_adjusted[np.where(min_rate_adjusted >= min_rate / (10 ** 6))[0]] = min_rate / (10 ** 6)
    min_rate_adjusted = min_rate_adjusted
    n_min_rate_adjusted = np.sum(min_rate_adjusted < min_rate / (10 ** 6))

    pl_vec = scaled_fsb_pl
    rate = []
    power_all = np.asarray(pl_vec)
    power_all = power_all.reshape((-1, 1)) * unit_power
    for j in range(n_cluster):
        rate_cluster = capacity_SU(power_all[j, :], j, channel_alloc_optimizer, power_all,
                                   priority_optimizer, h, B, noise_vec_cg,
                                   SNR_gap_optimizer)
        rate.append(rate_cluster)
    best_n_succ = np.sum(np.array(rate) >= min_rate_adjusted)
    best_throughput = np.sum(rate)
    debug_counter=0
    while True:
        # print('GA_new while loop',debug_counter)
        debug_counter+=1
        pl_vec = pl_vec + 1
        indices_rounddown = ((pl_vec - max_pl) > 0)
        pl_vec[indices_rounddown] = pl_vec[indices_rounddown] - 1

        rate = []
        pl_all = np.asarray(pl_vec)
        pl_all = pl_all.reshape((-1, 1)) * unit_power
        for j in range(n_cluster):
            rate_cluster = capacity_SU(power_all[j, :], j, channel_alloc_optimizer, power_all,
                                       priority_optimizer, h, B, noise_vec_cg,
                                       SNR_gap_optimizer)
            rate.append(rate_cluster)
        n_succ = np.sum(np.array(rate) >= min_rate_adjusted)
        throughput = np.sum(rate)
        # print('power ', pl_vec)
        if n_succ < best_n_succ or throughput < best_throughput or np.sum(pl_vec - max_pl) == 0:
            break
        else:
            pl_vec = pl_vec
            best_n_succ = n_succ
            best_throughput = throughput

    OneDim_opt_pl = pl_vec

    fit_hist = []

    pop = initialPopulation(popSize, n_cluster, fsb_pl, scaled_fsb_pl, OneDim_opt_pl, max_pl)

    for i_gen in range(generations):
        n_succ = np.zeros(popSize)
        throughput = np.zeros(popSize)

        for i in range(popSize):
            rate = []
            power_all = np.asarray(pop[i])
            power_all = power_all.reshape((-1,1)) * unit_power
            for j in range (n_cluster):
                rate_cluster = capacity_SU(power_all[j,:], j, channel_alloc_optimizer, power_all,
                                priority_optimizer, h, B, noise_vec_cg,
                                SNR_gap_optimizer)
                rate.append(rate_cluster)
            n_succ[i] = np.sum(np.array(rate) >= min_rate_adjusted)
            throughput[i] = np.sum(rate)

        pop = nextGeneration(pop, eliteSize, n_succ, throughput, n_cluster, mutationRate, fsb_pl, max_pl)

        fit_hist.append(rankAllocs(pop, n_succ, throughput)[0][1])

    bestAllocIndex = rankAllocs(pop, n_succ, throughput)[0][0]
    bestAlloc = pop[bestAllocIndex]
    bestAlloc = np.asarray(bestAlloc) * unit_power
    bestAlloc = list(bestAlloc)
    # plt.plot(fit_hist)
    # # plt.show()
    # plt.savefig('GA_curve')

    np.save('fit_AGA.npy', fit_hist)

    # Record rate for each cluster
    final_rate = []

    power_all = np.asarray(bestAlloc)
    power_all = power_all.reshape((-1, 1))
    for j in range(n_cluster):
        rate_cluster = capacity_SU(power_all[j, :], j, channel_alloc_optimizer, power_all,
                                   priority_optimizer, h, B, noise_vec_cg,
                                   SNR_gap_optimizer)
        final_rate.append(rate_cluster)
    np.save('rate_AGA_100.npy', final_rate)
    converge_n = generations
    for n in range(len(fit_hist)):
        if abs(fit_hist[-1] - fit_hist[n] < 0.01):
            converge_n = n
            break

    return bestAlloc, min_rate_adjusted, generations, converge_n, n_min_rate_adjusted