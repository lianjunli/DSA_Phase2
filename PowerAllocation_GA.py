from genetic_algorithm import *
from HelperFunc_CPO import capacity_SU

def PA_GA(h, B, noise_vec_cg, SU_power_optimizer, QoS_optimizer,SNR_gap_optimizer, channel_alloc_optimizer, priority_optimizer):
    n_cluster = h.shape[0]
    max_power = SU_power_optimizer[0]
    min_rate = QoS_optimizer[0]

    n_power_level= 100
    popSize = 100
    generations = 100
    eliteSize = 20
    mutationRate = 0.1

    fit_hist = []
    power_options = np.arange(0,max_power+1,max_power/n_power_level)

    pop = initialPopulation(popSize, n_cluster, power_options)

    for i_gen in range(generations):
        n_succ = np.zeros(popSize)
        throughput = np.zeros(popSize)

        for i in range(popSize):
            rate = []
            power_all = np.asarray(pop[i])
            power_all = power_all.reshape((-1,1))
            for j in range (n_cluster):
                rate_cluster = capacity_SU(power_all[j,:], j, channel_alloc_optimizer, power_all,
                                priority_optimizer, h, B, noise_vec_cg,
                                SNR_gap_optimizer)
                rate.append(rate_cluster)
            n_succ[i] = np.sum(np.array(rate) >= min_rate / (10**6))
            throughput[i] = np.sum(rate)

        pop = nextGeneration(pop, eliteSize, mutationRate, n_succ, throughput, power_options)

        fit_hist.append(rankAllocs(pop, n_succ, throughput)[0][1])

    bestAllocIndex = rankAllocs(pop, n_succ, throughput)[0][0]
    bestAlloc = pop[bestAllocIndex]
    converge_n = generations
    for n in range(len(fit_hist)):
        if abs(fit_hist[-1] - fit_hist[n] < 0.01):
            converge_n = n
            break
    # print('converge generation,', converge_n)

    # plt.plot(fit_hist)
    # plt.show()

    return bestAlloc, converge_n, generations