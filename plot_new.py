import numpy as np
import copy
import matplotlib.pyplot as plt

# 0	GA_converge_i
# 1	 AGA_converge_i
# 2	 GA_time
# 3	 AGA_time
# 4	GA_converge_time
# 5	 AGA_converge_time
# 6	 DC_time
# 7	 total_rate_GA
# 8	 total_rate_after_DC
# 9	 total_rate_after_DC_round
# 10	total_rate_AGA
# 11	 GA_feasible_number
# 12	AGA_feasible_number
# 13	 GA_new_time
# 14	 GAnew_converge_time
# 15	 total_rate_GA_new
# 16	 n_fsb_clusters_GA_new
# 17	 n_min_rate_adjusted
# 18	n_fsb_clusters_DC
# 19	rate_1d


GA = True
AGA = True
GA_new = False
DC = True
oneD_opt = True


result_final = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\test.npy')

result_final_avg = np.mean(result_final, axis = 1)
result_final_std = np.std(result_final, axis = 1)

n_cluster_set = range(20,101,20)

plt.figure()
if GA:
    plt.plot(n_cluster_set, result_final_avg[:,7], 'o-', label='GA rate')
    plt.fill_between(n_cluster_set, result_final_avg[:,7] -result_final_std[:,7], result_final_avg[:,7] + result_final_std[:,7], alpha=0.2)
if AGA:
    plt.plot(n_cluster_set, result_final_avg[:,10], 'o-', label='AGA rate')
    plt.fill_between(n_cluster_set, result_final_avg[:,10] -result_final_std[:,10], result_final_avg[:,10] + result_final_std[:,10], alpha=0.2)
if GA_new:
    plt.plot(n_cluster_set, result_final_avg[:,15], 'o-', label='GA_new rate')
    plt.fill_between(n_cluster_set, result_final_avg[:,15] -result_final_std[:,15], result_final_avg[:,15] + result_final_std[:,15], alpha=0.2)
if DC:
    plt.plot(n_cluster_set, result_final_avg[:,9], 'o-', label='DC_1d Rate Quantized')
    plt.fill_between(n_cluster_set, result_final_avg[:,9] -result_final_std[:,9], result_final_avg[:,9] + result_final_std[:,9], alpha=0.2)
if oneD_opt:
    plt.plot(n_cluster_set, result_final_avg[:,19], 'o-', label='1d power Rate')
    plt.fill_between(n_cluster_set, result_final_avg[:,19] -result_final_std[:,19], result_final_avg[:,19] + result_final_std[:,19], alpha=0.2)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Mbps')
plt.title('Network Sum Rate')

plt.figure()
if GA:
    plt.plot(n_cluster_set, result_final_avg[:,2], 'o-', label='GA time')
    plt.fill_between(n_cluster_set, result_final_avg[:,2] -result_final_std[:,2], result_final_avg[:,2] + result_final_std[:,2], alpha=0.2)
    plt.plot(n_cluster_set, result_final_avg[:,4] + result_final_avg[:,2]/100, 'o-', label='GA converge time')
    plt.fill_between(n_cluster_set, result_final_avg[:,4] -result_final_std[:,4], result_final_avg[:,4] + result_final_std[:,4], alpha=0.2)
if AGA:
    plt.plot(n_cluster_set, result_final_avg[:,3], 'o-', label='AGA time')
    plt.fill_between(n_cluster_set, result_final_avg[:,3] -result_final_std[:,3], result_final_avg[:,3] + result_final_std[:,3], alpha=0.2)
    plt.plot(n_cluster_set, result_final_avg[:,5]+result_final_avg[:,3]/100, 'o-', label='AGA converge time')
    plt.fill_between(n_cluster_set, result_final_avg[:,5] -result_final_std[:,5], result_final_avg[:,5] + result_final_std[:,5], alpha=0.2)
if GA_new:
    plt.plot(n_cluster_set, result_final_avg[:,13], 'o-', label='GA_new time')
    plt.fill_between(n_cluster_set, result_final_avg[:,13] -result_final_std[:,13], result_final_avg[:,13] + result_final_std[:,13], alpha=0.2)
    plt.plot(n_cluster_set, result_final_avg[:,14]+result_final_avg[:,13]/100, 'o-', label='GA_new converge time')
    plt.fill_between(n_cluster_set, result_final_avg[:,14] -result_final_std[:,14], result_final_avg[:,14] + result_final_std[:,14], alpha=0.2)
if DC:
    plt.plot(n_cluster_set, result_final_avg[:,6], 'o-', label='DC time')
    plt.fill_between(n_cluster_set, result_final_avg[:,6] -result_final_std[:,6], result_final_avg[:,6] + result_final_std[:,6], alpha=0.2)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Seconds')
plt.title('Time')

plt.figure()
if GA:
    plt.plot(n_cluster_set, result_final_avg[:,11], 'o-', label='GA feasible number')
    plt.fill_between(n_cluster_set, result_final_avg[:,11] -result_final_std[:,11], result_final_avg[:,11] + result_final_std[:,11], alpha=0.2)
if AGA:
    plt.plot(n_cluster_set, result_final_avg[:,12], 'o-', label='AGA feasible number')
    plt.fill_between(n_cluster_set, result_final_avg[:,12] -result_final_std[:,12], result_final_avg[:,12] + result_final_std[:,12], alpha=0.2)
if GA_new:
    plt.plot(n_cluster_set, result_final_avg[:,16], 'o-', label='GA_new feasible number')
    plt.fill_between(n_cluster_set, result_final_avg[:,16] -result_final_std[:,16], result_final_avg[:,16] + result_final_std[:,16], alpha=0.2)
if DC:
    plt.plot(n_cluster_set, result_final_avg[:,18], 'o-', label='DC feasible number')
    plt.fill_between(n_cluster_set, result_final_avg[:,18] -result_final_std[:,18], result_final_avg[:,18] + result_final_std[:,18], alpha=0.2)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.title('Number of feasible clusters')

# plt.figure()
# plt.plot(n_cluster_set, result_final_avg[:,-1], 'o-', label='GA_new number of relaxed min rate')
# plt.fill_between(n_cluster_set, result_final_avg[:,-1] -result_final_std[:,-1], result_final_avg[:,-1] + result_final_std[:,-1], alpha=0.2)
# plt.legend(loc=0)
# plt.grid()
# plt.xlabel('Number of clusters')
# # plt.ylabel('Mbps')
# plt.title('Number of relaxed min rate')

plt.show()

print('done')
