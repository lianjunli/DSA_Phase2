import numpy as np
import copy
import matplotlib.pyplot as plt
#[GA_converge_i, GA_time, GA_converge_time, DC_time, total_rate_GA, total_rate_after_DC, total_rate_after_DC_round]
# result = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_power_results_GA.npy')
# result_quantize = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_power_results_QuantizeDC.npy')
# result_final = copy.deepcopy(result_quantize)
# result_final[:,:,[0, 1, 2, 4]] = result[:,:, [5, 0, 1, 3]]

result_final = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_new_power_results_2000_part1.npy')
result_final_pre = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_new_power_results_2000_part2.npy')
result_final[-2:,:,:]=result_final_pre
result_DC = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_new_power_results_2000_DC.npy')
result_DC = result_DC.squeeze()
result_DC_1d = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\1d_2000.npy')
result_DC_1d = result_DC_1d.squeeze()


# result_1 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_power_results_1200.npy')
# result_2 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_power_results_1200_GA_new.npy')
# result_3 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\GA_power_results_1200_DC_GA_new.npy')
# result_final = result_2
# result_final[:,:,:-6] = result_1
# result_DC = result_3.squeeze()
# result_DC_1d = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\1d_1200.npy')
# result_DC_1d = result_DC_1d.squeeze()

result_final_avg = np.mean(result_final, axis = 1)
result_final_std = np.std(result_final, axis = 1)

n_cluster_set = range(20,101,20)

plt.figure()
plt.plot(n_cluster_set, result_final_avg[:,7], 'o-', label='GA rate')
plt.fill_between(n_cluster_set, result_final_avg[:,7] -result_final_std[:,7], result_final_avg[:,7] + result_final_std[:,7], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,10], 'o-', label='AGA rate')
plt.fill_between(n_cluster_set, result_final_avg[:,10] -result_final_std[:,10], result_final_avg[:,10] + result_final_std[:,10], alpha=0.2)
# plt.plot(n_cluster_set, result_final_avg[:,-4], 'o-', label='GA_new rate')
# plt.fill_between(n_cluster_set, result_final_avg[:,-4] -result_final_std[:,-4], result_final_avg[:,-4] + result_final_std[:,-4], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,-3], 'o-', label='GA_new rate')
plt.fill_between(n_cluster_set, result_final_avg[:,-3] -result_final_std[:,-3], result_final_avg[:,-3] + result_final_std[:,-3], alpha=0.2)

plt.plot(n_cluster_set, result_DC[:,9], 'o-', label='DC quantized rate')
plt.plot(n_cluster_set, result_DC_1d[:,9], 'o-', label='DC_1d Rate')
plt.plot(n_cluster_set, result_DC_1d[:,-1], 'o-', label='1d power Rate')
# plt.fill_between(n_cluster_set, result_final_avg[:,6] -result_final_std[:,6], result_final_avg[:,6] + result_final_std[:,6], alpha=0.2)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Mbps')
plt.title('Network Sum Rate')

plt.figure()
plt.plot(n_cluster_set, result_final_avg[:,2], 'o-', label='GA time')
plt.fill_between(n_cluster_set, result_final_avg[:,2] -result_final_std[:,2], result_final_avg[:,2] + result_final_std[:,2], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,4] + result_final_avg[:,2]/100, 'o-', label='GA converge time')
plt.fill_between(n_cluster_set, result_final_avg[:,4] -result_final_std[:,4], result_final_avg[:,4] + result_final_std[:,4], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,3], 'o-', label='AGA time')
plt.fill_between(n_cluster_set, result_final_avg[:,3] -result_final_std[:,3], result_final_avg[:,3] + result_final_std[:,3], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,5]+result_final_avg[:,3]/100, 'o-', label='AGA converge time')
plt.fill_between(n_cluster_set, result_final_avg[:,5] -result_final_std[:,5], result_final_avg[:,5] + result_final_std[:,5], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,-6], 'o-', label='GA_new time')
plt.fill_between(n_cluster_set, result_final_avg[:,-6] -result_final_std[:,-6], result_final_avg[:,-6] + result_final_std[:,-6], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,-5]+result_final_avg[:,-6]/100, 'o-', label='GA_new converge time')
plt.fill_between(n_cluster_set, result_final_avg[:,-5] -result_final_std[:,-5], result_final_avg[:,-5] + result_final_std[:,-5], alpha=0.2)
plt.plot(n_cluster_set, result_DC[:,6], 'o-', label='DC time')
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Seconds')
plt.title('Time')

plt.figure()
plt.plot(n_cluster_set, result_final_avg[:,-8], 'o-', label='GA feasible number')
plt.fill_between(n_cluster_set, result_final_avg[:,-8] -result_final_std[:,-8], result_final_avg[:,-8] + result_final_std[:,-8], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,-7], 'o-', label='AGA feasible number')
plt.fill_between(n_cluster_set, result_final_avg[:,-7] -result_final_std[:,-7], result_final_avg[:,-7] + result_final_std[:,-7], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,-3], 'o-', label='GA_new feasible number')
plt.fill_between(n_cluster_set, result_final_avg[:,-3] -result_final_std[:,-2], result_final_avg[:,-3] + result_final_std[:,-3], alpha=0.2)
plt.plot(n_cluster_set, result_final_avg[:,-3], 'o-', label='DC feasible number')
# plt.plot(n_cluster_set, result_final_avg[:,6], 'o-', label='DC quantized rate')
# plt.fill_between(n_cluster_set, result_final_avg[:,6] -result_final_std[:,6], result_final_avg[:,6] + result_final_std[:,6], alpha=0.2)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
# plt.ylabel('Mbps')
plt.title('Number of feasible clusters')

plt.figure()
plt.plot(n_cluster_set, result_final_avg[:,-1], 'o-', label='GA_new number of relaxed min rate')
plt.fill_between(n_cluster_set, result_final_avg[:,-1] -result_final_std[:,-1], result_final_avg[:,-1] + result_final_std[:,-1], alpha=0.2)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
# plt.ylabel('Mbps')
plt.title('Number of relaxed min rate')

plt.show()

print('done')
