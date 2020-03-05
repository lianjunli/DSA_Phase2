import numpy as np
import copy
import matplotlib.pyplot as plt

result_orig_1200 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\DC_orig_1200.npy')
result_dc_1200_orig_avg = np.mean(result_orig_1200, axis = 1)
result_dc_1200_orig_std = np.std(result_orig_1200, axis = 1)

result_order_1200 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\DC_order_1200.npy')
result_dc_1200_order_avg = np.mean(result_order_1200, axis = 1)
result_dc_1200_order_std = np.std(result_order_1200, axis = 1)

result_R_order_1200 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\DC_R_order_1200.npy')
result_dc_1200_R_order_avg = np.mean(result_R_order_1200, axis = 1)
result_dc_1200_R_order_std = np.std(result_R_order_1200, axis = 1)

result_orig_2000 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\DC_orig_2000.npy')
result_dc_2000_orig_avg = np.mean(result_orig_2000, axis = 1)
result_dc_2000_orig_std = np.std(result_orig_2000, axis = 1)

result_order_2000 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\DC_order_2000.npy')
result_dc_2000_order_avg = np.mean(result_order_2000, axis = 1)
result_dc_2000_order_std = np.std(result_order_2000, axis = 1)

result_R_order_2000 = np.load('D:\\VTProjects\\DSA_phase2 - GAPower\\saved_results\\DC_R_order_2000.npy')
result_dc_2000_R_order_avg = np.mean(result_R_order_2000, axis = 1)
result_dc_2000_R_order_std = np.std(result_R_order_2000, axis = 1)

n_cluster_set = range(20,101,20)

plt.figure()
plt.plot(n_cluster_set, result_dc_1200_orig_avg[:, 9], 'o-', label='DC_orig_1200')
plt.fill_between(n_cluster_set, result_dc_1200_orig_avg[:, 9] - result_dc_1200_orig_std[:, 9], result_dc_1200_orig_avg[:, 9] + result_dc_1200_orig_std[:, 9], alpha=0.2)
plt.plot(n_cluster_set, result_dc_1200_order_avg[:, 9], 'o-', label='DC_order_1200')
plt.fill_between(n_cluster_set, result_dc_1200_order_avg[:, 9] - result_dc_1200_order_std[:, 9], result_dc_1200_order_avg[:, 9] + result_dc_1200_order_std[:, 9], alpha=0.2)
plt.plot(n_cluster_set, result_dc_1200_R_order_avg[:, 9], 'o-', label='DC_R_order_1200')
plt.fill_between(n_cluster_set, result_dc_1200_R_order_avg[:, 9] - result_dc_1200_R_order_std[:, 9], result_dc_1200_R_order_avg[:, 9] + result_dc_1200_R_order_std[:, 9], alpha=0.2)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Mbps')
plt.title('Network Sum Rate 1200')


plt.figure()
plt.plot(n_cluster_set, result_dc_2000_orig_avg[:, 9], 'o-', label='DC_orig_2000')
plt.fill_between(n_cluster_set, result_dc_2000_orig_avg[:, 9] - result_dc_2000_orig_std[:, 9], result_dc_2000_orig_avg[:, 9] + result_dc_2000_orig_std[:, 9], alpha=0.2)
plt.plot(n_cluster_set, result_dc_2000_order_avg[:, 9], 'o-', label='DC_order_2000')
plt.fill_between(n_cluster_set, result_dc_2000_order_avg[:, 9] - result_dc_2000_order_std[:, 9], result_dc_2000_order_avg[:, 9] + result_dc_2000_order_std[:, 9], alpha=0.2)
plt.plot(n_cluster_set, result_dc_2000_R_order_avg[:, 9], 'o-', label='DC_R_order_2000')
plt.fill_between(n_cluster_set, result_dc_2000_R_order_avg[:, 9] - result_dc_2000_R_order_std[:, 9], result_dc_2000_R_order_avg[:, 9] + result_dc_2000_R_order_std[:, 9], alpha=0.2)

plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Mbps')
plt.title('Network Sum Rate 2000')

plt.figure()
plt.plot(n_cluster_set, result_dc_1200_orig_avg[:, 6], 'o-', label='DC_orig_1200')
plt.fill_between(n_cluster_set, result_dc_1200_orig_avg[:, 6] - result_dc_1200_orig_std[:, 6], result_dc_1200_orig_avg[:, 6] + result_dc_1200_orig_std[:, 6], alpha=0.2)
plt.plot(n_cluster_set, result_dc_1200_order_avg[:, 6], 'o-', label='DC_order_1200')
plt.fill_between(n_cluster_set, result_dc_1200_order_avg[:, 6] - result_dc_1200_order_std[:, 6], result_dc_1200_order_avg[:, 6] + result_dc_1200_order_std[:, 6], alpha=0.2)
plt.plot(n_cluster_set, result_dc_1200_R_order_avg[:, 6], 'o-', label='DC_R_order_1200')
plt.fill_between(n_cluster_set, result_dc_1200_R_order_avg[:, 6] - result_dc_1200_R_order_std[:, 6], result_dc_1200_R_order_avg[:, 6] + result_dc_1200_R_order_std[:, 6], alpha=0.2)

plt.plot(n_cluster_set, result_dc_2000_orig_avg[:, 6], 'o-', label='DC_orig_2000')
plt.fill_between(n_cluster_set, result_dc_2000_orig_avg[:, 6] - result_dc_2000_orig_std[:, 6], result_dc_2000_orig_avg[:, 6] + result_dc_2000_orig_std[:, 6], alpha=0.2)
plt.plot(n_cluster_set, result_dc_2000_order_avg[:, 6], 'o-', label='DC_order_2000')
plt.fill_between(n_cluster_set, result_dc_2000_order_avg[:, 6] - result_dc_2000_order_std[:, 6], result_dc_2000_order_avg[:, 6] + result_dc_2000_order_std[:, 6], alpha=0.2)
plt.plot(n_cluster_set, result_dc_2000_R_order_avg[:, 6], 'o-', label='DC_R_order_2000')
plt.fill_between(n_cluster_set, result_dc_2000_R_order_avg[:, 6] - result_dc_2000_R_order_std[:, 6], result_dc_2000_R_order_avg[:, 6] + result_dc_2000_R_order_std[:, 6], alpha=0.2)

plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Seconds')
plt.title('Time')

plt.figure()
plt.plot(n_cluster_set, result_dc_1200_orig_avg[:, 18], 'o-', label='DC_orig_1200')
plt.fill_between(n_cluster_set, result_dc_1200_orig_avg[:, 18] - result_dc_1200_orig_std[:, 18], result_dc_1200_orig_avg[:, 18] + result_dc_1200_orig_std[:, 18], alpha=0.2)
plt.plot(n_cluster_set, result_dc_1200_order_avg[:, 18], 'o-', label='DC_order_1200')
plt.fill_between(n_cluster_set, result_dc_1200_order_avg[:, 18] - result_dc_1200_order_std[:, 18], result_dc_1200_order_avg[:, 18] + result_dc_1200_order_std[:, 18], alpha=0.2)
plt.plot(n_cluster_set, result_dc_1200_R_order_avg[:, 18], 'o-', label='DC_R_order_1200')
plt.fill_between(n_cluster_set, result_dc_1200_R_order_avg[:, 18] - result_dc_1200_R_order_std[:, 18], result_dc_1200_R_order_avg[:, 18] + result_dc_1200_R_order_std[:, 18], alpha=0.2)

plt.plot(n_cluster_set, result_dc_2000_orig_avg[:, 18], 'o-', label='DC_orig_2000')
plt.fill_between(n_cluster_set, result_dc_2000_orig_avg[:, 18] - result_dc_2000_orig_std[:, 18], result_dc_2000_orig_avg[:, 18] + result_dc_2000_orig_std[:, 18], alpha=0.2)
plt.plot(n_cluster_set, result_dc_2000_order_avg[:, 18], 'o-', label='DC_order_2000')
plt.fill_between(n_cluster_set, result_dc_2000_order_avg[:, 18] - result_dc_2000_order_std[:, 18], result_dc_2000_order_avg[:, 18] + result_dc_2000_order_std[:, 18], alpha=0.2)
plt.plot(n_cluster_set, result_dc_2000_R_order_avg[:, 18], 'o-', label='DC_R_order_2000')
plt.fill_between(n_cluster_set, result_dc_2000_R_order_avg[:, 18] - result_dc_2000_R_order_std[:, 18], result_dc_2000_R_order_avg[:, 18] + result_dc_2000_R_order_std[:, 18], alpha=0.2)

plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of clusters')
plt.title('Number of feasible clusters')


plt.show()

print('done')