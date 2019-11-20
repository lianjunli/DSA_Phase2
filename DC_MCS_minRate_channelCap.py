import numpy as np
import copy
import networkx as nx
from itertools import combinations
from HelperFunc_CPO import objective_value_SU, capacity_SU, objective_value
import time
import cdd
import os

'''
This is the method of 
(1) restrict 𝑀^(𝑖) in the feasible region of node
(2) add edge points of node i's feasible region into 𝑀^(𝑖)
(3) Vertex enumeration method
(4) add minimum data rate constraints
𝑀^𝑖 is the set of points to approximate f(p), where the DC objective function is {max f(p) - g(p)}
'''

class PA_DC_MCS_minRate_channelCap:

    def __init__(self, power_alloc, channel_alloc, channel_gain, B, noise_vec, priority, SU_power,
                 minRate, SNR_gap, QAM_cap, channel_cap, objective_list, update_order, channel_gain_minR):
        self.power_alloc = copy.deepcopy(power_alloc)
        self.Succeed = False
        self.capacity_SU = np.zeros(power_alloc.shape[0])
        self.power_allocation(power_alloc, channel_alloc, channel_gain, B, noise_vec, priority, SU_power,
                              minRate, SNR_gap, QAM_cap, channel_cap, objective_list, update_order,channel_gain_minR)


    def max_distance(self, arr):
        max_distance = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                distance = np.linalg.norm(arr[i] - arr[j])

                if (distance > max_distance):
                    max_distance = distance
                    index = [i, j]
        return index

    def A(self, n, m, power_alloc, channel_alloc, channel_gain, noise_vec, SNR_gap):
        n_su = power_alloc.shape[0]
        inter_sum = 0
        for j in range(n_su):
            if (j != n):
                inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
        return channel_gain[n, n] / SNR_gap[n] / (inter_sum + noise_vec[n])

    def region_minRate(self, SU_index, gradient, minRate, power_alloc, channel_alloc, channel_gain, B, noise_vec, SNR_gap):
        n_channel = power_alloc.shape[1]
        p_dim = int(np.sum(channel_alloc[SU_index, :]))

        # All the elements of gradient should be >=0
        # If <=0, then multiplies with -1
        if (np.sum(gradient <= 0) == gradient.size):
            gradient = - gradient

        denominator = np.sum(channel_alloc[SU_index, :])
        numerator = - minRate[SU_index] / B
        index = 0
        for m in range(n_channel):
            if (channel_alloc[SU_index, m] == 1):
                a = self.A(SU_index, m, power_alloc, channel_alloc, channel_gain, noise_vec, SNR_gap)
                numerator = numerator + np.log2(a/gradient[index])
                index = index + 1

        k = 2 ** (numerator / denominator)

        p = np.zeros(p_dim)
        index = 0
        for m in range(n_channel):
            if (channel_alloc[SU_index, m] == 1):
                a = self.A(SU_index, m, power_alloc, channel_alloc, channel_gain, noise_vec, SNR_gap)
                p[index] = (a - k*gradient[index]) / (a*k*gradient[index])
                index = index + 1

        capacity = 0
        index = 0
        for m in range(n_channel):
            if (channel_alloc[SU_index, m] == 1):
                a = self.A(SU_index, m, power_alloc, channel_alloc, channel_gain, noise_vec, SNR_gap)
                capacity = capacity + B*np.log2(1+a*p[index])

                index = index + 1

        z = np.dot(gradient, p)

        return gradient, z, p

    def QAM_region(self, SU_index, power_alloc, channel_alloc, channel_gain, noise_vec, SNR_gap, QAM_cap):

        n = SU_index
        n_su = power_alloc.shape[0]
        n_channel = power_alloc.shape[1]
        QAM_max_power = np.zeros(n_channel)

        for m in range(n_channel):
            inter_sum = 0
            for j in range(n_su):
                if (j != n):
                    inter_sum = inter_sum + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]
            QAM_max_power[m] = (2**(QAM_cap[n]+1) - 1) * (inter_sum + noise_vec[n]) * (SNR_gap[n] / channel_gain[n, n])

        return QAM_max_power


    def power_allocation(self, power_alloc, channel_alloc, channel_gain, B, noise_vec, priority, SU_power,
                         minRate, SNR_gap, QAM_cap, channel_cap, objective_list, update_order,channel_gain_minR):

        n_su = power_alloc.shape[0]
        n_channel = power_alloc.shape[1]
        self.epsilon = 0.01
        self.n_search_node_total = 0

        tic = time.time()
        while (True):
            previous_power_alloc = copy.deepcopy(power_alloc)
            # When the user update restarts, initialize SU_max_power
            self.SU_max_power = copy.deepcopy(SU_power)

            for n in update_order:

                # If no channel is assigned to user n, skip the power update for user n
                if (np.sum(channel_alloc[n, :]) == 0):
                    power_alloc[n, :] = 0
                    objective_list['total'].append(
                        objective_value(channel_alloc, power_alloc, priority, channel_gain, B, noise_vec, SNR_gap))
                    continue

                '''
                Find all possible channel allocation for user n
                '''
                channel_index = list(np.where(channel_alloc[n,:]==1)[0])
                channel_index_comb = list(combinations(channel_index, channel_cap[n]))
                power_alloc_comb = [None] * len(channel_index_comb)
                capacity_comb = [None] * len(channel_index_comb)

                for index_iter in range(len(channel_index_comb)):

                    # Temporary channel allocation
                    tmp_channel_alloc = copy.deepcopy(channel_alloc)
                    tmp_channel_alloc[n, :] = 0
                    tmp_channel_alloc[n, channel_index_comb[index_iter]] = 1

                    # Add QAM capacity constraint for user n
                    self.QAM_max_power = self.QAM_region(n, power_alloc, tmp_channel_alloc, channel_gain, noise_vec, SNR_gap,
                                                         QAM_cap)

                    # Add minimum data rate constraint for user n
                    tmp_p_dim = int(np.sum(tmp_channel_alloc[n, :]))
                    self.minRate_h = np.zeros((1, tmp_p_dim))
                    self.minRate_b = np.zeros(1)
                    gradient = np.ones(tmp_p_dim)
                    h, b, p = self.region_minRate(n, gradient, minRate, power_alloc, tmp_channel_alloc, channel_gain_minR,
                                                  B, noise_vec, SNR_gap)
                    self.minRate_h[0, :] = -h
                    self.minRate_b[0] = -b


                    while (True):

                        # Initial maximum lower bound is -Infinity
                        self.lowerbound_global = -float("inf")

                        # Update the power of user n using DC programming (when a solution is found,
                        # self.power_alloc will be updated)
                        self.power_allocation_single_user(power_alloc, tmp_channel_alloc, channel_gain, B, noise_vec, priority, SNR_gap, n)
                        power_alloc_comb[index_iter] = copy.deepcopy(self.power_alloc)


                        '''
                        Check if user n meets the minimum data rate requirements
                        '''
                        self.capacity_SU[n] = capacity_SU(self.power_alloc[n, :], n, tmp_channel_alloc, self.power_alloc,
                                                          priority, channel_gain_minR, B, noise_vec, SNR_gap)
                        capacity_comb[index_iter] = self.capacity_SU[n]

                        if (self.capacity_SU[n] * (10**6) < minRate[n]):
                            # check if the user is possible to meet the minimum data rate
                            if(self.SU_max_power[n] < np.amin(self.minRate_b[-1] / self.minRate_h[-1, :])):
                                # print('The required data rate cannot be satisfied')
                                # os._exit(0)
                                return

                            # user n doesn't meet minimum data rate
                            gradient = np.zeros(tmp_p_dim)
                            grad_index = 0
                            for m in range(n_channel):
                                if (tmp_channel_alloc[n, m]==1):
                                    a = self.A(n, m, self.power_alloc, tmp_channel_alloc, channel_gain_minR, noise_vec, SNR_gap)
                                    gradient[grad_index] = a/(1+a*self.power_alloc[n, m])
                                    grad_index = grad_index + 1

                            h, b, p = self.region_minRate(n, gradient, minRate, self.power_alloc, tmp_channel_alloc,
                                                          channel_gain_minR, B, noise_vec, SNR_gap)

                            if ((minRate[n] - self.capacity_SU[n]* (10 ** 6)) / minRate[n] < 0.01):
                                # The difference with minimum data rate is within a threshold
                                tmp_p = np.zeros(n_channel)
                                tmp_p[np.where(tmp_channel_alloc[n,:]==1)] = p
                                self.power_alloc[n, :] = tmp_p
                                self.capacity_SU[n] = capacity_SU(self.power_alloc[n, :], n, tmp_channel_alloc,
                                                                  self.power_alloc, priority, channel_gain_minR, B, noise_vec, SNR_gap)
                                capacity_comb[index_iter] = self.capacity_SU[n]
                            else:
                                # Add more linear constraints to upper bound the convex set
                                self.minRate_h = np.vstack((self.minRate_h, -h))
                                self.minRate_b = np.append(self.minRate_b, -b)
                                print('minRate is not satisfied for %.2f Mbps' % (minRate[n]/(10**6) - self.capacity_SU[n]))
                                continue

                        '''
                        Check if other users meet the minimum data rate requirements
                        '''
                        unsatisfied_user = np.zeros(n_su)
                        for j in range(n_su):
                            if (j != n and np.sum(channel_alloc[j, :]) > 0):
                                gradient = np.ones(int(np.sum(channel_alloc[j, :])))
                                h, b, p = self.region_minRate(j, gradient, minRate, self.power_alloc, tmp_channel_alloc, channel_gain_minR, B, noise_vec, SNR_gap)
                                # Impossible to find user j's power allocation to let user j's minRate is satisfied
                                if ( b > SU_power[j] ):
                                    unsatisfied_user[j] = 1
                                # The search region of user j's power allocation should include original power allocation
                                if ( b > np.sum(power_alloc[j,:])):
                                    unsatisfied_user[j] = 1

                        if (np.sum(unsatisfied_user) == 0):
                            break
                        else:
                            #print('Other users minRate cannot be satisfied for %.2f mW' % max_sumPowerDiff)
                            # print('Other users minRate cannot be satisfied')
                            self.SU_max_power[n] = max(-self.minRate_b[0], np.sum(self.power_alloc[n, :]) - 100)
                            if (self.SU_max_power[n] == -self.minRate_b[0]):
                                # print('The required data rate cannot be satisfied')
                                break
                                os._exit(0)


                # Take the power allocation with the maximum capacity
                max_capacity_index = capacity_comb.index(max(capacity_comb))
                self.power_alloc[n, :] = power_alloc_comb[max_capacity_index][n, :]
                power_alloc[n, :] = power_alloc_comb[max_capacity_index][n, :]
                objective_list['total'].append(
                    objective_value(channel_alloc, power_alloc, priority, channel_gain, B, noise_vec, SNR_gap))


            toc = time.time()
            if (toc-tic) > 5:
                # print('iteration converge timeout')
                self.Succeed = True # found solution
                self.power_alloc = self.power_alloc * SU_power[0] / np.max(self.power_alloc)
                break

            if (np.amax(np.absolute(power_alloc - previous_power_alloc)) < 1 ):
                # print('power allocation is updated')
                # print('number of search nodes = %d' % self.n_search_node_total)
                self.Succeed = True # found solution
                self.power_alloc = self.power_alloc * SU_power[0] / np.max(self.power_alloc)
                break

    def power_allocation_single_user(self, power_alloc, channel_alloc, channel_gain, B, noise_vec,
                                     priority, SNR_gap, SU_index):

        n = SU_index
        n_channel = power_alloc.shape[1]

        G = nx.Graph()
        node_index = 1
        G.add_node(node_index)

        avail_channel = np.where(channel_alloc[n, :] == 1)

        initial_vertex = [np.zeros(n_channel)]

        for m in range(avail_channel[0].size):
            tmp_vertex = np.zeros(n_channel)
            tmp_vertex[avail_channel[0][m]] = self.SU_max_power[n]
            initial_vertex.append(copy.deepcopy(tmp_vertex))

        G.node[node_index]['vertex'] = copy.deepcopy(initial_vertex)
        G.node[node_index]['index'] = node_index
        G.node[node_index]['vertex_total'] = copy.deepcopy(initial_vertex)

        queue = []
        queue.append(G.node[node_index])


        power_lb_global = np.zeros(n_channel)

        n_search_node = 0
        tic = time.clock()
        while (True):
            if len(queue) == 0 or n_search_node > 5000: # make sure DC won't search forever
                # print(objective_value_SU(
                #     power_alloc[n, :], n, channel_alloc, power_alloc, priority, channel_gain, env, SNR_gap))
                # print(self.lowerbound_global)

                toc = time.clock()
                # print('number of search nodes = %d' % n_search_node)
                # print('Processing Time = %.2f' % (toc - tic))

                self.power_alloc[n, :] = power_lb_global
                break
            n_search_node = n_search_node + 1
            self.n_search_node_total = self.n_search_node_total + 1

            node = queue.pop(0)
            node_index = node['index']

            # Check if the node is inside the feasible region
            if (self.check_feasible(G.node[node_index], channel_alloc, n) == False):
                G.remove_node(node_index)
                continue

            # Calculate the upperbound and power_ub
            G.node[node_index]['power_ub'], G.node[node_index]['upperbound'] = \
                self.upperbound(G.node[node_index], power_alloc, channel_alloc, channel_gain, B, noise_vec, priority, n, SNR_gap)

            # Update the vertex_lb
            G.node[node_index]['vertex_lb'] = []
            for v in G.node[node_index]['vertex']:
                tmp1 = np.matmul(self.minRate_h, v[np.where(channel_alloc[n, :] == 1)]) - self.minRate_b
                tmp2 = v[np.where(channel_alloc[n, :] == 1)] - self.QAM_max_power[np.where(channel_alloc[n, :] == 1)]
                if (np.sum(tmp1 < 0) == tmp1.size and np.sum(tmp2 < 0) == tmp2.size):
                    G.node[node_index]['vertex_lb'].append(copy.deepcopy(v))

            tmp1 = np.matmul(self.minRate_h,
                             G.node[node_index]['power_ub'][np.where(channel_alloc[n, :] == 1)]) - self.minRate_b
            if (np.sum(tmp1 < 0) == tmp1.size):
                G.node[node_index]['vertex_lb'].append(G.node[node_index]['power_ub'])

            # Calculate the lowerbound and power_lb
            if (len(G.node[node_index]['vertex_lb']) > 0):
                G.node[node_index]['power_lb'], G.node[node_index]['lowerbound'] = \
                    self.lowerbound(G.node[node_index], power_alloc, channel_alloc, channel_gain, B, noise_vec, priority, n,
                                    SNR_gap)
            else:
                G.node[node_index]['lowerbound'] = -float("inf")

            if (G.node[node_index]['lowerbound'] > self.lowerbound_global):
                self.lowerbound_global = G.node[node_index]['lowerbound']
                power_lb_global = copy.deepcopy(G.node[node_index]['power_lb'])

            # remove current node from graph G
            vertex_total_ancestor = copy.deepcopy(G.node[node_index]['vertex_total'])
            power_ub_ancestor = copy.deepcopy(G.node[node_index]['power_ub'])
            G.remove_node(node_index)

            if (node['upperbound'] - self.lowerbound_global < 0):
                continue
            if (node['upperbound'] - self.lowerbound_global <= self.epsilon and
                    node['upperbound'] - self.lowerbound_global >= 0):
                continue

            node_index1 = 2 * node_index
            node_index2 = 2 * node_index + 1
            vertex_index_maxDis = self.max_distance(node['vertex'])
            tmp_vertex_list = copy.deepcopy(node['vertex'])
            vertex1 = node['vertex'][vertex_index_maxDis[0]]
            vertex2 = node['vertex'][vertex_index_maxDis[1]]
            vertex3 = copy.deepcopy((vertex1 + vertex2) / 2)

            tmp_vertex_list1 = copy.deepcopy(tmp_vertex_list)
            del tmp_vertex_list1[vertex_index_maxDis[0]]
            tmp_vertex_list1.append(vertex3)

            tmp_vertex_list2 = copy.deepcopy(tmp_vertex_list)
            del tmp_vertex_list2[vertex_index_maxDis[1]]
            tmp_vertex_list2.append(vertex3)

            G.add_node(node_index1)
            G.add_node(node_index2)
            G.node[node_index1]['index'] = node_index1
            G.node[node_index2]['index'] = node_index2
            G.node[node_index1]['vertex'] = tmp_vertex_list1
            G.node[node_index2]['vertex'] = tmp_vertex_list2

            G.node[node_index1]['vertex_total'] = copy.deepcopy(vertex_total_ancestor)
            G.node[node_index2]['vertex_total'] = copy.deepcopy(vertex_total_ancestor)
            G.node[node_index1]['vertex_total'].append(copy.deepcopy(vertex3))
            G.node[node_index2]['vertex_total'].append(copy.deepcopy(vertex3))
            G.node[node_index1]['vertex_total'].append(copy.deepcopy(power_ub_ancestor))
            G.node[node_index2]['vertex_total'].append(copy.deepcopy(power_ub_ancestor))

            queue.append(G.node[node_index1])
            queue.append(G.node[node_index2])

    def upperbound(self, Node, power_alloc, channel_alloc, channel_gain, B, noise_vec, priority, SU_index, SNR_gap):

        vertex_list = Node['vertex']
        vertex_total = Node['vertex_total']
        n_su = power_alloc.shape[0]
        n_channel = power_alloc.shape[1]
        n = SU_index
        p_dim = int(np.sum(channel_alloc[SU_index, :]))

        # Define f, df, g, dg, ddg
        C = np.zeros(n_channel)
        for m in range(n_channel):
            for j in range(n_su):
                if (j != n):
                    C[m] = C[m] + channel_alloc[j, m] * channel_gain[n, j] * power_alloc[j, m]

        D = np.zeros((n_su, n_channel))
        for m in range(n_channel):
            for k in range(n_su):
                if (k != n):
                    for j in range(n_su):
                        if (j != n and j != k):
                            D[k, m] = D[k, m] + channel_alloc[j, m] * channel_gain[k, j] * power_alloc[j, m]

        E = np.zeros((n_su, n_channel))
        for m in range(n_channel):
            for k in range(n_su):
                if (k != n):
                    for j in range(n_su):
                        if (j == k):
                            E[k, m] = E[k, m] + channel_alloc[j, m] * channel_gain[k, j] / SNR_gap[k] * power_alloc[j, m]
                        elif (j != n):
                            E[k, m] = E[k, m] + channel_alloc[j, m] * channel_gain[k, j] * power_alloc[j, m]

        def f(p):
            sum = 0
            for m in range(n_channel):
                sum = sum + channel_alloc[n, m] * priority[n] * B / (10 ** 6) \
                      * np.log2(1 + channel_gain[n, n] / SNR_gap[n] * p[m] / (C[m] + noise_vec[n]))
            for k in range(n_su):
                if (k != n):
                    for m in range(n_channel):
                        sum = sum + channel_alloc[k, m] * priority[k] * B / (10 ** 6) \
                              * np.log2(E[k, m] + noise_vec[k] + channel_alloc[n, m] * channel_gain[k, n] * p[m])
            return sum

        def df(p):
            df = np.zeros(n_channel)
            for m in range(n_channel):
                df[m] = df[m] + channel_alloc[n, m] * priority[n] * B / (10 ** 6) / np.log(2) * channel_gain[n, n] \
                        / SNR_gap[n] / (C[m] + noise_vec[n] + channel_gain[n, n] / SNR_gap[n] * p[m])
                for k in range(n_su):
                    if (k != n):
                        df[m] = df[m] + channel_alloc[k, m] * priority[k] * B / (10 ** 6) / np.log(2) * \
                                channel_alloc[n, m] * channel_gain[k, n] \
                                / (E[k, m] + noise_vec[k] + channel_alloc[n, m] * channel_gain[k, n] * p[m])
            return df

        def g(p, low_dim=False):
            if (low_dim):
                p_prev = copy.deepcopy(p)
                p = np.zeros(n_channel)
                p_prev_index = 0
                for k in range(n_channel):
                    if (channel_alloc[n, k] == 1):
                        p[k] = p_prev[p_prev_index]
                        p_prev_index = p_prev_index + 1

            sum = 0
            for k in range(n_su):
                if (k != n):
                    for m in range(n_channel):
                        sum = sum + channel_alloc[k, m] * priority[k] * B / (10 ** 6) \
                              * np.log2(D[k, m] + noise_vec[k] + channel_alloc[n, m] * channel_gain[k, n] * p[m])
            return sum

        def finite_power_solver(vertex_list, vertex_total):

            # feasible region for node i
            a_feasible_list, b_feasible_list = feasible_region()

            vertex_total = copy.deepcopy(vertex_total)

            # If previous power allocation is out of feasible region, add a vertex
            for j in range(p_dim + 1):
                i = len(vertex_total) - 1
                if (np.dot(a_feasible_list[j], vertex_total[i][np.where(channel_alloc[n, :] == 1)]) > (
                        b_feasible_list[j] + 0.01)):
                    del vertex_total[i]
                    # Find gravity center for vertex_list
                    v_gravity = np.zeros(n_channel)
                    for v in vertex_list:
                        v_gravity[np.where(channel_alloc[n, :] == 1)] \
                            = v_gravity[np.where(channel_alloc[n, :] == 1)] + v[np.where(channel_alloc[n, :] == 1)]
                    v_gravity = v_gravity / (p_dim + 1)
                    vertex_total.append(copy.deepcopy(v_gravity))
                    break

            # delete the vertices that are out of feasible region of node i
            for j in range(p_dim + 1):
                i = len(vertex_total) - 3
                while (True):
                    if (i < 0):
                        break
                    if (np.dot(a_feasible_list[j], vertex_total[i][np.where(channel_alloc[n, :] == 1)]) > (
                            b_feasible_list[j] + 0.01)):
                        del vertex_total[i]
                    i = i - 1

            A_arr = np.zeros((len(vertex_total) + (p_dim + 1) + p_dim + len(self.minRate_h), 1 + p_dim))
            b_arr = np.zeros((len(vertex_total) + (p_dim + 1) + p_dim + len(self.minRate_b), 1))


            # fM(p) constraint: t - fM(p) <= 0
            A_arr[0: len(vertex_total), 0] = 1
            k = 0
            for v in vertex_total:
                tmp = - df(v)
                tmp = tmp[np.where(channel_alloc[n, :] == 1)]
                A_arr[k, 1:] = copy.deepcopy(tmp)
                b_arr[k, 0] = f(v) - np.dot(v.T, df(v))
                k = k + 1

            # feasible region constraint
            for k in range(p_dim + 1):
                A_arr[len(vertex_total) + k, 0] = 0
                A_arr[len(vertex_total) + k, 1:] = a_feasible_list[k]
                b_arr[len(vertex_total) + k, 0] = b_feasible_list[k]

            # QAM capacity constraint
            A_arr[len(vertex_total) + (p_dim + 1): len(vertex_total) + (p_dim + 1) + p_dim, 1:] = np.identity(p_dim)

            b_arr[len(vertex_total) + (p_dim + 1): len(vertex_total) + (p_dim + 1) + p_dim, 0] = \
                self.QAM_max_power[np.where(channel_alloc[n, :] == 1)]

            # minimum data rate constraint
            A_arr[len(vertex_total) + (p_dim + 1) + p_dim:, 0] = 0
            A_arr[len(vertex_total) + (p_dim + 1) + p_dim:, 1:] = self.minRate_h
            b_arr[len(vertex_total) + (p_dim + 1) + p_dim:, 0] = self.minRate_b

            # b_arr - A_arr * x >= 0
            A = np.hstack((b_arr, -A_arr))
            A = np.round(A, 8)

            mat = cdd.Matrix(A, number_type='fraction')
            mat.rep_type = cdd.RepType.INEQUALITY
            poly = cdd.Polyhedron(mat)
            vertices = poly.get_generators()
            vertices_array = np.array(vertices, dtype=float)

            if (vertices_array.size == 0):
                # print("Cannot find a feasible solution!")
                return
                # os._exit()

            upperbound_max = -float("inf")
            p_sol_lowd = np.zeros(p_dim)
            for i in range(vertices_array.shape[0]):
                if (vertices_array[i, 0] == 1):
                    # v = vertices_array[i, 1:]
                    # q = np.round(np.matmul(A_arr, v) - b_arr.reshape(-1), 4)
                    # if (np.amax(q) < 0 + 0.001):
                    upperbound = vertices_array[i, 1] - g(vertices_array[i, 2:], low_dim=True)
                    if (upperbound > upperbound_max):
                        p_sol_lowd = vertices_array[i, 2:]
                        upperbound_max = upperbound


            p_sol = np.zeros(n_channel)
            p_lowd_index = 0
            for k in range(n_channel):
                if (channel_alloc[n, k] == 1):
                    p_sol[k] = p_sol_lowd[p_lowd_index]
                    p_lowd_index = p_lowd_index + 1

            return p_sol, upperbound_max

        def feasible_region():
            # store the constraints for the feasible region hx <= b
            h_list = []
            b_list = []

            vertex_ld = []
            for v in vertex_list:
                vertex_ld.append(v[np.where(channel_alloc[n, :] == 1)])

            for i in range(p_dim+1):
                vertex_hyperlane = copy.deepcopy(vertex_ld)
                vertex_out = vertex_hyperlane[i]
                del vertex_hyperlane[i]

                b = np.ones(p_dim)
                A = np.zeros((p_dim, p_dim))

                for k in range(len(vertex_hyperlane)):
                    A[k, :] = vertex_hyperlane[k]


                if (np.linalg.matrix_rank(A) == p_dim):
                    h = np.linalg.solve(A, b)
                    b = b[0]
                else:
                    while (True):
                        translation = np.random.rand(p_dim)
                        vertex_hyperlane_new = copy.deepcopy(vertex_hyperlane)
                        for k in range(len(vertex_hyperlane_new)):
                            vertex_hyperlane_new[k] = vertex_hyperlane[k] + translation

                        for k in range(len(vertex_hyperlane_new)):
                            A[k, :] = vertex_hyperlane_new[k]

                        if (np.linalg.matrix_rank(A) == p_dim):
                            h = np.linalg.solve(A, b)
                            b = b[0] - np.inner(h, translation)
                            break
                if ( np.inner(h, vertex_out) > b ):
                    h = -h
                    b = -b

                h_list.append(np.round(h, 8))
                b_list.append(np.round(b, 8))

            return h_list, b_list

        p_sol, upperbound = finite_power_solver(vertex_list, vertex_total)

        return p_sol, upperbound

    def lowerbound(self, Node, power_alloc, channel_alloc, channel_gain, B, noise_vec, priority, SU_index, SNR_gap):

        vertex_lb = Node['vertex_lb']

        max_capacity = 0
        for p in vertex_lb:
            capacity = objective_value_SU(p, SU_index, channel_alloc, power_alloc, priority, channel_gain, B, noise_vec, SNR_gap)
            if (capacity > max_capacity):
                max_p = copy.deepcopy(p)
                max_capacity = capacity

        return max_p, max_capacity

    def check_feasible(self, Node, channel_alloc, SU_index):
        vertex_list = Node['vertex']
        n = SU_index
        p_dim = int(np.sum(channel_alloc[n, :]))

        def feasible_region():
            # store the constraints for the feasible region hx <= b
            h_list = []
            b_list = []

            vertex_ld = []
            for v in vertex_list:
                vertex_ld.append(v[np.where(channel_alloc[n, :] == 1)])

            for i in range(p_dim + 1):
                vertex_hyperlane = copy.deepcopy(vertex_ld)
                vertex_out = vertex_hyperlane[i]
                del vertex_hyperlane[i]

                b = np.ones(p_dim)
                A = np.zeros((p_dim, p_dim))

                for k in range(len(vertex_hyperlane)):
                    A[k, :] = vertex_hyperlane[k]

                if (np.linalg.matrix_rank(A) == p_dim):
                    h = np.linalg.solve(A, b)
                    b = b[0]
                else:
                    while (True):
                        translation = np.random.rand(p_dim)
                        vertex_hyperlane_new = copy.deepcopy(vertex_hyperlane)
                        for k in range(len(vertex_hyperlane_new)):
                            vertex_hyperlane_new[k] = vertex_hyperlane[k] + translation

                        for k in range(len(vertex_hyperlane_new)):
                            A[k, :] = vertex_hyperlane_new[k]

                        if (np.linalg.matrix_rank(A) == p_dim):
                            h = np.linalg.solve(A, b)
                            b = b[0] - np.inner(h, translation)
                            break
                if (np.inner(h, vertex_out) > b):
                    h = -h
                    b = -b

                h_list.append(np.round(h, 8))
                b_list.append(np.round(b, 8))

            return h_list, b_list

        # feasible region for node i
        a_feasible_list, b_feasible_list = feasible_region()


        A_arr = np.zeros(((p_dim + 1) + p_dim + len(self.minRate_h), 1 + p_dim))
        b_arr = np.zeros(((p_dim + 1) + p_dim + len(self.minRate_b), 1))


        # feasible region constraint
        for k in range(p_dim + 1):
            A_arr[k, 0] = 0
            A_arr[k, 1:] = a_feasible_list[k]
            b_arr[k, 0] = b_feasible_list[k]



        # QAM capacity constraint
        A_arr[(p_dim + 1): (p_dim + 1) + p_dim, 1:] = np.identity(p_dim)

        b_arr[(p_dim + 1): (p_dim + 1) + p_dim, 0] = self.QAM_max_power[np.where(channel_alloc[n, :] == 1)]

        # minimum data rate constraint
        A_arr[(p_dim + 1) + p_dim:, 0] = 0
        A_arr[(p_dim + 1) + p_dim:, 1:] = self.minRate_h
        b_arr[(p_dim + 1) + p_dim:, 0] = self.minRate_b

        # b_arr - A_arr * x >= 0
        A = np.hstack((b_arr, -A_arr))
        A = np.round(A, 8)

        mat = cdd.Matrix(A, number_type='fraction')
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        vertices = poly.get_generators()
        vertices_array = np.array(vertices, dtype=float)

        if (vertices_array.size == 0):
            # print("Cannot find a feasible solution!")
            return False
        else:
            return True