import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

'''
The required scenario Optimizer
'''
class Winner2_Optimizer():
    def __init__(
            self,
            n_channel,
            n_cluster,
            n_user_cluster,
            # B,
            area
    ):
        self.n_channel = n_channel
        self.n_cluster = n_cluster
        self.n_user_cluster = n_user_cluster
        self.n_user = self.n_cluster * self.n_user_cluster


        # Set noise power per cluster per 1.25Mhz channel
        self.NoisePower_dBm = np.ones((self.n_cluster,self.n_channel)) * -105 #noise power per cluster in dBm #-93
        self.NoisePower_mat = 10**(self.NoisePower_dBm/10)



        # Set the carrier frequency (2 GHz)
        self.fc = 2
        # Set the path loss model exponent
        self.exponent = 36.8

        self.random_seed = 12  #12, 21
        np.random.seed(self.random_seed)

        #self._build_location_asymmetric()
        self._build_location(area)


    def _build_location(self,area):

        radius = 150 # 50
        self.area = area

        # Initialize the location of centers of the group
        center_y = np.random.uniform(0 + radius, self.area - radius, self.n_cluster)
        center_x = np.zeros(self.n_cluster)
        for n in range(self.n_cluster):
            center_x[n] = np.random.uniform(radius + (self.area - 2 * radius) / self.n_cluster * n,
                                            radius + (self.area - 2 * radius) / self.n_cluster * (n + 1))

        # Initialize the location of users in the group
        self.user_x = []
        self.user_y = []
        self.center_x = np.zeros(self.n_cluster)
        self.center_y = np.zeros(self.n_cluster)
        for n in range(self.n_cluster):
            self.user_x.append(np.zeros(self.n_user_cluster))
            self.user_y.append(np.zeros(self.n_user_cluster))

            theda = 2 * np.pi * np.random.uniform(0, 1, self.n_user_cluster)
            d = radius * np.random.uniform(0, 1, self.n_user_cluster)
            dx = d * np.cos(theda)
            dy = d * np.sin(theda)
            self.user_x[n] = center_x[n] + dx
            self.user_y[n] = center_y[n] + dy
            self.center_x[n] = np.mean(self.user_x[n])
            self.center_y[n] = np.mean(self.user_y[n])

        # self._plot_SU_location(self.area)

    def _build_location_asymmetric(self):

        self.area = 360

        radius = 50
        center_radius = 5

        # Initialize the location of centers of the group
        center_y = np.random.uniform(0 + radius, self.area - radius, self.n_cluster)
        center_x = np.zeros(self.n_cluster)
        for n in range(self.n_cluster):
            center_x[n] = np.random.uniform(radius + (self.area - 2 * radius) / self.n_cluster * n,
                                            radius + (self.area - 2 * radius) / self.n_cluster * (n + 1))

        # Initialize the location of users in the group
        self.user_x = []
        self.user_y = []
        self.center_x = np.zeros(self.n_cluster)
        self.center_y = np.zeros(self.n_cluster)
        for n in range(self.n_cluster):
            self.user_x.append(np.zeros(self.n_user_cluster))
            self.user_y.append(np.zeros(self.n_user_cluster))

            decision_x = 2 * np.random.randint(2, size=1) - 1
            decision_y = np.random.randint(3, size=1) - 1

            self.user_x[n][0:5] = center_x[n] + decision_x * (radius + center_radius) / 2 \
                                  + np.random.uniform(-radius/2, radius/2, 5)
            self.user_x[n][5:8] = center_x[n] - decision_x * (radius + center_radius) / 2 \
                                  + np.random.uniform(-radius / 2, radius / 2, 3)

            self.user_y[n][0:5] = center_y[n] + decision_y * (radius + center_radius) / 2
            self.user_y[n][5:8] = center_y[n] - decision_y * (radius + center_radius) / 2
            if (decision_y == 0):
                self.user_y[n][0:5] += np.random.uniform(-center_radius / 2, center_radius / 2, 5)
                self.user_y[n][5:8] += np.random.uniform(-center_radius / 2, center_radius / 2, 3)
            else:
                self.user_y[n][0:5] += np.random.uniform(-radius / 2, radius / 2, 5)
                self.user_y[n][5:8] += np.random.uniform(-radius / 2, radius / 2, 3)

            self.center_x[n] = np.mean(self.user_x[n])
            self.center_y[n] = np.mean(self.user_y[n])

        # self._plot_SU_location(self.area)

    def _PLdB(self, d):
        return 46.4 + self.exponent * np.log10(d) + 20 * np.log10(self.fc / 5)

    def _plot_SU_location(self, area):

        plt.figure(figsize=(6, 6))
        plt.ylim(0, area)
        plt.xlim(0, area)
        # Plot the locations
        for n in range(self.n_cluster):
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(self.user_x[n], self.user_y[n], 'o', color = color)
            plt.plot(self.center_x[n], self.center_y[n], '+', color=color)
            plt.annotate(
                "Cluster%d" % n, color = color,
                xy=(self.center_x[n], self.center_y[n]), xytext=(0, 20),
                textcoords='offset points', ha='center', va='bottom',
            )
        plt.ylabel('y', fontsize=14)
        plt.xlabel('x', fontsize=14)
        plt.show()

    def channel_gain(self, cluster_list, interfer='average'):

        n_cluster = len(cluster_list)
        channel_gain_mean = np.zeros((n_cluster, n_cluster))
        channel_gain_min = np.zeros((n_cluster, n_cluster))
        channel_gain_std_dB = np.zeros(n_cluster)

        '''
        Calculate the all transmission pairs' channel gain
        '''
        h_all = np.zeros((n_cluster * self.n_user_cluster, n_cluster * self.n_user_cluster))
        h_all_dB = np.zeros((n_cluster * self.n_user_cluster, n_cluster * self.n_user_cluster))

        for tx in range(n_cluster * self.n_user_cluster):
            for rx in range(n_cluster * self.n_user_cluster):
                if tx == rx:
                    h_all[tx, rx] = 0
                else:
                    tx_cluster = cluster_list[math.floor(tx / self.n_user_cluster)]
                    rx_cluster = cluster_list[math.floor(rx / self.n_user_cluster)]
                    d = np.sqrt((self.user_x[tx_cluster][tx % self.n_user_cluster] -
                                 self.user_x[rx_cluster][rx % self.n_user_cluster]) ** 2 +
                                (self.user_y[tx_cluster][tx % self.n_user_cluster] -
                                 self.user_y[rx_cluster][rx % self.n_user_cluster]) ** 2)
                    h_all[tx, rx] = np.float_power(10, -((self._PLdB(d)) / 10))
                    h_all_dB[tx, rx] = -self._PLdB(d)


        for c1 in range(len(cluster_list)):
            '''
            Calculate the desired channel gain
            '''
            h_dB = []
            cluster = cluster_list[c1]
            for tx in range(self.n_user_cluster):
                for rx in range(tx + 1, self.n_user_cluster):
                    d = np.sqrt((self.user_x[cluster][tx] - self.user_x[cluster][rx])**2 +
                                (self.user_y[cluster][tx] - self.user_y[cluster][rx])**2)
                    h_dB.append(-self._PLdB(d))
            h_dB = np.sort(h_dB)
            h_lin = np.float_power(10, h_dB/10)
            h_mean_harmonic = len(h_lin)/np.sum(1/h_lin)
            h_mean_dB = sum(h_dB) / len(h_dB)
            h_mean = np.float_power(10, h_mean_dB/10)
            h_min_dB = min(h_dB)
            h_min = np.float_power(10, h_min_dB/10)
            h_std_dB = statistics.stdev(h_dB)
            channel_gain_mean[c1, c1] = h_mean
            channel_gain_min[c1, c1] = h_min
            channel_gain_std_dB[c1] = h_std_dB

            '''
            Calculate the interfered channel gain
            '''
            for c2 in range(len(cluster_list)):
                if (c2 == c1):
                    continue
                cluster_inter = cluster_list[c2]
                h_dB = []
                for tx in range(self.n_user_cluster):
                    for rx in range(self.n_user_cluster):
                        d = np.sqrt((self.user_x[cluster_inter][tx] - self.user_x[cluster][rx]) ** 2 +
                                    (self.user_y[cluster_inter][tx] - self.user_y[cluster][rx]) ** 2)
                        h_dB.append(-self._PLdB(d))

                if (interfer == 'average'):
                    h_mean_dB = sum(h_dB)/len(h_dB)
                    channel_gain_mean[c1, c2] = np.float_power(10, h_mean_dB / 10)
                    channel_gain_min[c1, c2] = np.float_power(10, h_mean_dB / 10)
                elif (interfer == 'max'):
                    h_max_dB = max(h_dB)
                    channel_gain_mean[c1, c2] = np.float_power(10, h_max_dB / 10)
                    channel_gain_min[c1, c2] = np.float_power(10, h_max_dB / 10)

        return channel_gain_mean, channel_gain_min, channel_gain_std_dB, h_all, h_all_dB



class Winner2_LGS():
    def __init__(self, n_channel_type1, n_channel_type2, n_channel_type3,
                      n_group_type1, n_group_type2, n_group_type3,
                      n_su_group_type1, n_su_group_type2, n_su_group_type3):

        self.n_group_type1 = n_group_type1
        self.n_group_type2 = n_group_type2
        self.n_group_type3 = n_group_type3

        self.n_channel_type1 = n_channel_type1
        self.n_channel_type2 = n_channel_type2
        self.n_channel_type3 = n_channel_type3

        self.n_su_group_type1 = n_su_group_type1
        self.n_su_group_type2 = n_su_group_type2
        self.n_su_group_type3 = n_su_group_type3

        self.max_inter_type1 = 4
        self.max_inter_type2 = 3

        # Set the bandwidth (1.2MhZ)
        self.B = 1.2*(10**6)
        # Set the noise spectral density (10^(-10.4) mw/Hz)
        #self.Noise = np.float_power(10, -13.4)
        self.Noise = 4 * np.float_power(10, -16)
        # Set the carrier frequency (5 GHz)
        self.fc = 5

        self.random_seed = 12 # document is 12 10, 12, 14
        np.random.seed(self.random_seed)

        # build the locations of group 1, 2, 3
        self.area = 700 # 1200, 800

        self.radius_type1 = 100
        self.random_type1 = True
        self.minDis_type1 = True
        self._build_location(1)

        self.radius_type2 = 100
        self.random_type2 = True
        self.minDis_type2 = True
        #self._build_location(2)
        #self._build_location(3)


    def _build_location(self, type_index):

        area = self.area

        if (type_index == 1):
            n_group = self.n_group_type1

            # radius of the group
            radius = self.radius_type1

            # number of users in the group
            n_su_group = self.n_su_group_type1

            # The geometry within a user group
            randomLoc = self.random_type1

            # The distance measure between the groups
            minDis = self.minDis_type1

        elif (type_index == 2):
            n_group = self.n_group_type2

            # radius of the group
            radius = self.radius_type2

            # number of users in the group
            n_su_group = self.n_su_group_type2

            # The geometry within a user group
            randomLoc = self.random_type2

            # The distance measure between the groups
            minDis = self.minDis_type2


        # Initialize the location of centers of the group
        group_y = np.random.uniform(0 + radius, area - radius, n_group)
        group_x = np.zeros(n_group)
        for n in range(n_group):
            group_x[n] = np.random.uniform(radius + (area - 2 * radius) / n_group * n,
                                           radius + (area - 2 * radius) / n_group * (n + 1))

        # Initialize the location of users in the group
        SU_x = []
        SU_y = []
        for n in range(n_group):
            SU_x.append(np.zeros(n_su_group))
            SU_y.append(np.zeros(n_su_group))

            theda = np.zeros(n_su_group)

            if (randomLoc):
                theda = 2 * np.pi * np.random.uniform(0, 1, n_su_group)
                d = radius * np.random.uniform(0, 1, n_su_group)
                dx = d * np.cos(theda)
                dy = d * np.sin(theda)
                SU_x[n] = group_x[n] + dx
                SU_y[n] = group_y[n] + dy
            else:
                theda[0] = 2 * np.pi * np.random.uniform(0, 1)
                for i in range(1, n_su_group):
                    theda[i] = theda[0] + (2 * np.pi / n_su_group) * i
                dx = radius * np.cos(theda)
                dy = radius * np.sin(theda)
                SU_x[n] = group_x[n] + dx
                SU_y[n] = group_y[n] + dy

        # Compute the distance between different groups
        group_dis = np.zeros((n_group, n_group))
        if (minDis):
            for k1 in range(n_group):
                for k2 in range(k1 + 1, n_group):
                    group_dis[k1][k2] = float("inf")
                    for n1 in range(n_su_group):
                        for n2 in range(n_su_group):
                            tmp_dis = np.sqrt(np.float_power(SU_x[k1][n1] - SU_x[k2][n2], 2)
                                              + np.float_power(SU_y[k1][n1] - SU_y[k2][n2], 2))
                            if (tmp_dis < group_dis[k1][k2]):
                                group_dis[k1][k2] = tmp_dis
                    group_dis[k2][k1] = group_dis[k1][k2]

        else:
            for k1 in range(n_group):
                for k2 in range(k1 + 1, n_group):
                    group_dis[k1][k2] = \
                        np.sqrt(np.float_power(group_x[k1] - group_x[k2], 2)
                                + np.float_power(group_y[k1] - group_y[k2], 2))
                    group_dis[k2][k1] = group_dis[k1][k2]

        if (type_index == 1):
            self.group_y_type1 = group_y
            self.group_x_type1 = group_x
            self.SU_x_type1 = SU_x
            self.SU_y_type1 = SU_y
            self.group_dis_type1 = group_dis
            self._plot_SU_location(1)

        elif (type_index == 2):
            self.group_y_type2 = group_y
            self.group_x_type2 = group_x
            self.SU_x_type2 = SU_x
            self.SU_y_type2 = SU_y
            self.group_dis_type2 = group_dis
            self._plot_SU_location(2)


    def _plot_SU_location(self, type_index):

        if (type_index == 1):
            n_group = self.n_group_type1
            radius = self.radius_type1
            SU_x = self.SU_x_type1
            SU_y = self.SU_y_type1
            group_x = self.group_x_type1
            group_y = self.group_y_type1
        elif (type_index == 2):
            n_group = self.n_group_type2
            radius = self.radius_type2
            SU_x = self.SU_x_type2
            SU_y = self.SU_y_type2
            group_x = self.group_x_type2
            group_y = self.group_y_type2

        plt.figure(figsize=(8, 8))
        plt.ylim(0, self.area)
        plt.xlim(0, self.area)
        # Plot the locations
        for n in range(n_group):
            plt.plot(SU_x[n], SU_y[n], 'ko')
            labelstr = "G%d" % n
            plt.annotate(
                labelstr,
                xy=(group_x[n], group_y[n]), xytext=(0, 16),
                textcoords='offset points', ha='center', va='bottom',
            )
            circle = plt.Circle((group_x[n], group_y[n]), radius, color = 'k',
                                linestyle= '--', fill=False)
            plt.gca().add_artist(circle)
        plt.ylabel('y', fontsize=14)
        plt.xlabel('x', fontsize=14)


    def channel_gain_type1(self, cluster):
        n_group = len(cluster)
        n_su = n_group * self.n_su_group_type1
        SU_x = np.zeros(n_su)
        SU_y = np.zeros(n_su)
        for k in range(n_group):
            group = cluster[k]

            SU_x[k * self.n_su_group_type1: (k+1) * self.n_su_group_type1] = \
                self.SU_x_type1[group][:]
            SU_y[k * self.n_su_group_type1: (k+1) * self.n_su_group_type1] = \
                self.SU_y_type1[group][:]


        channel_gain = np.zeros((n_su, n_su))
        for k1 in range(n_su):
            for k2 in range(n_su):
                d = np.sqrt(np.float_power(SU_x[k1] - SU_x[k2], 2)
                            + np.float_power(SU_y[k1] - SU_y[k2], 2))
                if (k1 == k2):
                    channel_gain[k1, k2] = 0
                else:
                    channel_gain[k1, k2] = np.float_power(10, -((46.4 + 35 * np.log10(d) + 20 * np.log10(self.fc / 5)) / 10))

        return channel_gain