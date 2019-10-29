import numpy as np


class ChannelGrouping:
    def __init__(self, channel_IDs, unit_bandwidth):

        self.channel_IDs = channel_IDs

        self.n_channels = len(channel_IDs)

        self.contiguous_channels = []

        self.unit_bandwidth = unit_bandwidth

        temp_CG = [channel_IDs[0]]

        for i in range(1, self.n_channels):

            if channel_IDs[i] - channel_IDs[i-1] == 1:
                temp_CG.append(channel_IDs[i])
            else:
                self.contiguous_channels.append(temp_CG)
                temp_CG = [channel_IDs[i]]

        self.contiguous_channels.append(temp_CG)

        self.channel_groups = []
        for i in range(len(self.contiguous_channels)):
            if len(self.contiguous_channels[i]) / 4 < 1:
                for j in self.contiguous_channels[i]:
                    self.channel_groups.append([j])
            else:
                for j in range(int(len(self.contiguous_channels[i]) / 4)):
                    self.channel_groups.append(self.contiguous_channels[i][4 * j:4 * j + 4])
                    i_end = 4 * j + 4
                for j in range(i_end, len(self.contiguous_channels[i])):
                    self.channel_groups.append([self.contiguous_channels[i][j]])

        self.large_CGs = []
        self.small_CGs = []
        for CG in self.channel_groups:
            if len(CG) == 4:
                self.large_CGs.append(CG)
            else:
                self.small_CGs.append(CG)

        self.channel_groups = self.large_CGs + self.small_CGs

        self.n_CGs = len(self.channel_groups)

        self.n_channels_in_CGs = np.zeros(self.n_CGs, dtype=np.int32)
        for i_CG in range(self.n_CGs):
            self.n_channels_in_CGs[i_CG] = len(self.channel_groups[i_CG])

        self.bandwidth_CGs = self.unit_bandwidth * (10**6) * self.n_channels_in_CGs

        self.n_large_CGs = len(self.large_CGs)
        self.n_small_CGs = len(self.small_CGs)

    def split(self):
        if self.n_large_CGs > 0:
            splitted_CG = self.large_CGs[-1]
            new_CGs = []
            for ch in splitted_CG:
                new_CGs.append([ch])

            self.large_CGs = self.large_CGs[:-1]
            self.small_CGs = new_CGs + self.small_CGs

            self.channel_groups = self.large_CGs + self.small_CGs

            self.n_CGs = len(self.channel_groups)

            self.n_channels_in_CGs = np.zeros(self.n_CGs, dtype=np.int32)
            for i_CG in range(self.n_CGs):
                self.n_channels_in_CGs[i_CG] = len(self.channel_groups[i_CG])

            self.bandwidth_CGs = self.unit_bandwidth * (10 ** 6) * self.n_channels_in_CGs

            self.n_large_CGs = len(self.large_CGs)
            self.n_small_CGs = len(self.small_CGs)