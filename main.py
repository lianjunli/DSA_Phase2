from Environment_simulation import environment_simulation
from CPO import CPO
from Performance_evaluation import performance_evaluation

###----------------------SYSTEM INPUTS----------------------###

# Minimum data rate scaling up factor in dB
min_Rate_Margin = 0

# Shadow fading margin in dB (for environment simulation)
Log_Normal_sigma = 3

# Shadow fading margin in dB (for CPO)
shadow_Fading_Margin = 3

# Simulation area (square meter)
area = 600

# Intra cluster channel gain type used in minmum data constraint: min: minumu channel gain; average: geometric average channel gain
minRate_intra_gain_type = 1  # min = 2, average = 1

# Intra cluster channel gain type used in DC objective function: min: minumu channel gain; average: geometric average channel gain
DC_intra_gain_type = 1  # min = 2, average = 1

# The total number of clusters
n_cluster = 10

# Radio cluster ID
cluster_ID = list(range(2001,2001+n_cluster))

# The number of users per cluster
n_user_cluster = 8

# SNR gap in dB
SNR_gap_dB = [3.9] * n_cluster

# Radio cluster priority
priority = [1] * n_cluster

# Minimum data rate constraint (Mbps)
minRate = [1] * n_cluster

# Maximum power constraint (dBm)
maxPower = [30] * n_cluster

# Channel IDs
# channel_IDs = [2, 11, 12, 13, 14, 15, 16]
channel_IDs = [11]

# NoisePower
noise_mat=[]

###----------------------SYSTEM INPUTS END----------------------###

# simulate environment
h_mean, h_min, h_std_dB, h_all, h_all_dB, noise_mat \
    = environment_simulation(channel_IDs, n_cluster, n_user_cluster, area)

# channel allocation and power allocation
channel_allocation, channel_bandwidth, power_allocation, noise_vec, SUCCESS_INDICATOR \
    = CPO(min_Rate_Margin, h_mean, h_min, h_std_dB, shadow_Fading_Margin, minRate_intra_gain_type, DC_intra_gain_type,
          SNR_gap_dB, priority, minRate, maxPower, cluster_ID, channel_IDs, noise_mat, n_user_cluster)

# performance evaluation
if SUCCESS_INDICATOR:
    performance_evaluation(channel_allocation, channel_bandwidth, power_allocation, h_all, noise_vec, Log_Normal_sigma,
                           n_cluster, n_user_cluster, SNR_gap_dB, minRate)
