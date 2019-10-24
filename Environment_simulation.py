from Channel_model import Winner2_Optimizer

def environment_simulation(n_channel, n_cluster, n_user_cluster, area):
    env = Winner2_Optimizer(n_channel, n_cluster, n_user_cluster, area)
    h_mean, h_min, h_std_dB, h_all, h_all_dB, = env.channel_gain(list(range(n_cluster)))
    noise_mat = env.NoisePower_mat
    return h_mean, h_min, h_std_dB, h_all, h_all_dB, noise_mat