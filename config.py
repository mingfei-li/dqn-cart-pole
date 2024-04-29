class Config():
    num_episodes = 10000
    max_eps = 0.5
    min_eps = 0.05
    n_eps = 50_000
    max_lr = 5e-5
    min_lr = 1e-7
    n_lr = 500_000
    target_update_freq = 50
    batch_size = 32
    gamma = 0.99
    learning_start = 10000
    buffer_size = 50000
    grad_clip = 1e4