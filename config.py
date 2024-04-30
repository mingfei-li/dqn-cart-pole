class Config():
    num_episodes = 5000
    max_eps = 0.5
    min_eps = 0.1
    n_eps = 50_000
    max_lr = 5e-4
    min_lr = 5e-4
    n_lr = 500_000
    target_update_freq = 50
    batch_size = 32
    gamma = 0.99
    learning_start = 30_000
    buffer_size = 100_000
    grad_clip = 1e4