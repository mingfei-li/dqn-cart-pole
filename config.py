class Config():
    num_episodes_train = 4000
    max_eps = 0.5
    min_eps = 0.5
    n_eps = 50_000
    max_lr = 5e-4
    min_lr = 1e-5
    n_lr = 500_000
    target_update_freq = 50
    batch_size = 32
    gamma = 0.99
    learning_start = 10_000
    buffer_size = 50_000
    grad_clip = 1e4
    exp_id = "high-exploration-low-lr-grad-clip"