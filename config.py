class Config():
    num_episodes_train = 1500
    max_eps = 0.05
    min_eps = 0.05
    n_eps = 50_000
    max_lr = 1e-4
    min_lr = 1e-4
    n_lr = 500_000
    target_update_freq = 50
    batch_size = 256
    gamma = 0.99
    learning_start = 10_000
    buffer_size = 50_000
    grad_clip = 1e4
    exp_id = "exp-5:batchsize=256,lr=1e-4,width=128,adam-corrected"