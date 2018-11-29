HYPERPARAMS = {
        'env_name':             "Banana_Collector",
        'stop_scores':          13.0,
        'scores_window_size':   100,
        'train_episodes':       1800,

        # DQN agent parameters
        'replay_size':          100000,             # replay buffer size
        'replay_initial':       10000,              # replay buffer initialize
        'fix_target_updates':   1,                  # fix the target Q for the fix_target_updates
        'epsilon_start':        1.0,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.993,              # factor for decreasing epsilon

        # Q network parameters
        'hidden_layers':        [64, 64],           # hidden units and layers of Q-network
        'update_interval':      4,                  # network updating every update_interval steps
        'learning_rate':        5e-4,               # learning rate
        'thau':                 1e-3,               # for soft update of target parameters
        'gamma':                0.99,               # discount factor
        'batch_size':           64                  # minibatch size
}
