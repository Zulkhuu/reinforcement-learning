HYPERPARAMS = {
        #Training parameters
        'env_name':             "Banana_Collector",
        'stop_scores':          13.0,
        'scores_window_size':   100,
        'n_episodes':           2000,
        'epsilon_start':        1.0,                # starting value of epsilon
        'epsilon_final':        0.01,               # minimum value of epsilon
        'epsilon_decay':        0.995,              # factor for decreasing epsilon

        # DQN agent parameters
        'seed':                 0,                  # random seed
        'buffer_size':          100000,             # replay buffer size
        'batch_size':           64,                 # minibatch size
        'update_every':         4,                  # network updating every update_interval steps
        'learning_rate':        5e-4,               # learning rate
        'tau':                  1e-3,               # for soft update of target parameters
        'gamma':                0.99,               # discount factor

        # Q network parameters
        'state_size':           37,                 # state size
        'action_size':          4,                  # action size
        'fc1_units':            64,                 # Number of nodes in first hidden layer
        'fc2_units':            64,                 # Number of nodes in second hidden layer
}
