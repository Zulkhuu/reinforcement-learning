HYPERPARAMS = {
    'Tennis':
    {
        # Environment parameters
        'env_name':             'Tennis',           # environment name
        'solve_score':          0.5,                # score to solve environment
        'scores_window_size':   100,                # moving average window size
        'n_episodes':           2000,               # maximum number of episodes

        'agent':
        {
            # Agent parameters
            'agent_name':           'MADDPG',           # name
            'n_agents':             2,                  # number of agents
            'random_seed':          0,                  # random seed
            'buffer_size':          500000,             # replay buffer size
            'batch_size':           128,                # minibatch size
            'weight_decay':         0,#0.0001,             # weight decay
            'lr_actor':             1e-4,               # learning rate for actor
            'lr_critic':            1e-4,               # learning rate for critic
            'tau':                  1e-5,               # for soft update of target parameters
            'gamma':                0.99,               # discount factor

            # Q network parameters
            'state_size':           24,                 # state size
            'action_size':          2,                  # action size
            'fc1_units':            128,                # Number of nodes in first hidden layer
            'fc2_units':            150,                # Number of nodes in second hidden layer
        }
    },
    'Tennis1':
    {
        # Environment parameters
        'env_name':             'Tennis',           # environment name
        'solve_score':          0.1,                # score to solve environment
        'scores_window_size':   100,                # moving average window size
        'n_episodes':           2000,               # maximum number of episodes

        'agent':
        {
            # Agent parameters
            'agent_name':           'MADDPG',           # name
            'n_agents':             2,                  # number of agents
            'random_seed':          0,                  # random seed
            'buffer_size':          1000000,            # replay buffer size
            'batch_size':           128,                # minibatch size
            'weight_decay':         0,#0.0001,             # weight decay
            'lr_actor':             1e-3,               # learning rate for actor
            'lr_critic':            1e-3,               # learning rate for critic
            'tau':                  6e-2,               # for soft update of target parameters
            'gamma':                0.99,               # discount factor

            # Q network parameters
            'state_size':           24,                 # state size
            'action_size':          2,                  # action size
            'fc1_units':            256,                # Number of nodes in first hidden layer
            'fc2_units':            128,                # Number of nodes in second hidden layer
        }
    },
}
