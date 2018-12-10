import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from cycler import cycler

def plot_lr_actor():
    csv_filename = '../scores/DDPG_lra{}_lrc1E-04_batch128_fc:256:128.csv'
    plt_filename = '../docs/plots/lr_actor.png'

    learning_rates = ['5E-05', '1E-04', '5E-04']

    fig, ax = plt.subplots()

    for lra in learning_rates:
        filename = csv_filename.format(lra)
        df = pd.read_csv(filename, index_col = 0)
        plt.plot( df.index, 'scores', data=df,
            lw=2, label="Actor learning rate:{}".format(lra), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_lr_critic():
    csv_filename = '../scores/DDPG_lra1E-04_lrc{}_batch128_fc:256:128.csv'
    plt_filename = '../docs/plots/lr_critic.png'

    learning_rates = ['5E-05', '1E-04', '5E-04']

    fig, ax = plt.subplots()

    for lrc in learning_rates:
        filename = csv_filename.format(lrc)
        df = pd.read_csv(filename, index_col = 0)
        plt.plot( df.index, 'scores', data=df,
            lw=2, label="Critic learning rate:{}".format(lrc), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_batch_size():
    csv_filename = '../scores/DDPG_lra1E-04_lrc1E-04_batch{}_fc:256:128.csv'
    plt_filename = '../docs/plots/batch_sizes.png'

    batch_sizes = [256, 128, 64]

    fig, ax = plt.subplots()

    for batch_size in batch_sizes:
        filename = csv_filename.format(batch_size)
        df = pd.read_csv(filename, index_col = 0)
        plt.plot( df.index, 'scores', data=df,
            lw=2, label="Batch size:{}".format(batch_size), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_nn_size():
    csv_filename = '../scores/DDPG_lra1E-04_lrc1E-04_batch128_fc:{}.csv'
    plt_filename = '../docs/plots/nn_sizes.png'

    nn_sizes = ['400:300', '256:128', '128:64']

    fig, ax = plt.subplots()

    for nn_size in nn_sizes:
        filename = csv_filename.format(nn_size)
        df = pd.read_csv(filename, index_col = 0)
        plt.plot( df.index, 'scores', data=df,
            lw=2, label="hidden layer units:{}".format(nn_size), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_learning_curve(filename):
    csv_filename = '../scores/{:s}.csv'.format(filename)
    plt_filename = '../docs/plots/{:s}.png'.format(filename)

    df = pd.read_csv(csv_filename, index_col = 0)
    plt.figure(figsize=(10,5))
    #plt.axhline(13, color='red', lw=1, alpha=0.3) # Draw goal line
    plt.plot( df.index, 'scores', data=df, color='lime', lw=1, label="score per episode", alpha=0.4)
    plt.plot( df.index, 'average_scores', data=df, color='green', lw=2, label="average score")
    # Set labels and legends
    plt.xlabel('Episode')
    plt.xlim(0, len(df.index))
    plt.xticks(50*np.arange(int(len(df.index)/50+1)))
    plt.ylabel('Score')
    plt.yticks(3*np.arange(8))
    plt.title(filename)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    # Save figure
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

#Uncomment the necessary plot and run

# Plot neural network size
#plot_nn_size()

# Plot learning rate of actor
#plot_lr_actor()

# Plot learning rate of critic
#plot_lr_critic()

# Plot batch sizes
plot_batch_size()

# Plot learning curve of particular training
#plot_learning_curve('DDPG_lra1E-04_lrc1E-04_batch128_fc:256:128_solved256')
