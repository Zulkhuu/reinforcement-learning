import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from cycler import cycler

N_TRY = 3

def plot_lr():
    csv_filename = '../scores/MADDPG_lra{:.0E}_lrc{:.0E}_tau6E-02_nstart5.0_nt400_run{}.csv'
    plt_filename = '../docs/plots/learning_rates.png'

    learning_rates = [5E-03, 2E-03, 1E-03, 5E-04, 2E-04, 1E-04, 5E-05]

    fig = plt.figure(figsize=(10,5))

    labels = []
    for lr in learning_rates:
        for idx in range(N_TRY):
            filename = csv_filename.format(lr, lr, idx)
            df = pd.read_csv(filename, index_col = 0)
            label = "Learning rate:{}".format(lr)
            labels.append(label)
            plt.plot( df.index, 'average_scores', data=df,
                lw=2, label=label, alpha=0.8)

    #Use same color and only one label if labels are same
    ax = fig.gca()
    printed_labels = []
    for i, p in enumerate(ax.get_lines()):
        if p.get_label() in labels:
            idx = labels.index(p.get_label())
            p.set_c(ax.get_lines()[idx].get_c())
            if p.get_label() in printed_labels:
                p.set_label('_' + p.get_label())
            else:
                printed_labels.append(p.get_label())

    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_tau():
    csv_filename = '../scores/MADDPG_lra1E-03_lrc1E-03_tau{:.0E}_nstart5.0_nt400_run{}.csv'
    plt_filename = '../docs/plots/tau.png'

    taus = [1E-01, 6E-02, 5E-02, 1E-02, 5E-03, 1E-03]

    fig = plt.figure(figsize=(10,5))

    labels = []
    for tau in taus:
        for idx in range(N_TRY):
            filename = csv_filename.format(tau, idx)
            df = pd.read_csv(filename, index_col = 0)
            label = "Tau:{}".format(tau)
            labels.append(label)
            plt.plot( df.index, 'average_scores', data=df,
                lw=2, label=label, alpha=0.8)

    #Use same color and only one label if labels are same
    ax = fig.gca()
    printed_labels = []
    for i, p in enumerate(ax.get_lines()):
        if p.get_label() in labels:
            idx = labels.index(p.get_label())
            p.set_c(ax.get_lines()[idx].get_c())
            if p.get_label() in printed_labels:
                p.set_label('_' + p.get_label())
            else:
                printed_labels.append(p.get_label())

    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left')
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_noise_scale():
    csv_filename = '../scores/MADDPG_lra1E-03_lrc1E-03_tau6E-02_nstart{:.1f}_nt400_run{}.csv'
    plt_filename = '../docs/plots/noise_scale.png'

    # Exploration boost start
    noise_scales = [10, 7, 5, 3, 2, 1]

    fig = plt.figure(figsize=(10,5))

    labels = []
    for n_scale in noise_scales:
        for idx in range(N_TRY):
            filename = csv_filename.format(n_scale, idx)
            df = pd.read_csv(filename, index_col = 0)
            label = "Noise scale:{}".format(n_scale)
            labels.append(label)
            plt.plot( df.index, 'average_scores', data=df,
                lw=2, label=label, alpha=0.8)

    #Use same color and only one label if labels are same
    ax = fig.gca()
    printed_labels = []
    for i, p in enumerate(ax.get_lines()):
        if p.get_label() in labels:
            idx = labels.index(p.get_label())
            p.set_c(ax.get_lines()[idx].get_c())
            if p.get_label() in printed_labels:
                p.set_label('_' + p.get_label())
            else:
                printed_labels.append(p.get_label())

    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left')
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_noise_decay():
    csv_filename = '../scores/MADDPG_lra1E-03_lrc1E-03_tau1E-01_nstart7.0_ndecay{}_run{}.csv'
    plt_filename = '../docs/plots/noise_decay.png'

    # Exploration boost decay
    n_decays = [0.999, 0.996, 0.993, 0.99]

    fig = plt.figure(figsize=(10,5))

    labels = []
    for n_decay in n_decays:
        for idx in range(N_TRY):
            filename = csv_filename.format(n_decay, idx)
            df = pd.read_csv(filename, index_col = 0)
            label = "Noise decay:{}".format(n_decay)
            labels.append(label)
            plt.plot( df.index, 'average_scores', data=df,
                lw=2, label=label, alpha=0.8)

    #Use same color and only one label if labels are same
    ax = fig.gca()
    printed_labels = []
    for i, p in enumerate(ax.get_lines()):
        if p.get_label() in labels:
            idx = labels.index(p.get_label())
            p.set_c(ax.get_lines()[idx].get_c())
            if p.get_label() in printed_labels:
                p.set_label('_' + p.get_label())
            else:
                printed_labels.append(p.get_label())

    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left')
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_learning_curve(filename):
    csv_filename = '../scores/{:s}.csv'.format(filename)
    plt_filename = '../docs/plots/{:s}.png'.format(filename)

    df = pd.read_csv(csv_filename, index_col = 0)
    plt.figure(figsize=(10,5))
    plt.axhline(0.5, color='red', lw=1, alpha=0.3) # Draw goal line
    plt.plot( df.index, 'scores', data=df, color='lime', lw=1, label="score per episode", alpha=0.4)
    plt.plot( df.index, 'average_scores', data=df, color='green', lw=2, label="average score")
    # Set labels and legends
    plt.xlabel('Episode')
    plt.xlim(0, len(df.index))
    plt.ylabel('Score')
    plt.title(filename)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    # Save figure
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

#Uncomment the necessary plot and run

# Plot learning rate
#plot_lr()

# Plot tau
#plot_tau()

# Plot noise scale
#plot_noise_scale()

# Plot neural network size
#plot_noise_decay()

# Plot learning curve of particular training
plot_learning_curve('MADDPG_lra1E-03_lrc1E-03_tau1E-01_nstart7.0_ndecay0.999_solved369')
