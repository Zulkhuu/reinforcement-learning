import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from cycler import cycler

def plot_batch_vs_lr_fc():
    csv_filename = '../scores/lr_batchsize_fc_tuning.csv'
    plt_filename = '../docs/plots/batch_size_vs_learning_rate_fcunit.png'
    df = pd.read_csv(csv_filename, index_col = 0)

    fig, ax = plt.subplots()
    df = df.groupby(['batch_size', 'learning_rate', 'fc_units']).mean().reset_index()

    tdf = df.groupby(['learning_rate', 'fc_units']).mean().reset_index()
    tdf = tdf[['learning_rate', 'fc_units']]

    for index, row in tdf.iterrows():
        #print("batch_size:{:}, fc_units:{:}".format(row['batch_size'], row['fc_units']))
        plt.plot( 'batch_size', 'i_episode',
            data=df.loc[(df['fc_units'] == row['fc_units']) & (df['learning_rate'] == row['learning_rate'])],
            lw=2, label="learning_rate:{:} fc_units:{:}x{:}".format(row['learning_rate'], row['fc_units'], row['fc_units']), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Batch size')
    plt.xticks([32, 64, 128, 256])
    plt.ylabel('Episode')
    plt.title('Batch size vs learning rate and neural network layer size')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_buffer_vs_lr_updatet():
    csv_filename = '../scores/lr_buffersize_update_tuning.csv'
    plt_filename = '../docs/plots/buffer_size_vs_learning_rate_updatet.png'
    df = pd.read_csv(csv_filename, index_col = 0)

    fig, ax = plt.subplots()
    df = df.groupby(['buffer_size', 'learning_rate', 'update_every']).mean().reset_index()

    tdf = df.groupby(['learning_rate', 'update_every']).mean().reset_index()
    tdf = tdf[['learning_rate', 'update_every']]

    for index, row in tdf.iterrows():
        #print("batch_size:{:}, fc_units:{:}".format(row['batch_size'], row['fc_units']))
        plt.plot( 'buffer_size', 'i_episode',
            data=df.loc[(df['update_every'] == row['update_every']) & (df['learning_rate'] == row['learning_rate'])],
            lw=2, label="learning_rate:{:} update_every:{:}".format(row['learning_rate'], row['update_every']), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Buffer size')
    plt.xticks([50000, 100000, 200000])
    plt.ylabel('Episode')
    plt.title('Batch size vs learning rate and neural network layer size')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_lr_vs_batch_fc():
    csv_filename = '../scores/lr_batchsize_fc_tuning.csv'
    plt_filename = '../docs/plots/learning_rate_vs_batchsize_fcunit.png'
    df = pd.read_csv(csv_filename, index_col = 0)

    fig, ax = plt.subplots()
    df = df.groupby(['batch_size', 'learning_rate', 'fc_units']).mean().reset_index()

    tdf = df.groupby(['batch_size', 'fc_units']).mean().reset_index()
    tdf = tdf[['batch_size', 'fc_units']]

    for index, row in tdf.iterrows():
        plt.plot( 'learning_rate', 'i_episode',
            data=df.loc[(df['fc_units'] == row['fc_units']) & (df['batch_size'] == row['batch_size'])],
            lw=2, label="batch_size:{:} fc_units:{:}x{:}".format(row['batch_size'], row['fc_units'], row['fc_units']), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Learning rate')
    plt.xscale("log")
    plt.xticks([5e-5, 1e-4, 5e-4, 1e-3])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().get_major_formatter().set_useOffset(True)

    plt.ylabel('Episode')
    plt.title('Learning rate vs batch_size and neural network layer size')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_lr_vs_buffer_updatet():
    csv_filename = '../scores/lr_buffersize_update_tuning.csv'
    plt_filename = '../docs/plots/learning_rate_vs_buffersize_updateinterval.png'
    df = pd.read_csv(csv_filename, index_col = 0)

    fig, ax = plt.subplots()
    df = df.groupby(['buffer_size', 'learning_rate', 'update_every']).mean().reset_index()

    tdf = df.groupby(['buffer_size', 'update_every']).mean().reset_index()
    tdf = tdf[['buffer_size', 'update_every']]

    for index, row in tdf.iterrows():
        plt.plot( 'learning_rate', 'i_episode',
            data=df.loc[(df['update_every'] == row['update_every']) & (df['buffer_size'] == row['buffer_size'])],
            lw=2, label="buffer_size:{:} update_every:{:}".format(row['buffer_size'], row['update_every']), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Learning rate')
    plt.xscale("log")
    plt.xticks([5e-5, 1e-4, 5e-4, 1e-3])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().get_major_formatter().set_useOffset(True)

    plt.ylabel('Episode')
    plt.title('Learning rate vs buffer size and update interval')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.savefig(plt_filename, bbox_inches='tight')
    plt.show()

def plot_lr_vs_buffer():
    csv_filename = '../scores/lr_buffersize_update_tuning.csv'
    plt_filename = '../docs/plots/learning_rate_vs_buffersize.png'
    df = pd.read_csv(csv_filename, index_col = 0)

    fig, ax = plt.subplots()
    df = df.groupby(['buffer_size', 'learning_rate']).mean().reset_index()

    tdf = df.groupby(['buffer_size']).mean().reset_index()
    tdf = tdf[['buffer_size']]
    print(df)
    print(tdf)
    for index, row in tdf.iterrows():
        plt.plot( 'learning_rate', 'i_episode',
            data=df.loc[(df['buffer_size'] == row['buffer_size'])],
            lw=2, label="buffer_size:{:}".format(row['buffer_size']), alpha=0.8)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.xlabel('Learning rate')
    plt.xscale("log")
    plt.xticks([5e-5, 1e-4, 5e-4, 1e-3])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().get_major_formatter().set_useOffset(True)

    plt.ylabel('Episode')
    plt.title('Learning rate vs buffer size')
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

# Plot learning curve of particular training
plot_learning_curve('DDQN_lr1.0E-03_batch256_model64x64_buffer100000_update4_solved410')

# Plot learning rate vs batch_size and fully connected units of neural network
#plot_lr_vs_batch_fc()

# Plot learning rate vs buffer size and update interval
#plot_lr_vs_buffer_updatet()

# Plot learning rate vs buffer size and update interval
#plot_lr_vs_buffer()

# Plot buffer size vs learning rate and update interval
#plot_buffer_vs_lr_updatet()

# Plot Batch_size vs learning rate and fully connected units of neural network
#plot_batch_vs_lr_fc()
