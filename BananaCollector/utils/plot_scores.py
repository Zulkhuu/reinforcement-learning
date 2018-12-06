import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from cycler import cycler

df = pd.read_csv('BananaCollector/scores/lr_batchsize_fc_tuning.csv', index_col = 0)
#scores.head()

fig, ax = plt.subplots()

df = df.groupby(['batch_size', 'learning_rate', 'fc_units']).mean().reset_index()
print(df)

tdf = df.groupby(['batch_size', 'fc_units']).mean().reset_index()
tdf = tdf[['batch_size', 'fc_units']]

for index, row in tdf.iterrows():
    print("batch_size:{:}, fc_units:{:}".format(row['batch_size'], row['fc_units']))
    plt.plot( 'learning_rate', 'i_episode',
        data=df.loc[(df['fc_units'] == row['fc_units']) & (df['batch_size'] == row['batch_size'])],
        lw=2, label="batch_size:{:} fc_units:{:}".format(row['batch_size'], row['fc_units']), alpha=0.8)

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

plt.xlabel('Learning rate')
plt.xscale("log")
plt.xticks([5e-5, 1e-4, 5e-4, 1e-3])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().get_major_formatter().set_useOffset(True)

plt.ylabel('Episode')
plt.title('DDQN')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend()

#plt.savefig('dqn.png', bbox_inches='tight')
plt.show()

# Plot scores
plt.figure(figsize=(10,5))
#plt.axhline(13, color='red', lw=1, alpha=0.3)
plt.plot( df.index, 'scores', data=df, color='lime', lw=1, label="score", alpha=0.4)
plt.plot( df.index, 'average_scores', data=df, color='green', lw=2, label="average score")
# Set labels and legends
plt.xlabel('Episode')
plt.xlim(0, len(df.index))
plt.xticks(50*np.arange(int(len(df.index)/50+1)))
plt.ylabel('Score')
plt.yticks(3*np.arange(8))
plt.title('DDQN agent')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend()
# Save figure
plt.savefig('graphs/{:s}.png'.format(filename), bbox_inches='tight')
