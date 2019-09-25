#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

loss = pd.read_csv('run_003-tag-loss.csv', index_col=0)
val_loss = pd.read_csv('run_003-tag-val_loss.csv', index_col=0)
sns.relplot(x='Epoch', y='Loss', kind='line', data=loss)
sns.relplot(x='Epoch', y='Val_loss', kind='line', data=val_loss)
plt.show()


