import numpy as np
import subprocess as sp
import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVNULL = open(os.devnull, 'w')


class DF_writer(object):
    def __init__(self, columns):
        self.df = pd.DataFrame(columns=columns)
        self.columns = columns

    def append(self, **row_data):
        if set(self.columns) == set(row_data):
            s = pd.Series(row_data)
            self.df = self.df.append(s, ignore_index=True)


def plot_results(df, target_lib="", audio_format="", ext="png"):
    sns.set_style("whitegrid")

    ordered_libs = df.time.groupby(
        df.lib
    ).mean().sort_values().index.tolist()

    fig = plt.figure()

    g = sns.catplot(
        x="duration", 
        y="time", 
        kind='point',
        hue_order=ordered_libs,
        hue='lib', 
        data=df,
        height=6.6, 
        aspect=1,
    )

    g.savefig("benchmark_%s_%s_dur.%s" % (target_lib, audio_format, ext))

    fig = plt.figure()
    ax = sns.barplot(x="time", y="lib", data=df, order=ordered_libs, orient='h')
    fig.savefig("benchmark_%s_%s_bar.%s" % (target_lib, audio_format, ext), bbox_inches='tight')