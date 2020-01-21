import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


for package in ['pytorch', 'tf', 'np', 'metadata']:
    dfs = []
    for ext in ["wav", "mp3", "mp4", "ogg", "flac"]:
        dfs.append(
            pd.read_pickle("results/benchmark_%s_%s.pickle" % (package, ext))
        )

    df = pd.concat(dfs, ignore_index=True)

    sns.set_style("whitegrid")

    ordered_libs = df.time.groupby(
        df.lib
    ).mean().sort_values().index.tolist()

    fig = plt.figure()

    g = sns.catplot(
        x="time",
        y="lib",
        kind='bar',
        hue='ext',
        order=ordered_libs,
        data=df,
        height=6.6,
        aspect=1,
        legend=False
    )
    g.set(xscale="log")
    g.despine(left=True)
    plt.legend(loc='upper right')
    g.savefig("results/benchmark_%s.png" % package)
