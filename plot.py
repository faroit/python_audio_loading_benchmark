import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


for package in ['np', 'pytorch', 'tf', 'metadata']:
    dfs = []
    for ext in ["wav", "mp3", "mp4", "ogg", "flac"]:
        try:
            dfs.append(
                pd.read_pickle("results/benchmark_%s_%s.pickle" % (package, ext))
            )
        except FileNotFoundError:
            continue

    df = pd.concat(dfs, ignore_index=True)

    sns.set_style("whitegrid")

    ordered_exts = df.time.groupby(
        df.ext
    ).mean().sort_values().index.tolist()

    fig = plt.figure()

    g = sns.catplot(
        x="time",
        y="ext",
        kind='bar',
        hue='lib',
        order=ordered_exts,
        data=df,
        height=6.6,
        aspect=1,
        legend=False
    )
    g.set(xscale="log")
    g.despine(left=True)
    plt.legend(loc='upper right')
    g.savefig("results/benchmark_%s.png" % package)
