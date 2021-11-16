# This is just a copy-paste of the functions in the notebook, preferably look at the notebook to see the different functions associated with a visualization

from gensim.models import word2vec
from cleaning import get_filtered_df
from preprocessing import preprocess

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

_, w2v_model = preprocess(get_filtered_df())

def similar_words(w2v_model: word2vec.Word2Vec):
    return {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=6)] for search_term in ['vin', 'viand', 'fruit', 'legum', 'fromag', 'pain', 'farin', 'sucre', 'lait']}

labels_from_tsne = sum([[k] + v for k, v in similar_words().items()], [])

def plot_cluster1():
    words = sum([[k] + v for k, v in similar_words().items()], [])
    wvs = w2v_model.wv[words]

    tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(wvs)
    labels_from_tsne = words

    plt.figure(figsize=(14, 8))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels_from_tsne, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

def plot_cluster2():
    words = w2v_model.wv.index_to_key
    wvs = w2v_model.wv[words]

    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(wvs)
    labels = words

    plt.figure(figsize=(12, 6))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

    return TSNE

T = plot_cluster2().fit_transform(w2v_model.wv[labels_from_tsne])

def plot_dbscan():
    X = StandardScaler().fit_transform(T)

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=3).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Plot result

    plt.figure(figsize=(15, 8))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    for label, x, y in zip(labels_from_tsne, X[:, 0], X[:, 1]):
        plt.annotate(label, xy=(x + 0.01, y + 0.01), xytext=(0, 0), textcoords='offset points')
    plt.show()