# from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import numpy as np  

from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
from IPython import get_ipython

import matplotlib.pyplot as plt
import random
import os

from gensim.models.word2vec import Word2Vec

from cleaning import get_filtered_df
from preprocessing import preprocess

DIR = os.path.dirname(__file__)
VIZ_PATH = os.path.join(DIR, 'viz/word-embedding-plot.html')

def reduce_dimensions(model: Word2Vec):
    num_dimensions = 2 

    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)

    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=False):

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename= VIZ_PATH)


def plot_with_matplotlib(x_vals, y_vals, labels):

    random.seed(420)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 200)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    
    plt.show(block= True)

if __name__ == "__main__":
    df = get_filtered_df()
    _, model = preprocess(df)
    x_vals, y_vals, labels = reduce_dimensions(model)
    plot_with_plotly(x_vals, y_vals, labels)