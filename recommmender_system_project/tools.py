import pandas
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings: pandas.DataFrame) -> None:
    """Visualize the embeddings of the items in 2d.

    Parameters
    ----------
    embeddings : pandas.DataFrame
        A dataframe with item embeddings.

    Returns
    -------
    None
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000,learning_rate=10)
    tsne_results = tsne.fit_transform(embeddings)

    plt.scatter(tsne_results[:,0], tsne_results[:,1])
    plt.show()