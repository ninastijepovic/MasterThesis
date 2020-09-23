from sklearn.manifold import TSNE


def embed(features):
    if features is None or len(features.shape) != 2:
        raise ValueError('Data is not in correct format: np.array with shape (n, features)')
    #model_cnn = umap.UMAP()
    #return model_cnn.fit_transform(features)

    model_cnn = TSNE(n_components=2, random_state=0)
    return model_cnn.fit_transform(features)
