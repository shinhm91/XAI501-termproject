from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP

def create_basic_prep_pipe():

    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ])

    return pipe


def create_dim_red_prep_pipe(n_components=2):
    """
    Returns three dim reduction pipeline followed by scaling.
    """

    pca_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
        ])

    kernel_pca_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kpca', KernelPCA(n_components=n_components, kernel='rbf'))
        ])

    umap_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('umap', UMAP(n_components=n_components, n_neighbors=15))
        ])

    return pca_pipe, kernel_pca_pipe, umap_pipe