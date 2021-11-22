from xai.data.mvtec import MVTec_AD
from xai.preprocessing.utils import create_basic_prep_pipe, create_dim_red_prep_pipe


# load val, train data
DB_PATH = "./dataset/MVTecAD"
OUT_PATH = "./out/"

size = (86, 86)
cls = 'bottle'
dataset = MVTec_AD(DB_PATH, OUT_PATH, flatten=True, size=size)

# Split Train / Validation / Test data

train = dataset.read_mvtec(cls=cls, mode='train')
val = dataset.read_mvtec(cls=cls, mode='val')
test = dataset.read_mvtec(cls=cls, mode='test')

assert train['imgs'].shape == (209, 7396)
assert val['imgs'].shape == (53, 7396)
assert test['imgs'].shape == (83, 7396)

print("Data split completed")

## Preprocess

scaling_prep = create_basic_prep_pipe()
pca_pipe, kernel_pca_pipe, umap_pipe = create_dim_red_prep_pipe(n_components=3)

scaled_train_tfed = scaling_prep.fit_transform(train['imgs'])
pca_train_tfed = pca_pipe.fit_transform(train['imgs'])
kpca_train_tfed = kernel_pca_pipe.fit_transform(train['imgs'])
umap_train_tfed = umap_pipe.fit_transform(train['imgs'])

print("Preprocessing completed")

## Model

## SKLEARN PIPELINE (Last module: Estimator)

## Eval

## visualize