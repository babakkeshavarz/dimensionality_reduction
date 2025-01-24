import pandas as pd
import gzip
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def view_digit(example):
    label = y_train.loc[example]
    image = X_train.loc[example].values.reshape(28, 28)
    plt.title('Example: %d  Label: %d' % (example, label))
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()

def scatterPlot(xDF, yDF, algoName):
    tempDF = pd.DataFrame(data=xDF.loc[:, 0:1], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join='inner')
    tempDF.columns = ['First Vector', 'Second Vector', 'Label']
    sns.lmplot(x='First Vector', y='Second Vector', data=tempDF, hue='Label', fit_reg=False)
    ax = plt.gca()
    ax.set_title(algoName)
    plt.show()


file_path = '..\\datasets_unsupervised_aapatel\mnist_data\\mnist.pkl.gz'

f = gzip.open(file_path , 'rb')

train_set, validation_set, test_set = pickle.load(f, encoding='latin1')

f.close()

X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)



# view_digit(1)

############### Linear dimensionality reduction techniques ##################

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import SparsePCA


n_components = 784
random_state = 42

pca = PCA(n_components=n_components, random_state=random_state)
incrementalPCA = IncrementalPCA(n_components=n_components, batch_size=None) ## when the data is too large
sparsePCA = SparsePCA(n_components=n_components, alpha= 0.0001 , random_state=random_state , n_jobs=-1) ## when some degree of sparsity is required

X_train_pca = pca.fit_transform(X_train)
X_train_pca = pd.DataFrame(X_train_pca)

print("Variance of the first 10 dimensions: ", pca.explained_variance_ratio_[0:10])
print("Total information of PCA: ", sum(pca.explained_variance_ratio_))
print("Variance of the first 10 dimensions: ", sum(pca.explained_variance_ratio_[0:10]))


scatterPlot(X_train_pca, y_train, "PCA")



############### non-Linear dimensionality reduction techniques ##################
from sklearn.decomposition import KernelPCA
n_components = 100
kernel = 'rbf'
gamma = None
random_state = 42
n_jobs = -1

KernelPCA = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=random_state, n_jobs=n_jobs)

X_train_kernelpca= KernelPCA.fit_transform(X_train.loc[0:1000 , :])
X_train_kernelpca = pd.DataFrame(X_train_kernelpca)

# scatterPlot(X_train_kernelpca, y_train, "Kernel PCA")


################ SVD ######################################
# SVD is more efficient since it foregoes the calculation of the covariance matrix
from sklearn.decomposition import TruncatedSVD

n_components = 200
algorithm = 'randomized'
n_iter = 5
random_state = 42

svd = TruncatedSVD(n_components=n_components, algorithm=algorithm, n_iter=n_iter, random_state=random_state)

X_train_svd = svd.fit_transform(X_train)
X_train_svd = pd.DataFrame(X_train_svd)

scatterPlot(X_train_svd, y_train, 'SVD')


##################### Random Projection  ############################
# In general PCA works well on relatively low dimensional data. Random Projection is a good alternative for high dimensional data.
from sklearn.random_projection import GaussianRandomProjection

n_components = 'auto'
eps = 0.5
random_state = 42

GRP = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)

X_train_GRP = GRP.fit_transform(X_train)
X_train_GRP = pd.DataFrame(X_train_GRP)

scatterPlot(X_train_GRP, y_train, 'GRP')