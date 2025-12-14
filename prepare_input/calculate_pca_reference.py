import numpy as np
import sklearn.decomposition

def pca_ref(imgs, nComps=3):
    T = imgs.shape[2]
    mu = np.zeros(T)
    for t in range(T):
        temp = imgs[:, :, t]
        mu[t] = np.mean(temp)
        imgs[:, :, t] = temp - mu[t]

    imgs1 = imgs.reshape(imgs.shape[0] * imgs.shape[1], imgs.shape[2])

    pca = sklearn.decomposition.PCA()
    pca.fit(imgs1)

    new_imgs = np.dot(pca.transform(imgs1)[:, :nComps], pca.components_[:nComps, :])

    new_imgs = new_imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2])

    for t in range(T):
        new_imgs[:, :, t] = new_imgs[:, :, t] + mu[t]

    return new_imgs