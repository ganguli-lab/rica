import numpy as np
import theano
import theano.tensor as T
from sfo.sfo import SFO
from skimage.filter import gabor_kernel
from skimage.transform import resize

def generate_data(ndim, nsamples, nfeatures):
    """Generate data by drawing samples that are a sparse combination of gabor features
    """

    # build features
    features = list()
    for j in range(nfeatures):
        theta = np.pi * np.random.rand()
        sigma = 2*np.random.rand() + 1
        freq = 0.2 * np.random.rand() + 0.05
        features.append(resize(np.real(gabor_kernel(freq, theta=theta, sigma_x=sigma, sigma_y=sigma)), (ndim, ndim)).ravel())
    features = np.vstack(features)

    # draw sparse combinations
    X = np.random.laplace(size=(nsamples, nfeatures)) .dot (features)

    return X, features

class rica(object):

    def __init__(self, W0, X, penalty=0.01):

        assert W0.shape[1] == X.shape[1], 'Dimensions do not match. W: (k,n), X: (m,n)'

        self.penalty = penalty
        self.W_init = W0.copy()
        print('Compiling ...')

        # build objective and gradientV
        W = T.dmatrix('W')
        x = T.dvector('x')
        u = W.dot(x)
        obj = 0.5 * T.sum((W.T.dot(u) - x).ravel() ** 2) + self.penalty * T.sum(T.log(T.cosh(u)))

        self.f = theano.function([W, x], obj)
        self.df = theano.function([W, x], T.grad(obj, W))

        # initialize optimizer
        self.X = X
        self.data = [x for x in X]
        self.optimizer = SFO(self.f_df, W0, self.data)

        print('Done.')

    def f_df(self, W, x):
        return self.f(W, x), self.df(W, x)

    def check_grad(self):
        self.optimizer.check_grad()

    def fit(self, num_passes=10):
        W_hat = self.optimizer.optimize(num_passes=num_passes)
        self.coeffs = W_hat
        self.filters = W_hat.dot(self.X.T)

if __name__ == "__main__":

    # generate fake data
    ndim = 64
    nfeat = 4
    nsamples = 1000

    X, features = generate_data(int(np.sqrt(ndim)), nsamples, nfeat)

    # build RICA object
    W0 = np.random.randn(nfeat ,ndim)
    model = rica(W0, X, penalty=0.1)

    # optimize
    model.fit(num_passes=1)