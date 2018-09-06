import numpy as np

class NormalInverseWishart:
    '''
    Conjugate Prior of Multivariate Gaussian Distribution
    
    To reduce the heavy burden of calculating squared sum of all previous data points ∑_i(x_i * x_i.T)
    maintain the ∑_i(x_i * x_i.T) so for every new data point, only calculate x_i * x_i.T

    (l, mu, v, psi)  are the instance variables needed for calculating the likelihood of multivariate_t_distribution

    '''

    def __init__(self, dim):

        # self.sum: ∑_i(x_i * x_i.T), update when every data point is added or deleted
        # self.x_hat: mean of x points, calculated using incremental update equation
        # self.l, self.v, self.psi, self.mu0: initialized hyperparameters
        # self.num: number of data added for updating the prior
        
        self.dim = dim
        self.x_hat = np.zeros(dim) # [0 for i in range(dim)]
        self.sum = np.zeros((dim, dim)) #[0 for i in range(dim)]
        self.mu0 = np.zeros(dim)
        self.l = 1
        self.v = dim 
        self.psi = np.identity(dim)
        self.num = 0
        

    def additem(self, data):

        # Used for updating the x_hat, num, sum instance variables for later calculating new parameters for likelihood

        if self.num == 0:
            self.num += 1
            self.x_hat += data
            self.sum += np.dot(np.reshape(np.array(data), (dim, -1)), np.reshape(np.array(data), (-1, dim)))
        else:
            self.num += 1
            self.x_hat = self.x_hat + (self.x_hat - data) / self.num
            self.sum += np.dot(np.reshape(np.array(data), (dim, -1)), np.reshape(np.array(data), (-1, dim)))

    def delitem(self, data):

        # Used for updating the x_hat, num, sum instance variables for later calculating new parameters for likelihood
        # Because Gibbs Sampling is used for Dirichlet Process Gaussian Mixture model (in this case), delitem is used for sampling

        self.num -= 1
        self.x_hat = (self.x_hat - data / self.num) * self.num / (self.num - 1)
        self.sum -= np.dot(np.reshape(np.array(data), (dim, -1)), np.reshape(np.array(data), (-1, dim)))

    def likelihood(self, data):

        # likelihood method returns parameters needed for calculating likelihood of Poseterior predictive, which is multivariate-T-dist
        
        l = self.l + self.num
        mu = (self.l * self.mu0 + self.num * self.x_hat) / l
        v = self.v + self.num
        psi = self.psi + self.sum - \
                    (self.l + self.num) * np.dot(np.reshape(mu, (dim, -1)), np.reshape(mu, (-1, dim))) + \
                    self.l * np.dot(np.reshape(self.mu0, (dim, -1)), np.reshape(self.mu0, (-1, dim)))
        return l, mu, v, psi
