import numpy as np
class NormalInverseWishart:
    def __init__(self, dim):
        self.dim = dim
        self.x_hat = np.zeros(dim) # [0 for i in range(dim)]
        self.sum = np.zeros((dim, dim)) #[0 for i in range(dim)]
        self.mu0 = np.zeros(dim)
        self.l = 1
        self.v = dim 
        self.psi = np.identity(dim)
        self.num = 0
        

    def additem(self, data):
        if self.num == 0:
            self.num += 1
            self.x_hat += data
            self.sum += np.dot(np.reshape(np.array(data), (dim, -1)), np.reshape(np.array(data), (-1, dim)))
        else:
            self.num += 1
            self.x_hat = self.x_hat + (self.x_hat - data) / self.num
            self.sum += np.dot(np.reshape(np.array(data), (dim, -1)), np.reshape(np.array(data), (-1, dim)))

    def delitem(self, data):
        self.num -= 1
        self.x_hat = (self.x_hat - data / self.num) * self.num / (self.num - 1)
        self.sum -= np.dot(np.reshape(np.array(data), (dim, -1)), np.reshape(np.array(data), (-1, dim)))

    #def update_prior(self, data):
    #    l, mu, v, psi = self.likelihood(data)
    #    self.psi = psi
    #    self.l = l
    #    self.mu0 = mu
    #    self.v = v

    def likelihood(self, data):
        l = self.l + self.num
        mu = (self.l * self.mu0 + self.num * self.x_hat) / l
        v = self.v + self.num
        psi = self.psi + self.sum - \
                    (self.l + self.num) * np.dot(np.reshape(mu, (dim, -1)), np.reshape(mu, (-1, dim))) + \
                    self.l * np.dot(np.reshape(self.mu0, (dim, -1)), np.reshape(self.mu0, (-1, dim)))
        return l, mu, v, psi