import numpy as np
import matplotlib.pyplot as plt
class SimpleGaussianProcess:
    def __init__(self):
        self.thetas = [1,1,1,1,0.0001]
        self.covariance = []
        self.mu = []
    
    def _kernel(self, x, y):
        exponential = self.thetas[0] * np.exp(-0.5 * self.thetas[1] * np.sum((x - y)**2))
        bias = self.thetas[2]
        linear = self.thetas[3] * np.dot(x,y)
        return exponential + bias + linear
    
    def draw_multivariate_gaussian(self,mu,C):
        try:
            ndim = len(mu)
        except TypeError as e:
            ndim = 1            
        if ndim == 1:
            return mu + np.random.normal(mu, C, 1)
        z = np.random.standard_normal(ndim)
        [U,S,V] = np.linalg.svd(C)
        A = U * np.sqrt(S)
        return mu + np.dot(A,z)
    
    def train(self, data):
        self.covariance = np.reshape([self._kernel(x,y) for x in data for y in data], (len(data),len(data))) + np.linalg.inv(self.thetas[-1]*np.identity(len(data)))
        self.mu = np.zeros(len(data))
        
    def predict(self, x, data):
        ''' data 들어오면 기존 covariance matrix 활용해서 새로운 mu, covariance matrix 구하고
         그걸 가지고 새로운 데이터 포인트에서의 prediction 값 t를 예측한다'''
        new_cov = np.array([self._kernel(x,y) for y in data[0]])
        pred_mu = np.matmul(np.matmul(new_cov, np.linalg.inv(self.covariance)), data[1].T)
        pred_sigma = self._kernel(x,x) - np.matmul(np.matmul(new_cov, np.linalg.inv(self.covariance)),new_cov.T)        
        outcome = self.draw_multivariate_gaussian(pred_mu, pred_sigma)
        
        return outcome
        
        # self.covariane = np.concatenate(self.covariance, new_cov, axis = 1)
        # new_cov = new_cov.append(self.kernel(x,x) + self.thetas[4])
        # self.covariance = np.concatenate(self.covariance, new_cov, axis = 0)