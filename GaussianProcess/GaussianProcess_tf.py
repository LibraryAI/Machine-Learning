import tensorflow as tf
import random
import numpy as np

class GaussianProcess:
    def __init__(self):
        self.thetas = tf.Variable(tf.random_normal([5]))
        self.learning_rate = tf.placeholder(tf.float32, [1])
        self.num_iteration = tf.placeholder(tf.int64, [1])

    def kernel(self,x1,x2):
        exponential = tf.mul(tf.div(tf.slice(self.thetas, [1], [1]), 2.0), tf.sum(np.dot((x1-x2), (x1-x2))))
        exponential = tf.mul(tf.slice(self.thetas, [0], [1]), exponential)
        bias = tf.slice(self.thetas, [2], [1])
        lienar = tf.mul(tf.slice(self.thetas, [3], [1]), np.dot(x1,x2))
        outcome = tf.add(tf.add(exponentiial, bias), linear)
        return outcome
    
    def training(self, X, Y):
        '''하이퍼파라미터 옵티마이제이션            
            1. kernel method를 통해 Xdata의 covariance matrix 계산, 0 mean 
            2. log derivative of p(T|theta) w.r.t. thetas 계산
            3. minimize
            '''
        numData = len(Y)
        numDimension = len(X[0])
        
        obsX = tf.placeholder(tf.float32, [numData, numDimension])
        obsY = tf.placeholder(tf.float32, [numData, 1])
        
        # self.cov = tf.reshape([self.kernel(x1,x2) for x1 in obsX for x1 in obsX], shape = [numData, numData])
        # self.cov = tf.inv(tf.mul(tf.slice(self.thetas, [4], [1]), np.identity(numData)) + self.cov
        covLinear = []
        for i in range(numData):
            for j in range(numData):
                kernel_output = self.kernel(tf.slice(obsX, [i, 0], [1, numDimension]), tf.slice(obsX, [j, 0], [1, numDimension]))
                
                if i != j:
                    covLinear.append(kernel_output)
                else:
                    covLinear.append(kernel_output + tf.div(1.0, tf.slice(tf.thetas, [4], [1])))
        
        cov = tf.stack(covLinear)
        self.cov = tf.reshape(cov, [numData,numData])
        covInv = tf.inv(self.cov)
        
        negloglikelihood = 0
        for i in range(numData):
            k = tf.Variable(tf.ones([numData]))
            for j in range(numData):
                kernel_output = self.kernel(tf.slice(obsX, [i, 0], [1, numDimension]), 
                                            tf.slice(obsX, [j, 0], [1, numDimension]))
                indices = tf.constant([j])
                tempTensor = tf.Variable(tf.zeros([1]))
                tempTensor = tf.add(tempTensor, kernel_output)
                tf.scatter_update(k, tf.reshape(indices, [1,1]), tempTensor)
            
            c = tf.Variable(tf.zeros([1,1]))
            kernel_output = self.kernel(tf.slice(obsX, [i, 0], [1, numDimension]), 
                                        tf.slice(obsX, [i, 0], [1, numDimension]))
            c = tf.add(tf.add(c, kernel_output), tf.div(1.0, tf.slice(tf.thetas, [4], [1])))
            k= tf.reshape(k, [1, numData])
            
            pred_mu = tf.matmul(k, tf.matmul(covInv, obsY))
            pred_var = tf.sub(c, tf.matmul(tf.matmul(k, covInv), tf.transpose(k)))
            
            negloglikelihood = tf.add(negloglikelihood, tf.div(tf.pow(tf.sub(pred_mu, tf.slice(obsY, [i,0], [1,1])), 2), tf.scalar_mul(tf.constant(2.0), pred_var)))
        
        training = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(negloglikelihood)
        
    def predict(self, X, t_X, t_Y):
        '''새로운 데이터포인트(점)에 대한 예측값 생성'''
        numDimension = len(X[0])
        numData = len(tf.shape(t_X)[0])
        
        obs_X = tf.placeholder(tf.float32, [1, numDimension])
        train_X = tf.placeholder(tf.float32, [numData, numDimension])
        train_Y = tf.placeholder(tf.float32, [numData, 1])
        new_cov = []
        for i in range(tf.shape(self.cov)[0]):
            kernel_output = self.kernel(obs_X, tf.slice(train_X, [i, numDimension]))
            new_cov.append(kernel_output)
        new_cov = tf.reshape(tf.stack(new_cov), [1, numData])
        pred_mu = tf.matmul(tf.matmul(new_cov, tf.inv(self.cov)), train_Y)
        pred_sigma = tf.sub(tf.reshape(self.kernel(obs_X, obs_X), [1, 1]), 
                            tf.matmul(tf.matmul(new_cov, tf.inv(self.cov)), tf.transpose(new_cov)))
        return pred_mu, pred_sigma
        
    def sample(self, mean, cov):
        '''주어진 mean, cov의 dim에 맞춰 sampling'''
        
        num_data = len(tf.shape(cov)[0])
        if num_data == 1:
            return tf.add(mean, tf.random_normal([1], mean=mean, stddev=cov))
        z = tf.random_normal([num_data])
        s, u, v = tf.linalg.svd(cov)
        a = matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
        return tf.add(mean, tf.matmul(a, z))
