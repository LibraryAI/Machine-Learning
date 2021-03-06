'''
GaussianProcessClassifier
'''

import tensorflow as tf
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt

### Binomial Gaussian Process Classifier ###

class BinomialGPC:
	'''
	#### Likelihood ####
	class 가 두개인 심플 케이스라면
	p(Y=t_i|Θ) = σ(f(x;Θ))^t × (1-σ(f(x;Θ)))^(1-t)
	의 형태가 곧 one data point 가 t_i 로 assign 될 확률, 혹은 likelihood
	
	p(Y|Θ) = ∏(p(Y=t_i|Θ))

	Gaussian Process 를 사용한 Logistic classifier니깐 
	f(x;Θ) = outcome of gaussian = mu of predicted datapoint
	
	#### Prior ####
	p(Y|x,Θ) 에서 Θ 가 어디서 왔냐를 바라보는 측면에서 Bayesian Inference가 됨
	prior p(Θ ; x) = ∏(p(Y=t_i|Θ))
	'''
    def __init__(self):
        
        # thetas: θ_1, θ_2, θ_3, θ_4, θ_5, β

        self.thetas = tf.Variable(tf.random_normal([5]))

    def kernel(self,x1,x2):
        '''
        # 커널은 mapping function 에 의해 다른 feature space 로 변환된 두 vector 사이의 내적
        # 이 커널 함수는 세개의 커널로 이루어짐

               1) degree 2 의 polynomial kernel function ( exponential component)
                  = θ_0 * exp(-Θ_1 / 2 * l2norm(x1, x2)^2)

               2) degree 1 의 linear component
                  = Θ_3 * x1.T * x2

               3) Bias component
                  = Θ_2
        '''
        exponential = tf.multiply(tf.div(tf.slice(self.thetas, [1], [1]), 2.0), tf.reduce_sum(tf.multiply(np.subtract(x1, x2), np.subtract(x1, x2))))
        exponential = tf.multiply(tf.slice(self.thetas, [0], [1]), exponential)
        bias = tf.slice(self.thetas, [2], [1])
        linear = tf.multiply(tf.slice(self.thetas, [3], [1]), tf.reduce_sum(tf.multiply(x1,x2)))
        outcome = tf.add(tf.add(exponential, bias), linear)
        return outcome
    
    def train(self, X, Y, n_data):
        '''
        하이퍼파라미터 옵티마이제이션            
            1. kernel method를 통해 Xdata의 covariance matrix 계산, 0 mean 
            2. log derivative of p(T|theta) w.r.t. thetas 계산
            3. minimize
        '''
        numData = n_data # len(Y)
        numDimension = X.shape[1] # len(X[0])
        
        X = tf.cast(X, tf.float32)
        Y = tf.cast(Y, tf.float32)
        q_Y = tf.add(tf.scalar_mul(2.0, Y), -1.0)
        
        # self.cov = tf.reshape([self.kernel(x1,x2) for x1 in obsX for x1 in obsX], shape = [numData, numData])
        # self.cov = tf.inv(tf.multiply(tf.slice(self.thetas, [4], [1]), np.identity(numData)) + self.cov
        covLinear = []
        for i in range(numData):
            for j in range(numData):
                kernel_output = self.kernel(tf.slice(X, [i, 0], [1, numDimension]), tf.slice(X, [j, 0], [1, numDimension]))
                
                if i != j:
                    covLinear.append(kernel_output)
                else:
                    covLinear.append(tf.add(kernel_output, tf.div(1.0, tf.slice(self.thetas, [4], [1]))))
        
        cov = tf.stack(covLinear)
        cov = tf.convert_to_tensor(tf.reshape(cov, [numData,numData]), dtype=tf.float32)
        self.cov = cov
        
        covInv = self.pinv(cov)
        self.covInv = covInv


        #covInv = tf.matrix_inverse(self.cov)
        
        negloglikelihood = tf.constant([0], dtype = tf.float32)
        for i in range(numData):
            k = tf.Variable(tf.ones([numData]))
            for j in range(numData):
                kernel_output = self.kernel(tf.slice(X, [i, 0], [1, numDimension]), 
                                            tf.slice(X, [j, 0], [1, numDimension]))
                indices = tf.constant([j])
                tempTensor = tf.zeros([1])
                tempTensor = tf.add(tempTensor, kernel_output)
                tf.scatter_update(k, indices, tempTensor)
            
            c = tf.zeros([1])
            kernel_output = self.kernel(tf.slice(X, [i, 0], [1, numDimension]), 
                                        tf.slice(X, [i, 0], [1, numDimension]))
            c = tf.add(tf.add(c, kernel_output), tf.div(1.0, tf.slice(self.thetas, [4], [1])))
            k= tf.reshape(k, [1, numData])
            
            pred_mu = tf.matmul(tf.matmul(k, covInv), q_Y)
            pred_var = tf.subtract(c, tf.matmul(tf.matmul(k, covInv), tf.transpose(k)))

            probit = self.probit(pred_mu, pred_var)

            negloglikelihood = tf.add(negloglikelihood, 
            						  tf.negative(tf.add(tf.multiply(Y, tf.log(probit)), 
            						  	     tf.multiply(tf.subtract(1.0, Y), tf.log(tf.subtract(1.0, probit))))))
     
            return negloglikelihood

    def probit(self, mu, var):
    	mu_convol = tf.matmul(tf.sqrt(tf.add(1.0, tf.div(tf.scalar_mul(np.pi, var), 8))), mu)
        probit = tf.multiply(tf.div(1.0,tf.sqrt(tf.multiply(2.0, np.pi))), tf.exp(tf.multiply(tf.constant([-0.5]), tf.square(mu_convol))))	
        return probit
        
    def predict(self, obs_X, train_X, train_Y, n_data):

        '''
        새로운 데이터포인트(점)에 대한 
        Mean and covariance of P(t_new|T_train) 
        
        Multivariate normal distribution 의 conditional breakdown theorem을 이용해서 전체 covariance 중 새로 update 되는 부분만 구하면됨
        new_cov: new data point x_new 와 x_train 내의 전체 data의 kernel vector
        self.cov: x_train data 포인트 간의 kernel matrix
        
        μ_t_new = (new_cov.T) × (self.cov.inverse) × (train_Y)
        cov_t_new = (variance of x_new) － (new_cov.T) × (self.cov.inverse) × (new.cov)
        
        - Gaussian Process를 사용한 Regression 목적이라면 μ_t_new 가 정답~
        '''
        numDimension = obs_X.shape[1] # len(X[0])
        numData = n_data
        new_cov = []

        obs_X = tf.cast(obs_X, tf.float32)
        train_X = tf.cast(train_X, tf.float32)
        train_Y = tf.cast(train_Y, tf.float32)
        q_Y = tf.add(tf.scalar_mul(2.0, train_Y), -1.0)

        for i in range(tf.shape(self.cov)[0]):
            kernel_output = self.kernel(obs_X, tf.slice(train_X, [i, 0], [1, numDimension]))
            new_cov.append(kernel_output)
        new_cov = tf.reshape(tf.stack(new_cov), [1, numData])
        pred_mu = tf.matmul(tf.matmul(new_cov, tf.inv(self.cov)), train_Y)
        pred_sigma = tf.sub(tf.reshape(self.kernel(obs_X, obs_X), [1, 1]), 
                            tf.matmul(tf.matmul(new_cov, tf.inv(self.cov)), tf.transpose(new_cov)))
        probit = self.probit(pred_mu, pred_sigma)
        
        if probit[0] >= 0.5:
        	pred_class = 1
        else:
        	pred_class = -1
        return pred_class

    def pinv(self, A, reltol=1e-6):
        # Compute the SVD of the input matrix A
        s, u, v = tf.linalg.svd(A)
 
        # Invert s, clear entries lower than reltol*s[0].
        
        #atol = tf.reduce_max(s) * reltol
        #s = tf.boolean_mask(s, s > atol)

        s_inv = tf.diag(tf.div(1., s))
        return tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))

################################################################################################################################################################
n_epochs = 100
lr = .01
#batch_size = 64
#n_steps = int(X_train.shape[0] / batch_size)


total_features, total_target = load_breast_cancer(True)
total_target = list(map(lambda x: 1 if x== 'M' else 0, total_target))

train_features = scale(total_features[:250])
train_target = total_target[:250]
test_features = scale(total_features[250:350])
test_target = total_target[250:350]

numDimension = train_features.shape[1]

numDimension = train_features.shape[1]
numTrain = train_features.shape[0]
numTest = test_features.shape[0]

obs_X = tf.placeholder(tf.float32, [1, numDimension])
train_X = tf.placeholder(tf.float32, [None, numDimension])
train_Y = tf.placeholder(tf.float32, [None, 1])

test_X = tf.placeholder(tf.float32, [None, numDimension])
test_Y = tf.placeholder(tf.float32, [None, 1])

GP = BinomialGPC()
negloglikelihood = GP.train(train_X, train_Y, numTrain)
training_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss=negloglikelihood)

#sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#sess = tf.Session(config = sess_config)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

################################################################################################################################################################
points = [[], []]
for epoch in range(n_epochs):
    _, tr_loss = sess.run([training_op, negloglikelihood], feed_dict = {train_X : train_features,
                                                                       train_Y : train_target})
    
    if epoch % 10 == 0:
        points[0].append(epoch)
        points[1].append(tr_loss)

    if epoch % 25 == 0:
        print(tr_loss)

plt.plot(points[0], points[1], 'r--')
plt.axis([0, epochs, 50, 600])
plt.show()

################################################################################################################################################################
pred_op = GP.predict(test_X, train_X, train_Y)
points_pred = [[], []]
for i in range(test_features[0]):
    
    pred_mu, _ = sess.run(pred_op, feed_dict = {test_X : test_features[i],
                                                train_X : train_features
                                                train_Y : train_target})
    points_pred[0].append(i)
    points_pred[1].append(pred_mu)

plt.plot(points_pred[0], points_pred[1], 'r--')
plt.show()

plt.plot(points_pred[0], test_target)
plt.show()


#test_loss = calc(test_features, test_target)[1]
