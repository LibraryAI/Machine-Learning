import tensorflow as tf
import random
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt


class GaussianProcess:
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
            
            pred_mu = tf.matmul(tf.matmul(k, covInv), Y)
            pred_var = tf.subtract(c, tf.matmul(tf.matmul(k, covInv), tf.transpose(k)))
            
            negloglikelihood = tf.add(negloglikelihood, tf.div(tf.pow(tf.subtract(pred_mu, tf.slice(Y, [i, 0], [1, 1])[0]), 2), tf.scalar_mul(tf.constant(2.0), pred_var)))
     
            return negloglikelihood[0]

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

        for i in range(n_data):
            kernel_output = self.kernel(obs_X, tf.slice(train_X, [i, 0], [1, numDimension]))
            new_cov.append(kernel_output)
        new_cov = tf.reshape(tf.stack(new_cov), [1, numData])
        pred_mu = tf.matmul(tf.matmul(new_cov, self.covInv), train_Y)
        #pred_mu = tf.matmul(tf.matmul(k, covInv), Y)
        pred_sigma = tf.subtract(tf.reshape(self.kernel(obs_X, obs_X), [1, 1]), 
                            tf.matmul(tf.matmul(new_cov, self.covInv), tf.transpose(new_cov)))
        return pred_mu, pred_sigma
        
    def sample(self, mean, cov):
        
        # 주어진 mean, cov에 맞춰 sampling
        # 한 데이터 포인트에 대한 sampling 혹은
        # 데이터셋에 대한 trajectory를 sampling 
        
        num_data = len(tf.shape(cov)[0])
        if num_data == 1:
            return tf.add(mean, tf.random_normal([1], mean, stddev=cov))
        
        z = tf.random_normal([num_data])
        s, u, v = tf.linalg.svd(cov)
        a = matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
        return tf.add(mean, tf.matmul(a, z))

    def pinv(self, A, reltol=1e-6):
        # Compute the SVD of the input matrix A
        s, u, v = tf.linalg.svd(A)
 
        # Invert s, clear entries lower than reltol*s[0].
        
        #atol = tf.reduce_max(s) * reltol
        #s = tf.boolean_mask(s, s > atol)

        s_inv = tf.diag(tf.div(1., s))
        return tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))

################################################################################################################################################################
n_epochs = 500
lr = .01
#batch_size = 64
#n_steps = int(X_train.shape[0] / batch_size)


total_features, total_prices = load_boston(True)
total_features = scale(total_features)

train_features = total_features[:15]
train_prices = total_prices[:15]
test_features = total_features[15:20]
test_prices = total_prices[15:20]

numDimension = train_features.shape[1]
numTrain = train_features.shape[0]
numTest = test_features.shape[0]

train_prices = np.reshape(train_prices, (numTrain, 1))
test_prices = np.reshape(test_prices, (numTest, 1))

train_X = tf.placeholder(tf.float32, [None, numDimension])
train_Y = tf.placeholder(tf.float32, [None, 1])

train_XX = tf.placeholder(tf.float32, [None, numDimension])
train_YY = tf.placeholder(tf.float32, [None, 1])
test_X = tf.placeholder(tf.float32, [None, numDimension])
#test_Y = tf.placeholder(tf.float32, [None, 1])

GP = GaussianProcess()
negloglikelihood = GP.train(train_X, train_Y, numTrain)
training_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss=negloglikelihood)
pred_op = GP.predict(test_X, train_XX, train_YY, numTrain)


#sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#sess = tf.Session(config = sess_config)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

################################################################################################################################################################
points = [[], []]
for epoch in range(n_epochs):
    _, tr_loss = sess.run([training_op, negloglikelihood], feed_dict = {train_X : train_features,
                                                                       train_Y : train_prices})
    
    if epoch % 10 == 0:
        points[0].append(epoch)
        points[1].append(tr_loss)

    if epoch % 100 == 0:
        print(tr_loss)


plt.plot(points[0], points[1], 'r--')
plt.show()


################################################################################################################################################################
points_pred = [[], []]
for i in range(numTest):
    
    test_feature = np.reshape(test_features[i], (1, numDimension))
    pred_mu, _ = sess.run(pred_op, feed_dict = {train_XX : train_features,
                                                train_YY : train_prices,
                                                test_X : test_feature})
    points_pred[0].append(i)
    points_pred[1].append(pred_mu)

plt.plot(points_pred[0], points_pred[1], 'r--')
plt.show()

plt.plot(points_pred[0], test_prices)
plt.show()


#test_loss = calc(test_features, test_prices)[1]
