import numpy as np
import copy
from math import *

#(1) 새로운 데이터 포인트 관측 , sample
#(2) 해당 포인트를 위한 prior 계산, Chinese Restaurant Process 를 통한 
#    (1) 신규 클러스터 확률
#    (2) 기존 클러스터들에 대한 확률 
#(3) 해당 포인트를 위한 likelihood 계산, mixture component 즉 특정 클러스터의 가우시안 pdf 계산
#(4) maximum likelihood를 가지는 posterior ... 가 아니라 random sampling을 통한 확률 구간 


class DirichletPRocessGaussianMixture:
    '''
    Dirichlet Process Gaussian Mixture code

    - Base function: Normal Inverse Wishart distribution is used 
    - Mixing Coefficient(Prior): Chinese Restaurant Process is used for probability of assigning a new data to each cluster
    - Mixture Component(Likelihood): Multivariate Gaussian distribution is used, with NIW as a base function (prior of the gaussian)
    - Posterior: Prior × Likelihood, the probability a new dataset is sampled from a specific component

    1) Prior and Likelihood, and Posterior are calculated for existing n clusters and possible new cluster
    2) From the posterior of all clusters assign new data point 
    3) Using Collapsed Gibbs Sampling
        1) With the dataset run iteration
        2) If the data point is already assigned to any cluster, 
           (meaning the data is sampled at least once) delete the data from the cluster and get the likelihood and prior
           what if that data is sampled from that cluster
    
    - Collapsed gibbs sampling is enables for the de finetti's theorem holds for the chinese restaurant process
    - However, DPGM model is itself constrained that it is a stochastic process, this itself is not for the clustering purpose
      (modification will do)

    - My model does not assign clusters to all datapoints when initializing the class. I believe it is not recommended

    '''
    def __init__(self, alpha, prior, data, num_iter, z):
        
        # num_iter: number of iterations all data in the dataset should be done (num_iter > 1 needed)
        # prior: prior distribution
        # n_k: number of data points assigned to each cluster maintained. 
        #      If 0 data is assigned to a cluster, that cluster will be deleted
        # z: cluster name assigned to each datapoint. 
        #    max length is the size of the dataset
        # k_mixtures: mixture component. each element is a copy of a base distribution 
        
        self.num_iter = num_iter
        self.alpha = alpha
        self.prior = prior(data.shape[1])
        self.data = data
        self.num_samples, self.dim = data.shape
        self.n_k = [1]
        self.z = [1]
        self.num_clusters = 1
        self.k_mixtures = []
        self.k_mixtures.append(copy.deepcopy(self.prior))
        self.new_cluster = copy.deepcopy(self.prior)

    def calc_likelihood(self, data, idx):

        # calculate the likelihood of a mixture component
        # multivariate t distribution is used

    	l, mu, v, psi = self.k_mixtures[idx].likelihood(data)
        ll = self.multivariate_t_distribution(data, 
                                              mu, 
                                              psi * (1/l+1) / 1/l * (v - self.dim + 1), v - self.dim + 1, 
                                              self.dim)
        return ll

    def calc_prior_and_likelihood(self, data, idx):

        # calculate prior prob and likelihood for new sample of each cluster k using chinese restaurant process 

        crp_prior = []
        likelihood = []
        if sum(self.n_k) < self.num_samples:
            
            ###### prior ########
            for i in range(self.num_clusters):
                crp = self.n_k[i] / (self.alpha + sum(self.n_k))
                crp_prior.append(crp)
            crp_prior.append(self.alpha / (self.alpha + sum(self.n_k)))
            
            ###### likelihood ########
            for i in range(self.num_clusters):
                ll = self.calc_likelihood(data, i)
                likelihood.append(ll)
        else:

            k = self.z[idx]
            self.n_k[k-1] -= 1
            self.k_mixtures[k-1].delitem(data)

            if self.n_k[k-1] == 0:
                self.num_clusters -= 1
                self.k_mixtures.pop(k-1)
                self.n_k.pop(k-1)
                a = np.squeeze(np.argwhere(np.array(self.z) > k), axis = 1)
                for i in a:
                    self.z[i] -= 1
            
            ###### likelihood ########              
            for i in range(self.num_clusters):
                ll = self.calc_likelihood(data, i)
                likelihood.append(ll)                    

            ###### prior #########
            for i in range(self.num_clusters):
                crp = self.n_k[i] / (self.alpha + sum(self.n_k))
                crp_prior.append(crp)
            crp_prior.append(self.alpha / (self.alpha + sum(self.n_k)))            

        # for likelihood of a new cluster
        l = self.new_cluster.l
        v = self.new_cluster.v
        psi = self.new_cluster.psi
        mu = self.new_cluster.mu0
        ll = self.multivariate_t_distribution(data, mu, psi * (1/l+1) / 1/l * (v - self.dim + 1), v - self.dim + 1, self.dim)
        likelihood.append(ll)

        return crp_prior, likelihood

    def gibbs_sampling(self):

        # Collapsed gibbs sampling for updating posterior and assignment of a data point

        for _ in range(self.num_iter):
            for idx, data in range(self.data):
                crp_prior, likelihood = calc_prior_and_likelihood(data, idx)
                posterior = np.multiply(np.array(crp_prior), np.array(likelihood))
                posterior = posterior / sum(posterior)

                # simple assignment by random sampling and assign cluster
                random_sample =  np.random.rand() 
                k_new = int(np.sum(random_sample > np.cumsum(posterior)))

                if k_new > self.num_clusters:
                    self.num_clusters += 1
                    self.n_k.append(0)
                    self.k_mixtures.append(copy.deepcopy(self.prior))
                
                # update posterior of the assigned cluster prior and likelihood
                self.k_mixtures[k_new-1].additem(data)
                self.n_k[k_new-1] += 1
                
                if len(self.z) < self.num_samples:
                    self.z.append(k_new)
                else:
                    self.z[idx] = k_new

    def multivariate_t_distribution(x,mu,Sigma,df,d):
        
        '''
        Multivariate t-student density:
        output:
            the density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
            mu = mean (d dimensional numpy array or scalar)
            Sigma = scale matrix (dxd numpy array)
            df = degrees of freedom
            d: dimension


        this code is from https://stackoverflow.com/questions/29798795/multivariate-student-t-distribution-with-python, ''farhawa''
        '''
        Num = gamma(1. * (d+df)/2)
        Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(np.linalg.det(Sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), (x - mu)),1.* (d+df)/2))
        d = 1. * Num / Denom 
        return d

    #def print_probs(self):
        """
        Print the clusters and the MAP assignment of its Multinoulli parameters
        :return:
        """
    #    print('The MAP assignments of the clusters that we sampled')
    #    for i, k in enumerate(np.argsort(self.n_k)[::-1]):
    #        q = self.clusters_distros[k]
    #        map_assignment = q.get_posterior_multinoulli('map')
    #        print('Cluster %3i with %5i data and MAP %s' % (k, q.num, ' - '.join(['%5.2f' % prob for prob in map_assignment])))
