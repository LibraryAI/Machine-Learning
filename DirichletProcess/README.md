# Dirichlet Process Gaussian Mixture

This is a Python implementation of a Dirichlet Process Gaussian Mixture. Followings are references:
* [Dirichlet Process](https://www.stats.ox.ac.uk/~teh/research/npbayes/Teh2010a.pdf) by Yee Whye Teh <br>
* [Dirichlet Process Gaussian Mixture Models: Choice of the Base Distribution](http://mlg.eng.cam.ac.uk/pub/pdf/GoeRas10.pdf) by Dilan and Rasmussen <br>
* [The Infinite Gaussian Mixture Model](https://www.seas.harvard.edu/courses/cs281/papers/rasmussen-1999a.pdf) by Carl Edward Rasmussen

### Code explanation (Normal Wishart Inverse)
Wishart inverse distribution is a prior conjugate multivariate gaussian model. The distribution generates mu and covariance for gaussian model. Here it is a base distribution for dirichlet process since mixture components of the dirichlet model is gaussian. <a href="http://www.codecogs.com/eqnedit.php?latex=(\mu,&space;\Sigma)&space;\sim&space;NIW(\mu_0,&space;\lambda,&space;\Psi,&space;v&space;)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(\mu,&space;\Sigma)&space;\sim&space;NIW(\mu_0,&space;\lambda,&space;\Psi,&space;v&space;)" title="(\mu, \Sigma) \sim NIW(\mu_0, \lambda, \Psi, v )" /></a> <br>
* __additem()__ : Used for updating the x_hat, num, sum instance variables for later calculating new parameters for likelihood.

* __delitem()__ : Used for updating the x_hat, num, sum instance variables for later calculating new parameters for likelihood. Because Gibbs Sampling is used for Dirichlet Process Gaussian Mixture model (in this case), delitem is used for sampling.

* __likelihood()__ : Likelihood method returns parameters needed for calculating likelihood of Poseterior predictive, which is multivariate-T-dist.

### Code explanation (DPGM)
1) Prior and Likelihood, and Posterior are calculated for existing n clusters and possible new cluster<br>
2) From the posterior of all clusters assign new data point <br>
3) Using Collapsed Gibbs Sampling<br>
    1) With the dataset run iteration<br>
    2) If the data point is already assigned to any cluster, (meaning the data is sampled at least once) delete the data from the cluster and get the likelihood and prior what if that data is sampled from that cluster
* __gibbs_sampling()__ :<br>
Collapsed gibbs sampling is enabled for the de finetti's theorem holds for the chinese restaurant process.<br>

* __calc_prior_and_likelihood__ :<br>
For the likelihood, each gaussian model with mu and covariance from normal invert wishart is calculated with multivariate t distribution.<br>
Chinese restaurant process is used for assigning a new data point to mixture components in the model. As a possible number of clusters are unlimited, following holds for CRP <br>
  * For cluster k : <a href="http://www.codecogs.com/eqnedit.php?latex=p(\Theta_n_e_w|\Theta_1,&space;\Theta_2,&space;...,&space;\Theta_n_-_1,\alpha)&space;=&space;N_k/(\alpha&space;&plus;&space;N-1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(\Theta_n_e_w|\Theta_1,&space;\Theta_2,&space;...,&space;\Theta_n_-_1,\alpha)&space;=&space;N_k/(\alpha&space;&plus;&space;N-1)" title="p(\Theta_n_e_w|\Theta_1, \Theta_2, ..., \Theta_n_-_1,\alpha) = N_k/(\alpha + N-1)" /></a> <br>
  * For a new cluster : <a href="http://www.codecogs.com/eqnedit.php?latex=p(\Theta_n_e_w|\Theta_1,&space;\Theta_2,&space;...,&space;\Theta_n_-_1,\alpha)&space;=&space;\alpha/(\alpha&space;&plus;&space;N-1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(\Theta_n_e_w|\Theta_1,&space;\Theta_2,&space;...,&space;\Theta_n_-_1,\alpha)&space;=&space;\alpha/(\alpha&space;&plus;&space;N-1)" title="p(\Theta_n_e_w|\Theta_1, \Theta_2, ..., \Theta_n_-_1,\alpha) = \alpha/(\alpha + N-1)" /></a> <br>

* __calc_likelihood__ :<br>
Used for simplifying code<br>
* __multivariate_t_distribution__ :<br>
Calculates the density

### Developed with
Python 3.6
Python packages: numpy, copy, math

### How to run
```
python DPGM.py
```
