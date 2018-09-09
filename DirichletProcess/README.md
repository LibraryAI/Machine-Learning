# Dirichlet Process Gaussian Mixture

This is a Python implementation of a Dirichlet Process Gaussian Mixture. Followings are references:
* [Dirichlet Process](https://www.stats.ox.ac.uk/~teh/research/npbayes/Teh2010a.pdf) by Yee Whye Teh <br>
* [Dirichlet Process Gaussian Mixture Models: Choice of the Base Distribution](http://mlg.eng.cam.ac.uk/pub/pdf/GoeRas10.pdf) by Dilan and Rasmussen <br>
* [The Infinite Gaussian Mixture Model](https://www.seas.harvard.edu/courses/cs281/papers/rasmussen-1999a.pdf) by Carl Edward Rasmussen

### Code explanation (Normal Wishart Inverse)
Wishart inverse distribution is a prior conjugate multivariate gaussian model. The distribution generates mu and covariance for gaussian model. Here it is a base distribution for dirichlet process since mixture components of the dirichlet model is gaussian.<br>
<a href="http://www.codecogs.com/eqnedit.php?latex=(\mu,&space;\Sigma)&space;\sim&space;NIW(\mu_0,&space;\lambda,&space;\Psi,&space;v&space;)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(\mu,&space;\Sigma)&space;\sim&space;NIW(\mu_0,&space;\lambda,&space;\Psi,&space;v&space;)" title="(\mu, \Sigma) \sim NIW(\mu_0, \lambda, \Psi, v )" /></a>
* __additem()__ : Used for updating the x_hat, num, sum instance variables for later calculating new parameters for likelihood.

* __delitem()__ : Used for updating the x_hat, num, sum instance variables for later calculating new parameters for likelihood. Because Gibbs Sampling is used for Dirichlet Process Gaussian Mixture model (in this case), delitem is used for sampling.

* __likelihood()__ : Likelihood method returns parameters needed for calculating likelihood of Poseterior predictive, which is multivariate-T-dist.

### Code explanation (DPGM)



### Developed with
Python 3.6
Python packages: numpy, copy, math
