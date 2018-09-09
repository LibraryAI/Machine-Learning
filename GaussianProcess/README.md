Gaussian Process 
==================
This is a Tensorflow implementation of a Gaussian Process. Followings are references:

* [Gaussian Process for Regression: A Quick Introduction](https://arxiv.org/pdf/1505.02965.pdf) by M. Ebden
* [Gaussian Process for Machine Learning CH2, CH3](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf) by  C. E. Rasmussen & C. K. I. Williams, the MIT Press 2006

### Code explanation
* __kernel()__<br>
Kernel is a vector dot product of two vectors transformed to different feature space by a mapping function<br>
Here kernel function uses three different kernel components and add them: <br>
exponential component, degree 1 linear component, and bias component each represents the strength of distance between data points, intrinsic bias.<br>
_Returns tensor shape [1]_ <br>
* __train()__<br>
(1) First calculates covariance matrix of n by n size where n is the size of the training data.<br>
(2) Calculate inverse of the covariance matrix with pinv class method<br>
(3) Calculate negative log likelihood of each data in the training data. Will be used later with adam optimizer to minimize.<br>
* __predict()__<br>
Predict performs a regression for a new data point. <a href="http://www.codecogs.com/eqnedit.php?latex=p(t_n_&plus;_1|T_N)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(t_n_&plus;_1|T_N)" title="p(t_n_+_1|T_N)" /></a><br>
<a href="http://www.codecogs.com/eqnedit.php?latex=p(t_n_&plus;_1|T_N)&space;=&space;N(t_n&plus;1|k^{T}cov_N^{-1}T_N,&space;c-k^{T}cov_N^{-1}k)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(t_n_&plus;_1|T_N)&space;=&space;N(t_n&plus;1|k^{T}cov_N^{-1}T_N,&space;c-k^{T}cov_N^{-1}k)" title="p(t_n_+_1|T_N) = N(t_n+1|k^{T}cov_N^{-1}T_N, c-k^{T}cov_N^{-1}k)" /></a> is how the regression of a new data point given the training data is performed.<br>
* __sample()__

### Differences from the reference
* __Kernel fucntion__<br>
Gaussian Process for Regression: A Quick Introduction uses kernel function<br>
<a href="http://www.codecogs.com/eqnedit.php?latex=K(x,&space;x')&space;=&space;\Theta&space;_1&space;*&space;exp(-(x-x')^2/2\Theta_2)&space;&plus;&space;\Theta_3×δ(x,x')" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K(x,&space;x')&space;=&space;\Theta&space;_1&space;*&space;exp(-(x-x')^2/2\Theta_2)&space;&plus;&space;\Theta_3×δ(x,x')" title="K(x, x') = \Theta _1 * exp(-(x-x')^2/2\Theta_2) + \Theta_3×δ(x,x')" /></a><br>
which uses Kronecker delta function. However, in this implementation kernel function of<br>
<a href="http://www.codecogs.com/eqnedit.php?latex=K(x,&space;x')&space;=&space;\Theta&space;_0*&space;exp(\Theta_1/2*(x-x')^2)&space;&plus;&space;\Theta_2&space;&plus;&space;\Theta_3*Dot(x,&space;x')" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K(x,&space;x')&space;=&space;\Theta&space;_0*&space;exp(\Theta_1/2*(x-x')^2)&space;&plus;&space;\Theta_2&space;&plus;&space;\Theta_3*Dot(x,&space;x')" title="K(x, x') = \Theta _0* exp(\Theta_1/2*(x-x')^2) + \Theta_2 + \Theta_3*Dot(x, x')" /></a><br>
which is comprised of three kernel components.

* __Probabilistic programming framework__<br>
Differentiation of log likelihood of w.r.t thetas of kernel function does not have a closed form solution and require approximatioin for the MLE. 
I used automatic gradient descent method of TensorFlow instead of long derivation.

### Developed with
TensorFlow <br>
Python 3.6.3 <br>
Python packages: numpy, Scikit-Learn(for dataset  and scaling)

### How to run
```
python GaussianProcess_tensroflow.py
```
