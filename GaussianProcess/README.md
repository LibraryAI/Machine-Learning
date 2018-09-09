Gaussian Process 
==================
This is a Tensorflow implementation of a Gaussian Process. Followings are references:

* [Gaussian Process for Regression: A Quick Introduction](https://arxiv.org/pdf/1505.02965.pdf) by M. Ebden
* [Gaussian Process for Machine Learning CH2, CH3](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf) by  C. E. Rasmussen & C. K. I. Williams, the MIT Press 2006

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
