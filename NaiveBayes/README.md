Naive Bayes Classifier
==================
This is a python implementation of naive bayes classifier for filtering spam mail with 2 lables spam and ham.<br>
Spam mail dataset has x variables of 48 features as relative frequency of word counts of "free", "money" and so on.

### Code explanation
* __train()__<br>
<a href="http://www.codecogs.com/eqnedit.php?latex=p(\Theta)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(\Theta)" title="p(\Theta)" /></a> : prior probability of data in each class<br>
<a href="http://www.codecogs.com/eqnedit.php?latex=p(x_i|\Theta)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(x_i|\Theta)" title="p(x_i|\Theta)" /></a> : likelihood of features in each class following conditional independence assumption
* __predict()__<br>
Calculate log posterior probability of a new data vector for each classes<br>
<a href="http://www.codecogs.com/eqnedit.php?latex=p(y&space;=&space;spam&space;|&space;x_n_e_w)&space;=&space;p(x_n_e_w|\Theta_s_p_a_m)*p(\Theta_s_p_a_m)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(y&space;=&space;spam&space;|&space;x_n_e_w)&space;=&space;p(x_n_e_w|\Theta_s_p_a_m)*p(\Theta_s_p_a_m)" title="p(y = spam | x_n_e_w) = p(x_n_e_w|\Theta_s_p_a_m)*p(\Theta_s_p_a_m)" /></a><br>
<a href="http://www.codecogs.com/eqnedit.php?latex=p(y=ham|x_n_e_w)=p(x_n_e_w|\Theta_h_a_m)*p(\Theta_h_a_m)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(y=ham|x_n_e_w)=p(x_n_e_w|\Theta_h_a_m)*p(\Theta_h_a_m)" title="p(y=ham|x_n_e_w)=p(x_n_e_w|\Theta_h_a_m)*p(\Theta_h_a_m)" /></a><br>
<a href="http://www.codecogs.com/eqnedit.php?latex=p(x_n_e_w|\Theta)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(x_n_e_w|\Theta)" title="p(x_n_e_w|\Theta)" /></a>, <a href="http://www.codecogs.com/eqnedit.php?latex=p(\Theta)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p(\Theta)" title="p(\Theta)" /></a> are precalculated from train method.<br>
Compare log posterior of each classes and assign class to a bigger prob.

### Developed with
Python 3.6.3 <br>
Python packages: numpy, pandas

### How to run
```
python NaiveBayes.py
```
