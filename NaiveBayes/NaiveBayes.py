'''
나이브 베이즈 분류기를 이용해 스팸메일을 분류하기
스팸메일 데이터셋은 "free", "money" 등과 같은 48개 단어 키워드의 상대적인 빈도
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

# 데이터를 불러옵니다.
datafile = open('data/spambase.data', 'r')
data = []
for line in datafile:
    line = [float(element) for element in line.rstrip('\n').split(',')]
    data.append(np.asarray(line))

X = [data[i][:num_features] for i in range(len(data))]
y = [int(data[i][-1]) for i in range(len(data))]

# train, test 데이터셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

class NaiveBayesClassifier(object):
	# 48개의 feature를 이용합니다
	def __init__(self, num_features=48):
        self.num_features = num_features

    def log_likelihood_naivebayes(self, feature_vector, Class):
    	assert len(feature_vector) == self.num_features
	    log_likelihood = 0.0 #log-likelihood를 사용해 underflow 회피
    	if Class == 0:
        	for feature_index in range(len(feature_vector)):
            	if feature_vector[feature_index] == 1: #feature present
                	log_likelihood += np.log10(likelihoods_ham[feature_index])
            	elif feature_vector[feature_index] == 0: #feature absent
                	log_likelihood += np.log10(1.0 - likelihoods_ham[feature_index])
    	elif Class == 1:
        	for feature_index in range(len(feature_vector)):
            	if feature_vector[feature_index] == 1:
                	log_likelihood += np.log10(likelihoods_spam[feature_index])
            	elif feature_vector[feature_index] == 0:
                	log_likelihood += np.log10(1.0 - likelihoods_spam[feature_index])

	    return log_likelihood

	# Maximum A Priori(MAP) inference를 이용해 사후확률이 가장 큰 class를 고르기
	def class_posteriors(self, feature_vector):
    	log_likelihood_ham = self.log_likelihood_naivebayes(feature_vector, Class = 0)
	    log_likelihood_spam = self.log_likelihood_naivebayes(feature_vector, Class = 1)

    	log_posterior_ham = log_likelihood_ham + log_prior_ham
	    log_posterior_spam = log_likelihood_spam + log_prior_spam

    	return log_posterior_ham, log_posterior_spam

	def spam_classify(self, document):
    	feature_vector = [int(element>0.0) for element in document]
	    log_posterior_ham, log_posterior_spam = self.class_posteriors(feature_vector)
    	if log_posterior_ham > log_posterior_spam:
        	return 0
    	else:
        	return 1

    def train(self, X_train, y_train):
		# Likelihood estimator 만들기
		# 스팸 클래스와 햄 클래스 나누기
		X_train_spam = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1]
		X_train_ham = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]

		# 각 클래스의 feature에 대한 likelihood 구하기	
		self.likelihoods_ham = np.mean(X_train_ham, axis = 0)/100,0
		self.likelihoods_spam = np.mean(X_train_spam, axis = 0)/100.0
		self.likelihoods_ham = self.likelihoods_ham[0]        

		# 각 class의 prior를 계산
		num_ham = float(len(X_train_ham))
		num_spam = float(len(X_train_spam))

		prior_probability_ham = num_ham / (num_ham + num_spam)
		prior_probability_spam = num_spam / (num_ham + num_spam)

		self.log_prior_ham = np.log10(prior_probability_ham)
		self.log_prior_spam = np.log10(prior_probability_spam)

        return None

    def predict(self, X_test):
    	predictions = []
		for case in X_test:
    		predictions.append(self.spam_classify(case))
    	
    	return predictions
        

NB = NaiveBayesClassifier()
NB.train(X_train, y_train)
pred = NB.predict(X_test)

def evaluate_performance(predictions, ground_truth_labels):
    correct_count = 0.0
    for item_index in range(len(predictions)):
        if predictions[item_index] == ground_truth_labels[item_index]:
            correct_count += 1.0
    accuracy = correct_count/len(predictions)
    return accuracy

accuracy_naivebayes = evaluate_performance(pred, y_test)
print(accuracy_naivebayes)