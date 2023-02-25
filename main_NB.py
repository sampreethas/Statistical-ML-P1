#!/usr/bin/env python
# coding: utf-8

# In[1]:


#to import the necessary libraries

import numpy as np      #to check if numpy is properly installed or not - python library with support for algorithms, arrays etc
import scipy.io as sc   #open source python library used for mathematical and scientific problems - built on numpy extension
import math             #to perform basic math calculations
import matplotlib.pyplot as plt   #to plot the graphs


# In[2]:


data_input = sc.loadmat('mnist_data.mat')  #loading the contents of .mat file or a given dataset 'mnist_data.mat' 


# In[3]:



#data_input contains below features

#the set of inputs according to .txt file 
#trX - training set, each row represents a digit
#trY - training labels, 0 and 1 represent digit 7 and 8 respectively
#tsX - testing set, each row represents a digit
#tsY - testing labels, 0 and 1 represent digit 7 and 8 respectively
#Ploting the input mnist data
reshaped_data = np.reshape(data_input["trX"][-1],(28,28))
plt.imshow(reshaped_data, cmap = "gray")


# In[4]:


#extracting the testing and training data from loaded dataset
X_train, X_test, Y_train, Y_test = data_input['trX'], data_input['tsX'], data_input['trY'], data_input['tsY']


# In[5]:


#task 1 - to extract features from input X for both training and testing set
#The average of all pixel values in the image
#The standard deviation of all pixel values in the image

#Stack arrays in sequence vertically - vertical concatenation
#taking transpose for better understanding 
#for numpy package, .mean() and .std() is used for mean and standard deviation calculation

X_train = np.vstack([data_input['trX'].mean(axis = 1), data_input['trX'].std(axis = 1)]).T
X_test = np.vstack([data_input['tsX'].mean(axis = 1), data_input['tsX'].std(axis = 1)]).T
print(X_train)
print(X_test)


# In[9]:


def accuracy_NB(y_actual, y_predict):
    accuracy = np.sum(y_actual == y_predict, axis = 0)/len(y_actual)
    return accuracy


# In[8]:


#The class for Gaussian NaiveBayes Classifier
#Naive Bayes Classifier follows bayes rule which is defined as - P(Y|X) = (P(X|Y) * P(Y)) / P(X)
#where P(X|Y) is the likelihood, P(Y) is the prior probability, 
#P(Y|X) is the posterior probability 
#P(X) scales the posterior probability/scaling factor, which is not considered for calculation


class Gaussian_NaiveBayes_Classifier():
    #fit_data: takes a sample of data for one variable and fits a data distribution, 
    #mean and variance calculations for each class features
    def fit_data(value, x_train, y_train):
        value.x_train, value.y_train = x_train, y_train
        value.classes = np.unique(y_train)
        value.features = []
       
        #there are two labels for two digits according to problem statement, 0 for 7, 1 for 8 
        for i, cl in enumerate(value.classes):
            x_train_cl = x_train[np.where(y_train == cl)]
            value.features.append([])
            
            for j in x_train_cl.T:
                features = {"mean": j.mean(), "variance": j.var()}
                value.features[i].append(features)
                
    #prior probability calculation with total number of samples             
    def prior_calculation(value, cl):
        rate = np.mean(value.y_train == cl)
        return rate
    
   #calculating the likelihood function P(X|Y) for naive bayes classifier
    #gaussian likelihood formula calculated as product of two variables - for simpler calculation  
    def likelihood_Calculation(value, mean, var, x_train):
        var1 = 1.0 / math.sqrt(2.0 * math.pi * var)
        var2 = math.exp(-(math.pow(x_train - mean, 2) / (2 * var)))
        return var1*var2
 
    # posterior probability calculation which is the product of prior and likelihood
    #Return the class with the largest posterior probability -  class y with highest probability
    #initializing posterior as prior for calculations 
    
    def posterior_calculation(value, sample_data):
        posterior_data = []
        for i, cl in enumerate(value.classes):
            posterior_calc = value.prior_calculation(cl)
            for feature_value, parameter_value in zip(sample_data, value.features[i]):
                likelihood_calc = value.likelihood_Calculation(parameter_value["mean"], parameter_value["variance"], feature_value)
                posterior_calc *= likelihood_calc
            posterior_data.append(posterior_calc)
        return value.classes[np.argmax(posterior_data)]
    
     #predicting the class labels of samples in X
    
    def predict_output(value, x_train):
        y_pred = [value.posterior_calculation(sample_data) for sample_data in x_train]
        return y_pred
    
    
                


# In[10]:


def main():
    Ytr, Ytest = data_input['trY'].squeeze().T, data_input['tsY'].squeeze().T
    class_obj = Gaussian_NaiveBayes_Classifier()
    class_obj.fit_data(X_train,Ytr)
    y_predict = class_obj.predict_output(X_test)
    accuracy_score_final = accuracy_NB(Ytest, y_predict)
    print("The accuracy of Naive Bayes Classifier is:", accuracy_score_final)
main()

