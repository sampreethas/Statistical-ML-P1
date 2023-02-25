#!/usr/bin/env python
# coding: utf-8

# In[25]:


#to import the necessary libraries

import numpy as np      #to check if numpy is properly installed or not - python library with support for algorithms, arrays etc
import scipy.io as sc   #open source python library used for mathematical and scientific problems - built on numpy extension
import math             #to perform basic math calculations
import matplotlib.pyplot as plt   #to plot the graphs


# In[26]:


data_input = sc.loadmat('mnist_data.mat')  #loading the contents of .mat file or a given dataset 'mnist_data.mat' 


# In[27]:



#data_input contains below features

#the set of inputs according to .txt file 
#trX - training set, each row represents a digit
#trY - training labels, 0 and 1 represent digit 7 and 8 respectively
#tsX - testing set, each row represents a digit
#tsY - testing labels, 0 and 1 represent digit 7 and 8 respectively
#Ploting the input mnist data
reshaped_data = np.reshape(data_input["trX"][-1],(28,28))
plt.imshow(reshaped_data, cmap = "gray")


# In[28]:


#extracting the testing and training data from loaded dataset
X_train, X_test, Y_train, Y_test = data_input['trX'], data_input['tsX'], data_input['trY'], data_input['tsY']


# In[29]:


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


# In[30]:


def accuracy_LR(y_actual, y_predict):
    accuracy = np.sum(y_actual == y_predict, axis = 0)/len(y_actual)
    return accuracy


# In[31]:


#logistic Regression using gradient ascent method
#using sigmoid function - In order to map predicted values to probabilities (varies between 0 and 1)
#sigmoid function definition: c(x) = 1/(1+exp(-x)) - squeezes the value to 0 and 1
#Logistic regression maps continuous x to binary y


class sigmoid_definition():
    def __call__(data, x):
        res = 1 / (1+np.exp(-x))
        return res
    
    def derivative_sigmoid(data, x):
        grad = data.func(x)*(1 - data.func(x))
        return grad
        
        


# In[32]:


# to view the sigmoid function definition or representation

x_axis = np.linspace(-10, 10, 200)
z_axis = 1/(1 + np.exp(-x_axis))
  
plt.plot(x_axis, z_axis)
plt.xlabel("x")
plt.ylabel("sigmoid_definition(X)")
  
plt.show()


# In[33]:


#task 2 - logistic regression
#logistic regression using gradient ascent method


class LogisticRegression_gradient():
   
    #__init__ used as constructor of oops concept to initialize or assign values to the data members of the class 
    #when an object of class is created
    def __init__(self, learning_rate=.01, gradient_ascent = True):
        self.parameter_value = None
        self.learning_rate = learning_rate
        self.gradient_ascent = gradient_ascent
        self.sigmoid = sigmoid_definition()

    #to get the dimension of array
    def parameter_initialization(self, data_x):
        feature_value = np.shape(data_x)[1]
        threshold = 1 / math.sqrt(feature_value)
        self.parameter_value = np.random.uniform(-threshold, threshold, (feature_value,))

    #fitdata function implementation for n iterations, make a new prediction
    #gradient ascent method - move in the direction of the loss function with respect to parameters to reduce the loss/error
    
    def fit_data(self, data_x, data_y, iteration_count = 10000):
        self.parameter_initialization(data_x)
        for i in range(iteration_count):
            y_pred = self.sigmoid(data_x.dot(self.parameter_value))
            if self.gradient_ascent:
                self.parameter_value += self.learning_rate * (data_y - y_pred).dot(data_x)

    #to predict the output using rounding concept
    def predict_output(self, data_x):
        y_pred = np.round(self.sigmoid(data_x.dot(self.parameter_value))).astype(int)
        return y_pred


# In[34]:


def main():
    Ytr, Ytest = data_input['trY'].squeeze().T, data_input['tsY'].squeeze().T
    class_obj2 = LogisticRegression_gradient()
    class_obj2.fit_data(X_train,Ytr)
    y_predict = class_obj2.predict_output(X_test)
    accuracy_score_LR = accuracy_LR(Ytest, y_predict)
    print("The accuracy for Logistic Regression using gradient ascent:",accuracy_score_LR)
    
main()

