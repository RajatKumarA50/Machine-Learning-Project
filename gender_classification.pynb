# Gender classification using voice
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

data = pd.read_csv(r"C:\Users\Rajat\Documents\CTC\voice.csv")

data.info()

data.label.value_counts()

data['label'] = [1 if i=='male' else 0 for i in data.label]
data.label.value_counts()

# data selection
x_data = data.drop(['label'], axis=1) # it is a matrix excluding label feature
y = data.label.values # it is a vector wich contains only label feature

y

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
x.head()

# train test split (we split our data into 2 parts: train and test. Test part is 20% of all data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42) # 0.2=20%

# take transpose of all these partial data
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

# initialize w: weight and b: bias
dimension = 20
def initialize(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    return w,b

# sigmoid function
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# check sigmoid function
sigmoid(0)

def cost(y_head, y_train):
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost_value = np.sum(loss)/x_train.shape[1] # for scaling
    return cost_value

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    cost_value = cost(y_head, y_train)
    
    # backward propagation
    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = (np.sum(y_head-y_train))/x_train.shape[1]
    
    return cost_value, derivative_weight, derivative_bias

def logistic_regression(x_train, x_test, y_train, y_test, learning_rate, num_iteration):
    w,b = initialize(dimension)
    cost_list = []
    index = []
    for i in range(num_iteration):
        cost_value, derivative_weight, derivative_bias = forward_backward_propagation(w,b,x_train,y_train)
        
        # updating weight and bias
        w = w-learning_rate*derivative_weight
        b = b-learning_rate*derivative_bias

        if i % 10 == 0:
            index.append(i)
            cost_list.append(cost_value)
            print('cost after iteration {}: {}'.format(i,cost_value))
    # in for loop above, we have obtained final values of parameters(weight and bias): machine has learnt them 
           
    z_final = np.dot(w.T,x_test)+b
    z_final_sigmoid = sigmoid(z_final) #z_final value after sigmoid function
    
    # prediction
    y_prediction = np.zeros((1,x_test.shape[1]))
    # if z_final_sigmoid is bigger than 0.5, our prediction is sign 1 (y_head_=1)
    # if z_final_sigmoid is smaller than 0.5, our prediction is sign 0 (y_head_=0)
    for i in range(z_final_sigmoid.shape[1]):
        if z_final_sigmoid[0,i]<= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
            
    # print test errors
    print('test accuracy: {} %'.format(100-np.mean(np.abs(y_prediction-y_test))*100))
    
    # plot iteration vs cost function
    plt.figure(figsize=(15,10))
    plt.plot(index, cost_list)
    plt.xticks(index, rotation='vertical')
    plt.xlabel('number of iteration', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.show()     

# run the program
# Firstly, learning_rate and num_iteration are chosen randomly. Then it is tuned accordingly
logistic_regression(x_train, x_test, y_train, y_test, learning_rate=1.5, num_iteration=200)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train.T,y_train.T)

# prediction of test data
log_reg.predict(x_test.T)

# actual values
y_test

print('test_accuracy: {}'.format(log_reg.score(x_train.T,y_train.T)))

y_test

#Prediction using some random data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
voice_data=pd.read_csv(r"C:\Users\Rajat\Documents\CTC\voice.csv")
x=voice_data.drop(columns=['label'])
y=voice_data['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
score=accuracy_score(y_test,predictions)
score
##syntax  model.predict([[  ,  ,   ,  , ]])
predictions=model.predict([[0.160042863,0.061398068,0.132186474,0.113617859,0.222639527,0.109021668,2.500834323,11.11512135,0.924541066,0.480473769,0.119684833,0.160042863,0.112151479,0.014513788,0.238095238,0.41809082,0.087890625,0.786132813,0.698242188,0.594738595
],[0.167119878,0.060909655,0.159188691,0.112206515,0.224757222,0.112550707,2.659695426,12.05930119,0.912057418,0.413139856,0.110485556,0.167119878,0.111678906,0.021834061,0.25,0.517578125,0.004882813,0.786132813,0.78125,0.375]])
predictions
