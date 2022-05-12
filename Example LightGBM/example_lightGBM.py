#importing standard libraries
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import graphviz

import lightgbm as lgb

#loading our training dataset 'adult.csv' with name 'data' using pandas
data=pd.read_csv('./adult.csv',header=None)

#Assigning names to the columns
data.columns=['age','workclass','fnlwgt','education','education-num','marital_Status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','Income']

#Label Encoding our target variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l=LabelEncoder()
l.fit(data.Income)

data.Income=Series(l.transform(data.Income)) #label encoding our target variable

#One Hot Encoding of the Categorical features
one_hot_workclass=pd.get_dummies(data.workclass)
one_hot_education=pd.get_dummies(data.education)

#removing categorical features
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True)

#Merging one hot encoded features with our dataset 'data'
data=pd.concat([data,one_hot_workclass,one_hot_education],axis=1)

#Here our target variable is 'Income' with values as 1 or 0.
#Separating our data into features dataset x and our target dataset y
x=data.drop('Income',axis=1)
y=data.Income

#Imputing missing values in our target variable
y.fillna(y.mode()[0],inplace=True)

#Now splitting our dataset into test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

train_data=lgb.Dataset(x_train,label=y_train)

#setting parameters for lightgbm
param = {'num_leaves':150, 'objective':'binary','max_depth':3,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']

#training our model using light gbm
num_round=50
lgbm=lgb.train(param,train_data,num_round)

graph = lgb.create_tree_digraph(lgbm)
graph.render(view=True)