import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
from sklearn.tree import export_graphviz
#import pydotplus
import pickle

original_df=pd.read_csv(r'C:\Users\jaswanth\Downloads\learnbay\python class lernbay\datasets for practice\utkarsh_prctie_dataset\bank.csv')
copy_df=original_df.copy()

   ######    makng objcts datatypes to int using sklearn library ########

def preprocessor(copy_df):
    res_df = copy_df.copy()
    le = preprocessing.LabelEncoder()
    res_df['job'] = le.fit_transform(res_df['job'])
    res_df['marital'] = le.fit_transform(res_df['marital'])
    res_df['education'] = le.fit_transform(res_df['education'])
    res_df['default'] = le.fit_transform(res_df['default'])
    res_df['housing'] = le.fit_transform(res_df['housing'])
    res_df['month'] = le.fit_transform(res_df['month'])
    res_df['loan'] = le.fit_transform(res_df['loan'])
    res_df['contact'] = le.fit_transform(res_df['contact'])
    res_df['day_of_week'] = le.fit_transform(res_df['day'])
    res_df['poutcome'] = le.fit_transform(res_df['poutcome'])
    res_df['deposit'] = le.fit_transform(res_df['deposit'])
    return res_df


 #### calling my model ##### 
encoded_df = preprocessor(copy_df)
encoded_df


### MODEL BUILDING ####

x = encoded_df.drop(['deposit'],axis =1).values
y = encoded_df['deposit'].values
  
  # 1--splitting 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2)


## modeling ###
# Decision tree with depth = 8

#class decision_tree:

  #def __init__(self,x_train,y_train,x_test,y_test):

model_dt_8 = DecisionTreeClassifier(random_state=1, max_depth=14, criterion = "gini", min_samples_split=12,
min_samples_leaf=14,)
model_dt_8.fit(x_train, y_train)
model_dt_8_score_train = model_dt_8.score(x_train, y_train)
print("Training score: ",model_dt_8_score_train)
model_dt_8_score_test = model_dt_8.score(x_test, y_test)
print("Testing score: ",model_dt_8_score_test)
result=model_dt_8.score(x_test,y_test)


### saving using pickle ####

pickle.dump(model_dt_8,open('model_8_d.pkl',  'wb'))


#### predicting using saved model(.pkl,.sav)



