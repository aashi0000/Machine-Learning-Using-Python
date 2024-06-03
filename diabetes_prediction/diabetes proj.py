import numpy as np
import pandas as pd
import sklearn.datasets as ds
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('diabetes_prediction\diabetes.csv')

#handle missing vals if any

x=df.drop(columns='Outcome',axis=1)#features
y=df.Outcome #targets

#splitting data into training and test

#data standardization- ek hi range m krdeta data ko
#meaning standard devn should be one, mean should be zero
#newval= (x-mean)/std_dev
scaler=StandardScaler()
x=scaler.fit_transform(x)
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2 )
print(x_train.std())
print(x_test.std())

#training model
classifier= svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

#training accuracy score
x_train_pred=classifier.predict(x_train)
train_accuracy=accuracy_score(x_train_pred,y_train)
print("training data accuracy: ",train_accuracy)
#testing accuracy score
x_test_pred=classifier.predict(x_test)
test_accuracy=accuracy_score(x_test_pred,y_test)
print("testing data accuracy: ",test_accuracy)

#pred sys

#taking input
inp=(1,85,66,29,0,26.6,0.351,31)
inp=np.asarray(inp)
inp_reshaped=inp.reshape(1,-1)
#stdzn the input data
inp_std=scaler.transform(inp_reshaped)
print(inp_std)
pred=classifier.predict(inp_std)
if(pred[0]==0):
    print("person is not diabetic")
else:
    print("person is diabetic")