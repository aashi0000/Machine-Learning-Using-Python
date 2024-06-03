import numpy as np
import pandas as pd
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset=ds.load_breast_cancer()
df=pd.DataFrame(dataset.data, columns=dataset.feature_names)

x=df #features
y=dataset.target  #targets

#splitting data into training and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3 )

#data standardization
#meaning standard devn should be one, mean should be zero
#newval= (x-mean)/std_dev
scaler=StandardScaler()
scaler.fit(x_train)
x_train_std=scaler.transform(x_train)
x_test_std=scaler.transform(x_test)
print(x_train_std.std())
print(x_test_std.std())



#handling imbalanced datasets
#col ke basis pe div df into parts
#part to replace ke no of rows count kro- nr
#sample from bachi hui bachi hui vals nr rows
#concatenate both-> eq no of vals of both
