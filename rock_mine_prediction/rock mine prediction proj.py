import numpy as np
import pandas as pd
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv(r'rock_mine_prediction\sonar.csv',header=None)
# print(df.describe())
# print(df[60].value_counts())
# print(df.groupby(60).mean())

#handle missing vals if any

#separating features and target
x=df.drop(columns=60,axis=1)
y=df[60]
print(x)
#standardizing data
scaler=StandardScaler()
x=scaler.fit_transform(x)

#separating training and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)
#stratify se eq num of r and m


#fitting model with training data
model=LogisticRegression()
model.fit(x_train,y_train)

#model accuracy (train)- predict new x vals and check with old y vals
x_train_pred=model.predict(x_train)
train_accuracy=accuracy_score(x_train_pred,y_train)
print("training data accuracy: ",train_accuracy)

#now for test data
x_test_pred=model.predict(x_test)
test_accuracy=accuracy_score(x_test_pred,y_test)
print("testing data accuracy: ",test_accuracy)


#making a predictive system
inp=(0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
inp=np.asarray(inp)
inp_reshaped=inp.reshape(1,-1)
pred=model.predict(inp_reshaped)
if(pred[0]=='R'):
    print("according to input data, object is a rock")
else:
    print("according to input data, object is a mine")
