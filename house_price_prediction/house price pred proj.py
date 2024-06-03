import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'price']
df = pd.read_csv(r'house_price_prediction\housing.csv', header=None, delimiter=r"\s+", names=column_names)

#understanding correlation
#correlation=df.corr()
#constructing heatmap
#plt.figure(figsize=(10,10))
#sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
# plt.show()

x=df.drop(['price'],axis=1)
y=df.price

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)

#model training
model=XGBRegressor()
model.fit(x_train,y_train)

#model eval
train_pred=model.predict(x_train)
#r squared error
score_1=metrics.r2_score(y_train,train_pred)
#mean abs err
score_2=metrics.mean_absolute_error(y_train,train_pred)
print("r sq err: ",score_1)
print("mean abs err: ",score_2)


#visualizing actual and predicted prices
plt.scatter(y_train,train_pred)
plt.xlabel('actual prices')
plt.ylabel('predicted prices')
plt.title('actual vs pred prices')
plt.show()

test_pred=model.predict(x_test)
#r squared error
score_1=metrics.r2_score(y_test,test_pred)
#mean abs err
score_2=metrics.mean_absolute_error(y_test,test_pred)
print("r sq err: ",score_1)
print("mean abs err: ",score_2)