#missing vals handled by either dropping or replacing vals


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 


#reading csv file into dataset
dataset=pd.read_csv('basics\Placement_Data_Full_Class.csv')

#seeing no of missing vals
print(dataset.isnull().sum())

#analyzing distribution of vals
sns.distplot(dataset.salary)
plt.show()

#by replacing missing vals
#when no skewness- normally dist: use mean
#when skewness- use median mode


# #using median
dataset['salary'].fillna(dataset['salary'].median(),inplace=True)
print(dataset.isnull().sum())
sns.distplot(dataset.salary)
plt.show()

#by dropping
# dataset=dataset.dropna(how='any')
# sns.distplot(dataset.salary)
# plt.show()
