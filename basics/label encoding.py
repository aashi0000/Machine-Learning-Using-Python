import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r'unit 1\breastcancer.csv')
# print(list(df.columns.values))
# print(dataset)
# df['diagnosis']= dataset.target()
print(df.diagnosis.value_counts())
lab_enc= LabelEncoder()
labels=lab_enc.fit_transform(df.diagnosis)
#alphabetical order m labels assign krta
df['target']=labels
print(df.diagnosis,df.target)

