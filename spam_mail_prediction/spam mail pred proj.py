import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#reading file into dataframe
df=pd.read_csv('spam_mail_prediction\mail_data.csv')
#filtering dataframe for vals that are not null
df=df.where(pd.notnull(df),'')

#label encoding
#spam=0,ham=1
df.loc[df.Category=='spam','Category',]=0
df.loc[df.Category=='ham','Category',]=1

#separating text and labels
x=df.Message
y=df.Category

#train-test splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)

#feature extraction for textual data (data->feature vectors)

#fit all mails into vectorizer function 
feat_ext=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
#include words with min tfidf  score of 1
#stopwords common words used in english ..taaki spam wala doesnt recognize these

x_train=feat_ext.fit_transform(x_train)
x_test=feat_ext.transform(x_test)
#each sentence in x get a certain score
y_train=y_train.astype('int')
y_test=y_test.astype('int')

#training model
model=LogisticRegression()
model.fit(x_train,y_train)

#evaluating predicted vals
pred_train=model.predict(x_train)
accur_train=accuracy_score(pred_train,y_train)
print("training accuracy: ",accur_train)

pred_test=model.predict(x_test)
accur_test=accuracy_score(pred_test,y_test)
print("testing accuracy: ",accur_test)

#building predictive system

#input
inp=["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
inp=feat_ext.transform(inp)
pred=model.predict(inp)
if(pred==0):
    print("this was a spam mail")
else:
    print("this was a legit mail")


