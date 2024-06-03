import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class svm_classifier():
    def __init__(self,learning_rate,no_of_iterations,lambda_parameter):
        self.no_of_iterations=no_of_iterations
        self.learning_rate=learning_rate
        self.lambda_parameter=lambda_parameter
    
    def fit(self,x,y):
        self.m,self.n=x.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.x=x
        self.y=y
        for i in range(self.no_of_iterations):
            self.update_weights()
        
        
    def update_weights(self):
        y_label=np.where(self.y<=0,-1,1)
        for index, x_i in enumerate(self.x):
            condition= y_label[index]*(np.dot(x_i,self.w)-self.b) >= 1
            if condition==True:
                dw=2*self.lambda_parameter*self.w
                db=0
            else:
                dw=2*self.lambda_parameter*self.w-np.dot(x_i,y_label[index])
                db=y_label[index]
            self.w=self.w-self.learning_rate*dw
            self.b=self.b-self.learning_rate*db
        
    def predict(self,x):
        output=np.dot(x,self.w)-self.b
        predicted_labels=np.sign(output)
        y_hat=np.where(predicted_labels<1,0,1)
        return y_hat

def main():
    
    diabetes_data=pd.read_csv(r"svm_from_scratch_code\diabetes.csv")

    features=diabetes_data.drop(columns="Outcome",axis=1)
    target=diabetes_data['Outcome']

    scaler=StandardScaler()
    features=scaler.fit_transform(features)

    x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.28,random_state=2)
    classifier=svm_classifier(learning_rate=0.001,no_of_iterations=1000,lambda_parameter=0.01)
    classifier.fit(x_train,y_train)

    x_train_pred=classifier.predict(x_train)
    train_acc=accuracy_score(y_train,x_train_pred)
    print("training accuracy: ",train_acc)

    x_test_pred=classifier.predict(x_test)
    test_acc=accuracy_score(y_test,x_test_pred)
    print("testing accuracy: ",test_acc)

    input_data=(1,85,66,29,0,26.6,0.351,31)
    input_data=np.asarray(input_data)
    input_data=input_data.reshape(1,-1)
    input_data=scaler.fit_transform(input_data)
    pred=classifier.predict(input_data)
    print(pred)
    input_data=(6,148,72,35,0,33.6,0.627,50)
    input_data=np.asarray(input_data)
    input_data=input_data.reshape(1,-1)
    input_data=scaler.transform(input_data)
    pred=classifier.predict(input_data)
    print(pred)

if __name__ == "__main__" :	 
	main()   