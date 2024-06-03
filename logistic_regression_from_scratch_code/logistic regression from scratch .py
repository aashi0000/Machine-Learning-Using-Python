
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class NewLogisticRegression() : 
	def __init__( self, learning_rate, iterations ) :		 
		self.learning_rate = learning_rate		 
		self.iterations = iterations 
			 
	def fit( self, X, Y ) :		 
				 
		self.m, self.n = X.shape		 
		self.W = np.zeros( self.n )		 
		self.b = 0		
		self.X = X		 
		self.Y = Y 
			
		for i in range( self.iterations ) :			 
			self.update_weights()			 
		return self
	
	def update_weights( self ) :		 
		A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) ) 
			
		tmp = ( A - self.Y.T )		 
		tmp = np.reshape( tmp, self.m )		 
		dW = np.dot( self.X.T, tmp ) / self.m		 
		db = np.sum( tmp ) / self.m 
		 
		self.W = self.W - self.learning_rate * dW	 
		self.b = self.b - self.learning_rate * db 
		
		return self
	
	def predict( self, X ) :	 
		pred = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )		 
		Y = np.where( pred > 0.5, 1, 0 )		 
		return Y 

def main():
    df=pd.read_csv(r'logistic_regression_from_scratch_code\diabetes.csv')
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1:].values 
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 2 ) 
    classifier=NewLogisticRegression(learning_rate=0.01,iterations=2000)
    classifier.fit(X_train,Y_train)
    x_train_pred=classifier.predict(X_train)
    train_acc=accuracy_score(Y_train,x_train_pred)
    print("accuracy of training data: ",train_acc)
    x_test_pred=classifier.predict(X_test)
    test_acc=accuracy_score(Y_test,x_test_pred)
    print("accuracy of testing data: ",test_acc)
    #predictive system
    inp=(6,148,72,35,0,33.6,0.627,50)
    inp=np.asarray(inp)
    inp=inp.reshape(1,-1)
    inp=scaler.transform(inp)
    pred=classifier.predict(inp)
    if(pred[0]==0):
        print("person is NOT diabetic")
    else:
        print("person IS diabetic")
    
if __name__ == "__main__" :	 
	main()
