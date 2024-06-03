import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

class LassoRegression() : 
    def __init__( self, learning_rate, iterations, lambda_parameter ) : 
        self.learning_rate = learning_rate 	
        self.iterations = iterations 
        self.lambda_parameter = lambda_parameter
    
    def update_weights(self):
        y_pred=self.predict(self.X)
        dw=np.zeros(self.n)
        for i in range(self.n):
            if self.W[i]>0:
                dw[i]=(-(2*(self.X[:,i]).dot(self.Y-y_pred))+self.lambda_parameter)/self.m
            else:
                dw[i]=(-(2*(self.X[:,i]).dot(self.Y-y_pred))-self.lambda_parameter)/self.m
        db=-2*np.sum(self.Y-y_pred)/self.m
        self.W=self.W-self.learning_rate*dw
        self.b=self.b-self.learning_rate*db
    def fit( self, X, Y ) : 
        self.m, self.n = X.shape 
        self.W = np.zeros( self.n ) 
        self.b = 0
        self.X = X 
        self.Y = Y 	
        for i in range( self.iterations ) : 
            self.update_weights() 
        return self
    
    def predict( self, X ) : 
	    return X.dot( self.W ) + self.b 
 
def main():
    df = pd.read_csv( r'unit 3\salary_data.csv' ) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,1].values 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 2)
    model=LassoRegression(learning_rate=0.001,iterations=1000,lambda_parameter=0.01)
    model.fit(X_train,Y_train)
    
    y_prediction=model.predict(X_test)
    
    plt.scatter( X_test, Y_test, color = 'red' ) #for scatterplot- dots
    plt.plot( X_test, y_prediction, color = 'blue' ) #for line
    plt.title( 'Salary vs Experience' ) 
    plt.xlabel( 'Years of Experience' ) 
    plt.ylabel( 'Salary' ) 
    plt.show() 


if __name__=="__main__":
    main()
 
    
 

