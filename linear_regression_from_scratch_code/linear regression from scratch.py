import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
# Linear Regression 
class LinearRegression() : 
	def __init__( self, learning_rate, iterations ) : 
		self.learning_rate = learning_rate 	
		self.iterations = iterations 
	#func for training 		
	def fit( self, X, Y ) : 
		self.m, self.n = X.shape 
		self.W = np.zeros( self.n ) 
		self.b = 0
		self.X = X 
		self.Y = Y 
		# gradient descent learning 	
		for i in range( self.iterations ) : 
			self.update_weights() 
		return self
	# func to update weights
	def update_weights( self ) : 
		Y_pred = self.predict( self.X ) 	
		dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) / self.m 
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
		self.W = self.W - self.learning_rate * dW 
		self.b = self.b - self.learning_rate * db 
		return self
	
	def predict( self, X ) : 
		return X.dot( self.W ) + self.b 
	
def main() : 
	df = pd.read_csv( r'linear_regression_from_scratch_code\salary_data.csv' ) 
	X = df.iloc[:,:-1].values 
	Y = df.iloc[:,1].values 
	X_train, X_test, Y_train, Y_test = train_test_split( 
	X, Y, test_size = 0.33, random_state = 2) 
	model = LinearRegression( iterations = 1000, learning_rate = 0.02 ) 
	model.fit( X_train, Y_train ) 
	
	Y_pred = model.predict( X_test ) 
	
	# print( "Predicted values ", np.round( Y_pred[:3], 2 ) ) 
	# print( "Real values	 ", Y_test[:3] ) 
	print( "weight : ", model.W[0] )  
	print( "bias : ",  model.b) 
	
	# Visualization on test set 
	plt.scatter( X_test, Y_test, color = 'red' ) #for scatterplot- dots
	plt.plot( X_test, Y_pred, color = 'blue' ) #for line
	plt.title( 'Salary vs Experience' ) 
	plt.xlabel( 'Years of Experience' ) 
	plt.ylabel( 'Salary' ) 
	plt.show() 
	
if __name__ == "__main__" : 
	
	main()
