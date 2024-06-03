from sklearn import datasets,linear_model
from sklearn.model_selection import cross_val_score
dia=datasets.load_diabetes()
x=dia.data[:150]
y=dia.target[:150]
model=linear_model.Lasso()
print(cross_val_score(model,x,y,cv=3))
# cv means data div into 3 subsets/folds
