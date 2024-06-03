
import pandas as pd
#df = pd.read_csv(r"C:\Users\aashi\OneDrive\Desktop\new\data.csv")
#print(df.head())

#n1=int(input("enter first no : "))
#n2=int(input("enter second no : "))
#print("sum is ",n1+n2) 

# mylist=['1',2,45,"mylol"]
# mylist.append('a')
# print(mylist)

#list with []
#tuple with ()
#set with {}
#dictionary with {:}

#if elif else m brackets m condn+ dono ke baad :
#for i in range(5) or i in seq :
#while condn:
#for func defn-> def funcname(paras):

#numpy for matrix wagehra->numpy arrays
# import numpy as np
# newlist=[1,2,3,4,5]
# np_array=np.array(newlist)
# print(np_array)
# print(np_array.shape)
# c=np.array([(1,2,3,4),(5,6,7,8)],dtype=float)
# print(c)
#np.add(a,b)..sim subtract multiply divide transpose reshape

# import pandas as pd
# from sklearn.datasets import load_wine
# wine_dataset=load_wine()
# # print(wine_dataset)
# # print(type(wine_dataset))
# wine_df=pd.DataFrame(wine_dataset.data,columns=wine_dataset.feature_names)
# print(wine_df.head())

#dataset load kro toh dataframe m convert krna hota
#csv file se direct dataframe milta
# wine_df.to_csv('wine.csv')


# print(wine_df.value_counts('hue'))
#count no of vals (har val ka)in hue

#groupby(colname) kr skte 
# df.mean() sbke mean ke liye
# print(wine_df.std()) for standard deviation
#similarly .min .max
# df.describe() saare ke liye


# wine_df['ash2']= wine_dataset.target() adding a col
# print(wine_df.drop(axis=1,columns='alcohol'))
# .drop() index=jo bhi num/ column=jo bhi col axis=0 if row,1 if col } to rem a row/col
# print(wine_df.iloc[3]) locating row, iloc[:,0] locating col
# print(wine_df.corr()) for checking +ve,-ve correlation of vars w respect to each other (directly/inversely prop)

import matplotlib.pyplot as plt
import numpy as np
# x=np.linspace(0,10,100)
# #100 evenly spaced vals b/w 0 and 10
# y=np.sin(x)
# z=np.cos(x)
# plt.plot(x,y) #to plot
# plt.xlabel('vals')
# plt.ylabel('sin')
# plt.title('yassss')
# plt.plot(x,z) #to plot
# plt.xlabel('vals')
# plt.ylabel('cos')
# plt.show() #to display plot
#do kiye toh ek m hi ayenge

#->for generalized plot
#fig1=plt.figure()
#ax=fig1.add_axes([0,0,1,1]) mtlb 0,0 se shuru and 1,1 tk jata area of plot
#ax.plotnm(paras) phir plt.show()

import seaborn as sns 
# tips=sns.load_dataset('tips')
# # print(tips.head())
# sns.set_theme()
# sns.relplot(data=tips, x='total_bill',y='tip',col='time',hue='smoker',style='smoker',size='size')
# plt.show()
#col to diff by, hue- colour by certain col, style by col,size of symbol by other col
#can also build basic plots like scatterplots,countplots,barplots,distribution plots
#heatmap se correlation matrix nikalna








