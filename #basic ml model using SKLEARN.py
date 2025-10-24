import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied=np.array([20,50,32,65,23,43,10,5,22,35,29,5,56]).reshape(-1,1)
scores=np.array([56,83,47,93,47,82,45,23,55,67,57,4,89]).reshape(-1,1) #actual connected realtionship not random numbers

#_______________________________________________________________________________
#they are in horizontal format, for sklearn we need vertical format             |
#reshaping the arrays to vertical format using reshape(-1,1)                    |
#we need to reshape because sklearn expects 2D arrays for features and targets  |
#outliers can affect the model performance significantly, for small data        |
#sets, accuracy gets shitty with outliers                                       |
# __________________________________________________________________________

#training the model
model=LinearRegression()
model.fit(time_studied,scores) #making the optimized line
#and the model is done

print(model.predict(np.array([20]).reshape(-1,1)))

#visualizing the model
plt.scatter(time_studied,scores)  #original graph of data points
plt.plot(np.linspace(0,70,100).reshape(-1,1),model.predict(np.linspace(0,70,100).reshape(-1,1)),'r')

#______________________________________________________________________
#linspace generates 100 points between 0 and 70 for a smooth line      |
#we needed to reshape it to vertical format for prediction             |
#model.predict gives the predicted scores for those time studied values|
#_______________________________________________________________________

plt.ylim(0,100)
plt.show()

#testing the model
#eg. 80% on training and 20% on testing

time_train,time_test,scores_train,scores_test=train_test_split(time_studied,scores,test_size=0.2) #0.2 means 20% test data
model2=LinearRegression()
model2.fit(time_train,scores_train)

print(model2.score(time_test,scores_test))#gives accuracy of the model on test data

plt.scatter(time_train,scores_train)
plt.plot(np.linspace(0,70,100).reshape(-1,1),model.predict(np.linspace(0,70,100).reshape(-1,1)),'r')
plt.show()


