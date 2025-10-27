#K-nn model

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data=load_breast_cancer()
#print(data.feature_names,data.target_names)

x_train,x_test,y_train,y_test=train_test_split(np.array(data.data),np.array(data.target),test_size=0.2) #does not take the first 80 and last 20 so, it shuffles everytime

clf=KNeighborsClassifier(n_neighbors=3) #3 nearest neighbors will be considered
clf.fit(x_train,y_train)

print(clf.score(x_test,y_test)) #gives accuracy of the model on test data

print(clf.predict([x_test[10]])) #predicting the first data point of test data