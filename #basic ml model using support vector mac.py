#basic ml model using support vector machines

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  #just to compare the accuracy

data=load_breast_cancer()

X=data.data
Y=data.target

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

clf=SVC(kernel="linear",C=3)  #better classifier for this dataset
clf.fit(x_train,y_train)
#linear kernel is better because the data is linearly separable
#others kernels can also be used like rbf, poly etc
#c is the regularization parameter (soft margin)

clf2=KNeighborsClassifier(n_neighbors=5)
clf2.fit(x_train,y_train)

print(clf.score(x_test,y_test))
print(clf2.score(x_test,y_test))

