#basic ml model using Sklearns's logistic regression model
#This is a binary classification


#IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


#PREPARE THE DATA

data={
    'Hours_Studied':[1,2,3,4,5,6,7,8,9,10],
    'Score':[12,25,32,40,55,65,72,80,90,77]
}
df=pd.DataFrame(data)


#CREATE A BINARY TARGET :PASS OR FAIL

df['Pass']=(df['Score']>=50).astype(int)

X=df[['Hours_Studied']]  #feature
y=df['Pass']  #target (0 or 1)


#SPLIT THE DATA

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=40)


#TRAIN THE LOGISTIC REGRESSION MODEL

model=LogisticRegression()
model.fit(X_train,y_train)


#MAKE PREDICTIONS

y_pred=model.predict(X_test)
print("Predictions:",y_pred)


#EVALUATE THE MODEL

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))


#VISUALIZE DECSISION BOUNDARY

X_sorted=np.linspace(0,12,100).reshape(-1,1)
y_prob=model.predict_proba(X_sorted)[:,1]

plt.scatter(X,y,color='blue',label='Actual Pass/Fail')
plt.plot(X_sorted,y_prob,color='red',label='Predicted Probability(Pass)')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression : Pass vs Fail')
plt.legend()
plt.show()
