#basic ml model using SKLEARN's linear regression model
#This is a binary classification
#lr model learn relationship between actual and our studied score then make a prediction on the test data and measure the accuracy in some sort of evaluation metric

#STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#------------------------------------------------------------
#STEP 2: PREPARE THE DATA
                         #needs to clean if data is very large
data={
    'Hours_Studied':[1,2,3,4,5,6,7,8,9,10],
    'Score':[12,25,32,40,55,65,72,80,90,77]
}
df=pd.DataFrame(data) #ValueError: All arrays must be of the same length

#Features and target
X=df[['Hours_Studied']] #input {dataframe} #"Passing a set as an indexer is not supported. Use a list instead." 
                        #Expected a 2-dimensional container but got <class 'pandas.core.series.Series'> instead. Pass a DataFrame containing a single row (i.e. single sample) or a single column (i.e. single feature) instead.
y=df['Score'] #output {series}

#---------------------------------------------------------------
#STEP 3: SPLIT THE DATA

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)

#------------------------------------------------------------------
#STEP 4: TRAIN THE MODEL

model=LinearRegression() #teaching the model to predict the score
                          #lr model learn relationship between actual and our studied score then make a prediction on the test data and measure the accuracy in some sort of evaluation metric
model.fit(X_train,y_train)

#---------------------------------------
#STEP 5: MAKE PREDICTIONS

y_pred=model.predict(X_test)
print("Predictions:",y_pred)

#------------------------------------
#STEP 6: EVALUATE THE MODEL

mse=mean_squared_error(y_test,y_pred) #tells how far we were from the predicted result  
print("MEAN SQUARED ERROR:",mse)

#---------------------------------------

#STEP 7: VISUALIZE RESULTS
#convert X to 1D array for plotting

X_1d=X['Hours_Studied'].values
y_pred_full=model.predict(X).flatten()

plt.scatter(X_1d,y,color='blue',label='Actual Scores')
plt.plot(X_1d,y_pred_full,color='red',label='Predicted Line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours Studied vs Score Prediction')
plt.legend()
plt.show()