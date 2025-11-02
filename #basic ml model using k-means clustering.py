#basic ml model using k-means clustering
#unsupervised


from sklearn.cluster import KMeans
from sklearn.preprocessing import scale  #preprocessing purpose used to standardize the data
from sklearn.datasets import load_digits

digits=load_digits()
data=scale(digits.data)  

model=KMeans(n_clusters=10,init="random",n_init=10) #init=random means random initialization of centroids
#n_init=10 means the algorithm will run 10 times with different centroid seeds

model.fit(data)

#we dont have to test the model as it is unsupervised learning

print(model.predict(data[3:5]))