# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:44:00 2018

@author: Akhilesh
"""

#1. Assumption is that the minority samples are critical to predict

#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns

#setting basic visualization
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)
plt.rc("font", size=14)

#Importing the data
dataset = pd.read_csv("Dataset_spine.csv")

#checking dimensions,missed values and maxima/minima
dataset.shape
dataset.isnull().sum(axis=0)
dataset.info()
dataset.describe()

#visualising the data flow

plt.plot(dataset)
plt.legend(labels=['1','2','3','4','5','6','7','8','9','10','11','12'])
plt.savefig('Back.png',dpi=400)
plt.show()

#checking the optimal number of cluster we can get
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters =i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('Inertia.png',dpi=400)
plt.show()

#Applying Kmeans clustering
kmeans=KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_Kmeans=kmeans.fit_predict(dataset)
Class=pd.DataFrame(y_Kmeans,columns=['class'])
dataset=pd.concat([dataset,Class],axis=1)

#Plotting Count after clustering
sns.countplot(x=dataset['class'],palette='hls')
plt.savefig('countclass_plot.png',dpi=400)
plt.show()



#Dividing independent variable and dependent variable
dataset.shape
Y=dataset.iloc[:,12].values
X=dataset.iloc[:,0:12].values

#Applying PCA for 2D visualisation
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
dataset['x']=pca.fit_transform(X)[:,0]
dataset['y']=pca.fit_transform(X)[:,1]

#cluster Visulaisation
plt.scatter(dataset[y_Kmeans==0]["x"],dataset[y_Kmeans==0]["y"],s=100,c='red',label='SpineProblem')
plt.scatter(dataset[y_Kmeans==1]["x"],dataset[y_Kmeans==1]["y"],s=100,c='green',label='OtherProblem')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.title('Clusters of Patients')
plt.legend()
plt.savefig('BackPain.png',dpi=400)
plt.show()


#splitting data into training and test set
training_feature, test_feature, training_target, test_target = train_test_split(X,Y,test_size=0.2)


#Training the Data through Decison Tree
classifier=DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5,min_samples_split=5,random_state=100)
classifier.fit(training_feature,training_target)

#Testing the Model using test data
y_pred=classifier.predict(test_feature)

#Evaluation

#1.Checking Confusion Metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_target,y_pred)

#2.Visualising Confusion matrix
unique=np.unique(test_target)
fig, ax= plt.subplots(figsize=(22,22))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=unique,yticklabels=unique)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix2.png',dpi=100)
plt.show()


#3.Applying 3 cross validation
accuracy=cross_val_score(classifier,training_feature,training_target, cv=3)
accuracy.mean()
accuracy.std()

#4.CVgridSearch for pruning activities
from sklearn.model_selection import GridSearchCV
params= [{'max_depth':[1,2,3,4,5],'min_samples_leaf':[3,4,5],'min_samples_split':[3,4,5]}]
grid_search=GridSearchCV(estimator=classifier,param_grid=params,scoring='accuracy',cv=3)
grid_search=grid_search.fit(training_feature,training_target)
best_accuracy=grid_search.best_score_
best_param= grid_search.best_params_
grid_search.cv_results_

#5.Prining Accuracy Score
print("Accuracy on training set: {:.3f}".format(classifier.score(training_feature,training_target)))
print("Accuracy on test set: {:.3f}".format(classifier.score(test_feature,test_target)))

#6.Printing Precision,Recall,Accuracy
from sklearn import metrics
print(metrics.classification_report(test_target,y_pred))







