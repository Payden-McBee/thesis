# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:55:25 2016

@author: Payden McBee
"""


from sklearn.metrics import accuracy_score
import numpy as np

from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA

if 1:
    trainingSet_features=np.load('trainingSet4Bfeatures.npy')
    trainingSet_labels=np.load('trainingSet4Blabels.npy')
    testSet_features=np.load('testSet4Bfeatures.npy')
    testSet_labels=np.load('testSet4Blabels.npy')
else:
    trainingSet_features=np.load('trainingSet9Bfeatures.npy')
    trainingSet_labels=np.load('trainingSet9Blabels.npy')
    testSet_features=np.load('testSet9Bfeatures.npy')
    testSet_labels=np.load('testSet9Blabels.npy')  

nnC = KNeighborsClassifier(n_neighbors=5)
nnC.fit(trainingSet_features, trainingSet_labels) 
nnC_predictions = nnC.predict(testSet_features)
print(accuracy_score(testSet_labels,nnC_predictions))

svmC = svm.SVC()
svmC.fit(trainingSet_features, trainingSet_labels) 
svmCpredictions = svmC.predict(testSet_features)
print(accuracy_score(testSet_labels,svmCpredictions))


rfC = RandomForestClassifier(n_estimators=100)
rfC.fit(trainingSet_features, trainingSet_labels) 
rfC_predictions = rfC.predict(testSet_features)
print(accuracy_score(testSet_labels,rfC_predictions))

ldaC = LDA(solver='lsqr')
ldaC.fit(trainingSet_features, trainingSet_labels) 
ldaC_predictions = ldaC.predict(testSet_features)
print(accuracy_score(testSet_labels,ldaC_predictions))