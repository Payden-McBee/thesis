# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:55:25 2016

@author: Payden McBee
"""
from sklearn import svm 
from sklearn.metrics import accuracy_score
import numpy as np

trainingSet4Bfeatures=np.load('trainingSet4Bfeatures.npy')
trainingSet4Blabels=np.load('trainingSet4Blabels.npy')
testSet4Bfeatures=np.load('testSet4Bfeatures.npy')
testSet4Blabels=np.load('testSet4Blabels.npy')

clf = svm.SVC()
clf.fit(trainingSet4Bfeatures, trainingSet4Blabels) 
predictions = clf.predict(testSet4Bfeatures)
print(accuracy_score(testSet4Blabels,predictions))

trainingSet9Bfeatures=np.load('trainingSet9Bfeatures.npy')
trainingSet9Blabels=np.load('trainingSet9Blabels.npy')
testSet9Bfeatures=np.load('testSet9Bfeatures.npy')
testSet9Blabels=np.load('testSet9Blabels.npy')  

clf2 = svm.SVC()
clf2.fit(trainingSet9Bfeatures, trainingSet9Blabels) 
predictions2 = clf2.predict(testSet9Bfeatures)
print(accuracy_score(testSet9Blabels,predictions2))