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

def main():
    
    for question in range(3,18):
        
        print("Question ", question, " Percent Accuracy")

        trainingSet_features, trainingSet_labels, testSet_features, testSet_labels = loadTrainingAndTestData(question)
        #print(len(trainingSet_features))
        #print(trainingSet_labels)
        #print(len(testSet_features))
        #print(len(testSet_labels))
        
        #print(trainingSet_labels)
        nnC = KNeighborsClassifier(n_neighbors=5)
        nnC.fit(trainingSet_features, trainingSet_labels) 
        nnC_predictions = nnC.predict(testSet_features)
        print("Nearest Neighbor: %.2f" % (100*accuracy_score(testSet_labels,nnC_predictions)),"%")

        svmC = svm.SVC()
        svmC.fit(trainingSet_features, trainingSet_labels) 
        svmCpredictions = svmC.predict(testSet_features)
        print("Support Vector Machines: %.2f" % (100*accuracy_score(testSet_labels,svmCpredictions)),"%")

        rfC = RandomForestClassifier(n_estimators=100)
        rfC.fit(trainingSet_features, trainingSet_labels) 
        rfC_predictions = rfC.predict(testSet_features)
        print("Random Forrest:  %.2f" % (100*accuracy_score(testSet_labels,rfC_predictions)),"%")

        ldaC = LDA(solver='lsqr')
        ldaC.fit(trainingSet_features, trainingSet_labels) 
        ldaC_predictions = ldaC.predict(testSet_features)
        print("Linear Discriminant Analysis Classifier: %.2f" % (100*accuracy_score(testSet_labels,ldaC_predictions)),"%")

def loadTrainingAndTestData(question):
    
    trainingSet_features_str = 'trainingSet' + str(question) + 'features.npy'
    trainingSet_labels_str = 'trainingSet' + str(question) + 'labels.npy'
    testSet_features_str = 'testSet' + str(question) + 'features.npy'
    testSet_labels_str = 'testSet' + str(question) + 'labels.npy'
    trainingSet_features = np.load(trainingSet_features_str)
    trainingSet_labels = np.load(trainingSet_labels_str)
    testSet_features = np.load(testSet_features_str)
    testSet_labels = np.load(testSet_labels_str)
    return trainingSet_features, trainingSet_labels, testSet_features, testSet_labels
    
if __name__ == '__main__':
    main()