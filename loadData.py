#import sys
#sys.path.append('/D:/Draper/Features/useThese')
import numpy as np
import csv
for x in range(0,1):
    print('imported script run begun', x) 
    
def main():
    text_file = open("bot_features_160309.txt", "r")
    #lines = text_file.read()#.split(',')
    lines = text_file.read().splitlines()
    text_file.close()
    #print(lines)
    
    numLines = len(lines)
    #numElements = lines[0].count(',') + 1;
    
    #villages = list(numLines)
    cropHealth = []
    cropHealthPerc = []
    edgeDen = []
    percDenEdges = []
    percEdges = []
    
    #import image data from .txt file
    
    villages = []
    
    for i in range(0,numLines):
        
        village = lines[i].split(',')
        villages.append(village)
        
    v = np.array(villages)
        
    IMGfeatures = v[:,2:]
    IMGvillages = v[:,0]
    IMGdates = v[:,1]
        
    #calc time distance in days
        
        
    #import survey data from excel file
    import pandas as pd
    xl = pd.ExcelFile("SurveyImage_All_withPCA.xlsx")
    df = xl.parse("SurveyImage_All_withPCA")
    excelFile = np.array(df)
    q1Index = 20
    q100Index = 222
    surveyAnswers = excelFile[:,q1Index:q100Index]
    TOWNVILL = 4
    surveyVillages = excelFile[:,TOWNVILL]
    IQ = 224
    LE = 290
    LH = 293
    MB = 313
    MD = 315
    MH = 319
        
    featureSet1 = excelFile[:,IQ:LE]
    featureSet2 = excelFile[:,LH:MB]
    featureSet3 = excelFile[:,MD:MH]
        
    featureSet = np.concatenate((featureSet1, featureSet2, featureSet3), axis=1)
        
    #do this after you choose the question, MAKE IT A FUNCTION PER QUESTION 
    #check for 9 (no response) and toss out response
    totalNumResponses = len(surveyVillages)
    numQuestions = len(surveyAnswers[0])
    responseMask = np.zeros([totalNumResponses,numQuestions])
    
    noResponse = 9;
    surveyVillagesProc =  [[] for k in range(numQuestions)]
    surveyAnswersProc = [[] for k in range(numQuestions)]
    numResPerQuestion = np.zeros(numQuestions)
    nR = 0
    for question in range(0,numQuestions):
        for i in range(0,totalNumResponses):
            if not (surveyAnswers[i,question] == noResponse or surveyAnswers[i,question] == 999):
                responseMask[i,question] = 1
                surveyVillagesProc[question].append(surveyVillages[i])
                surveyAnswersProc[question].append(surveyAnswers[i][question])
                numResPerQuestion[question] += 1
                    
    #aggregate responses and average
    newVilMask = np.zeros([totalNumResponses,numQuestions])
              
    for question in range(0,numQuestions):
        for i in range(0,totalNumResponses):
            if responseMask[i,question] == 1:
                if i == 0:
                    newVilMask[i,question] = 1
                elif not (surveyVillages[i] == surveyVillages[i-1]):
                    newVilMask[i,question] = 1

    numNewVilPerQ = sum(newVilMask)
    
    #count responses
    numFeatures = len(featureSet[0])
                                        
    numResPerVilPerQ = []
    featPerVperQ = []
    surResPerVperQ = []
    for question in range(0,numQuestions):
        numResPerVil = np.zeros(int(numNewVilPerQ[question]))
        featPerVil = np.zeros([int(numNewVilPerQ[question]),numFeatures])
        surResPerVil = np.zeros(int(numNewVilPerQ[question]))                                     
        currentVil = -1
        for i in range(0,totalNumResponses):
            if newVilMask[i,question] == 1:
                currentVil += 1
                featPerVil[currentVil,:] = featureSet[i,:]
                surResPerVil[currentVil] = surveyAnswers[i,question]
                numResPerVil[currentVil] = 1
            elif responseMask[i,question] == 1:
                    numResPerVil[currentVil] += 1
                    featPerVil[currentVil,:] = featPerVil[currentVil,:] + featureSet[i,:]
                    surResPerVil[currentVil] = surResPerVil[currentVil] + surveyAnswers[i,question]
                   
        numResPerVilPerQ.append(numResPerVil.copy())
        featPerVperQ.append(featPerVil.copy())
        surResPerVperQ.append(surResPerVil.copy())
    
    #Average survey and features
    for question in range(0,numQuestions):
        for currentVil in range(0,int(numNewVilPerQ[question])):
            featPerVperQ[question][currentVil] = featPerVperQ[question][currentVil]/numResPerVilPerQ[question][currentVil]
            surResPerVperQ[question][currentVil] = surResPerVperQ[question][currentVil]/numResPerVilPerQ[question][currentVil]
                                                                
    #for questions 4A, 4B, 5, 6A, 6B, 7A, 7B (indicies 3-9)
    for question in range(3,18):
        if question < 10:
            threshold = 2.9
        elif question < 15:
            threshold = 0.8
        else:
            threshold = 0.4
        
        binaryClasses(surResPerVperQ,numNewVilPerQ,featPerVperQ,question,threshold)
        
    print("terminado")
        
def binaryClasses(surResPerVperQ, numNewVilPerQ, featPerVperQ, question, threshold):
    labels = np.zeros(len(surResPerVperQ[question]))
    for currentVil in range(0,int(numNewVilPerQ[question])):
        if surResPerVperQ[question][currentVil] > threshold :
            labels[currentVil] = 1
        else:
            labels[currentVil] = 0
    avg = sum(labels)/len(labels)
    print("Average of labels: ", avg)
    
    featureVector = featPerVperQ[question].copy()
    trainingEnd = int(len(featureVector)*0.8)
                
    trainingSet_features = featureVector[0:trainingEnd,:]
    trainingSet_labels = labels[0:trainingEnd]
    testSet_features = featureVector[trainingEnd:,:]
    testSet_labels = labels[trainingEnd:]
                
    saveTrainingAndTestData(question,trainingSet_features,trainingSet_labels, testSet_features, testSet_labels)
                                                            

                                                                            
def saveTrainingAndTestData(question,trainingSet_features,trainingSet_labels, testSet_features, testSet_labels):
    
    trainingSet_features_str = 'trainingSet' + str(question) + 'features'
    trainingSet_labels_str = 'trainingSet' + str(question) + 'labels'
    testSet_features_str = 'testSet' + str(question) + 'features'
    testSet_labels_str = 'testSet' + str(question) + 'labels'
    np.save(trainingSet_features_str, trainingSet_features)
    np.save(trainingSet_labels_str,   trainingSet_labels)
    np.save(testSet_features_str,     testSet_features)
    np.save(testSet_labels_str,       testSet_labels)
    

if __name__ == '__main__':
    main()