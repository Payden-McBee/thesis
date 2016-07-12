#import sys
#sys.path.append('/D:/Draper/Features/useThese')
import numpy as np
import csv
for x in range(0,1):
    print('imported script run begun', x) 
    

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
    
    numResPerVilPerQ.append(numResPerVil)
    featPerVperQ.append(featPerVil)
    surResPerVperQ.append(surResPerVil)

#Average survey and features
for question in range(0,numQuestions):
    for currentVil in range(0,int(numNewVilPerQ[question])):
        featPerVperQ[question][currentVil] = featPerVperQ[question][currentVil]/numResPerVilPerQ[question][currentVil]
        surResPerVperQ[question][currentVil] = surResPerVperQ[question][currentVil]/numResPerVilPerQ[question][currentVil]

#choose 2 q's
question4B = 4
labels4B = np.zeros(len(surResPerVperQ[question4B]))
for currentVil in range(0,int(numNewVilPerQ[question4B])):
    if surResPerVperQ[question4B][currentVil] > 2 :
        labels4B[currentVil] = 0
    else:
        labels4B[currentVil] = 1
featureVector4B = featPerVperQ[question4B]
trainingEnd4B = int(len(featureVector4B)*0.8)

trainingSet4Bfeatures = featureVector4B[0:trainingEnd4B,:]
trainingSet4Blabels = labels4B[0:trainingEnd4B]
testSet4Bfeatures = featureVector4B[trainingEnd4B:,:]
testSet4Blabels = labels4B[trainingEnd4B:]

np.save('trainingSet4Bfeatures',trainingSet4Bfeatures)
np.save('trainingSet4Blabels',trainingSet4Blabels)
np.save('testSet4Bfeatures',testSet4Bfeatures)
np.save('testSet4Blabels',testSet4Blabels)

question9B = 16
labels9B = np.zeros(len(surResPerVperQ[question9B]))
for currentVil in range(0,int(numNewVilPerQ[question9B])):
    if surResPerVperQ[question9B][currentVil] > 0.5:
        labels9B[currentVil] = 0
    else:
        labels9B[currentVil] = 1
featureVector9B = featPerVperQ[question9B] 
trainingEnd9B = int(len(featureVector9B)*0.8)

trainingSet9Bfeatures = featureVector9B[0:trainingEnd9B,:]
trainingSet9Blabels = labels9B[0:trainingEnd9B]
testSet9Bfeatures = featureVector9B[trainingEnd9B:,:]
testSet9Blabels = labels9B[trainingEnd9B:]        

np.save('trainingSet9Bfeatures',trainingSet9Bfeatures)
np.save('trainingSet9Blabels',trainingSet9Blabels)
np.save('testSet9Bfeatures',testSet9Bfeatures)
np.save('testSet9Blabels',testSet9Blabels)


print("terminado")