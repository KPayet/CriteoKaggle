from helperFunctions import *

import sys
import getopt
import os
from pyspark import SparkContext

sc = SparkContext(appName="CriteoKaggle")

# Load data
rawData = (sc.textFile("./dac_sample.txt", 10) # change value here for number of partitions
		     .map(lambda x: x.replace('\t', ',')))


weights = [.97, .03]
seed = 1234

rawTrainData, rawValidationData = rawData.randomSplit(weights, seed)
# Cache the data for performance
rawTrainData.cache()
rawValidationData.cache()

# Make sure that label is simply Id for the test set, i.e. the line of the observation + 59999999
rawTestSet = (sc.textFile("test.txt", 10) # change value here for number of partitions
                .map(lambda x: x.replace('\t', ','))
				.zipWithIndex()
				.map(lambda t: str( t[1] + 60000000 ) + "," + t[0] ))


# Create the dictionary for one-hot encoding from the train set and compute number of features in train set

trainFeatures = rawTrainData.map(parse)

OHEDict = makeOHEDict(trainFeatures)
numTrainOHEFeatures = len(OHEDict.keys())

# One-Hot encoding of data in train, valid and test sets

OHETrainData = rawTrainData.map(lambda point: parse2OHE(point, OHEDict, numTrainOHEFeatures))
OHETrainData.cache()

OHEValidationData = rawValidationData.map(lambda point: parse2OHE(point, OHEDict, numTrainOHEFeatures))
OHEValidationData.cache()

# in the test set, the label of the LabeledPoint will simply be the observation ID required in Kaggle submission
testOHE = rawTestSet.map(lambda point: parse2OHE(point, OHEDict, numTrainOHEFeatures))
testOHE.cache()

# Logistic regression on OHE features

from pyspark.mllib.classification import LogisticRegressionWithSGD

numIters = 50
stepSize = 10.
regParam = 1e-6

model0 = LogisticRegressionWithSGD.train(OHETrainData, numIters, stepSize, 1.0, None, regParam, 'l2', True)

# Model evaluation

validLogLossModel0 = evaluateModel(model0, OHEValidationData)

# Predict on the test set

testPredictions = (testOHE.map(lambda p: (p.label, getCTRProb(p.features, model0.weights, model0.intercept))) # in test label is simply observation Id, needed for Kaggle submission.
                          .map(lambda t: str(int(t[0])) + "," + str(t[1]))
                          .coalesce(1)
                          .saveAsTextFile("./predictions.csv"))


 # Using feature hashing instead of OHE

hashedTrainData = rawTrainData.map(lambda point: createHashedPoint(point, 2**15))
hashedTrainData.cache()
hashedValidationData = rawValidationData.map(lambda point: createHashedPoint(point, 2**15))
hashedValidationData.cache()


# retrain a logistic regression, but do some grid search before

# Initialize variables using values from initial model training
bestModel = None
bestLogLoss = float("inf")

stepSizes = [1, 5, 10, 20]
regParams = [0.000001, 0.0001, 0.001, 0.01]

for stepSize in stepSizes:
    for regParam in regParams:
        model = LogisticRegressionWithSGD.train(hashedTrainData, 500, stepSize, regParam=regParam, regType='l2', intercept=True)
        logLossVa = (hashedValidationData.map(lambda p: (p.label, getCTRProb(p.features, model.weights, model.intercept)))
                                         .map(lambda p: computeLogLoss(p[1], p[0]))
                                         .reduce(lambda a,b: a+b))/hashedValidationData.count()
#        logLossVa = evaluateModel(model, hashedValidationData)
        if (logLossVa < bestLogLoss):
            bestModel = model
            bestLogLoss = logLossVa


# predict on test set

testHashed = rawTestSet.map(lambda point: createHashedPoint(point, 2**15))

testPredictions = (testHashed.map(lambda p: (p.label, getCTRProb(p.features, bestModel.weights, bestModel.intercept))) # in test label is simply observation Id, needed for Kaggle submission.
                             .map(lambda t: str(int(t[0])) + "," + str(t[1]))
                             .coalesce(1)
                             .saveAsTextFile("predictions_2.csv"))







