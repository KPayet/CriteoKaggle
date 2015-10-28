"""
This file contains the helper functions for preparing the data, training the model ....
"""

"""
Function used to one-hot encode categorical features of the dataset
i.e. turn the categorical features into binary features (K binary features for a K-level nominal feature)
The produced encoding is stored in a SparseVector for space efficiency
"""
from pyspark.mllib.linalg import SparseVector
def oneHotEncode(features, featDict, numFeatures):

    N = [];

    for feat in features:
        if featDict.has_key(feat):
            N.append(featDict[feat])
    N = sorted(N)

    return SparseVector(numFeatures, N, [1.]*len(N))


"""
This function creates a dictionary used for one-hot encoding the data
"""
def makeOHEDict(input):

    distinctFeats = (input
                       .flatMap(lambda s: s)
                       .distinct())
    dic = (distinctFeats
                .zipWithIndex()
                .collectAsMap())
    return dic

"""
Converts the comma separated input observations into a list of (feature ID, value) tuples
feature ID = [0, numFeatures - 1]
The function returns only the features, not the label (which is the first value of the string).
"""
def parse(point):

    i=0
    pList = []
    for feat in point.split(',')[1:]:
        pList.append((i, feat))
        i=i+1

    return pList

"""
Transforms the raw string format of observations into a LabeledPoint.
For test data, the string is modified so that the first number is the observation ID required for Kaggle submission
"""
from pyspark.mllib.regression import LabeledPoint
def parse2OHE(point, dict, numFeatures):

    label = point.split(',')[0]
    feats = parse(point)

    encodedFeats = oneHotEncode(feats, dict, numFeatures)

    return LabeledPoint(label, encodedFeats)

from math import log

"""
This function computes the logloss
"""
def computeLogLoss(prob, y):

    epsilon = 10e-12

    if prob == 0:
        prob = prob + epsilon
    elif prob == 1:
        prob = prob - epsilon

    if y == 1:
        loss = -log(prob)
    else:
        loss = -log(1-prob)

    return loss

"""
This function computes the probability of CTR given observed features X, and weights for a logistic regression model W, and the W0
"""
from math import exp
def getCTRProb(X, W, W0):

    rawPrediction = W0 + X.dot(W)

    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    return 1/(1+exp(-rawPrediction))

"""
Computes the log loss on the a data set to evaluate the model
"""

def evaluateModel(model, data):

    logLoss = (data.map(lambda p: (p.label, getCTRProb(p.features, model.weights, model.intercept)))
                 .map(lambda p: computeLogLoss(p[1],p[0]))
                 .reduce(lambda a,b: a+b))/data.count()

    return logLoss

"""
Hash function used for the feature hashing trick
"""
from collections import defaultdict
import hashlib

def hashThis(nBuckets = 2**16, features):

    mapping = {}

    for i, cat in features:
        featureString = cat + str(i)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % nBuckets)

    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)

"""
This function uses the parse function defined above, and the hash function
to create for each observation a LabeledPoint, with label, and where features is a
SparseVector containing the hashed features.
"""
def createHashedPoint(point, nBuckets):

    splitPoint = point.split(',')
    label = splitPoint[0]
    feats = parse(point)

    hashedFeats = hashThis(nBuckets, feats)

    nonZeroIndices = sorted([key for key in hashedFeats])
    nonZeroValues = [hashedFeats[key] for key in nonZeroIndices]

    return LabeledPoint(label, SparseVector(nBuckets, nonZeroIndices, nonZeroValues))

