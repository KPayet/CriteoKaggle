MultiLogLoss <- function(act, pred)
{
    eps = 1e-15;
    nr <- nrow(pred)
    pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
    pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
    ll = sum(act*log(pred) + (1-act)*log(1-pred))
    ll = ll * -1/(nrow(act))      
    return(ll);
}

require(readr)

rawDataTrain = read_delim("dac_sample.txt", delim="\t", col_names = F)
rawDataTest = read_delim("test.txt", delim="\t", col_names = F)

#countNA = function(x) {N = table(is.na(x)); return(N);}

#apply(X = data, MARGIN = 2, FUN = countNA)

# use only integer features for the moment

# intData = rawData[,1:14]
# nEntries = nrow(intData)

# We have a lot of missing data:

# for(i in 2:14) {
#     N = table(is.na(intData[,i]))
#     if(length(N)>1) {
#         N = N[[2]]
#     }
#     else {
#         if(!is.na(intData[1,i])) {
#             N = 0
#         }
#         else 
#             N = nEntries
#     }
#     print(paste("NA fraction for Integer feature", i, ":", N/nEntries))
# }

# There are many ways to deal with this:
# If a feature has mainly missing values, it is sometimes better to simply discard it
# Here the feature for which we might want to do that is X13, with 77+% of missing values
# Second option is to impute the value. Either fill it with 0, or the mean/median of the column
# Third is to impute the missing values using a regression model
# We will first impute these missing values by filling with the mean/median.
# Given the size of the real dataset, and how noisy it is (lots of missing values,
# + we're going to have to make lots of approximations/drop some variables, regression imputation is 
# not really realistic at that time)
#
# Compute the means and medians for each feature
# means = apply(intData, MARGIN = 2, mean, na.rm=T) #mean
# medians = apply(intData, MARGIN = 2, median, na.rm=T) #medians

# intDataImputed = intData
# 
# intDataImputed$X2[is.na(intDataImputed$X2)] = rep(means[2], length(intDataImputed$X2[is.na(intDataImputed$X2)]))
# intDataImputed$X3[is.na(intDataImputed$X3)] = rep(means[3], length(intDataImputed$X3[is.na(intDataImputed$X3)]))
# intDataImputed$X4[is.na(intDataImputed$X4)] = rep(means[4], length(intDataImputed$X4[is.na(intDataImputed$X4)]))
# intDataImputed$X5[is.na(intDataImputed$X5)] = rep(means[5], length(intDataImputed$X5[is.na(intDataImputed$X5)]))
# intDataImputed$X6[is.na(intDataImputed$X6)] = rep(means[6], length(intDataImputed$X6[is.na(intDataImputed$X6)]))
# intDataImputed$X7[is.na(intDataImputed$X7)] = rep(means[7], length(intDataImputed$X7[is.na(intDataImputed$X7)]))
# intDataImputed$X8[is.na(intDataImputed$X8)] = rep(means[8], length(intDataImputed$X8[is.na(intDataImputed$X8)]))
# intDataImputed$X9[is.na(intDataImputed$X9)] = rep(means[9], length(intDataImputed$X9[is.na(intDataImputed$X9)]))
# intDataImputed$X10[is.na(intDataImputed$X10)] = rep(means[10], length(intDataImputed$X10[is.na(intDataImputed$X10)]))
# intDataImputed$X11[is.na(intDataImputed$X11)] = rep(means[11], length(intDataImputed$X11[is.na(intDataImputed$X11)]))
# intDataImputed$X12[is.na(intDataImputed$X12)] = rep(means[12], length(intDataImputed$X12[is.na(intDataImputed$X12)]))
# intDataImputed$X13[is.na(intDataImputed$X13)] = rep(means[13], length(intDataImputed$X13[is.na(intDataImputed$X13)]))
# intDataImputed$X14[is.na(intDataImputed$X14)] = rep(means[14], length(intDataImputed$X14[is.na(intDataImputed$X14)]))

# since I'm just working with a subset of data, I'm going to use the mice package. It is quite long,
# and in the end, produces results that are close to what I would have had with only putting to 0 ?
# most values are imputed to zero.
# Perhaps mean imputation is better in the end...

# require(mice)
# 
# if(!file.exists("intDataImputed.df")) {
#     intDataImputed = complete(mice(intData))
# } else {
#     intDataImputed = readRDS("intDataImputed.df")
# }


# for(i in 2:14) {
#     N = table(is.na(intDataImputed[,i]))
#     if(length(N)>1) {
#         N = N[[2]]
#     }
#     else {
#         if(!is.na(intDataImputed[1,i])) {
#             N = 0
#         }
#         else 
#             N = nEntries
#     }
#     print(paste("NA fraction for Integer feature", i, ":", N/nEntries))
}

# The fraction of missing data is now zero.

# We can now train our first logistic regression model

# intDataImputed = cbind(intData$X1, intDataImputed)
# names(intDataImputed)[1] = "X1"

# Shuffle the dataset

# intDataImputed = intDataImputed[sample(1:nrow(intDataImputed), replace = FALSE),]

# Average removal, and standard deviation division

# require(caret)

# intDataImputedScaled = predict(preProcess(intDataImputed[,2:14]), intDataImputed)
# 
# intDataImputedScaled = cbind(intDataImputed$X1, intDataImputedScaled)
# names(intDataImputedScaled)[1] = "X1"

# glm

# model = glm(X1~., data = intDataImputedScaled, family = "binomial")
# pred = predict(object = model, newdata = intDataImputedScaled, type="response")
# 
# labels = intDataImputedScaled$X1[!is.na(pred)]
# pred = pred[!is.na(pred)]
# 
# require(ROCR)
# predROC = prediction(pred, labels)
# perfROC = performance(predROC, "tpr", "fpr")
# plot(perfROC, colorize=T)

# xgboost

# require(xgboost)

# require(caTools)
# 
# split = sample.split(intDataImputed$X1, SplitRatio = 0.8)
# 
# train = intDataImputed[split,]
# train = list(data = as.matrix(train[,2:14]), label = train$X1)
# test = intDataImputed[!split,]
# test = list(data = as.matrix(test[,2:14]), label = test$X1)
# 
# dtrain = xgb.DMatrix(data = train$data, label = train$label)
# dtest = xgb.DMatrix(data = test$data, label = test$label)

# rm(list = list(intData, intDataImputed, intDataImputedScaled))

# xgModel = xgboost(data = dtrain, nrounds = 100, objective = "binary:logistic", eval_metric="logloss")
# 
# pred = predict(xgModel, train$data)
# labels = train$label
# 
# require(ROCR)

# predROC = prediction(pred, labels)
# perfROC = performance(predROC, "tpr", "fpr")
# performance(predROC, "auc")@"y.values"[[1]]
# plot(perfROC, colorize=T)

# predTest = predict(xgModel, test$data)
# labelTest = test$label
# 
# predROCTest = prediction(predTest, labelTest)
# perfROCTest = performance(predROC, "tpr", "fpr")
# performance(predROCTest, "auc")@"y.values"[[1]]
# plot(perfROCTest, colorize=T)
# 
# MultiLogLoss(act = labelTest, predTest)

# Using k-fold cross validation to find best parameters with xgboost

# cvList=list()
# i = 1
# 
# for(md in seq(4,10,2)) {
#     for(g in seq(0.5, 3.5, 1.5)) {
#         for(mw in c(0.5, 1, 5)) {
#             for(nround in c(50,200,500)) {
#                 
#                 writeLines(paste("\nStep",i,"\n"))
#                 eta = -6.75e-4*nround + 0.3675
#                 
#                 xgbParams = list(max.depth=md, eta=eta, gamma = g, min_child_weight = mw, 
#                                  max_delta_step = 0, subsample = 1, colsample_bytree = 1, 
#                                  silent=1)
#                 
#                 cvRes = xgb.cv(xgbParams, dtrain, nround, nfold = 10, 
#                               objective = 'binary:logistic', eval_metric = 'logloss',
#                               maximize = FALSE, early.stop.round = 5, showsd = F, verbose = F)
#                 
#                 cvList[[i]] = list(max.depth=md, eta=eta, nrounds=nround, gamma=g, 
#                                    min_child_weight=mw, test.logloss = min(cvRes$test.logloss.mean),
#                                    bestIteration = which.min(cvRes$test.logloss.mean))
#                 
#                 writeLines("\n")
#                 print(as.data.frame(cvList[[i]]))
#                 i = i+1
#             }
#         }
#     }
# }
# 
# cvDF = as.data.frame(do.call(rbind, cvList))
# for(i in 1:7) {cvDF[,i] = as.numeric(cvDF[,i])}
# 
# ggplot(data = cvDF, aes(x=test.logloss)) + geom_histogram() + facet_grid(max.depth~.)

# xgb.cv(params = list(max.depth=6, eta=0.03, gamma = 3.5, min_child_weight = 5,
#                      max_delta_step = 0, subsample = 1, colsample_bytree = 1, silent=1), 
#        dtrain, 500, nfold = 10, objective = 'binary:logistic', eval_metric = 'logloss',
#        maximize = FALSE, early.stop.round = 5, showsd = F, verbose = F)
# 0.470697 - best

# also tested depth 4, 6, 8, with nrounds = 500, 1000, 2000

# for now, best model = max.depth=6, eta=0.03, gamma = 3.5, min_child_weight = 5,
#                      max_delta_step = 0, subsample = 1, colsample_bytree = 1, nround = 500


#
# Here I want to test keeping missing values, and letting xgboost dealing with it
# I simply set missing values to -666, and pass it to the missing flag

# intDataNA = intData
# 
# intDataNA$X2[is.na(intDataNA$X2)] = rep(-666, length(intDataNA$X2[is.na(intDataNA$X2)]))
# intDataNA$X3[is.na(intDataNA$X3)] = rep(-666, length(intDataNA$X3[is.na(intDataNA$X3)]))
# intDataNA$X4[is.na(intDataNA$X4)] = rep(-666, length(intDataNA$X4[is.na(intDataNA$X4)]))
# intDataNA$X5[is.na(intDataNA$X5)] = rep(-666, length(intDataNA$X5[is.na(intDataNA$X5)]))
# intDataNA$X6[is.na(intDataNA$X6)] = rep(-666, length(intDataNA$X6[is.na(intDataNA$X6)]))
# intDataNA$X7[is.na(intDataNA$X7)] = rep(-666, length(intDataNA$X7[is.na(intDataNA$X7)]))
# intDataNA$X8[is.na(intDataNA$X8)] = rep(-666, length(intDataNA$X8[is.na(intDataNA$X8)]))
# intDataNA$X9[is.na(intDataNA$X9)] = rep(-666, length(intDataNA$X9[is.na(intDataNA$X9)]))
# intDataNA$X10[is.na(intDataNA$X10)] = rep(-666, length(intDataNA$X10[is.na(intDataNA$X10)]))
# intDataNA$X11[is.na(intDataNA$X11)] = rep(-666, length(intDataNA$X11[is.na(intDataNA$X11)]))
# intDataNA$X12[is.na(intDataNA$X12)] = rep(-666, length(intDataNA$X12[is.na(intDataNA$X12)]))
# intDataNA$X13[is.na(intDataNA$X13)] = rep(-666, length(intDataNA$X13[is.na(intDataNA$X13)]))
# intDataNA$X14[is.na(intDataNA$X14)] = rep(-666, length(intDataNA$X14[is.na(intDataNA$X14)]))
# 
# intDataNA = intDataNA[sample(1:nrow(intDataNA), replace = FALSE),]
# 
# splitNA = sample.split(intDataNA$X1, SplitRatio = 0.8)
# 
# trainNA = intDataNA[splitNA,]
# trainNA = list(data = as.matrix(trainNA[,2:14]), label = trainNA$X1)
# testNA = intDataNA[!splitNA,]
# testNA = list(data = as.matrix(testNA[,2:14]), label = testNA$X1)
# 
# dtrainNA = xgb.DMatrix(data = trainNA$data, label = trainNA$label)
# dtestNA = xgb.DMatrix(data = testNA$data, label = testNA$label)
# 
# cvList=list()
# i = 1
# 
# for(md in seq(4,8,2)) {
#     for(g in seq(2.5, 3.5, 0.5)) {
#         for(mw in c(1, 5, 10)) {
#             for(nround in c(200,500)) {
#                 
#                 writeLines(paste("\nStep",i,"\n"))
#                 eta = -6.75e-4*nround + 0.3675
#                 
#                 xgbParams = list(max.depth=md, eta=eta, gamma = g, min_child_weight = mw, 
#                                  max_delta_step = 0, subsample = 1, colsample_bytree = 1, 
#                                  silent=1)
#                 
#                 cvRes = xgb.cv(xgbParams, dtrainNA, nround, nfold = 10, 
#                               objective = 'binary:logistic', eval_metric = 'logloss',
#                               maximize = FALSE, early.stop.round = 5, showsd = F, verbose = F, missing = -666)
#                 
#                 cvList[[i]] = list(max.depth=md, eta=eta, nrounds=nround, gamma=g, 
#                                    min_child_weight=mw, test.logloss = min(cvRes$test.logloss.mean),
#                                    bestIteration = which.min(cvRes$test.logloss.mean))
#                 
#                 writeLines("\n")
#                 print(as.data.frame(cvList[[i]]))
#                 i = i+1
#             }
#         }
#     }
# }
# 
# cvDFNA = as.data.frame(do.call(rbind, cvList))
# for(i in 1:7) {cvDFNA[,i] = as.numeric(cvDFNA[,i])}
# # in the end, looks like min_child_weight doesn't matter that much. However, we get improvement on gamma = 3 instead of 3.5.
# # so, i'm going to run some more test to find a best gamma value. nrounds still seems better when large, max.depth 6, 
# 
# cvList=list()
# i = 1
# 
# for(md in 5:7) {
#     for(g in seq(2.5, 3.5, 0.25)) {
#         for(mw in c(1)) {
#             for(nround in c(200,500)) {
#                 for(eta in c(0.02, 0.03)) {
#                     writeLines(paste("\nStep",i,"\n"))
# 
#                     xgbParams = list(max.depth=md, eta=eta, gamma = g, min_child_weight = mw, 
#                                      max_delta_step = 0, subsample = 1, colsample_bytree = 1, 
#                                      silent=1)
#                     
#                     cvRes = xgb.cv(xgbParams, dtrainNA, nround, nfold = 10, 
#                                    objective = 'binary:logistic', eval_metric = 'logloss',
#                                    maximize = FALSE, early.stop.round = 5, showsd = F, verbose = F, missing = -666)
#                     
#                     cvList[[i]] = list(max.depth=md, eta=eta, nrounds=nround, gamma=g, 
#                                        min_child_weight=mw, test.logloss = min(cvRes$test.logloss.mean),
#                                        bestIteration = which.min(cvRes$test.logloss.mean))
#                     
#                     writeLines("\n")
#                     print(as.data.frame(cvList[[i]]))
#                     i = i+1
#                 }
#                 
#             }
#         }
#     }
# }
# 
# cvDFNA2 = as.data.frame(do.call(rbind, cvList))
# for(i in 1:7) {cvDFNA2[,i] = as.numeric(cvDFNA2[,i])}
# 
# # max.depth  eta nrounds gamma min_child_weight test.logloss bestIteration
# #        6  0.03     500  3.00                1     0.469905           231
# # can now train on all train set.
# 
# xgbParams = list(max.depth=6, eta=0.03, gamma = 3, min_child_weight = 1, 
#                  max_delta_step = 0, subsample = 0.8, colsample_bytree = 0.8, 
#                  silent=1)
# 
# xgModel = xgboost(params = xgbParams, data = dtrain, nrounds = 250, 
#                   objective = "binary:logistic", eval_metric="logloss", verbose = F)
# 
# predTest = predict(xgModel, test$data)
# labelTest = test$label
# 
# MultiLogLoss(act = cbind(labelTest, 1-labelTest), cbind(predTest, 1-predTest)) # 0.944 :-(

# in the following, we need to introduce the categorical features

rm(list = ls()[ls()!="rawDataTrain"])

# rawData[1] is label, rawData[2:14] are integer features, rawData[15:] are categorical features.
# The categorical features need to be converted from hexa to decimal integers, then we must one hot encode them.
# This is going to make the number of features explode

prepDataTrain = rawDataTrain

for(i in 2:14) {
  prepDataTrain[,i][is.na(prepDataTrain[,i])] = rep(-666, length(prepDataTrain[,i][is.na(prepDataTrain[,i])]))
}

prepDataTest = rawDataTest

for(i in 2:14) {
  prepDataTest[,i][is.na(prepDataTest[,i])] = rep(-666, length(prepDataTest[,i][is.na(prepDataTest[,i])]))
}

# this is how we are going to prepare the categorical features

# require(int64)
# 
# testCatFeat = prepData$X15
# testCatFeat = as.character(lapply(testCatFeat, FUN = function(x){paste("0x",x,sep="")}))
# testCatFeat = as.numeric(testCatFeat)
# testCatFeat = as.factor(testCatFeat)
# oneHottestCatFeat = model.matrix(~ testCatFeat) # becomes 100000 x 541 binary variables.

# To prepare the whole dataset to use with model.matrix, we need to turn each categorical feature from character to numeric and then to factor.
# The problem is that some observations have missing data: for categorical data, this is the empty string "". We need a way to deal with this.
# The way I see it now: For each feature, find the rows with empty string, then, change features to numeric. The empty strings should be now NA
# Then, use model.matrix to One-Hot encode. Then, we should have a NA column for each features that had missing values (e.g. X40NA). This has 
# to be tested. Then either we use the indices for rows we found before, and put all features corresponding to the parent categorical feature
# that had empty string to -666.
#. The features that have missing category are: 17, 18, 20, 26, 30, 33, 34, 35, 36 (81%), 38, 39, 40.

missingDataTrainX17 = prepDataTrain$X17==""
missingDataTrainX18 = prepDataTrain$X18==""
missingDataTrainX20 = prepDataTrain$X20==""
missingDataTrainX26 = prepDataTrain$X26==""
missingDataTrainX30 = prepDataTrain$X30==""
missingDataTrainX33 = prepDataTrain$X33==""
missingDataTrainX34 = prepDataTrain$X34==""
missingDataTrainX35 = prepDataTrain$X35==""
missingDataTrainX36 = prepDataTrain$X36==""
missingDataTrainX38 = prepDataTrain$X38==""
missingDataTrainX39 = prepDataTrain$X39==""
missingDataTrainX40 = prepDataTrain$X40==""

for(i in 15:40){
    testCatFeat = prepDataTrain[,i]
    testCatFeat = as.character(sapply(testCatFeat, FUN = function(x){paste("0x",x,sep="")}))
    testCatFeat = as.numeric(testCatFeat)
    testCatFeat = as.factor(testCatFeat)
    prepDataTrain[,i] = testCatFeat
    print(i)
}
rm(list = c("testCatFeat", "i"))

# testData = prepData[1:1000,]
# outData = model.matrix( X1~ .-1, data = testData[,1:15])

####
#### J'en ?tais ? preparer les categorical features. A ce stade, elles sont toutes converties en num?rique. Il ne reste plus qu'?
#### les mettre en factor, mais il y a des NA. Et il faut trouver comment g?rer les NA quand on one-hot encode.
#### Il y a un probl?me avec la taille des features. Dans le cas de X15, pour 1000 rows, il y a 541 features, ce qui cr?e une matrice de ~4MB
#### Mais, par exemple, pour X17, il y a 43869 diff?rentes valeurs, ce qui cr?e quelques choses de beaucoup trop gros, et model.matrix plante.
#### Il faut donc r?gler ce probl?me en premier: comment utiliser model.matrix, malgr? la possible taille de l'output.
#### Il semble que le probl?me vienne de la quantit? de m?moire utilis?e durant le calcul, parce que l'output devrait ?tre suffisamment "petit"
#### Je m'appr?tais ? essayer sparse.model.matrix
####
#### 18/09/2015:
#### Essay? model.matrix sur AWS (r3.xlarge), mais pour tout le sample, ?a marche pas non plus, m?me avec 30 Go de m?moire
####
#### 19/09/2015
#### Essai  de sparse.model.matrix

require(Matrix)

# outData = sparse.model.matrix(~ .-1, data = testData[,2:40]) # works for that

#### Essai avec tout le sample

# features = sparse.model.matrix(X1~ .-1, data = prepData[,1:40]) # ?a marche aussi

#### Ca a l'air de marcher de cette mani?re.
#### Malheureusement, ?a enl?ve les observations o? il y a des NA.
#### Moi ce que je veux, c'est que si, pour une observation, j'ai un NA dans une categorical feature, alors dans ma matrix finale, 
#### j'ai l'observation, mais avec toutes les colonnes correspondant aux diff?rent level de la feature mis ? -666
#### Ou alors, je remplace simplement les missing values par le mode de la feature en question... Dans ce cas on introduit un biais.
#### Je vais d'abbord tester avec sparse.model.matrix, sans me soucier des NA, et trainer avec xgboost pour voir ce que ?a donne.
#### Mais ce n'est pas possible de ne pas se soucier des NA, parce que ?a nous fait perdre ~ 90% des observations.
#### Donc, je fais avec le mode imputation.

for(i in 15:40) {
    
    catFeat = prepDataTrain[,i]
    topLevel = names(which.max(table(catFeat)))
    prepDataTrain[,i][is.na(prepDataTrain[,i])] = rep(topLevel, length(prepDataTrain[,i][is.na(prepDataTrain[,i])]))
    print(i)
}
rm(list = c("i", "catFeat", "topLevel"))

require(xgboost)
require(caTools)

split = sample.split(prepDataTrain$X1, SplitRatio = 0.8)

train = prepDataTrain[split,]
trainFeats = sparse.model.matrix(~ .-1, data = train[,2:40])
train = list(data = trainFeats, label = train$X1)
valid = prepDataTrain[!split,]
validFeats = sparse.model.matrix(~ .-1, data = valid[,2:40])
valid = list(data = validFeats, label = valid$X1)

dtrain = xgb.DMatrix(data = train$data, label = train$label)
dvalid = xgb.DMatrix(data = valid$data, label = valid$label)

rm(list = c("train", "trainFeats", "validFeats", "split"))
# # max.depth  eta nrounds gamma min_child_weight test.logloss bestIteration
# #        6  0.03     500  3.00                1     0.469905           231

xgbParams = list(max.depth=6, eta=0.03, gamma = 3, min_child_weight = 1, 
                                  max_delta_step = 0, subsample = 0.8, colsample_bytree = 0.8, silent=1)

xgModel = xgb.train(params = xgbParams, data = dtrain, watchlist = list(train=dtrain, valid=dvalid), 
                    nrounds = 50, objective = "binary:logistic", eval_metric="logloss", verbose = 1)

predValid = predict(xgModel, dvalid)
labelValid = valid$label

MultiLogLoss(act = cbind(labelValid, 1-labelValid), cbind(predValid, 1-predValid))

ptrain = predict(xgModel, dtrain, outputmargin=TRUE)
pvalid = predict(xgModel, dvalid, outputmargin=TRUE)
setinfo(dtrain, "base_margin", ptrain)
setinfo(dvalid, "base_margin", pvalid)
xgModel = xgb.train(params = xgbParams, data = dtrain, watchlist = list(train=dtrain, valid=dvalid), 
                    nrounds = 250, objective = "binary:logistic", eval_metric="logloss", verbose = 1)

#### Tout fonctionne. xgb.train me donne un logloss de 0.44951 sur le test set, mais celui de Kaggle 0.8990805
#### 2. Essayer de voir si on peut train tout le vrai dataset sur AWS, et combien ça me donne en score // il faut utiliser une machine de 60Go de RAM
####  02/10/2015 !!!!!! 
####  On n'arrive pas a train le vrai train set sur AWS. Ou alors il faut une machine avec 120 ou 240 Go de RAM.
####   Avec 1000000 d'obs, on a toujours un MultiLogLoss pas terrible (et toujours 2x celui de xgboost ???)
####   La prochaine etape, c'est donc de voir comment gerer les missing features dans le test set, et de faire une submission
####   sur Kaggle, afin d'avoir une meilleure idée de la performance.
####     - Puis, en utilisant le vrai test set et en essayant de faire une submission sur Kaggle
####          - Pour ca, il  faut voir comment gerer les levels absents du train set
####            Premier test, ne simplement rien faire, et voir si xgboost fonctionne
####            Pas sur de pouvoir traiter sur mon portable. Parce qu'il n'y aura pas assez de m�moire pour faire le sparse matrix step du test set

#### Et ensuite, avec le hashing trick ?