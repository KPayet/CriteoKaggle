source("criteo_helper.R")

require(readr)

prepDataTrain = read_delim("train.txt", delim="\t", col_names = F)
prepDataTrain = prepDataTrain[sample(1:nrow(prepDataTrain), nrow(prepDataTrain), replace=F),]
prepDataTest = read_delim("test.txt", delim="\t", col_names = F)

require(caret)
preproc = preProcess(prepDataTrain[,2:14])
prepDataTrain[,2:14] = predict(preproc, prepDataTrain[,2:14])

names(prepDataTest) = names(prepDataTrain)[2:40]
prepDataTest[,1:13] = predict(preproc, prepDataTest[,1:13])

# make all NA in numeric features -666, to use the missing = -666 flag in xgboost

for(i in 2:14) {
  prepDataTrain[,i][is.na(prepDataTrain[,i])] = rep(-666, length(prepDataTrain[,i][is.na(prepDataTrain[,i])]))
}

for(i in 1:13) {
  prepDataTest[,i][is.na(prepDataTest[,i])] = rep(-666, length(prepDataTest[,i][is.na(prepDataTest[,i])]))
}

# prepare categorical features
# the test set has levels that don't appear in the train set, that must be dealt with
# We simply consider that unseen feature levels that appear in the test set are missing values
# The list below is declared to save all levels that appear in the train set
trainLevelsForFeat = list()

# train set
for(i in 15:40){
    testCatFeat = prepDataTrain[,i]
    testCatFeat = as.character(sapply(testCatFeat, FUN = function(x){paste("0x",x,sep="")}))
    testCatFeat = as.numeric(testCatFeat)
    testCatFeat[is.na(testCatFeat)] = -666
    testCatFeat = as.factor(testCatFeat)
    trainLevelsForFeat[[i]] = levels(testCatFeat) # record which levels are in the train set
    prepDataTrain[,i] = testCatFeat
    print(i)
}
rm(list = c("testCatFeat", "i"))

for(i in 14:39){
  testCatFeat = prepDataTest[,i]
  testCatFeat = as.character(sapply(testCatFeat, FUN = function(x){paste("0x",x,sep="")}))
  testCatFeat = as.numeric(testCatFeat)
  testCatFeat[is.na(testCatFeat)] = -666
  testCatFeat[ !( testCatFeat %in% trainLevelsForFeat[[i + 1]] ) ] = -666 # set levels that were not in the train set to missing
  testCatFeat = as.factor(testCatFeat)
  prepDataTest[,i] = testCatFeat
  print(i)
}
rm(list = c("testCatFeat", "i"))

# one hot encoding method, using sparse.model.matrix
# I used it before moving to the hashing trick
# Simply kept here for reference

# require(Matrix)
# require(xgboost)
# require(caTools)

# split = sample.split(prepDataTrain$X1, SplitRatio = 0.8)

# train = prepDataTrain[split,]
# trainFeats = sparse.model.matrix(~ .-1, data = train[,2:40])
# train = list(data = trainFeats, label = train$X1)
# valid = prepDataTrain[!split,]
# validFeats = sparse.model.matrix(~ .-1, data = valid[,2:40])
# valid = list(data = validFeats, label = valid$X1)

# dtrain = xgb.DMatrix(data = train$data, label = train$label)
# dvalid = xgb.DMatrix(data = valid$data, label = valid$label)

# rm(list = c("train", "trainFeats", "validFeats", "split"))

# xgbParams = list(max.depth=9, eta=0.01, gamma = 3, min_child_weight = 1, 
                 # max_delta_step = 0, subsample = 0.8, colsample_bytree = 0.8, silent=1, scale_pos_weight=1.)

# xgModel = xgb.train(params = xgbParams, data = dtrain, watchlist = list(train=dtrain, valid=dvalid), 
                    # nrounds = 250, objective = "binary:logistic", eval_metric="logloss", verbose = 1)

# # Continued training
# ptrain = predict(xgModel, dtrain, outputmargin=TRUE)
# pvalid = predict(xgModel, dvalid, outputmargin=TRUE)
# setinfo(dtrain, "base_margin", ptrain)
# setinfo(dvalid, "base_margin", pvalid)
# xgModel = xgb.train(params = xgbParams, data = dtrain, watchlist = list(train=dtrain, valid=dvalid), 
                    # nrounds = 100, objective = "binary:logistic", eval_metric="logloss", verbose = 1)

# ## Predictions on the test set, and make submission

# testFeats = sparse.model.matrix(~., data = prepDataTest)
# dtest = xgb.DMatrix(data = testFeats)
# predTest = predict(xgModel, dtest)

# predDF = data.frame(ID=1:length(predTest)+59999999, Predicted=predTest)
# write.csv(predDF, "predictions_1.csv", row.names=F)



# Model data using the Hashing "trick" instead of sparse.model.matrix

require(FeatureHashing)
require(xgboost)
require(caTools)

split = sample.split(prepDataTrain$X1, SplitRatio = 0.97)

train = prepDataTrain[split,]
trainHashFeats = hashed.model.matrix(~ .-1, data = train[,2:40], hash.size = 2^11, transpose = F)
train = list(data = trainHashFeats, label = train$X1)
valid = prepDataTrain[!split,]
validHashFeats = hashed.model.matrix(~ .-1, data = valid[,2:40], hash.size = 2^11, transpose = F)
valid = list(data = validHashFeats, label = valid$X1)

# saveRDS(train, "trainListHashed2048.rds")
# saveRDS(valid, "validListHashed2048.rds")
# train = readRDS("trainListHashed2048.rds")
# valid = readRDS("validListHashed2048.rds")

dtrain = xgb.DMatrix(data = train$data, label = train$label)
dvalid = xgb.DMatrix(data = valid$data, label = valid$label)

# do cross validation using xgb.cv to find best parameters
cvResults = grid.Search(dvalid, .md = seq(4, 8, 2), .gamma = seq(1.5, 3.5, 0.5), 
								.minChildWeight = c(1, 5, 10), .nround = c(100, 200, 500),
								nFolds = 5)

# testFeats = readRDS("testListHashed2048.rds")
testFeats = hashed.model.matrix(~ .-1, data = prepDataTest, hash.size = 2^11, transpose = F)
# saveRDS(testFeats, "testListHashed2048.rds")
dtest = xgb.DMatrix(data = testFeats)

xgbParams = list(max.depth=12, eta=0.1, gamma = 7.5, min_child_weight = 5, missing = -666,
                 max_delta_step = 1, subsample = 1, colsample_bytree = 1, silent=1)


# train for one round, simply to initiate the xgboost model
# then, training is done in the loop below
xgModel = xgb.train(params = xgbParams, data = dtrain, watchlist = list(train=dtrain, valid=dvalid), 
                    nrounds = 1, objective = "binary:logistic", eval_metric="logloss", verbose = 1)

alreadyDone = 1

# The loop below simply saves the model every 100 rounds
# This is simply because I trained on AWS using spot request, 
# and didn't want the training to be wasted if the spot was closed 
for(i in c(seq(100, 1000, 100), 2000)) {
  
  nMoreRounds = i - alreadyDone
  
  # Continued training
  ptrain = predict(xgModel, dtrain, outputmargin=TRUE)
  pvalid = predict(xgModel, dvalid, outputmargin=TRUE)
  setinfo(dtrain, "base_margin", ptrain)
  setinfo(dvalid, "base_margin", pvalid)
  
  xgModel = xgb.train(params = xgbParams, data = dtrain, watchlist = list(train=dtrain, valid=dvalid), 
                      nrounds = nMoreRounds, objective = "binary:logistic", eval_metric="logloss", verbose = 1)
  
  # Predictions on the test set, and make submission file
  predTest = predict(xgModel, dtest)
  predDF = data.frame(Id=1:length(predTest)+59999999, Predicted=predTest) # format required by Kaggle for submission
  write.csv(predDF, paste("predictions_",i,"_hashed2048.csv", sep = ""), row.names=F)
  saveRDS(xgModel, paste("model_",i,"_rounds.xgb", sep=""))
  
  # copy to s3 to save
#   system(command = paste("sudo aws s3 cp predictions_",i,"_hashed1024.csv s3://kpayets3/", sep=""), 
#          intern = F, ignore.stdout = T, ignore.stderr = T)
#   system(command = paste("sudo aws s3 cp model_",i,"_rounds.xgb s3://kpayets3/", sep=""), 
#          intern = F, ignore.stdout = T, ignore.stderr = T)
  writeLines(c(paste("sudo aws s3 cp predictions_",i,"_hashed2048.csv s3://kpayets3/", sep=""),
               paste("sudo aws s3 cp model_",i,"_rounds.xgb s3://kpayets3/", sep=""),
               paste("sudo rm predictions_",i,"_hashed2048.csv", sep=""),
               paste("sudo rm model_",i,"_rounds.xgb", sep="")),
             con = "copyTos3", sep = "\n")
  
  alreadyDone = i
}
