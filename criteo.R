source("criteo_helper.R")

require(readr)

train = read_delim("train.txt", delim="\t", col_names = F)
train = train[sample(1:nrow(train), nrow(train), replace=F),] # shuffle train set

test = read_delim("test.txt", delim="\t", col_names = F)

# I split the training set into train and validation set right from the beginning, so that the valid set is treated like the test set
# I don't want any information leakage
require(caTools)
split = sample.split(train$X1, SplitRatio = 0.97) # probably makes shuffling redundant

valid = train[!split,]
train = train[split,]

# scale the numeric features in the three datasets
require(caret)
preproc = preProcess(train[,2:14])
train[,2:14] = predict(preproc, train[,2:14])

valid[,2:14] = predict(preproc, valid[,2:14])

names(test) = names(train)[2:40]
test[,1:13] = predict(preproc, test[,1:13])

# make all NA in numeric features -666, to use the missing = -666 flag in xgboost

for(i in 2:14) {
  train[,i][is.na(train[,i])] = -666
  valid[,i][is.na(valid[,i])] = -666
}

for(i in 1:13) {
  test[,i][is.na(test[,i])] = -666
}

# prepare categorical features
# the test set has levels that don't appear in the train set, that must be dealt with
# We simply consider that unseen feature levels that appear in the test set are missing values
# The list below is declared to save all levels that appear in the train set
trainLevelsForFeat = list()

# train set
for(i in 15:40){
    testCatFeat = train[,i]
    # testCatFeat = as.character(sapply(testCatFeat, FUN = function(x){paste("0x",x,sep="")}))
    # testCatFeat = as.numeric(testCatFeat)
    # testCatFeat[testCatFeat = ""] = -666
    # testCatFeat = as.factor(testCatFeat)
	if(!( "" %in% levels(testCatFeat) )) { levels(testCatFeat) = c(levels(testCatFeat), "") }
    trainLevelsForFeat[[i]] = levels(testCatFeat) # record which levels are in the train set
    # train[,i] = testCatFeat
    print(i)
}
rm(list = c("testCatFeat", "i"))

for(i in 15:40){
  testCatFeat = valid[,i]
  # testCatFeat = as.character(sapply(testCatFeat, FUN = function(x){paste("0x",x,sep="")}))
  # testCatFeat = as.numeric(testCatFeat)
  # testCatFeat[testCatFeat = ""] = -666
  if(!( "" %in% levels(testCatFeat) )) { levels(testCatFeat) = c(levels(testCatFeat), "") }
  testCatFeat[ !( testCatFeat %in% trainLevelsForFeat[[i]] ) ] = "" # set levels that were not in the train set to missing
  testCatFeat = as.factor(testCatFeat)
  valid[, i] = testCatFeat
  print(i)
}
rm(list = c("testCatFeat", "i"))

for(i in 14:39){
  testCatFeat = test[,i]
  # testCatFeat = as.character(sapply(testCatFeat, FUN = function(x){paste("0x",x,sep="")}))
  # testCatFeat = as.numeric(testCatFeat)
  # testCatFeat[testCatFeat = ""] = -666
  if(!( "" %in% levels(testCatFeat) )) { levels(testCatFeat) = c(levels(testCatFeat), "") }
  testCatFeat[ !( testCatFeat %in% trainLevelsForFeat[[i + 1]] ) ] = "" # set levels that were not in the train set to missing
  testCatFeat = as.factor(testCatFeat)
  test[, i] = testCatFeat
  print(i)
}
rm(list = c("testCatFeat", "i"))

# one hot encoding method, using sparse.model.matrix
# I used it before moving to the hashing trick
# Simply kept here for reference

# require(Matrix)
# require(xgboost)
# require(caTools)

# split = sample.split(train$X1, SplitRatio = 0.8)

# train = train[split,]
# trainFeats = sparse.model.matrix(~ .-1, data = train[,2:40])
# train = list(data = trainFeats, label = train$X1)
# valid = train[!split,]
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

# testFeats = sparse.model.matrix(~., data = test)
# dtest = xgb.DMatrix(data = testFeats)
# predTest = predict(xgModel, dtest)

# predDF = data.frame(ID=1:length(predTest)+59999999, Predicted=predTest)
# write.csv(predDF, "predictions_1.csv", row.names=F)



# Model data using the Hashing "trick" instead of sparse.model.matrix

require(FeatureHashing)
require(xgboost)

# split = sample.split(train$X1, SplitRatio = 0.97) # this is done at the beginning in the new version

# train = train[split,]
trainHashFeats = hashed.model.matrix(~ .-1, data = train[,2:40], hash.size = 2^16, transpose = F)
train = list(data = trainHashFeats, label = train$X1)
# valid = train[!split,]
validHashFeats = hashed.model.matrix(~ .-1, data = valid[,2:40], hash.size = 2^16, transpose = F)
valid = list(data = validHashFeats, label = valid$X1)

rm(list = c("trainHashFeats", "validHashFeats"))

# saveRDS(train, "trainListHashed2048.rds")
# saveRDS(valid, "validListHashed2048.rds")
train = readRDS("trainListHashed2048_2.rds")
valid = readRDS("validListHashed2048_2.rds")

dtrain = xgb.DMatrix(data = train$data[1:1000000,], label = train$label[1:1000000])
dvalid = xgb.DMatrix(data = valid$data, label = valid$label)

# do cross validation using xgb.cv to find best parameters
# cvResults = grid.Search(dvalid, .md = seq(6, 14, 2), .gamma = seq(1.5, 9.5, 0.5), 
# 								.minChildWeight = c(1, 5, 10), .nround = c(100, 200, 500, 1000),
# 								nFolds = 5)

testFeats = readRDS("testListHashed2048_2.rds")
# testFeats = hashed.model.matrix(~ .-1, data = test, hash.size = 2^16, transpose = F)
# saveRDS(testFeats, "testListHashed2048.rds")
dtest = xgb.DMatrix(data = testFeats)

xgbParams = list(max.depth=12, eta=0.1, gamma = 10, min_child_weight = 5, missing = -666,
                 max_delta_step = 1, subsample = 1, colsample_bytree = 1, silent=1)


# train for one round, simply to initiate the xgboost model
# then, training is done in the loop below
xgModel = xgb.train(params = xgbParams, data = dtrain, watchlist = list(train=dtrain, valid=dvalid), 
                    nrounds = 1, objective = "binary:logistic", eval_metric="logloss", verbose = 1)

alreadyDone = 1

# The loop below simply saves the model every 100 rounds
# This is simply because I trained on AWS using spot request, 
# and didn't want the training to be wasted if the spot was closed 
for(i in c(seq(50, 100, 50))) {
  
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
  write.csv(predDF, paste("predictions_",i,"_hashed65k.csv", sep = ""), row.names=F)
  saveRDS(xgModel, paste("model_",i,"_rounds.xgb", sep=""))
  
  # copy to s3 to save
  writeLines(c(paste("sudo aws s3 cp predictions_",i,"_hashed65k.csv s3://kpayets3/", sep=""),
               paste("sudo aws s3 cp model_",i,"_rounds.xgb s3://kpayets3/", sep=""),
               paste("sudo rm predictions_",i,"_hashed65k.csv", sep=""),
               paste("sudo rm model_",i,"_rounds.xgb", sep="")),
             con = "copyTos3", sep = "\n")
  
  alreadyDone = i
}
