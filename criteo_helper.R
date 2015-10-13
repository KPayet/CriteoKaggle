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

require(xgboost)
# dvalid should: dvalid = xgb.DMatrix(data = valid$data, label = valid$label)
#
grid.Search <- function(dvalid, .md, .gamma, .minChildWeight, .nround, .subsample = c(1.), .colsample = c(1.), .maxDelta = 0, .posWeight = 1.) {
  
    cvList=list()
    i = 1
    
    for(md in .md) {
        for(g in .gamma) {
            for(mw in .minChildWeight) {
                for(nround in .nround) {
                    for(subsample in .subsample) {
                        for(colsample in .colsample) {
                            writeLines(paste("\nStep",i,"\n"))
                            eta = -6.75e-4*nround + 0.3675 # we consider eta as a function of nrounds, because usually, as nrounds grows, we have to reduce eta
                            
                            xgbParams = list(max.depth=md, eta=eta, gamma = g, min_child_weight = mw, 
                                             max_delta_step = .maxDelta, subsample = subsample, colsample_bytree = colsample, 
                                             silent=1, scale_pos_weight = .posWeight)
                            
                            cvRes = xgb.cv(xgbParams, dvalid, nround, nfold = 10, 
                                          objective = 'binary:logistic', eval_metric = 'logloss',
                                          maximize = FALSE, early.stop.round = 5, showsd = F, verbose = F, missing = -666)
                            
                            cvList[[i]] = list(max.depth=md, eta=eta, nrounds=nround, gamma=g, 
                                               min_child_weight=mw, subsample = subsample, colsample_bytree = colsample,
                                               test.logloss = min(cvRes$test.logloss.mean),
                                               bestIteration = which.min(cvRes$test.logloss.mean))
                            
                            writeLines("\n")
                            print(as.data.frame(cvList[[i]]))
                            i = i+1
                        }
                    }
                }
            }
        }
    }
  
}

