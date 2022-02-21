library(xgboost)

MyBoost_cont_OWplugin<-function(dat, otr.test, dat.test, PredVar ){
  ### little trick to embed trt01p,aval and evnt into labels for Xgboost input  ###
  labels <- dat.test$trt01p
  dtest <- xgb.DMatrix(as.matrix(subset(dat.test, select=-c(trt01p, Y))), label = labels)
  
  #attach other info to the dtrain data
  attr(dtest, "dcopy") = as.matrix(subset(dat.test, select=-c(trt01p, Y)))  
  attr(dtest, "Y") = dat.test$Y
  
  fit = MyBoost_cont_sub_OW_plugin(dat)
  
  pred <- predict(fit, dtest)
  pred <- 1/(1+exp(-pred))
  idx = (pred >0.5) == otr.test
  acc = mean( idx )
  
  sens = mean( idx[otr.test==1] , na.rm=T)  
  spec = mean(idx[otr.test!=1]  , na.rm=T)
  importance = xgb.importance(fit$feature_names, fit)$Feature
  rank = unlist(sapply(PredVar, function(j)(which(importance==j)) ))
  rank = rank[PredVar]
  
  return( list( acc, sens, spec, pred, rank ))
}

MyBoost_cont_sub_OW_plugin <- function(dat){
  N <- nrow(dat)
  labels <- dat$trt01p
  dtrain <- xgb.DMatrix(as.matrix(subset(dat, select=-c(trt01p, Y))), label = labels)
  
  #attach other info to the dtrain data
  attr(dtrain, "Y") = dat$Y
  
  #defined gradient  
  Myloss <- function(preds, dtrain) {
    ## (0) Decode y , trt01p ##
    trt01p <- getinfo(dtrain, "label")
    Y = attr(dtrain, "Y")
    
    arm.val <- c(1, 0)
    
    ## (1) Get data ready##
    km.dat <- data.frame(trt01p, Y)
    n<-dim(km.dat)[1]
    km.dat$id<-c(1:n)
    km.dat$pred <- 1/(1+exp(-preds))
    km.dat$predg <- exp(preds)/(1+exp(preds))^2  
    
    ## The weigth will be updated if Propensity score is needed.
    km.dat$W1A1 = km.dat$W1A0 =km.dat$W0A1=km.dat$W0A0 = rep(0.5, n)
    
    ## (2) Set up gradient and Hessian ##
    value.diff1 <- 0 #V1(p) - V0(p)
    value.diff2  <- 0 #V1(1-p) - V0(1-p)
    
    sub1 <- subset(km.dat, trt01p==arm.val[1])  #A1
    sub2 <- subset(km.dat, trt01p==arm.val[2])  #A0
    
    
    if (nrow(sub1)==0){
      V1A1 = V2A1 = gV1A1 = gV2A1 = 0
    } else{
      denomV1A1 = sum(sub1$pred*sub1$W1A1)
      denomV2A1 = sum((1-sub1$pred)*sub1$W0A1)
      
      V1A1 = sum(sub1$Y*sub1$pred*sub1$W1A1)/denomV1A1
      V2A1 = sum(sub1$Y*(1-sub1$pred)*sub1$W0A1)/denomV2A1
      V1A1 = ifelse(is.na(V1A1), 0, V1A1)
      V2A1 = ifelse(is.na(V2A1), 0, V2A1)
      
      gV1A1 = (denomV1A1 * ((km.dat$trt01p == arm.val[1])*km.dat$Y*km.dat$W1A1) - 
                 sum(sub1$Y*sub1$W1A1*sub1$pred)*( (km.dat$trt01p ==arm.val[1])*km.dat$W1A1) )/(denomV1A1^2)
      gV2A1 = (denomV2A1 * ((km.dat$trt01p == arm.val[1])*km.dat$Y*km.dat$W0A1) - 
                 sum(sub1$Y*sub1$W0A1*(1-sub1$pred))*((km.dat$trt01p ==arm.val[1])*km.dat$W0A1) )/(-denomV2A1^2)
      
      gV1A1 = ifelse(is.na(gV1A1), 0, gV1A1)
      gV2A1 = ifelse(is.na(gV2A1), 0, gV2A1)
      
    }
    
    if(nrow(sub2)==0){
      V1A0 = V2A0 = gV1A0 = gV2A0 = 0
    } else{
      denomV1A0 = sum(sub2$pred*sub2$W1A0)
      denomV2A0 = sum((1-sub2$pred)*sub2$W0A0)
      
      V1A0 =  sum(sub2$Y*sub2$pred*sub2$W1A0)/denomV1A0
      V2A0 =  sum(sub2$Y*(1-sub2$pred)*sub2$W0A0)/denomV2A0
      V1A0 = ifelse(is.na(V1A0), 0, V1A0)
      V2A0 = ifelse(is.na(V2A0), 0, V2A0)
      
      gV1A0 = (denomV1A0 * ((km.dat$trt01p == arm.val[2])*km.dat$Y*km.dat$W1A0) - 
                 sum(sub2$Y*sub2$W1A0*sub2$pred)*( (km.dat$trt01p ==arm.val[2])*km.dat$W1A0) )/(denomV1A0^2)
      
      gV2A0 = (denomV2A0 * ((km.dat$trt01p == arm.val[2])*km.dat$Y*km.dat$W0A0) - 
                 sum(sub2$Y*sub2$W0A0*(1-sub2$pred))*((km.dat$trt01p ==arm.val[2])*km.dat$W0A0) )/(-denomV2A0^2)
      
      gV1A0 = ifelse(is.na(gV1A0), 0, gV1A0)
      gV2A0 = ifelse(is.na(gV2A0), 0, gV2A0)
      
    }
    
    
    value.diff1 = V1A1 - V1A0
    value.diff2 = V2A1 - V2A0
    
    ## Gradient ##
    value.diff1.g <- gV1A1 - gV1A0 
    value.diff2.g <- gV2A1 - gV2A0
    
    err <- (-1)*( sum(km.dat$pred)*value.diff1 -  sum(1-km.dat$pred)*value.diff2  )
    
    g.p <- (sum(km.dat$pred)*value.diff1.g + value.diff1 - sum(1-km.dat$pred)*value.diff2.g + value.diff2) #JZ-eq(2)*-1, same order
    g <-  km.dat$predg*(-1)*g.p 
    g <- g[order(km.dat$id)]
    h <-rep(0.00001,n)
    
    return(list(grad = g, hess = h))
    
  }
  
  #defined error function
  evalerror <- function(preds, dtrain) {
    ## (0) Decode y, trt01p##
    trt01p <- getinfo(dtrain, "label")
    Y = attr(dtrain, "Y")
    
    arm.val <- c(1,0)
    
    ## (1) Get Time to event Data Ready ##
    km.dat <- data.frame(trt01p, Y)
    n<-dim(km.dat)[1]
    km.dat$id<-c(1:n)
    km.dat$pred <- 1/(1+exp(-preds))
    
    ##add propensity score estimation
    km.dat$W1A1 = km.dat$W1A0 =km.dat$W0A1=km.dat$W0A0 = rep(0.5, n )
    
    n<-dim(km.dat)[1]
    
    ## (2) caluclate value function ##
    value.diff1 <- 0 #V1(p) - V0(p)
    value.diff2  <- 0 #V1(1-p) - V0(1-p)
    
    sub1 <- subset(km.dat, trt01p ==arm.val[1])  
    sub2 <- subset(km.dat, trt01p ==arm.val[2])  
    
    if (nrow(sub1)==0){
      V1A1 = V2A1 =0
    } else{
      V1A1 = sum(sub1$Y*sub1$pred*sub1$W1A1)/sum(sub1$pred*sub1$W1A1)
      V2A1 = sum(sub1$Y*(1-sub1$pred)*sub1$W0A1)/sum((1-sub1$pred)*sub1$W0A1)
      
      V1A1  = ifelse(is.na(V1A1), 0, V1A1)
      V2A1 =   ifelse(is.na(V2A1), 0, V2A1)
    }
    
    if(nrow(sub2)==0){
      V1A0 = V2A0 =0
    } else{
      V1A0 =  sum(sub2$Y*sub2$pred*sub2$W1A0)/sum(sub2$pred*sub2$W1A0)
      V2A0 =  sum(sub2$Y*(1-sub2$pred)*sub2$W0A0)/sum((1-sub2$pred)*sub2$W0A0)
      
      V1A0  = ifelse(is.na(V1A0), 0, V1A0)
      V2A0 =   ifelse(is.na(V2A0), 0, V2A0)
    }
    
    value.diff1 = V1A1 - V1A0
    value.diff2 = V2A1 - V2A0
    
    err <- (-1)*( sum(km.dat$pred)*value.diff1 -  sum(1-km.dat$pred)*value.diff2  )
    
    
    
    # print(paste0("OTRErr-", err))
    return(list(metric = "OTR_error", value = err))
  }
  
  ### Let's boost ###
  
  ##add propensity score estimation
  cvfit = tryCatch( cv.glmnet(x=as.matrix(subset(dat, select=-c(trt01p, Y))), 
                              y=dat$trt01p, family = "binomial",
                              nfolds = 5, gamma=0.5, nlambda=50) ,
                    error=function(e) {e}  
  )
  
  if ("error" %in% class(cvfit)) {
    W = 0.5
  }  else{
    W = predict(cvfit, newx = as.matrix(subset(dat, select=-c(trt01p, Y))), s = "lambda.min", type = "response")[,]
  }
  
  indweight = rep(0,N)
  indweight[labels==1] =   1-W[labels==1]
  indweight[labels==0] =   W[labels==0]
  
  
  # Grid Search #
  hyper_grid <- expand.grid(
    eta = c(0.001,.005, .01, .05, .1),
    max_depth = c(2,4,6),
    #subsample = c(.65, .8, 1),
    #colsample_bytree = c(.8, 1),
    #lambda =c(1,3,5),
    optimal_trees = 0,
    min_error = 0
  )
  
  cat("CV to find the optimal parameter setting \n")
  for(i in 1:nrow(hyper_grid)) {
    print(i)
    
    # create parameter list #
    params <- list(
      eta = hyper_grid$eta[i],
      max_depth = hyper_grid$max_depth[i],
      lambda = 1,
      min_child_weight = 0,
      subsample = 1,
      colsample_bytree = 1
    )
    
    
    # train model
    xgb.tune <- xgb.cv(
      params = params,
      data =  dtrain, #as.matrix(dat),
      #label = labels,
      nrounds = 500,
      nfold = 5,
      objective = Myloss,
      eval_metric = evalerror,
      maximize= F,
      verbose = 0,               # silent,
      early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
    )
    
    # add min training error and trees to grid
    hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_OTR_error_mean)
    hyper_grid$min_error[i] <- min(xgb.tune$evaluation_log$test_OTR_error_mean)
  }
  
  ### Train a model based on the best CV parameter set ###
  hyper_grid <- hyper_grid[order(hyper_grid$min_error),]
  #cat(" Top Five Fitting per CV \n")
  #print (hyper_grid[1:5,])
  
  param <- list(max_depth = hyper_grid$max_depth[1], eta = hyper_grid$eta[1], silent = 1,objective = Myloss,
                eval_metric = evalerror,verbose = 1,lambda=1,base_score=0,colsample_bytree=1,min_child_weight=0,
                weight=indweight)
  
  watchlist <- list(train = dtrain)
  
  cat("Train Model based on Optimal Parameter Setting from CV \n")
  
  model <- xgb.train(param, dtrain, nrounds = hyper_grid$optimal_trees[1],watchlist)
  
  cat('Model Fitting Finished \n')
  
  return(model)
}
