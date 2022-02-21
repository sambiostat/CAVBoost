library(xgboost)
library(MASS)
library(dplyr)

#difference adjusted by sample size
MyBoost_main_N <- function(dat, otr.test, dat.test, PredVar ){
  ### little trick to embed trt01p,aval and evnt into labels for Xgboost input  ###
  labels <- dat.test$trt01p
  dtest <- xgb.DMatrix(as.matrix(subset(dat.test, select=-c(trt01p, Y))), label = labels)
  
  attr(dtest, "dcopy") = as.matrix(subset(dat.test, select=-c(trt01p, Y)))  
  attr(dtest, "Y") = dat.test$Y
  fit = MyBoost_diff_N(dat)
  
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

MyBoost_diff_N <- function(dat ){
  N <- nrow(dat)
  labels <- dat$trt01p
  dtrain <- xgb.DMatrix(as.matrix(subset(dat, select=-c(trt01p, Y))), label = labels)
  
  attr(dtrain, "Xcopy") = as.matrix(subset(dat, select=-c(trt01p, Y)))  
  attr(dtrain, "Y") = dat$Y
  
  #defined gradient  
  Myloss <- function(preds, dtrain) {
    stepsize = 0.001
    #print('loss')
    ## (0) Decode y , trt01p ##
    trt01p <- getinfo(dtrain, "label")
    dcopy = attr(dtrain, "Xcopy")
    Y = attr(dtrain, "Y")
    
    #dcopy = as.matrix(cbind(1,dcopy)) ## add intercept ~Z
    Y = matrix(Y, ncol=1)
    
    n = length(Y)
    pred <- 1/(1+exp(-preds))
    predg <- exp(preds)/(1+exp(preds))^2 
    
    ## (1) Set up gradient and Hessian #
    X = cbind(dcopy,trt01p)
    nX = ncol(X)
    
    if ( sum( pred ) >0){
      #print("sub1")
      #weght and center the matrix
      X1 = as.matrix( scale(  diag(sqrt(pred) ) %*% X ) )
      Y1 = as.matrix( scale( Y * sqrt(pred) ) )
      
      Xsqr1=  t(X1) %*% X1 # (X^TWX), p by p matrix
      Xsqr1_inv = tryCatch( ginv( Xsqr1),     
                            error= function(e) { matrix(0, nrow=nX, ncol=nX)})
      beta1_hat =  Xsqr1_inv %*% t( X1 ) %*%  Y1
      V1 = beta1_hat[nX]*sum( pred ) 
      
      rm(X1, Y1 , Y1_hat,Xsqr1, Xsqr1_inv,beta1_hat )
      
      gV1 =  sapply(1:n, function(j) {
        pred_diff = pred
        pred_diff[j] = pred_diff[j] + stepsize
        pred_diff[pred_diff>1] =1
        pred_diff[pred_diff<0] =0
        
        X_diff <- scale( diag(sqrt(pred_diff) ) %*% X  )
        Y_diff <- scale( Y * sqrt(pred_diff) )
        
        Xsqr1_diff=  t(X_diff) %*% X_diff # (X^TWX), p by p matrix
        Xsqr1_inv_diff = tryCatch( ginv( Xsqr1_diff),     
                                   error= function(e) { matrix(0, nrow=nX, ncol=nX)})
        
        beta1_hat_diff =  Xsqr1_inv_diff %*% t( X_diff ) %*%  Y_diff
        V1_diff = beta1_hat_diff[nX]* sum(pred_diff)

        return( (V1_diff- V1)/ stepsize) 
      }
      )
      
    } else{
      V1 = 0
      gV1 = 0
    }
    
    if (sum( 1-pred ) >0){
      #weght and center the matrix
      pred[pred>1] = 1
      pred[pred<0] = 0
      X2 = scale(  diag(sqrt(1-pred) )%*% X )
      Y2 = scale( Y * sqrt(1-pred) )
      
      Xsqr2=  t(X2) %*% X2 # (X^TWX), p by p matrix
      Xsqr2_inv = tryCatch( ginv( Xsqr2),     
                            error= function(e) { matrix(0, nrow=nX, ncol=nX)})
      
      beta2_hat =  Xsqr2_inv %*% t( X2 ) %*%  Y2
      
      V2 = beta2_hat[nX]*sum(1 - pred)
      
      rm(X2, Xsqr2, Xsqr2_inv , beta2_hat)
      
      gV2 =  sapply(1:n, function(j) {
        pred_diff = pred
        pred_diff[j] = pred_diff[j] + stepsize #pred_diff[j] - stepsize
        pred_diff[pred_diff > 1] =1
        pred_diff[pred_diff < 0] =0
        
        X2_diff <- scale( diag(sqrt(1-pred_diff) ) %*% X  )
        Y2_diff <- scale( Y * sqrt(1-pred_diff) )
        
        Xsqr2_diff=  t(X2_diff) %*% X2_diff # (X^TWX), p by p matrix
        Xsqr2_inv_diff = tryCatch( ginv( Xsqr2_diff),     
                                   error= function(e) { matrix(0, nrow=nX, ncol=nX)})
        
        beta2_hat_diff =  Xsqr2_inv_diff %*% t( X2_diff ) %*%  Y2_diff
         
        V2_diff =  beta2_hat_diff[nX]*sum(1 - pred_diff)
        return( (V2_diff- V2)/ stepsize) 
      }
      )
    }else{
      V2 = 0
      gV2 = 0
    }
    
    err <-   -(V1 - V2)
    g.p <-   -(gV1  - gV2) 
    
    g <-   predg*g.p 
    h <- rep(0.00001,n)
    #print(err)
    return(list(grad = g, hess = h))
  }
  
  #defined error function
  evalerror <- function(preds, dtrain) {
    #print("error")
    trt01p <- getinfo(dtrain, "label")
    dcopy = attr(dtrain, "Xcopy")
    Y = attr(dtrain, "Y")
    
    #dcopy = as.matrix(cbind(1,dcopy))
    Y = matrix(Y, ncol=1)
    A = matrix(trt01p, ncol=1)
    n = length(Y)
    pred <- 1/(1+exp(-preds))
    predg <- exp(preds)/(1+exp(preds))^2 
    
    ## (2) Set up error
    X = cbind(dcopy, A)
    nX = ncol(X)
    if (sum( pred ) >0){
      #weght and center the matrix
      X1 = scale(   diag(sqrt(pred) )%*% X)
      Y1 = scale( Y * sqrt(pred) )
      
      Xsqr1=  t(X1) %*% X1 # (X^TWX), p by p matrix
      Xsqr1_inv = tryCatch( ginv( Xsqr1),     
                            error= function(e) { matrix(0, nrow=nX, ncol=nX)})
      
      beta1_hat =  Xsqr1_inv %*% t( X1 ) %*%  Y1
      V1 = tryCatch( beta1_hat[nX]*sum(pred), warning= function(w) { 0 } )    
      V1 = ifelse( is.na(V1), 0, V1 ) 
    }else{
      V1 = 0
    }
    
    if (sum( 1-pred ) >0){
      #weght and center the matrix
      X2 = scale(   diag(sqrt(1-pred) )%*% X)
      Y2 = scale( Y * sqrt(1-pred) )
      
      Xsqr2=  t(X2) %*% X2 # (X^TWX), p by p matrix
      Xsqr2_inv = tryCatch( ginv( Xsqr2),     
                            error= function(e) { matrix(0, nrow=nX, ncol=nX)})
      
      beta2_hat =  Xsqr2_inv %*% t( X2 ) %*%  Y2
      V2 = tryCatch( beta2_hat[nX]*sum(1-pred), warning= function(w) { 0 } )    
      V2 = ifelse( is.na(V2), 0, V2 ) 
    }else{
      V2= 0
    }
    
    err <-  -as.vector(V1  - V2)
    if( is.na(err)) err = Inf 
    
    #print(c('err0'=err, 'nP'=sum(pred), 'nQ' = sum(1-pred), "V1"=V1, "V2"=V2 ,
    #        "EYdiff1" = sum(Y[A==1]*pred)/sum(pred) ,"EYdiff1" = sum(Y[A==1]*(1-pred))/sum(1-pred) 
    #        )       )
    return(list(metric = "OTR_error", value = err))
  }
  
  ### Let's boost ###
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
      maximize=F,
      verbose = 0,               # silent,
      early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
    )
    
    # add min training error and trees to grid
    hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_OTR_error_mean)
    hyper_grid$min_error[i] <- min(xgb.tune$evaluation_log$test_OTR_error_mean)
  }
  
  ### Train a model based on the best CV parameter set ###
  hyper_grid <- hyper_grid[order(hyper_grid$min_error),]
  cat(" Top Five Fitting per CV \n")
  print (hyper_grid[1:5,])
  
  param <- list(max_depth = hyper_grid$max_depth[1], eta = hyper_grid$eta[1], silent = 1,objective = Myloss,
                eval_metric = evalerror,verbose = 1,lambda=1,base_score=0,colsample_bytree=1,min_child_weight=0)
  
  watchlist <- list(train = dtrain)
  
  cat("Train Model based on Optimal Parameter Setting from CV \n")
  
  model <- xgb.train(param, dtrain, nrounds = hyper_grid$optimal_trees[1],watchlist)
  
  cat('Model Fitting Finished \n')
  
  return(model)
}
