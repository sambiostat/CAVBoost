setwd("/SFS/scratch/zhanjinc/ANCOVABoost/code/")
 
source("Competitor.R")
source("AncovaBoost_Diff.R")
source("AncovaBoost_SD.R")
source("ContBoost.R")
source('ContBoost_IPW_plugin.R')
source('ContBoost_OW_plugin.R')

source('GenModel.R')
 

sim_main <- function(scenario=0, N=5100, M=100){
  
  data <- MySimData(s=scenario, N=N, M=M)
  
  dat.train <- data[[1]]
  dat.test <- data[[2]]
  otr.test <- data[[4]]
  PredVar <- data[[5]]
  
  Ancova_N_res =tryCatch ( MyBoost_main_N(dat.train, otr.test=otr.test, dat.test, PredVar )  ,
                                error = function(e) {list(NA, NA, NA, NA, NA)}
  )
  
  Ancova_SD_res =tryCatch ( MyBoost_main_sd(dat.train, otr.test=otr.test, dat.test, PredVar)  ,
                            error = function(e) {list(NA, NA, NA, NA, NA)}
  )
  
  
  Cont_Boost_res =tryCatch ( MyBoost_cont(dat.train, otr.test=otr.test, dat.test, PredVar)  ,
                             error = function(e) {list(NA, NA, NA, NA, NA)}
  )
  
  Cont_BoostIPW_res =tryCatch ( MyBoost_cont_IPWplugin(dat.train, otr.test=otr.test, dat.test, PredVar)  ,
                             error = function(e) { list(NA, NA, NA, NA, NA)}
  )
  
  Cont_BoostOW_res =tryCatch ( MyBoost_cont_OWplugin(dat.train, otr.test=otr.test, dat.test, PredVar)  ,
                                error = function(e) { list(NA, NA, NA, NA, NA)}
  )
   
  ##for other competitor
  sgm_list= c(0.1, 1, 10, 100)
  c_list= 4^(-1:2)
  m = 5
  X <- dat.train %>% select(-trt01p, -Y)
  Xnames = colnames(X)
  X = as.matrix(X)
  
  ##Convert the Tr to -1/+1
  Tr <- dat.train$trt01p*2-1
  Y <- dat.train$Y
  RY = glm(Y ~  X)
  YR =  RY$residuals
  
  eval = function( predITR, itr){
    itr= itr*2 - 1
    idx = {predITR== itr}
    acc = mean( idx )
    sen =  mean( idx[otr.test==1], na.rm=T )  
    spec =  mean( idx[otr.test!=1], na.rm=T )
    return( list(acc, sen, spec,predITR))
    }
  
  ##rwl under rbf kernel
  X.test <- as.matrix(  dat.test %>% select(-trt01p, -Y) )
  
  rwl.fit1 <- tryCatch(ROWL(X, Tr, YR, pi=rep(0.5, nrow(dat.train)  ), kernel ="rbf", clinear=c_list, sigma=sgm_list,m=m, residual=F),
                       error=function(e) {e} )
  
  if ("error" %in% class(rwl.fit1)) {
    rwl_rbf_res = list(NA, NA, NA, NA, NA)
    print(rwl.fit1)
  }else{
    optTr.rwl1 <-  predict(rwl.fit1, X.test)
    rwl_rbf_res <- tryCatch(   eval(optTr.rwl1, otr.test),
                            error=function(e) {list(NA, NA, NA, NA, NA)} )
  }
  
   
  
  ## classic qlearning
  qfit1 <- tryCatch (Qlearning (H = as.matrix(X), A= (Tr), R= (Y),m = m), 
                     error=function(e) {e})
  
  if ("error" %in% class(qfit1)){
    qfit1_res = list(NA, NA, NA,  NA)
    qRANK = rep(NA, length(PredVar))
  }else{
    print("qfit")
    qPred <- predict(qfit1, X.test)
    optTr.qfit1 <- qPred[["opt_trt"]]
    qfit1_res =tryCatch( eval(optTr.qfit1, otr.test),
                         error=function(e) {list(NA, NA, NA, NA, NA)} )
    if (length (qfit1_res) ==4) {
      qCoef <- abs( qPred[["coef.est" ]] )
      ##qCoef corresponds to cbind(1,X, A, A*X); we only need later half
      qCoef <- qCoef[(length(qCoef) /2 +2): length(qCoef) ]
      XnamesNotSelected <- Xnames[qCoef ==0]
      XnamesSelected <-  Xnames[qCoef != 0]
      
      qImportance <- order(qCoef[qCoef != 0], decreasing=T)
      
      qRANK = sapply(PredVar, function(j){
        if (j %in% XnamesNotSelected){
          return (length( XnamesSelected )+ length(XnamesSelected )/2)
        }else{
          qImportance[ Xnames ==j ]
        }
      } )
      qRANK = unlist(qRANK)[PredVar]
      
    } else{
      qRANK = rep(NA, length(PredVar))
    }
  } 
  print(qRANK )
  
  dat.train$Y = YR
  Res_Boost_res = tryCatch( MyBoost_cont(dat.train, otr.test, dat.test, PredVar  ),
                            error = function(e) {list(NA, NA, NA, NA, NA)}
  )
  print(Res_Boost_res[1:3])
  
  
  
  VarRank =tryCatch( cbind.data.frame( Ancova_N_res[[5]], Ancova_SD_res[[5]], Ancova_Var_res[[5]], Cont_Boost_res[[5]], Cont_BoostIPW_res[[5]],Cont_BoostOW_res[[5]],
                              Res_Boost_res[[5]],qRANK, rep(NA, length(PredVar)), rep(NA, length(PredVar)) ), 
                     error=function(e) {e})
  if ("error" %in% class(VarRank)){
    print(VarRank )
    VarRank= tryCatch( cbind.data.frame( Ancova_N_res[[5]], Ancova_SD_res[[5]], Ancova_Var_res[[5]], Cont_Boost_res[[5]], Cont_BoostIPW_res[[5]],Cont_BoostOW_res[[5]],
                               Res_Boost_res[[5]], rep(NA, length(PredVar)), rep(NA, length(PredVar)), rep(NA, length(PredVar)) ), 
                       error=function(e) {  NA  })
  }   
  

  out = list( accuracy= c( Ancova_N_res[[1]], Ancova_SD_res[[1]],   Cont_Boost_res[[1]], Cont_BoostIPW_res[[1]], Cont_BoostOW_res[[1]],
                           Res_Boost_res[[1]], qfit1_res[[1]],   rwl_rbf_res[[1]] ),
        
        sensitivity = c( Ancova_N_res[[2]], Ancova_SD_res[[2]],   Cont_Boost_res[[2]], Cont_BoostIPW_res[[2]],Cont_BoostOW_res[[2]], 
                         Res_Boost_res[[2]],  qfit1_res[[2]],   rwl_rbf_res[[2]] ),
        
        Specificity = cbind.data.frame( Ancova_N_res[[3]], Ancova_SD_res[[3]],  Cont_Boost_res[[3]], Cont_BoostIPW_res[[3]],Cont_BoostOW_res[[3]],
                                 Res_Boost_res[[3]],qfit1_res[[3]],  rwl_rbf_res[[3]] ) , 
      
        VarRank = VarRank
        
  )
  
  return(out)
  
}
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  