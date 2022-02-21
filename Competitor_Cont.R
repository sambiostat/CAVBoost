library(e1071)
library(kernlab)
library(glmnet)

##added Qlearning, Jan 2021
Qlearning  <- function (H, A, R,  m = 10) {
  gcinfo(F)
  n = length(A)
  X = cbind(H, A, diag(A) %*% H)
  colnames(X) <- c(paste("H",1:dim(H)[2], sep = ""),"A",paste("AH",1:dim(H)[2], sep = ""))
  
  co = coef(lm(R ~ X))
  
  XX1 = cbind(rep(1, n), H, rep(1, n), diag(n) %*% H)
  XX2 = cbind(rep(1, n), H, rep(-1, n), -diag(n) %*% H)
  Q1 = XX1 %*% co
  Q2 = XX2 %*% co
  Q = apply(cbind(Q1, Q2), 1, max)
  Qsingle = list(co = co, Q = Q)
  class(Qsingle) = "qlearn"
  Qsingle
}

predict.qlearn <- function(qlrn, X) {
  gcinfo(F)
  nn <- dim(X)[1]
  XX1 = cbind(rep(1, nn), X, rep(1, nn), diag(nn) %*% X)
  XX2 = cbind(rep(1, nn), X, rep(-1, nn), -diag(nn) %*% X)
  Q1 = XX1 %*% qlrn$co
  Q2 = XX2 %*% qlrn$co
  pred <- as.numeric(Q1 - Q2)
  z_hat <- I(pred > 0)*1
  opt_trt <- 2*z_hat - 1
  rm(XX1, XX2, Q1, Q2,z_hat)
  
  out <- list( pred = pred, opt_trt = opt_trt, coef.est = as.numeric(qlrn$co))
  return(out)
}


ROWL<-function (H, A, R2, pi = rep(1, n), pentype = "lasso", kernel = "linear", residual=TRUE,
                sigma = c(0.03, 0.05, 0.07), clinear = 2^(-2:2), m = 4, e = 1e-05)
{
  npar = length(clinear)
  n = length(A)
  p = dim(H)[2]
  if (residual==TRUE & max(R2) != min(R2)) {
    if (pentype == "lasso") {
      cvfit = cv.glmnet(H, R2, nfolds = m)
      co = as.matrix(predict(cvfit, s = "lambda.min", type = "coeff"))
    }
    else if (pentype == "LSE") {
      co = coef(lm(R2 ~ H))
    }
    else stop(gettextf("'pentype' is the penalization type for the regression step of Olearning, the default is 'lasso',\nit can also be 'LSE' without penalization"))
    r = R2 - cbind(rep(1, n), H) %*% co
  }
  else r = R2
  rand = sample(m, n, replace = TRUE)
  r = r/pi
  if (kernel == "linear") {
    V = matrix(0, m, npar)
    for (i in 1:m) {
      this = (rand != i)
      X = H[this, ]
      Y = A[this]
      R = r[this]
      Xt = H[!this, ]
      Yt = A[!this]
      Rt = r[!this]
      for (j in 1:npar) {
        model = wsvm(X, Y, R, C = clinear[j], e = e)
        YP = predict(model, Xt)
        V[i, j] = sum(Rt * (YP == Yt))/sum(YP == Yt)
      }
    }
    mimi = colMeans(V)
    best = which.max(mimi)
    cbest = clinear[best]
    model = wsvm(H, A, r, C = cbest, e = e)
  }
  if (kernel == "rbf") {
    nsig = length(sigma)
    V = array(0, c(npar, nsig, m))
    for (i in 1:m) {
      this = (rand != i)
      X = H[this, ]
      Y = A[this]
      R = r[this]
      Xt = H[!this, ]
      Yt = A[!this]
      Rt = r[!this]
      for (j in 1:npar) {
        for (s in 1:nsig) {
          model = wsvm(X, Y, R, "rbf", sigma = sigma[s],
                       C = clinear[j], e = e)
          YP = predict(model, Xt)
          V[j, s, i] = sum(Rt * (YP == Yt))/sum(YP ==
                                                  Yt)
        }
      }
    }
    mimi = apply(V, c(1, 2), mean)
    best = which(mimi == max(mimi), arr.ind = TRUE)
    bestC = clinear[best[1]]
    bestSig = sigma[best[2]]
    print(bestC)
    print(bestSig)
    model = wsvm(H, A, r, "rbf", bestSig, C = bestC, e = e)
  }
  model
}


wsvm<-function(X, A, wR, kernel='linear',sigma=0.05,C=1,e=0.00001){
  wAR=A*wR
  if (kernel=='linear'){
    K=X%*%t(X)
  }
  else if (kernel=='rbf'){
    rbf=rbfdot(sigma=sigma)
    K=kernelMatrix(rbf,X)
  } else stop(gettextf("Kernel function should be 'linear' or 'rbf'"))
  
  H=K*(wAR%*%t(wAR))
  n=length(A)
  solution=ipop(-abs(wR),H,t(A*wR),0,numeric(n),rep(C,n),0,maxiter=100)
  alpha=primal(solution)
  alpha1=alpha*wR*A
  if (kernel=='linear'){
    w=t(X)%*%alpha1 #parameter for linear
    fitted=X%*%w
    rm=sign(wR)*A-fitted
  } else if (kernel=='rbf'){
    #there is no coefficient estimates for gaussian kernel
    #but there is fitted value, first we compute the fitted value without adjusting for bias
    fitted=K%*%alpha1
    rm=sign(wR)*A-fitted
  }
  Imid =(alpha < C-e) & (alpha > e)
  rmid=rm[Imid==1];
  if (sum(Imid)>0){
    bias=mean(rmid)
  } else {
    Iup=((alpha<e)&(A==-sign(wR)))|((alpha>C-e)&(A==sign(wR)))
    Ilow=((alpha<e)&(A==sign(wR)))|((alpha>C-e)&(A==-sign(wR)))
    rup=rm[Iup]
    rlow=rm[Ilow]
    bias=(min(rup)+max(rlow))/2}
  fit=bias+fitted
  if (kernel=='linear') {
    model=list(alpha1=alpha1,bias=bias,fit=fit,beta=w)
    class(model)<-'linearcl'
  } else if (kernel=='rbf') {
    model=list(alpha1=alpha1,bias=bias,fit=fit,sigma=sigma,X=X)
    class(model)<-'rbfcl'}
  return (model)
}


predict.linearcl<-function(object,x,...){
  predict=sign(object$bias+x%*%object$beta)
}


predict.rbfcl<-function(object,x,...){
  rbf=rbfdot(sigma=object$sigma)
  n=dim(object$X)[1]
  if (is.matrix(x)) xm=dim(x)[1]
  else if (is.vector(x)) xm=1
  else stop('x must be vector or matrix')
  if (xm==1){ K <- apply(object$X,1,rbf,y=x)
  }else{   K<- matrix(0, xm, n)
  for (i in 1:xm) {K[i,]=apply(object$X,1,rbf,y=x[i,]) }}
  
  predict=sign(object$bias+K%*%object$alpha1)
}
