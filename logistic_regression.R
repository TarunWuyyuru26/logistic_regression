
# Clearing the Environment
  rm(list = ls(all = TRUE))

# Creating some Custom Functionalities for performing Logistic Regression
  
  #Sigmoid function
    sigmoid=function(x) {1/(1+exp(-x))}
  
  #Fit logistic regression
    fit_logit=function(x,y,intercept=T,tol=10e-5,max_it=100){
      ##Type conversion
      if (!is.matrix(x))
      {
        x=as.matrix(x)
      }
      if (!is.matrix(y))
      {
        y=as.matrix(y)
      }
      ##Add intercept is required
      if (intercept)
      {
        x=cbind(x,1)
      }
      ##Algorithm initialization
      iterations=0
      converged=F
      ##Weights are initialized to 1 
      coeffs=matrix(1,dim(x)[2])
      
      ##Updates the weight until the max number of iter
      ##Or the termination criterion is met
      while (iterations<max_it& !converged)
      {
        iterations=iterations+1
        nu<-sigmoid(x %*% coeffs)
        old_pred=sigmoid(x %*% coeffs)
        nu_diag=diag(nu[,1])
        ##Weights update
        coeffs=coeffs + solve(t(x) %*% nu_diag %*% x)%*% t(x) %*% (y-nu)
        ##compute mse to check termination
        mse=mean((y-sigmoid(x%*%coeffs))^2)
        ##Stop computation if tolerance is reached
        if (mse<tol)
        {
          converged=T
        }
        
      }
      ##Creates the logit objects 
      my_logit=list(intercept=intercept)
      my_logit[['coeffs']]=coeffs
      my_logit[['preds']]=sigmoid(x%*%coeffs)
      my_logit[['residuals']]=my_logit[['preds']]-y
      my_logit[['mse']]=mean(my_logit[['residuals']]^2)
      my_logit[['iteration']]=iterations
      attr(my_logit, "class") <- "my_logit"
      return(my_logit)
    }

  #Predict the outcome on new data
    predict.my_logit<-function(my_logit,x,probs=T,..){
      if (!is.matrix(x))
      {
        x=as.matrix(x)
      }
      if (my_logit[['intercept']])
      {
        x=cbind(x,1)
      }
      if (probs)
      {
        sigmoid(x %*% my_logit[['coeffs']])
      }
      else
      {
        sigmoid(x %*% my_logit[['coeffs']])>0.5
      }
    }

# Implementation

  # Converting the Problem statement into Binary Class problem;
  # i'm assigning Positive values to setosa instances and negative values to other instances in iris dataset
    y = iris[,5]=='setosa'
    lr_model = fit_logit(iris[,1:4],y)
    lr_model

  # Predictions
    preds = predict(my_logit = lr_model, x = iris[,1:4], probs = TRUE)
    y_hat = as.character(ifelse(preds > 0.5, '1', '0'))
    unique(y_hat)
    
  # Confusion Matrix
    library(caret)
    
    y = iris$Species == 'setosa'; y = ifelse(y == TRUE, '1', '0')
    
    class(y)
    class(y_hat)
    
    y = as.factor(y)
    y_hat = as.factor(y_hat)
    
    confusionMatrix(data = y, reference = y_hat)
