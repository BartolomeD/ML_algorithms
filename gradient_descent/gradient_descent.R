' Function for linear regression parameter estimation with gradient descent.

# Arguments:
    y:      array, the target variable
    x:      matrix, the trainset features
    n_iter: int, number of gradient descent iterations
    alpha:  float, the learning rate
    plot:   bool, if TRUE regression line is plotted
    
# Returns:
    theta:  array, linear regression parameters
'
g_descent <- function(y, x, n_iter = 1000, alpha = 0.01, plot = FALSE) {
  
  # create dataframe from predictor data
  x <- as.data.frame(x)
  
  # normalize the data
  norm_data <- (cbind(y, x) - min(cbind(y, x))) / (max(cbind(y, x)) - min(cbind(y, x)))
  observed <- norm_data[,1]
  data <- cbind(rep(1, nrow(x)), norm_data[,-1])
  h <- matrix(NA, nrow = nrow(data), ncol = ncol(data))
  
  # create initial theta matrix
  if (ncol(x) == 1) {
    theta <- c(mean(data[,-1]), rep(0, ncol(x))) 
  } else {
    theta <- c(mean(colMeans(data[,-1])), rep(0, ncol(x)))
  }
    
  # initialize gradient descent iterations
  for (i in 1:n_iter) {
    
    # create predictions with generated theta values
    for (j in 1:ncol(data)) {
      h[,j] <- theta[j]*data[,j]
    }
    prediction <- rowSums(h)
    
    # update the theta values using gradient descent
    theta[1] <- theta[1] - alpha * sum(prediction - observed)
    for (d in 2:ncol(data)) {
      theta[d] <- theta[d] - alpha * sum((prediction - observed) * data[,d])
    }
  }
  
  # return error if thetas do not converge
  if (is.nan(theta[1]) | is.nan(theta[2])) {
    return('Error: the theta-values did not converge. Lower alpha or increase the number of iterations')
  }
  
  # reverse normalize the data
  theta[1] <- theta[1] * (max(cbind(x,y)) - min(cbind(x,y))) + min(cbind(x,y))
  observed <- observed * (max(cbind(x,y)) - min(cbind(x,y))) + min(cbind(x,y))
  data <- data[,-1] * (max(x) - min(x)) + min(x) 
  
  # plot the regression line
  if (plot) {
    if (ncol(x) == 1) {
      plot(x = data, y = observed, bty = 'n')
      abline(theta[1], theta[2], col = 'blue')
    } else {
      print('Error: plot function only supports two-dimensional data.')
    }
  }
  
  # print the theta-values
  return(theta)
}
