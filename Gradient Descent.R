# formulate nD Gradient Descent algorithm
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

# formulate evaluation method
evaluate <- function(y, x, theta, eval) {
  x <- as.data.frame(x)
  
  if (length(theta)-1 != ncol(x)) {
    return('Error: theta and data dimension mismatch.')
  }
  
  if (eval == 'mae') {
    return(mean(abs(y - (theta[1] + t(theta[-1] * t(x))))))
  } else if (eval == 'rmse') {
    return(sqrt(mean((y - (theta[1] + t(theta[-1] * t(x))))^2)))
  } else if (eval == 'mse') {
    return(mean((y - (theta[1] + t(theta[-1] * t(x))))^2))
  } else {
    return('Error: enter valid evaluation measure. Supported metrics are: mae, rmse and mse.')
  }
}

# load a test dataset
data(cars)

# generate theta parameters using GDescent()
theta_GDescent <- g_descent(y = cars[,1], x = cars[,2])
# generate theta parameters using lm()
theta_lm <- as.vector(lm(cars[,1] ~ cars[,2])$coefficient)
# generate theta parameters manually
theta_manual <- c(NA, NA)
theta_manual[2] <- (sum((cars[,2] - mean(cars[,2]))*(cars[,1] - mean(cars[,1])))) / (sum((cars[,2] - mean(cars[,2]))^2))
theta_manual[1] <- mean(cars[,1]) - theta_manual[2] * mean(cars[,2])

# plot the data with GDescent() and lm() hypotheses
plot(y = cars[,1], x = cars[,2], bty = 'n', xlab = 'speed', ylab = 'distance')
abline(theta_GDescent[1], theta_GDescent[2], col = 'red')
abline(lm(cars[,1] ~ cars[,2]), col = 'blue')
legend(90, 10, c('lm( )', 'g_descent( )'), lty = c(1, 1), lwd = c(2, 2), col = c('blue', 'red'))

# calculate loss for GDescent()
evaluate(y = cars[,1], x = cars[,2], theta = theta_GDescent, eval = 'mae')
# calculate loss for lm()
evaluate(y = cars[,1], x = cars[,2], theta = theta_lm, eval = 'mae')
# calculate loss for manual
evaluate(y = cars[,1], x = cars[,2], theta = theta_manual, eval = 'mae')

# running time GDescent()
t1 <- proc.time()
g_descent(y = cars[,1], x = cars[,2])
t2 <- proc.time()
t2-t1

# running time of lm()
t1 <- proc.time()
as.vector(lm(cars[,1] ~ cars[,2])$coefficient)
t2 <- proc.time()
t2-t1
