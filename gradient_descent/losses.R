loss <- function(y, x, theta, eval) {
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