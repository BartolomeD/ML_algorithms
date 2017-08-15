source('gradient_descent.R')
source('losses.R')

# load a test dataset
data(cars)

# generate theta parameters using g_descent()
theta_gdescent <- g_descent(y = cars[,1], x = cars[,2])
# generate theta parameters using lm()
theta_lm <- as.vector(lm(cars[,1] ~ cars[,2])$coefficient)
# generate theta parameters manually
theta_manual <- c(NA, NA)
theta_manual[2] <- (sum((cars[,2] - mean(cars[,2]))*(cars[,1] - mean(cars[,1])))) / (sum((cars[,2] - mean(cars[,2]))^2))
theta_manual[1] <- mean(cars[,1]) - theta_manual[2] * mean(cars[,2])

# plot the data with g_descent() and lm() hypotheses
plot(y = cars[,1], x = cars[,2], bty = 'n', xlab = 'speed', ylab = 'distance')
abline(theta_gdescent[1], theta_gdescent[2], col = 'red')
abline(lm(cars[,1] ~ cars[,2]), col = 'blue')
legend(90, 10, c('lm( )', 'g_descent( )'), lty = c(1, 1), lwd = c(2, 2), col = c('blue', 'red'))

# calculate loss for g_descent()
loss(y = cars[,1], x = cars[,2], theta = theta_gdescent, eval = 'mae')
# calculate loss for lm()
loss(y = cars[,1], x = cars[,2], theta = theta_lm, eval = 'mae')
# calculate loss for manual
loss(y = cars[,1], x = cars[,2], theta = theta_manual, eval = 'mae')

# running time g_descent()
t1 <- proc.time()
g_descent(y = cars[,1], x = cars[,2])
t2 <- proc.time()
t2-t1

# running time of lm()
t1 <- proc.time()
as.vector(lm(cars[,1] ~ cars[,2])$coefficient)
t2 <- proc.time()
t2-t1