library(stabs)
library(hdi)
library(randomForest)
library(knockoff)
library(leaps)
library(VSURF)

data <- read.table("low.txt", header = T)
X <- as.matrix(data[, 2:ncol(data)])
y <- data$y
alpha <- 0.05

# forward ---
sub_fwd = regsubsets(y~., data = data, method = "forward")
summary(sub_fwd)

coef(sub_fwd, 4)

plot(sub_fwd, scale = "adjr2")
plot(sub_fwd, scale = "bic")



# pca ----
pca <- prcomp(X, scale = TRUE,  rank. = 10)
summary(pca)



pca.select <- function(x, y, ncomponents=3, nvariables=3){
  pca <- prcomp(X, scale = TRUE,  rank. = ncomponents)
  selected <- sapply(1:ncomponents, function(x) which(pca$rotation[, x] %in% sort(abs(pca$rotation[, x]), decreasing = T)[1:nvariables]))
  selected <- unique(unlist(selected))
  selected
}



fit <- multi.split(x=X, y=y, B=10,
                   fraction=0.5, ci=TRUE, ci.level = 1-alpha,
                   model.selector = pca.select,
                   verbose = TRUE)

fit


# lasso ----
fit <- multi.split(x=X, y=y, B=10,
                   fraction=0.5, ci=TRUE, ci.level = 1-alpha,
                   model.selector = lasso.cv,
                   verbose = TRUE)

fit


# random forest ----
rf <- randomForest(X, y, ntree = 500, nodesize = 10, importance = T)
sort(rf$importance[,2], decreasing = T)

# 16 10 17 9

selected <- which(rf$importance[,2] > quantile(rf$importance[,2], 0.75))
selected

rf.select <-  function (x, y, ntree = 200, nodesize = 5, quantile = 0.8, ...){
  rf <- randomForest(x, y, ntree = ntree, nodesize = nodesize, importance = T, ...)

  sel <- which(rf$importance[,2] > quantile(rf$importance[,2], quantile))
  sel
}

fit <- multi.split(x=X, y=y, B=10,
            fraction=0.5, ci=TRUE, ci.level = 1-alpha,
            model.selector = rf.select,
            verbose = TRUE)

fit


# stability selection ----
fit <- stabsel(x = X, y = y,
               fitfun = glmnet.lasso,
               q = 10,
               cutoff = 0.75)
plot(fit)
fit
#   14  15  16  19


rf.stability <-  function (x, y, q, ntree = 200, nodesize = 10, quantile = 0.5, ...){
  rf <- randomForest(x[, sample(1:ncol(x), q)], y, ntree = ntree, nodesize = nodesize, importance = T)

  sel <- which(rf$importance[,2] > quantile(rf$importance[,2], quantile))
  ret <- logical(ncol(x))
  ret[sel] <- TRUE
  names(ret) <- colnames(x)

  return(list(selected = ret, path = NULL))
}

fit <- stabsel(x = X, y = y,
               fitfun = rf.stability,
               q = 10,
               cutoff = 0.8)

plot(fit)
fit


#knock-off ----
result = knockoff.filter(X, y, knockoffs = create.fixed, statistic = stat.glmnet_lambdasmax)
print(result)



result = knockoff.filter(X, y)
result


# VSURF ----

vsurf <- VSURF(X, y, ntree = 500,  parallel = TRUE, ncores = 10)
vsurf$varselect.thres
# 8  3  9 11 16  4

vsurf.select <-  function (x, y, ntree = 500, ...){
  vsurf_threshold <- VSURF_thres(x, y, ntree = ntree, parallel = TRUE, ncores = 10)
  vsurf_threshold$varselect.thres
}

fit <- multi.split(x=X, y=y, B=10,
                   fraction=0.5, ci=TRUE, ci.level = 1-alpha,
                   model.selector = vsurf.select,
                   verbose = TRUE)

fit

selected <- c(10, 16)
model <- lm(y ~ ., data[, selected+1])
summary(model)
