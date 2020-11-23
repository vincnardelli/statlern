library(devtools)
install_github(repo="ryantibs/conformal", subdir="conformalInference")
library(conformalInference)
library(randomForest)
library(xgboost)
library(ggplot2)


train <- read.table("train.txt", header = T)
test <- read.table("test.txt", header = T)

x_train <- as.matrix(train[, 1:5])
x_test <- as.matrix(test)
y_train <- train$y


alpha = 0.05

# vediamo come performa il modello sul train!
lm <- lm(y~., train)
lm_predict<- predict(lm, newx = x_train)
mse_lm <- mean((lm_predict - y_train)^2)
plot(x_train[,2], y_train)


cv <- cv.glmnet(x_train, y_train)
cv$lambda.min

model <- glmnet(x_train, y_train, lambda = cv$lambda.min)
model_fit <- predict(model, x_train, s = cv$lambda.min)
mse_lasso <- mean((model_fit - y_train)^2)

rf <- randomForest(x_train, y_train, ntree = 200, nodesize = 2)
rf
mse_rf <- mean((rf$predicted - y_train)^2)


#split for XGB
train_id <- sample(1:nrow(x_train), 0.80*nrow(x_train))
test_id <- which(!(1:nrow(x_train) %in% train_id))
length(train_id) + length(test_id)


xgb_train = xgb.DMatrix(data = x_train[train_id, ], label = y_train[train_id])
xgb_test = xgb.DMatrix(data = x_train[test_id, ], label = y_train[test_id])


xgbc = xgboost(data = xgb_train, max.depth = 2, nrounds = 75)
print(xgbc)

pred_y = predict(xgbc, xgb_train)
mean((y_train[train_id] - pred_y)^2)


pred_y = predict(xgbc, xgb_test)
mean((y_train[test_id] - pred_y)^2)


xgb_all = xgb.DMatrix(data = x_train, label = y_train)
xgbc = xgboost(data = xgb_all, max.depth = 2, nrounds = 75)
pred_y = predict(xgbc, xgb_all)
mean((y_train - pred_y)^2)

mse_xgboost <- mean((pred_y - y_train)^2)


# lm
funs_lm = lm.funs(lambda = 0)
outsplit_lm = conformal.pred.split(x_train,
                                   y_train,
                                   x_train,
                                   alpha=alpha,
                                   seed=0,
                                   train.fun=funs_lm$train,
                                   predict.fun=funs_lm$predict,
                                   verbose = T)

cov_split_lm = colMeans(outsplit_lm$lo <= y_train & y_train <= outsplit_lm$up)
cov_split_lm
len_split_lm = colMeans(outsplit_lm$up - outsplit_lm$lo)
len_split_lm


# lasso
funs_lasso = lasso.funs(cv = T)
outsplit_lasso = conformal.pred.split(x_train,
                                      y_train,
                                      x_train,
                                      alpha=alpha,
                                      seed=0,
                                      train.fun=funs_lasso$train,
                                      predict.fun=funs_lasso$predict,
                                      verbose = T)

cov_split_lasso = colMeans(outsplit_lasso$lo <= y_train & y_train <= outsplit_lasso$up)
cov_split_lasso
len_split_lasso = colMeans(outsplit_lasso$up - outsplit_lasso$lo)
len_split_lasso


# random forest
funs_rf = rf.funs(ntree = 200, nodesize = 2)

outsplit_rf = conformal.pred.split(x_train,
                                   y_train,
                                   x_train,
                                   alpha=alpha,
                                   seed=0,
                                   train.fun=funs_rf$train,
                                   predict.fun=funs_rf$predict,
                                   verbose = T)

cov_split_rf = colMeans(outsplit_rf$lo <= y_train & y_train <= outsplit_rf$up)
cov_split_rf
len_split_rf = colMeans(outsplit_rf$up - outsplit_rf$lo)
len_split_rf



outsplit_xgb = conformal.pred.split(x_train,
                                    y_train,
                                    x_train,
                                    alpha=alpha,
                                    seed=0,
                                    train.fun=function(x_train, y_train) xgboost(data = xgb.DMatrix(data = x_train, label = y_train), max.depth = 3, nrounds = 50),
                                    predict.fun=function(model, x_train) predict(model, xgb.DMatrix(data = x_train)),
                                    verbose = T)



# xgboost
train.fun = function(x, y, out = NULL) {
  xgb_mat = xgb.DMatrix(data = x, label = y)
  return(xgboost(data = xgb_mat, max.depth = 2, nrounds = 75))
}
predict.fun = function(out, newx) {
  newx_mat = xgb.DMatrix(data = newx)
  return(predict(out, newx_mat))
}

outsplit_xgb = conformal.pred.split(x_train,
                                    y_train,
                                    x_train,
                                    alpha=alpha,
                                    seed=0,
                                    train.fun=train.fun,
                                    predict.fun=predict.fun,
                                    verbose = T)

cov_split_xgb = colMeans(outsplit_xgb$lo <= y_train & y_train <= outsplit_xgb$up)
cov_split_xgb
len_split_xgb = colMeans(outsplit_xgb$up - outsplit_xgb$lo)
len_split_xgb

mse <- c(mse_lm, mse_lasso, mse_rf, mse_xgboost)
coverage <- c(cov_split_lm, cov_split_lasso, cov_split_rf, cov_split_xgb)
length <- c(len_split_lm, len_split_lasso, len_split_rf, len_split_xgb)

mse
coverage
length
which.min(length[coverage > 0.9])


# prediction

outsplit_xgb = conformal.pred.split(x_train,
                                    y_train,
                                    x_test,
                                    alpha=alpha,
                                    seed=0,
                                    train.fun=train.fun,
                                    predict.fun=predict.fun,
                                    verbose = T)



# output
output <- cbind(outsplit_xgb$lo, outsplit_xgb$up)

write.table(output, "output.txt", row.names=F, col.names = F)
