library(devtools)
install_github(repo="ryantibs/conformal", subdir="conformalInference")
library(conformalInference)
library(randomForest)
library(xgboost)


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


train.fun = function(x, y, out = NULL) {
  xgb_mat = xgb.DMatrix(data = x, label = y)
  return(xgboost(data = xgb_mat, max.depth = 2, nrounds = 75))
}
predict.fun = function(out, newx) {
  newx_mat = xgb.DMatrix(data = newx)
  return(predict(out, newx_mat))
}


funs_list <- list(lm.funs(lambda = 0),
               lasso.funs(cv = T),
               rf.funs(ntree = 200, nodesize = 2),
               list("train"= train.fun,
                    "predict" = predict.fun))

outsplit_train <- function(funs=NULL,
                           x_train,
                           y_train,
                           alpha=0.05,
                           seed=0){

    conf <- conformal.pred.split(x_train,
                         y_train,
                         x_train,
                         alpha=alpha,
                         seed=0,
                         train.fun=funs$train,
                         predict.fun=funs$predict,
                         verbose = T)

  return(list("conf"=conf,
              "coverage" = colMeans(conf$lo <= y_train & y_train <= conf$up),
              "length" = colMeans(conf$up - conf$lo)))

}


results <- lapply(1:length(funs_list), function(x) outsplit_train(funs=funs_list[[x]], x_train, y_train, alpha))

coverage <- sapply(1:length(funs_list), function(x) results[[x]]$coverage)
length <- sapply(1:length(funs_list), function(x) results[[x]]$length)

coverage
length

mse <- c(mse_lm, mse_lasso, mse_rf, mse_xgboost)

mse
coverage
length
best_model
best_model <- which.min(length[coverage > 0.9])


outsplit_best <- conformal.pred.split(x=x_train,
                             y=y_train,
                             x0=x_test,
                             alpha=alpha,
                             seed=0,
                             train.fun=funs_list[[best_model]]$train,
                             predict.fun=funs_list[[best_model]]$predict,
                             verbose = T)


# save output
output <- cbind(outsplit_best$lo, outsplit_best$up)
write.table(output, "output.txt", row.names=F, col.names = F)


library(ggplot2)
library(patchwork)



a <- data.frame(y=y_train, "lo"=results[[1]]$conf$lo, "up"=results[[1]]$conf$up) %>%
  arrange(y) %>%
  mutate(id = 1:nrow(x_train)) %>%
  ggplot() +
  geom_linerange(aes(x=id, ymin=lo, ymax=up), size=3, color="gray") +
  geom_point(aes(id, y)) +
  theme_minimal() +
  ylim(90, 150) +
  ggtitle("lm")

b <- data.frame(y=y_train, "lo"=results[[2]]$conf$lo, "up"=results[[2]]$conf$up) %>%
  arrange(y) %>%
  mutate(id = 1:nrow(x_train)) %>%
  ggplot() +
  geom_linerange(aes(x=id, ymin=lo, ymax=up), size=3, color="gray") +
  geom_point(aes(id, y)) +
  theme_minimal() +
  ylim(90, 150) +
  ggtitle("lasso")

c <- data.frame(y=y_train, "lo"=results[[3]]$conf$lo, "up"=results[[3]]$conf$up) %>%
  arrange(y) %>%
  mutate(id = 1:nrow(x_train)) %>%
  ggplot() +
  geom_linerange(aes(x=id, ymin=lo, ymax=up), size=3, color="gray") +
  geom_point(aes(id, y)) +
  theme_minimal() +
  ylim(90, 150) +
  ggtitle("rf")

d <- data.frame(y=y_train, "lo"=results[[4]]$conf$lo, "up"=results[[4]]$conf$up) %>%
  arrange(y) %>%
  mutate(id = 1:nrow(x_train)) %>%
  ggplot() +
  geom_linerange(aes(x=id, ymin=lo, ymax=up), size=3, color="gray") +
  geom_point(aes(id, y)) +
  theme_minimal() +
  ylim(90, 150) +
  ggtitle("xgb")

(a | b | c | d)

