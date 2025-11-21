# ===========================================================
#  1. Load Libraries
# ===========================================================

packages <- c("tidyverse","caret","pROC","randomForest","xgboost",
              "e1071","glmnet","corrplot","MLmetrics","ranger",
              "doParallel","kernlab")

lapply(packages, require, character.only = TRUE)

# ===========================================================
#  2. Load Data
# ===========================================================

df <- read.csv("C:/Git/diabetes_binary_health_indicators_BRFSS2015.csv")
df$Diabetes_binary <- factor(df$Diabetes_binary, levels=c(0,1), labels=c("No","Yes"))

# ===========================================================
#  3. Train-Test Split
# ===========================================================

set.seed(123)
train_idx <- createDataPartition(df$Diabetes_binary, p=.8, list=FALSE)

train <- df[train_idx,]
test  <- df[-train_idx,]

# ===========================================================
#  4. Preprocessing
# ===========================================================

predictors <- setdiff(names(train), "Diabetes_binary")

preproc <- preProcess(train[, predictors], method=c("center","scale"))
train_scaled <- train; train_scaled[, predictors] <- predict(preproc, train[, predictors])
test_scaled  <- test;  test_scaled[, predictors]  <- predict(preproc, test[, predictors])

# ===========================================================
#  5. Train Control
# ===========================================================

ctrl <- trainControl(
  method="cv", number=5,
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  savePredictions=TRUE
)

# ===========================================================
#  MODEL 1 — Logistic Regression
# ===========================================================

log_model <- train(
  Diabetes_binary ~ .,
  data=train_scaled,
  method="glm",
  family="binomial",
  trControl=ctrl,
  metric="ROC"
)

log_model
log_pred <- predict(log_model, test_scaled, type="prob")[,2]
roc_log <- roc(test_scaled$Diabetes_binary, log_pred)
auc_log <- auc(roc_log)
auc_log

# ===========================================================
#  MODEL 2 — Random Forest (FAST VERSION)
# ===========================================================

library(doParallel)
stopImplicitCluster()
registerDoSEQ()   # caret runs single-threaded

rf_grid <- expand.grid(
  mtry = c(4, 6),
  splitrule = "gini",
  min.node.size = 5
)

rf_model <- train(
  Diabetes_binary ~ .,
  data = train,
  method = "ranger",
  trControl = trainControl(
    method="cv",
    number=3,
    classProbs=TRUE,
    summaryFunction=twoClassSummary,
    allowParallel=FALSE      # <-- explicitly disable parallel
  ),
  tuneGrid = rf_grid,
  metric = "ROC",
  num.trees = 100,
  importance = "impurity"
)

rf_model
rf_pred <- predict(rf_model, test, type="prob")[,2]
roc_rf <- roc(test$Diabetes_binary, rf_pred)
auc_rf <- auc(roc_rf)
auc_rf

# ===========================================================
#  MODEL 3 — XGBoost
# ===========================================================

xgb_grid <- expand.grid(
  nrounds=200,
  max_depth=c(3,4,5),
  eta=0.1,
  gamma=0,
  colsample_bytree=0.8,
  min_child_weight=1,
  subsample=0.8
)

xgb_model <- train(
  Diabetes_binary ~ .,
  data=train,
  method="xgbTree",
  trControl=ctrl,
  tuneGrid=xgb_grid,
  metric="ROC"
)

xgb_model
xgb_pred <- predict(xgb_model, test, type="prob")[,2]
roc_xgb <- roc(test$Diabetes_binary, xgb_pred)
auc_xgb <- auc(roc_xgb)
auc_xgb

# ===========================================================
# MODEL 4 — SUPER FAST SVM USING LiblineaR  (CORRECTED)
# ===========================================================

library(LiblineaR)

# Convert labels to numeric 0/1
y_train <- ifelse(train_scaled$Diabetes_binary == "Yes", 1, 0)
y_test  <- ifelse(test_scaled$Diabetes_binary == "Yes", 1, 0)

# Matrices
x_train <- as.matrix(train_scaled[, predictors])
x_test  <- as.matrix(test_scaled[, predictors])

# Grid
cost_grid <- c(0.1, 1)

best_auc <- 0
best_model <- NULL

for (C in cost_grid) {
  
  svm_fit <- LiblineaR(
    data = x_train,
    target = y_train,
    type = 0,        # fast linear logistic regression
    cost = C,
    bias = TRUE,
    verbose = FALSE
  )
  
  svm_prob <- predict(svm_fit, x_test, proba = TRUE)$probabilities[,2]
  roc_svm <- roc(test_scaled$Diabetes_binary, svm_prob)
  auc_svm <- auc(roc_svm)
  cat("Cost:", C, "AUC:", auc_svm, "\n")
  
  if (auc_svm > best_auc) {
    best_auc <- auc_svm
    best_model <- svm_fit
  }
}

# Final AUC
svm_auc <- best_auc
svm_auc

# Final prediction using the selected best model
svm_pred <- predict(best_model, x_test, proba = TRUE)$probabilities[,2]
roc_svm <- roc(test_scaled$Diabetes_binary, svm_pred)
auc_svm <- auc(roc_svm)
auc_svm



# ===========================================================
#  MODEL 5 — LASSO Logistic Regression
# ===========================================================

x <- model.matrix(Diabetes_binary ~ ., train_scaled)[,-1]
y <- train_scaled$Diabetes_binary
x_test <- model.matrix(Diabetes_binary ~ ., test_scaled)[,-1]

lasso <- cv.glmnet(x, y, family="binomial", alpha=1, type.measure="auc")

lasso_prob <- predict(lasso, newx=x_test, s="lambda.min", type="response")
lasso_prob <- as.numeric(lasso_prob)

roc_lasso <- roc(test_scaled$Diabetes_binary, lasso_prob)
auc_lasso <- auc(roc_lasso)
auc_lasso

# ===========================================================
#  Combine Model Results
# ===========================================================

results <- data.frame(
  Model=c("Logistic Regression","Random Forest","XGBoost","SVM (Radial)","LASSO"),
  AUC=c(auc_log, auc_rf, auc_xgb, auc_svm, auc_lasso)
) %>% arrange(desc(AUC))

print(results)

# ===========================================================
#  Plot all ROC curves
# ===========================================================

plot(roc_log, col="blue", main="ROC Curves for All Models", lwd=2)
plot(roc_rf, col="green", add=TRUE, lwd=2)
plot(roc_xgb, col="red", add=TRUE, lwd=2)
plot(roc_svm, col="purple", add=TRUE, lwd=2)
plot(roc_lasso, col="black", add=TRUE, lwd=2)

legend("bottomright",
       legend=c("Logistic","Random Forest","XGBoost","SVM","LASSO"),
       col=c("blue","green","red","purple","black"),
       lwd=2)
