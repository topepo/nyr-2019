## -----------------------------------------------------------------------------
## Code for Machine Learning with caret at NYR in 2019 by Max Kuhn

## -----------------------------------------------------------------------------
## Prelims

library(tidymodels)

thm <- theme_bw() + 
  theme(
    panel.background = element_rect(fill = "transparent", colour = NA), 
    plot.background = element_rect(fill = "transparent", colour = NA),
    legend.position = "top",
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.key = element_rect(fill = "transparent", colour = NA)
  )
theme_set(thm)

## -----------------------------------------------------------------------------

load("churn.RData")

## -----------------------------------------------------------------------------

library(tidymodels)

two_class_example %>% head(4)

two_class_example %>% 
	conf_mat(truth = truth, estimate = predicted)

two_class_example %>% 
	accuracy(truth = truth, estimate = predicted)

## -----------------------------------------------------------------------------

library(caret)

confusionMatrix(two_class_example$predicted, 
                two_class_example$truth)

two_class_example %>% 
	conf_mat(truth = truth, estimate = predicted)

## -----------------------------------------------------------------------------

roc_obj <- 
  two_class_example %>% 
  roc_curve(truth, Class1)

two_class_example %>% roc_auc(truth, Class1)

# (there's also pr_curve and)
two_class_example %>% pr_auc(truth, Class1)

## -----------------------------------------------------------------------------

autoplot(roc_obj) + thm


## -----------------------------------------------------------------------------

str(churn)
table(churn$class)

## -----------------------------------------------------------------------------

library(caret)

set.seed(394)
churn_split <- createDataPartition(churn$class, list = FALSE, p = 3/4)

churn_train <- churn[ churn_split,]
churn_test  <- churn[-churn_split,]

## -----------------------------------------------------------------------------

library(caret)
ctrl <- trainControl(
	method = "cv",
	# Also predict the probabilities
	classProbs = TRUE,
	# Compute the ROC AUC as well as the sens and  
	# spec from the default 50% cutoff. The 
	# function `twoClassSummary` will produce those. 
	summaryFunction = twoClassSummary,
	savePredictions = "final",
	sampling = "down"
)

## -----------------------------------------------------------------------------

library(rpart)
library(partykit)

mod <- rpart(class ~ ., data = churn_train)
plot(as.party(mod))

## -----------------------------------------------------------------------------

set.seed(5515)
cart_mod <- train(
	x = churn_train %>% dplyr::select(-class), 
	y = churn_train$class,
	method = "rpart",
	metric = "ROC",
	tuneLength = 20,
	trControl = ctrl
)

cart_mod$finalModel

## -----------------------------------------------------------------------------

ggplot(cart_mod) + scale_x_log10()

## -----------------------------------------------------------------------------

cart_smaller <- update(cart_mod, param = list(cp = .003))

## -----------------------------------------------------------------------------

cart_mod$results$cp[12]
cart_smaller <- update(cart_mod, param = list(cp = cart_mod$results$cp[12]))

getTrainPerf(cart_smaller)
getTrainPerf(cart_mod)

## -----------------------------------------------------------------------------

cart_smaller %>%
  pluck("pred") %>%
  group_by(Resample) %>%
  roc_curve(obs, yes) %>%
  ungroup() %>%
  ggplot() +
  aes(x = 1 - specificity, y = sensitivity,
      col = Resample, group = Resample) +
  geom_path(alpha = .5)  +
  geom_abline(col = "red", alpha = .5, lty = 2) +
  theme(legend.position = "none")

## -----------------------------------------------------------------------------

approx_roc_curve <- function(x, label) {
  x %>%
    pluck("pred") %>%
    roc_curve(obs, yes) %>%
    mutate(model = label)
}
approx_roc_curve(cart_smaller, "CART") %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path()  +
  geom_abline(col = "red", alpha = .5, lty = 2)

## -----------------------------------------------------------------------------

confusionMatrix(cart_smaller)

## -----------------------------------------------------------------------------

cart_imp <- varImp(cart_smaller, scale = FALSE, 
                   surrogates = FALSE, 
                   competes = FALSE)
ggplot(cart_imp) + xlab("")

## -----------------------------------------------------------------------------

set.seed(5515)
cart_bag <- train(
	x = churn_train %>% dplyr::select(-class), 
	y = churn_train$class,
	method = "treebag",
	metric = "ROC",
	trControl = ctrl
)

cart_bag
## -----------------------------------------------------------------------------

confusionMatrix(cart_bag)

## -----------------------------------------------------------------------------

all_curves <-
  approx_roc_curve(cart_smaller, "CART") %>%
  bind_rows(approx_roc_curve(cart_bag, "Bagged CART"))

ggplot(all_curves) +
  aes(x = 1 - specificity, y = sensitivity,
      group = model, col = model) +
  geom_path()  +
  geom_abline(col = "red", alpha = .5, lty = 2)

## -----------------------------------------------------------------------------

bag_imp <- varImp(cart_bag, scale = FALSE)
ggplot(bag_imp) + xlab("")

## -----------------------------------------------------------------------------

no_dummies <- 
  recipe(class ~ ., data = churn_train) %>%
	# step_bin2factor(dummy_var_1, dummy_var_2) %>%  <- the sauce     
	step_zv(all_predictors())

nb_grid <- expand.grid(usekernel = TRUE, fL = 0, adjust = 1)

## -----------------------------------------------------------------------------

set.seed(5515)
nb_mod <- train(
	no_dummies,
	data = churn_train,
	method = "nb",
	metric = "ROC",
	tuneGrid = nb_grid,
	trControl = ctrl
)

nb_mod

## -----------------------------------------------------------------------------

all_curves <-
  all_curves %>%
  bind_rows(approx_roc_curve(nb_mod, "Naive Bayes"))

ggplot(all_curves) +
  aes(x = 1 - specificity, y = sensitivity,
      group = model, col = model) +
  geom_path()  +
  geom_abline(col = "red", alpha = .5, lty = 2)

## -----------------------------------------------------------------------------

test_res <- churn_test %>%
	dplyr::select(class) %>%
	mutate(
		prob = predict(nb_mod, churn_test, type = "prob")[, "yes"],
		pred = predict(nb_mod, churn_test)
	)
roc_auc(test_res, class, prob)
getTrainPerf(nb_mod)

## -----------------------------------------------------------------------------

test_roc <- roc_curve(test_res, class, prob)

## -----------------------------------------------------------------------------

ggplot(test_roc) +
  aes(x = 1 - specificity, y = sensitivity) +
  geom_path()  +
  geom_abline(col = "red", alpha = .5, lty = 2)


ggplot(test_res, aes(x = prob)) + 
  geom_histogram(binwidth = .04) + 
  facet_wrap( ~ class) +
  xlab("Pr[Churn]")

## -----------------------------------------------------------------------------

confusionMatrix(
  data = test_res$pred,
  reference = test_res$class
)
