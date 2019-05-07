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

library(AmesHousing)
ames <- make_ames()

## -----------------------------------------------------------------------------

library(caret)
library(dplyr)  # load this _after_ caret
ames <- 
  make_ames() %>% 
  # Remove quality-related predictors
  dplyr::select(-matches("Qu"))
nrow(ames)

# Make sure that you get the same random numbers
set.seed(4595)
in_train <- createDataPartition(ames$Sale_Price, p = 3/4, list = FALSE)

ames_train <- ames[ in_train,]
ames_test  <- ames[-in_train,]

nrow(ames_train)/nrow(ames)

## -----------------------------------------------------------------------------

ggplot(ames_train, aes(x = Sale_Price)) + 
  geom_line(stat = "density", trim = TRUE) + 
  geom_line(data = ames_test, 
            stat = "density", 
            trim = TRUE, col = "red") 

## -----------------------------------------------------------------------------

## model_fn(Sale_Price ~ Neighborhood + Year_Sold + Neighborhood:Year_Sold, data = ames_train)

## model_fn(Sale_Price ~ ., data = ames_train)

## model_fn(log10(Sale_Price) ~ ns(Longitude, df = 3) + ns(Latitude, df = 3), data = ames_train)

## # Usually, the variables must all be numeric
## pre_vars <- c("Year_Sold", "Longitude", "Latitude")
## model_fn(x = ames_train[, pre_vars],
##          y = ames_train$Sale_Price)

## -----------------------------------------------------------------------------

simple_lm <- lm(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train)

simple_lm_values <- broom::augment(simple_lm)
names(simple_lm_values)

summary(simple_lm)

## -----------------------------------------------------------------------------

library(caret)
ctrl <-
  trainControl(
    method = "cv",            # defaults to 10-fold
    savePredictions = "final" # save the holdout predictions 
  )

## -----------------------------------------------------------------------------

set.seed(5616)
lm_fit <- train(log10(Sale_Price) ~ Latitude + Longitude, 
                data = ames_train, 
                method = "lm", 
                trControl = ctrl)
lm_fit

## -----------------------------------------------------------------------------

library(purrr)
holdout_results <- 
  lm_fit %>% 
  pluck("pred") %>% 
  arrange(rowIndex) %>% 
  bind_cols(ames_train)

holdout_results %>% 
  dplyr::select(obs, pred, Resample, Neighborhood, Alley) %>% 
  dplyr::slice(1:7)

## -----------------------------------------------------------------------------

simple_knn <- knnreg(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train, k = 2)

naive_rmse <- caret::RMSE(pred = predict(simple_knn, ames_train), obs = log10(ames_train$Sale_Price))
naive_rmse

## -----------------------------------------------------------------------------

set.seed(5616)
knn_fit <- train(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train,
                 method = "knn", trControl = ctrl,
                 tuneGrid = data.frame(k = 2))
getTrainPerf(knn_fit)

# Before:
naive_rmse

## -----------------------------------------------------------------------------

rs <- resamples(list(lm = lm_fit, knn = knn_fit), metric = "RMSE")

rmse_values <- 
  rs %>% 
  pluck("values") %>% 
  dplyr::select(Resample, `lm~RMSE`, `knn~RMSE`) %>% 
  dplyr::rename(lm = `lm~RMSE`, knn = `knn~RMSE`) 
rmse_corr <- cor(rmse_values$lm, rmse_values$knn)

## -----------------------------------------------------------------------------

rmse_values %>% 
  gather(model, RMSE, -Resample) %>% 
  ggplot(aes(x = model, y = RMSE, group = Resample, col = Resample)) + 
  geom_line() + 
  theme(legend.position = "none")

## -----------------------------------------------------------------------------

compare_models(lm_fit, knn_fit)

## -----------------------------------------------------------------------------

# Leave the others at their defaults
knn_grid <- data.frame(k = 1:20)

ctrl <- trainControl(method = "cv", savePredictions = "final", returnResamp = "all")

## -----------------------------------------------------------------------------

set.seed(5616)
knn_tuned <- 
  train(log10(Sale_Price) ~ Longitude + Latitude, 
        data = ames_train,
        method = "knn", 
        trControl = ctrl,
        tuneGrid = knn_grid)

knn_tuned

## -----------------------------------------------------------------------------

ggplot(knn_tuned)

## -----------------------------------------------------------------------------

getTrainPerf(knn_tuned)
knn_tuned$bestTune

# since `savePredictions = "final"`:
knn_tuned$pred %>% slice(1:3)

## -----------------------------------------------------------------------------

knn_tuned %>% pluck("resample") %>% slice(1:3)

## -----------------------------------------------------------------------------

ggplot(knn_tuned) + 
  geom_line(data = knn_tuned$resample, 
            aes(group = Resample, col = Resample), 
            alpha = .3, lwd = 1) + 
  theme(legend.position = "none")

## -----------------------------------------------------------------------------

# Based on final model with optimized `k`
predict(knn_tuned, ames_test %>% slice(1:3))

