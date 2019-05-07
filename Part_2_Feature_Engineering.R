## -----------------------------------------------------------------------------
## Code for Machine Learning with caret at NYR in 2019 by Max Kuhn

## -----------------------------------------------------------------------------
## Prelims

library(caret)
library(ggplot2)

thm <- theme_bw() + 
  theme(
    panel.background = element_rect(fill = "transparent", colour = NA), 
    plot.background = element_rect(fill = "transparent", colour = NA),
    legend.position = "top",
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.key = element_rect(fill = "transparent", colour = NA)
  )
theme_set(thm)

options(digits = 3, width = 120)

## -----------------------------------------------------------------------------
## Previously...

library(AmesHousing)
library(dplyr)
ames <- make_ames() %>% 
  dplyr::select(-matches("Qu"))

# Make sure that you get the same random numbers
set.seed(110490)
in_train <- createDataPartition(ames$Sale_Price, p = 3/4, list = FALSE)

ames_train <- ames[ in_train,]
ames_test  <- ames[-in_train,]

## -----------------------------------------------------------------------------

ggplot(ames_train, aes(x = Neighborhood)) + geom_bar() + coord_flip() + xlab("")

## -----------------------------------------------------------------------------

mod_rec <- 
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) %>%
  step_log(Sale_Price, base = 10)

## -----------------------------------------------------------------------------

mod_rec <- recipe(
    Sale_Price ~ Longitude + Latitude + Neighborhood, 
    data = ames_train
  ) %>%
  step_log(Sale_Price, base = 10) %>%
  
  # Lump factor levels that occur in 
  # <= 5% of data as "other"
  step_other(Neighborhood, threshold = 0.05) %>%
  
  # Create dummy variables for _any_ factor variables
  step_dummy(all_nominal())

mod_rec

## -----------------------------------------------------------------------------

mod_rec_trained <- prep(mod_rec, training = ames_train, verbose = TRUE)

mod_rec_trained

## -----------------------------------------------------------------------------

ames_test_dummies <- bake(mod_rec_trained, new_data = ames_test)
names(ames_test_dummies)

## -----------------------------------------------------------------------------

price_breaks <- (1:6)*(10^5)

ggplot(
    ames_train, 
    aes(x = Year_Built, y = Sale_Price)
  ) + 
  geom_point(alpha = 0.4) +
  scale_x_log10() + 
  scale_y_continuous(
    breaks = price_breaks, 
    trans = "log10"
  ) +
  geom_smooth(method = "loess")

library(MASS) # to get robust linear regression model

ggplot(
    ames_train, 
    aes(x = Year_Built, 
        y = Sale_Price)
  ) + 
  geom_point(alpha = 0.4) +
  scale_y_continuous(
    breaks = price_breaks, 
    trans = "log10"
  ) + 
  facet_wrap(~ Central_Air, nrow = 2) +
  geom_smooth(method = "rlm") 

## -----------------------------------------------------------------------------

mod1 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air, data = ames_train)
mod2 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air + Year_Built:Central_Air, data = ames_train)
anova(mod1, mod2)

## -----------------------------------------------------------------------------

recipe(Sale_Price ~ Year_Built + Central_Air, data = ames_train) %>%
  step_log(Sale_Price) %>%
  step_dummy(Central_Air) %>%
  step_interact(~ starts_with("Central_Air"):Year_Built) %>%
  prep(training = ames_train) %>%
  juice() %>%
  # select a few rows with different values
  slice(153:157)

## -----------------------------------------------------------------------------

ames_rec <- recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + 
                      Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area +
                      Central_Air + Longitude + Latitude,
                    data = ames_train) %>%
  step_log(Sale_Price, base = 10) %>%
  step_BoxCox(Lot_Area, Gr_Liv_Area) %>%
  step_other(Neighborhood, threshold = 0.05)  %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ starts_with("Central_Air"):Year_Built) %>%
  step_bs(Longitude, Latitude, options = list(df = 5))

## -----------------------------------------------------------------------------

ggplot(ames_train, 
       aes(x = Longitude, y = Sale_Price)) + 
  geom_point(alpha = .5) + 
  geom_smooth(
    method = "lm", 
    formula = y ~ splines::bs(x, 5), 
    se = FALSE
  ) + 
  scale_y_log10()

## -----------------------------------------------------------------------------

ggplot(ames_train, 
       aes(x = Latitude, y = Sale_Price)) + 
  geom_point(alpha = .5) + 
  geom_smooth(
    method = "lm", 
    formula = y ~ splines::bs(x, 5), 
    se = FALSE
  ) + 
  scale_y_log10()

## -----------------------------------------------------------------------------

ctrl <- trainControl(method = "cv", savePredictions = "final")

set.seed(5616)
lm_complex <- train(ames_rec, data = ames_train, method = "lm",  trControl = ctrl)

## -----------------------------------------------------------------------------

lm_complex %>%
  pluck("pred") %>%
  mutate(
    obs = 10^obs,
    pred = 10^pred
  ) %>%
ggplot(aes(x = obs, y = pred))  +
  geom_smooth(se = FALSE, col = "red") +
  geom_abline(lty = 2) +
  geom_point(alpha = .4)

## -----------------------------------------------------------------------------
## Extra/Backup Slides

data(segmentationData)

segmentationData <- segmentationData[, c("EqSphereAreaCh1", "PerimCh1", "Class", "Case")]
names(segmentationData)[1:2] <- paste0("Predictor", LETTERS[1:2])

segmentationData$Class <- factor(ifelse(segmentationData$Class == "PS", "One", "Two"))

bivariate_data_train <- subset(segmentationData, Case == "Train")
bivariate_data_test  <- subset(segmentationData, Case == "Test")

bivariate_data_train$Case <- NULL
bivariate_data_test$Case  <- NULL

ggplot(bivariate_data_test, 
       aes(x = PredictorA, 
           y = PredictorB,
           color = Class)) +
  geom_point(alpha = .3, cex = 1.5) + 
  theme(legend.position = "top") 

## -----------------------------------------------------------------------------

bivariate_rec <- recipe(Class ~ PredictorA + PredictorB, 
                         data = bivariate_data_train) %>%
  step_BoxCox(all_predictors())

bivariate_rec <- prep(bivariate_rec, training = bivariate_data_train, verbose = FALSE)

inverse_test <- bake(bivariate_rec, new_data = bivariate_data_test, everything())

ggplot(inverse_test, 
       aes(x = 1/PredictorA, 
           y = 1/PredictorB,
           color = Class)) +
  geom_point(alpha = .3, cex = 1.5) + 
  theme(legend.position = "top") 

## -----------------------------------------------------------------------------

ggplot(inverse_test, 
       aes(x = 1/PredictorA, 
           y = 1/PredictorB,
           color = Class)) +
  geom_point(alpha = .3, cex = 1.5) + 
  theme(legend.position = "top") 

## -----------------------------------------------------------------------------

bivariate_pca <- 
  recipe(Class ~ PredictorA + PredictorB, data = bivariate_data_train) %>%
  step_BoxCox(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_pca(all_predictors()) %>%
  prep(training = bivariate_data_test, verbose = FALSE)

pca_test <- bake(bivariate_pca, new_data = bivariate_data_test)

# Put components axes on the same range
pca_rng <- extendrange(c(pca_test$PC1, pca_test$PC2))

ggplot(pca_test, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(alpha = .2, cex = 1.5) + 
  theme(legend.position = "top") +
  xlim(pca_rng) + ylim(pca_rng) + 
  xlab("Principal Component 1") + ylab("Principal Component 2") 

## -----------------------------------------------------------------------------

ggplot(pca_test, 
       aes(x = PC1, 
           y = PC2,
           color = Class)) +
  geom_point(alpha = .2, cex = 1.5) + 
  theme(legend.position = "top") +
  xlim(pca_rng) + ylim(pca_rng) + 
  xlab("Principal Component 1") + 
  ylab("Principal Component 2") 

