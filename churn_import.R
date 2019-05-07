# https://www.kaggle.com/shrutimechlearn/churn-modelling

library(tidymodels)
library(readr)

churn <-
  read_csv("~/Downloads/Churn_Modelling.csv") %>%
  dplyr::select(-RowNumber, -CustomerId, -Surname) %>% 
  mutate(
    class = ifelse(Exited == 1, "yes", "no"),
    class = factor(class, levels = c("yes", "no")),
    IsActiveMember = ifelse(IsActiveMember == 1, "yes", "no"),
    HasCrCard = ifelse(HasCrCard == 1, "yes", "no")
    ) %>%
  mutate_if(
    is.character, as.factor
  ) %>% 
  dplyr::rename(
    credit_score = CreditScore,
    geography = Geography,
    gender = Gender,
    age = Age,
    tenure = Tenure, 
    balance = Balance,
    num_prod = NumOfProducts,
    credit_card = HasCrCard,
    active = IsActiveMember,
    salary = EstimatedSalary
  ) %>% 
  dplyr::select(-Exited)

save(churn, file = "churn.RData")

if (!interactive())
  q("no")
