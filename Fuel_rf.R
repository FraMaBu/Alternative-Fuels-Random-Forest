rm(list =c())

library(tidymodels)
library(themis)
library(ranger)
library(vip)


##### 
### speed up computation with parallel processing
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

#####
###Step 1 Loading and first factor engineering 
df = read.csv(file = "fuel_data.csv", stringsAsFactors = T)

df$fuel = as.factor(df$fuel)

str(df[,2:12])

######
###Step 2: Initial observations and splitting the data

df %>% count(fuel) %>% mutate(prop = n/sum(n))

set.seed(10)
data_split <- initial_split(df, strata = fuel, prob = 0.75)

# Create data frames for the two sets: (75% split)
train_df <- training(data_split)
test_df  <- testing(data_split)


train_df %>% count(fuel) %>% mutate(prop = n/sum(n))
test_df %>% count(fuel) %>% mutate(prop = n/sum(n))


######
###Step 3 Data Preprocessing

#Create the Recipe for the random forest (bal?nced and imbalanced)
rec_bal <- recipe(fuel ~ ., data = train_df) %>%
  update_role(imo.number, new_role = "ID") %>%
  step_knnimpute(all_predictors())  %>%
  step_zv(all_predictors()) %>%
  step_downsample(fuel)

#inspect the recipe
rec_bal


#####
###Step 4 build models with parsnip

#Define the rf with two hyperparameters to tune to find best model
rf_mod_tune <-  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")


#####
###Step 5 workflow

#Design workflows for logisitc and rf regressions
rf_wf <-  workflow() %>% add_model(rf_mod_tune) %>% add_recipe(rec_bal)

#####
###Step 6 Fit the untuned models with default hyperparameters and evaluate model performance

#resample the training data 10 times for cross validation and evaluating our
#model without using the training set
set.seed(10)
folds <- vfold_cv(train_df, v = 5)
folds


###Step7 Hypertune a model with grid search

#Set up the Grid to search for candidates, we choose 25 candidate models,
#then we tune the model with our 10 resamples of the data

#set metrics
cls_metrics <- metric_set(specificity, sensitivity, roc_auc, j_index)

set.seed(10)
rf_tuned = tune_grid(rf_wf, resamples = folds, grid = 50, metrics = cls_metrics,
                     control = control_grid(verbose = T))


rf_tuned %>% show_best(metric = "roc_auc",n = 10) 
rf_tuned %>% show_best(metric = "j_index",n = 5)

autoplot(rf_tuned)
# we can see that our tuned candidate performs slightly better than the model with standard parameters
# lets collect the parameters from the best candidate
# Note also from the plot that the roc is very robust to different parameter values

rf_best_params <- rf_tuned %>% select_best("j_index")
#our best model has the parameters 2 and 4

#Lets finalize our tune model with the best parameters
rf_model_final <- rf_mod_tune %>% finalize_model(rf_best_params)
rf_model_final

rf_wf_final = workflow() %>% add_model(rf_model_final) %>% add_recipe(rec_bal)

rf_fit_final = last_fit(rf_wf_final, data_split, metrics = cls_metrics)

collect_metrics(rf_fit_final)

#Explore the results with roc curve and vip chart
pred = collect_predictions(rf_fit_final)
pred

rf_fit_final %>% collect_predictions() %>% roc_curve(fuel , .pred_0) %>% autoplot()

rf_fit_final %>% pluck(".workflow", 1) %>% pull_workflow_fit() %>%  vip(num_features = 10)


