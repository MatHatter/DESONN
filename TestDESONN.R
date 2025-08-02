source("DESONN.R")
# source("DESONN_20240629_v6.R")
# Initialize activation functions
# self$activation_functions <- vector("list", self$num_layers)
# self$activation_functions_learn <- vector("list", self$num_layers)
library(caret)
###############################################################################
function(test1){
  # test1
  # Define input_size, output_size, and num_networks
  # Generate synthetic Rdata for training
  # ML_NN <- TRUE
  # ML_NN <- FALSE
  # numeric_columns <- NULL
  # normalize <- TRUE
  # # numeric_columns <- c('age', 'creatinine_phosphokinase', 'ejection_fraction',
  #                                                    # 'platelets', 'serum_creatinine', 'serum_sodium', 'time')
  # total_num_samples <- 1500
  # input_size <- 10
  # hidden_sizes <- c(1,5)
  # # dropout_rates <- c(0.1,0.2,0.3)
  # dropout_rates <- NULL
  # dropout_rates_learn <- dropout_rates
  # # hidden_sizes <- NULL
  # output_size <- 1
  # num_networks <- 1
  # if(!ML_NN){
  #     N <- input_size + output_size  # Multiplier for data generation (not directly applicable here)
  # }else{
  #     N <- input_size + sum(hidden_sizes) + output_size
  # }
  # threshold <- 0.98
  # num_samples <- if (!missing(total_num_samples)) total_num_samples else num_samples
  # # Define the number of samples for each set
  # num_validation_samples <- 200
  # num_test_samples <- 300
  # num_training_samples <- total_num_samples - numvalidation_samples - num_test_samples
  #
  # # Create a random permutation of row indices
  # indices <- sample(1:total_num_samples)
  #
  # # Split the indices into training, validation, and test sets
  # train_indices <- indices[1:num_training_samples]
  # validation_indices <- indices[(num_training_samples + 1):(num_training_samples + num_validation_samples)]
  # test_indices <- indices[(num_training_samples + num_validation_samples + 1):total_num_samples]
  #
  # # # Base seed
  # # base_seed <- 437
  # #
  # # # Generate datasets
  # # set.seed(base_seed)
  # X <- matrix(runif(total_num_samples * input_size), ncol = input_size)
  # y <- as.matrix(apply(X, 1, function(x) sum(sin(x)) + rnorm(1, sd = 0.1)))
  #
  # # # Create training, validation, and test sets
  # X_train <- X[train_indices, ]
  # y_train <- y[train_indices, ]
  #
  # # X_validation <- X[validation_indices, ]
  # # y_validation <- y[validation_indices, ]
  # #
  # # X_test <- X[test_indices, ]
  # # y_test <- y[test_indices, ]
  #
  # X <- as.matrix(X_train)
  # y <- as.matrix(y_train)
}

# # Define parameters
init_method <- "he" #variance_scaling" #glorot_uniform" #"orthogonal" #"orthogonal" #lecun" #xavier"
optimizer <- "adagrad" #"lamb" #ftrl #nag #"sgd" #NULL "rmsprop" #adam #sgd_momentum #lookahead #adagrad
lookahead_step <- 20
batch_normalize_data <- TRUE
shuffle_bn <- FALSE
gamma_bn <- .6
beta_bn <- .6
epsilon_bn <- 1e-6  # Increase for numerical stability
momentum_bn <- 0.9 # Improved convergence
is_training_bn <- TRUE
beta1 <- .9 # Standard Adam value
beta2 <- 0.8 # Slightly lower for better adaptabilit
lr <- .001
lambda <- 0.00025
num_epochs <- 3
custom_scale <- .05

# ML_NN <- TRUE
ML_NN <- TRUE
train <- TRUE
# hidden_sizes <- NULL
hidden_sizes <- c(64, 32)

activation_functions <- list(relu, relu, sigmoid) #hidden layers + output layer


activation_functions_learn <- list(relu, relu, sigmoid) #list(relu, bent_identity, sigmoid) #list("elu", bent_identity, "sigmoid") # list(NULL, NULL, NULL, NULL) #activation_functions #list("relu", "custom_activation", NULL, "relu")  #"custom_activation"
epsilon <- 1e-6
loss_type <- "CrossEntropy" #NULL #'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'

dropout_rates <- list(0.2) # NULL for output layer

dropout_rates_learn <- dropout_rates

num_layers <- length(hidden_sizes) + 1
output_size <- 1  # For binary classification
num_networks <- 1  # Number of trees in the Random Forest

# threshold <- 0.98  # Classification threshold (not directly used in Random Forest)

# Load the dataset
data <- read.csv("C:/Users/wfky1/Downloads/heart_failure_clinical_records.csv")

# Check for missing values
sum(is.na(data))


# Assuming there are no missing values, or handle them if they exist
# Convert categorical variables to factors if any
data <- data %>%
  mutate(across(where(is.character), as.factor))

input_columns <- setdiff(colnames(data), "DEATH_EVENT")
Rdata <- data[, input_columns]
labels <- data$DEATH_EVENT
input_size <- ncol(Rdata)

if(!ML_NN) {
  N <- input_size + output_size  # Multiplier for data generation (not directly applicable here)
} else {
  N <- input_size + sum(hidden_sizes) + output_size
}


library(readxl)

# Load file
Rdata_predictions <- read_excel("Rdata_predictions.xlsx", sheet = "Rdata_Predictions")

# Deceptively healthy flag
Rdata_predictions$deceptively_healthy <- ifelse(
  Rdata_predictions$serum_creatinine < quantile(Rdata_predictions$serum_creatinine, 0.10, na.rm = TRUE) &
    Rdata_predictions$age < quantile(Rdata_predictions$age, 0.15, na.rm = TRUE) &
    Rdata_predictions$creatinine_phosphokinase < quantile(Rdata_predictions$creatinine_phosphokinase, 0.20, na.rm = TRUE),
  1, 0
)



# Extract vectors
probs            <- suppressWarnings(as.numeric(Rdata_predictions[[15]][1:800]))
labels           <- suppressWarnings(as.numeric(Rdata_predictions[[13]][1:800]))
deceptive_flags  <- suppressWarnings(as.numeric(Rdata_predictions$deceptively_healthy[1:800]))
risky_flags      <- suppressWarnings(as.numeric(Rdata_predictions$risky[1:800]))  # <- You must have this column!

# NA checks
check_na <- function(vec, name) {
  if (any(is.na(vec))) {
    cat(paste0("X NA in '", name, "' at Excel rows:\n"))
    print(which(is.na(vec)) + 1)
    stop(paste("Fix NA in", name))
  }
}
check_na(probs, "probs")
check_na(labels, "labels")
check_na(deceptive_flags, "deceptive_flags")
check_na(risky_flags, "risky")

# Error vector
errors <- abs(probs - labels)

# Base weights
base_weights <- rep(1, length(labels))

# 👇 Apply rule-based scaling
# Boost deaths overall
base_weights[labels == 1] <- base_weights[labels == 1] * 2

# Strong boost for risky deaths
base_weights[labels == 1 & risky_flags == 1] <- base_weights[labels == 1 & risky_flags == 1] * log(20) * 4

# Optional: boost deceptive healthy deaths (the hard cases)
base_weights[labels == 1 & deceptive_flags == 1] <- base_weights[labels == 1 & deceptive_flags == 1] * 3


# Blend with error
raw_weights <- base_weights * errors
raw_weights <- pmin(pmax(raw_weights, 0.05), 23)

# Final adaptive weights
sample_weights <- 0.6 * base_weights + 0.4 * raw_weights
sample_weights <- sample_weights / mean(sample_weights)

stopifnot(length(sample_weights) == length(labels))
cat("✅ Sample weights created with 'risky' boost. Mean:", round(mean(sample_weights), 4), "\n")
sample_weights <- NULL

# Split the data into features (X) and target (y)
X <- data %>% dplyr::select(-DEATH_EVENT)
y <- data %>% dplyr::select(DEATH_EVENT)

# cols_to_remove <- names(p_values)[p_values > 0.05]
#
# # Remove those columns from the dataset
# X_reduced <- X %>% select(-all_of(cols_to_remove))

# columns_to_remove <- c("anaemia", "diabetes", "high_blood_pressure", "platelets", "serum_sodium", "sex", "smoking", "time")

# Remove specified columns
# data_reduced <- data %>% select(-all_of(columns_to_remove))
numeric_columns <- c('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
                     'serum_creatinine', 'serum_sodium', 'time')

# Split the data into features (X) and target (y)
# X <- data_reduced %>% select(-DEATH_EVENT)
y <- data %>% dplyr::select(DEATH_EVENT)
colname_y <- colnames(y)
# Define the number of samples for each set
set.seed(111)
total_num_samples <- nrow(data)
# Define num_samples
num_samples <- if (!missing(total_num_samples)) total_num_samples else num_samples
num_validation_samples <- 800
num_test_samples <- 800
num_training_samples <- total_num_samples - num_validation_samples - num_test_samples

# Create a random permutation of row indices
indices <- sample(1:total_num_samples)

# Split the indices into training, validation, and test sets
train_indices <- indices[1:num_training_samples]
validation_indices <- indices[(num_training_samples + 1):(num_training_samples + num_validation_samples)]
test_indices <- indices[(num_training_samples + num_validation_samples + 1):total_num_samples]

# Create training, validation, and test sets
X_train <- X[train_indices, ]
y_train <- y[train_indices, ]

X_validation <- X[validation_indices, ]
y_validation <- y[validation_indices, ]

X_test <- X[test_indices, ]
y_test <- y[test_indices, ]

# ===== APPLY LOG TRANSFORMATION =====
# Apply log1p to avoid issues with zero values (log1p(x) = log(1 + x))
# X_train$creatinine_phosphokinase <- pmin(X_train$creatinine_phosphokinase, 3000)
# X_validation$creatinine_phosphokinase <- pmin(X_validation$creatinine_phosphokinase, 3000)
# X_test$creatinine_phosphokinase <- pmin(X_test$creatinine_phosphokinase, 3000)



# $$$$$$$$$$$$$ Feature scaling without leakage (standardization first)
X_train_scaled <- scale(X_train)
center <- attr(X_train_scaled, "scaled:center")
scale_ <- attr(X_train_scaled, "scaled:scale")

X_validation_scaled <- scale(X_validation, center = center, scale = scale_)
X_test_scaled <- scale(X_test, center = center, scale = scale_)

# $$$$$$$$$$$$$ Further rescale to prevent exploding activations
max_val <- max(abs(X_train_scaled))
if (max_val > 1) {
  Rdata <- Rdata / max_val  # range will be roughly [-1, 1]
}

X_train_scaled <- X_train_scaled / max_val
X_validation_scaled <- X_validation_scaled / max_val
X_test_scaled <- X_test_scaled / max_val

# $$$$$$$$$$$$$ Sanity check of unscaled and scaled data
cat("=== Unscaled Rdata summary (X_train) ===\n")
print(summary(as.vector(X_train)))
cat("First 5 rows of unscaled X_train:\n")
print(X_train[1:5, 1:min(5, ncol(X_train))])

cat("=== Scaled Rdata summary (X_train_scaled) ===\n")
print(summary(as.vector(X_train_scaled)))
cat("First 5 rows of scaled X_train_scaled:\n")
print(X_train_scaled[1:5, 1:min(5, ncol(X_train_scaled))])

# $$$$$$$$$$$$$ Overwrite training matrix for model training
X <- as.matrix(X_train_scaled)
y <- as.matrix(y_train)

X_validation <- as.matrix(X_validation_scaled)
y_validation <- as.matrix(y_validation)

# X <- as.matrix(X_test_scaled)
# y <- as.matrix(y_test)




# X <- as.matrix(X_test)
colnames(y) <- colname_y

binary_flag <- is_binary(y)

# # Perform Random Forest-based feature selection on training data
# library(randomForest)
# 
# rf_data <- as.data.frame(X)
# rf_data$DEATH_EVENT <- as.factor(y)
# 
# set.seed(42)
# rf_model <- randomForest(DEATH_EVENT ~ ., data = rf_data, importance = TRUE)
# 
# # Compute feature importance and select features above median
# importance_scores <- importance(rf_model, type = 2)[, 1]  # MeanDecreaseGini
# threshold <- mean(importance_scores)
# selected_features <- names(importance_scores[importance_scores > threshold])
# 
# # Filter feature matrix to selected important features
# X <- as.matrix(rf_data[, selected_features, drop = FALSE])
# 
# # Update input size for neural network initialization
# input_size <- ncol(X)
# 
# numeric_columns <- intersect(numeric_columns, selected_features)



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#    ██████  ██████  ███    ██ ████████ ██████   ██████  ██          ██████   █████  ███    ██ ███████ ██  $$$$$$$$$$$$$$
#  ██      ██    ██ ████   ██    ██    ██   ██ ██    ██ ██          ██   ██ ██   ██ ████   ██ ██      ██   $$$$$$$$$$$$$$
# ██      ██    ██ ██ ██  ██    ██    ██████  ██    ██ ██          ██████  ███████ ██ ██  ██ █████   ██    $$$$$$$$$$$$$$
#██      ██    ██ ██  ██ ██    ██    ██   ██ ██    ██ ██          ██      ██   ██ ██  ██ ██ ██      ██     $$$$$$$$$$$$$$
#██████  ██████  ██   ████    ██    ██   ██  ██████  ███████     ██      ██   ██ ██   ████ ███████ ███████ $$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
showMeanBoxPlots <- FALSE
Losses_At_Optimal_Epoch_filenumber <- 3
writeTofiles <- FALSE
#########################################################################################################################
target_metric_name_best <- 'MSE' #<<-
target_metric_name_worst <- 'MSE' #<<-
metric_name <- 'MSE'
#########################################################################################################################

nruns <- 5
verbose <<- FALSE
hyperparameter_grid_setup <- TRUE
reg_type = "L2" #"Max_Norm" #"L2" #Max_Norm" #"Group_Lasso" #"L1_L2"

# input_size <- 13 # This should match the actual number of features in your data
# hidden_size <- 2
loading_ensemble_1_run_ids <- FALSE
#########################################################################################################################
never_ran_flag <- TRUE

results <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)
# if(hyperparameter_grid_setup){
#     loading_ensemble_1_run_ids <- TRUE #change back to false
# }else if(!hyperparameter_grid_setup){
# loading_ensemble_1_run_ids <- TRUE
# }
# results <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)
# Iterate over each row of the hyperparameter grid
# for (j in 1:nruns) {

plot_robustness <- FALSE
predict_models <- FALSE
use_loaded_weights <- FALSE
saveToDisk <- FALSE
Run1 <- TRUE
Run2 <- FALSE
Run3 <- FALSE
Run1.1 <- FALSE
Run1.2 <- FALSE
# === Step 1: Hyperparameter setup ===
hyperparameter_grid_setup <- FALSE  # Set to FALSE to run a single combo manually
results <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)
j <- 1
if (hyperparameter_grid_setup) {
  # Define grid of learning rates and regularization values
  lr_vals <- c(0.3)
  lambda_vals <- c(0.000028, 0.000011)
  hyper_grid <- expand.grid(lr = lr_vals, lambda = lambda_vals)
  
  # Loop through all combinations
  for (j in 1:nrow(hyper_grid)) {
    lr <- hyper_grid$lr[j]
    lambda <- hyper_grid$lambda[j]
    
    # === Initialize DESONN model ===
    if (ML_NN) {
      DESONN_model <- DESONN$new(
        num_networks = num_networks,
        input_size = input_size,
        hidden_sizes = hidden_sizes,
        output_size = output_size,
        N = N,
        lambda = lambda,
        ensemble_number = j,
        ML_NN = ML_NN,
        method = init_method,
        custom_scale = custom_scale
      )
    } else {
      DESONN_model <- DESONN$new(
        num_networks = num_networks,
        input_size = input_size,
        output_size = output_size,
        N = N,
        lambda = lambda,
        ensemble_number = j,
        ML_NN = ML_NN,
        method = init_method,
        custom_scale = custom_scale
      )
    }
    
    # === Train the model ===
    run_result <- DESONN_model$train(
      Rdata = X,
      labels = y,
      lr = lr,
      ensemble_number = j,
      num_epochs = num_epochs,
      threshold = threshold,
      reg_type = reg_type,
      numeric_columns = numeric_columns,
      activation_functions_learn = activation_functions_learn,
      activation_functions = activation_functions,
      dropout_rates_learn = dropout_rates_learn,
      dropout_rates = dropout_rates,
      optimizer = optimizer,
      beta1 = beta1,
      beta2 = beta2,
      epsilon = epsilon,
      lookahead_step = lookahead_step,
      batch_normalize_data = batch_normalize_data,
      gamma_bn = gamma_bn,
      beta_bn = beta_bn,
      epsilon_bn = epsilon_bn,
      momentum_bn = momentum_bn,
      is_training_bn = is_training_bn,
      shuffle_bn = shuffle_bn,
      loss_type = loss_type,
      sample_weights = sample_weights,
      X_validation = X_validation,
      y_validation = y_validation
    )
    
    # === Save result to results table ===
    results <- rbind(results, data.frame(
      lr = lr,
      lambda = lambda,
      accuracy = run_result$accuracy
    ))
    
    cat("Finished grid run", j, "of", nrow(hyper_grid), "-> Accuracy:", run_result$accuracy, "\n")
  }
  
} else {
  # === Manual run with provided lr1 and lambda1 ===
  lr <- lr
  lambda <- lambda
  
  if (ML_NN) {
    DESONN_model <- DESONN$new(
      num_networks = num_networks,
      input_size = input_size,
      hidden_sizes = hidden_sizes,
      output_size = output_size,
      N = N,
      lambda = lambda,
      ensemble_number = 1,
      ML_NN = ML_NN,
      method = init_method,
      custom_scale = custom_scale
    )
  } else {
    DESONN_model <- DESONN$new(
      num_networks = num_networks,
      input_size = input_size,
      output_size = output_size,
      N = N,
      lambda = lambda,
      ensemble_number = 1,
      ML_NN = ML_NN,
      method = init_method,
      custom_scale = custom_scale
    )
  }
  
  run_result <- DESONN_model$train(
    Rdata = X,
    labels = y,
    lr = lr,
    ensemble_number = 1,
    num_epochs = num_epochs,
    threshold = threshold,
    reg_type = reg_type,
    numeric_columns = numeric_columns,
    activation_functions_learn = activation_functions_learn,
    activation_functions = activation_functions,
    dropout_rates_learn = dropout_rates_learn,
    dropout_rates = dropout_rates,
    optimizer = optimizer,
    beta1 = beta1,
    beta2 = beta2,
    epsilon = epsilon,
    lookahead_step = lookahead_step,
    batch_normalize_data = batch_normalize_data,
    gamma_bn = gamma_bn,
    beta_bn = beta_bn,
    epsilon_bn = epsilon_bn,
    momentum_bn = momentum_bn,
    is_training_bn = is_training_bn,
    shuffle_bn = shuffle_bn,
    loss_type = loss_type,
    sample_weights = sample_weights,
    X_validation = X_validation,
    y_validation = y_validation
  )
  
  # Save manual run result
  results <- rbind(results, data.frame(
    lr = lr,
    lambda = lambda,
    accuracy = run_result$accuracy
  ))
  
  cat("Manual run -> Accuracy:", run_result$accuracy, "\n")
}
learnOnlyTrainingRun <- FALSE
update_weights <- TRUE
update_biases <- TRUE

# === Step 4: Save all results ===
saveRDS(results, file = "results.rds")
write.csv(results, file = "results.csv", row.names = FALSE)

if(saveToDisk){
  
  run_results_saved <- list(run_results_1_1)#, run_results_1_2, run_results_1_3, run_results_1_4, run_results_1_5)
  
  # Define the base file name
  base_file_name <- "run_results_saved"
  
  # Define a function to generate a new file name
  generate_new_file_name <- function(base_name) {
    i <- 1
    new_file_name <- paste0(base_name, "_", i, ".rds")
    while (file.exists(new_file_name)) {
      i <- i + 1
      new_file_name <- paste0(base_name, "_", i, ".rds")
    }
    return(new_file_name)
  }
  
  # Check if the file already exists
  if (file.exists(paste0(base_file_name, ".rds"))) {
    # Generate a new file name
    file_name <- generate_new_file_name(base_file_name)
  } else {
    # Use the base file name
    file_name <- paste0(base_file_name, ".rds")
  }
  
  # Save the data to an RDS file
  saveRDS(run_results_saved, file_name)
  
  cat("Data saved to file:", file_name, "\n")
}