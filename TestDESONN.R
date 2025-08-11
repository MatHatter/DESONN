source("DESONN.R")
# source("DESONN_20240629_v6.R")
# Initialize activation functions
# self$activation_functions <- vector("list", self$num_layers)
# self$activation_functions_learn <- vector("list", self$num_layers)

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
set.seed(111)
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
lr <- .121
lambda <- 0.0003
num_epochs <- 3
custom_scale <- .05
threshold <- .98
# ML_NN <- TRUE
ML_NN <- TRUE
train <- TRUE
learnOnlyTrainingRun <- FALSE
update_weights <- TRUE
update_biases <- TRUE
# hidden_sizes <- NULL
hidden_sizes <- c(32, 12)

activation_functions <- list(relu, relu, sigmoid) #hidden layers + output layer


activation_functions_learn <- list(relu, relu, sigmoid) #list(relu, bent_identity, sigmoid) #list("elu", bent_identity, "sigmoid") # list(NULL, NULL, NULL, NULL) #activation_functions #list("relu", "custom_activation", NULL, "relu")  #"custom_activation"
epsilon <- 1e-6
loss_type <- "CrossEntropy" #NULL #'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'

dropout_rates <- list(0.1) # NULL for output layer

dropout_rates_learn <- dropout_rates

num_layers <- length(hidden_sizes) + 1
output_size <- 1  # For binary classification

threshold_function <- tune_threshold_accuracy
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

viewTables <- FALSE
Losses_At_Optimal_Epoch_filenumber <- 3
writeTofiles <- FALSE
#########################################################################################################################

#########################################################################################################################


verbose <- TRUE
hyperparameter_grid_setup <- TRUE
reg_type = "L2" #"Max_Norm" #"L2" #Max_Norm" #"Group_Lasso" #"L1_L2"

# input_size <- 13 # This should match the actual number of features in your data
# hidden_size <- 2
loading_ensemble_1_run_ids <- FALSE
#########################################################################################################################
never_ran_flag <- TRUE

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

# === Step 1: Hyperparameter setup ===
hyperparameter_grid_setup <- FALSE  # Set to FALSE to run a single combo manually
## ====== REQUIRED PACKAGES (only if not already loaded) ======
suppressWarnings({
  if (!requireNamespace("dplyr", quietly = TRUE))   stop("Please install.packages('dplyr')")
  if (!requireNamespace("tidyr", quietly = TRUE))   stop("Please install.packages('tidyr')")
  if (!requireNamespace("purrr", quietly = TRUE))   stop("Please install.packages('purrr')")
})
library(dplyr)
library(tidyr)
library(purrr)

## ====== GLOBALS ======
results   <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)

`%||%` <- function(a,b) if (is.null(a) || length(a)==0) b else a

# You can set these BEFORE sourcing the file. Defaults below are only fallbacks.
num_networks  <- get0("num_networks", ifnotfound = 1)
num_ensembles <- get0("num_ensembles", ifnotfound = 1)

# ---- decide mode (single vs ensemble) ----
do_ensemble <- !(is.null(num_ensembles) || is.na(num_ensembles) || num_ensembles <= 0 || (num_networks %||% 1) <= 1)

# firstRun is only used to build the MAIN holder in ensemble mode
firstRun <- if (do_ensemble) TRUE else FALSE

j <- 1
ensembles <- list(main_ensemble = vector("list"), temp_ensemble = vector("list"))

metric_name <- "MSE"
viewTables  <- FALSE

## ====== HELPERS (needed in both modes) ======
is_real_serial <- function(x) is.character(x) && length(x) == 1 && !is.na(x) && nzchar(x)
.metric_minimize <- function(m) grepl("mse|mae|rmse|error|loss|quantization_error|topographic_error", tolower(m))

main_meta_var  <- function(i) sprintf("Ensemble_Main_1_model_%d_metadata", as.integer(i))
temp_meta_var  <- function(e,i) sprintf("Ensemble_Temp_%d_model_%d_metadata", as.integer(e), as.integer(i))

.resolve_metric_from_pm <- function(pm, metric_name) {
  if (is.null(pm)) return(NA_real_)
  if (is.list(pm) || is.environment(pm)) {
    val <- pm[[metric_name]]; if (!is.null(val)) return(as.numeric(val)[1])
    nm <- names(pm)
    if (!is.null(nm)) {
      hit <- which(tolower(nm) == tolower(metric_name))
      if (length(hit)) return(as.numeric(pm[[ nm[hit[1]] ]])[1])
    }
  }
  if (is.atomic(pm) && !is.null(names(pm))) {
    nm <- names(pm)
    if (metric_name %in% nm) return(as.numeric(pm[[metric_name]])[1])
    hit <- which(tolower(nm) == tolower(metric_name))
    if (length(hit)) return(as.numeric(pm[[ hit[1] ]])[1])
  }
  if (is.data.frame(pm)) {
    if (metric_name %in% names(pm)) return(as.numeric(pm[[metric_name]][1]))
    hit <- which(tolower(names(pm)) == tolower(metric_name))
    if (length(hit)) return(as.numeric(pm[[ hit[1] ]][1]))
    cn <- tolower(names(pm))
    if (all(c("metric","value") %in% cn)) {
      midx <- which(cn == "metric")[1]; vidx <- which(cn == "value")[1]
      rows <- which(tolower(pm[[midx]]) == tolower(metric_name))
      if (length(rows)) return(as.numeric(pm[[vidx]][ rows[1] ]))
    }
  }
  NA_real_
}

serial_to_meta_name <- function(serial) {
  if (!is_real_serial(serial)) return(NA_character_)
  p <- strsplit(serial, "\\.")[[1]]
  if (length(p) < 3) return(NA_character_)
  e <- suppressWarnings(as.integer(p[1])); i <- suppressWarnings(as.integer(p[3]))
  if (is.na(e) || is.na(i)) return(NA_character_)
  if (e == 1) sprintf("Ensemble_Main_%d_model_%d_metadata", e, i)
  else        sprintf("Ensemble_Temp_%d_model_%d_metadata", e, i)
}

get_metric_by_serial <- function(serial, metric_name) {
  var <- serial_to_meta_name(serial)
  if (nzchar(var) && exists(var, envir = .GlobalEnv)) {
    md <- get(var, envir = .GlobalEnv)
    return(.resolve_metric_from_pm(md$performance_metric, metric_name))
  }
  NA_real_
}

.collect_vals <- function(serials, metric_name) {
  if (!length(serials)) return(data.frame(serial = character(), value = numeric()))
  data.frame(
    serial = as.character(serials),
    value  = vapply(serials, get_metric_by_serial, numeric(1), metric_name),
    stringsAsFactors = FALSE
  )
}
## ====== Control panel flags ======
viewAllPlots <- FALSE  # TRUE shows all plots regardless of individual flags
verbose      <- FALSE  # TRUE enables additional plot/debug output

#SONN plots
accuracy_plot     <- TRUE   # show training accuracy/loss
saturation_plot   <- FALSE   # show output saturation
max_weight_plot   <- TRUE   # Yes max weight magnitude

#DESONN plots
performance_high_mean_plots <- TRUE
performance_low_mean_plots <- TRUE
relevance_high_mean_plots <- TRUE
relevance_low_mean_plots <- TRUE


## ====== SINGLE RUN (no logs, no lineage, no temp/prune/add) ======
if (!do_ensemble) {
  cat("Single-run mode → training one model, skipping all ensemble/logging.\n")
  
  main_model <- DESONN$new(
    num_networks    = 1L,
    input_size      = input_size,
    hidden_sizes    = hidden_sizes,
    output_size     = output_size,
    N               = N,
    lambda          = lambda,
    ensemble_number = 1L,
    ensembles       = NULL,
    ML_NN           = ML_NN,
    method          = init_method,
    custom_scale    = custom_scale
  )
  
  # Set per-SONN plotting flags before training (if SONNs already exist)
  if (length(main_model$ensemble)) {
    for (m in seq_along(main_model$ensemble)) {
      main_model$ensemble[[m]]$SONNModelViewPlotsConfig <- list(
        accuracy_plot   = isTRUE(accuracy_plot),
        saturation_plot = isTRUE(saturation_plot),
        max_weight_plot = isTRUE(max_weight_plot),
        viewAllPlots    = isTRUE(viewAllPlots),
        verbose         = isTRUE(verbose)
      )
    }
  }
  
  invisible(main_model$train(
    Rdata=X, labels=y, lr=lr, ensemble_number=1L, num_epochs=num_epochs,
    threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
    activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
    dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
    beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
    batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
    epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
    shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights,
    X_validation=X_validation, y_validation=y_validation, threshold_function=threshold_function, ML_NN=ML_NN,
    train=train, verbose=verbose
  ))
  
  # Optionally expose it the same way your ensemble code does:
  ensembles$main_ensemble[[1]] <- main_model
  
  # Quick optional summary (does not touch logs)
  if (!is.null(main_model$performance_metric)) {
    cat("\nSingle model performance_metric:\n"); print(main_model$performance_metric)
  }
  if (!is.null(main_model$relevance_metric)) {
    cat("\nSingle model relevance_metric:\n"); print(main_model$relevance_metric)
  }
}



## ====== ENSEMBLE MODE (everything below is unchanged but wrapped) ======
if (do_ensemble) {
  
  # -------- logs + lineage tables only needed in ensemble mode --------
  snapshot_main_serials_meta <- function() {
    vars <- grep("^Ensemble_Main_1_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
    if (!length(vars)) return(character())
    ord <- as.integer(sub("^Ensemble_Main_1_model_(\\d+)_metadata$", "\\1", vars))
    vars <- vars[order(ord)]
    vapply(vars, function(v) {
      md <- get(v, envir = .GlobalEnv)
      s  <- md$model_serial_num
      if (!is.null(s) && nzchar(as.character(s))) as.character(s) else NA_character_
    }, character(1))
  }
  
  get_temp_serials_meta <- function(j) {
    e <- j + 1
    vars <- grep(sprintf("^Ensemble_Temp_%d_model_\\d+_metadata$", e), ls(.GlobalEnv), value = TRUE)
    if (!length(vars)) return(character())
    ord <- as.integer(sub(sprintf("^Ensemble_Temp_%d_model_(\\d+)_metadata$", e), "\\1", vars))
    vars <- vars[order(ord)]
    vapply(vars, function(v) {
      md <- get(v, envir = .GlobalEnv)
      s  <- md$model_serial_num
      if (!is.null(s) && nzchar(as.character(s))) as.character(s) else NA_character_
    }, character(1))
  }
  
  if (is.null(ensembles$tables)) ensembles$tables <- list()
  
  if (is.null(ensembles$tables$movement_log)) {
    ensembles$tables$movement_log <- data.frame(
      iteration      = integer(),
      phase          = character(),
      slot           = integer(),
      role           = character(),
      serial         = character(),
      metric_name    = character(),
      metric_value   = numeric(),
      current_serial = character(),
      message        = character(),
      timestamp      = as.POSIXct(character()),
      stringsAsFactors = FALSE
    )
    rownames(ensembles$tables$movement_log) <- character(0)
  }
  if (is.null(ensembles$tables$movement_log_path)) {
    ensembles$tables$movement_log_path <- file.path(getwd(), sprintf("movement_log_%s.rds", format(Sys.time(), "%Y%m%d_%H%M%S")))
  }
  
  if (is.null(ensembles$tables$change_log)) {
    ensembles$tables$change_log <- data.frame(
      iteration    = integer(),
      role         = character(),
      serial       = character(),
      metric_name  = character(),
      metric_value = numeric(),
      message      = character(),
      timestamp    = as.POSIXct(character()),
      stringsAsFactors = FALSE
    )
  }
  if (is.null(ensembles$tables$change_log_path)) {
    ensembles$tables$change_log_path <- file.path(getwd(), sprintf("change_log_%s.rds", format(Sys.time(), "%Y%m%d_%H%M%S")))
  }
  
  if (is.null(ensembles$tables$lineage)) ensembles$tables$lineage <- vector("list", length = 0)
  
  init_lineage_if_needed <- function(main_serials) {
    n <- length(main_serials)
    if (length(ensembles$tables$lineage) < n) {
      length(ensembles$tables$lineage) <<- n
    }
    for (i in seq_len(n)) {
      if (is.null(ensembles$tables$lineage[[i]]) || !length(ensembles$tables$lineage[[i]])) {
        s <- main_serials[i]
        if (is_real_serial(s)) ensembles$tables$lineage[[i]] <<- c(as.character(s))
      }
    }
  }
  lineage_append <- function(slot, new_serial) {
    if (slot <= 0) return()
    if (is.null(ensembles$tables$lineage[[slot]])) ensembles$tables$lineage[[slot]] <<- character(0)
    ensembles$tables$lineage[[slot]] <<- c(ensembles$tables$lineage[[slot]], as.character(new_serial))
  }
  lineage_current <- function(slot) {
    v <- ensembles$tables$lineage[[slot]]
    if (!length(v)) return(NA_character_)
    v[length(v)]
  }
  ensure_lineage_columns <- function() {
    max_depth <- 0L
    for (v in ensembles$tables$lineage) max_depth <- max(max_depth, length(v))
    for (k in seq_len(max_depth)) {
      nm <- sprintf("lineage_%d", k)
      if (!nm %in% names(ensembles$tables$movement_log)) {
        if (nrow(ensembles$tables$movement_log))
          ensembles$tables$movement_log[[nm]] <- rep(NA_character_, nrow(ensembles$tables$movement_log))
        else
          ensembles$tables$movement_log[[nm]] <- character(0)
      }
    }
  }
  
  .align_rows_to_log <- function(rows) {
    logdf  <- ensembles$tables$movement_log
    target <- names(logdf)
    nlog   <- nrow(logdf)
    nrows  <- nrow(rows)
    add_missing_to_rows <- setdiff(target, names(rows))
    if (length(add_missing_to_rows)) {
      for (nm in add_missing_to_rows) {
        if (nm %in% c("iteration","slot")) {
          rows[[nm]] <- rep(NA_integer_, nrows)
        } else if (nm %in% c("metric_value")) {
          rows[[nm]] <- rep(NA_real_, nrows)
        } else if (nm %in% c("timestamp")) {
          rows[[nm]] <- as.POSIXct(rep(NA, nrows))
        } else {
          rows[[nm]] <- rep(NA_character_, nrows)
        }
      }
    }
    extra_cols <- setdiff(names(rows), target)
    if (length(extra_cols)) {
      for (nm in extra_cols) {
        col <- rows[[nm]]
        if (is.integer(col)) {
          ensembles$tables$movement_log[[nm]] <<- if (nlog) rep(NA_integer_, nlog) else integer(0)
        } else if (is.numeric(col)) {
          ensembles$tables$movement_log[[nm]] <<- if (nlog) rep(NA_real_, nlog) else numeric(0)
        } else if (inherits(col, "POSIXct")) {
          ensembles$tables$movement_log[[nm]] <<- if (nlog) as.POSIXct(rep(NA, nlog)) else as.POSIXct(character())
        } else {
          ensembles$tables$movement_log[[nm]] <<- if (nlog) rep(NA_character_, nlog) else character(0)
        }
      }
      target <- names(ensembles$tables$movement_log)
    }
    rows <- rows[, target, drop = FALSE]
    rownames(rows) <- NULL
    rows
  }
  
  append_movement_block_named <- function(rows, row_ids, fill_lineage_for_slots = NULL) {
    if (!NROW(rows)) return(invisible())
    ensure_lineage_columns()
    line_cols <- grep("^lineage_\\d+$", names(ensembles$tables$movement_log), value = TRUE)
    for (nm in line_cols) if (!(nm %in% names(rows))) rows[[nm]] <- NA_character_
    if (!("current_serial" %in% names(rows))) rows$current_serial <- NA_character_
    if (!is.null(fill_lineage_for_slots)) {
      fs <- rep_len(fill_lineage_for_slots, nrow(rows))
      for (idx in seq_len(nrow(rows))) {
        slot <- fs[idx]
        if (!is.na(slot) && slot > 0) {
          rows$current_serial[idx] <- lineage_current(slot)
          v <- ensembles$tables$lineage[[slot]] %||% character(0)
          if (length(v)) {
            for (k in seq_along(v)) {
              nm <- sprintf("lineage_%d", k)
              if (!(nm %in% names(rows))) rows[[nm]] <- NA_character_
              rows[[nm]][idx] <- v[k]
            }
          }
        }
      }
    }
    rows <- .align_rows_to_log(rows)
    rn <- as.character(row_ids)
    rn[rn == ""] <- paste0("ChangeRow_", seq_len(sum(rn == "")), "_", format(Sys.time(), "%H%M%S%OS3"))
    rownames(rows) <- rn
    ensembles$tables$movement_log <<- rbind(ensembles$tables$movement_log, rows)
    saveRDS(ensembles$tables$movement_log, ensembles$tables$movement_log_path)
  }
  
  append_change_rows <- function(rows) {
    if (!NROW(rows)) return(invisible())
    ensembles$tables$change_log <<- rbind(ensembles$tables$change_log, rows)
    saveRDS(ensembles$tables$change_log, ensembles$tables$change_log_path)
  }
  
  prune_network_from_ensemble <- function(ensembles, target_metric_name_worst) {
    minimize <- .metric_minimize(target_metric_name_worst)
    main_serials <- snapshot_main_serials_meta()
    main_vals    <- if (length(main_serials)) vapply(main_serials, get_metric_by_serial, numeric(1), target_metric_name_worst) else numeric(0)
    tbl <- data.frame(slot = seq_along(main_serials), serial = main_serials, value = main_vals, stringsAsFactors = FALSE)
    cat("\n==== PRUNE DIAGNOSTICS ====\n")
    cat("Metric:", target_metric_name_worst, " | Direction:", if (minimize) "MINIMIZE (lower better)" else "MAXIMIZE (higher better)", "\n")
    if (NROW(tbl)) print(tbl, row.names = FALSE) else { cat("(no main rows)\n"); return(NULL) }
    if (all(!is.finite(tbl$value))) { cat("No finite main values; abort prune.\n"); return(NULL) }
    worst_idx <- if (minimize) which.max(tbl$value) else which.min(tbl$value)
    worst_row <- tbl[worst_idx, , drop = FALSE]
    cat(sprintf("Chosen WORST serial = %s | value=%.6f\n", worst_row$serial, worst_row$value))
    list(
      removed_network   = ensembles$main_ensemble[[1]],
      updated_ensemble  = ensembles,
      worst_model_index = 1L,
      worst_serial      = worst_row$serial,
      worst_value       = as.numeric(worst_row$value)
    )
  }
  
  add_network_to_ensemble <- function(ensembles, target_metric_name_best, removed_network, ensemble_number, worst_model_index,
                                      removed_serial, removed_value) {
    minimize <- .metric_minimize(target_metric_name_best)
    temp_serials <- get_temp_serials_meta(ensemble_number)
    temp_vals    <- if (length(temp_serials)) vapply(temp_serials, get_metric_by_serial, numeric(1), target_metric_name_best) else numeric(0)
    
    cat("\n==== ADD DIAGNOSTICS ====\n")
    if (!length(temp_serials)) {
      cat("No TEMP serials; abort add.\n")
      return(list(updated_ensemble = ensembles$main_ensemble, removed_network = removed_network, added_network = NULL, added_serial = NA_character_))
    }
    
    temp_tbl <- data.frame(temp_serial = temp_serials, value = temp_vals, stringsAsFactors = FALSE)
    print(temp_tbl, row.names = FALSE)
    
    best_idx <- if (minimize) which.min(temp_tbl$value) else which.max(temp_tbl$value)
    best_row <- temp_tbl[best_idx, , drop = FALSE]
    
    removed_val <- removed_value
    if (!is.finite(removed_val) && is_real_serial(removed_serial)) {
      removed_val <- get_metric_by_serial(removed_serial, target_metric_name_best)
    }
    if (!is.finite(removed_val)) {
      removed_val <- .resolve_metric_from_pm(removed_network$performance_metric, target_metric_name_best)
    }
    
    cat(sprintf("Compare TEMP(best) %s=%.6f vs REMOVED %s on %s (%s better)\n",
                best_row$temp_serial, best_row$value,
                if (is.finite(removed_val)) sprintf("%.6f", removed_val) else "NA",
                target_metric_name_best, if (minimize) "lower" else "higher"))
    
    if (!is.finite(best_row$value) || !is.finite(removed_val)) {
      cat("→ INDETERMINATE (NA) — keep removed.\n")
      return(list(updated_ensemble = ensembles$main_ensemble, removed_network = removed_network, added_network = NULL, added_serial = NA_character_))
    }
    
    do_replace <- if (minimize) (best_row$value < removed_val) else (best_row$value > removed_val)
    if (!do_replace) {
      cat("→ KEEP REMOVED (TEMP not better).\n")
      return(list(updated_ensemble = ensembles$main_ensemble, removed_network = removed_network, added_network = NULL, added_serial = NA_character_))
    }
    
    parts <- strsplit(removed_serial, "\\.")[[1]]
    worst_slot <- suppressWarnings(as.integer(parts[3]))
    if (!is.finite(worst_slot) || is.na(worst_slot)) worst_slot <- 1L
    
    temp_parts <- strsplit(best_row$temp_serial, "\\.")[[1]]
    temp_e  <- suppressWarnings(as.integer(temp_parts[1]))
    temp_i  <- suppressWarnings(as.integer(temp_parts[3]))
    
    tvar <- temp_meta_var(temp_e, temp_i)
    mvar <- main_meta_var(worst_slot)
    
    if (!exists(tvar, envir = .GlobalEnv)) {
      cat("→ ERROR: TEMP metadata var not found:", tvar, "— aborting replace.\n")
      return(list(updated_ensemble = ensembles$main_ensemble, removed_network = removed_network, added_network = NULL, added_serial = NA_character_))
    }
    
    tmd <- get(tvar, envir = .GlobalEnv)
    tmd$model_serial_num <- best_row$temp_serial
    assign(mvar, tmd, envir = .GlobalEnv)
    
    cat(sprintf("→ REPLACED MAIN metadata slot %d: %s -> %s\n", worst_slot, removed_serial, best_row$temp_serial))
    lineage_append(worst_slot, best_row$temp_serial)
    
    list(
      updated_ensemble = ensembles$main_ensemble,
      removed_network  = removed_network,
      added_network    = NULL,
      added_serial     = best_row$temp_serial,
      worst_slot       = worst_slot
    )
  }
  
  
  ## ====== PHASE 1: build MAIN holder ======
  if (isTRUE(firstRun)) {
    cat("First run: initializing main_ensemble\n")
    main_model <- DESONN$new(
      num_networks    = max(1L, as.integer(num_networks)),
      input_size      = input_size,
      hidden_sizes    = hidden_sizes,
      output_size     = output_size,
      N               = N,
      lambda          = lambda,
      ensemble_number = 1,
      ensembles       = ensembles,
      ML_NN           = ML_NN,
      method          = init_method,
      custom_scale    = custom_scale
    )
    
    # Per‑SONN plotting flags
    if (length(main_model$ensemble)) {
      for (m in seq_along(main_model$ensemble)) {
        main_model$ensemble[[m]]$SONNModelViewPlotsConfig <- list(
          accuracy_plot   = isTRUE(accuracy_plot),
          saturation_plot = isTRUE(saturation_plot),
          max_weight_plot = isTRUE(max_weight_plot),
          viewAllPlots    = isTRUE(viewAllPlots),
          verbose         = isTRUE(verbose)
        )
      }
    }
    
    # Per‑DESONN plotting flags (MAIN)
    if (length(main_model$ensemble)) {
      for (m in seq_along(main_model$ensemble)) {
        main_model$ensemble[[m]]$DESONNModelViewPlotsConfig <- list(
          performance_high_mean_plots = isTRUE(performance_high_mean_plots),
          performance_low_mean_plots  = isTRUE(performance_low_mean_plots),
          relevance_high_mean_plots   = isTRUE(relevance_high_mean_plots),
          relevance_low_mean_plots    = isTRUE(relevance_low_mean_plots),
          viewAllPlots                = isTRUE(viewAllPlots),
          verbose                     = isTRUE(verbose)
        )
      }
    }
    
    invisible(main_model$train(
      Rdata=X, labels=y, lr=lr, ensemble_number=1, num_epochs=num_epochs,
      threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
      activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
      dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
      beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
      batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
      epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
      shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights,
      X_validation=X_validation, y_validation=y_validation, threshold_function=threshold_function, ML_NN=ML_NN,
      train=train, verbose=verbose
    ))
    ensembles$main_ensemble[[1]] <- main_model
    firstRun <- FALSE
  }
  
  ## ====== LOOP ======
  debug_prune <- TRUE
  num_ensembles <- as.integer(num_ensembles %||% 2L)
  
  for (j in 1:num_ensembles) {
    cat("\n— Iteration", j, ": build TEMP and run prune/add —\n")
    ts_iter <- Sys.time()
    
    main_before <- snapshot_main_serials_meta()
    init_lineage_if_needed(main_before)
    main_tbl_before <- .collect_vals(main_before, metric_name)
    rows_before <- if (length(main_before)) {
      data.frame(
        iteration = j, phase = "main_before", slot = seq_along(main_before),
        role = "main", serial = main_before, metric_name = metric_name,
        metric_value = main_tbl_before$value, current_serial = NA_character_,
        message = "", timestamp = ts_iter, stringsAsFactors = FALSE
      )
    } else data.frame()
    ids_before <- if (length(main_before)) vapply(seq_along(main_before), function(i) main_meta_var(i), character(1)) else character(0)
    append_movement_block_named(rows_before, ids_before, fill_lineage_for_slots = if (length(main_before)) seq_along(main_before) else NULL)
    
    ensembles$temp_ensemble <- vector("list", 1)
    temp_model <- DESONN$new(
      num_networks    = num_networks,
      input_size      = input_size,
      hidden_sizes    = hidden_sizes,
      output_size     = output_size,
      N               = N,
      lambda          = lambda,
      ensemble_number = j + 1,
      ensembles       = ensembles,
      ML_NN           = ML_NN,
      method          = init_method,
      custom_scale    = custom_scale
    )
    ensembles$temp_ensemble[[1]] <- temp_model
    
    # Per‑SONN plotting flags for TEMP
    if (length(temp_model$ensemble)) {
      for (m in seq_along(temp_model$ensemble)) {
        temp_model$ensemble[[m]]$SONNModelViewPlotsConfig <- list(
          accuracy_plot   = isTRUE(accuracy_plot),
          saturation_plot = isTRUE(saturation_plot),
          max_weight_plot = isTRUE(max_weight_plot),
          viewAllPlots    = isTRUE(viewAllPlots),
          verbose         = isTRUE(verbose)
        )
      }
    }
    
    # Per-DESONN plotting flags for TEMP
    if (length(temp_model$ensemble)) {
      for (m in seq_along(temp_model$ensemble)) {
        temp_model$ensemble[[m]]$DESONNModelViewPlotsConfig <- list(
          performance_high_mean_plots = isTRUE(performance_high_mean_plots),
          performance_low_mean_plots  = isTRUE(performance_low_mean_plots),
          relevance_high_mean_plots   = isTRUE(relevance_high_mean_plots),
          relevance_low_mean_plots    = isTRUE(relevance_low_mean_plots),
          viewAllPlots                = isTRUE(viewAllPlots),
          verbose                     = isTRUE(verbose)
        )
      }
    }
    
    invisible(temp_model$train(
      Rdata=X, labels=y, lr=lr, ensemble_number=j+1, num_epochs=num_epochs,
      threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
      activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
      dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
      beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
      batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
      epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
      shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights,
      X_validation=X_validation, y_validation=y_validation, threshold_function=threshold_function, ML_NN=ML_NN,
      train=train, verbose=verbose
    ))
    
    t_sers <- get_temp_serials_meta(j)
    t_vals <- if (length(t_sers)) vapply(t_sers, get_metric_by_serial, numeric(1), metric_name) else numeric(0)
    rows_temp <- if (length(t_sers)) {
      data.frame(
        iteration = j, phase = "temp", slot = NA_integer_,
        role = "temp", serial = t_sers, metric_name = metric_name,
        metric_value = t_vals, current_serial = NA_character_,
        message = "", timestamp = ts_iter, stringsAsFactors = FALSE
      )
    } else data.frame()
    ids_temp <- if (length(t_sers)) vapply(seq_along(t_sers), function(i) temp_meta_var(j+1, i), character(1)) else character(0)
    append_movement_block_named(rows_temp, ids_temp)
    
    pruned <- prune_network_from_ensemble(ensembles, metric_name)
    removed_serial <- NA_character_
    added_serial   <- NA_character_
    worst_slot     <- NA_integer_
    
    if (!is.null(pruned)) {
      added <- add_network_to_ensemble(
        ensembles               = pruned$updated_ensemble,
        target_metric_name_best = metric_name,
        removed_network         = pruned$removed_network,
        ensemble_number         = j,
        worst_model_index       = pruned$worst_model_index,
        removed_serial          = pruned$worst_serial,
        removed_value           = pruned$worst_value
      )
      ensembles$main_ensemble <- added$updated_ensemble
      removed_serial <- pruned$worst_serial
      added_serial   <- added$added_serial %||% NA_character_
      worst_slot     <- added$worst_slot %||% NA_integer_
      
      if (is_real_serial(removed_serial) && is_real_serial(added_serial)) {
        rrow <- data.frame(
          iteration=j, phase="removed", slot=worst_slot, role="removed",
          serial=removed_serial, metric_name=metric_name,
          metric_value=get_metric_by_serial(removed_serial, metric_name),
          current_serial=NA_character_,
          message=sprintf("%s replaced by %s", removed_serial, added_serial),
          timestamp=ts_iter, stringsAsFactors = FALSE
        )
        arow <- data.frame(
          iteration=j, phase="added", slot=worst_slot, role="added",
          serial=added_serial, metric_name=metric_name,
          metric_value=get_metric_by_serial(added_serial, metric_name),
          current_serial=NA_character_,
          message=sprintf("%s replaced by %s", removed_serial, added_serial),
          timestamp=ts_iter, stringsAsFactors = FALSE
        )
        append_movement_block_named(rrow, "", fill_lineage_for_slots = worst_slot)
        append_movement_block_named(arow, "", fill_lineage_for_slots = worst_slot)
        
        append_change_rows(rbind(
          data.frame(iteration=j, role="removed", serial=removed_serial, metric_name=metric_name,
                     metric_value=get_metric_by_serial(removed_serial, metric_name),
                     message=sprintf("%s replaced by %s", removed_serial, added_serial),
                     timestamp=ts_iter, stringsAsFactors = FALSE),
          data.frame(iteration=j, role="added",   serial=added_serial,   metric_name=metric_name,
                     metric_value=get_metric_by_serial(added_serial, metric_name),
                     message=sprintf("%s replaced by %s", removed_serial, added_serial),
                     timestamp=ts_iter, stringsAsFactors = FALSE)
        ))
      }
    }
    
    main_after <- snapshot_main_serials_meta()
    main_tbl_after <- .collect_vals(main_after, metric_name)
    rows_after <- if (length(main_after)) {
      data.frame(
        iteration = j, phase = "main_after", slot = seq_along(main_after),
        role = "main", serial = main_after, metric_name = metric_name,
        metric_value = main_tbl_after$value, current_serial = NA_character_,
        message = if (is_real_serial(removed_serial) && is_real_serial(added_serial))
          sprintf("%s replaced by %s", removed_serial, added_serial) else "",
        timestamp = ts_iter, stringsAsFactors = FALSE
      )
    } else data.frame()
    ids_after <- if (length(main_after)) vapply(seq_along(main_after), function(i) main_meta_var(i), character(1)) else character(0)
    append_movement_block_named(rows_after, ids_after, fill_lineage_for_slots = if (length(main_after)) seq_along(main_after) else NULL)
  }
}




if (saveToDisk) {
  
  # Save main ensemble and both temp ensembles used during the 2 iterations
  ensemble_results <- list(
    main_ensemble    = ensembles$main_ensemble,
    temp_ensemble_1  = temp_ensemble_1,
    temp_ensemble_2  = temp_ensemble_2
  )
  
  # Define the base file name
  base_file_name <- "ensemble_results"
  
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
  
  # Determine file name
  if (file.exists(paste0(base_file_name, ".rds"))) {
    file_name <- generate_new_file_name(base_file_name)
  } else {
    file_name <- paste0(base_file_name, ".rds")
  }
  
  # Save the full ensemble results
  saveRDS(ensemble_results, file_name)
  cat("Data saved to file:", file_name, "\n")
}
