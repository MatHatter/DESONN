source("DESONN.R")
source("utils/utils.R")
source("utils/bootstrap_metadata.R")
# source("DESONN_20240629_v6.R")
# Initialize activation functions
# self$activation_functions <- vector("list", self$num_layers)
# self$activation_functions_learn <- vector("list", self$num_layers)

###############################################################################
function(test1){ #fake function/ just for toggling
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
beta2 <- 0.8 # Slig1htly lower for better adaptabilit
lr <- .121
lr_decay_rate  <- 0.5
lr_decay_epoch <- 20
lr_min <- 1e-6
lambda <- 0.0003
num_epochs <- 117
validation_metrics <- TRUE
custom_scale <- .05

ML_NN <- TRUE

learnOnlyTrainingRun <- FALSE
update_weights <- TRUE
update_biases <- TRUE
# hidden_sizes <- NULL
hidden_sizes <- c(64, 32)

activation_functions <- list(relu, relu, sigmoid) #hidden layers + output layer


activation_functions_learn <- list(relu, relu, sigmoid) #list(relu, bent_identity, sigmoid) #list("elu", bent_identity, "sigmoid") # list(NULL, NULL, NULL, NULL) #activation_functions #list("relu", "custom_activation", NULL, "relu")  #"custom_activation"
epsilon <- 1e-6
loss_type <- "CrossEntropy" #NULL #'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'

dropout_rates <- list(0.1) # NULL for output layer


threshold_function <- tune_threshold_accuracy
# threshold <- .98  # Classification threshold (not directly used in Random Forest)

dropout_rates_learn <- dropout_rates

num_layers <- length(hidden_sizes) + 1
output_size <- 1  # For binary classification
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
# Rdata_predictions <- read_excel("Rdata_predictions.xlsx", sheet = "Rdata_Predictions")

# Deceptively healthy flag
# Rdata_predictions$deceptively_healthy <- ifelse(
#   Rdata_predictions$serum_creatinine < quantile(Rdata_predictions$serum_creatinine, 0.10, na.rm = TRUE) &
#     Rdata_predictions$age < quantile(Rdata_predictions$age, 0.15, na.rm = TRUE) &
#     Rdata_predictions$creatinine_phosphokinase < quantile(Rdata_predictions$creatinine_phosphokinase, 0.20, na.rm = TRUE),
#   1, 0
# )



# Extract vectors
# probs            <- suppressWarnings(as.numeric(Rdata_predictions[[15]][1:800]))
# labels           <- suppressWarnings(as.numeric(Rdata_predictions[[13]][1:800]))
# deceptive_flags  <- suppressWarnings(as.numeric(Rdata_predictions$deceptively_healthy[1:800]))
# risky_flags      <- suppressWarnings(as.numeric(Rdata_predictions$risky[1:800]))  # <- You must have this column!

# NA checks
# check_na <- function(vec, name) {
#   if (any(is.na(vec))) {
#     cat(paste0("X NA in '", name, "' at Excel rows:\n"))
#     print(which(is.na(vec)) + 1)
#     stop(paste("Fix NA in", name))
#   }
# }
# check_na(probs, "probs")
# check_na(labels, "labels")
# check_na(deceptive_flags, "deceptive_flags")
# check_na(risky_flags, "risky")
# 
# # Error vector
# errors <- abs(probs - labels)
# 
# # Base weights
# base_weights <- rep(1, length(labels))
# 
# # 👇 Apply rule-based scaling
# # Boost deaths overall
# base_weights[labels == 1] <- base_weights[labels == 1] * 2
# 
# # Strong boost for risky deaths
# base_weights[labels == 1 & risky_flags == 1] <- base_weights[labels == 1 & risky_flags == 1] * log(20) * 4
# 
# # Optional: boost deceptive healthy deaths (the hard cases)
# base_weights[labels == 1 & deceptive_flags == 1] <- base_weights[labels == 1 & deceptive_flags == 1] * 3
# 
# 
# # Blend with error
# raw_weights <- base_weights * errors
# raw_weights <- pmin(pmax(raw_weights, 0.05), 23)
# 
# # Final adaptive weights
# sample_weights <- 0.6 * base_weights + 0.4 * raw_weights
# sample_weights <- sample_weights / mean(sample_weights)
# 
# stopifnot(length(sample_weights) == length(labels))
# cat("✅ Sample weights created with 'risky' boost. Mean:", round(mean(sample_weights), 4), "\n")
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
numeric_columns <- c('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',  'serum_creatinine', 'serum_sodium', 'time')

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


## =========================
## DESONN Runner – Modes
## =========================
## SCENARIO A: Single-run only (no ensemble, ONE model)
# do_ensemble         <- FALSE
# num_networks        <- 1L
# num_temp_iterations <- 0L   # ignored when do_ensemble = FALSE
#
## SCENARIO B: Single-run, MULTI-MODEL (no ensemble)
# do_ensemble         <- FALSE
# num_networks        <- 3L          # e.g., run 5 models in one DESONN instance
# num_temp_iterations <- 0L
#
## SCENARIO C: Main ensemble only (no TEMP/prune-add)
# do_ensemble         <- TRUE
# num_networks        <- 3L          # example main size
# num_temp_iterations <- 0L
#
## SCENARIO D: Main + TEMP iterations (prune/add enabled)
do_ensemble         <- TRUE
num_networks        <- 5L          # example main size
num_temp_iterations <- 2L          # MAIN + 1 TEMP pass (set higher for more TEMP passes)
#
## You can set the above variables BEFORE sourcing this file. The defaults below are fallbacks.

## ====== GLOBALS ======
results   <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)

`%||%` <- function(a,b) if (is.null(a) || length(a)==0) b else a

# You can set these BEFORE sourcing the file. Defaults below are only fallbacks.
num_networks        <- get0("num_networks", ifnotfound = 1L)
num_temp_iterations <- get0("num_temp_iterations", ifnotfound = 0L)   # 0 = MAIN only (no TEMP)
do_ensemble         <- get0("do_ensemble", ifnotfound = FALSE)         # TRUE ⇒ run MAIN (+ TEMP if >0)

# firstRun is only used to build the MAIN holder in ensemble mode
firstRun <- TRUE

j <- 1L
ensembles <- list(main_ensemble = vector("list"), temp_ensemble = vector("list"))

metric_name <- "MSE"
viewTables  <- FALSE


## ====== Control panel flags ======
viewAllPlots <- FALSE  # TRUE shows all plots regardless of individual flags
verbose      <- FALSE  # TRUE enables additional plot/debug output

# SONN plots
accuracy_plot     <- FALSE    # show training accuracy/loss
saturation_plot   <- FALSE   # show output saturation
max_weight_plot   <- FALSE    # show max weight magnitude

# DESONN plots
performance_high_mean_plots <- FALSE
performance_low_mean_plots  <- FALSE
relevance_high_mean_plots   <- FALSE
relevance_low_mean_plots    <- FALSE

# =========================
# Session bootstrap & flags (TRAIN | PREDICT:STATELESS | PREDICT:STATEFUL)
# =========================
# =========================
# Phase 0 — toggles & mode
# =========================

# Toggle — set TRUE to export RDS metadata and (optionally) clear env, then stop.
# prepare_disk_only <- TRUE
prepare_disk_only <- get0("prepare_disk_only", ifnotfound = FALSE)  # one-shot RDS export helper

# Flag specific to disk-only prepare / selection
prepare_disk_only_FROM_RDS <- TRUE
prepare_disk_only_FROM_RDS <- get0("prepare_disk_only_FROM_RDS", ifnotfound = FALSE)
# Modes:
# "train"
#   "predict:stateless"
#   "predict:stateful"
MODE <- "predict:stateless"
MODE <- get0("MODE", ifnotfound = "train")

# Prediction scope (only used when MODE starts with "predict:")
PREDICT_SCOPE <- "all"
PREDICT_SCOPE    <- get0("PREDICT_SCOPE",    ifnotfound = "one")
PREDICT_SELECTOR <- get0("PREDICT_SELECTOR", ifnotfound = "by_metric")
TARGET_METRIC    <- get0("TARGET_METRIC",    ifnotfound = "accuracy")

# Optional candidate filters
KIND_FILTER      <- get0("KIND_FILTER",      ifnotfound = c("Main","Temp"))
ENS_FILTER       <- get0("ENS_FILTER",       ifnotfound = NULL)
MODEL_FILTER     <- get0("MODEL_FILTER",     ifnotfound = NULL)

# (Kept only for one-shot export convenience)
BM_NAME_HINT    <- get0("BM_NAME_HINT", ifnotfound = NULL)
BM_PREFER_KIND  <- get0("BM_PREFER_KIND",  ifnotfound = c("Main","Temp"))
BM_PREFER_ENS   <- get0("BM_PREFER_ENS",   ifnotfound = c(0L, 1L))
BM_PREFER_MODEL <- get0("BM_PREFER_MODEL", ifnotfound = 1L)   # legacy keeps your old default

# ---- NEW: prepare-disk choice flags (adds 3-choice behavior) ----------
# Choices: "first" | "all" | "pick"
PREPARE_DISK_CHOICE <- get0("PREPARE_DISK_CHOICE", ifnotfound = "all")
# Clear environment after saving (matches prior behavior of bm_prepare_disk_only):
PREPARE_CLEAR_ENV   <- get0("PREPARE_CLEAR_ENV",   ifnotfound = TRUE)

# Default metadata dir (if not already set elsewhere)
.BM_DIR <- get0(".BM_DIR", ifnotfound = "artifacts")

# =========================================
# Phase 1 — one-shot export & (optional) clean
# =========================================
if (isTRUE(prepare_disk_only)) {
  
  # Snapshot the clear flag BEFORE any helper that might wipe env
  .CLEAR_AFTER <- isTRUE(get0("PREPARE_CLEAR_ENV", ifnotfound = TRUE))
  saved_names  <- character(0)  # will hold names of models we want to keep in env
  
  # -------------------------------
  # A) Legacy exact-name path (BM_NAME_HINT)
  # -------------------------------
  if (!is.null(BM_NAME_HINT) && nzchar(BM_NAME_HINT)) {
    rds_path <- bm_prepare_disk_only(
      name_hint    = BM_NAME_HINT,
      prefer_kind  = BM_PREFER_KIND,
      prefer_ens   = BM_PREFER_ENS,
      prefer_model = BM_PREFER_MODEL,
      dir          = .BM_DIR
    )
    cat(sprintf("[prepare_disk_only] Saved metadata to: %s\n", rds_path))
    
    # record the object name so we can keep it in env after clearing
    saved_names <- unique(c(saved_names, sub("\\.rds$", "", basename(rds_path))))
    
  } else {
    
    # -------------------------------
    # B) Choice-based export ("first" | "all" | "pick")
    #     — DO NOT clear inside helper
    # -------------------------------
    rds_paths <- tryCatch(
      bm_prepare_disk_by_choice(
        choice       = PREPARE_DISK_CHOICE,   # "first" | "all" | "pick"
        kind_filter  = KIND_FILTER,
        ens_filter   = ENS_FILTER,
        model_filter = MODEL_FILTER,
        dir          = .BM_DIR,
        clear_env    = FALSE                  # ← prevent premature wipe
      ),
      error = function(e) {
        cat("\n[prepare_disk_only] ERROR during bm_prepare_disk_by_choice:\n")
        message(e)
        stop(e)
      }
    )
    
    if (!length(rds_paths)) {
      cat("[prepare_disk_only] No candidates saved (zero-length result).\n")
    } else {
      cat("[prepare_disk_only] Saved metadata files:\n")
      for (p in unique(rds_paths)) cat("  - ", p, "\n", sep = "")
    }
    
    # record all object names so we can keep them in env after clearing
    saved_names <- unique(c(saved_names, sub("\\.rds$", "", basename(rds_paths))))
  }
  
  # -------------------------------
  # C) Clear, but KEEP the saved models in the Environment
  # -------------------------------
  if (.CLEAR_AFTER) {
    if (length(saved_names)) {
      if (exists("bm_clear_env_except", inherits = TRUE)) {
        bm_clear_env_except(keep_names = saved_names)
      } else {
        all_objs <- ls(envir = .GlobalEnv, all.names = FALSE)
        to_rm    <- setdiff(all_objs, saved_names)
        if (length(to_rm)) rm(list = to_rm, envir = .GlobalEnv)
      }
      gc()
      cat("[prepare_disk_only] Environment cleared (kept saved models). End of phase-1.\n")
    } else {
      # nothing saved? fall back to full clear
      rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)
      gc()
      cat("[prepare_disk_only] Environment cleared. End of phase-1.\n")
    }
  }
  
  # Final stop/quit
  if (exists(".hard_stop", mode = "function")) .hard_stop() else stop("prepare_disk_only done.")
}

# Derive boolean once from MODE (no redundancy)
train <- identical(MODE, "train")

# ======================= PREDICT-ONLY (single knob: PREDICT_SCOPE) =======================
if (!train) {
  cat("Predict-only mode (train == FALSE).\n")
  
  # -------------------------------
  # Config / scope
  # -------------------------------
  predict_mode  <- if (exists("MODE", inherits = TRUE) && identical(MODE, "predict:stateful")) "stateful" else "stateless"
  PREDICT_SCOPE <- get0("PREDICT_SCOPE", ifnotfound = "all", inherits = TRUE)
  PREDICT_SCOPE <- match.arg(PREDICT_SCOPE, c("one","group-best","all","pick","single"))
  if (identical(PREDICT_SCOPE, "single")) PREDICT_SCOPE <- "one"
  PICK_INDEX <- as.integer(get0("PICK_INDEX", ifnotfound = 1L, inherits = TRUE))
  
  TARGET_METRIC <- get0("TARGET_METRIC", ifnotfound = "accuracy", inherits = TRUE)
  KIND_FILTER   <- get0("KIND_FILTER",   ifnotfound = c("Main","Temp"), inherits = TRUE)
  ENS_FILTER    <- get0("ENS_FILTER",    ifnotfound = NULL, inherits = TRUE)
  MODEL_FILTER  <- get0("MODEL_FILTER",  ifnotfound = NULL, inherits = TRUE)
  
  PREDICT_ONLY_FROM_RDS <- get0("PREDICT_ONLY_FROM_RDS", ifnotfound = TRUE, inherits = TRUE)
  
  # -------------------------------
  # Build candidate list from env/RDS
  # -------------------------------
  df_all <- bm_list_all()
  if (!nrow(df_all)) stop("No models found in env/RDS (bm_list_all() returned 0 rows).")
  
  # apply filters
  if (length(KIND_FILTER))    df_all <- df_all[df_all$kind  %in% KIND_FILTER, , drop = FALSE]
  if (!is.null(ENS_FILTER))   df_all <- df_all[df_all$ens   %in% ENS_FILTER,  , drop = FALSE]
  if (!is.null(MODEL_FILTER)) df_all <- df_all[df_all$model %in% MODEL_FILTER,, drop = FALSE]
  if (!nrow(df_all)) stop("No candidates after applying KIND/ENS/MODEL filters.")
  
  # rank by metric
  df_all$metric_value <- vapply(seq_len(nrow(df_all)), function(i) {
    meta_i <- tryCatch(bm_select_exact(df_all$kind[i], df_all$ens[i], df_all$model[i]), error = function(e) NULL)
    if (is.null(meta_i)) return(NA_real_)
    .get_metric_from_meta(meta_i, TARGET_METRIC)
  }, numeric(1))
  minimize <- .metric_minimize(TARGET_METRIC)
  ok <- is.finite(df_all$metric_value)
  df_ranked <- if (any(ok)) {
    ord <- if (minimize) order(df_all$metric_value) else order(df_all$metric_value, decreasing = TRUE)
    df_all[ord, , drop = FALSE]
  } else df_all
  
  # --- DE-DUPLICATE same (kind, ens, model), prefer RDS unless flag set ---
  if (!"source" %in% names(df_ranked)) df_ranked$source <- NA_character_
  if (isTRUE(PREDICT_ONLY_FROM_RDS)) {
    source_priority <- c("rds","file","disk","env","memory","workspace")
  } else {
    source_priority <- c("env","memory","workspace","rds","file","disk")
  }
  df_ranked$._src_rank <- match(tolower(df_ranked$source), source_priority)
  df_ranked$._src_rank[is.na(df_ranked$._src_rank)] <- 99L
  ord <- with(df_ranked, order(kind, ens, model, ._src_rank, na.last = TRUE))
  df_ranked <- df_ranked[ord, , drop = FALSE]
  dups <- duplicated(df_ranked[, c("kind","ens","model")])
  if (any(dups)) {
    kept   <- df_ranked[!dups, , drop = FALSE]
    tossed <- df_ranked[ dups, , drop = FALSE]
    cat(sprintf("De-duped %d duplicate entries (preferring %s).\n",
                sum(dups), if (isTRUE(PREDICT_ONLY_FROM_RDS)) "RDS" else "env"))
    df_ranked <- kept
  }
  df_ranked$._src_rank <- NULL
  
  # Scope mapping
  scope_rows <- switch(PREDICT_SCOPE,
                       "one"        = df_ranked[1, , drop = FALSE],
                       "group-best" = {
                         out <- list()
                         m1 <- subset(df_ranked, kind=="Main" & ens==1)
                         if (nrow(m1)) out[[length(out)+1]] <- m1[1, , drop = FALSE]
                         te <- subset(df_ranked, kind=="Temp")
                         if (nrow(te)) for (e in sort(unique(te$ens))) out[[length(out)+1]] <- te[te$ens==e, , drop = FALSE][1, , drop = FALSE]
                         do.call(rbind, out)
                       },
                       "all"        = df_ranked,
                       "pick"       = {
                         if (PICK_INDEX < 1L || PICK_INDEX > nrow(df_ranked))
                           stop(sprintf("PICK_INDEX=%d out of range [1..%d]", PICK_INDEX, nrow(df_ranked)))
                         df_ranked[PICK_INDEX, , drop = FALSE]
                       },
                       df_ranked[1, , drop = FALSE]
  )
  if (!nrow(scope_rows)) stop("No rows selected for prediction after scope resolution.")
  
  # -------------------------------
  # Common X/y fallback
  # -------------------------------
  X_common <- get0("X_validation", envir = .GlobalEnv, inherits = TRUE, ifnotfound =
                     get0("X_test",       envir = .GlobalEnv, inherits = TRUE, ifnotfound =
                            get0("X",            envir = .GlobalEnv, inherits = TRUE, ifnotfound = NULL)))
  y_common <- get0("y_validation", envir = .GlobalEnv, inherits = TRUE, ifnotfound =
                     get0("y_test",       envir = .GlobalEnv, inherits = TRUE, ifnotfound =
                            get0("y",            envir = .GlobalEnv, inherits = TRUE, ifnotfound = NULL)))
  y_common <- .normalize_y(y_common)
  
  # -------------------------------
  # Helpers
  # -------------------------------
  .digest_or <- function(obj) if (requireNamespace("digest", quietly=TRUE)) tryCatch(digest::digest(obj, algo="xxhash64"), error=function(e) "NA") else "digest_pkg_missing"
  .weight_signature <- function(meta) {
    w <- tryCatch({
      if (!is.null(meta$best_model_metadata$best_weights_record))
        meta$best_model_metadata$best_weights_record else meta$best_weights_record
    }, error=function(e) NULL)
    if (!is.null(w)) return(.digest_or(w))
    .digest_or(meta)
  }
  
  # -------------------------------
  # Predict per model
  # -------------------------------
  results <- vector("list", length = nrow(scope_rows))
  pred_sigs <- character(nrow(scope_rows))
  
  for (i in seq_len(nrow(scope_rows))) {
    kind  <- as.character(scope_rows$kind[i]); ens <- as.integer(scope_rows$ens[i]); model <- as.integer(scope_rows$model[i])
    varname <- sprintf("Ensemble_%s_%d_model_%d_metadata", kind, ens, model)
    meta <- if (exists(varname, envir=.GlobalEnv)) get(varname, envir=.GlobalEnv) else bm_select_exact(kind, ens, model)
    if (is.null(meta)) { warning(sprintf("Skipping %s/%d/%d: no metadata.", kind, ens, model)); next }
    
    xin <- .choose_X_from_meta(meta); yin <- .choose_y_from_meta(meta)
    Xi <- if (!is.null(xin)) xin$X else X_common
    yi <- if (!is.null(yin)) .normalize_y(yin$y) else y_common
    if (is.null(Xi)) { warning(sprintf("Skipping %s/%d/%d: no X.", kind, ens, model)); next }
    
    Xi <- .align_by_names_safe(Xi, X_common)
    Xi <- .apply_scaling_if_any(as.matrix(Xi), meta)
    
    pred_raw <- .safe_run_predict(X=Xi, meta=meta, model_index=1L, ML_NN=TRUE)
    P <- .as_pred_matrix(pred_raw)
    
    wsig <- .weight_signature(meta)
    psig <- .digest_or(round(P[seq_len(min(nrow(P),2000)),,drop=FALSE],6))
    pred_sigs[i] <- psig
    
    cat(sprintf("→ Model(kind=%s, ens=%d, model=%d) preds=%dx%d\n", kind, ens, model, nrow(P), ncol(P)))
    cat(sprintf("   · serial=%s | w_sig=%s | pred_sig=%s\n", as.character(meta$model_serial_num %||% sprintf("%d.%d.%d", ens,0L,model)), wsig, psig))
    
    # ---- Metrics
    acc_val <- prec_val <- rec_val <- f1_val <- NA_real_
    tp <- fp <- fn <- tn <- NA_integer_
    tuned_thr <- tuned_acc <- tuned_prec <- tuned_rec <- tuned_f1 <- NA_real_
    
    if (!is.null(yi) && nrow(P)>0) {
      n <- min(nrow(P), length(yi))
      if (nrow(P)!=length(yi)) { warning(sprintf("[%s/%d/%d] preds=%d labels=%d trunc→%d",kind,ens,model,nrow(P),length(yi),n)); P<-P[1:n,,drop=FALSE]; yi<-yi[1:n] }
      if (ncol(P)==1L) { y_pred <- as.integer(as.numeric(P[,1])>=0.5) } else { y_pred <- as.integer((max.col(P)-1L)>0L) }
      y_true01 <- as.integer(yi>0)
      acc_val <- mean(y_pred==y_true01); tp<-sum(y_pred==1&y_true01==1); fp<-sum(y_pred==1&y_true01==0); fn<-sum(y_pred==0&y_true01==1); tn<-sum(y_pred==0&y_true01==0)
      prec_val <- tp/(tp+fp+1e-8); rec_val<-tp/(tp+fn+1e-8); f1_val<-2*prec_val*rec_val/(prec_val+rec_val+1e-8)
      
      if (is.function(get0("accuracy", inherits=TRUE))) {
        cat("   · accuracy(): baseline @0.5\n")
        tryCatch(accuracy(SONN=NULL,Rdata=Xi,labels=yi,predicted_output=P,verbose=TRUE), error=function(e){cat("     (err) ");message(e)})
      }
      
      if (is.function(get0("accuracy_tuned", inherits=TRUE))) {
        cat("   · accuracy_tuned(): searching threshold\n")
        tuning_metric <- tolower(get0("TARGET_METRIC", ifnotfound="accuracy", inherits=TRUE))
        allowed <- c("accuracy","f1","precision","recall","macro_f1","macro_precision","macro_recall")
        if (!tuning_metric %in% allowed) tuning_metric <- "accuracy"
        tuned <- tryCatch(
          accuracy_tuned(SONN=NULL,Rdata=Xi,labels=yi,predicted_output=P,
                         metric_for_tuning=tuning_metric,threshold_grid=seq(0.05,0.95,by=0.01),verbose=TRUE),
          error=function(e){cat("     (err) ");message(e);NULL}
        )
        if (!is.null(tuned)) {
          `%||%` <- function(x,y) if (is.null(x)) y else x
          tuned_acc  <- 100*as.numeric(tuned$accuracy %||% NA_real_)
          tuned_prec <- as.numeric(tuned$precision %||% NA_real_)
          tuned_rec  <- as.numeric(tuned$recall %||% NA_real_)
          tuned_f1   <- as.numeric(tuned$F1 %||% NA_real_)
          tuned_thr  <- as.numeric((tuned$details %||% list())$best_threshold %||% NA_real_)
        }
      }
    }
    
    results[[i]] <- list(kind=kind, ens=ens, model=model,
                         data_source=if (!is.null(xin)) paste0("embedded:",xin$tag) else "fallback:common",
                         n_pred_rows=nrow(P), accuracy=acc_val, precision=prec_val, recall=rec_val, f1=f1_val,
                         tuned_threshold=tuned_thr, tuned_accuracy_percent=tuned_acc,
                         tuned_precision=tuned_prec, tuned_recall=tuned_rec, tuned_f1=tuned_f1,
                         tp=tp,fp=fp,fn=fn,tn=tn)
  }
  
  results <- Filter(Negate(is.null), results)
  if (!length(results)) stop("No successful predictions.")
  
  rows <- lapply(results, function(z) data.frame(
    kind=z$kind, ens=z$ens, model=z$model, data_source=z$data_source,
    n_pred_rows=z$n_pred_rows, accuracy=z$accuracy, precision=z$precision,
    recall=z$recall, f1=z$f1, tuned_threshold=z$tuned_threshold,
    tuned_accuracy_percent=z$tuned_accuracy_percent, tuned_precision=z$tuned_precision,
    tuned_recall=z$tuned_recall, tuned_f1=z$tuned_f1,
    tp=z$tp, fp=z$fp, fn=z$fn, tn=z$tn, stringsAsFactors=FALSE))
  results_df <- do.call(rbind, rows); rownames(results_df)<-NULL
  assign("PREDICT_RESULTS_TABLE", results_df, .GlobalEnv)
  
  cat(sprintf("[predict] MODE=%s | SCOPE=%s | METRIC=%s | models=%d\n", predict_mode,PREDICT_SCOPE,TARGET_METRIC,nrow(results_df)))
  print(results_df, digits=6)
  if (length(pred_sigs)) { cat("Prediction signature counts:\n"); print(table(pred_sigs)) }
} else {
  ## =========================================================================================
  ## SINGLE-RUN MODE (no logs, no lineage, no temp/prune/add) — covers Scenario A & Scenario B
  ## =========================================================================================
  if (!isTRUE(do_ensemble)) {
    cat(sprintf("Single-run mode → training %d model%s inside one DESONN instance, skipping all ensemble/logging.\n",
                as.integer(num_networks), if (num_networks == 1L) "" else "s"))
    
    # IMPORTANT: respect num_networks here so Scenario B works (multi-model, no ensembles)
    main_model <- DESONN$new(
      num_networks    = max(1L, as.integer(num_networks)),
      input_size      = input_size,
      hidden_sizes    = hidden_sizes,
      output_size     = output_size,
      N               = N,
      lambda          = lambda,
      ensemble_number = 0L,
      ensembles       = NULL,      # single run: no ensembles tracking
      ML_NN           = ML_NN,
      method          = init_method,
      custom_scale    = custom_scale
    )
    # NOTE: The code below still uses the word "ensemble" for consistency,
    # even in single-run mode. It is not a true ensemble in that case,
    # but changing the naming caused downstream issues with plotting logic
    # (especially where plots are returned as lists). To avoid the heavy
    # rework needed just for naming, we keep the "ensemble" wording here
    # and simply document the distinction.
    # Apply per-SONN plotting flags to all internal models
    # Apply per‑SONN plotting flags to all internal models
    if (length(main_model$ensemble)) {
      for (m in seq_along(main_model$ensemble)) {
        main_model$ensemble[[m]]$PerEpochViewPlotsConfig <- list(
          accuracy_plot   = isTRUE(accuracy_plot),
          saturation_plot = isTRUE(saturation_plot),
          max_weight_plot = isTRUE(max_weight_plot),
          viewAllPlots    = isTRUE(viewAllPlots),
          verbose         = isTRUE(verbose)
        )
        main_model$ensemble[[m]]$FinalUpdatePerformanceandRelevanceViewPlotsConfig <- list(
          performance_high_mean_plots = isTRUE(performance_high_mean_plots),
          performance_low_mean_plots  = isTRUE(performance_low_mean_plots),
          relevance_high_mean_plots   = isTRUE(relevance_high_mean_plots),
          relevance_low_mean_plots    = isTRUE(relevance_low_mean_plots),
          viewAllPlots                = isTRUE(viewAllPlots),
          verbose                     = isTRUE(verbose)
        )
      }
    }
    # R defaults to doubles for numbers unless you explicitly add the L.
    invisible(main_model$train(
      Rdata=X, labels=y, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch,
      lr_min=lr_min, ensemble_number=0L, num_epochs=num_epochs,
      threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
      activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
      dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
      beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
      batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
      epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
      shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights,
      X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
      train=train, viewTables=viewTables, verbose=verbose
    ))
    
    # Keep the ID on the run object too (belt-and-suspenders)
    main_model$ensemble_number <- 0L
    
    # Attach the run into the top-level container (so downstream code sees it)
    ensembles <- attach_run_to_container(ensembles, main_model)
    
    # Optional: quick tree view
    print_ensembles_summary(ensembles)
    
    # Optional: expose under ensembles$main_ensemble[[1]] for downstream convenience
    if (exists("ensembles", inherits = TRUE) && is.list(ensembles)) {
      if (is.null(ensembles$main_ensemble)) ensembles$main_ensemble <- list()
      ensembles$main_ensemble[[1]] <- main_model
    }
    
    # Optional summaries
    if (!is.null(main_model$performance_metric)) {
      cat("\nSingle run performance_metric (DESONN-level):\n"); print(main_model$performance_metric)
    }
    if (!is.null(main_model$relevance_metric)) {
      cat("\nSingle run relevance_metric (DESONN-level):\n"); print(main_model$relevance_metric)
    }
    # If you want per-model metrics, iterate main_model$ensemble here as needed.
    
  } 
  else {
    ## ==========================================================
    ## SINGLE-RUN or ENSEMBLE MODE
    ## ==========================================================
    
    # ---------- stable-slot placeholder ----------
    EMPTY_SLOT <- structure(list(.empty_slot = TRUE), class = "EMPTY_SLOT")
    
    # ---------- metadata compose helpers (for both modes) ----------
    `%||%` <- get0("%||%", ifnotfound = function(x, y) if (is.null(x)) y else x)
    
    main_meta_var <- function(i) sprintf("Ensemble_Main_1_model_%d_metadata", as.integer(i))
    temp_meta_var <- function(e,i) sprintf("Ensemble_Temp_%d_model_%d_metadata", as.integer(e), as.integer(i))
    
    .list_main_vars <- function() {
      v <- grep("^Ensemble_Main_(0|1)_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
      if (!length(v)) v <- grep("^Ensemble_Main_1_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
      v
    }
    .get_vars_as_list <- function(vars) {
      vars <- vars[nzchar(vars) & vapply(vars, exists, logical(1), envir = .GlobalEnv)]
      setNames(lapply(vars, get, envir = .GlobalEnv), vars)
    }
    .serial_to_var <- function(s) {
      if (!is.character(s) || !nzchar(s)) return(NA_character_)
      p <- strsplit(s, "\\.")[[1]]
      e <- suppressWarnings(as.integer(p[1])); i <- suppressWarnings(as.integer(p[3]))
      if (!is.finite(e) || !is.finite(i)) return(NA_character_)
      if (e == 1L) {
        v1 <- main_meta_var(i)
        if (exists(v1, envir = .GlobalEnv)) return(v1)
        v0 <- sprintf("Ensemble_Main_0_model_%d_metadata", i)
        if (exists(v0, envir = .GlobalEnv)) return(v0)
        return(NA_character_)
      } else temp_meta_var(e, i)
    }
    .snapshot_main_serials_safe <- function() {
      if (exists("snapshot_main_serials_meta", mode = "function")) {
        s <- try(snapshot_main_serials_meta(), silent = TRUE)
        if (!inherits(s, "try-error") && length(s)) return(s)
      }
      mv <- .list_main_vars()
      if (!length(mv)) return(character())
      ord <- suppressWarnings(as.integer(sub("^Ensemble_Main_(0|1)_model_(\\d+)_metadata$", "\\2", mv)))
      mv <- mv[order(ord)]
      vapply(mv, function(v) {
        md <- get(v, envir = .GlobalEnv)
        as.character(md$model_serial_num %||% NA_character_)
      }, character(1))
    }
    .latest_temp_e <- function() {
      vs <- grep("^Ensemble_Temp_(\\d+)_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
      if (!length(vs)) return(NA_integer_)
      max(suppressWarnings(as.integer(sub("^Ensemble_Temp_(\\d+)_model_\\d+_metadata$", "\\1", vs))))
    }
    compose_metadata_views <- function(ensembles) {
      if (is.null(ensembles$original)) ensembles$original <- list()
      
      # Originals (all MAIN & all TEMP waves)
      main_vars <- .list_main_vars()
      ensembles$original$main_1 <- .get_vars_as_list(main_vars)
      
      t_all <- grep("^Ensemble_Temp_(\\d+)_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
      if (length(t_all)) {
        Es <- sort(unique(as.integer(sub("^Ensemble_Temp_(\\d+)_model_\\d+_metadata$", "\\1", t_all))))
        for (e in Es) {
          vars_e <- grep(sprintf("^Ensemble_Temp_%d_model_\\d+_metadata$", e), t_all, value = TRUE)
          ensembles$original[[sprintf("temp_%d", e)]] <- .get_vars_as_list(vars_e)
        }
      }
      
      # MAIN — current composition by slot (after prune/add if any)
      cur_main_serials <- .snapshot_main_serials_safe()
      if (length(cur_main_serials)) {
        cur_vars <- vapply(cur_main_serials, .serial_to_var, character(1))
        cur_vars <- cur_vars[nzchar(cur_vars)]
        ensembles$main <- .get_vars_as_list(cur_vars)
      } else {
        # single-run fallback: current == originals
        ensembles$main <- .get_vars_as_list(main_vars)
      }
      
      # TEMP — latest wave only
      e_last <- .latest_temp_e()
      if (is.finite(e_last)) {
        vars_e <- grep(sprintf("^Ensemble_Temp_%d_model_\\d+_metadata$", e_last),
                       ls(.GlobalEnv), value = TRUE)
        ensembles$temp <- .get_vars_as_list(vars_e)
      } else {
        ensembles$temp <- list()
      }
      
      # conveniences
      ensembles$main_names     <- names(ensembles$main)
      ensembles$temp_names     <- names(ensembles$temp)
      ensembles$original_names <- lapply(ensembles$original, names)
      
      ensembles
    }
    
    # =========================================================================================
    # Branch A: SINGLE-RUN (no ensembles/prune/add)  — your Scenario A/B
    # =========================================================================================
    if (!isTRUE(do_ensemble)) {
      cat(sprintf("Single-run mode → training %d model%s inside one DESONN instance, skipping ensemble logic.\n",
                  as.integer(num_networks), if (num_networks == 1L) "" else "s"))
      
      main_model <- DESONN$new(
        num_networks    = max(1L, as.integer(num_networks)),
        input_size      = input_size,
        hidden_sizes    = hidden_sizes,
        output_size     = output_size,
        N               = N,
        lambda          = lambda,
        ensemble_number = 0L,
        ensembles       = NULL,
        ML_NN           = ML_NN,
        method          = init_method,
        custom_scale    = custom_scale
      )
      
      # Apply per-SONN plotting flags
      if (length(main_model$ensemble)) {
        for (m in seq_along(main_model$ensemble)) {
          main_model$ensemble[[m]]$PerEpochViewPlotsConfig <- list(
            accuracy_plot   = isTRUE(accuracy_plot),
            saturation_plot = isTRUE(saturation_plot),
            max_weight_plot = isTRUE(max_weight_plot),
            viewAllPlots    = isTRUE(viewAllPlots),
            verbose         = isTRUE(verbose)
          )
          main_model$ensemble[[m]]$FinalUpdatePerformanceandRelevanceViewPlotsConfig <- list(
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
        Rdata=X, labels=y, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch,
        lr_min=lr_min, ensemble_number=0L, num_epochs=num_epochs,
        threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
        activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
        dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
        beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
        batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
        epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
        shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights,
        X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
        train=train, viewTables=viewTables, verbose=verbose
      ))
      
      # tuck the run into the container (compat with downstream)
      if (is.null(ensembles$main_ensemble)) ensembles$main_ensemble <- list()
      ensembles$main_ensemble[[1]] <- main_model
      
      # mirror metadata views (original/main/temp)
      ensembles <- compose_metadata_views(ensembles)
      
      # optional: quick summary
      if (exists("print_ensembles_summary", mode = "function")) {
        print_ensembles_summary(ensembles)
      }
      
    } else {
      
      # =======================================================================================
      # Branch B: ENSEMBLE MODE (MAIN + TEMP prune/add) — your Scenario C/D
      # =======================================================================================
      
      # ------- logging helpers (unchanged except for returning top-level 'ensembles' where needed) -------
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
            if (nm %in% c("iteration","slot")) rows[[nm]] <- rep(NA_integer_, nrows)
            else if (nm %in% c("metric_value")) rows[[nm]] <- rep(NA_real_, nrows)
            else if (nm %in% c("timestamp")) rows[[nm]] <- as.POSIXct(rep(NA, nrows))
            else rows[[nm]] <- rep(NA_character_, nrows)
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
        }
        rows <- rows[, names(ensembles$tables$movement_log), drop = FALSE]
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
      
      # ---------- PRUNE ----------
      prune_network_from_ensemble <- function(ensembles, target_metric_name_worst) {
        minimize     <- .metric_minimize(target_metric_name_worst)
        main_serials <- snapshot_main_serials_meta()
        main_vals    <- if (length(main_serials)) vapply(main_serials, get_metric_by_serial, numeric(1), target_metric_name_worst) else numeric(0)
        
        tbl <- data.frame(slot = seq_along(main_serials), serial = main_serials, value = main_vals, stringsAsFactors = FALSE)
        cat("\n==== PRUNE DIAGNOSTICS ====\n")
        cat("Metric:", target_metric_name_worst, " | Direction:", if (minimize) "MINIMIZE (lower better)" else "MAXIMIZE (higher better)", "\n")
        if (!NROW(tbl)) { cat("(no main rows)\n"); return(NULL) }
        print(tbl, row.names = FALSE)
        if (all(!is.finite(tbl$value))) { cat("No finite main values; abort prune.\n"); return(NULL) }
        
        worst_idx  <- if (minimize) which.max(tbl$value) else which.min(tbl$value)
        worst_row  <- tbl[worst_idx, , drop = FALSE]
        worst_slot <- as.integer(worst_row$slot)
        cat(sprintf("Chosen WORST serial = %s | value=%.6f | slot=%d\n", worst_row$serial, worst_row$value, worst_slot))
        
        if (!(length(ensembles$main_ensemble) >= 1L)) { cat("No main container; abort.\n"); return(NULL) }
        main_container <- ensembles$main_ensemble[[1]]
        if (is.null(main_container$ensemble) || !length(main_container$ensemble)) { cat("Main has no models; abort.\n"); return(NULL) }
        if (worst_slot < 1L || worst_slot > length(main_container$ensemble)) { cat("Worst slot OOB; abort.\n"); return(NULL) }
        
        removed_model <- main_container$ensemble[[worst_slot]]
        main_container$ensemble[[worst_slot]] <- EMPTY_SLOT
        ensembles$main_ensemble[[1]] <- main_container
        
        list(
          removed_network     = removed_model,
          updated_ensembles   = ensembles,
          worst_model_index   = worst_slot,
          worst_slot          = worst_slot,
          worst_serial        = as.character(worst_row$serial),
          worst_value         = as.numeric(worst_row$value)
        )
      }
      
      # ---------- ADD ----------
      add_network_to_ensemble <- function(ensembles, target_metric_name_best,
                                          removed_network, ensemble_number,
                                          worst_model_index, removed_serial, removed_value) {
        minimize     <- .metric_minimize(target_metric_name_best)
        temp_serials <- get_temp_serials_meta(ensemble_number)
        temp_vals    <- if (length(temp_serials)) vapply(temp_serials, get_metric_by_serial, numeric(1), target_metric_name_best) else numeric(0)
        
        cat("\n==== ADD DIAGNOSTICS ====\n")
        if (!length(temp_serials)) {
          cat("No TEMP serials; abort add.\n")
          return(list(
            updated_ensembles = ensembles,
            removed_network   = removed_network,
            added_network     = NULL,
            added_serial      = NA_character_,
            added_value       = NA_real_,
            worst_slot        = worst_model_index
          ))
        }
        
        temp_tbl <- data.frame(temp_serial = temp_serials, value = temp_vals, stringsAsFactors = FALSE)
        print(temp_tbl, row.names = FALSE)
        
        best_idx <- if (minimize) which.min(temp_tbl$value) else which.max(temp_tbl$value)
        best_row <- temp_tbl[best_idx, , drop = FALSE]
        best_val <- as.numeric(best_row$value)
        
        removed_val <- removed_value
        if (!is.finite(removed_val) && is_real_serial(removed_serial)) {
          removed_val <- get_metric_by_serial(removed_serial, target_metric_name_best)
        }
        
        cat(sprintf("Compare TEMP(best) %s=%.6f vs REMOVED %s on %s (%s better)\n",
                    best_row$temp_serial, best_val,
                    if (is.finite(removed_val)) sprintf("%.6f", removed_val) else "NA",
                    target_metric_name_best, if (minimize) "lower" else "higher"))
        
        if (!is.finite(best_val) || !is.finite(removed_val) ||
            !(if (minimize) best_val < removed_val else best_val > removed_val)) {
          cat("→ KEEP REMOVED (TEMP not better or NA).\n")
          return(list(
            updated_ensembles = ensembles,
            removed_network   = removed_network,
            added_network     = NULL,
            added_serial      = NA_character_,
            added_value       = NA_real_,
            worst_slot        = worst_model_index
          ))
        }
        
        worst_slot <- as.integer(worst_model_index)
        if (!(length(ensembles$main_ensemble) >= 1L)) { cat("No main container; abort add.\n"); return(invisible()) }
        main_container <- ensembles$main_ensemble[[1]]
        if (is.null(main_container$ensemble) || !length(main_container$ensemble)) { cat("Main has no models; abort add.\n"); return(invisible()) }
        if (worst_slot < 1L || worst_slot > length(main_container$ensemble)) { cat("Worst slot OOB; abort add.\n"); return(invisible()) }
        
        temp_parts <- strsplit(best_row$temp_serial, "\\.")[[1]]
        temp_model_index <- suppressWarnings(as.integer(temp_parts[3]))
        if (!is.finite(temp_model_index) || is.na(temp_model_index)) {
          cat("Could not parse TEMP model index; abort add.\n")
          return(list(
            updated_ensembles = ensembles,
            removed_network   = removed_network,
            added_network     = NULL,
            added_serial      = NA_character_,
            added_value       = NA_real_,
            worst_slot        = worst_slot
          ))
        }
        
        if (!(length(ensembles$temp_ensemble) >= 1L) || is.null(ensembles$temp_ensemble[[1]]$ensemble)) {
          cat("TEMP container not available; abort add.\n")
          return(list(
            updated_ensembles = ensembles,
            removed_network   = removed_network,
            added_network     = NULL,
            added_serial      = NA_character_,
            added_value       = NA_real_,
            worst_slot        = worst_slot
          ))
        }
        temp_container <- ensembles$temp_ensemble[[1]]
        if (temp_model_index < 1L || temp_model_index > length(temp_container$ensemble)) {
          cat("TEMP model index OOB; abort add.\n")
          return(list(
            updated_ensembles = ensembles,
            removed_network   = removed_network,
            added_network     = NULL,
            added_serial      = NA_character_,
            added_value       = NA_real_,
            worst_slot        = worst_slot
          ))
        }
        
        candidate_model <- temp_container$ensemble[[temp_model_index]]
        main_container$ensemble[[worst_slot]] <- candidate_model
        ensembles$main_ensemble[[1]] <- main_container
        
        # metadata swap to reflect move
        temp_e  <- suppressWarnings(as.integer(temp_parts[1]))
        tvar <- temp_meta_var(temp_e, temp_model_index)
        mvar <- main_meta_var(worst_slot)
        if (exists(tvar, envir = .GlobalEnv)) {
          tmd <- get(tvar, envir = .GlobalEnv)
          tmd$model_serial_num <- as.character(best_row$temp_serial)
          assign(mvar, tmd, envir = .GlobalEnv)
        }
        
        lineage_append(worst_slot, best_row$temp_serial)
        cat(sprintf("→ REPLACED MAIN model slot %d: %s -> %s (and updated metadata)\n",
                    worst_slot, removed_serial, best_row$temp_serial))
        
        list(
          updated_ensembles = ensembles,
          removed_network   = removed_network,
          added_network     = candidate_model,
          added_serial      = as.character(best_row$temp_serial),
          added_value       = best_val,
          worst_slot        = worst_slot
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
          ensemble_number = 1L,
          ensembles       = ensembles,
          ML_NN           = ML_NN,
          method          = init_method,
          custom_scale    = custom_scale
        )
        
        if (length(main_model$ensemble)) {
          for (m in seq_along(main_model$ensemble)) {
            main_model$ensemble[[m]]$PerEpochViewPlotsConfig <- list(
              accuracy_plot   = isTRUE(accuracy_plot),
              saturation_plot = isTRUE(saturation_plot),
              max_weight_plot = isTRUE(max_weight_plot),
              viewAllPlots    = isTRUE(viewAllPlots),
              verbose         = isTRUE(verbose)
            )
            main_model$ensemble[[m]]$FinalUpdatePerformanceandRelevanceViewPlotsConfig <- list(
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
          Rdata=X, labels=y, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch,
          lr_min=lr_min, ensemble_number=1L, num_epochs=num_epochs,
          threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
          activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
          dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
          beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
          batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
          epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
          shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights,
          X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
          train=train, viewTables=viewTables, verbose=verbose
        ))
        ensembles$main_ensemble[[1]] <- main_model
        firstRun <- FALSE
        
        # mirror metadata views immediately
        ensembles <- compose_metadata_views(ensembles)
      }
      
      ## ====== PHASE 2: TEMP iterations (prune/add) ======
      debug_prune <- TRUE
      num_temp_iterations <- as.integer(num_temp_iterations %||% 0L)
      
      if (num_temp_iterations > 0L) {
        for (j in seq_len(num_temp_iterations)) {
          cat("\n— TEMP Iteration", j, ": build TEMP and run prune/add —\n")
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
          
          ensembles$temp_ensemble <- vector("list", 1L)
          temp_model <- DESONN$new(
            num_networks    = max(1L, as.integer(num_networks)),
            input_size      = input_size,
            hidden_sizes    = hidden_sizes,
            output_size     = output_size,
            N               = N,
            lambda          = lambda,
            ensemble_number = j + 1L,
            ensembles       = ensembles,
            ML_NN           = ML_NN,
            method          = init_method,
            custom_scale    = custom_scale
          )
          ensembles$temp_ensemble[[1]] <- temp_model
          
          if (length(temp_model$ensemble)) {
            for (m in seq_along(temp_model$ensemble)) {
              temp_model$ensemble[[m]]$PerEpochViewPlotsConfig <- list(
                accuracy_plot   = isTRUE(accuracy_plot),
                saturation_plot = isTRUE(saturation_plot),
                max_weight_plot = isTRUE(max_weight_plot),
                viewAllPlots    = isTRUE(viewAllPlots),
                verbose         = isTRUE(verbose)
              )
              temp_model$ensemble[[m]]$FinalUpdatePerformanceandRelevanceViewPlotsConfig <- list(
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
            Rdata=X, labels=y, lr=lr, ensemble_number=j+1L, num_epochs=num_epochs,
            threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
            activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
            dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
            beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
            batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
            epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
            shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights,
            X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
            train=train, viewTables=viewTables, verbose=verbose
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
          ids_temp <- if (length(t_sers)) vapply(seq_along(t_sers), function(i) temp_meta_var(j+1L, i), character(1)) else character(0)
          append_movement_block_named(rows_temp, ids_temp)
          
          pruned <- prune_network_from_ensemble(ensembles, metric_name)
          removed_serial <- NA_character_
          added_serial   <- NA_character_
          worst_slot     <- NA_integer_
          
          if (!is.null(pruned)) {
            added <- add_network_to_ensemble(
              ensembles               = pruned$updated_ensembles,
              target_metric_name_best = metric_name,
              removed_network         = pruned$removed_network,
              ensemble_number         = j,
              worst_model_index       = pruned$worst_model_index,
              removed_serial          = pruned$worst_serial,
              removed_value           = pruned$worst_value
            )
            
            ensembles      <- added$updated_ensembles
            removed_serial <- pruned$worst_serial
            added_serial   <- added$added_serial %||% NA_character_
            worst_slot     <- added$worst_slot   %||% pruned$worst_slot %||% NA_integer_
            
            removed_val_cached <- pruned$worst_value
            added_val_cached   <- added$added_value
            
            if (is_real_serial(removed_serial) && is_real_serial(added_serial)) {
              rrow <- data.frame(
                iteration     = j, phase = "removed", slot = worst_slot, role = "removed",
                serial        = removed_serial, metric_name = metric_name,
                metric_value  = removed_val_cached,
                current_serial= NA_character_,
                message       = sprintf("%s replaced by %s", removed_serial, added_serial),
                timestamp     = ts_iter, stringsAsFactors = FALSE
              )
              arow <- data.frame(
                iteration     = j, phase = "added", slot = worst_slot, role = "added",
                serial        = added_serial, metric_name = metric_name,
                metric_value  = added_val_cached,
                current_serial= NA_character_,
                message       = sprintf("%s replaced by %s", removed_serial, added_serial),
                timestamp     = ts_iter, stringsAsFactors = FALSE
              )
              append_movement_block_named(rrow, "", fill_lineage_for_slots = worst_slot)
              append_movement_block_named(arow, "", fill_lineage_for_slots = worst_slot)
              
              append_change_rows(rbind(
                data.frame(iteration=j, role="removed", serial=removed_serial,
                           metric_name=metric_name, metric_value=removed_val_cached,
                           message=sprintf("%s replaced by %s", removed_serial, added_serial),
                           timestamp=ts_iter, stringsAsFactors=FALSE),
                data.frame(iteration=j, role="added",   serial=added_serial,
                           metric_name=metric_name, metric_value=added_val_cached,
                           message=sprintf("%s replaced by %s", removed_serial, added_serial),
                           timestamp=ts_iter, stringsAsFactors=FALSE)
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
          
          # refresh metadata mirrors after each iteration
          ensembles <- compose_metadata_views(ensembles)
        }
      }
      
      # final refresh (even if no temp iters)
      ensembles <- compose_metadata_views(ensembles)
    }
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
