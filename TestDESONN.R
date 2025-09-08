source("DESONN.R")
source("utils/utils.R")
source("utils/bootstrap_metadata.R")

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
test_metrics <- TRUE
custom_scale <- .05

ML_NN <- TRUE

learnOnlyTrainingRun <- FALSE
update_weights <- TRUE
update_biases <- TRUE
# hidden_sizes <- NULL
hidden_sizes <- c(64, 32)

# Activation functions applied in forward pass during prediction | predict(). # hidden layers + output layer
activation_functions <- list(relu, relu, sigmoid)  

# Activation functions applied in forward pass during training | learn() # You can keep them the same as predict() or customize, e.g. list(relu, selu, sigmoid) 
activation_functions_learn <- activation_functions
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
  # Rdata <- Rdata / max_val  # range will be roughly [-1, 1]  (Rdata not defined here)
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

# ==============================================================
# Choose whether to use scaled or raw data for NN training
# ==============================================================

scaledData <- TRUE   # <<<<<< set to FALSE to use raw data

if (isTRUE(scaledData)) {
  # $$$$$$$$$$$$$ Overwrite training matrix with scaled data
  X <- as.matrix(X_train_scaled)
  y <- as.matrix(y_train)
  
  X_validation <- as.matrix(X_validation_scaled)
  y_validation <- as.matrix(y_validation)
  
  X_test <- as.matrix(X_test_scaled)
  y_test <- as.matrix(y_test)
  
} else {
  # $$$$$$$$$$$$$ Overwrite training matrix with raw (unscaled) data
  X <- as.matrix(X_train)
  y <- as.matrix(y_train)
  
  X_validation <- as.matrix(X_validation)
  y_validation <- as.matrix(y_validation)
  
  X_test <- as.matrix(X_test)
  y_test <- as.matrix(y_test)
}

colnames(y) <- colname_y

binary_flag <- is_binary(y)


# ==============================================================
# Optional Random Forest-based feature selection (default OFF)
# ==============================================================

importanceFeaturesOnly <- FALSE   # default: don't filter features

if (isTRUE(importanceFeaturesOnly)) {
  library(randomForest)
  
  # --- Train RF on TRAIN split (X, y) ---
  rf_data <- as.data.frame(X)                         
  rf_data$DEATH_EVENT <- as.factor(as.vector(y[, 1])) # ensure 1D factor
  
  set.seed(42)
  rf_model <- randomForest(DEATH_EVENT ~ ., data = rf_data, importance = TRUE)
  
  # Compute feature importance and select features above mean
  importance_scores <- importance(rf_model, type = 2)[, 1]  # MeanDecreaseGini
  threshold <- mean(importance_scores)
  selected_features <- names(importance_scores[importance_scores > threshold])
  
  # Safety net if none pass the threshold
  if (length(selected_features) == 0L) {
    k <- min(10L, length(importance_scores))
    selected_features <- names(sort(importance_scores, decreasing = TRUE))[seq_len(k)]
  }
  
  # Helper: enforce same columns & order; add any missing as zeros
  ensure_feature_columns <- function(M, wanted) {
    M <- as.matrix(M)
    miss <- setdiff(wanted, colnames(M))
    if (length(miss)) {
      M <- cbind(M, matrix(0, nrow = nrow(M), ncol = length(miss),
                           dimnames = list(NULL, miss)))
    }
    M[, wanted, drop = FALSE]
  }
  
  # ---- Apply the filter to ALL splits (train/val/test) ----
  X            <- ensure_feature_columns(X,            selected_features)
  X_validation <- ensure_feature_columns(X_validation, selected_features)
  X_test       <- ensure_feature_columns(X_test,       selected_features)
  
  # Update input size for neural network initialization
  input_size <- ncol(X)
  
  # Keep numeric_columns in sync (if present)
  if (exists("numeric_columns")) {
    numeric_columns <- intersect(numeric_columns, selected_features)
  }
  
  # (optional) quick checks
  stopifnot(identical(colnames(X), colnames(X_validation)),
            identical(colnames(X), colnames(X_test)))
  cat(sprintf("[RF] kept %d features; input_size=%d\n",
              length(selected_features), input_size))
}

# ==============================================================
# Adaptive Sample Weights (default OFF)
# ==============================================================

sampleWeights <- FALSE   # <-- toggle this flag; default = FALSE

if (isTRUE(sampleWeights)) {
  # --- Assume you already have P (n×1 probs) and yi (length n labels in {0,1}) ---
  probs  <- as.numeric(P[, 1])
  labels <- as.numeric(yi)
  
  # Optional: build flags from in-memory features (use your own logic).
  # If you have the *unscaled* data frame used for this split (e.g., X_validation_raw),
  # compute flags on that; otherwise set them to 0s as a safe default.
  deceptive_flags <- rep(0L, length(labels))
  risky_flags     <- rep(0L, length(labels))
  
  # Example (only if you have the needed columns in a DF called X_raw with same row order):
  # q <- function(x, p) quantile(x, p, na.rm = TRUE)
  # deceptive_flags <- as.integer(
  #   X_raw$serum_creatinine < q(X_raw$serum_creatinine, 0.10) &
  #   X_raw$age               < q(X_raw$age,               0.15) &
  #   X_raw$creatinine_phosphokinase < q(X_raw$creatinine_phosphokinase, 0.20)
  # )
  # risky_flags <- as.integer( ...your risky rule here... )
  
  # Sanity
  stopifnot(length(probs) == length(labels),
            length(deceptive_flags) == length(labels),
            length(risky_flags) == length(labels))
  
  # Error magnitude
  errors <- abs(probs - labels)
  
  # Base weights
  base_weights <- rep(1, length(labels))
  base_weights[labels == 1] <- base_weights[labels == 1] * 2
  base_weights[labels == 1 & risky_flags == 1] <- base_weights[labels == 1 & risky_flags == 1] * log(20) * 4
  base_weights[labels == 1 & deceptive_flags == 1] <- base_weights[labels == 1 & deceptive_flags == 1] * 3
  
  # Blend with error + clip
  raw_weights <- base_weights * errors
  raw_weights <- pmin(pmax(raw_weights, 0.05), 23)
  
  # Final adaptive weights (normalized)
  sample_weights <- 0.6 * base_weights + 0.4 * raw_weights
  sample_weights <- sample_weights / mean(sample_weights)
  
  cat("✅ Sample weights created. Mean =", sprintf("%.4f", mean(sample_weights)), "\n")
  
} else {
  sample_weights <- NULL
  cat("ℹ️ Sample weights disabled (sampleWeights=FALSE).\n")
}



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
# num_networks        <- 30L          # e.g., run 5 models in one DESONN instance
# num_temp_iterations <- 0L
#
## SCENARIO C: Main ensemble only (no TEMP/prune-add)
do_ensemble         <- TRUE
num_networks        <- 2L          # example main size
num_temp_iterations <- 0L
#
## SCENARIO D: Main + TEMP iterations (prune/add enabled)
# do_ensemble         <- TRUE
# num_networks        <- 5L          # example main size
# num_temp_iterations <- 6L          # MAIN + 1 TEMP pass (set higher for more TEMP passes)
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

metric_name <- "accuracy"
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
# prepare_disk_only <- TRUE
# prepare_disk_only <- TRUE
PREPARE_DISK_ONLY <- get0("PREPARE_DISK_ONLY", ifnotfound = FALSE)  # one-shot RDS export helper

# Flag specific to disk-only prepare / selection
prepare_disk_only_FROM_RDS <- FALSE
prepare_disk_only_FROM_RDS <- get0("prepare_disk_only_FROM_RDS", ifnotfound = FALSE)
# Modes:
# "train"
  # "predict:stateless"
# "predict:stateful"
# MODE <- "train"
MODE <- "predict:stateless"
MODE <- get0("MODE", ifnotfound = "train")

PREDICT_ONLY_FROM_RDS  <- isTRUE(get0("PREDICT_ONLY_FROM_RDS", inherits=TRUE, ifnotfound=FALSE))
# Prediction scope (only used when MODE starts with "predict:")
PREDICT_SCOPE <- "all"
PREDICT_SCOPE    <- get0("PREDICT_SCOPE",    ifnotfound = "one")
PICK_INDEX       <- as.integer(get0("PICK_INDEX", 1L))    # used when scope="pick"
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
PREPARE_CLEAR_ENV <- FALSE
PREPARE_CLEAR_ENV   <- get0("PREPARE_CLEAR_ENV",   ifnotfound = TRUE)

# ---- NEW FLAG: clear environment after saving EXCEPT the models (keep models in env) ----
# If this is TRUE, PREPARE_CLEAR_ENV is forced FALSE (mutually exclusive).
PREPARE_CLEAR_ENV_EXCEPT_MODELS <- TRUE
PREPARE_CLEAR_ENV_EXCEPT_MODELS <- get0("PREPARE_CLEAR_ENV_EXCEPT_MODELS", ifnotfound = FALSE)

# Enforce mutual exclusion at runtime
if (isTRUE(PREPARE_CLEAR_ENV_EXCEPT_MODELS)) {
  PREPARE_CLEAR_ENV <- FALSE
}

# Default metadata dir (if not already set elsewhere)
.BM_DIR <- get0(".BM_DIR", ifnotfound = "artifacts")

# Helper: strip trailing _YYYYMMDD_HHMMSS from a base name
.strip_ts <- function(x) sub("_(\\d{8}_\\d{6})$", "", x)

# =========================================
# Phase 1 — one-shot export & (optional) clean
# =========================================
if (isTRUE(prepare_disk_only)) {
  
  # Snapshot the clear flags BEFORE any helper that might wipe env
  .CLEAR_EXCEPT <- isTRUE(PREPARE_CLEAR_ENV_EXCEPT_MODELS)  # strong override branch
  .CLEAR_AFTER  <- isTRUE(PREPARE_CLEAR_ENV)                # only used if .CLEAR_EXCEPT is FALSE
  saved_names   <- character(0)  # will hold names of models we want to keep in env (if needed)
  ts_tag        <- format(Sys.time(), "%Y%m%d_%H%M%S")  # timestamp for filenames
  
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
    # rename file with timestamp
    if (file.exists(rds_path)) {
      rds_new <- sub("\\.rds$", sprintf("_%s.rds", ts_tag), rds_path)
      file.rename(rds_path, rds_new)
      rds_path <- rds_new
    }
    cat(sprintf("[prepare_disk_only] Saved metadata to: %s\n", rds_path))
    
    # record the object name (strip timestamp) so we can keep it in env after clearing
    base_no_ext <- sub("\\.rds$", "", basename(rds_path))
    saved_names <- unique(c(saved_names, .strip_ts(base_no_ext)))
    
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
    
    # rename each file with timestamp
    rds_new_paths <- character(0)
    if (length(rds_paths)) {
      for (p in unique(rds_paths)) {
        if (file.exists(p)) {
          p_new <- sub("\\.rds$", sprintf("_%s.rds", ts_tag), p)
          file.rename(p, p_new)
          rds_new_paths <- c(rds_new_paths, p_new)
        }
      }
      cat("[prepare_disk_only] Saved metadata files:\n")
      for (p in rds_new_paths) cat("  - ", p, "\n", sep = "")
    } else {
      cat("[prepare_disk_only] No candidates saved (zero-length result).\n")
    }
    
    # record all object names (strip timestamp) so we can keep them in env after clearing
    base_no_ext <- sub("\\.rds$", "", basename(rds_new_paths))
    saved_names <- unique(c(saved_names, .strip_ts(base_no_ext)))
  }
  
  # -------------------------------
  # C) Clear environment policy
  # -------------------------------
  if (.CLEAR_EXCEPT) {
    # Clear everything EXCEPT the saved models/metadata objects (by name)
    if (length(saved_names)) {
      if (exists("bm_clear_env_except", inherits = TRUE)) {
        bm_clear_env_except(keep_names = saved_names)
      } else {
        all_objs <- ls(envir = .GlobalEnv, all.names = FALSE)
        to_rm    <- setdiff(all_objs, saved_names)
        if (length(to_rm)) rm(list = to_rm, envir = .GlobalEnv)
      }
      gc()
      cat("[prepare_disk_only] Environment cleared EXCEPT saved models (kept in env). End of phase-1.\n")
    } else {
      # nothing saved? full clear fallback
      rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)
      gc()
      cat("[prepare_disk_only] Environment fully cleared (no saved models to keep). End of phase-1.\n")
    }
  } else if (.CLEAR_AFTER) {
    # Legacy behavior: clear after saving; keep saved models/metadata in env
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
  } else {
    cat("[prepare_disk_only] No environment clear requested. End of phase-1.\n")
  }
  
  # Final stop/quit
  if (exists(".hard_stop", mode = "function")) .hard_stop() else stop("prepare_disk_only done.")
}


# Derive boolean once from MODE (no redundancy)
train <- identical(MODE, "train")
INPUT_SPLIT    <- "auto"   # or "test" / "train" / "auto"
USE_EMBEDDED_X <- FALSE          # keep FALSE to ensure it uses your chosen split

## —— Ensembling
ENABLE_ENSEMBLE_AVG   <- TRUE
ENABLE_ENSEMBLE_WAVG  <- TRUE
ENABLE_ENSEMBLE_VOTE  <- TRUE
ENSEMBLE_WEIGHT_COLUMN <- "tuned_f1"  # falls back automatically if missing
ENSEMBLE_RESPECT_MINIMIZE <- TRUE
ENSEMBLE_VOTE_USE_TUNED_THRESH <- TRUE
# ENSEMBLE_VOTE_QUORUM <- 4L      # optional explicit quorum

## —— Printing
PREDICT_FULL_PRINT  <- TRUE      # show everything
PREDICT_HEAD_N      <- 100L      # if not full-printing
PREDICT_PRINT_MAX   <- 1e7
PREDICT_PRINT_WIDTH <- 240L
PREDICT_USE_TIBBLE  <- TRUE

## —— Artifacts
ARTIFACTS_DIR       <- file.path(getwd(), "artifacts")
PREDICT_RDS_DEBUG   <- FALSE     # set TRUE if model_rds shows NA

# ======================= PREDICT-ONLY (choose split via INPUT_SPLIT) =======================
if (!train) {
  cat("Predict-only mode (train == FALSE).\n")
  
  # -------------------------------
  # Config / scope
  # -------------------------------
  predict_mode  <- if (identical(MODE, "predict:stateful")) "stateful" else "stateless"
  scope_opt     <- match.arg(PREDICT_SCOPE, c("one","group-best","all","pick","single"))
  if (identical(scope_opt, "single")) scope_opt <- "one"
  PICK_INDEX    <- as.integer(PICK_INDEX)
  
  # -------------------------------
  # Build candidate list from env/RDS
  # -------------------------------
  df_all <- bm_list_all()
  if (!nrow(df_all)) stop("No models found in env/RDS (bm_list_all() returned 0 rows).")
  
  if (length(KIND_FILTER))    df_all <- df_all[df_all$kind  %in% KIND_FILTER, , drop = FALSE]
  if (!is.null(ENS_FILTER))   df_all <- df_all[df_all$ens   %in% ENS_FILTER,  , drop = FALSE]
  if (!is.null(MODEL_FILTER)) df_all <- df_all[df_all$model %in% MODEL_FILTER,, drop = FALSE]
  if (!nrow(df_all)) stop("No candidates after applying KIND/ENS/MODEL filters.")
  
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
  
  # De-duplicate (prefer RDS unless flag set)
  if (!"source" %in% names(df_ranked)) df_ranked$source <- NA_character_
  source_priority <- if (isTRUE(PREDICT_ONLY_FROM_RDS))
    c("rds","file","disk","env","memory","workspace") else
      c("env","memory","workspace","rds","file","disk")
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
  
  # -------------------------------
  # Scope mapping
  # -------------------------------
  scope_rows <- switch(scope_opt,
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
  # Resolve split (INPUT_SPLIT) and build common X/y
  # -------------------------------
  `%||%` <- function(x,y) if (is.null(x)) y else x
  
  resolve_split <- function() {
    s <- tolower(INPUT_SPLIT)
    if (s == "test") {
      return(list(
        X = get0("X_test", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL),
        y = get0("y_test", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL),
        tag = "X_test", chosen = "test"
      ))
    } else if (s == "validation") {
      return(list(
        X = get0("X_validation", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL),
        y = get0("y_validation", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL),
        tag = "X_validation", chosen = "validation"
      ))
    } else if (s == "train") {
      return(list(
        X = get0("X", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL),
        y = get0("y", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL),
        tag = "X_train", chosen = "train"
      ))
    } else { # auto: validation → test → train
      Xv <- get0("X_validation", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL)
      yv <- get0("y_validation", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL)
      if (!is.null(Xv) && !is.null(yv)) return(list(X=Xv, y=yv, tag="auto(val→test→train)", chosen="validation"))
      Xt <- get0("X_test", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL)
      yt <- get0("y_test", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL)
      if (!is.null(Xt) && !is.null(yt)) return(list(X=Xt, y=yt, tag="auto(val→test→train)", chosen="test"))
      Xr <- get0("X", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL)
      yr <- get0("y", envir=.GlobalEnv, inherits=TRUE, ifnotfound=NULL)
      if (!is.null(Xr) && !is.null(yr)) return(list(X=Xr, y=yr, tag="auto(val→test→train)", chosen="train"))
      return(list(X=NULL, y=NULL, tag="auto(val→test→train)", chosen="unknown"))
    }
  }
  
  sel <- resolve_split()
  X_common <- sel$X; y_common <- sel$y
  if (is.null(X_common) || is.null(y_common)) stop(sprintf("INPUT_SPLIT='%s' not found.", INPUT_SPLIT))
  y_common <- .normalize_y(y_common)
  
  # -------------------------------
  # Helpers
  # -------------------------------
  .digest_or <- function(obj) {
    if (requireNamespace("digest", quietly=TRUE)) {
      tryCatch(digest::digest(obj, algo="xxhash64"), error=function(e) "NA")
    } else "digest_pkg_missing"
  }
  
  # Resolve RDS strictly from artifacts/, report if used
  .resolve_model_rds <- function(kind, ens, model,
                                 artifacts_dir = get0("ARTIFACTS_DIR", inherits=TRUE, ifnotfound=file.path(getwd(),"artifacts"))) {
    out <- list(file = NA_character_, used = FALSE)
    if (!dir.exists(artifacts_dir)) return(out)
    files <- tryCatch(
      list.files(artifacts_dir, pattern="\\.[Rr][Dd][Ss]$", full.names=TRUE, recursive=TRUE, include.dirs=FALSE),
      error = function(e) character(0)
    )
    if (!length(files)) return(out)
    b <- basename(files)
    
    # prefer exact then prefixed common names created by prepare_disk_only
    patts <- c(
      sprintf("^model_%s_%d_%d\\.[Rr][Dd][Ss]$", kind, ens, model),
      sprintf("^model_%s_%d_%d_.*\\.[Rr][Dd][Ss]$", kind, ens, model),
      sprintf("^Ensemble_%s_%d_model_%d.*\\.[Rr][Dd][Ss]$", kind, ens, model)
    )
    hit_idx <- Reduce(`|`, lapply(patts, function(p) grepl(p, b, ignore.case = FALSE)))
    hits <- files[hit_idx]
    if (!length(hits)) return(out)
    info <- file.info(hits)
    hits <- hits[order(info$mtime, decreasing = TRUE)]
    out$file  <- basename(hits[1])
    out$used  <- TRUE
    out
  }
  
  .slot_wsig <- function(meta, slot_index) {
    if (exists("hash_meta_weights", mode = "function")) return(hash_meta_weights(meta, slot_index))
    w <- tryCatch({
      if (!is.null(meta$best_model_metadata$best_weights_record))
        meta$best_model_metadata$best_weights_record else meta$best_weights_record
    }, error=function(e) NULL)
    if (!is.null(w)) return(.digest_or(w))
    .digest_or(meta)
  }
  .get_expected_feature_names <- function(meta) {
    nms <- NULL
    try({ nms <- meta$feature_names %||% meta$input_names %||% meta$colnames }, silent = TRUE)
    if (is.null(nms)) {
      try({
        cs <- meta$preprocess %||% meta$scaler %||% meta$best_model_metadata$preprocess
        if (!is.null(cs)) {
          nms <- names(cs$center) %||% colnames(cs$center) %||%
            names(cs$scale)  %||% colnames(cs$scale)
        }
      }, silent = TRUE)
    }
    if (is.null(nms)) {
      try({
        xin <- .choose_X_from_meta(meta)
        if (!is.null(xin) && !is.null(colnames(xin$X))) nms <- colnames(xin$X)
      }, silent = TRUE)
    }
    if (!is.null(nms)) nms <- as.character(nms)
    nms
  }
  .ensure_columns <- function(X, expected_names) {
    X <- as.matrix(X)
    if (is.null(expected_names)) return(X)
    miss <- setdiff(expected_names, colnames(X))
    if (length(miss)) {
      X <- cbind(X, matrix(0, nrow = nrow(X), ncol = length(miss),
                           dimnames = list(NULL, miss)))
    }
    X <- X[, expected_names, drop = FALSE]
    X
  }
  
  # -------------------------------
  # Predict per model
  # -------------------------------
  results   <- vector("list", length = nrow(scope_rows))
  pred_sigs <- character(nrow(scope_rows))
  P_list    <- vector("list", length = nrow(scope_rows))
  meta_list <- vector("list", length = nrow(scope_rows))
  
  for (i in seq_len(nrow(scope_rows))) {
    kind  <- as.character(scope_rows$kind[i])
    ens   <- as.integer(scope_rows$ens[i])
    model <- as.integer(scope_rows$model[i])
    
    varname <- sprintf("Ensemble_%s_%d_model_%d_metadata", kind, ens, model)
    meta <- if (exists(varname, envir=.GlobalEnv)) get(varname, envir=.GlobalEnv) else bm_select_exact(kind, ens, model)
    if (is.null(meta)) { warning(sprintf("Skipping %s/%d/%d: no metadata.", kind, ens, model)); next }
    
    # Respect INPUT_SPLIT unless embedded-X explicitly allowed
    xin <- if (isTRUE(USE_EMBEDDED_X)) .choose_X_from_meta(meta) else NULL
    Xi  <- if (!is.null(xin)) xin$X else X_common
    yi  <- if (isTRUE(USE_EMBEDDED_X) && !is.null(.choose_y_from_meta(meta))) {
      .normalize_y(.choose_y_from_meta(meta)$y)
    } else y_common
    if (is.null(Xi)) { warning(sprintf("Skipping %s/%d/%d: no X.", kind, ens, model)); next }
    
    # Feature alignment to model expectation
    expected  <- .get_expected_feature_names(meta)
    orig_cols <- colnames(Xi)
    Xi <- .ensure_columns(Xi, expected)
    if (!is.null(expected)) {
      added <- setdiff(expected, orig_cols)
      dropped <- setdiff(orig_cols, expected)
      if (length(added) || length(dropped)) {
        cat(sprintf("   · feature-align: +%d (missing→0), -%d (extra dropped), final_cols=%d\n",
                    length(added), length(dropped), ncol(Xi)))
        if (length(added))   cat(sprintf("     + %s\n", paste(utils::head(added, 10), collapse=", ")))
        if (length(dropped)) cat(sprintf("     - %s\n", paste(utils::head(dropped, 10), collapse=", ")))
      }
    }
    # Safety check if model has fixed expected size
    input_size <- tryCatch({
      meta$input_size %||% ncol(.choose_X_from_meta(meta)$X)
    }, error = function(e) NULL)
    if (!is.null(input_size) && ncol(Xi) != input_size) {
      stop(sprintf("Input feature count (%d) doesn’t match model’s expected input (%d). Provide matching columns or store feature_names in metadata.",
                   ncol(Xi), input_size))
    }
    
    # Align to reference & scale if any
    Xi <- .align_by_names_safe(Xi, X_common)
    Xi <- .apply_scaling_if_any(as.matrix(Xi), meta)
    
    # Prediction (safe)
    pred_raw <- tryCatch(.safe_run_predict(X = Xi, meta = meta, model_index = model, ML_NN = TRUE),
                         error = function(e) { message("! predict failed for ", as.character(meta$model_serial_num %||% "NA"), ": ", conditionMessage(e)); NULL })
    if (is.null(pred_raw)) {
      P <- matrix(numeric(0), nrow=0, ncol=1)
    } else {
      P <- .as_pred_matrix(pred_raw)
    }
    P_list[[i]] <- P
    meta_list[[i]] <- meta
    
    # RDS from artifacts? (true if prepare_disk_only produced matching file)
    rds_info <- .resolve_model_rds(kind, ens, model)
    model_rds_name <- rds_info$file
    artifact_used  <- if (isTRUE(rds_info$used)) "yes" else "no"
    
    # Signatures
    wsig <- .slot_wsig(meta, model)
    psig <- .digest_or(round(P[seq_len(min(nrow(P),2000)),,drop=FALSE],6))
    pred_sigs[i] <- psig
    
    # Console diagnostics
    input_tag  <- if (!is.null(xin)) paste0("embedded:", xin$tag %||% "X") else sel$tag
    split_used <- if (!is.null(xin)) paste0("embedded:", xin$tag %||% "X") else sel$chosen
    cat(sprintf("→ Model(kind=%s, ens=%d, model=%d) preds=%dx%d\n", kind, ens, model, nrow(P), ncol(P)))
    cat(sprintf("   · input_source=%s | rows=%d, cols=%d\n", input_tag, nrow(Xi), ncol(Xi)))
    cat(sprintf("   · serial=%s | w_sig=%s | pred_sig=%s | artifact_rds=%s\n",
                as.character(meta$model_serial_num %||% sprintf("%d.%d.%d", ens,0L,model)), wsig, psig, artifact_used))
    
    # ---- Baseline metrics
    acc_val <- prec_val <- rec_val <- f1_val <- NA_real_
    tp <- fp <- fn <- tn <- NA_integer_
    if (!is.null(yi) && nrow(P)>0) {
      n <- min(nrow(P), length(yi))
      if (nrow(P)!=length(yi)) {
        warning(sprintf("[%s/%d/%d] preds=%d labels=%d trunc→%d",kind,ens,model,nrow(P),length(yi),n))
        P <- P[1:n,,drop=FALSE]; P_list[[i]] <- P; yi <- yi[1:n]
      }
      if (ncol(P)==1L) { y_pred <- as.integer(as.numeric(P[,1])>=0.5) } else { y_pred <- as.integer((max.col(P)-1L)>0L) }
      y_true01 <- as.integer(yi>0)
      acc_val <- mean(y_pred==y_true01); tp<-sum(y_pred==1&y_true01==1); fp<-sum(y_pred==1&y_true01==0); fn<-sum(y_pred==0&y_true01==1); tn<-sum(y_pred==0&y_true01==0)
      prec_val <- tp/(tp+fp+1e-8); rec_val<-tp/(tp+fn+1e-8); f1_val<-2*prec_val*rec_val/(prec_val+rec_val+1e-8)
    }
    
    # ---- Tuned metrics (0–1, not percent)
    tuned_thr <- tuned_acc <- tuned_prec <- tuned_rec <- tuned_f1 <- NA_real_
    if (is.function(get0("accuracy_tuned", inherits=TRUE)) && !is.null(yi) && nrow(P)>0) {
      tuning_metric <- tolower(get0("TARGET_METRIC", ifnotfound="accuracy", inherits=TRUE))
      allowed <- c("accuracy","f1","precision","recall","macro_f1","macro_precision","macro_recall")
      if (!tuning_metric %in% allowed) tuning_metric <- "accuracy"
      tuned <- tryCatch(
        accuracy_tuned(SONN=NULL, Rdata=Xi, labels=yi, predicted_output=P,
                       metric_for_tuning=tuning_metric,
                       threshold_grid=seq(0.05,0.95,by=0.01), verbose=FALSE),
        error=function(e) NULL
      )
      if (!is.null(tuned)) {
        tuned_acc  <- as.numeric(tuned$accuracy %||% NA_real_)
        tuned_prec <- as.numeric(tuned$precision %||% NA_real_)
        tuned_rec  <- as.numeric(tuned$recall %||% NA_real_)
        tuned_f1   <- as.numeric(tuned$F1 %||% NA_real_)
        tuned_thr  <- as.numeric((tuned$details %||% list())$best_threshold %||% NA_real_)
      }
    }
    
    # Round numerics to 6 decimals for storage
    r6 <- function(x) if (is.na(x)) NA_real_ else round(x, 6)
    
    results[[i]] <- list(kind=kind, ens=ens, model=model,
                         data_source=input_tag,
                         split_used=split_used,
                         n_pred_rows=nrow(P),
                         accuracy=r6(acc_val), precision=r6(prec_val), recall=r6(rec_val), f1=r6(f1_val),
                         tuned_threshold=r6(tuned_thr), tuned_accuracy=r6(tuned_acc),
                         tuned_precision=r6(tuned_prec), tuned_recall=r6(tuned_rec), tuned_f1=r6(tuned_f1),
                         tp=tp, fp=fp, fn=fn, tn=tn,
                         w_sig=.slot_wsig(meta, model), pred_sig=psig,
                         model_rds=model_rds_name,
                         artifact_used=artifact_used)
  }
  
  results <- Filter(Negate(is.null), results)
  if (!length(results)) stop("No successful predictions.")
  
  # ----- Build numeric results_df (metrics rounded to 6); WITHOUT model_rds for now
  rows <- lapply(results, function(z) {
    data.frame(
      kind=z$kind, ens=z$ens, model=z$model, data_source=z$data_source,
      split_used=z$split_used,
      n_pred_rows=z$n_pred_rows, accuracy=z$accuracy, precision=z$precision,
      recall=z$recall, f1=z$f1,
      tuned_threshold=z$tuned_threshold, tuned_accuracy=z$tuned_accuracy,
      tuned_precision=z$tuned_precision, tuned_recall=z$tuned_recall, tuned_f1=z$tuned_f1,
      tp=z$tp, fp=z$fp, fn=z$fn, tn=z$tn,
      w_sig=z$w_sig, pred_sig=z$pred_sig,
      stringsAsFactors=FALSE)
  })
  results_df <- do.call(rbind, rows); rownames(results_df) <- NULL
  
  # ===========================================
  # ENSEMBLE COMBINES (avg / weighted / vote)
  # ===========================================
  have_multi <- length(P_list) >= 2 && all(vapply(P_list, function(P) is.matrix(P) && nrow(P)>0, logical(1)))
  ensemble_rows  <- NULL
  ensemble_preds <- list()
  
  # helpers (reuse in ensembles)
  .metrics_from_probs <- function(p, y, thr = 0.5) {
    if (is.null(y)) return(list(acc=NA, prec=NA, rec=NA, f1=NA, tp=NA, fp=NA, fn=NA, tn=NA))
    y01 <- as.integer(y > 0)
    yhat <- as.integer(p >= thr)
    tp <- sum(yhat==1 & y01==1); fp <- sum(yhat==1 & y01==0)
    fn <- sum(yhat==0 & y01==1); tn <- sum(yhat==0 & y01==0)
    acc <- mean(yhat==y01)
    prec <- tp / (tp + fp + 1e-8)
    rec  <- tp / (tp + fn + 1e-8)
    f1   <- 2*prec*rec / (prec + rec + 1e-8)
    list(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)
  }
  .tune_soft <- function(p, y, metric = tolower(TARGET_METRIC), grid = seq(0.05,0.95,by=0.01)) {
    if (is.null(y)) return(list(thr=NA, acc=NA, prec=NA, rec=NA, f1=NA))
    res <- tryCatch(
      tune_threshold_accuracy(predicted_output = matrix(as.numeric(p), ncol=1),
                              labels = matrix(as.numeric(y), ncol=1),
                              metric = switch(metric,
                                              "f1"="f1","precision"="precision","recall"="recall",
                                              "macro_f1"="macro_f1","macro_precision"="macro_precision","macro_recall"="macro_recall",
                                              "accuracy"),
                              threshold_grid = grid,
                              verbose = FALSE),
      error = function(e) NULL
    )
    if (is.null(res)) return(list(thr=NA, acc=NA, prec=NA, rec=NA, f1=NA))
    thr <- as.numeric(res$thresholds)
    y01 <- as.integer(y > 0)
    yhat <- as.integer(as.numeric(p) >= thr)
    tp <- sum(yhat==1 & y01==1); fp <- sum(yhat==1 & y01==0)
    fn <- sum(yhat==0 & y01==1)
    prec <- tp / (tp + fp + 1e-8); rec <- tp / (tp + fn + 1e-8)
    f1 <- 2*prec*rec / (prec + rec + 1e-8)
    list(thr=thr, acc=as.numeric(res$tuned_accuracy), prec=prec, rec=rec, f1=f1)
  }
  r6 <- function(x) if (is.na(x)) NA_real_ else round(x, 6)
  .make_row <- function(kind_label, mets, tuned, N, sel_tag, sel_chosen) {
    data.frame(
      kind  = kind_label, ens = 999L, model = 1L,
      data_source = sel_tag, split_used = sel_chosen,
      n_pred_rows = N,
      accuracy = r6(mets$acc), precision = r6(mets$prec), recall = r6(mets$rec), f1 = r6(mets$f1),
      tuned_threshold = r6(tuned$thr %||% NA_real_),
      tuned_accuracy  = r6(tuned$acc %||% NA_real_),
      tuned_precision = r6(tuned$prec %||% NA_real_),
      tuned_recall    = r6(tuned$rec %||% NA_real_),
      tuned_f1        = r6(tuned$f1 %||% NA_real_),
      tp = mets$tp, fp = mets$fp, fn = mets$fn, tn = mets$tn,
      w_sig = "ENSEMBLE", pred_sig = "ENSEMBLE",
      stringsAsFactors = FALSE
    )
  }
  
  if (have_multi && (ENABLE_ENSEMBLE_AVG || ENABLE_ENSEMBLE_WAVG || ENABLE_ENSEMBLE_VOTE)) {
    extract_prob <- function(P) {
      if (is.null(P) || !is.matrix(P) || nrow(P) == 0) return(NULL)
      if (ncol(P) == 1L) as.numeric(P[,1]) else as.numeric(P[,1])  # assume col1 = positive class prob
    }
    probs <- lapply(P_list, extract_prob)
    lens  <- vapply(probs, length, integer(1))
    N <- min(lens[lens > 0])
    
    if (is.finite(N) && N > 0 && sum(lens >= N) >= 2) {
      probs_mat <- do.call(cbind, lapply(probs, function(v) v[1:N]))
      yi_use <- if (!is.null(y_common)) .normalize_y(y_common)[1:N] else NULL
      
      # 1) Simple average
      if (ENABLE_ENSEMBLE_AVG) {
        p_avg <- rowMeans(probs_mat, na.rm = TRUE)
        m_avg <- .metrics_from_probs(p_avg, yi_use, thr = 0.5)
        t_avg <- .tune_soft(p_avg, yi_use)
        ensemble_rows <- rbind(ensemble_rows, .make_row("Ensemble_avg", m_avg, t_avg, N, sel$tag, sel$chosen))
        ensemble_preds[["Ensemble_avg"]] <- matrix(p_avg, ncol=1)
      }
      # 2) Weighted average
      if (ENABLE_ENSEMBLE_WAVG) {
        pick_weights <- function(df, colname) {
          cn <- tolower(colname)
          if (cn %in% tolower(names(df))) {
            nm <- names(df)[tolower(names(df)) == cn][1]
            as.numeric(df[[nm]])
          } else if ("tuned_f1" %in% names(df) && any(is.finite(df$tuned_f1))) df$tuned_f1
          else if ("f1" %in% names(df) && any(is.finite(df$f1)))               df$f1
          else as.numeric(df$accuracy %||% rep(1, nrow(df)))
        }
        base_w <- pick_weights(results_df, ENSEMBLE_WEIGHT_COLUMN)
        base_w[!is.finite(base_w)] <- 0
        if (isTRUE(ENSEMBLE_RESPECT_MINIMIZE) && .metric_minimize(TARGET_METRIC)) {
          mx <- max(base_w[is.finite(base_w)], na.rm = TRUE); base_w <- (mx - base_w)
        }
        if (sum(base_w) <= 0) base_w[] <- 1
        w <- base_w / sum(base_w)
        p_wavg <- as.numeric(probs_mat %*% w)
        m_wavg <- .metrics_from_probs(p_wavg, yi_use, thr = 0.5)
        t_wavg <- .tune_soft(p_wavg, yi_use)
        ensemble_rows <- rbind(ensemble_rows, .make_row("Ensemble_wavg", m_wavg, t_wavg, N, sel$tag, sel$chosen))
        ensemble_preds[["Ensemble_wavg"]] <- matrix(p_wavg, ncol=1)
      }
      # 3) Vote
      if (ENABLE_ENSEMBLE_VOTE) {
        thr_vec <- if (isTRUE(ENSEMBLE_VOTE_USE_TUNED_THRESH) && "tuned_threshold" %in% names(results_df)) {
          tv <- as.numeric(results_df$tuned_threshold); tv[!is.finite(tv)] <- 0.5; tv
        } else rep(0.5, ncol(probs_mat))
        vote_mat  <- sweep(probs_mat, 2, thr_vec, FUN = ">=") * 1L
        vote_frac <- rowMeans(vote_mat, na.rm = TRUE)   # soft in [0,1]
        q <- if (exists("ENSEMBLE_VOTE_QUORUM", inherits=TRUE) && !is.null(ENSEMBLE_VOTE_QUORUM)) {
          as.integer(get0("ENSEMBLE_VOTE_QUORUM", inherits=TRUE))
        } else ceiling(ncol(probs_mat)/2)
        y_vote_hard <- as.integer(rowSums(vote_mat, na.rm = TRUE) >= q)
        m_vote_soft <- .metrics_from_probs(vote_frac, yi_use, thr = 0.5)
        t_vote_soft <- .tune_soft(vote_frac, yi_use)
        m_vote_hard <- .metrics_from_probs(as.numeric(y_vote_hard), yi_use, thr = 0.5)
        ensemble_rows <- rbind(
          ensemble_rows,
          .make_row("Ensemble_vote_soft", m_vote_soft, t_vote_soft, N, sel$tag, sel$chosen),
          .make_row("Ensemble_vote_hard", m_vote_hard, NULL,        N, sel$tag, sel$chosen)
        )
        ensemble_preds[["Ensemble_vote_soft"]] <- matrix(vote_frac, ncol=1)
        ensemble_preds[["Ensemble_vote_hard"]] <- matrix(as.numeric(y_vote_hard), ncol=1)
      }
    }
  }
  
  # Append ensemble rows safely: align columns before rbind
  if (!is.null(ensemble_rows) && nrow(ensemble_rows)) {
    missing_cols <- setdiff(names(results_df), names(ensemble_rows))
    if (length(missing_cols)) for (cc in missing_cols) ensemble_rows[[cc]] <- NA
    ensemble_rows <- ensemble_rows[, names(results_df), drop = FALSE]
    results_df <- rbind(results_df, ensemble_rows)
  }
  
  # ---- NOW add model_rds + artifact_used columns (NA / "no" for ensembles)
  model_rds_vec <- vapply(results, function(z) z[["model_rds"]] %||% NA_character_, character(1))
  artifact_used_vec <- vapply(results, function(z) z[["artifact_used"]] %||% "no", character(1))
  n_ens <- if (!is.null(ensemble_rows)) nrow(ensemble_rows) else 0L
  results_df$model_rds     <- c(model_rds_vec, rep(NA_character_, n_ens))
  results_df$artifact_used <- c(artifact_used_vec, rep("no", n_ens))
  
  # Expose numeric table to .GlobalEnv
  assign("PREDICT_RESULTS_TABLE", results_df, .GlobalEnv)
  
  # ---- Print with fixed 6 decimals (flags read from .GlobalEnv)
  fmt_cols <- c("accuracy","precision","recall","f1",
                "tuned_threshold","tuned_accuracy","tuned_precision","tuned_recall","tuned_f1")
  results_df_print <- results_df
  for (cc in fmt_cols) results_df_print[[cc]] <- sprintf("%.6f", as.numeric(results_df_print[[cc]]))
  
  PREDICT_FULL_PRINT  <- get0("PREDICT_FULL_PRINT",  inherits = TRUE, ifnotfound = FALSE)
  PREDICT_HEAD_N      <- as.integer(get0("PREDICT_HEAD_N",      inherits = TRUE, ifnotfound = 50L))
  PREDICT_PRINT_MAX   <- as.numeric(get0("PREDICT_PRINT_MAX",   inherits = TRUE, ifnotfound = 1e7))
  PREDICT_PRINT_WIDTH <- as.integer(get0("PREDICT_PRINT_WIDTH", inherits = TRUE, ifnotfound = 200L))
  PREDICT_USE_TIBBLE  <- get0("PREDICT_USE_TIBBLE",  inherits = TRUE, ifnotfound = TRUE)
  
  if (isTRUE(PREDICT_FULL_PRINT)) {
    old_opts <- options(max.print = PREDICT_PRINT_MAX, width = PREDICT_PRINT_WIDTH)
    on.exit(options(old_opts), add = TRUE)
  }
  
  cat(sprintf("[predict] MODE=%s | SCOPE=%s | METRIC=%s | models=%d\n",
              predict_mode, scope_opt, TARGET_METRIC, nrow(results_df_print)))
  
  if (isTRUE(PREDICT_USE_TIBBLE) && requireNamespace("tibble", quietly = TRUE)) {
    tb <- tibble::as_tibble(results_df_print)
    if (isTRUE(PREDICT_FULL_PRINT)) {
      print(tb, n = Inf, width = Inf)
    } else {
      print(tb, n = PREDICT_HEAD_N, width = Inf)
      if (nrow(tb) > PREDICT_HEAD_N) {
        cat(sprintf("... (%d more rows — set PREDICT_FULL_PRINT=TRUE to show all)\n",
                    nrow(tb) - PREDICT_HEAD_N))
      }
    }
  } else {
    if (isTRUE(PREDICT_FULL_PRINT)) {
      print(results_df_print, row.names = FALSE, right = TRUE)
    } else {
      to_show <- utils::head(results_df_print, PREDICT_HEAD_N)
      print(to_show, row.names = FALSE, right = TRUE)
      if (nrow(results_df_print) > nrow(to_show)) {
        cat(sprintf("... (%d more rows — set PREDICT_FULL_PRINT=TRUE to show all)\n",
                    nrow(results_df_print) - nrow(to_show)))
      }
    }
  }
  
  ## ---- NOTE about prepare_disk_only vs PREDICT_ONLY_FROM_RDS ----
  if (PREPARE_DISK_ONLY && PREDICT_ONLY_FROM_RDS) {
    cat("\nNOTE: prepare_disk_only was run AND PREDICT_ONLY_FROM_RDS=TRUE.\n")
    cat("      Models’ .rds are loaded strictly from artifacts/ (see `model_rds` column).\n\n")
  } else if (PREPARE_DISK_ONLY && !PREDICT_ONLY_FROM_RDS) {
    cat("\nNOTE: prepare_disk_only was run, but PREDICT_ONLY_FROM_RDS=FALSE.\n")
    cat("      Artifacts exist, but env/metadata may be preferred depending on source priority.\n\n")
  } else if (!PREPARE_DISK_ONLY && PREDICT_ONLY_FROM_RDS) {
    cat("\nNOTE: PREDICT_ONLY_FROM_RDS=TRUE but prepare_disk_only not run.\n")
    cat("      No artifacts available; model_rds will likely be NA.\n\n")
  } else {
    cat("\nNOTE: Neither prepare_disk_only nor PREDICT_ONLY_FROM_RDS were set.\n")
    cat("      Models came from in-memory/env metadata; no artifact .rds used.\n\n")
  }
  
  # -------------------------------
  # SAVE to artifacts/ with mode & scope in filename
  # -------------------------------
  artifacts_dir <- file.path(getwd(), "artifacts")
  if (!dir.exists(artifacts_dir)) dir.create(artifacts_dir, recursive = TRUE, showWarnings = FALSE)
  ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  scope_tag <- gsub("[^A-Za-z0-9_-]+", "-", tolower(scope_opt))
  mode_tag  <- gsub("[^A-Za-z0-9_-]+", "-", tolower(predict_mode))
  rds_path  <- file.path(artifacts_dir, sprintf("predictions_%s_scope-%s_%s.rds", mode_tag, scope_tag, ts_stamp))
  
  # compact prediction matrices keyed by model tag
  pred_named <- list()
  for (i in seq_along(P_list)) {
    if (is.null(P_list[[i]])) next
    tag <- sprintf("%s_%d_%d", scope_rows$kind[i], scope_rows$ens[i], scope_rows$model[i])
    pred_named[[tag]] <- P_list[[i]]
  }
  if (length(ensemble_preds)) pred_named <- c(pred_named, ensemble_preds)
  
  predict_pack <- list(
    saved_at        = Sys.time(),
    predict_mode    = predict_mode,
    PREDICT_SCOPE   = scope_opt,
    TARGET_METRIC   = TARGET_METRIC,
    minimize_metric = minimize,
    flags = list(
      MODE = MODE,
      PREDICT_ONLY_FROM_RDS = PREDICT_ONLY_FROM_RDS,
      KIND_FILTER = KIND_FILTER, ENS_FILTER = ENS_FILTER, MODEL_FILTER = MODEL_FILTER,
      PICK_INDEX = PICK_INDEX,
      INPUT_SPLIT = INPUT_SPLIT,
      USE_EMBEDDED_X = USE_EMBEDDED_X,
      ENABLE_ENSEMBLE_AVG = ENABLE_ENSEMBLE_AVG,
      ENABLE_ENSEMBLE_WAVG = ENABLE_ENSEMBLE_WAVG,
      ENABLE_ENSEMBLE_VOTE = ENABLE_ENSEMBLE_VOTE,
      ENSEMBLE_WEIGHT_COLUMN = ENSEMBLE_WEIGHT_COLUMN,
      ENSEMBLE_RESPECT_MINIMIZE = ENSEMBLE_RESPECT_MINIMIZE,
      ENSEMBLE_VOTE_USE_TUNED_THRESH = ENSEMBLE_VOTE_USE_TUNED_THRESH
    ),
    candidates_ranked = df_ranked,
    scope_rows        = scope_rows,
    results_table     = results_df,   # numeric, rounded to 6
    prediction_sigs   = pred_sigs,
    predictions       = pred_named
  )
  
  saveRDS(predict_pack, rds_path)
  cat(sprintf("[predict] Saved predictions to: %s\n", rds_path))
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
    '%||%' <- get0("%||%", ifnotfound = function(x, y) if (is.null(x)) y else x)
    
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
    
    # =====================================================================
    # >>>>>>>>>>>>>>>  MAIN LOG (Resulting MAIN winners)  <<<<<<<<<<<<<<<<<
    # =====================================================================
    # Safe null-coalescing already set; bring small shims here so both branches can call them
    .is_real_serial <- get0("is_real_serial",
                            ifnotfound = function(s) is.character(s) && length(s) == 1 && nzchar(s))
    
    # Prefer your existing snapshot function if defined (used inside append)
    snapshot_main_serials_meta <- get0("snapshot_main_serials_meta", ifnotfound = function() {
      vars <- grep("^Ensemble_Main_(0|1)_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
      if (!length(vars)) return(character())
      ord <- suppressWarnings(as.integer(sub("^Ensemble_Main_(0|1)_model_(\\d+)_metadata$", "\\1", vars)))
      vars <- vars[order(ord)]
      vapply(vars, function(v) {
        md <- get(v, envir = .GlobalEnv)
        as.character(md$model_serial_num %||% NA_character_)
      }, character(1))
    })
    
    # Use your .collect_vals if present, otherwise safe fallback
    .collect_vals_safe <- local({
      cv <- get0(".collect_vals", inherits = TRUE)
      if (is.function(cv)) {
        function(serials, metric_name) cv(serials, metric_name)
      } else {
        function(serials, metric_name) {
          get_metric_by_serial <- get0("get_metric_by_serial", inherits = TRUE)
          if (!is.function(get_metric_by_serial)) {
            stop("get_metric_by_serial() not found and .collect_vals() not available.")
          }
          data.frame(
            serial = as.character(serials),
            value  = vapply(as.character(serials),
                            function(s) suppressWarnings(as.numeric(get_metric_by_serial(s, metric_name))),
                            numeric(1)),
            stringsAsFactors = FALSE
          )
        }
      }
    })
    
    # lineage shims (no-ops if lineage not used)
    .lineage_current <- get0("lineage_current", ifnotfound = function(slot) NA_character_)
    .ensure_lineage_columns <- get0("ensure_lineage_columns", ifnotfound = function() {})
    
    ensure_main_log_init <- function() {
      if (is.null(ensembles$tables)) ensembles$tables <<- list()
      if (is.null(ensembles$tables$main_log)) {
        ensembles$tables$main_log <<- data.frame(
          iteration    = integer(),
          phase        = character(),
          slot         = integer(),
          serial       = character(),
          metric_name  = character(),
          metric_value = numeric(),
          current_serial = character(),
          message      = character(),
          timestamp    = as.POSIXct(character()),
          stringsAsFactors = FALSE
        )
        rownames(ensembles$tables$main_log) <<- character(0)
      }
      if (is.null(ensembles$tables$main_log_path)) {
        ensembles$tables$main_log_path <<- file.path(
          getwd(), sprintf("main_log_%s.rds", format(Sys.time(), "%Y%m%d_%H%M%S"))
        )
      }
    }
    .align_rows_to_main_log <- function(rows) {
      logdf  <- ensembles$tables$main_log
      target <- names(logdf)
      nlog   <- nrow(logdf)
      nrows  <- nrow(rows)
      add_missing <- setdiff(target, names(rows))
      if (length(add_missing)) {
        for (nm in add_missing) {
          if (nm %in% c("iteration","slot")) rows[[nm]] <- rep(NA_integer_, nrows)
          else if (nm %in% c("metric_value")) rows[[nm]] <- rep(NA_real_, nrows)
          else if (nm %in% c("timestamp")) rows[[nm]] <- as.POSIXct(rep(NA, nrows))
          else rows[[nm]] <- rep(NA_character_, nrows)
        }
      }
      extras <- setdiff(names(rows), target)
      if (length(extras)) {
        for (nm in extras) {
          col <- rows[[nm]]
          if (is.integer(col)) {
            ensembles$tables$main_log[[nm]] <<- if (nlog) rep(NA_integer_, nlog) else integer(0)
          } else if (is.numeric(col)) {
            ensembles$tables$main_log[[nm]] <<- if (nlog) rep(NA_real_, nlog) else numeric(0)
          } else if (inherits(col, "POSIXct")) {
            ensembles$tables$main_log[[nm]] <<- if (nlog) as.POSIXct(rep(NA, nlog)) else as.POSIXct(character())
          } else {
            ensembles$tables$main_log[[nm]] <<- if (nlog) rep(NA_character_, nlog) else character(0)
          }
        }
        target <- names(ensembles$tables$main_log)
      }
      rows <- rows[, target, drop = FALSE]
      rownames(rows) <- NULL
      rows
    }
    append_main_log_snapshot <- function(phase, iteration = NA_integer_, message = "") {
      ensure_main_log_init()
      .ensure_lineage_columns()
      main_serials <- snapshot_main_serials_meta()
      metric_name  <- get0("metric_name", ifnotfound = get0("TARGET_METRIC", ifnotfound = "accuracy", inherits = TRUE), inherits = TRUE)
      ts_now <- Sys.time()
      if (!length(main_serials)) {
        empty_row <- data.frame(
          iteration     = as.integer(iteration),
          phase         = as.character(phase),
          slot          = NA_integer_,
          serial        = NA_character_,
          metric_name   = as.character(metric_name),
          metric_value  = NA_real_,
          current_serial= NA_character_,
          message       = as.character(message),
          timestamp     = ts_now,
          stringsAsFactors = FALSE
        )
        empty_row <- .align_rows_to_main_log(empty_row)
        ensembles$tables$main_log <<- rbind(ensembles$tables$main_log, empty_row)
        saveRDS(ensembles$tables$main_log, ensembles$tables$main_log_path)
        return(invisible())
      }
      vals <- .collect_vals_safe(main_serials, metric_name)
      rows <- data.frame(
        iteration     = rep_len(as.integer(iteration), length(main_serials)),
        phase         = rep_len(as.character(phase),   length(main_serials)),
        slot          = seq_along(main_serials),
        serial        = as.character(main_serials),
        metric_name   = rep_len(as.character(metric_name), length(main_serials)),
        metric_value  = as.numeric(vals$value),
        current_serial= NA_character_,
        message       = rep_len(as.character(message), length(main_serials)),
        timestamp     = rep(ts_now, length(main_serials)),
        stringsAsFactors = FALSE
      )
      if (!is.null(ensembles$tables$lineage) && length(ensembles$tables$lineage)) {
        max_depth <- 0L
        for (v in ensembles$tables$lineage) max_depth <- max(max_depth, length(v))
        if (max_depth > 0L) {
          for (k in seq_len(max_depth)) {
            nm <- sprintf("lineage_%d", k)
            if (!(nm %in% names(rows))) rows[[nm]] <- NA_character_
          }
        }
        for (i in seq_len(nrow(rows))) {
          s <- rows$slot[i]
          rows$current_serial[i] <- .lineage_current(s)
          v <- ensembles$tables$lineage[[s]] %||% character(0)
          if (length(v)) {
            for (k in seq_along(v)) {
              nm <- sprintf("lineage_%d", k)
              rows[[nm]][i] <- v[k]
            }
          }
        }
      }
      rows <- .align_rows_to_main_log(rows)
      ensembles$tables$main_log <<- rbind(ensembles$tables$main_log, rows)
      saveRDS(ensembles$tables$main_log, ensembles$tables$main_log_path)
      invisible()
    }
    # =====================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<  MAIN LOG ENDS  >>>>>>>>>>>>>>>>>>>>>>>>>>>
    # =====================================================================
    
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
      
      # >>> MAIN winners snapshot for SINGLE-RUN
      append_main_log_snapshot(phase = "single_final", iteration = NA_integer_, message = "Single-run MAIN composition")
      
    } 
    else {
      
      # =======================================================================================
      # Branch B: ENSEMBLE MODE (MAIN + TEMP prune/add)
      # =======================================================================================
      
      # ------- logging helpers -------
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
          
          # >>> MAIN winners snapshot BEFORE prune/add
          append_main_log_snapshot(phase = "main_before", iteration = j, message = "Pre TEMP/prune-add MAIN composition")
          
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
          
          # >>> MAIN winners snapshot AFTER prune/add
          append_main_log_snapshot(phase = "main_after", iteration = j, message = "Post TEMP/prune-add MAIN composition")
          
          # refresh metadata mirrors after each iteration
          ensembles <- compose_metadata_views(ensembles)
        }
      }
      
      # final refresh (even if no temp iters)
      ensembles <- compose_metadata_views(ensembles)
      
      # >>> FINAL MAIN winners snapshot
      append_main_log_snapshot(phase = "final", iteration = NA_integer_, message = "Final MAIN composition")
      
      # =======================================================================================
      # NEW: Build grouped metrics from MAIN metadata (no dependency on viewTables/train return)
      # =======================================================================================
      collect_grouped_from_main <- function(ens) {
        perf_rows <- list()
        relev_rows <- list()
        
        # ens$main is a named list of metadata lists like "Ensemble_Main_1_model_k_metadata"
        if (!is.null(ens$main) && length(ens$main)) {
          nm <- names(ens$main)
          for (i in seq_along(ens$main)) {
            md <- ens$main[[i]]
            if (is.null(md) || !is.list(md)) next
            serial <- as.character(md$model_serial_num %||% NA_character_)
            slot   <- suppressWarnings(as.integer(sub(".*model_(\\d+)_metadata$", "\\1", nm[i])))
            if (!is.finite(slot)) slot <- i
            
            # performance_metric: a named list of numerics or sublists; flatten shallow numerics only
            if (!is.null(md$performance_metric) && is.list(md$performance_metric)) {
              for (mn in names(md$performance_metric)) {
                val <- md$performance_metric[[mn]]
                # only take numeric scalars (skip nested lists like accuracy_tuned$details)
                if (is.numeric(val) && length(val) == 1L && is.finite(val)) {
                  perf_rows[[length(perf_rows)+1L]] <- data.frame(
                    slot = slot,
                    serial = serial,
                    metric_name = mn,
                    metric_value = as.numeric(val),
                    stringsAsFactors = FALSE
                  )
                }
              }
            }
            
            # relevance_metric: similar handling
            if (!is.null(md$relevance_metric) && is.list(md$relevance_metric)) {
              for (mn in names(md$relevance_metric)) {
                val <- md$relevance_metric[[mn]]
                if (is.numeric(val) && length(val) == 1L && is.finite(val)) {
                  relev_rows[[length(relev_rows)+1L]] <- data.frame(
                    slot = slot,
                    serial = serial,
                    metric_name = mn,
                    metric_value = as.numeric(val),
                    stringsAsFactors = FALSE
                  )
                }
              }
            }
          }
        }
        
        perf_df  <- if (length(perf_rows))  do.call(rbind, perf_rows)  else NULL
        relev_df <- if (length(relev_rows)) do.call(rbind, relev_rows) else NULL
        
        list(perf_df = perf_df, relev_df = relev_df)
      }
      
      grouped <- collect_grouped_from_main(ensembles)
      
      if (is.null(ensembles$tables)) ensembles$tables <- list()
      if (!is.null(grouped$perf_df) && NROW(grouped$perf_df)) {
        ensembles$tables$performance_grouped_main <- grouped$perf_df
      } else {
        ensembles$tables$performance_grouped_main <- NULL
      }
      if (!is.null(grouped$relev_df) && NROW(grouped$relev_df)) {
        ensembles$tables$relevance_grouped_main <- grouped$relev_df
      } else {
        ensembles$tables$relevance_grouped_main <- NULL
      }
      
    }
    




    
  }
  
  
  
  
}



saveToDisk <- TRUE

if (saveToDisk) {
  # ---- ensure output dir ----
  out_dir <- "artifacts_runs"
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  # ---- locate main ensemble (required) ----
  if (!exists("ensembles", inherits = TRUE) || is.null(ensembles$main_ensemble)) {
    stop("[saveToDisk] ensembles$main_ensemble is missing or NULL; cannot save.")
  }
  main_ensemble_obj <- ensembles$main_ensemble
  
  # ---- collect temp ensembles (optional) ----
  temp_list <- list()
  
  # 1) If your code maintains a list like ensembles$temp_ensembles, grab it
  if (!is.null(ensembles$temp_ensembles) && length(ensembles$temp_ensembles) > 0) {
    temp_list <- ensembles$temp_ensembles
  }
  
  # 2) Also sweep the global env for any temp_ensemble_<n> variables and add them
  temp_sym_names <- ls(envir = .GlobalEnv, pattern = "^temp_ensemble_\\d+$")
  if (length(temp_sym_names)) {
    for (nm in temp_sym_names) {
      obj <- get(nm, envir = .GlobalEnv)
      # avoid duplicate names from ensembles$temp_ensembles
      if (is.null(names(temp_list)) || !(nm %in% names(temp_list))) {
        temp_list[[nm]] <- obj
      }
    }
  }
  
  # ---- bundle payload ----
  ensemble_results <- list(
    main_ensemble  = main_ensemble_obj,
    temp_ensembles = temp_list  # may be an empty list, which is fine
  )
  
  # ---- filename logic (no overwrite) ----
  base_file_name <- file.path(out_dir, "ensemble_results")
  
  generate_new_file_name <- function(base_name) {
    i <- 1L
    candidate <- paste0(base_name, "_", i, ".rds")
    while (file.exists(candidate)) {
      i <- i + 1L
      candidate <- paste0(base_name, "_", i, ".rds")
    }
    candidate
  }
  
  file_name <- if (file.exists(paste0(base_file_name, ".rds"))) {
    generate_new_file_name(base_file_name)
  } else {
    paste0(base_file_name, ".rds")
  }
  
  # ---- save ----
  saveRDS(ensemble_results, file_name)
  cat("Data saved to file:", file_name, "\n")
  cat(sprintf("[saveToDisk] Included %d temp ensemble(s).\n", length(temp_list)))
}
