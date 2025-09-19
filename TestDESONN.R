source("DESONN.R")
source("utils/utils.R")
source("utils/bootstrap_metadata.R")
library(readxl)
library(dplyr)
set.seed(30)
# # Define parameters
## =========================
## Classification mode
## =========================
# CLASSIFICATION_MODE <- "multiclass"   # "binary" | "multiclass" | "regression"
CLASSIFICATION_MODE <- "binary"
# CLASSIFICATION_MODE <- "regression"

init_method <- "he" #variance_scaling" #glorot_uniform" #"orthogonal" #"orthogonal" #lecun" #xavier"
optimizer <- "adagrad" #"lamb" #ftrl #nag #"sgd" #NULL "rmsprop" #adam #sgd_momentum #lookahead #adagrad
lookahead_step <- 5
batch_normalize_data <- TRUE
shuffle_bn <- FALSE
gamma_bn <- .6
beta_bn <- .6
epsilon_bn <- 1e-6  # Increase for numerical stability
momentum_bn <- 0.9 # Improved convergence
is_training_bn <- TRUE
beta1 <- .9 # Standard Adam value
beta2 <- 0.8 # Slightly lower for better adaptability
# lr <- .122
lr <- .125
lr_decay_rate  <- 0.5
lr_decay_epoch <- 20
lr_min <- 1e-5
lambda <- 0.00028
# lambda <- 0.00013
num_epochs <- 200
validation_metrics <- TRUE
test_metrics <- TRUE
custom_scale <- 1.04349



ML_NN <- TRUE

learnOnlyTrainingRun <- FALSE
update_weights <- TRUE
update_biases <- TRUE

if (CLASSIFICATION_MODE == "binary") {
  use_biases <- TRUE
} else if (CLASSIFICATION_MODE == "regression") {
  use_biases <- TRUE
}
# hidden_sizes <- NULL
hidden_sizes <- c(64, 32)

# Activation functions applied in forward pass during prediction | predict(). # hidden layers + output layer
if (CLASSIFICATION_MODE == "binary") {
  activation_functions <- list(relu, relu, sigmoid)
  
} else if (CLASSIFICATION_MODE == "multiclass") {
  activation_functions <- list(relu, relu, softmax)
  
} else if (CLASSIFICATION_MODE == "regression") {
  activation_functions <- list(relu, relu, identity)
}


# Activation functions applied in forward pass during training | learn() # You can keep them the same as predict() or customize, e.g. list(relu, selu, sigmoid) 
activation_functions_learn <- activation_functions
epsilon <- 1e-7
loss_type <- "CrossEntropy" #NULL #'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'

dropout_rates <- list(0.10, 0.00) # NULL for output layer


threshold_function <- tune_threshold_accuracy
# threshold <- .98  # Classification threshold (not directly used in Random Forest)

dropout_rates_learn <- dropout_rates

num_layers <- length(hidden_sizes) + 1
output_size <- 1  # For binary classification

## Kaggle Data
runData <- FALSE
if(runData){
  library(reticulate)
  
  # Ensure we're using reticulate's default Python env (created above)
  # If you prefer, you can create/use your own venv and call use_virtualenv() instead.
  
  # 1) Install kagglehub into the active Python used by reticulate
  py_install("kagglehub", pip = TRUE)
  
  # 2) Download with kagglehub from Python, returning the path back to R
  res <- py_run_string("
import kagglehub
p = kagglehub.dataset_download('matiflatif/walmart-complete-stocks-dataweekly-updated')
print('Downloaded to:', p)
path = p
")
  src_path <- res$path
  
  # 3) Copy the downloaded dataset into your target folder
  dst_path <- "C:/Users/wfky1/Desktop/DESONN/data"
  dir.create(dst_path, showWarnings = FALSE, recursive = TRUE)
  # Copy everything (preserves directories); overwrite existing
  file.copy(from = list.files(src_path, full.names = TRUE),
            to   = dst_path,
            recursive = TRUE, overwrite = TRUE)
  
  cat("Copied dataset from:\n ", src_path, "\n→\n ", dst_path, "\n")
}


## -------------------------
## Load the dataset
## -------------------------
if (CLASSIFICATION_MODE == "binary") {
  data <- read.csv("data/heart_failure_clinical_records.csv")
  dependent_variable <- "DEATH_EVENT"
} else if (CLASSIFICATION_MODE == "multiclass") {
  data <- read.csv("data/train_multiclass_customer_segmentation.csv")
  dependent_variable <- "Segmentation"
} else if (CLASSIFICATION_MODE == "regression") {
  data <- read.csv("data/WMT_1970-10-01_2025-03-15.csv")
  # --- NEW: enforce time order + create future target (close[t+1]) ---
  stopifnot("date" %in% names(data))
  data <- data %>% arrange(date)
  data <- data %>% mutate(future_close = dplyr::lead(close, 1L))
  data <- data %>% filter(!is.na(future_close))
  
  # --- NEW: set dependent variable to future_close ---
  dependent_variable <- "future_close"
} else {
  stop("CLASSIFICATION_MODE must be 'binary' or 'multiclass'")
}


## Quick NA check
na_count <- sum(is.na(data))
cat("[split] NA count:", na_count, "\n")


# Assuming there are no missing values, or handle them if they exist
# Convert categorical variables to factors if any
data <- data %>% mutate(across(where(is.character), as.factor))



## -------------------------
## Features/labels (full)
## -------------------------
input_columns <- setdiff(colnames(data), dependent_variable)
Rdata  <- data[, input_columns, drop = FALSE]
labels <- data[, dependent_variable, drop = FALSE]  # keep as data.frame (1 col)

if (CLASSIFICATION_MODE %in% c("binary", "regression")) {
  input_size  <- ncol(Rdata)
  output_size <- 1L
}

if (!ML_NN) {
  N <- input_size + output_size
} else {
  N <- input_size + sum(hidden_sizes) + output_size
}

reduce_data <- TRUE

if (CLASSIFICATION_MODE == "regression" && reduce_data) {
  stopifnot(is.character(dependent_variable), length(dependent_variable) == 1)
  stopifnot(dependent_variable %in% names(data))
  
  # Keep only raw OHLCV + date, plus the dependent variable (future_close)
  base_keep <- c("date", "open", "high", "low", "close", "volume")
  keep_cols <- unique(c(base_keep, dependent_variable))
  
  data_reduced <- data %>%
    dplyr::select(dplyr::any_of(keep_cols))
  
  # Split into X (features at time t) and y (close at t+1)
  X <- data_reduced %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y <- data_reduced[[dependent_variable]]
  
  # Safety checks (no leakage)
  if ("adj_close" %in% names(X)) {
    stop("Leak detected: 'adj_close' showed up in X unexpectedly.")
  }
  if (!is.null(y) && anyNA(y)) {
    warning("Target 'y' contains NA values.")
  }
  
  # Optional: make 'date' numeric if present
  if ("date" %in% names(X)) {
    d <- X[["date"]]
    if (inherits(d, "POSIXt")) {
      X[["date"]] <- as.numeric(as.Date(d))
    } else if (inherits(d, "Date")) {
      X[["date"]] <- as.numeric(d)
    } else {
      suppressWarnings({ parsed <- as.Date(d) })
      if (!all(is.na(parsed))) {
        X[["date"]] <- as.numeric(parsed)
      } # else leave as-is
    }
  }
}

# Keep your existing fallback exactly as written
if (CLASSIFICATION_MODE == "regression" && reduce_data) {
  X <- data_reduced %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y <- data_reduced %>% dplyr::select(dplyr::all_of(dependent_variable))
} else {
  X <- data %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y <- data %>% dplyr::select(dplyr::all_of(dependent_variable))
}

colname_y <- colnames(y)

# ensure downstream uses lagged setup (place right after your X/y fallback)
# Rdata  <- X
# labels <- y

if (CLASSIFICATION_MODE == "binary") {
  numeric_columns <- c('age','creatinine_phosphokinase','ejection_fraction',
                       'platelets','serum_creatinine','serum_sodium','time')
} else if (CLASSIFICATION_MODE == "multiclass") {
  numeric_columns <- c("Age","Work_Experience","Family_Size")
} else if (CLASSIFICATION_MODE == "regression") {
  # Predicting future_close (t+1) from today's OHLCV (t)
  numeric_columns <- c("date","open","high","low","close","volume")
  
}





## -------------------------
## Split selector
## -------------------------
USE_TIME_SPLIT <- TRUE  # toggle here: TRUE=new chrono split, FALSE=old random split

if (USE_TIME_SPLIT) {
  ## -------------------------
  ## Time-ordered split
  ## -------------------------
  stopifnot(nrow(X) == nrow(y))
  total_num_samples <- nrow(X)
  
  p_train <- 0.70
  p_val   <- 0.15
  
  num_training_samples   <- max(1L, floor(p_train * total_num_samples))
  num_validation_samples <- max(1L, floor(p_val   * total_num_samples))
  num_test_samples       <- max(0L, total_num_samples - num_training_samples - num_validation_samples)
  
  train_indices      <- seq_len(num_training_samples)
  validation_indices <- if (num_validation_samples > 0L)
    seq(from = max(train_indices) + 1L,
        length.out = num_validation_samples)
  else integer()
  test_indices       <- if (num_test_samples > 0L)
    seq(from = max(c(train_indices, validation_indices)) + 1L,
        length.out = num_test_samples)
  else integer()
  
  X_train      <- X[train_indices,      , drop = FALSE]; y_train      <- y[train_indices,      , drop = FALSE]
  X_validation <- X[validation_indices, , drop = FALSE]; y_validation <- y[validation_indices, , drop = FALSE]
  X_test       <- X[test_indices,       , drop = FALSE]; y_test       <- y[test_indices,       , drop = FALSE]
  
  cat(sprintf("[SPLIT chrono] train=%d val=%d test=%d\n",
              nrow(X_train), nrow(X_validation), nrow(X_test)))
} else {
  ## -------------------------
  ## Old robust random split
  ## -------------------------
  total_num_samples <- nrow(X)
  desired_val  <- 800L
  desired_test <- 800L
  
  num_validation_samples <- min(desired_val,  floor(total_num_samples / 3))
  num_test_samples       <- min(desired_test, floor((total_num_samples - num_validation_samples) / 2))
  num_training_samples   <- total_num_samples - num_validation_samples - num_test_samples
  
  indices <- sample.int(total_num_samples)
  train_indices      <- indices[seq_len(num_training_samples)]
  validation_indices <- indices[seq(from = num_training_samples + 1L,
                                    length.out = num_validation_samples)]
  test_indices       <- indices[seq(from = num_training_samples + num_validation_samples + 1L,
                                    length.out = num_test_samples)]
  
  X_train      <- X[train_indices,      , drop = FALSE]; y_train      <- y[train_indices,      , drop = FALSE]
  X_validation <- X[validation_indices, , drop = FALSE]; y_validation <- y[validation_indices, , drop = FALSE]
  X_test       <- X[test_indices,       , drop = FALSE]; y_test       <- y[test_indices,       , drop = FALSE]
  
  cat(sprintf("[SPLIT random] train=%d val=%d test=%d\n",
              nrow(X_train), nrow(X_validation), nrow(X_test)))
}



## IMPORTANT:
## When you train, pass X_train / y_train — NOT the full Rdata/labels.
## Likewise, for validation, forward X_validation and compare with y_validation.

# ===== OPTIONAL LOG/CLIP TRANSFORMATION (kept from your "below" block; commented) =====
# Apply log1p to avoid issues with zero values (log1p(x) = log(1 + x))
# X_train$creatinine_phosphokinase      <- pmin(X_train$creatinine_phosphokinase, 3000)
# X_validation$creatinine_phosphokinase <- pmin(X_validation$creatinine_phosphokinase, 3000)
# X_test$creatinine_phosphokinase       <- pmin(X_test$creatinine_phosphokinase, 3000)

# ------------------------------------------------------------
# BINARY PATH (untouched)  vs  MULTICLASS PATH (make numeric + labels)
# ------------------------------------------------------------
if (CLASSIFICATION_MODE == "binary") {
  
  # $$$$$$$$$$$$$ Feature scaling without leakage (standardization first)
  X_train_scaled <- scale(X_train)
  center <- attr(X_train_scaled, "scaled:center")
  scale_ <- attr(X_train_scaled, "scaled:scale")
  
  X_validation_scaled <- scale(X_validation, center = center, scale = scale_)
  X_test_scaled       <- scale(X_test,       center = center, scale = scale_)
  
  # $$$$$$$$$$$$$ Further rescale to prevent exploding activations (keep parity)
  max_val <- suppressWarnings(max(abs(X_train_scaled)))
  if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1

  X_train_scaled      <- X_train_scaled      / max_val
  X_validation_scaled <- X_validation_scaled / max_val
  X_test_scaled       <- X_test_scaled       / max_val
  
  # ==============================================================
  # Choose whether to use scaled or raw data for NN training
  # ==============================================================
  scaledData <- TRUE   # <<<<<< set to FALSE to use raw data
  
  if (isTRUE(scaledData)) {
    X <- as.matrix(X_train_scaled)
    y <- as.matrix(y_train)
    
    X_validation <- as.matrix(X_validation_scaled)
    y_validation <- as.matrix(y_validation)
    
    X_test <- as.matrix(X_test_scaled)
    y_test <- as.matrix(y_test)
  } else {
    X <- as.matrix(X_train)
    y <- as.matrix(y_train)
    
    X_validation <- as.matrix(X_validation)
    y_validation <- as.matrix(y_validation)
    
    X_test <- as.matrix(X_test)
    y_test <- as.matrix(y_test)
  }
  
  colnames(y) <- colname_y
  
  # ----- diagnostics (binary has no one-hot by design) -----
  cat("=== Unscaled Rdata summary (X_train) ===\n")
  print(summary(as.vector(as.matrix(X_train))))
  cat("First 5 rows of unscaled X_train:\n")
  print(as.matrix(X_train)[1:5, 1:min(5, ncol(X_train)), drop = FALSE])
  
  cat("=== Scaled Rdata summary (X_train_scaled) ===\n")
  print(summary(as.vector(as.matrix(X_train_scaled))))
  cat("First 5 rows of scaled X_train_scaled:\n")
  print(as.matrix(X_train_scaled)[1:5, 1:min(5, ncol(X_train_scaled)), drop = FALSE])
  
  cat("Dimensions of scaled sets:\n")
  cat("Training:",   dim(X), "\n")
  cat("Validation:", dim(X_validation), "\n")
  cat("Test:",       dim(X_test), "\n")
  
  cat("Any NAs in scaled sets:\n")
  cat("Training:",   anyNA(X), "\n")
  cat("Validation:", anyNA(X_validation), "\n")
  cat("Test:",       anyNA(X_test), "\n")
  
} else if (CLASSIFICATION_MODE == "multiclass") {
  
  cat("\n==================== [MC] START ====================\n")
  
  # ---------- A) Row-ID setup so we can align y to X ----------
  # Ensure row names on split frames reflect original row indices
  if (is.null(rownames(X_train)))      rownames(X_train)      <- as.character(train_indices)
  if (is.null(rownames(X_validation))) rownames(X_validation) <- as.character(validation_indices)
  if (is.null(rownames(X_test)))       rownames(X_test)       <- as.character(test_indices)
  
  # ---------- A0) Numeric imputation with TRAIN medians (prevents NA drops downstream) ----------
  impute_with_train_median <- function(df_train, df_other) {
    num_cols <- names(df_train)[vapply(df_train, is.numeric, TRUE)]
    for (nm in num_cols) {
      med <- suppressWarnings(median(df_train[[nm]], na.rm = TRUE))
      if (!is.finite(med) || is.na(med)) med <- 0
      if (nm %in% names(df_train))   df_train[[nm]][is.na(df_train[[nm]])] <- med
      if (nm %in% names(df_other))   df_other[[nm]][is.na(df_other[[nm]])] <- med
    }
    list(train = df_train, other = df_other)
  }
  tmp <- impute_with_train_median(X_train, X_validation); X_train <- tmp$train; X_validation <- tmp$other
  tmp <- impute_with_train_median(X_train, X_test);       X_test  <- tmp$other
  
  # Quick predictor type scan
  pred_types <- vapply(X_train, function(col) {
    if (is.numeric(col)) "numeric"
    else if (is.factor(col)) "factor"
    else class(col)[1]
  }, character(1))
  n_num <- sum(pred_types == "numeric")
  n_fac <- sum(pred_types == "factor")
  cat("[mc] predictors summary: numeric =", n_num, " | factor =", n_fac, " | total =", length(pred_types), "\n")
  if (n_fac > 0) {
    cat("[mc] factor columns:\n"); print(names(which(pred_types == "factor")))
  }
  
  all_numeric <- all(pred_types == "numeric")
  cat("[mc] all_numeric predictors? ->", all_numeric, "\n")
  
  # ---------- B) Build X (features) ----------
  if (all_numeric) {
    cat("[mc] PATH A: numeric-only (no one-hot for X)\n")
    
    X_train_scaled <- scale(as.data.frame(X_train))
    center <- attr(X_train_scaled, "scaled:center")
    scale_ <- attr(X_train_scaled, "scaled:scale"); scale_[!is.finite(scale_) | scale_ == 0] <- 1
    
    X_validation_scaled <- sweep(sweep(as.matrix(as.data.frame(X_validation)), 2, center, "-"), 2, scale_, "/")
    X_test_scaled       <- sweep(sweep(as.matrix(as.data.frame(X_test)),       2, center, "-"), 2, scale_, "/")
    
    X            <- as.matrix(X_train_scaled)
    X_validation <- as.matrix(X_validation_scaled)
    X_test       <- as.matrix(X_test_scaled)
    
  } else {
    cat("[mc] PATH B: one-hot features with model.matrix() + consistent TRAIN terms + row alignment\n")
    
    # Make factor NAs explicit so rows are preserved as levels
    X_train_f      <- dplyr::mutate(X_train,      dplyr::across(where(is.factor), ~forcats::fct_explicit_na(., "(Missing)")))
    X_validation_f <- dplyr::mutate(X_validation, dplyr::across(where(is.factor), ~forcats::fct_explicit_na(., "(Missing)")))
    X_test_f       <- dplyr::mutate(X_test,       dplyr::across(where(is.factor), ~forcats::fct_explicit_na(., "(Missing)")))
    
    # Build design on TRAIN ONLY to lock columns
    mm_terms <- terms(~ . - 1, data = X_train_f)
    X_train_mm      <- model.matrix(mm_terms, data = X_train_f)
    X_validation_mm <- model.matrix(mm_terms, data = X_validation_f)
    X_test_mm       <- model.matrix(mm_terms, data = X_test_f)
    
    cat("[mc] dim(X_train_mm)=", paste(dim(X_train_mm), collapse="×"),
        " | dim(X_val_mm)=", paste(dim(X_validation_mm), collapse="×"),
        " | dim(X_test_mm)=", paste(dim(X_test_mm), collapse="×"), "\n")
    
    # Scale with train stats
    X_train_scaled <- scale(X_train_mm)
    center <- attr(X_train_scaled, "scaled:center")
    scale_ <- attr(X_train_scaled, "scaled:scale"); scale_[!is.finite(scale_) | scale_ == 0] <- 1
    
    X_validation_scaled <- sweep(sweep(X_validation_mm, 2, center, "-"), 2, scale_, "/")
    X_test_scaled       <- sweep(sweep(X_test_mm,       2, center, "-"), 2, scale_, "/")
    
    X            <- as.matrix(X_train_scaled)
    X_validation <- as.matrix(X_validation_scaled)
    X_test       <- as.matrix(X_test_scaled)
  }
  
  # Stabilize magnitude (parity with your binary path)
  max_val <- suppressWarnings(max(abs(X)))
  if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1
  X            <- X            / max_val
  X_validation <- X_validation / max_val
  X_test       <- X_test       / max_val
  
  cat("[mc] dim(X) train/val/test: ",
      paste(dim(X), collapse="×"), " / ",
      paste(dim(X_validation), collapse="×"), " / ",
      paste(dim(X_test), collapse="×"), "\n")
  
  # ---------- C) Align y to X rows (train/val/test) ----------
  pull_y_vec <- function(obj) {
    if (is.matrix(obj)) as.vector(obj[, 1, drop = TRUE])
    else if (is.data.frame(obj)) as.vector(obj[[1]])
    else as.vector(obj)
  }
  align_y_to_X <- function(X_mat, y_df, idx_vec) {
    kept_rn <- rownames(X_mat)
    if (is.null(kept_rn)) kept_rn <- as.character(idx_vec[seq_len(nrow(X_mat))])
    pos <- match(kept_rn, as.character(idx_vec))
    if (anyNA(pos)) stop("[mc][align] Could not map X rownames to original indices.")
    pull_y_vec(y_df)[pos]
  }
  
  y_vec_tr <- align_y_to_X(X,            y_train,      train_indices)
  y_vec_va <- align_y_to_X(X_validation, y_validation, validation_indices)
  y_vec_te <- align_y_to_X(X_test,       y_test,       test_indices)
  
  cat("[mc] lens y_vec (aligned): train/val/test = ",
      length(y_vec_tr), "/", length(y_vec_va), "/", length(y_vec_te), "\n")
  
  # ---------- D) One-hot labels (shared levels from full dataset) ----------
  y_full_vec  <- pull_y_vec(y)
  levels_y    <- levels(factor(y_full_vec))
  output_size <- length(levels_y)
  cat("[mc] class levels (", output_size, "): ", paste(levels_y, collapse=", "), "\n", sep="")
  
  to_one_hot <- function(v, lvls) {
    idx <- match(v, lvls)
    m <- matrix(0L, nrow = length(v), ncol = length(lvls))
    m[cbind(seq_along(idx), idx)] <- 1L
    colnames(m) <- lvls
    m
  }
  
  y_train_one_hot_aligned <- to_one_hot(y_vec_tr, levels_y)
  y_validation_one_hot    <- to_one_hot(y_vec_va, levels_y)
  y_test_one_hot          <- to_one_hot(y_vec_te, levels_y)
  
  # Back-compat alias if other code references y_train_one_hot
  y_train_one_hot <- y_train_one_hot_aligned
  
  cat("[mc] dim(y one-hot) train/val/test: ",
      paste(dim(y_train_one_hot_aligned), collapse="×"), " / ",
      paste(dim(y_validation_one_hot), collapse="×"), " / ",
      paste(dim(y_test_one_hot), collapse="×"), "\n")
  
  # Keep original y matrices too
  y            <- as.matrix(y_train); colnames(y) <- colname_y
  y_validation <- as.matrix(y_validation)
  y_test       <- as.matrix(y_test)
  
  # ---------- E) Single source of truth for training ----------
  Rdata       <- X                               # training features
  labels      <- y_train_one_hot_aligned         # training labels aligned to X
  input_size  <- ncol(Rdata)
  output_size <- ncol(labels)
  
  cat("[mc] FINAL dim(Rdata)=", paste(dim(Rdata), collapse="×"),
      " | dim(labels)=", paste(dim(labels), collapse="×"),
      " | input_size=", input_size, " | output_size=", output_size, "\n")
  
  if (nrow(Rdata) != nrow(labels)) {
    stop(sprintf("[mc][FATAL] Row mismatch persists: nrow(Rdata)=%d vs nrow(labels)=%d.\nCheck alignment prints above.",
                 nrow(Rdata), nrow(labels)))
  }
  
  # ---------- F) Recompute N now that sizes are final ----------
  if (!ML_NN) {
    N <- input_size + output_size
  } else {
    N <- input_size + sum(hidden_sizes) + output_size
  }
  cat("[mc] N =", N, "\n")
  
  cat("===================== [MC] END =====================\n\n")
} else if (CLASSIFICATION_MODE == "regression") {
  cat("\n==================== [REG] START ====================\n")
  
  # ---------- A) Optional: make date numeric if present ----------
  make_date_numeric <- function(df) {
    if (!"date" %in% names(df)) return(df)
    d <- df[["date"]]
    if (inherits(d, "POSIXt")) {
      df[["date"]] <- as.numeric(as.Date(d))       # days since epoch
    } else if (inherits(d, "Date")) {
      df[["date"]] <- as.numeric(d)                # days since epoch
    } else {
      suppressWarnings({ parsed <- as.Date(d) })
      if (all(is.na(parsed))) {
        warning("[reg] 'date' column could not be parsed; converting to NA (will be imputed).")
        df[["date"]] <- NA_real_
      } else {
        df[["date"]] <- as.numeric(parsed)
      }
    }
    df
  }
  X_train      <- make_date_numeric(X_train)
  X_validation <- make_date_numeric(X_validation)
  X_test       <- make_date_numeric(X_test)
  
  # ---------- B) Numeric imputation with TRAIN medians (no leakage) ----------
  impute_with_train_median <- function(df_train, df_other) {
    num_cols <- names(df_train)[vapply(df_train, is.numeric, TRUE)]
    for (nm in num_cols) {
      med <- suppressWarnings(median(df_train[[nm]], na.rm = TRUE))
      if (!is.finite(med) || is.na(med)) med <- 0
      if (nm %in% names(df_train)) df_train[[nm]][is.na(df_train[[nm]])] <- med
      if (nm %in% names(df_other)) df_other[[nm]][is.na(df_other[[nm]])] <- med
    }
    list(train = df_train, other = df_other)
  }
  tmp <- impute_with_train_median(X_train, X_validation); X_train <- tmp$train; X_validation <- tmp$other
  tmp <- impute_with_train_median(X_train, X_test);       X_test  <- tmp$other
  
  # ---------- C) Feature scaling without leakage ----------
  X_train_df <- as.data.frame(X_train)
  X_val_df   <- as.data.frame(X_validation)
  X_test_df  <- as.data.frame(X_test)
  
  num_mask <- vapply(X_train_df, is.numeric, TRUE)
  if (!any(num_mask)) stop("[reg] No numeric predictors found after preprocessing.")
  
  X_train_num <- as.matrix(X_train_df[, num_mask, drop = FALSE])
  X_val_num   <- as.matrix(X_val_df[,   num_mask, drop = FALSE])
  X_test_num  <- as.matrix(X_test_df[,  num_mask, drop = FALSE])
  
  X_train_scaled <- scale(X_train_num)
  center <- attr(X_train_scaled, "scaled:center")
  scale_ <- attr(X_train_scaled, "scaled:scale"); scale_[!is.finite(scale_) | scale_ == 0] <- 1
  
  X_validation_scaled <- sweep(sweep(X_val_num,  2, center, "-"), 2, scale_, "/")
  X_test_scaled       <- sweep(sweep(X_test_num, 2, center, "-"), 2, scale_, "/")
  
  # ---------- D) Further rescale to prevent exploding activations ----------
  max_val <- suppressWarnings(max(abs(X_train_scaled)))
  if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1
  
  # --- Save training-time preprocessing for predict() ---
  feature_names <- colnames(X_train_num)  # exact order used to train
  train_medians <- vapply(as.data.frame(X_train_df[, feature_names, drop = FALSE]),
                          function(col) suppressWarnings(median(col, na.rm = TRUE)), numeric(1))
  train_medians[!is.finite(train_medians)] <- 0
  
  # ---------- E) y handling (train-only): optional z-score ----------
  SCALE_Y_WITH_ZSCORE <- FALSE  # TRUE = train on z-scored y; FALSE = raw y
  
  `%||%` <- get0("%||%", inherits = TRUE, ifnotfound = function(x, y) if (is.null(x)) y else x)
  
  # 1) Robust y extraction (vector)
  y_vec <- if (is.matrix(y_train) || is.data.frame(y_train)) {
    as.numeric(y_train[, 1])
  } else {
    as.numeric(y_train)
  }
  stopifnot(length(y_vec) == NROW(X_train))
  
  # 2) Decide transform type (train-only)
  is_regression <- identical(tolower(CLASSIFICATION_MODE %||% "regression"), "regression")
  use_zscore    <- is_regression && is.numeric(y_vec) && isTRUE(SCALE_Y_WITH_ZSCORE)

  if (use_zscore) {
    y_center <- mean(y_vec, na.rm = TRUE)
    y_scale  <- stats::sd(y_vec, na.rm = TRUE)
    if (!is.finite(y_scale) || y_scale == 0) y_scale <- 1
    
    y_vec_scaled <- (y_vec - y_center) / y_scale
    
    target_transform <- list(
      type   = "zscore",
      params = list(center = y_center, scale = y_scale)
    )
    y_trained_scaled <- TRUE
  } else {
    # Train on raw y (predict-only does identity inverse)
    y_vec_scaled <- y_vec
    target_transform <- list(
      type   = "identity",
      params = list(center = 0, scale = 1)
    )
    y_trained_scaled <- FALSE
  }
  

  # 3) one-col numeric matrix for training
  y <- matrix(as.numeric(y_vec_scaled), ncol = 1L)
  storage.mode(y) <- "double"
  colnames(y) <- colname_y
  
  # ---------- F) SINGLE source of truth object (no meta writes) ----------
  # Build preprocessScaledData containing BOTH X and y handling so predict-only can invert correctly.
  center_vec <- setNames(as.numeric(center[feature_names]),        feature_names)
  scale_vec  <- setNames(as.numeric(scale_[feature_names]),        feature_names)
  med_vec    <- setNames(as.numeric(train_medians[feature_names]), feature_names)
  
  preprocessScaledData <- list(
    # X preprocess
    feature_names     = as.character(feature_names),
    center            = center_vec,
    scale             = scale_vec,
    max_val           = as.numeric(max_val),
    divide_by_max_val = TRUE,
    train_medians     = med_vec,
    date_policy       = "as.Date -> numeric days; char parsed via as.Date()",
    used_scaled_X     = TRUE,
    scaler            = "standardize+divide_by_max",
    imputer           = "train_median",
    input_size        = ncol(X_train_num),
    
    # y / target transform lives HERE
    target_transform  = target_transform,      # identity or zscore
    y_trained_scaled  = isTRUE(y_trained_scaled)
  )
  
  # Make available to outer scope (for your writer / store_metadata)
  assign("preprocessScaledData", preprocessScaledData, inherits = TRUE)
  assign("target_transform",     target_transform,     inherits = TRUE)
  
  # ---------- (Optional but recommended) mirror into meta BEFORE you write the RDS ----------
  # This does NOT write to disk here; it only ensures the in-memory `meta` will contain the fields
  # when your existing store_metadata() persists it.
  # ---------- Add model-critical configs into meta ----------
  if (exists("meta", inherits = TRUE)) {
    try({
      meta$preprocessScaledData <- preprocessScaledData
      meta$target_transform     <- target_transform
      meta$train_target_center  <- target_transform$params$center %||% NA_real_
      meta$train_target_scale   <- target_transform$params$scale  %||% NA_real_
      
      # ✅ NEW: store activations and dropout etc.
      meta$activation_functions <- activation_functions
      meta$dropout_rates        <- dropout_rates
      meta$hidden_sizes         <- self$hidden_sizes %||% meta$hidden_sizes
      meta$output_size          <- self$output_size %||% meta$output_size
      meta$ML_NN                <- self$ML_NN %||% meta$ML_NN
    }, silent = TRUE)
  }
  
  
  # ---------- (Helpful debug) ----------
  if (isTRUE(get0("DEBUG_PREDICT", inherits = TRUE, ifnotfound = TRUE))) {
    if (use_zscore) {
      cat(sprintf("[y-train] using zscore: center=%.6f scale=%.6f | y sd(raw)=%.6f sd(scaled)=%.6f\n",
                  y_center, y_scale, stats::sd(y_vec), stats::sd(y_vec_scaled)))
    } else {
      cat(sprintf("[y-train] using identity target transform | y sd=%.6f\n", stats::sd(y_vec)))
    }
  }

  # `y` is now ready for training; `preprocessScaledData`/`target_transform` are ready for store_metadata().
  
  
  
  # ---------- G) Feed scaled inputs to the NN (divide-by-max applied) ----------
  X_train_scaled      <- X_train_scaled      / max_val
  X_validation_scaled <- X_validation_scaled / max_val
  X_test_scaled       <- X_test_scaled       / max_val
  
  scaledData <- TRUE  # use scaled inputs by default
  if (isTRUE(scaledData)) {
    X <- as.matrix(X_train_scaled)
    # IMPORTANT: keep y computed above (possibly scaled) — do NOT overwrite with raw y_train
    X_validation <- as.matrix(X_validation_scaled)
    X_test       <- as.matrix(X_test_scaled)
  } else {
    X <- as.matrix(X_train_num)
    X_validation <- as.matrix(X_val_num)
    X_test       <- as.matrix(X_test_num)
  }
  
  # Ensure y is 1 column numeric
  if (ncol(y) != 1L) {
    y <- matrix(as.numeric(y[, 1]), ncol = 1L)
  } else {
    storage.mode(y) <- "double"
  }
  colnames(y) <- colname_y
  
  # ---------- H) Diagnostics ----------
  cat("=== [reg] Unscaled X_train (numeric subset) summary ===\n")
  print(summary(as.vector(X_train_num)))
  cat("First 5 rows of unscaled numeric X_train:\n")
  print(X_train_num[1:5, 1:min(5, ncol(X_train_num)), drop = FALSE])
  
  cat("=== [reg] Scaled X_train summary ===\n")
  print(summary(as.vector(X)))
  cat("First 5 rows of scaled X (train):\n")
  print(X[1:5, 1:min(5, ncol(X)), drop = FALSE])
  
  cat("[reg] Dimensions (train/val/test):\n")
  cat("X:",            paste(dim(X), collapse="×"),          "\n")
  cat("X_validation:", paste(dim(X_validation), collapse="×"), "\n")
  cat("X_test:",       paste(dim(X_test), collapse="×"),      "\n")
  cat("[reg] Any NAs?  train:", anyNA(X),
      "  val:", anyNA(X_validation),
      "  test:", anyNA(X_test), "\n")
  
  # ---------- I) Final wiring into trainer ----------
  Rdata       <- X
  labels      <- y
  input_size  <- ncol(Rdata)
  output_size <- 1L
  
  cat("[reg] FINAL dim(Rdata)=", paste(dim(Rdata), collapse="×"),
      " | dim(labels)=", paste(dim(labels), collapse="×"),
      " | input_size=", input_size, " | output_size=", output_size, "\n")
  
  if (nrow(Rdata) != nrow(labels)) {
    stop(sprintf("[reg][FATAL] Row mismatch: nrow(Rdata)=%d vs nrow(labels)=%d.", nrow(Rdata), nrow(labels)))
  }

  if (!ML_NN) {
    N <- input_size + output_size
  } else {
    N <- input_size + sum(hidden_sizes) + output_size
  }
  cat("[reg] N =", N, "\n")
  cat("==================== [REG] END ====================\n\n")
}



if (CLASSIFICATION_MODE == "binary"){
  preprocessScaledData <- NULL
}else if (CLASSIFICATION_MODE == "multiclass") {
  preprocessScaledData <- NULL
}


if (CLASSIFICATION_MODE == "multiclass") {
  input_size  <- ncol(Rdata)                    # after model.matrix/processing
  output_size <- ncol(y_train_one_hot_aligned)  # number of classes (fixed name)
}


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
reg_type = "L1" #"Max_Norm" #"L2" #Max_Norm" #"Group_Lasso" #"L1_L2"

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
do_ensemble         <- FALSE
num_networks        <- 1L
num_temp_iterations <- 0L   # ignored when do_ensemble = FALSE
#
## SCENARIO B: Single-run, MULTI-MODEL (no ensemble)
# do_ensemble         <- FALSE
# num_networks        <- 30L          # e.g., run 5 models in one DESONN instance
# num_temp_iterations <- 0L
#
## SCENARIO C: Main ensemble only (no TEMP/prune-add)
# do_ensemble         <- TRUE
# num_networks        <- 2L          # example main size
# num_temp_iterations <- 0L
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

prepare_disk_only <- FALSE
# prepare_disk_only <- TRUE

prepare_disk_only <- get0("prepare_disk_only", ifnotfound = FALSE)  # one-shot RDS export helper

# Flag specific to disk-only prepare / selection
prepare_disk_only_FROM_RDS <- FALSE
prepare_disk_only_FROM_RDS <- get0("prepare_disk_only_FROM_RDS", ifnotfound = FALSE)
# Modes:

MODE <- "train"
# num_epochs <- 3
# MODE <- "predict:stateless"
MODE <- get0("MODE", ifnotfound = "train")
PREDICT_ONLY_FROM_RDS <- TRUE
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
.BM_DIR <- "artifacts"
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
INPUT_SPLIT    <- "test"   # or "test" / "train" / "auto"
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
############################################################
# PREDICT-ONLY MODE (!train) — cleaned & fixed for regression
############################################################
if (!train) {
  cat("Predict-only mode (train == FALSE).\n")
  
  # ------------------------------------------------------------
  # Common toggles / helpers
  # ------------------------------------------------------------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  DEBUG_PREDICT       <- isTRUE(get0("DEBUG_PREDICT",       inherits=TRUE, ifnotfound=TRUE))
  DEBUG_RUNPRED       <- isTRUE(get0("DEBUG_RUNPRED",       inherits=TRUE, ifnotfound=FALSE))
  DEBUG_SAFERUN       <- isTRUE(get0("DEBUG_SAFERUN",       inherits=TRUE, ifnotfound=FALSE))
  DEBUG_ASPM          <- isTRUE(get0("DEBUG_ASPM",          inherits=TRUE, ifnotfound=FALSE))
  DEBUG_INVERSE       <- isTRUE(get0("DEBUG_INVERSE",       inherits=TRUE, ifnotfound=TRUE))
  STRICT_LINEAR_HEAD  <- isTRUE(get0("STRICT_LINEAR_HEAD",  inherits=TRUE, ifnotfound=FALSE))
  WARN_TINY_RANGE     <- isTRUE(get0("WARN_TINY_RANGE",     inherits=TRUE, ifnotfound=TRUE))
  
  # Classification behavior toggles (explicit, no inference)
  APPLY_CLASS_SOFTMAX           <- isTRUE(get0("APPLY_CLASS_SOFTMAX",           inherits=TRUE, ifnotfound=TRUE))
  BINARY_USE_SIGMOID_WHEN_1COL  <- isTRUE(get0("BINARY_USE_SIGMOID_WHEN_1COL",  inherits=TRUE, ifnotfound=TRUE))
  CLASS_THRESHOLD               <- as.numeric(get0("CLASS_THRESHOLD",           inherits=TRUE, ifnotfound=0.5))
  
  .dbg <- function(...) if (isTRUE(DEBUG_PREDICT)) cat("[DBG] ", sprintf(...), "\n", sep = "")
  
  r6 <- function(x) {
    if (is.null(x)) return(NA_real_)
    if (is.list(x)) {
      x <- unlist(x, recursive = TRUE, use.names = FALSE)
      if (!length(x)) return(NA_real_)
    }
    suppressWarnings({
      xn <- as.numeric(x[1])
      if (!is.finite(xn)) return(NA_real_)
      round(xn, 6)
    })
  }
  
  # ------------------------------------------------------------
  # Utilities
  # ------------------------------------------------------------

  
  .coerce_to_fn <- function(x) {
    if (is.function(x)) return(x)
    if (is.character(x) && length(x) == 1L) {
      nm <- tolower(trimws(x))
      if (exists(x, inherits = TRUE) && is.function(get(x, inherits = TRUE))) return(get(x, inherits = TRUE))
      if (nm %in% c("identity","linear","id")) { f <- function(z) z; attr(f,"name") <- "identity"; return(f) }
      stop(sprintf("Unknown activation name '%s' (no function found).", x))
    }
    if (is.null(x)) { f <- function(z) z; attr(f,"name") <- "identity"; return(f) }
    stop("Activation entry is neither function nor single character name.")
  }
  .normalize_af <- function(af, n_layers) {
    if (is.null(af)) af <- list()
    if (!is.list(af)) af <- as.list(af)
    af <- lapply(af, .coerce_to_fn)
    if (length(af) < n_layers) {
      pad <- replicate(n_layers - length(af), { f <- function(z) z; attr(f,"name") <- "identity"; f }, simplify = FALSE)
      af <- c(af, pad)
    } else if (length(af) > n_layers) {
      af <- af[seq_len(n_layers)]
    }
    af
  }
  .force_linear_head_if_regression <- function(mode, af) {
    if (identical(tolower(mode %||% ""), "regression") && length(af)) {
      af[[length(af)]] <- (function(z) z); attr(af[[length(af)]], "name") <- "identity"
    }
    af
  }
  .resolve_af <- function(model_meta, global_af) {
    if (!is.null(model_meta$model$activation_functions)) {
      list(af = model_meta$model$activation_functions, source = "artifact:model$activation_functions")
    } else if (!is.null(model_meta$activation_functions)) {
      list(af = model_meta$activation_functions,       source = "artifact:activation_functions")
    } else {
      list(af = global_af,                              source = "global activation_functions")
    }
  }
  
  .get_tt_strict <- function(meta_local) {
    tt <- meta_local$target_transform %||%
      (meta_local$preprocessScaledData %||% list())$target_transform %||%
      (meta_local$model %||% list())$target_transform %||%
      tryCatch({
        j <- meta_local$target_transform_json
        if (length(j) && is.character(j) && nzchar(j)) jsonlite::fromJSON(j) else NULL
      }, error = function(e) NULL)
    
    if (is.null(tt) || is.null(tt$type)) {
      list(type = "identity", params = list(center = 0, scale = 1))
    } else {
      p <- tt$params %||% list()
      p$center <- p$center %||% 0
      p$scale  <- p$scale  %||% 1
      list(type = tolower(as.character(tt$type)), params = p)
    }
  }
  
  .apply_target_inverse_strict <- function(P_raw_local, meta_local) {
    tt    <- .get_tt_strict(meta_local)
    ttype <- tolower(as.character(tt$type %||% "identity"))
    c0    <- as.numeric(tt$params$center %||% 0)
    s0    <- as.numeric(tt$params$scale  %||% 1)
    if (!is.finite(s0) || s0 == 0) s0 <- 1
    
    if (isTRUE(DEBUG_INVERSE)) {
      cat("[INV] ttype=", ttype, " | params=center,scale → ", r6(c0), ",", r6(s0), "\n", sep = "")
    }
    
    if (ttype == "zscore")  return(P_raw_local * s0 + c0)
    if (ttype == "minmax") {
      mn <- as.numeric(tt$params$min %||% 0); mx <- as.numeric(tt$params$max %||% 1)
      return(P_raw_local * (mx - mn) + mn)
    }
    if (ttype == "affine") {
      a <- as.numeric(tt$params$a); b <- as.numeric(tt$params$b)
      if (!is.finite(a) || !is.finite(b)) stop("predict(): invalid affine params a/b.")
      return(a + b * P_raw_local)
    }
    if (abs(c0) > 1e-12 || abs(s0 - 1) > 1e-12) {
      warning("target_transform 'identity' has nontrivial center/scale; treating as zscore. (Fix artifact writer.)")
      return(P_raw_local * s0 + c0)
    }
    P_raw_local
  }
  
  # ------------------------------------------------------------
  # Mode / scope
  # ------------------------------------------------------------
  MODE                 <- get0("MODE",           inherits=TRUE, ifnotfound="predict:stateless")
  PREDICT_SCOPE        <- get0("PREDICT_SCOPE",  inherits=TRUE, ifnotfound="all")
  INPUT_SPLIT          <- get0("INPUT_SPLIT",    inherits=TRUE, ifnotfound="test")
  TARGET_METRIC        <- get0("TARGET_METRIC",  inherits=TRUE, ifnotfound="accuracy")
  CLASSIFICATION_MODE  <- tolower(get0("CLASSIFICATION_MODE", inherits = TRUE, ifnotfound = "regression"))
  mode_type            <- match.arg(CLASSIFICATION_MODE, c("regression","binary","multiclass"))
  is_regression        <- identical(mode_type, "regression")
  is_binary            <- identical(mode_type, "binary")
  is_multiclass        <- identical(mode_type, "multiclass")
  is_classif           <- !is_regression
  
  predict_mode  <- if (identical(MODE, "predict:stateful")) "stateful" else "stateless"
  scope_opt_in  <- match.arg(PREDICT_SCOPE, c("one","group-best","all","pick","single"))
  scope_opt     <- if (identical(scope_opt_in, "single")) "one" else scope_opt_in
  PICK_INDEX    <- as.integer(get0("PICK_INDEX", inherits=TRUE, ifnotfound=NA_integer_))
  
  .dbg("MODE=%s | PREDICT_SCOPE=%s | INPUT_SPLIT=%s | MODE_TYPE=%s",
       as.character(MODE), as.character(PREDICT_SCOPE), as.character(INPUT_SPLIT), mode_type)
  
  # ------------------------------------------------------------
  # Registry & ranking (RDS metadata only)
  # ------------------------------------------------------------
  df_all <- bm_list_all()
  if (!nrow(df_all)) stop("No models found in RDS registry (bm_list_all() returned 0 rows).")
  
  KIND_FILTER  <- get0("KIND_FILTER",  inherits=TRUE, ifnotfound=character())
  ENS_FILTER   <- get0("ENS_FILTER",   inherits=TRUE, ifnotfound=NULL)
  MODEL_FILTER <- get0("MODEL_FILTER", inherits=TRUE, ifnotfound=NULL)
  
  if (length(KIND_FILTER))    df_all <- df_all[df_all$kind  %in% KIND_FILTER, , drop = FALSE]
  if (!is.null(ENS_FILTER))   df_all <- df_all[df_all$ens   %in% ENS_FILTER,  , drop = FALSE]
  if (!is.null(MODEL_FILTER)) df_all <- df_all[df_all$model %in% MODEL_FILTER,, drop = FALSE]
  if (!nrow(df_all)) stop("No candidates after applying KIND/ENS/MODEL filters.")
  
  df_all$metric_value <- vapply(seq_len(nrow(df_all)), function(i) {
    meta_i <- tryCatch(bm_select_exact(df_all$kind[i], df_all$ens[i], df_all$model[i]),
                       error = function(e) NULL)
    if (is.null(meta_i)) return(NA_real_)
    .get_metric_from_meta(meta_i, TARGET_METRIC)
  }, numeric(1))
  
  minimize <- .metric_minimize(TARGET_METRIC)
  ok       <- is.finite(df_all$metric_value)
  df_ranked <- if (any(ok)) {
    ord <- if (minimize) order(df_all$metric_value) else order(df_all$metric_value, decreasing = TRUE)
    df_all[ord, , drop = FALSE]
  } else df_all
  
  if (!"source" %in% names(df_ranked)) df_ranked$source <- NA_character_
  source_priority <- c("rds","file","disk","env","memory","workspace")
  df_ranked$._src_rank <- match(tolower(df_ranked$source), source_priority)
  df_ranked$._src_rank[is.na(df_ranked$._src_rank)] <- 99L
  ord <- with(df_ranked, order(kind, ens, model, ._src_rank, na.last = TRUE))
  df_ranked <- df_ranked[ord, , drop = FALSE]
  dups <- duplicated(df_ranked[, c("kind","ens","model")])
  if (any(dups)) {
    kept   <- df_ranked[!dups, , drop = FALSE]
    tossed <- df_ranked[ dups, , drop = FALSE]
    cat(sprintf("De-duped %d duplicate entries (preferring RDS).\n", sum(dups)))
    df_ranked <- kept
  }
  df_ranked$._src_rank <- NULL
  if (!nrow(df_ranked)) stop("No candidates available to select scope from.")
  
  .best_in_group_idx <- function(idx) {
    vals <- df_ranked$metric_value[idx]
    if (all(!is.finite(vals))) return(idx[1L])
    if (minimize) idx[ which.min(vals) ] else idx[ which.max(vals) ]
  }
  
  scope_rows <- switch(
    scope_opt,
    "one" = df_ranked[1L, , drop = FALSE],
    "group-best" = {
      key <- paste(df_ranked$kind, df_ranked$ens, sep = ":::")
      split_idx <- split(seq_len(nrow(df_ranked)), key)
      pick <- vapply(split_idx, .best_in_group_idx, integer(1))
      df_ranked[as.integer(pick), , drop = FALSE]
    },
    "all"  = df_ranked,
    "pick" = {
      if (is.finite(PICK_INDEX) && PICK_INDEX >= 1L && PICK_INDEX <= nrow(df_ranked)) {
        df_ranked[PICK_INDEX, , drop = FALSE]
      } else {
        warning(sprintf("PICK_INDEX=%s out of range; defaulting to top-1.", as.character(PICK_INDEX)))
        df_ranked[1L, , drop = FALSE]
      }
    },
    { warning(sprintf("Unknown scope '%s'; defaulting to top-1.", scope_opt)); df_ranked[1L, , drop = FALSE] }
  )
  rownames(scope_rows) <- NULL
  cat(sprintf("Scope selection → %s | %d candidate(s)\n", scope_opt, nrow(scope_rows)))
  
  # ------------------------------------------------------------
  # Core predict loop (STRICT: metadata-only)
  # ------------------------------------------------------------
  results   <- vector("list", length = nrow(scope_rows))
  pred_sigs <- character(nrow(scope_rows))
  P_list    <- vector("list", length = nrow(scope_rows))
  
  .digest_or <- function(obj) {
    if (requireNamespace("digest", quietly=TRUE)) {
      tryCatch(digest::digest(obj, algo="xxhash64"), error=function(e) "NA")
    } else "digest_pkg_missing"
  }
  .resolve_model_rds <- function(kind, ens, model,
                                 artifacts_dir = get0("ARTIFACTS_DIR", inherits=TRUE, ifnotfound=file.path(getwd(),"artifacts"))) {
    out <- list(file = NA_character_, used = FALSE)
    if (!dir.exists(artifacts_dir)) return(out)
    files <- tryCatch(list.files(artifacts_dir, pattern="\\.[Rr][Dd][Ss]$", full.names=TRUE, recursive=TRUE, include.dirs=FALSE),
                      error = function(e) character(0))
    if (!length(files)) return(out)
    b <- basename(files)
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
  
  .load_model_meta_rds <- function(kind, ens, model, artifacts_dir = NULL) {
    adir <- artifacts_dir %||% get0(".BM_DIR", inherits = TRUE, ifnotfound = "artifacts")
    if (!dir.exists(adir)) stop(sprintf("Artifacts dir not found: %s", adir))
    files <- list.files(adir, pattern = "\\.[Rr][Dd][Ss]$", full.names = TRUE, recursive = TRUE, include.dirs = FALSE)
    if (!length(files)) return(NULL)
    b <- basename(files)
    esc <- function(x) gsub("([][(){}.+*?^$|\\\\-])", "\\\\\\1", x, perl = TRUE)
    pat_strict <- paste0("(?i)(?:^|_)Ensemble_", esc(kind), "_", ens, "_model_", model, "_metadata_.*\\.[Rr][Dd][Ss]$")
    hit <- grepl(pat_strict, b, perl = TRUE)
    if (!any(hit)) {
      pat_loose <- paste0("(?i)", esc(kind), ".*model[^A-Za-z0-9]+", model, ".*metadata.*\\.[Rr][Dd][Ss]$")
      hit <- grepl(pat_loose, b, perl = TRUE)
    }
    cand <- files[hit]
    if (!length(cand)) return(NULL)
    info <- file.info(cand)
    file <- cand[order(info$mtime, decreasing = TRUE)][1L]
    tryCatch({
      m <- readRDS(file); attr(m, "artifact_path") <- file; m
    }, error = function(e) {
      warning(sprintf("Failed to read meta RDS: %s (%s)", file, conditionMessage(e))); NULL
    })
  }
  
  for (i in seq_len(nrow(scope_rows))) {
    kind  <- as.character(scope_rows$kind[i])
    ens   <- as.integer(scope_rows$ens[i])
    model <- as.integer(scope_rows$model[i])
    
    # 1) Load metadata (strict)
    model_meta <- .load_model_meta_rds(kind, ens, model)
    if (is.null(model_meta)) { warning(sprintf("Skipping %s/%d/%d: no artifact metadata.", kind, ens, model)); next }
    meta <- model_meta
    cat(sprintf("[META] loaded from: %s\n", attr(meta, "artifact_path") %||% "<unknown>"))
    
    # Echo target transform for this model (after meta is loaded)
    tt <- meta$target_transform %||%
      (meta$preprocessScaledData %||% list())$target_transform %||%
      (meta$model %||% list())$target_transform
    
    if (!is.null(tt)) {
      cat(sprintf("[predict-only] target_transform: type=%s | center=%s | scale=%s\n",
                  tt$type %||% "?", 
                  as.character((tt$params %||% list())$center %||% NA_real_),
                  as.character((tt$params %||% list())$scale  %||% NA_real_)))
    } else {
      cat("[predict-only] target_transform: <missing in metadata>\n")
    }
    
    # 2) Resolve activations; enforce linear head only for regression
    af_global <- get0("activation_functions", inherits = TRUE, ifnotfound = NULL)
    res_af    <- .resolve_af(meta, af_global)
    n_layers  <- tryCatch(length(meta$weights), error=function(e) NA_integer_)
    if (!is.finite(n_layers) || n_layers <= 0L) n_layers <- tryCatch(length(meta$model$weights), error=function(e) NA_integer_)
    if (!is.finite(n_layers) || n_layers <= 0L) n_layers <- 1L
    activation_functions_resolved <- .normalize_af(res_af$af, n_layers)
    activation_functions_resolved <- .force_linear_head_if_regression(mode_type, activation_functions_resolved)
    last <- activation_functions_resolved[[length(activation_functions_resolved)]]
    nm   <- tolower(trimws(attr(last, "name") %||% ""))
    is_lin <- {
      v <- c(-2,-1,0,1,2); outp <- try(last(v), silent = TRUE)
      is.numeric(outp) && length(outp) == 5 && all(abs(outp - v) <= 1e-12)
    } || nm %in% c("identity","linear","id") || identical(last, base::identity)
    cat(sprintf("[AF-TRACE] resolved from %s | layers=%d | last=%s | last_is_linear=%s\n",
                res_af$source, n_layers, if (nm=="") "<NULL>" else nm, is_lin))
    if (STRICT_LINEAR_HEAD && is_regression && !is_lin) stop("Regression requires linear head but last activation is not linear.")
    
    # 3) Choose split strictly from meta
    split_lower <- tolower(INPUT_SPLIT)
    if (split_lower == "test")       { Xi_raw <- meta$X_test;       yi_raw <- meta$y_test;       split_used <- "test" }
    else if (split_lower == "validation"){ Xi_raw <- meta$X_validation; yi_raw <- meta$y_validation; split_used <- "validation" }
    else if (split_lower == "train") { Xi_raw <- meta$X %||% meta$X_train; yi_raw <- meta$y %||% meta$y_train; split_used <- "train" }
    else {
      Xi_raw <- meta$X_validation;  yi_raw <- meta$y_validation; split_used <- "validation"
      if (is.null(Xi_raw) || is.null(yi_raw)) { Xi_raw <- meta$X_test; yi_raw <- meta$y_test; split_used <- "test" }
      if (is.null(Xi_raw) || is.null(yi_raw)) { Xi_raw <- meta$X %||% meta$X_train; yi_raw <- meta$y %||% meta$y_train; split_used <- "train" }
    }
    if (is.null(Xi_raw) || is.null(yi_raw)) { warning(sprintf("Skipping %s/%d/%d: split '%s' not found in metadata.", kind, ens, model, INPUT_SPLIT)); next }
    
    # 4) Strict coercion & feature alignment (by names from meta)
    .as_numeric_vector_strict <- function(v, nm = "y") {
      # matrix -> numeric vector (column-major), preserve length
      if (is.matrix(v)) {
        if (ncol(v) > 1L) stop(sprintf("[coerce:%s] matrix has %d columns; expected 1.", nm, ncol(v)), call. = FALSE)
        v <- v[, 1L, drop = TRUE]
      }
      
      # data.frame -> first (and only) column
      if (is.data.frame(v)) {
        if (ncol(v) != 1L) stop(sprintf("[coerce:%s] data.frame has %d columns; expected 1.", nm, ncol(v)), call. = FALSE)
        v <- v[[1L]]
      }
      
      # list / list-column -> take first scalar per element
      if (is.list(v)) {
        v <- vapply(v, function(x) {
          if (is.null(x)) return(NA_real_)
          if (is.list(x)) x <- unlist(x, recursive = TRUE, use.names = FALSE)
          suppressWarnings(as.numeric(if (length(x)) x[1] else NA))
        }, numeric(1))
      }
      
      # factor / character -> numeric
      if (is.factor(v)) v <- as.character(v)
      if (!is.numeric(v)) suppressWarnings(v <- as.numeric(v))
      
      if (!is.numeric(v)) stop(sprintf("[coerce:%s] could not coerce to numeric.", nm), call. = FALSE)
      if (!length(v))     stop(sprintf("[coerce:%s] zero-length after coercion.", nm), call. = FALSE)
      v
    }

    .as_numeric_matrix_strict <- function(X, nm = "X") {
      if (is.matrix(X) && is.numeric(X)) return(X)
      if (is.vector(X) && !is.list(X))  { X <- matrix(X, ncol = 1L); colnames(X) <- nm }
      if (is.data.frame(X) || is.matrix(X)) {
        Xdf <- as.data.frame(X, stringsAsFactors = FALSE)
        for (cc in names(Xdf)) {
          col <- Xdf[[cc]]
          if (is.list(col)) {
            col <- vapply(col, function(x) {
              if (is.null(x)) return(NA_real_)
              if (is.list(x)) x <- unlist(x, recursive = TRUE, use.names = FALSE)
              suppressWarnings(as.numeric(if (length(x)) x[1] else NA))
            }, numeric(1))
          } else if (is.factor(col)) {
            col <- as.numeric(as.character(col))
          } else if (!is.numeric(col)) {
            suppressWarnings(col <- as.numeric(col))
          }
          Xdf[[cc]] <- col
        }
        Xmat <- as.matrix(Xdf); storage.mode(Xmat) <- "double"
        bad <- which(vapply(seq_len(ncol(Xmat)), function(j) !any(is.finite(Xmat[,j])), logical(1)))
        if (length(bad)) stop(sprintf("[coerce:%s] columns entirely non-numeric/non-finite: %s", nm, paste(colnames(Xmat)[bad], collapse = ", ")), call. = FALSE)
        return(Xmat)
      }
      stop(sprintf("[coerce:%s] unsupported type: %s", nm, paste(class(X), collapse=",")), call. = FALSE)
    }
    
    if (is.data.frame(Xi_raw)) {
      has_list <- vapply(Xi_raw, is.list, logical(1))
      if (any(has_list)) cat(sprintf("[WARN] %d list-column(s) detected in Xi_raw: %s\n", sum(has_list), paste(names(Xi_raw)[has_list], collapse=", ")))
    }
    
    Xi <- .as_numeric_matrix_strict(Xi_raw, nm = "X")
    yi <- .as_numeric_vector_strict(yi_raw,  nm = "y")
    
    if (NROW(Xi_raw) != length(yi))
      stop(sprintf("[LABEL-CHK] NROW(X)=%d vs len(y)=%d on split '%s'.",
                   NROW(Xi_raw), length(yi), split_used))
    
    
    expected <- tryCatch({
      nms <- meta$feature_names %||% meta$input_names %||% meta$colnames
      if (is.null(nms)) colnames(Xi) else as.character(nms)
    }, error = function(e) colnames(Xi))
    
    
    orig_cols <- colnames(Xi)
    miss <- setdiff(expected, orig_cols)
    if (length(miss)) Xi <- cbind(Xi, matrix(0, nrow = nrow(Xi), ncol = length(miss), dimnames = list(NULL, miss)))
    Xi <- Xi[, expected, drop = FALSE]
    .dbg("feature-align(meta-only): +%d missing→0 | -%d dropped | final_cols=%d",
         length(setdiff(expected %||% character(0), orig_cols %||% character(0))),
         length(setdiff(orig_cols %||% character(0), expected %||% character(0))),
         ncol(Xi))
    
    input_size <- tryCatch(meta$input_size, error = function(e) NULL)
    if (!is.null(input_size) && !is.na(input_size) && input_size > 0L && ncol(Xi) != input_size) {
      stop(sprintf("Input feature count (%d) doesn’t match model’s expected input (%d).", ncol(Xi), input_size))
    }
    
    assign("LAST_APPLIED_X", Xi, .GlobalEnv)
    
    # 5) Predict (STRICT path) — with target transform probe
    `%||%` <- function(x, y) if (is.null(x)) y else x
    
    print("hello")
    


    cat("[SAFE-DBG] BEFORE .run_predict | X summary:",
        sprintf("mean=%f sd=%f min=%f max=%f\n",
                mean(as.matrix(X)), sd(as.matrix(X)),
                min(as.matrix(X)), max(as.matrix(X))))
    
    out <- .safe_run_predict(
      X = Xi, meta = meta, model_index = model, ML_NN = TRUE
    )
    
    # --- ALWAYS normalize immediately to a double matrix ---
    P_raw <- .as_pred_matrix(out, mode = "regression", meta = meta,
                             DEBUG = get0("DEBUG_ASPM", inherits = TRUE, ifnotfound = FALSE))
    
    # (Optional) keep a nice column name
    if (is.null(colnames(P_raw))) colnames(P_raw) <- "pred"
    
    # --- Safe debug prints that don't assume list shape ---
    str(P_raw)
    cat("[SAFE-DBG] AFTER .run_predict | head(P_raw):",
        paste(head(as.numeric(P_raw)), collapse = ", "),
        "\n")
    cat("[SAFE-DBG] P_raw summary:",
        sprintf("dims=%dx%d mean=%f sd=%f min=%f p50=%f max=%f\n",
                nrow(P_raw), ncol(P_raw),
                mean(P_raw), stats::sd(P_raw),
                min(P_raw), as.numeric(stats::median(P_raw)), max(P_raw)))
    
    # From here on, ALWAYS use P_raw (matrix). Never use pred_raw$predicted_output.
    
    
    if (!is.matrix(P_raw) || nrow(P_raw) == 0L) {
      warning(sprintf("Skipping %s/%d/%d: empty predictions.", kind, ens, model))
      next
    }
    
    assign("LAST_SAFERUN_OUT", P_raw, .GlobalEnv)
    
    cat(sprintf("[PRED-ONLY] raw_head: mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f\n",
                mean(P_raw), stats::sd(P_raw),
                min(P_raw), stats::median(as.vector(P_raw)), max(P_raw)))
    
    # ---- Target transform probe ----
    tt <- meta$target_transform %||%
      (meta$preprocessScaledData %||% list())$target_transform %||%
      (meta$model %||% list())$target_transform %||%
      list(type = "identity", params = list(center = 0, scale = 1))
    
    cat("[PRED-ONLY] target_transform:\n")


    
    
    # --- Safely fetch target transform from metadata (with fallbacks) ---
    .safe_tt <- function(meta) {
      `%||%` <- function(x, y) if (is.null(x)) y else x
      
      tt <- meta$target_transform %||%
        (meta$preprocessScaledData %||% list())$target_transform %||%
        (meta$model %||% list())$target_transform %||%
        list(type = "identity", params = list(center = 0, scale = 1))
      
      tt$type   <- tolower(tt$type %||% "identity")
      tt$params <- tt$params %||% list(center = 0, scale = 1)
      tt
    }
    
    tt  <- .safe_tt(meta)
    cen <- tt$params$center %||% NA_real_
    scl <- tt$params$scale  %||% NA_real_
    
    cat(sprintf("[PRED-ONLY] target_transform: type=%s | center=%s | scale=%s\n",
                tt$type, as.character(cen), as.character(scl)))
    
   
    
    
    
    # 6) Mode-specific post-processing (NO inference)
    if (is_regression) {
      P <- .apply_target_inverse_strict(P_raw, meta)
      # ===== LEAKAGE MINIMAL PROBE (post-predict; uses P and yi) =====
      {
        yv <- suppressWarnings(as.numeric(yi))
        pv <- suppressWarnings(as.numeric(P[, 1]))
        ok <- is.finite(yv) & is.finite(pv)
        yv <- yv[ok]; pv <- pv[ok]
        
        sst <- sum((yv - mean(yv))^2)
        sse <- sum((pv - yv)^2)
        r2  <- if (sst > 0) 1 - sse / sst else NA_real_
        
        set.seed(42)
        perm <- sample.int(length(yv))
        r2_shuf <- if (sst > 0) 1 - sum((yv - pv[perm])^2) / sst else NA_real_
        
        cat(sprintf("[LEAK-POST] R2=%.6f | R2(shuffled preds)=%.6f\n", r2, r2_shuf))
        
        leak_flag <- is.finite(r2) && r2 > 0.99 && is.finite(r2_shuf) && r2_shuf < 0.20
        
        # optional: quick feature->target correlation snapshot
        top_abs_cor <- tryCatch({
          Xi_num <- suppressWarnings(as.matrix(Xi))
          storage.mode(Xi_num) <- "double"
          cs <- suppressWarnings(cor(Xi_num, yv, use = "pairwise.complete.obs"))
          o <- order(abs(cs[,1]), decreasing = TRUE)
          head(setNames(cs[o, 1], colnames(Xi_num)[o]), 5)
        }, error = function(e) NA)
        
        diag <- list(r2 = r2, r2_shuf = r2_shuf, top_abs_cor = top_abs_cor,
                     n = length(yv), timestamp = Sys.time())
        fn <- file.path("artifacts", sprintf("leak_probe_%s.rds",
                                             format(Sys.time(), "%Y%m%d_%H%M%S")))
        saveRDS(diag, fn)
        cat("[LEAK-POST] Diagnostics saved to: ", fn, "\n", sep = "")
        
        # Enforce via env toggle (default TRUE)
        STRICT_LEAKAGE_GUARD <- isTRUE(get0("STRICT_LEAKAGE_GUARD", inherits = TRUE, ifnotfound = TRUE))
        if (STRICT_LEAKAGE_GUARD && leak_flag) {
          stop("[LEAK-GUARD] High R² + low shuffled R² → likely leakage (target/proxy in features, or prefit on test). Aborting.")
        }
      }
      # ===== END LEAKAGE MINIMAL PROBE =====
      
      
      # === INSERT PROBE HERE ===
      # probe_preds_vs_labels(P, yi, tag = "PREDICT", save_global = TRUE)
      # cat("[STOP-PROBE !train] after inverse:\n")
      
      # Use the split labels you've already loaded earlier in the loop:
      #   yi_raw  = labels as loaded from meta (may be df/matrix/vector)
      #   yi      = numeric vector coerced from yi_raw (you created this above)
      is_na_count <- sum(is.na(yi), na.rm = TRUE)
      cat("  yi NA count -> ", is_na_count, "\n", sep = "")
      
      # Coerce a clean numeric vector for y summaries
      y_vec <- suppressWarnings(as.numeric(yi))
      
      cat("  preds summary -> min=", min(P, na.rm=TRUE),
          " mean=", mean(P, na.rm=TRUE),
          " max=", max(P, na.rm=TRUE), "\n", sep="")
      
      cat("  y_vec summary -> min=", min(y_vec, na.rm=TRUE),
          " mean=", mean(y_vec, na.rm=TRUE),
          " max=", max(y_vec, na.rm=TRUE), "\n", sep="")
      
      # stop("STOP-PROBE !train: checked inverse outputs")
      
      
      if (isTRUE(DEBUG_PREDICT)) cat(sprintf("[DBG] post-inverse P: mean=%.6f sd=%.6f\n", mean(P[,1]), stats::sd(P[,1])))
      
      # --- INSERT THIS GUARD + CALIBRATION (regression only) ---
      # 0) apply saved affine calibration if present
      if (!is.null(meta$calibration) && identical(meta$calibration$type, "affine")) {
        a <- as.numeric(meta$calibration$a); b <- as.numeric(meta$calibration$b)
        P <- a + b * P
        if (isTRUE(DEBUG_PREDICT)) cat(sprintf("[CAL] applied saved affine: a=%.6f b=%.6f\n", a, b))
      } else {
        # 1) detect scale mismatch (huge variance gap) and auto-calibrate once
        y_vec <- suppressWarnings(as.numeric(yi))
        p_vec <- suppressWarnings(as.numeric(P[,1]))
        if (length(y_vec) > 1L && length(p_vec) > 1L) {
          var_ratio <- stats::var(y_vec, na.rm=TRUE) / max(stats::var(p_vec, na.rm=TRUE), .Machine$double.eps)
          has_tt <- !is.null(meta$target_transform) || !is.null(meta$preprocessScaledData$target_transform)
          if (!has_tt && is.finite(var_ratio) && var_ratio > 100) {
            fit <- stats::lm(y_vec ~ p_vec)  # y ≈ a + b * p
            a <- unname(stats::coef(fit)[1]); b <- unname(stats::coef(fit)[2])
            P <- a + b * P
            if (isTRUE(DEBUG_PREDICT)) cat(sprintf("[AUTO-CAL] a=%.6f b=%.6f | var_ratio=%.1f\n", a, b, var_ratio))
            # persist so future runs auto-apply (optional; only if you have a path handy)
            meta$calibration <- list(type="affine", a=a, b=b, note="auto-derived in predict-only")
            if (exists("META_RDS_PATH", inherits=TRUE)) try(saveRDS(meta, get("META_RDS_PATH", inherits=TRUE)), silent=TRUE)
          }
        }
      }
      # --- END INSERT ---
    } else {
# ------------------------------------------------------------
# Mode-specific post-processing
# ------------------------------------------------------------
if (is_regression) {
  # Regression: keep raw values, apply inverse if needed
  P <- .apply_target_inverse_strict(P_raw, meta)

} else if (is_binary) {
  if (ncol(P_raw) == 1L) {
    # Binary, 1-col: already sigmoid probabilities
    P <- P_raw
    yhat <- as.integer(P[,1] > CLASS_THRESHOLD)
  } else {
    # Binary, 2-col: softmax, then column 2 is "positive"
    mx <- apply(P_raw, 1, max)
    ex <- exp(P_raw - mx)
    sm <- rowSums(ex)
    P  <- ex / sm
    yhat <- as.integer(max.col(P, ties.method = "first") == 2L)
  }

} else if (is_multiclass) {
  # Multiclass: softmax, then argmax
  mx <- apply(P_raw, 1, max)
  ex <- exp(P_raw - mx)
  sm <- rowSums(ex)
  P  <- ex / sm
  yhat <- max.col(P, ties.method = "first")
}

    }      
    
    # ---- R2 DEBUG BLOCK (paste right after P is computed) ----
    cat("\n[R2-DBG] mode=regression? ", isTRUE(identical(mode_type, "regression")), 
        " | split=", split_used, "\n", sep="")
    
    cat(sprintf("[QUICK] rows(X)=%d | len(y)=%d\n",
                NROW(Xi_raw),
                if (!is.null(dim(yi_raw))) NROW(yi_raw) else length(yi_raw)))
    
    # 1) y checks
    y <- suppressWarnings(as.numeric(yi))
    cat(sprintf("[R2-DBG] y: n=%d | anyNA=%s | anyInf=%s | mean=%s | var=%s | range=[%s,%s]\n",
                length(y), any(is.na(y)), any(!is.finite(y)),
                format(mean(y, na.rm=TRUE)), format(stats::var(y, na.rm=TRUE)),
                format(min(y, na.rm=TRUE)), format(max(y, na.rm=TRUE))))
    
    # 2) P checks (use the first column if matrix)
    p <- suppressWarnings(as.numeric(P[,1]))
    cat(sprintf("[R2-DBG] p: n=%d | anyNA=%s | anyInf=%s | mean=%s | sd=%s | range=[%s,%s]\n",
                length(p), any(is.na(p)), any(!is.finite(p)),
                format(mean(p, na.rm=TRUE)), format(stats::sd(p, na.rm=TRUE)),
                format(min(p, na.rm=TRUE)), format(max(p, na.rm=TRUE))))
    
    # 3) shape checks
    if (length(y) != length(p)) {
      stop(sprintf("[R2-DBG] length mismatch: y=%d vs p=%d (feature/label split misaligned).", length(y), length(p)))
    }
    
    # 4) SST/SSE and manual R²
    sst <- sum( (y - mean(y, na.rm=TRUE))^2 )
    sse <- sum( (p - y)^2 )
    cat(sprintf("[R2-DBG] sst=%s | sse=%s\n", format(sst), format(sse)))
    
    if (!is.finite(sst) || sst <= 0) {
      cat("[R2-DBG] R² undefined: target variance (SST) <= 0. (Is y constant or degenerate on this split?)\n")
    } else {
      r2_manual <- 1 - sse / sst
      cor_py <- suppressWarnings(stats::cor(p, y, use="complete.obs"))
      rmse_m <- sqrt(mean((p - y)^2))
      mae_m  <- mean(abs(p - y))
      cat(sprintf("[R2-DBG] r2_manual=%s | corr=%s | rmse=%s | mae=%s\n",
                  format(r2_manual), format(cor_py), format(rmse_m), format(mae_m)))
    }
    # ---- END R2 DEBUG BLOCK ----
    
    
    # 7) Metrics
    SONN          <- get0("SONN", inherits = TRUE, ifnotfound = NULL)
    verbose_flag  <- isTRUE(get0("verbose", inherits = TRUE, ifnotfound = FALSE))
    yi_vec        <- if (!is.null(yi)) as.numeric(yi) else NULL
    
    if (is_regression) {
      mse_val   <- tryCatch(MSE(SONN, Xi, yi_vec, "regression", P, verbose_flag),    error = function(e) NA_real_)
      mae_val   <- tryCatch(MAE(SONN, Xi, yi_vec, "regression", P, verbose_flag),    error = function(e) NA_real_)
      rmse_val  <- tryCatch(RMSE(SONN, Xi, yi_vec, "regression", P, verbose_flag),   error = function(e) NA_real_)
      r2_val    <- tryCatch(R2(SONN, Xi, yi_vec, "regression", P, verbose_flag),     error = function(e) NA_real_)
      mape_val  <- tryCatch(MAPE(SONN, Xi, yi_vec, "regression", P, verbose_flag),   error = function(e) NA_real_)
      smape_val <- tryCatch(SMAPE(SONN, Xi, yi_vec, "regression", P, verbose_flag),  error = function(e) NA_real_)
      wmape_val <- tryCatch(WMAPE(SONN, Xi, yi_vec, "regression", P, verbose_flag),  error = function(e) NA_real_)
      mase_val  <- tryCatch(MASE(SONN, Xi, yi_vec, "regression", P, verbose_flag),   error = function(e) NA_real_)
      
      if (DEBUG_PREDICT && nrow(P) > 0 && length(yi_vec) == nrow(P) && all(is.finite(yi_vec))) {
        p  <- as.numeric(P[, 1])
        sse <- sum((p - yi_vec)^2); sst <- sum((yi_vec - mean(yi_vec))^2)
        r2_manual <- if (is.finite(sst) && sst > 0) 1 - sse/sst else NA_real_
        cor_py <- suppressWarnings(stats::cor(p, yi_vec))
        rmse <- sqrt(mean((p - yi_vec)^2)); mae <- mean(abs(p - yi_vec))
        .dbg("MANUAL: R2=%.6f | corr=%.6f | RMSE=%.6f | MAE=%.6f", r2_manual, cor_py, rmse, mae)
      }
      
      results[[i]] <- list(
        kind=kind, ens=ens, model=model, data_source=split_used, split_used=split_used, n_pred_rows=nrow(P),
        quantization_error=NA_real_, topographic_error=NA_real_, clustering_quality_db=NA_real_,
        accuracy=NA_real_, precision=NA_real_, recall=NA_real_, f1=NA_real_,
        tuned_threshold=NA_real_, tuned_accuracy=NA_real_, tuned_precision=NA_real_,
        tuned_recall=NA_real_, tuned_f1=NA_real_,
        MSE=r6(mse_val), MAE=r6(mae_val), RMSE=r6(rmse_val), R2=r6(r2_val),
        MAPE=r6(mape_val), SMAPE=r6(smape_val), WMAPE=r6(wmape_val), MASE=r6(mase_val),
        ndcg=NA_real_, diversity=NA_real_, serendipity=NA_real_, hit_rate=NA_real_,
        w_sig=.digest_or(meta), pred_sig=.digest_or(round(P[seq_len(min(nrow(P), 2000)), , drop = FALSE], 6)),
        model_rds=.resolve_model_rds(kind, ens, model)$file,
        artifact_used=if (.resolve_model_rds(kind, ens, model)$used) "yes" else "no"
      )
      
    } else {
      # Explicit binary / multiclass path
      acc <- prec <- rec <- f1s <- NA_real_
      
      if (is_binary) {
        # Thresholding @ CLASS_THRESHOLD (default 0.5)
        if (ncol(P) == 1L) {
          yhat <- as.integer(P[,1] >= CLASS_THRESHOLD)
        } else {
          # If model outputs 2 logits/probs for binary, take class 2 as "positive"
          yhat <- as.integer(max.col(P, ties.method="first") == 2L)
        }
        ybin <- yi_vec
        if (!all(ybin %in% c(0,1))) {
          # Map to 0/1 using max label as positive
          mx <- suppressWarnings(max(ybin, na.rm = TRUE)); ybin <- as.integer(ybin == mx)
        }
        TP <- sum(yhat==1 & ybin==1); FP <- sum(yhat==1 & ybin==0)
        FN <- sum(yhat==0 & ybin==1); TN <- sum(yhat==0 & ybin==0)
        acc <- (TP+TN)/length(ybin)
        prec <- if ((TP+FP)>0) TP/(TP+FP) else NA_real_
        rec  <- if ((TP+FN)>0) TP/(TP+FN) else NA_real_
        f1s  <- if (is.finite(prec) && is.finite(rec) && (prec+rec)>0) 2*prec*rec/(prec+rec) else NA_real_
        
      } else if (is_multiclass) {
        yhat <- max.col(P, ties.method="first")
        ymc  <- if (is.matrix(yi_raw) && ncol(yi_raw)>1) max.col(yi_raw, "first") else as.integer(yi_vec)
        acc  <- mean(yhat == ymc)
      }
      
      results[[i]] <- list(
        kind=kind, ens=ens, model=model, data_source=split_used, split_used=split_used, n_pred_rows=nrow(P),
        quantization_error=NA_real_, topographic_error=NA_real_, clustering_quality_db=NA_real_,
        accuracy=r6(acc), precision=r6(prec), recall=r6(rec), f1=r6(f1s),
        tuned_threshold=NA_real_, tuned_accuracy=NA_real_, tuned_precision=NA_real_,
        tuned_recall=NA_real_, tuned_f1=NA_real_,
        MSE=NA_real_, MAE=NA_real_, RMSE=NA_real_, R2=NA_real_,
        MAPE=NA_real_, SMAPE=NA_real_, WMAPE=NA_real_, MASE=NA_real_,
        ndcg=NA_real_, diversity=NA_real_, serendipity=NA_real_, hit_rate=NA_real_,
        w_sig=.digest_or(meta), pred_sig=.digest_or(round(P[seq_len(min(nrow(P), 2000)), , drop = FALSE], 6)),
        model_rds=.resolve_model_rds(kind, ens, model)$file,
        artifact_used=if (.resolve_model_rds(kind, ens, model)$used) "yes" else "no"
      )
    }
    
    P_list[[i]] <- P
    
    cat(sprintf("→ Model(kind=%s, ens=%d, model=%d) preds=%dx%d\n", kind, ens, model, nrow(P), ncol(P)))
    cat(sprintf("   · input_source=%s | rows=%d, cols=%d\n", split_used, nrow(Xi), ncol(Xi)))
    cat(sprintf("   · pred_sig=%s | artifact_rds=%s\n",
                results[[i]]$pred_sig,
                if (identical(results[[i]]$artifact_used,"yes")) "yes" else "no"))
  }
  
  results <- Filter(Negate(is.null), results)
  if (!length(results)) stop("No successful predictions.")
  
  rows <- lapply(results, function(z) {
    data.frame(
      kind=z$kind, ens=z$ens, model=z$model, data_source=z$data_source, split_used=z$split_used,
      n_pred_rows=z$n_pred_rows,
      quantization_error=z$quantization_error, topographic_error=z$topographic_error, clustering_quality_db=z$clustering_quality_db,
      accuracy=z$accuracy, precision=z$precision, recall=z$recall, f1=z$f1,
      tuned_threshold=z$tuned_threshold, tuned_accuracy=z$tuned_accuracy, tuned_precision=z$tuned_precision,
      tuned_recall=z$tuned_recall, tuned_f1=z$tuned_f1,
      MSE=z$MSE, MAE=z$MAE, RMSE=z$RMSE, R2=z$R2,
      MAPE=z$MAPE, SMAPE=z$SMAPE, WMAPE=z$WMAPE, MASE=z$MASE,
      ndcg=z$ndcg, diversity=z$diversity, serendipity=z$serendipity, hit_rate=z$hit_rate,
      w_sig=z$w_sig, pred_sig=z$pred_sig, model_rds=z$model_rds, artifact_used=z$artifact_used,
      stringsAsFactors=FALSE
    )
  })
  results_df <- do.call(rbind, rows); rownames(results_df) <- NULL
  
  # order columns: keep artifact fields last
  tail_cols <- c("model_rds","artifact_used")
  head_cols <- setdiff(names(results_df), tail_cols)
  results_df <- results_df[, c(head_cols, tail_cols), drop = FALSE]
  
  # prune empty columns (but keep identifiers)
  .col_has_value <- function(v) { if (is.numeric(v)) any(is.finite(v)) else any(!(is.na(v) | v==""), na.rm=TRUE) }
  keep_cols <- names(results_df)[vapply(results_df, .col_has_value, logical(1))]
  must_keep <- c("kind","ens","model","data_source","split_used","n_pred_rows","w_sig","pred_sig","model_rds","artifact_used")
  keep_cols <- union(keep_cols, must_keep)
  keep_cols <- intersect(c(head_cols, tail_cols), keep_cols)
  results_df <- results_df[, keep_cols, drop = FALSE]
  assign("PREDICT_RESULTS_TABLE", results_df, .GlobalEnv)
  
  # pretty print
  PREDICT_FULL_PRINT  <- get0("PREDICT_FULL_PRINT",  inherits = TRUE, ifnotfound = FALSE)
  PREDICT_HEAD_N      <- as.integer(get0("PREDICT_HEAD_N",      inherits = TRUE, ifnotfound = 50L))
  PREDICT_PRINT_MAX   <- as.numeric(get0("PREDICT_PRINT_MAX",   inherits = TRUE, ifnotfound = 1e7))
  PREDICT_PRINT_WIDTH <- as.integer(get0("PREDICT_PRINT_WIDTH", inherits = TRUE, ifnotfound = 200L))
  PREDICT_USE_TIBBLE  <- get0("PREDICT_USE_TIBBLE",  inherits = TRUE, ifnotfound = TRUE)
  
  cat(sprintf("[predict] MODE=%s | SCOPE=%s | METRIC=%s | rows=%d | mode_type=%s\n",
              predict_mode, scope_opt, TARGET_METRIC, nrow(results_df), mode_type))
  if (isTRUE(DEBUG_PREDICT) && is_regression) .dbg("Regression path confirmed. (Inverse-transform applied using artifact meta only.)")
  
  if (isTRUE(PREDICT_FULL_PRINT)) {
    old_opts <- options(max.print = PREDICT_PRINT_MAX, width = PREDICT_PRINT_WIDTH)
    on.exit(options(old_opts), add = TRUE)
  }
  if (isTRUE(PREDICT_USE_TIBBLE) && requireNamespace("tibble", quietly = TRUE)) {
    tb <- tibble::as_tibble(results_df)
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
    to_show <- if (isTRUE(PREDICT_FULL_PRINT)) results_df else utils::head(results_df, PREDICT_HEAD_N)
    print(to_show, row.names = FALSE, right = TRUE)
    if (!isTRUE(PREDICT_FULL_PRINT) && nrow(results_df) > nrow(to_show)) {
      cat(sprintf("... (%d more rows — set PREDICT_FULL_PRINT=TRUE to show all)\n",
                  nrow(results_df) - nrow(to_show)))
    }
  }
  
  # save compact prediction pack
  artifacts_dir <- file.path(getwd(), "artifacts")
  if (!dir.exists(artifacts_dir)) dir.create(artifacts_dir, recursive = TRUE, showWarnings = FALSE)
  ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  scope_tag <- gsub("[^A-Za-z0-9_-]+", "-", tolower(scope_opt))
  mode_tag  <- gsub("[^A-Za-z0-9_-]+", "-", tolower(predict_mode))
  rds_path  <- file.path(artifacts_dir, sprintf("predictions_%s_scope-%s_%s.rds", mode_tag, scope_tag, ts_stamp))
  
  pred_named <- list()
  for (i in seq_along(P_list)) {
    if (is.null(P_list[[i]])) next
    tag <- sprintf("%s_%d_%d", scope_rows$kind[i], scope_rows$ens[i], scope_rows$model[i])
    pred_named[[tag]] <- P_list[[i]]
  }
  
  predict_pack <- list(
    saved_at        = Sys.time(),
    predict_mode    = predict_mode,
    PREDICT_SCOPE   = scope_opt,
    TARGET_METRIC   = TARGET_METRIC,
    minimize_metric = minimize,
    flags = list(
      MODE = MODE,
      KIND_FILTER = KIND_FILTER, ENS_FILTER = ENS_FILTER, MODEL_FILTER = MODEL_FILTER,
      PICK_INDEX = PICK_INDEX,
      INPUT_SPLIT = INPUT_SPLIT,
      CLASSIFICATION_MODE = mode_type,
      APPLY_CLASS_SOFTMAX = APPLY_CLASS_SOFTMAX,
      BINARY_USE_SIGMOID_WHEN_1COL = BINARY_USE_SIGMOID_WHEN_1COL,
      CLASS_THRESHOLD = CLASS_THRESHOLD
    ),
    candidates_ranked = df_ranked,
    scope_rows        = scope_rows,
    results_table     = results_df,
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
    # Apply per-SONN plotting flags to all internal models
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
    
    # ============================
    # Multi-seed (50 runs) section
    # ============================
    if (!dir.exists("artifacts_runs")) dir.create("artifacts_runs")
    
    seeds    <- 1:10

    acc_rows <- vector("list", length(seeds))
    
    for (i in seq_along(seeds)) {
      s <- seeds[i]
      set.seed(s)
      
      # Build a fresh DESONN for each seed so init differs per run
      run_model <- DESONN$new(
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
      
      # Apply the same per-SONN plotting flags to this run_model
      if (length(run_model$ensemble)) {
        for (m in seq_along(run_model$ensemble)) {
          run_model$ensemble[[m]]$PerEpochViewPlotsConfig <- list(
            accuracy_plot   = isTRUE(accuracy_plot),
            saturation_plot = isTRUE(saturation_plot),
            max_weight_plot = isTRUE(max_weight_plot),
            viewAllPlots    = isTRUE(viewAllPlots),
            verbose         = isTRUE(verbose)
          )
          run_model$ensemble[[m]]$FinalUpdatePerformanceandRelevanceViewPlotsConfig <- list(
            performance_high_mean_plots = isTRUE(performance_high_mean_plots),
            performance_low_mean_plots  = isTRUE(performance_low_mean_plots),
            relevance_high_mean_plots   = isTRUE(relevance_high_mean_plots),
            relevance_low_mean_plots    = isTRUE(relevance_low_mean_plots),
            viewAllPlots                = isTRUE(viewAllPlots),
            verbose                     = isTRUE(verbose)
          )
        }
      }
      
      # Train for this seed
      model_results <- run_model$train(
        Rdata=X, labels=y, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch,
        lr_min=lr_min, ensemble_number=0L, num_epochs=num_epochs, use_biases=use_biases,
        threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns, CLASSIFICATION_MODE=CLASSIFICATION_MODE,
        activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
        dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
        beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
        batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
        epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
        shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights, preprocessScaledData=preprocessScaledData,
        X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
        train=train, viewTables=viewTables, verbose=verbose
      )
      
      # Extract accuracy for this run (robustly)
      acc <- tryCatch(
        as.numeric(model_results$performance_relevance_data$accuracy_percent),
        error = function(e) NA_real_
      )
      
      acc_rows[[i]] <- data.frame(run_index = i, seed = s, accuracy_percent = acc)
      
      cat(sprintf("Seed %d → accuracy_percent = %s\n",
                  s, if (is.na(acc)) "NA" else format(acc, digits = 6)))
      
      # After the final run, keep that model as main_model for downstream code
      if (i == length(seeds)) {
        main_model <- run_model
      }
    }
    
    # Save combined table to RDS
    results_table <- do.call(rbind, acc_rows)
    out_path <- file.path(
      "artifacts_runs",
      sprintf("accuracy_runs_%s_50seeds.rds", format(Sys.time(), "%Y%m%d_%H%M%S"))
    )
    saveRDS(results_table, out_path)
    cat("Saved multi-seed results to: ", out_path, "\n")
    
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
    
    # Optional summaries (for the last seed’s model)
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
        lr_min=lr_min, ensemble_number=0L, num_epochs=num_epochs, use_biases=use_biases,
        threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns, CLASSIFICATION_MODE=CLASSIFICATION_MODE,
        activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
        dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
        beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
        batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
        epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
        shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights, preprocessScaledData=preprocessScaledData,
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
          lr_min=lr_min, ensemble_number=1L, num_epochs=num_epochs, use_biases=use_biases,
          threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns,
          activation_functions_learn=activation_functions_learn, activation_functions=activation_functions,
          dropout_rates_learn=dropout_rates_learn, dropout_rates=dropout_rates, optimizer=optimizer,
          beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
          batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
          epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
          shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights, preprocessScaledData=preprocessScaledData,
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
            threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns, CLASSIFICATION_MODE=CLASSIFICATION_MODE,
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