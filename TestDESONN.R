source("DESONN.R")
source("utils/utils.R")
source("utils/bootstrap_metadata.R")
library(readxl)
library(dplyr)

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
threshold <- .5  # Classification threshold (not directly used in Random Forest)

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
if(train) {
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
    # if (!dir.exists("artifacts_runs")) dir.create("artifacts_runs")
    
    seeds <- 1:50
    acc_rows <- vector("list", length(seeds))
    metrics_rows <- list()
    
    # Timestamp for uniqueness
    ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    
    # Number of seeds
    s <- as.character(length(seeds))
    
    # File paths with seed count in the name
    agg_pred_file <- file.path("artifacts", sprintf("predictions_stateless_scope-one_%s_%s_seeds.rds", ts_stamp, s))
    agg_metrics_file <- file.path("artifacts", sprintf("metrics_test_%s_%s_seeds.rds", ts_stamp, s))
    
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
      
      
      
      flat <- tryCatch(
        rapply(
          list(
            performance_metric = model_results$performance_relevance_data$performance_metric,
            relevance_metric   = model_results$performance_relevance_data$relevance_metric
          ),
          f = function(z) z, how = "unlist"
        ),
        error = function(e) setNames(vector("list", 0L), character(0))
      )
      
      # keep only length-1 atomics
      if (length(flat)) {
        L <- as.list(flat)
        flat <- flat[vapply(L, is.atomic, logical(1)) & lengths(L) == 1L]
      }
      
      # exclude bulky keys + drop NAs
      nms <- names(flat)
      if (length(nms)) {
        drop <- grepl("custom_relative_error_binned", nms, fixed = TRUE) |
          grepl("grid_used", nms, fixed = TRUE) |
          grepl("(^|\\.)details(\\.|$)", nms)   # drop any *.details.*
        keep <- !drop & !is.na(flat)
        flat <- flat[keep]
        nms  <- names(flat)
      }
      
      # --- build a named list so column names are metric names (not values) ---
      if (length(flat) == 0L) {
        row_df <- data.frame(run_index = i, seed = s, stringsAsFactors = FALSE)
      } else {
        out <- setNames(vector("list", length(flat)), nms)
        # numeric if possible, else character — preserve names
        num <- suppressWarnings(as.numeric(flat))
        for (j in seq_along(flat)) {
          out[[j]] <- if (!is.na(num[j])) num[j] else as.character(flat[[j]])
        }
        row_df <- as.data.frame(out, check.names = TRUE, stringsAsFactors = FALSE)
        row_df <- cbind(data.frame(run_index = i, seed = s, stringsAsFactors = FALSE), row_df)
      }
      
      # --- add best-* fields (robust, no helpers) ---
      row_df$best_train_acc  <- tryCatch(as.numeric(model_results$best_train_acc),  error = function(e) NA_real_)
      row_df$best_epoch_train<- tryCatch(as.integer(model_results$best_epoch_train),error = function(e) NA_integer_)
      row_df$best_val_acc    <- tryCatch(as.numeric(model_results$best_val_acc),    error = function(e) NA_real_)
      row_df$best_val_epoch  <- tryCatch(as.integer(model_results$best_val_epoch),  error = function(e) NA_integer_)
      
      metrics_rows[[i]] <- row_df
      if (i == length(seeds)) main_model <- run_model
      cat(sprintf("Seed %d → collected %d metrics\n", s, max(0, ncol(row_df) - 2L)))
      
      #RUN TEST RESULTS TABLE
      # evaluate via ENV (example) or set LOAD_FROM_RDS=TRUE to read artifacts
      desonn_predict_eval(
        LOAD_FROM_RDS = FALSE,
        ENV_META_NAME = "Ensemble_Main_0_model_1_metadata",
        INPUT_SPLIT   = "test",
        CLASSIFICATION_MODE = CLASSIFICATION_MODE,
        RUN_INDEX = i,
        SEED      = s,
        OUTPUT_DIR = "artifacts",
        SAVE_METRICS_RDS = FALSE,                             # avoid per-seed metrics files
        METRICS_PREFIX   = "metrics_test",
        SAVE_PREDICTIONS_COLUMN_IN_RDS = FALSE,              # keep lightweight agg file
        AGG_PREDICTIONS_FILE = agg_pred_file,                # << aggregate here
        AGG_METRICS_FILE     = agg_metrics_file              # << and here
      )
      
    }
    
    # --- bind with base-R fill ---
    if (length(metrics_rows) == 0L) {
      results_table <- data.frame()
    } else {
      results_table <- metrics_rows[[1]]
      if (length(metrics_rows) > 1L) {
        for (k in 2:length(metrics_rows)) {
          x <- results_table
          y <- metrics_rows[[k]]
          for (m in setdiff(names(y), names(x))) x[[m]] <- NA
          for (m in setdiff(names(x), names(y))) y[[m]] <- NA
          ord <- union(names(x), names(y))
          results_table <- rbind(x[, ord, drop = FALSE], y[, ord, drop = FALSE])
        }
      }
    }
    
    # remove "performance_metric." / "relevance_metric." prefixes
    colnames(results_table) <- sub("^(performance_metric|relevance_metric)\\.", "", colnames(results_table))
    
    # keep best_train_acc/epoch + best_val_epoch, but DROP best_val_acc before final table
    if ("best_val_acc" %in% names(results_table)) results_table$best_val_acc <- NULL
    
    
    # save
    dir.create("artifacts", showWarnings = FALSE, recursive = TRUE)
    out_path <- file.path(
      "artifacts",
      sprintf("train_acc_validation_metrics_runs_%s_%s_seeds.rds",
              format(Sys.time(), "%Y%m%d_%H%M%S"), s)
    )
    saveRDS(results_table, out_path)
    cat("Saved multi-seed metrics table to:", out_path, " | rows=", nrow(results_table),
        " cols=", ncol(results_table), "\n")
  
    
    
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
                # only take numeric scalars (skip nested lists like accuracy_precision_recall_f1_tuned$details)
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



desonn_predict_eval <- function(
    LOAD_FROM_RDS = FALSE,                            # if TRUE, load meta RDS; else from ENV object
    ENV_META_NAME = "Ensemble_Main_0_model_1_metadata",
    INPUT_SPLIT   = "test",                           # "test" | "validation" | "train"
    CLASSIFICATION_MODE,                              # "binary" | "multiclass" | "regression"
    RUN_INDEX,
    SEED,
    OUTPUT_DIR = "artifacts",                         # files saved here
    SAVE_METRICS_RDS = TRUE,                          # write the flattened metrics table RDS (unless AGG_METRICS_FILE provided)
    METRICS_PREFIX = "metrics_test",                  # artifacts/<prefix>_runXXX_seedY_YYYYmmdd_HHMMSS.rds
    SAVE_PREDICTIONS_COLUMN_IN_RDS = get0("SAVE_PREDICTIONS_COLUMN_IN_RDS", inherits=TRUE, ifnotfound=FALSE),
    AGG_PREDICTIONS_FILE = NULL,                      # if provided, append all seeds into one predictions RDS here
    AGG_METRICS_FILE     = NULL                       # if provided, append all seeds into one metrics RDS table here
) {
  # ---------------- helpers ----------------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  r6 <- function(x) {
    if (is.null(x)) return(NA_real_)
    if (is.list(x)) x <- unlist(x, recursive = TRUE, use.names = FALSE)
    suppressWarnings({
      xn <- as.numeric(x[1]); if (!is.finite(xn)) return(NA_real_); round(xn, 6)
    })
  }
  digest_safe <- function(x) {
    if (!requireNamespace("digest", quietly=TRUE)) return(NA_character_)
    tryCatch(digest::digest(x, algo="xxhash64"), error=function(e) NA_character_)
  }
  .as_numeric_vector_strict <- function(v, nm = "y") {
    if (is.matrix(v)) {
      if (ncol(v) > 1L) stop(sprintf("[coerce:%s] matrix has %d cols (expect 1)", nm, ncol(v)))
      v <- v[,1L, drop=TRUE]
    }
    if (is.data.frame(v)) {
      if (ncol(v) != 1L) stop(sprintf("[coerce:%s] data.frame has %d cols (expect 1)", nm, ncol(v)))
      v <- v[[1L]]
    }
    if (is.list(v)) {
      v <- vapply(v, function(x) {
        if (is.null(x)) return(NA_real_)
        if (is.list(x)) x <- unlist(x, recursive=TRUE, use.names=FALSE)
        suppressWarnings(as.numeric(if (length(x)) x[1] else NA))
      }, numeric(1))
    }
    if (is.factor(v)) v <- as.character(v)
    if (!is.numeric(v)) suppressWarnings(v <- as.numeric(v))
    if (!is.numeric(v)) stop(sprintf("[coerce:%s] cannot coerce to numeric", nm))
    if (!length(v)) stop(sprintf("[coerce:%s] zero-length", nm))
    v
  }
  .as_numeric_matrix_strict <- function(X, nm = "X") {
    if (is.matrix(X) && is.numeric(X)) return(X)
    if (is.vector(X) && !is.list(X)) { X <- matrix(X, ncol=1L); colnames(X) <- nm }
    if (is.data.frame(X) || is.matrix(X)) {
      Xdf <- as.data.frame(X, stringsAsFactors=FALSE)
      for (cc in names(Xdf)) {
        col <- Xdf[[cc]]
        if (is.list(col)) {
          col <- vapply(col, function(x) {
            if (is.null(x)) return(NA_real_)
            if (is.list(x)) x <- unlist(x, recursive=TRUE, use.names=FALSE)
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
      if (length(bad)) stop(sprintf("[coerce:%s] entirely non-numeric cols: %s", nm, paste(colnames(Xmat)[bad], collapse=", ")))
      return(Xmat)
    }
    stop(sprintf("[coerce:%s] unsupported type: %s", nm, paste(class(X), collapse=",")))
  }
  .get_tt_strict <- function(meta_local) {
    tt <- meta_local$target_transform %||%
      (meta_local$preprocessScaledData %||% list())$target_transform %||%
      (meta_local$model %||% list())$target_transform %||%
      tryCatch({
        j <- meta_local$target_transform_json
        if (length(j) && is.character(j) && nzchar(j)) jsonlite::fromJSON(j) else NULL
      }, error=function(e) NULL)
    if (is.null(tt) || is.null(tt$type)) {
      list(type="identity", params=list(center=0, scale=1))
    } else {
      p <- tt$params %||% list(); p$center <- p$center %||% 0; p$scale <- p$scale %||% 1
      list(type=tolower(as.character(tt$type)), params=p)
    }
  }
  .apply_target_inverse_strict <- function(P_raw_local, meta_local) {
    tt <- .get_tt_strict(meta_local)
    ttype <- tolower(tt$type %||% "identity")
    c0 <- as.numeric(tt$params$center %||% 0)
    s0 <- as.numeric(tt$params$scale  %||% 1)
    if (!is.finite(s0) || s0 == 0) s0 <- 1
    if (ttype == "zscore")  return(P_raw_local * s0 + c0)
    if (ttype == "minmax")  { mn <- as.numeric(tt$params$min %||% 0); mx <- as.numeric(tt$params$max %||% 1); return(P_raw_local * (mx - mn) + mn) }
    if (ttype == "affine")  { a <- as.numeric(tt$params$a); b <- as.numeric(tt$params$b); if (!is.finite(a) || !is.finite(b)) stop("invalid affine"); return(a + b * P_raw_local) }
    if (abs(c0) > 1e-12 || abs(s0 - 1) > 1e-12) {
      warning("target_transform 'identity' has nontrivial center/scale; treating as zscore."); return(P_raw_local * s0 + c0)
    }
    P_raw_local
  }
  
  # ---------- safe, mode-aware predict shims ----------
  .safe_run_predict <- function(X, meta, model_index = 1L, ML_NN = TRUE, verbose = FALSE, debug = FALSE) {
    if (exists(".run_predict", inherits = TRUE)) {
      return(tryCatch(
        .run_predict(X = X, meta = meta, model_index = model_index, ML_NN = ML_NN,
                     verbose = verbose, debug = debug),
        error = function(e) { message("[.run_predict] failed, falling back: ", conditionMessage(e)); NULL }
      ))
    }
    mdl <- meta$model %||% NULL
    if (!is.null(mdl) && is.function(mdl$predict)) {
      return(tryCatch(mdl$predict(Rdata = X, weights = meta$weights_record %||% meta$weights,
                                  biases = meta$biases_record %||% meta$biases,
                                  activation_functions = meta$activation_functions %||% NULL,
                                  verbose = verbose, debug = debug),
                      error=function(e) NULL))
    }
    SONN_local <- get0("SONN", inherits = TRUE, ifnotfound = NULL)
    if (!is.null(SONN_local) && is.function(SONN_local$predict)) {
      return(tryCatch(SONN_local$predict(Rdata = X, weights = meta$weights_record %||% meta$weights,
                                         biases = meta$biases_record %||% meta$biases,
                                         activation_functions = meta$activation_functions %||% NULL,
                                         verbose = verbose, debug = debug),
                      error=function(e) NULL))
    }
    stop("No available predict method (.run_predict / meta$model$predict / SONN$predict).")
  }
  
  .as_pred_matrix <- function(pred_obj, mode = c("binary","multiclass","regression"), meta, DEBUG = FALSE) {
    mode <- match.arg(mode)
    if (is.list(pred_obj) && !is.null(pred_obj$predicted_output)) {
      P <- pred_obj$predicted_output
    } else if (is.matrix(pred_obj)) {
      P <- pred_obj
    } else if (is.data.frame(pred_obj)) {
      P <- as.matrix(pred_obj)
    } else {
      stop("[as_pred] Unsupported prediction object type.")
    }
    storage.mode(P) <- "double"
    if (!is.matrix(P) || nrow(P) == 0L) stop("[as_pred] empty prediction matrix")
    if (mode == "regression" && ncol(P) > 1L) { if (DEBUG) message("[as_pred] regression but >1 cols; using col 1"); P <- P[,1,drop=FALSE] }
    P
  }
  
  # ---------------- config + env bits ----------------
  CLASSIFICATION_MODE <- tolower(CLASSIFICATION_MODE)
  stopifnot(CLASSIFICATION_MODE %in% c("binary","multiclass","regression"))
  CLASS_THRESHOLD <- as.numeric(get0("CLASS_THRESHOLD", inherits=TRUE, ifnotfound=0.5))
  SONN           <- get0("SONN",     inherits=TRUE, ifnotfound=NULL)
  verbose_flag   <- isTRUE(get0("verbose", inherits=TRUE, ifnotfound=FALSE))
  
  cat("[CFG] SPLIT=", INPUT_SPLIT, " | CLASS_MODE=", CLASSIFICATION_MODE,
      " | RUN_INDEX=", RUN_INDEX, " | SEED=", SEED, " | OUT=", OUTPUT_DIR, "\n", sep="")
  
  # ---------------- load meta ----------------
  meta <- NULL
  if (LOAD_FROM_RDS) {
    adir <- get0(".BM_DIR", inherits=TRUE, ifnotfound="artifacts")
    if (!dir.exists(adir)) stop(sprintf("Artifacts dir not found: %s", adir))
    patt <- paste0("(?i)(?:^|_)Ensemble_Main_0_model_1_metadata_.*\\.[Rr][Dd][Ss]$")
    files <- list.files(adir, pattern="\\.[Rr][Dd][Ss]$", full.names=TRUE, recursive=TRUE, include.dirs=FALSE)
    hit <- grepl(paste0(patt), basename(files), perl=TRUE)
    if (!any(hit)) stop("No RDS metadata found for Ensemble_Main_0_model_1_metadata in '", adir, "'.")
    cand <- files[hit]; info <- file.info(cand)
    file <- cand[order(info$mtime, decreasing=TRUE)][1L]
    meta <- tryCatch({ m <- readRDS(file); attr(m,"artifact_path") <- file; m }, error=function(e) NULL)
    if (is.null(meta)) stop("Failed to read metadata RDS: ", file)
    cat("[LOAD] meta: RDS → ", attr(meta,"artifact_path"), "\n", sep="")
  } else {
    if (!exists(ENV_META_NAME, inherits=TRUE)) stop("ENV meta object not found: ", ENV_META_NAME)
    meta <- get(ENV_META_NAME, inherits=TRUE)
    cat("[LOAD] meta: ENV → ", ENV_META_NAME, "\n", sep="")
  }
  
  # ---------------- pick split strictly from meta ----------------
  sl <- tolower(INPUT_SPLIT)
  if (sl == "test")            { Xi_raw <- meta$X_test;       yi_raw <- meta$y_test;       split_used <- "test" }
  else if (sl == "validation") { Xi_raw <- meta$X_validation; yi_raw <- meta$y_validation; split_used <- "validation" }
  else if (sl == "train")      { Xi_raw <- meta$X %||% meta$X_train; yi_raw <- meta$y %||% meta$y_train; split_used <- "train" }
  else {
    Xi_raw <- meta$X_validation; yi_raw <- meta$y_validation; split_used <- "validation"
    if (is.null(Xi_raw) || is.null(yi_raw)) { Xi_raw <- meta$X_test; yi_raw <- meta$y_test; split_used <- "test" }
    if (is.null(Xi_raw) || is.null(yi_raw)) { Xi_raw <- meta$X %||% meta$X_train; yi_raw <- meta$y %||% meta$y_train; split_used <- "train" }
  }
  if (is.null(Xi_raw) || is.null(yi_raw)) stop("Requested split not present in metadata: ", INPUT_SPLIT)
  cat(sprintf("[SPLIT] %s | rows(X)=%d | cols(X)=%d\n", split_used, NROW(Xi_raw), NCOL(Xi_raw)))
  
  # ---------------- coerce + align ----------------
  if (is.data.frame(Xi_raw)) {
    has_list <- vapply(Xi_raw, is.list, logical(1))
    if (any(has_list)) cat(sprintf("[WARN] %d list-cols in X: %s\n", sum(has_list), paste(names(Xi_raw)[has_list], collapse=", ")))
  }
  Xi <- .as_numeric_matrix_strict(Xi_raw, nm="X")
  yi <- .as_numeric_vector_strict(yi_raw,  nm="y")
  if (NROW(Xi_raw) != length(yi)) stop(sprintf("[LABEL-CHK] NROW(X)=%d vs len(y)=%d", NROW(Xi_raw), length(yi)))
  expected <- tryCatch({ nms <- meta$feature_names %||% meta$input_names %||% meta$colnames; if (is.null(nms)) colnames(Xi) else as.character(nms) }, error=function(e) colnames(Xi))
  orig_cols <- colnames(Xi); miss <- setdiff(expected, orig_cols)
  if (length(miss)) Xi <- cbind(Xi, matrix(0, nrow=nrow(Xi), ncol=length(miss), dimnames=list(NULL, miss)))
  Xi <- Xi[, expected, drop=FALSE]
  
  # ---------------- predict (safe + stateless) ----------------
  t0 <- proc.time()
  out <- .safe_run_predict(
    X = Xi, meta = meta, model_index = 1L, ML_NN = TRUE,
    verbose = isTRUE(get0("VERBOSE_RUNPRED", inherits=TRUE, ifnotfound=FALSE)),
    debug   = isTRUE(get0("DEBUG_RUNPRED",   inherits=TRUE, ifnotfound=FALSE))
  )
  P_raw <- .as_pred_matrix(
    out, mode = if (CLASSIFICATION_MODE=="regression") "regression" else "binary",
    meta = meta, DEBUG = isTRUE(get0("DEBUG_ASPM", inherits=TRUE, ifnotfound=FALSE))
  )
  if (is.null(colnames(P_raw))) colnames(P_raw) <- "pred"
  if (!is.matrix(P_raw) || nrow(P_raw) == 0L) stop("Empty predictions from model.")
  t_pred <- as.numeric((proc.time() - t0)[["elapsed"]])
  
  cat(sprintf("[PRED] dims=%dx%d | mean=%f sd=%f | min=%f p50=%f max=%f\n",
              nrow(P_raw), ncol(P_raw), mean(P_raw), stats::sd(P_raw),
              min(P_raw), as.numeric(stats::median(P_raw)), max(P_raw)))
  
  # ---------------- post-process per mode ----------------
  if (CLASSIFICATION_MODE == "regression") {
    P <- .apply_target_inverse_strict(P_raw, meta)
  } else if (CLASSIFICATION_MODE == "binary") {
    if (ncol(P_raw) == 1L) {
      P <- P_raw
    } else {
      mx <- apply(P_raw, 1, max); ex <- exp(P_raw - mx); sm <- rowSums(ex)
      P  <- matrix((ex / sm)[, 2L], ncol=1L)  # positive class prob
    }
  } else { # multiclass
    mx <- apply(P_raw, 1, max); ex <- exp(P_raw - mx); sm <- rowSums(ex)
    P  <- ex / sm
  }
  
  # ---------------- metrics ----------------
  yi_vec <- as.numeric(yi)
  acc <- prec <- rec <- f1s <- NA_real_; cm_base <- NULL
  if (CLASSIFICATION_MODE == "binary") {
    y_true <- if (all(yi_vec %in% c(0,1))) as.integer(yi_vec) else as.integer(yi_vec >= 0.5)
    p_pos  <- as.numeric(P[,1]); thr <- CLASS_THRESHOLD
    yhat   <- as.integer(p_pos >= thr)
    TP <- sum(yhat==1 & y_true==1); FP <- sum(yhat==1 & y_true==0)
    TN <- sum(yhat==0 & y_true==0); FN <- sum(yhat==0 & y_true==1)
    acc <- (TP + TN) / length(y_true)
    prec <- if ((TP+FP)>0) TP/(TP+FP) else 0
    rec  <- if ((TP+FN)>0) TP/(TP+FN) else 0
    f1s  <- if ((prec+rec)>0) 2*prec*rec/(prec+rec) else 0
    cm_base <- list(TP=TP, FP=FP, TN=TN, FN=FN)
  } else if (CLASSIFICATION_MODE == "multiclass") {
    yhat <- max.col(P, ties.method="first")
    ymc  <- if (is.matrix(yi_raw) && ncol(yi_raw)>1) max.col(yi_raw, "first") else as.integer(yi_vec)
    acc  <- mean(yhat == ymc)
    K <- max(yhat, ymc, na.rm=TRUE)
    macro_prec <- macro_rec <- macro_f1 <- numeric(K)
    for (k in seq_len(K)) {
      TPk <- sum(yhat==k & ymc==k)
      FPk <- sum(yhat==k & ymc!=k)
      FNk <- sum(yhat!=k & ymc==k)
      pk <- if ((TPk+FPk)>0) TPk/(TPk+FPk) else 0
      rk <- if ((TPk+FNk)>0) TPk/(TPk+FNk) else 0
      fk <- if ((pk+rk)>0) 2*pk*rk/(pk+rk) else 0
      macro_prec[k] <- pk; macro_rec[k] <- rk; macro_f1[k] <- fk
    }
    prec <- mean(macro_prec); rec <- mean(macro_rec); f1s <- mean(macro_f1)
  }
  
  # regression-style metrics (NA where not applicable)
  mse_val   <- tryCatch(MSE(SONN, Xi, yi_vec, "regression", P, verbose_flag),   error=function(e) NA_real_)
  mae_val   <- tryCatch(MAE(SONN, Xi, yi_vec, "regression", P, verbose_flag),   error=function(e) NA_real_)
  rmse_val  <- tryCatch(RMSE(SONN, Xi, yi_vec, "regression", P, verbose_flag),  error=function(e) NA_real_)
  r2_val    <- tryCatch(R2(SONN, Xi, yi_vec, "regression", P, verbose_flag),    error=function(e) NA_real_)
  mape_val  <- tryCatch(MAPE(SONN, Xi, yi_vec, "regression", P, verbose_flag),  error=function(e) NA_real_)
  smape_val <- tryCatch(SMAPE(SONN, Xi, yi_vec, "regression", P, verbose_flag), error=function(e) NA_real_)
  wmape_val <- tryCatch(WMAPE(SONN, Xi, yi_vec, "regression", P, verbose_flag), error=function(e) NA_real_)
  mase_val  <- tryCatch(MASE(SONN, Xi, yi_vec, "regression", P, verbose_flag),  error=function(e) NA_real_)
  
  tuned <- tryCatch(
    accuracy_precision_recall_f1_tuned(
      SONN = SONN, Rdata = Xi, labels = yi_vec,
      CLASSIFICATION_MODE = CLASSIFICATION_MODE, predicted_output = P,
      metric_for_tuning = get0("METRIC_FOR_TUNING", inherits=TRUE, ifnotfound="accuracy"),
      threshold_grid    = get0("THRESHOLD_GRID",    inherits=TRUE, ifnotfound=seq(0.05,0.95,by=0.01)),
      verbose = isTRUE(get0("TUNED_VERBOSE", inherits=TRUE, ifnotfound=FALSE))
    ),
    error=function(e) {
      message("[tuned] failed: ", conditionMessage(e))
      list(accuracy=NA_real_, precision=NA_real_, recall=NA_real_, f1=NA_real_,
           confusion_matrix=NULL, details=list(best_threshold=NA_real_, tuned_by="error"))
    }
  )
  
  mem_bytes <- tryCatch(as.numeric(utils::object.size(list(Xi=Xi,P=P,meta=meta))), error=function(e) NA_real_)
  
  cat(sprintf("[METR] acc=%.6f | prec=%.6f | rec=%.6f | f1=%.6f | tuned_thr=%s | tuned_f1=%s | RMSE=%s\n",
              r6(acc), r6(prec), r6(rec), r6(f1s), r6(tuned$details$best_threshold), r6(tuned$f1), r6(rmse_val)))
  
  # ---------------- build flattened row like training ----------------
  performance_metric <- list(
    quantization_error    = NA_real_,
    topographic_error     = NA_real_,
    clustering_quality_db = NA_real_,
    MSE   = r6(mse_val),  MAE = r6(mae_val),  RMSE = r6(rmse_val),  R2   = r6(r2_val),
    MAPE  = r6(mape_val), SMAPE = r6(smape_val), WMAPE = r6(wmape_val), MASE = r6(mase_val),
    accuracy  = r6(acc), precision = r6(prec), recall = r6(rec), f1_score = r6(f1s),
    confusion_matrix = cm_base,
    accuracy_precision_recall_f1_tuned = tuned,
    speed = r6(t_pred), speed_learn = NA_real_, memory_usage = r6(mem_bytes), robustness = NA_real_,
    custom_relative_error_binned = NA
  )
  relevance_metric <- list(
    hit_rate=NA_real_, ndcg=NA_real_, diversity=NA_real_, serendipity=NA_real_,
    precision_boolean=NA_real_, recall=NA_real_, f1_score=NA_real_, mean_precision=NA_real_,
    novelty=NA_real_
  )
  
  flat <- tryCatch(
    rapply(list(performance_metric=performance_metric, relevance_metric=relevance_metric),
           f=function(z) z, how="unlist"),
    error=function(e) setNames(vector("list",0L), character(0))
  )
  if (length(flat)) {
    L <- as.list(flat)
    flat <- flat[vapply(L, is.atomic, logical(1)) & lengths(L) == 1L]
  }
  nms <- names(flat)
  if (length(nms)) {
    drop <- grepl("custom_relative_error_binned", nms, fixed=TRUE) |
      grepl("grid_used", nms, fixed=TRUE) |
      grepl("(^|\\.)details(\\.|$)", nms)
    keep <- !drop & !is.na(flat)
    flat <- flat[keep]; nms <- names(flat)
  }
  if (length(flat) == 0L) {
    row_df <- data.frame(run_index=RUN_INDEX, seed=SEED, stringsAsFactors=FALSE)
  } else {
    out <- setNames(vector("list", length(flat)), nms)
    num <- suppressWarnings(as.numeric(flat))
    for (j in seq_along(flat)) out[[j]] <- if (!is.na(num[j])) num[j] else as.character(flat[[j]])
    row_df <- cbind(data.frame(run_index=RUN_INDEX, seed=SEED, stringsAsFactors=FALSE),
                    as.data.frame(out, check.names=TRUE, stringsAsFactors=FALSE))
  }
  colnames(row_df) <- sub("^performance_metric\\.", "", colnames(row_df))
  colnames(row_df) <- sub("^relevance_metric\\.",   "", colnames(row_df))
  
  # ---------------- compact summary (results_df) ----------------
  pred_hash <- tryCatch(digest_safe(round(P[seq_len(min(nrow(P), 2000)), , drop=FALSE], 6)), error=function(e) NA_character_)
  results_df <- data.frame(
    kind = if (LOAD_FROM_RDS) "RDS" else "ENV", ens = 0L, model = 1L,
    split_used = split_used, n_pred_rows = nrow(P),
    accuracy=r6(acc), precision=r6(prec), recall=r6(rec), f1=r6(f1s),
    tuned_threshold = r6(tuned$details$best_threshold),
    tuned_accuracy  = r6(tuned$accuracy), tuned_precision=r6(tuned$precision),
    tuned_recall    = r6(tuned$recall),   tuned_f1      = r6(tuned$f1),
    MSE=r6(mse_val), MAE=r6(mae_val), RMSE=r6(rmse_val), R2=r6(r2_val),
    MAPE=r6(mape_val), SMAPE=r6(smape_val), WMAPE=r6(wmape_val), MASE=r6(mase_val),
    pred_sig = pred_hash,
    model_rds = if (LOAD_FROM_RDS) basename(attr(meta,"artifact_path") %||% NA_character_) else NA_character_,
    artifact_used = if (LOAD_FROM_RDS) "yes" else "no",
    stringsAsFactors=FALSE
  )
  
  cat(sprintf("[RESULTS] rows=%d | cols=%d | mode=%s\n", nrow(results_df), ncol(results_df), CLASSIFICATION_MODE))
  print(utils::head(results_df, 10))
  
  # ---------------- save files ----------------
  dir.create(OUTPUT_DIR, recursive=TRUE, showWarnings=FALSE)
  ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  
  # ---- PREDICTIONS: aggregate-or-file ----
  if (!is.null(AGG_PREDICTIONS_FILE)) {
    # Create or append
    pack <- if (file.exists(AGG_PREDICTIONS_FILE)) {
      tryCatch(readRDS(AGG_PREDICTIONS_FILE), error=function(e) NULL)
    } else NULL
    if (is.null(pack) || !is.list(pack) || is.null(pack$entries)) {
      pack <- list(
        schema_version = "pred-agg-v1",
        created_at     = Sys.time(),
        flags          = list(CLASSIFICATION_MODE=CLASSIFICATION_MODE),
        meta_source    = if (LOAD_FROM_RDS) (attr(meta,"artifact_path") %||% NA_character_) else ENV_META_NAME,
        entries        = list(),    # per-seed entries
        seeds          = integer(0)
      )
    }
    key <- sprintf("seed_%s_run_%03d", as.character(SEED), as.integer(RUN_INDEX))
    entry <- list(
      run_index   = RUN_INDEX,
      seed        = SEED,
      results_df  = results_df,
      prediction_sig = pred_hash
    )
    if (isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) {
      entry$predictions <- list(Ensemble_Main_0_model_1 = P)
    }
    pack$entries[[key]] <- entry
    if (!(SEED %in% pack$seeds)) pack$seeds <- sort(unique(c(pack$seeds, SEED)))
    saveRDS(pack, AGG_PREDICTIONS_FILE)
    cat("[SAVE] predictions (aggregate) → ", AGG_PREDICTIONS_FILE,
        if (!isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) " (payload omitted)" else "",
        " | appended key=", key, "\n", sep="")
    predictions_path <- AGG_PREDICTIONS_FILE
  } else {
    predictions_path <- file.path(
      OUTPUT_DIR,
      sprintf("predictions_stateless_scope-one_src-%s_%s.rds", if (LOAD_FROM_RDS) "rds" else "env", ts_stamp)
    )
    predict_pack <- list(
      schema_version = "pred-v2",
      saved_at       = Sys.time(),
      predict_mode   = "stateless",
      flags = list(
        INPUT_SPLIT                   = INPUT_SPLIT,
        CLASSIFICATION_MODE           = CLASSIFICATION_MODE,
        CLASS_THRESHOLD               = CLASS_THRESHOLD,
        SAVE_PREDICTIONS_COLUMN_IN_RDS = isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)
      ),
      meta_source     = if (LOAD_FROM_RDS) (attr(meta,"artifact_path") %||% NA_character_) else ENV_META_NAME,
      results_table   = results_df,
      prediction_sigs = results_df$pred_sig
    )
    if (isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) {
      predict_pack$predictions <- list(Ensemble_Main_0_model_1 = P)
    }
    # Hard safety strip
    if (!isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS) && !is.null(predict_pack$predictions)) predict_pack$predictions <- NULL
    saveRDS(predict_pack, predictions_path)
    cat("[SAVE] predictions → ", predictions_path,
        if (!isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) " (payload omitted)" else "",
        "\n", sep="")
  }
  
  # ---- METRICS: aggregate-or-file ----
  if (!is.null(AGG_METRICS_FILE)) {
    met <- if (file.exists(AGG_METRICS_FILE)) {
      tryCatch(readRDS(AGG_METRICS_FILE), error=function(e) NULL)
    } else NULL
    if (is.null(met) || !is.data.frame(met)) met <- row_df[0, , drop=FALSE]
    # align columns
    common <- union(colnames(met), colnames(row_df))
    if (!all(common %in% colnames(met)))  for (cc in setdiff(common, colnames(met)))  met[[cc]] <- NA
    if (!all(common %in% colnames(row_df)))for (cc in setdiff(common, colnames(row_df)))row_df[[cc]] <- NA
    met <- rbind(met[,common,drop=FALSE], row_df[,common,drop=FALSE])
    saveRDS(met, AGG_METRICS_FILE)
    cat("[SAVE] metrics (aggregate)    → ", AGG_METRICS_FILE, " | total_rows=", nrow(met), "\n", sep="")
    metrics_path <- AGG_METRICS_FILE
  } else {
    metrics_path <- file.path(
      OUTPUT_DIR,
      sprintf("%s_run%03d_seed%s_%s.rds", METRICS_PREFIX,
              ifelse(is.na(RUN_INDEX), 0L, RUN_INDEX),
              ifelse(is.na(SEED), "NA", SEED),
              ts_stamp)
    )
    if (SAVE_METRICS_RDS) {
      saveRDS(row_df, metrics_path)
      cat("[SAVE] metrics     → ", metrics_path, " | rows=", nrow(row_df), " cols=", ncol(row_df), "\n", sep="")
    } else {
      metrics_path <- NA_character_
    }
  }
  
  invisible(list(
    results_df      = results_df,
    flat_table      = row_df,
    predictions_rds = predictions_path,
    metrics_rds     = metrics_path
  ))
}
















saveToDisk <- FALSE

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