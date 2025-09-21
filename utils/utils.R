# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Ensemble Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
# 
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

probe_preds_vs_labels <- function(preds, labs, tag = "GENERIC", save_global = FALSE) {
  r2_val <- tryCatch({
    ss_tot <- sum((labs - mean(labs))^2, na.rm = TRUE)
    ss_res <- sum((labs - preds)^2, na.rm = TRUE)
    1 - ss_res / ss_tot
  }, error = function(e) NA_real_)
  
  cat(sprintf(
    "[PROBE-R2 %s] preds min=%.4f mean=%.4f max=%.4f\n",
    tag, min(preds), mean(preds), max(preds)
  ))
  cat(sprintf(
    "[PROBE-R2 %s] labs  min=%.4f mean=%.4f max=%.4f\n",
    tag, min(labs), mean(labs), max(labs)
  ))
  cat(sprintf("[PROBE-R2 %s] R^2=%.6f (n=%d)\n",
              tag, r2_val, length(preds)))
  
  if (save_global) {
    dbg <- list(
      tag   = tag,
      preds = list(min = min(preds), mean = mean(preds), max = max(preds)),
      labs  = list(min = min(labs),  mean = mean(labs),  max = max(labs)),
      r2    = r2_val,
      n     = length(preds)
    )
    
    # Separate globals for train vs predict
    if (grepl("^TRAIN", tag)) {
      assign("probe_last_train", dbg, envir = .GlobalEnv)
    } else if (grepl("^PREDICT", tag)) {
      assign("probe_last_predict", dbg, envir = .GlobalEnv)
    } else {
      assign("probe_last_dbg", dbg, envir = .GlobalEnv)
    }
  }
}

probe_last_layer <- function(weights, biases, y, tag = "GENERIC", save_global = TRUE) {
  W_last <- weights[[length(weights)]]
  b_last <- biases[[length(biases)]]
  
  stats <- list(
    tag = tag,
    W_last = list(
      dims  = dim(W_last),
      mean  = mean(W_last),
      sd    = sd(W_last),
      min   = min(W_last),
      max   = max(W_last)
    ),
    b_last = list(
      len   = length(b_last),
      mean  = mean(b_last),
      sd    = sd(b_last),
      min   = min(b_last),
      max   = max(b_last),
      head  = head(b_last, 10L)
    ),
    y = list(
      n     = length(y),
      mean  = mean(y),
      sd    = sd(y),
      min   = min(y),
      max   = max(y),
      head  = head(y, 10L)
    )
  )
  
  cat(sprintf(
    "[LASTLAYER %s] W dims=%s | mean=%.6f sd=%.6f range=[%.3f, %.3f]\n",
    tag, paste(dim(W_last), collapse="x"),
    stats$W_last$mean, stats$W_last$sd,
    stats$W_last$min, stats$W_last$max
  ))
  cat(sprintf(
    "[LASTLAYER %s] b len=%d | mean=%.6f range=[%.3f, %.3f]\n",
    tag, stats$b_last$len,
    stats$b_last$mean, stats$b_last$min, stats$b_last$max
  ))
  cat(sprintf(
    "[LASTLAYER %s] y n=%d | mean=%.6f sd=%.6f range=[%.3f, %.3f]\n",
    tag, stats$y$n,
    stats$y$mean, stats$y$sd,
    stats$y$min, stats$y$max
  ))
  
  if (save_global) {
    # store in global env
    assign(paste0("probe_last_layer_", tag), stats, envir = .GlobalEnv)
    
    # save RDS snapshot in artifacts
    fname <- sprintf("artifacts/probe_last_layer_%s_%s.rds",
                     tag, format(Sys.time(), "%Y%m%d_%H%M%S"))
    saveRDS(stats, fname)
    cat("[LASTLAYER] Snapshot saved to:", fname, "\n")
  }
  
  invisible(stats)
}




## ===== Shared plot filename helper =====
# Builds a filename prefixer for a specific context
# utils_plots.R

make_fname_prefix <- function(do_ensemble,
                              num_networks = NULL,
                              total_models = NULL,
                              ensemble_number = NULL,
                              model_index = NULL,
                              who) {
  if (missing(who) || !nzchar(who)) stop("'who' must be 'SONN' or 'DESONN'")
  who <- toupper(who)
  
  if (is.null(total_models))
    total_models <- if (!is.null(num_networks)) num_networks else get0("num_networks", ifnotfound = 1L)
  
  as_int_or_na <- function(x) {
    if (is.null(x) || length(x) == 0 || is.na(x)) return(NA_integer_)
    as.integer(x)
  }
  
  ens <- as_int_or_na(ensemble_number)
  mod <- as_int_or_na(model_index)
  tot <- as_int_or_na(total_models); if (is.na(tot)) tot <- 1L
  
  if (isTRUE(do_ensemble)) {
    return(function(base_name) {
      paste0(
        "DESONN", if (!is.na(ens)) paste0("_", ens),    # omit if NA
        "_SONN",  if (!is.na(mod)) paste0("_", mod),
        "_", base_name
      )
    })
  }
  
  if (!is.na(tot) && tot > 1L) {
    if (who == "SONN") {
      return(function(base_name) {
        prefix <- if (!is.na(mod)) sprintf("SONN_%dof%d_", mod, tot) else sprintf("SONN_of%d_", tot)
        paste0(prefix, base_name)
      })
    } else if (who == "DESONN") {
      return(function(base_name) paste0(sprintf("SONN_%d-%d_", 1L, tot), base_name))
    } else {
      stop("invalid 'who'")
    }
  }
  
  # single-model case
  function(base_name) {
    paste0("SONN", if (!is.na(mod)) paste0("_", mod), "_", base_name)
  }
}

# ---- helper for
#  prepare_disk_only
# ----- tiny helper: cleanly stop script right here -----
.hard_stop <- function(msg = "[prepare_disk_only] Done; stopping script.") {
  cat(msg, "\n")
  if (interactive()) {
    # In RStudio: stop evaluating this script, but do NOT kill the session
    stop(invisible(structure(list(message = msg),
                             class = c("simpleError","error","condition"))),
         call. = FALSE)
  } else {
    # From Rscript/terminal: exit process
    quit(save = "no")
  }
}

#function used in Optimzers.R
lookahead_update <- function(params, grads_list, lr, beta1, beta2, epsilon, lookahead_step, base_optimizer, epoch, lambda) {
  updated_params_list <- list()
  
  cat(">> Lookahead optimizer running\n")
  
  # grads_list is just the matrix for this layer
  grad_matrix <- if (is.list(grads_list)) grads_list[[1]] else grads_list
  
  if (is.null(grad_matrix)) stop("Missing gradient matrix")
  
  # ✅ FIXED: Don't double-index
  param_list <- params
  
  param <- param_list$param
  m <- param_list$m
  v <- param_list$v
  r <- param_list$r
  slow_param <- param_list$slow_weights
  lookahead_counter <- param_list$lookahead_counter
  lookahead_step_layer <- param_list$lookahead_step
  
  if (is.null(lookahead_counter)) {
    lookahead_counter <- 0
    cat("Initialized lookahead_counter = 0\n")
  }
  
  if (is.null(lookahead_step_layer)) {
    lookahead_step_layer <- lookahead_step
  }
  
  if (base_optimizer == "adam_update") {
    m <- beta1 * m + (1 - beta1) * grad_matrix
    v <- beta2 * v + (1 - beta2) * (grad_matrix^2)
    
    m_hat <- m / (1 - beta1^epoch)
    v_hat <- v / (1 - beta2^epoch)
    
    update <- lr * m_hat / (sqrt(v_hat) + epsilon)
    param <- param - update
    
  } else if (base_optimizer == "lamb_update") {
    m <- beta1 * m + (1 - beta1) * grad_matrix
    v <- beta2 * v + (1 - beta2) * (grad_matrix^2)
    
    m_hat <- m / (1 - beta1^epoch)
    v_hat <- v / (1 - beta2^epoch)
    
    r1 <- sqrt(sum(param^2))
    r2 <- sqrt(sum((m_hat / (sqrt(v_hat) + epsilon))^2))
    ratio <- ifelse(r1 == 0 | r2 == 0, 1, r1 / r2)
    
    update <- lr * ratio * m_hat / (sqrt(v_hat) + epsilon)
    param <- param - update
    
  } else {
    stop("Unsupported base optimizer in lookahead_update()")
  }
  
  lookahead_counter <- lookahead_counter + 1
  if (lookahead_counter >= lookahead_step_layer) {
    cat(">> Lookahead sync\n")
    slow_param <- param
    lookahead_counter <- 0
  }
  
  updated_params_list <- list(
    param = param,
    m = m,
    v = v,
    r = r,
    slow_weights = slow_param,
    lookahead_counter = lookahead_counter,
    lookahead_step = lookahead_step_layer,
    weights_update = update
  )
  
  return(updated_params_list)
}


# =======================
# PREDICT-ONLY HELPERS (for !train)
# =======================

# Needed: convert best_*_record to clean {weights,biases} for SL/ML prediction
if (!exists("normalize_records", inherits = TRUE)) {
  normalize_records <- function(wrec, brec, ML_NN) {
    if (ML_NN) {
      stopifnot(is.list(wrec), is.list(brec))
      list(
        weights = lapply(wrec, function(w) as.matrix(w)),
        biases  = lapply(brec, function(b) as.numeric(if (is.matrix(b)) b else b))
      )
    } else {
      list(
        weights = as.matrix(if (is.list(wrec)) wrec[[1]] else wrec),
        biases  = as.numeric(if (is.list(brec)) brec[[1]] else brec)
      )
    }
  }
}

# Needed: pulls best_*_record (supports nested best_model_metadata)
if (!exists("extract_best_records", inherits = TRUE)) {
  extract_best_records <- function(meta, ML_NN, model_index = 1L) {
    src <- if (!is.null(meta$best_model_metadata) &&
               !is.null(meta$best_model_metadata$best_weights_record)) {
      meta$best_model_metadata
    } else {
      meta
    }
    if (is.null(src$best_weights_record) || is.null(src$best_biases_record))
      stop("Metadata does not contain best_*_record fields.")
    normalize_records(
      wrec = src$best_weights_record[[model_index]],
      brec = src$best_biases_record[[model_index]],
      ML_NN = ML_NN
    )
  }
}

# ===== helpers used by predict-only flow =====
`%||%` <- function(a,b) if (is.null(a) || length(a)==0) b else a

.get_in <- function(x, path) {
  cur <- x
  for (p in path) {
    if (is.null(cur)) return(NULL)
    if (is.list(cur) && !is.null(cur[[p]])) cur <- cur[[p]] else return(NULL)
  }
  cur
}

.choose_X_from_meta <- function(meta) {
  cands <- list(
    c("datasets","X_validation"), c("datasets","X_val"), c("datasets","X_test"),
    c("X_validation"), c("X_val"), c("X_test"),
    c("datasets","X_validation_scaled"), c("X_validation_scaled"),
    c("datasets","X_train"), c("X")
  )
  for (cand in cands) {
    v <- .get_in(meta, cand); if (!is.null(v)) return(list(X=v, tag=paste(cand, collapse="/")))
  }
  NULL
}

.choose_y_from_meta <- function(meta) {
  cands <- list(
    c("datasets","y_validation"), c("datasets","y_val"), c("datasets","y_test"),
    c("y_validation"), c("y_val"), c("y_test"),
    c("datasets","y_train"), c("y")
  )
  for (cand in cands) {
    v <- .get_in(meta, cand); if (!is.null(v)) return(list(y=v, tag=paste(cand, collapse="/")))
  }
  NULL
}

.normalize_y <- function(y) {
  if (is.null(y)) return(NULL)
  if (is.list(y) && length(y) == 1L) y <- y[[1]]
  if (is.data.frame(y)) y <- y[[1]]
  if (is.matrix(y) && ncol(y) == 1L) y <- y[,1]
  if (is.factor(y)) y <- as.integer(y) - 1L
  as.numeric(y)
}

.align_by_names_safe <- function(Xi, Xref) {
  Xi <- as.matrix(Xi)
  if (is.null(Xref)) return(Xi)
  Xref <- as.matrix(Xref)
  if (is.null(colnames(Xi)) || is.null(colnames(Xref))) return(Xi)
  keep <- intersect(colnames(Xref), colnames(Xi))
  if (!length(keep)) return(Xi)
  as.matrix(Xi[, keep, drop = FALSE])
}

.apply_scaling_if_any <- function(X, meta) {
  pp <- meta$preprocessScaledData %||% meta$preprocess %||% meta$scaler
  X <- as.matrix(X); storage.mode(X) <- "double"
  if (is.null(pp)) {
    assign("LAST_APPLIED_X", X, .GlobalEnv)
    return(X)
  }
  # Column order + add missing with 0
  exp_names <- pp$feature_names %||% colnames(X)
  miss <- setdiff(exp_names, colnames(X))
  if (length(miss)) {
    X <- cbind(X, matrix(0, nrow=nrow(X), ncol=length(miss),
                         dimnames=list(NULL, miss)))
  }
  X <- X[, exp_names, drop=FALSE]
  
  # Impute with train medians (no leakage)
  if (!is.null(pp$train_medians)) {
    for (nm in intersect(names(pp$train_medians), colnames(X))) {
      idx <- is.na(X[, nm]); if (any(idx)) X[idx, nm] <- pp$train_medians[[nm]]
    }
  }
  
  # Z-score with train center/scale
  if (!is.null(pp$center) && !is.null(pp$scale)) {
    sc <- pp$scale; sc[!is.finite(sc) | sc==0] <- 1
    X <- sweep(sweep(X, 2, pp$center, "-"), 2, sc, "/")
  }
  
  # Optional extra compression — disabled unless explicitly TRUE and >1
  if (isTRUE(pp$divide_by_max_val)) {
    mv <- as.numeric(pp$max_val %||% 1)
    if (is.finite(mv) && mv > 1) X <- X / mv
  }
  
  assign("LAST_APPLIED_X", X, .GlobalEnv)  # lets you inspect later
  X
}


# ---- Target transform helpers ----------------------------------------------
.get_target_transform <- function(meta) {
  meta$target_transform %||%
    (tryCatch(meta$preprocessScaledData$target_transform, error=function(e) NULL)) %||%
    (tryCatch(meta$preprocess$target_transform,         error=function(e) NULL))
}

._invert_target <- function(pred, tt, DEBUG=FALSE) {
  if (is.null(tt)) return(pred)
  type <- tolower(tt$type %||% "identity")
  par  <- tt$params %||% list()
  v <- as.numeric(pred[,1])
  
  if (type == "identity") {
    # do nothing
  } else if (type == "standardize") {
    mu <- par$y_mean; sdv <- par$y_sd
    if (is.finite(mu) && is.finite(sdv) && sdv > 0) v <- v * sdv + mu
    if (DEBUG) cat("[ASPM-DBG] inverse standardize applied\n")
  } else if (type == "minmax") {
    ymin <- par$y_min; ymax <- par$y_max
    if (is.finite(ymin) && is.finite(ymax)) v <- v * (ymax - ymin) + ymin
    if (DEBUG) cat("[ASPM-DBG] inverse minmax applied\n")
  } else if (type == "log") {
    base <- par$base %||% exp(1)
    v <- if (identical(base, 10)) 10^v else if (identical(base, 2)) 2^v else exp(v)
    if (!is.null(par$shift)) v <- v + par$shift
    if (DEBUG) cat("[ASPM-DBG] inverse log applied\n")
  } else if (type == "boxcox") {
    lambda <- par$lambda
    if (is.finite(lambda)) {
      if (abs(lambda) < 1e-8) v <- exp(v) else v <- (lambda*v + 1)^(1/lambda)
      if (!is.null(par$shift)) v <- v + par$shift
      if (DEBUG) cat("[ASPM-DBG] inverse boxcox applied\n")
    }
  } else if (type == "affine") {
    a <- par$a %||% 0; b <- par$b %||% 1
    v <- a + b * v
    if (DEBUG) cat("[ASPM-DBG] inverse affine applied\n")
  } else {
    if (DEBUG) cat("[ASPM-DBG] unknown target_transform; left as-is\n")
  }
  
  pred[,1] <- v
  pred
}

# ---- minimal predict-time prep using saved train scalers ----
prep_predict_X <- function(X_new, meta) {
  pp <- meta$preprocessScaledData
  stopifnot(!is.null(pp))
  
  # 1) same date handling as train
  if ("date" %in% names(X_new)) {
    d <- X_new[["date"]]
    if (inherits(d, "POSIXt"))      X_new[["date"]] <- as.numeric(as.Date(d))
    else if (inherits(d, "Date"))   X_new[["date"]] <- as.numeric(d)
    else {
      parsed <- suppressWarnings(as.Date(d))
      X_new[["date"]] <- if (all(is.na(parsed))) NA_real_ else as.numeric(parsed)
    }
  }
  
  # 2) align columns to training order
  fn <- pp$feature_names
  miss <- setdiff(fn, names(X_new))
  if (length(miss)) X_new[miss] <- NA_real_
  X_new <- X_new[, fn, drop = FALSE]
  
  # 3) numeric + TRAIN-median impute (no leakage)
  Xn <- as.data.frame(X_new)
  for (j in seq_along(fn)) {
    v <- suppressWarnings(as.numeric(Xn[[j]]))
    v[!is.finite(v)] <- NA_real_
    v[is.na(v)] <- pp$train_medians[[j]]
    Xn[[j]] <- v
  }
  Xn <- as.matrix(Xn); storage.mode(Xn) <- "double"
  
  # 4) apply TRAIN center/scale once (+ max_val if used in train)
  Xs <- sweep(sweep(Xn, 2, pp$center, "-"), 2, pp$scale, "/")
  mv <- pp$max_val %||% 1
  if (is.finite(mv) && mv != 0) Xs <- Xs / mv
  Xs
}


## =========================
## DEBUG TOGGLES (set TRUE)
## =========================
DEBUG_MODE_HELPER <- TRUE
DEBUG_ASPM        <- TRUE   # .as_pred_matrix()
DEBUG_SAFERUN     <- TRUE   # .safe_run_predict()
DEBUG_RUNPRED     <- TRUE   # .run_predict() shim

## ------------------------------------------------------------------------
## Null-coalescing helper
## ------------------------------------------------------------------------
`%||%` <- function(x, y) if (is.null(x)) y else x

## ------------------------------------------------------------------------
## Debug utilities (lightweight, no deps)
## ------------------------------------------------------------------------
if (!exists("LAST_DEBUG", inherits = TRUE)) {
  LAST_DEBUG <- new.env(parent = emptyenv())
}

# --- Drop-in replacement: robust, no integer overflow warnings ---
.hash_vec <- function(x) {
  if (requireNamespace("digest", quietly = TRUE)) {
    return(substr(digest::digest(x, algo = "xxhash64", serialize = TRUE), 1, 16))
  }
  nx <- suppressWarnings(as.numeric(x))
  if (!length(nx)) return("len0")
  fin <- is.finite(nx)
  if (!any(fin)) {
    s_len <- length(nx); s_sum <- 0; s_mean <- 0; s_sd <- 0
  } else {
    nx <- nx[fin]
    s_len <- length(nx); s_sum <- sum(nx); s_mean <- mean(nx); s_sd <- stats::sd(nx)
  }
  v <- abs(c(s_len, s_sum, s_mean, s_sd)) + c(1, 2, 3, 4)
  v[!is.finite(v)] <- 0
  v <- floor(v * 1e6)
  MOD <- (2^31 - 1)
  v <- as.double(v %% MOD)
  iv <- as.integer(v)
  paste(sprintf("%08x", iv), collapse = "")
}

.peek_num <- function(x, k = 6) {
  v <- tryCatch(as.numeric(x), error = function(e) numeric())
  if (!length(v)) return("")
  paste(sprintf("%.6f", utils::head(v, k)), collapse = ", ")
}

.summarize_matrix <- function(M) {
  nr <- NROW(M); nc <- NCOL(M)
  rng <- tryCatch(range(M, finite = TRUE), error=function(e) c(NA_real_, NA_real_))
  sprintf("dims=%sx%s | mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
          nr, nc, mean(M), sd(as.vector(M)), rng[1], stats::median(as.vector(M)), rng[2])
}

.in_range01 <- function(M) {
  rng <- suppressWarnings(range(M, finite = TRUE))
  isTRUE(is.finite(rng[1]) && is.finite(rng[2]) && rng[1] >= 0 && rng[2] <= 1)
}

## ------------------------------------------------------------------------
## Mode helper (global → meta → default) with optional tracing
## ------------------------------------------------------------------------
.get_mode <- function(meta) {
  g <- get0("CLASSIFICATION_MODE", inherits = TRUE, ifnotfound = NULL)
  m <- tryCatch(meta$CLASSIFICATION_MODE, error = function(e) NULL)
  final <- tolower(g %||% m %||% "regression")
  if (isTRUE(get0("DEBUG_MODE_HELPER", inherits = TRUE, ifnotfound = FALSE))) {
    cat(sprintf(
      "[MODE-DBG %s] resolved mode='%s' | global=%s | meta=%s | default=regression\n",
      format(Sys.time(), "%H:%M:%S"),
      final,
      if (is.null(g)) "NULL" else as.character(g),
      if (is.null(m)) "NULL" else as.character(m)
    ))
  }
  final
}

## ------------------------------------------------------------------------
## Output normalization (mode-aware; add unscale for regression)
## ------------------------------------------------------------------------
.as_pred_matrix <- function(pred, mode = NULL, meta = NULL,
                            DEBUG = get0("DEBUG_ASPM", inherits = TRUE, ifnotfound = FALSE)) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  stamp <- format(Sys.time(), "%H:%M:%S")
  if (isTRUE(DEBUG)) {
    cat(sprintf("[ASPM-DBG %s] entry: class=%s len=%s\n",
                stamp, paste0(class(pred), collapse=","), length(pred)))
    if (is.list(pred)) cat("[ASPM-DBG] list names: ", paste(names(pred), collapse=", "), "\n")
  }
  
  # --- unwrap list containers (outer and inner) ---
  if (is.list(pred) && "predicted_output" %in% names(pred)) {
    pred <- pred$predicted_output
  }
  if (is.list(pred) && length(pred) == 1L) {
    pred <- pred[[1L]]
  }
  
  # --- normalize types ---
  if (is.null(pred) || length(pred) == 0L) {
    if (isTRUE(DEBUG)) cat("[ASPM-DBG] empty → returning 0x1 matrix\n")
    return(matrix(numeric(0), nrow = 0, ncol = 1))
  }
  if (is.data.frame(pred)) pred <- as.matrix(pred)
  if (is.vector(pred))     pred <- matrix(as.numeric(pred), ncol = 1L)
  if (is.list(pred))       pred <- matrix(as.numeric(unlist(pred)), ncol = 1L)
  
  pred <- as.matrix(pred)
  storage.mode(pred) <- "double"
  
  # --- resolve mode ---
  resolved_mode <- tolower(
    mode %||%
      get0("CLASSIFICATION_MODE", inherits = TRUE, ifnotfound = NULL) %||%
      (tryCatch(meta$CLASSIFICATION_MODE, error = function(e) NULL)) %||%
      "regression"
  )
  
  if (isTRUE(DEBUG)) {
    cat(sprintf("[ASPM-DBG %s] mode=%s | BEFORE squash: %s | hash=%s | head=[%s]\n",
                stamp, resolved_mode, .summarize_matrix(pred), .hash_vec(pred), .peek_num(pred)))
    cat(sprintf("[ASPM-DBG] in[0,1]? %s\n", .in_range01(pred)))
  }
  assign("LAST_ASPM_IN", pred, envir = LAST_DEBUG)
  
  # --- apply mode-specific transforms ---
  if (identical(resolved_mode, "binary")) {
    if (!.in_range01(pred)) {
      if (isTRUE(DEBUG)) cat("[ASPM-DBG] applying sigmoid (binary)\n")
      pred <- 1 / (1 + exp(-pred))
    }
  } else if (identical(resolved_mode, "multiclass")) {
    if (NCOL(pred) > 1 && !.in_range01(pred)) {
      if (isTRUE(DEBUG)) cat("[ASPM-DBG] applying softmax (multiclass)\n")
      mx <- apply(pred, 1, max)
      ex <- exp(pred - mx)
      sm <- rowSums(ex)
      pred <- ex / sm
    }
  } else {
    if (isTRUE(DEBUG)) cat("[ASPM-DBG] regression mode: no squashing applied\n")
    tt <- try(.get_target_transform(meta), silent = TRUE)
    if (!inherits(tt, "try-error") && !is.null(tt)) {
      pred <- tryCatch(
        ._invert_target(pred, tt, DEBUG = isTRUE(DEBUG)),
        error = function(e) {
          if (isTRUE(DEBUG)) cat("[ASPM-DBG] invert skipped: ", conditionMessage(e), "\n")
          pred
        }
      )
    }
  }
  
  if (isTRUE(DEBUG)) {
    cat(sprintf("[ASPM-DBG %s] mode=%s | AFTER squash/unscale: %s | hash=%s | head=[%s]\n",
                stamp, resolved_mode, .summarize_matrix(pred), .hash_vec(pred), .peek_num(pred)))
  }
  assign("LAST_ASPM_OUT", pred, envir = LAST_DEBUG)
  pred
}


.is_linear_verbose <- function(af, PROBE = TRUE) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  .now <- function() format(Sys.time(), "%H:%M:%S")
  .attr_name <- function(f) tolower(trimws(attr(f, "name") %||% ""))
  .is_identity_fn <- function(f) {
    if (!is.function(f)) return(FALSE)
    nm <- .attr_name(f)
    if (nm %in% c("identity","linear","id")) return(TRUE)
    if (identical(f, base::identity)) return(TRUE)
    # try to match a global 'identity' if user defined one
    if (exists("identity", inherits = TRUE)) {
      g <- get("identity", inherits = TRUE)
      if (is.function(g) && identical(f, g)) return(TRUE)
    }
    if (!PROBE) return(FALSE)
    v <- c(-2,-1,0,1,2)
    ok <- tryCatch({
      out <- f(v)
      is.numeric(out) && length(out) == length(v) && max(abs(out - v)) <= 1e-12
    }, error = function(e) {
      cat(sprintf("[AF-TRACE %s] probe error: %s\n", .now(), conditionMessage(e)))
      FALSE
    })
    ok
  }
  
  cat(sprintf("\n[AF-TRACE %s] activation_functions candidate:\n", .now()))
  if (is.null(af)) { cat("  <NULL>\n"); return(FALSE) }
  if (!is.list(af)) af <- as.list(af)
  
  for (i in seq_along(af)) {
    ai     <- af[[i]]
    is_fun <- is.function(ai)
    nm     <- if (is_fun) .attr_name(ai) else NA_character_
    cls    <- paste0(class(ai), collapse = ",")
    # short deparse/description
    desc <- tryCatch({
      paste(utils::capture.output(str(ai, give.attr = TRUE, vec.len = 6L)), collapse = " ")
    }, error = function(e) paste0("<str error: ", conditionMessage(e), ">"))
    
    cat(sprintf("  [%02d] class=%s | is.function=%s | name=%s\n      desc=%s\n",
                i, cls, is_fun, if (is.na(nm) || nm == "") "<NULL>" else nm, desc))
  }
  
  last <- af[[length(af)]]
  last_is_linear <-
    (is.character(last) && tolower(trimws(last)) %in% c("linear","identity","id")) ||
    (is.function(last)  && .is_identity_fn(last))
  
  # extra diags for last
  if (is.function(last)) {
    nm  <- .attr_name(last)
    sig <- tryCatch(paste(deparse(last, nlines = 1L), collapse = ""), error = function(e) "<deparse error>")
    cat(sprintf("  [HEAD %s] name=%s | signature=%s | linear?=%s\n",
                .now(), if (nm=="") "<NULL>" else nm, sig, last_is_linear))
  } else {
    cat(sprintf("  [HEAD %s] last is %s | value=%s | linear?=%s\n",
                .now(), paste0(class(last), collapse=","), as.character(last)[1], last_is_linear))
  }
  
  cat(sprintf("  -> last_is_linear=%s\n\n", last_is_linear))
  last_is_linear
}





## ------------------------------------------------------------------------
## Safe wrapper
## ------------------------------------------------------------------------
# ===========================================
# .safe_run_predict (passes verbose/debug through)
# ===========================================
.safe_run_predict <- function(
    X, meta, model_index = 1L, ML_NN = TRUE, ...,
    verbose = get0("VERBOSE_SAFERUN", inherits = TRUE, ifnotfound = FALSE),
    debug   = get0("DEBUG_SAFERUN",   inherits = TRUE, ifnotfound = FALSE),
    DEBUG   = get0("DEBUG_SAFERUN",   inherits = TRUE, ifnotfound = FALSE)
) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  vrb <- isTRUE(verbose)
  dbg <- isTRUE(DEBUG) || isTRUE(debug)
  stamp <- format(Sys.time(), "%H:%M:%S")
  
  if (isTRUE(dbg)) {
    cat(sprintf("[SAFE-DBG %s] enter .safe_run_predict | X dims=%d x %d\n",
                stamp, NROW(X), NCOL(X)))
    cat(sprintf("[SAFE-DBG %s] X summary: mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                stamp, mean(X), sd(as.vector(X)), min(X), max(X)))
  }
  
  out <- tryCatch(
    .run_predict(
      X = X,
      meta = meta,
      model_index = model_index,
      ML_NN = ML_NN,
      ...,
      verbose = vrb,
      debug   = dbg
    ),
    error = function(e) {
      if (isTRUE(dbg)) message("[SAFE-DBG] .run_predict error: ", conditionMessage(e))
      list(predicted_output = matrix(numeric(0), nrow = 0, ncol = 1))
    }
  )
  
  if (isTRUE(dbg)) {
    raw <- out$predicted_output %||% out
    cat(sprintf("[SAFE-DBG %s] raw preds head=%s | mean=%.6f | sd=%.6f\n",
                stamp,
                paste(head(as.numeric(raw)), collapse=", "),
                mean(as.numeric(raw)), sd(as.numeric(raw))))
  }
  
  res <- .as_pred_matrix(out, mode = .get_mode(meta), meta = meta,
                         DEBUG = get0("DEBUG_ASPM", inherits = TRUE, ifnotfound = FALSE))
  if (isTRUE(dbg)) {
    cat(sprintf("[SAFE-DBG %s] ASPM result dims=%d x %d | mean=%.6f | sd=%.6f\n",
                stamp, nrow(res), ncol(res), mean(res), sd(as.vector(res))))
  }
  res
}




## ------------------------------------------------------------------------
## Predict shim (stateless, uses extract_best_records) — MODE-AWARE
## ------------------------------------------------------------------------
if (!exists(".run_predict", inherits = TRUE)) {
  .run_predict <- function(
    X, meta,
    model_index   = 1L,
    ML_NN         = TRUE,
    expected_mode = NULL,
    ...,
    verbose = get0("VERBOSE_RUNPRED", inherits = TRUE, ifnotfound = FALSE),
    debug   = get0("DEBUG_RUNPRED",   inherits = TRUE, ifnotfound = FALSE)
  ) {
    `%||%` <- function(x, y) if (is.null(x)) y else x
    near <- function(a, b, tol = 1e-12) all(is.finite(a)) && all(is.finite(b)) && max(abs(a - b)) <= tol
    
    if (is.null(meta)) stop(".run_predict: 'meta' is NULL")
    X <- as.matrix(X); storage.mode(X) <- "double"
    if (nrow(X) == 0) return(list(predicted_output = matrix(numeric(0), nrow = 0, ncol = 1)))
    
    vrb <- isTRUE(verbose)
    dbg <- isTRUE(debug)
    stamp <- format(Sys.time(), "%H:%M:%S")
    
    ## ---- Resolve expected mode ----
    if (is.null(expected_mode) || !nzchar(expected_mode)) {
      expected_mode <- tolower(get0("CLASSIFICATION_MODE", inherits = TRUE,
                                    ifnotfound = meta$CLASSIFICATION_MODE %||% "regression"))
    } else {
      expected_mode <- tolower(expected_mode)
    }
    if (!expected_mode %in% c("binary", "multiclass", "regression")) expected_mode <- "regression"
    if (dbg) cat(sprintf("[MODE-DBG %s] expected_mode='%s'\n", stamp, expected_mode))
    
    ## ---- Extract best weights/biases ----
    rec <- extract_best_records(meta, ML_NN = ML_NN, model_index = model_index)
    
    if (dbg) {
      wdims <- tryCatch(dim(rec$weights[[1]]), error = function(e) NULL)
      cat("[RUNPRED-DBG] have weights/biases: ",
          paste0(length(rec$weights), "W/", length(rec$biases), "b"),
          " | W1 dims=", if (is.null(wdims)) "NA" else paste(wdims, collapse = "x"), "\n", sep = "")
    }
    
    ## ---- Model config ----
    input_size   <- ncol(X)
    hidden_sizes <- meta$hidden_sizes %||% meta$model$hidden_sizes
    output_size  <- as.integer(meta$output_size %||% 1L)
    num_networks <- as.integer(meta$num_networks %||% length(meta$best_weights_record) %||% 1L)
    N            <- as.integer(meta$N %||% nrow(X))
    lambda       <- as.numeric(meta$lambda %||% 0)
    init_method  <- meta$init_method %||% "xavier"
    custom_scale <- meta$custom_scale %||% NULL
    
    if (dbg) cat("[RUNPRED-DBG] meta names:", paste(names(meta), collapse=","), "\n")
    activation_functions <- meta$activation_functions %||%
      (meta$model$activation_functions %||% (meta$preprocessScaledData$activation_functions %||% NULL))
    if (is.null(activation_functions)) {
      stop("[RUNPRED-ERR] activation_functions not found in meta.")
    }
    
    ML_NN <- isTRUE(meta$ML_NN) || isTRUE(ML_NN)
    
    # Last activation quick probe (debug-only)
    last_af <- tryCatch(activation_functions[[length(activation_functions)]], error = function(e) NULL)
    last_nm <- tolower(tryCatch(attr(last_af, "name"), error = function(e) NULL) %||% "")
    is_linear <- FALSE
    if (is.function(last_af)) {
      v <- c(-2,-1,0,1,2)
      outp <- try(last_af(v), silent = TRUE)
      if (!inherits(outp, "try-error") && is.numeric(outp) && length(outp) == length(v)) {
        is_linear <- near(as.numeric(outp), v, tol = 1e-12) || identical(last_af, base::identity) ||
          last_nm %in% c("identity","linear","id")
      }
    }
    if (dbg) {
      cat(sprintf("[HEAD-DBG %s] last_activation='%s' | last_is_linear=%s | mode=%s\n",
                  stamp, if (nzchar(last_nm)) last_nm else "<unknown>", is_linear, expected_mode))
      if (!is_linear && expected_mode != "regression") {
        cat("[ACT-DBG] Non-linear head is correct for classification — no regression flattening concern.\n")
      }
    }
    
    ## ---- Build a single-SONN wrapper and predict ----
    main_model <- DESONN$new(
      num_networks    = num_networks,
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
    sonn_idx  <- min(model_index, length(main_model$ensemble))
    model_obj <- main_model$ensemble[[sonn_idx]]
    
    call_args <- list(
      Rdata   = X,
      weights = rec$weights,
      biases  = rec$biases,
      activation_functions = activation_functions,
      verbose = vrb,
      debug   = dbg
    )
    
    # WARNING HANDLER: be silent unless dbg==TRUE
    out <- withCallingHandlers(
      do.call(model_obj$predict, call_args),
      warning = function(w) {
        msg <- conditionMessage(w)
        if (grepl("\\[ACT-DBG\\].*Last activation is NOT linear", msg)) {
          if (identical(expected_mode, "regression")) {
            # let it through for true regression
            return(invokeRestart("muffleWarning"))  # or comment this to show the original warning
          } else {
            if (dbg) message(sprintf("[ACT-DBG] Non-linear last activation during predict; expected for '%s'. Silencing.", expected_mode))
            invokeRestart("muffleWarning")
            return(invisible())
          }
        }
        # otherwise, do nothing and let other warnings behave normally
      }
    )
    
    ## ---- Normalize output ----
    pred <- out$predicted_output %||% out
    if (is.list(pred) && length(pred) == 1L && !is.matrix(pred)) pred <- pred[[1]]
    if (is.data.frame(pred)) pred <- as.matrix(pred)
    if (is.vector(pred))     pred <- matrix(as.numeric(pred), ncol = 1)
    pred <- as.matrix(pred); storage.mode(pred) <- "double"
    
    if (dbg) cat(sprintf("[RUNPRED-DBG %s] raw model out: %dx%d\n", stamp, nrow(pred), ncol(pred)))
    
    # optional capture hook (silent)
    try({
      env <- get("LAST_DEBUG", inherits = TRUE)
      assign("LAST_RUNPRED_OUT", pred, envir = env)
    }, silent = TRUE)
    
    list(predicted_output = pred)
  }
}



##helpers for
## if (!isTRUE(do_ensemble)) else {} in TestDESONN.R

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

##helpers for 
## =========================================================================================
## SINGLE-RUN MODE (no logs, no lineage, no temp/prune/add) — covers Scenario A & Scenario B
## =========================================================================================
## if (!isTRUE(do_ensemble)) in TestDESONN.R

`%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a

# Role helper: 0/1 => main, 2+ => temp
ensemble_role <- function(ensemble_number) {
  if (is.na(ensemble_number) || ensemble_number <= 1L) "main" else "temp"
}

# Attach a DESONN run into the top-level container in a consistent way
attach_run_to_container <- function(ensembles_container, desonn_run) {
  stopifnot(is.list(ensembles_container))
  role <- ensemble_role((desonn_run$ensemble_number %||% 0L))
  if (role == "main") {
    ensembles_container$main_ensemble <- ensembles_container$main_ensemble %||% list()
    # single-run: place at [[1]]; real ensembles can append
    if (length(ensembles_container$main_ensemble) == 0L) {
      ensembles_container$main_ensemble[[1]] <- desonn_run
    } else {
      ensembles_container$main_ensemble[[length(ensembles_container$main_ensemble) + 1L]] <- desonn_run
    }
  } else {
    ensembles_container$temp_ensemble <- ensembles_container$temp_ensemble %||% list()
    ensembles_container$temp_ensemble[[length(ensembles_container$temp_ensemble) + 1L]] <- desonn_run
  }
  ensembles_container
}

# Get models inside a DESONN run (SONN list) safely
get_models <- function(desonn_run) {
  x <- tryCatch(desonn_run$ensemble, error = function(...) NULL)
  if (is.list(x)) x else list()
}

# Pretty + explained summary (works for E=0 single-run and real ensembles)
print_ensembles_summary <- function(ensembles_container,
                                    explain = TRUE,
                                    show_models = TRUE,
                                    max_models = 5L) {
  `%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a
  role_of <- function(e) if (is.na(e) || e <= 1L) "main" else "temp"
  
  me <- ensembles_container$main_ensemble %||% list()
  te <- ensembles_container$temp_ensemble %||% list()
  
  cat("=== ENSEMBLES SUMMARY ===\n")
  
  if (isTRUE(explain)) {
    cat(
      "Legend:\n",
      "  • E = ensemble_number label (E=0 denotes single-run labeling).\n",
      "  • R lists are 1-based, so the single run lives at main_ensemble[[1]].\n",
      "  • Role: 'main' when E ∈ {0,1}; 'temp' when E ≥ 2.\n",
      "  • 'models' = SONN models inside a DESONN run.\n\n", sep = ""
    )
  }
  
  # ---- Main runs ----
  cat(sprintf("Main ensembles (runs): %d\n", length(me)))
  for (i in seq_along(me)) {
    run  <- me[[i]]
    e    <- as.integer(run$ensemble_number %||% (i - 1L))
    mods <- tryCatch(run$ensemble, error = function(...) NULL)
    if (!is.list(mods)) mods <- list()
    
    label <- if (e == 0L) "single-run" else "main"
    cat(sprintf("  E=%d (%s) at main_ensemble[[%d]]: %d model(s)\n", e, label, i, length(mods)))
    
    if (isTRUE(show_models) && length(mods)) {
      upto <- min(length(mods), as.integer(max_models))
      for (m in seq_len(upto)) {
        mdl <- mods[[m]]
        hs  <- tryCatch(mdl$hidden_sizes, error = function(...) NULL)
        nl  <- tryCatch(mdl$num_layers,    error = function(...) NULL)
        act <- tryCatch({
          aa <- mdl$activation_functions
          if (is.list(aa)) {
            vapply(aa, function(f) attr(f, "name") %||% "?", character(1))
          } else {
            NULL
          }
        }, error = function(...) NULL)
        
        cat(sprintf("    M=%d | layers=%s | hidden=%s",
                    m,
                    if (length(nl)) paste0(nl, collapse = ",") else "?",
                    if (length(hs)) paste0(hs, collapse = ",") else "?"))
        if (length(act)) cat(sprintf(" | activations=%s", paste0(act, collapse = ",")))
        cat("\n")
      }
      if (length(mods) > upto) cat(sprintf("    … (%d more models not shown)\n", length(mods) - upto))
    }
  }
  
  # ---- Temp runs ----
  cat(sprintf("Temp ensembles (runs): %d\n", length(te)))
  for (i in seq_along(te)) {
    run  <- te[[i]]
    e    <- as.integer(run$ensemble_number %||% (i + 1L))
    mods <- tryCatch(run$ensemble, error = function(...) NULL)
    if (!is.list(mods)) mods <- list()
    cat(sprintf("  E=%d (temp) at temp_ensemble[[%d]]: %d model(s)\n", e, i, length(mods)))
  }
  
  invisible(NULL)
}



## helper for prune and add
EMPTY_SLOT <- structure(list(.empty_slot = TRUE), class = "EMPTY_SLOT")
is_empty_slot <- function(x) is.list(x) && inherits(x, "EMPTY_SLOT")







tune_threshold_accuracy <- function(predicted_output, labels,
                                    metric = c("accuracy", "f1", "precision", "recall",
                                               "macro_f1", "macro_precision", "macro_recall"),
                                    threshold_grid = seq(0.05, 0.95, by = 0.01),
                                    verbose = FALSE) {
  grid <- threshold_grid
  metric <- match.arg(metric)
  
  # --- Sanitize 'grid' (avoid passing function/env/list/NULL and clamp to (0,1)) ---
  if (missing(grid) || is.null(grid) || is.function(grid) || is.environment(grid) || is.list(grid)) {
    grid <- seq(0.05, 0.95, by = 0.01)
  } else {
    grid <- tryCatch(as.numeric(unlist(grid, use.names = FALSE)), error = function(e) numeric(0))
    grid <- grid[is.finite(grid)]
    grid <- grid[grid > 0 & grid < 1]     # thresholds must be in (0,1)
    grid <- sort(unique(grid))
    if (length(grid) == 0L) grid <- seq(0.05, 0.95, by = 0.01)
  }
  
  # --- Coerce to numeric matrices and align columns ---
  P <- as.matrix(predicted_output); storage.mode(P) <- "double"
  L <- as.matrix(labels);           storage.mode(L) <- "double"
  
  nL <- ncol(L); nP <- ncol(P); K <- max(nL, nP)
  if (K < 1L) K <- 1L
  
  if (ncol(P) < K) {
    total_needed <- nrow(L) * K
    replicated <- rep(as.vector(P), length.out = total_needed)
    P <- matrix(replicated, nrow = nrow(L), ncol = K, byrow = FALSE)
  } else if (ncol(P) > K) {
    P <- P[, 1:K, drop = FALSE]
  }
  
  # =========================
  # Binary
  # =========================
  if (K == 1L) {
    y_true <- as.numeric(L[, 1])
    y_true01 <- as.integer(ifelse(y_true > 0, 1L, 0L))
    
    best_t <- NA_real_; best_val <- -Inf; best_pred <- NULL
    for (t in grid) {
      y_pred <- as.integer(P[, 1] >= t)
      TP <- sum(y_pred == 1 & y_true01 == 1)
      FP <- sum(y_pred == 1 & y_true01 == 0)
      FN <- sum(y_pred == 0 & y_true01 == 1)
      prec <- TP / (TP + FP + 1e-8)
      rec  <- TP / (TP + FN + 1e-8)
      
      val <- switch(metric,
                    accuracy  = mean(y_pred == y_true01),
                    f1        = 2 * prec * rec / (prec + rec + 1e-8),
                    precision = prec,
                    recall    = rec
      )
      if (val > best_val) { best_val <- val; best_t <- t; best_pred <- y_pred }
    }
    tuned_acc <- mean(best_pred == y_true01)
    if (verbose) cat(sprintf("[Binary] best_t=%.3f | tuned_%s=%.6f\n", best_t, metric, best_val))
    return(list(
      thresholds     = best_t,
      y_pred_class   = best_pred,
      tuned_score    = as.numeric(best_val),
      tuned_accuracy = as.numeric(tuned_acc),
      metric_used    = metric
    ))
  }
  
  # =========================
  # Multiclass
  # =========================
  if (ncol(L) > 1L) {
    y_true_ids <- max.col(L, ties.method = "first")
  } else {
    cls <- as.integer(L[, 1]); if (min(cls, na.rm = TRUE) == 0L) cls <- cls + 1L
    cls[cls < 1L] <- 1L
    cls[cls > K] <- K
    y_true_ids <- cls
  }
  
  thr <- numeric(K)
  for (k in seq_len(K)) {
    y_true01 <- as.integer(y_true_ids == k)
    best_tk <- NA_real_; best_valk <- -Inf
    for (t in grid) {
      y_pred01 <- as.integer(P[, k] >= t)
      TP <- sum(y_pred01 == 1 & y_true01 == 1)
      FP <- sum(y_pred01 == 1 & y_true01 == 0)
      FN <- sum(y_pred01 == 0 & y_true01 == 1)
      prec <- TP / (TP + FP + 1e-8)
      rec  <- TP / (TP + FN + 1e-8)
      
      val <- switch(metric,
                    macro_f1        = 2 * prec * rec / (prec + rec + 1e-8),
                    macro_precision = prec,
                    macro_recall    = rec,
                    accuracy        = mean(y_pred01 == y_true01)
      )
      if (val > best_valk) { best_valk <- val; best_tk <- t }
    }
    thr[k] <- best_tk
  }
  
  # Apply thresholds with masked argmax + fallback
  masked <- P
  for (k in seq_len(K)) masked[, k] <- ifelse(P[, k] >= thr[k], P[, k], -Inf)
  y_pred_ids <- max.col(masked, ties.method = "first")
  all_neg_inf <- !is.finite(apply(masked, 1, max))
  if (any(all_neg_inf)) {
    y_pred_ids[all_neg_inf] <- max.col(P[all_neg_inf, , drop = FALSE], ties.method = "first")
  }
  
  # Evaluate tuned metrics
  tuned_acc <- mean(y_pred_ids == y_true_ids)
  
  # If optimizing a macro metric, report its value from the confusion matrix
  tuned_score <- tuned_acc
  if (metric %in% c("macro_f1", "macro_precision", "macro_recall")) {
    tab <- table(factor(y_true_ids, levels = 1:K), factor(y_pred_ids, levels = 1:K))
    TPk <- diag(tab)
    FPk <- colSums(tab) - TPk
    FNk <- rowSums(tab) - TPk
    Prec_k <- ifelse((TPk + FPk) > 0, TPk / (TPk + FPk), 0)
    Rec_k  <- ifelse((TPk + FNk) > 0, TPk / (TPk + FNk), 0)
    if (metric == "macro_precision") tuned_score <- mean(Prec_k)
    if (metric == "macro_recall")    tuned_score <- mean(Rec_k)
    if (metric == "macro_f1") {
      F1_k <- ifelse((Prec_k + Rec_k) > 0, 2 * Prec_k * Rec_k / (Prec_k + Rec_k), 0)
      tuned_score <- mean(F1_k)
    }
  }
  
  if (verbose) {
    cat(sprintf("[Multiclass] tuned_acc=%.6f | metric=%s\n", tuned_acc, metric))
    cat(" thresholds: ", paste0(sprintf("%.3f", thr), collapse = ", "), "\n")
  }
  return(list(
    thresholds     = thr,
    y_pred_class   = y_pred_ids,
    tuned_score    = as.numeric(tuned_score),
    tuned_accuracy = as.numeric(tuned_acc),
    metric_used    = metric
  ))
}



evaluate_classification_metrics <- function(preds, labels) {
  labels <- as.vector(labels)
  preds <- as.vector(preds)
  
  TP <- sum(preds == 1 & labels == 1)
  FP <- sum(preds == 1 & labels == 0)
  FN <- sum(preds == 0 & labels == 1)
  
  precision <- TP / (TP + FP + 1e-8)
  recall <- TP / (TP + FN + 1e-8)
  F1 <- 2 * precision * recall / (precision + recall + 1e-8)
  
  return(list(precision = precision, recall = recall, F1 = F1))
}

##helpers for calculate_performance_grouped 

aggregate_predictions <- function(predicted_output_list,
                                  method = c("mean","median","vote"),
                                  weights = NULL) {
  method <- match.arg(method)
  mats <- lapply(predicted_output_list, function(x) { x <- as.matrix(x); storage.mode(x) <- "double"; x })
  
  N <- nrow(mats[[1]]); K <- ncol(mats[[1]]); M <- length(mats)
  stopifnot(all(vapply(mats, nrow, 1L) == N),
            all(vapply(mats, ncol, 1L) == K))
  
  if (is.null(weights)) weights <- rep(1/M, M) else {
    stopifnot(length(weights) == M); weights <- weights / sum(weights)
  }
  
  if (method == "median") {
    arr <- simplify2array(mats)        # N × K × M
    return(apply(arr, c(1,2), stats::median, na.rm = TRUE))
  }
  
  # mean / vote → elementwise weighted mean, keeps N × K
  X <- do.call(cbind, lapply(mats, function(m) as.vector(m)))  # (N*K) × M
  out <- as.numeric(X %*% weights)                             # (N*K)
  matrix(out, nrow = N, ncol = K)
}

# Full replacement (compact + multiclass-safe)
# Full replacement — compact, NA-safe, binary/multiclass
pick_representative_sonn <- function(SONN_list, predicted_output_list, labels) {
  to_class <- function(y) {
    if (is.matrix(y)) {
      # Argmax per row; keep NA if an entire row is NA
      apply(y, 1L, function(r) if (all(is.na(r))) NA_integer_ else which.max(r))
    } else if (is.factor(y)) {
      as.integer(y)
    } else if (is.character(y)) {
      as.integer(factor(y))
    } else {
      as.integer(y)
    }
  }
  
  f1_macro <- function(P, Y) {
    yt <- to_class(Y)
    yp <- if (is.matrix(P) && ncol(P) > 1L) {
      apply(P, 1L, function(r) if (all(is.na(r))) NA_integer_ else which.max(r))
    } else {
      p1 <- if (is.matrix(P)) P[, 1L, drop = TRUE] else P
      as.integer(as.numeric(p1) >= 0.5)
    }
    n <- min(length(yt), length(yp)); if (!n) return(0)
    yt <- yt[seq_len(n)]; yp <- yp[seq_len(n)]
    cls <- sort(unique(c(yt, yp))); if (!length(cls)) return(0)
    
    f1s <- sapply(cls, function(k) {
      tp <- sum(yp == k & yt == k, na.rm = TRUE)
      fp <- sum(yp == k & yt != k, na.rm = TRUE)
      fn <- sum(yp != k & yt == k, na.rm = TRUE)
      if (tp == 0 && fp == 0 && fn == 0) return(0)
      pr <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
      rc <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
      if ((pr + rc) == 0) 0 else 2 * pr * rc / (pr + rc)
    })
    mean(f1s, na.rm = TRUE)
  }
  
  if (!length(predicted_output_list)) return(SONN_list[[1L]])
  scores <- vapply(seq_along(predicted_output_list),
                   function(i) f1_macro(predicted_output_list[[i]], labels),
                   numeric(1))
  SONN_list[[which.max(scores)]]
}


# -------------------------------
# Helpers (keep top-level in file)
# -------------------------------
# ---- Safe, name-aligning flattener for per-model metrics ---------------------
# metric_list: list of length M; each element like list(metrics=<named list>, names=<char>)
# run_id:      character/list of length M with model labels
flatten_metrics_to_df <- function(metric_list, run_id) {
  if (is.null(metric_list) || length(metric_list) == 0) return(NULL)
  
  # Build one wide row per model
  rows <- lapply(seq_along(metric_list), function(i) {
    mi <- metric_list[[i]]
    if (is.null(mi) || is.null(mi$metrics)) return(NULL)
    
    # Flatten nested lists to atomic named vector
    flat <- tryCatch(
      unlist(mi$metrics, recursive = TRUE, use.names = TRUE),
      error = function(e) NULL
    )
    if (is.null(flat) || length(flat) == 0) return(NULL)
    
    # Coerce to one-row data.frame
    df <- as.data.frame(as.list(flat), stringsAsFactors = FALSE)
    
    # Ensure syntactically valid, unique names
    names(df) <- make.names(names(df), unique = TRUE)
    
    # Attach model label
    df$Model_Name <- if (!is.null(run_id) && length(run_id) >= i) run_id[[i]] else paste0("Model_", i)
    
    df
  })
  
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(NULL)
  
  # Align columns across all rows
  all_names <- unique(unlist(lapply(rows, names), use.names = FALSE))
  
  rows <- lapply(rows, function(df) {
    missing <- setdiff(all_names, names(df))
    if (length(missing)) {
      # fill missing metrics with NA (numeric); keep Model_Name as character
      for (nm in missing) df[[nm]] <- NA_real_
    }
    # Order columns consistently
    df <- df[all_names]
    df
  })
  
  # Bind safely (names now match); avoid base rbind name-matching issues
  wide <- do.call(rbind, rows)
  rownames(wide) <- NULL
  
  # Long tidy form (Model_Name + Metric + Value)
  metric_cols <- setdiff(names(wide), "Model_Name")
  
  # Use tidyr if available; otherwise base reshape
  if (requireNamespace("tidyr", quietly = TRUE)) {
    long <- tidyr::pivot_longer(
      wide,
      cols = dplyr::all_of(metric_cols),
      names_to = "Metric",
      values_to = "Value"
    )
  } else {
    # Base fallback
    long <- stats::reshape(
      wide,
      varying = metric_cols,
      v.names = "Value",
      timevar = "Metric",
      times = metric_cols,
      idvar = "Model_Name",
      direction = "long"
    )
    long <- long[ , c("Model_Name", "Metric", "Value")]
  }
  
  # Best-effort numeric coercion for Value
  suppressWarnings(long$Value <- as.numeric(long$Value))
  
  long
}

summarize_grouped <- function(long_df) {
  if (is.null(long_df) || !nrow(long_df)) return(NULL)
  
  keep <- with(long_df, tapply(Value, Metric, function(v) any(is.finite(v))))
  keep_metrics <- names(keep)[keep]
  df <- long_df[long_df$Metric %in% keep_metrics, c("Model_Name","Metric","Value"), drop = FALSE]
  if (!nrow(df)) return(NULL)
  
  stats_mat <- do.call(rbind, lapply(split(df$Value, df$Metric), function(v) {
    v <- v[is.finite(v)]
    c(mean = mean(v, na.rm = TRUE),
      median = stats::median(v, na.rm = TRUE),
      n = length(v))
  }))
  data.frame(Metric = rownames(stats_mat), stats_mat, row.names = NULL, check.names = FALSE)
}

##helpers for performance and relevance metric fun.

# Sanitize a numeric threshold grid
sanitize_grid <- function(grid, default = seq(0.05, 0.95, by = 0.01)) {
  if (is.null(grid) || is.function(grid) || is.environment(grid) || is.list(grid)) {
    return(default)
  }
  g <- suppressWarnings(as.numeric(grid))
  g <- unique(g[is.finite(g)])
  if (!length(g)) default else sort(g)
}

# Collapse label matrix to a 0/1 vector when possible
# - 1 col: >0 -> 1 else 0
# - 2 col: argmax -> {0,1}
# - >=3 col: return NULL (treat as multiclass upstream)
labels_to_binary_vec <- function(L) {
  if (is.null(ncol(L)) || ncol(L) == 1L) {
    return(ifelse(as.vector(L) > 0, 1L, 0L))
  }
  if (ncol(L) == 2L) {
    return(max.col(L, ties.method = "first") - 1L)  # 0/1
  }
  NULL
}

# Extract a single positive-class probability vector from predictions
# - 1 col: treat as P(pos)
# - 2 col: use column 2 as P(pos) (common {neg,pos})
# - >=3 col: return NULL (multiclass)
preds_to_pos_prob <- function(P) {
  if (is.null(ncol(P)) || ncol(P) == 1L) {
    p <- as.vector(P)
  } else if (ncol(P) == 2L) {
    p <- as.vector(P[, 2, drop = TRUE])
  } else {
    return(NULL)
  }
  p[!is.finite(p)] <- 0
  p <- pmin(pmax(p, 0), 1)
  p
}

# Decide if the task is binary based on labels (preferred) or predictions
infer_is_binary <- function(L, P) {
  Lb <- labels_to_binary_vec(L)
  if (!is.null(Lb)) {
    uniqL <- unique(na.omit(Lb))
    return(list(is_binary = length(uniqL) <= 2 && all(uniqL %in% c(0,1)), Lb = Lb))
  }
  # If labels inconclusive (>=3 cols), use predictions shape as hint
  list(is_binary = (!is.null(P) && ncol(P) <= 2L), Lb = NULL)
}

# Build one-hot matrix (N x K) from class ids (1..K)
one_hot_from_ids <- function(ids, K, N = NULL, strict = TRUE) {
  # rows = N or length(ids) by default
  if (is.null(N)) N <- length(ids)
  
  # initialize integer matrix
  M <- matrix(0L, nrow = N, ncol = K)
  
  # finiteness check on numeric-ish values (handles NA/NaN/Inf)
  ids_num <- suppressWarnings(as.numeric(ids))
  ok_finite <- is.finite(ids_num)
  
  # optional strict check: used ids must be whole numbers (e.g., 3.0 is fine, 3.2 is not)
  if (strict) {
    non_whole <- ok_finite & (abs(ids_num - round(ids_num)) > .Machine$double.eps^0.5)
    if (any(non_whole, na.rm = TRUE)) {
      stop("one_hot_from_ids: non-integer class ids detected among finite values.")
    }
  }
  
  # coerce to integer indices (after checks)
  ids_int <- suppressWarnings(as.integer(round(ids_num)))
  
  # valid positions: finite, in range [1, K]
  ok <- ok_finite & !is.na(ids_int) & ids_int >= 1L & ids_int <= K
  
  if (any(ok)) {
    M[cbind(seq_len(N)[ok], ids_int[ok])] <- 1L
  }
  M
}


# Safe call to a user-provided metrics helper (optional)
safe_eval_metrics <- function(y_pred_class, y_true01, verbose = FALSE) {
  out <- NULL
  try({
    out <- evaluate_classification_metrics(y_pred_class, y_true01)
  }, silent = !verbose)
  out
}


# =========================
# Metric helpers (general)
# =========================

.metric_minimize <- get0(".metric_minimize", ifnotfound = function(metric_name) {
  m <- tolower(metric_name)
  
  # metrics that should be minimized
  minimize_patterns <- "(loss|mse|mae|rmse|logloss|cross.?entropy|error|nll)"
  if (grepl(minimize_patterns, m)) return(TRUE)
  
  # explicit maximize overrides (always higher is better)
  maximize_set <- c("accuracy", "precision", "recall", "f1", "ndcg", "clustering_quality_db")
  if (m %in% maximize_set) return(FALSE)
  
  # default: maximize
  FALSE
})


# Robust metric fetcher: searches performance_* and relevance_* (incl. nested)
.get_metric_from_meta <- function(meta, metric_name) {
  if (is.null(meta)) return(NA_real_)
  # --- normalizer helpers ---
  .norm <- function(x) tolower(gsub("[^a-z0-9]+", "", trimws(as.character(x))))
  .is_num <- function(x) is.numeric(x) && length(x) == 1L && is.finite(x)
  
  # --- gather candidate maps (shallow + useful nested) ---
  maps <- list()
  if (!is.null(meta$performance_metric))    maps <- c(maps, list(performance_metric = meta$performance_metric))
  if (!is.null(meta$performance_metrics))   maps <- c(maps, list(performance_metrics = meta$performance_metrics))
  if (!is.null(meta$metrics))               maps <- c(maps, list(metrics = meta$metrics))
  if (!is.null(meta$relevance_metric))      maps <- c(maps, list(relevance_metric = meta$relevance_metric))
  if (!is.null(meta$relevance_metrics))     maps <- c(maps, list(relevance_metrics = meta$relevance_metrics))
  
  # specific nested commonly used in your structure
  acc_tuned <- tryCatch(meta$performance_metric$accuracy_tuned, error = function(e) NULL)
  if (!is.null(acc_tuned)) {
    maps <- c(maps, list(accuracy_tuned = acc_tuned))
    if (!is.null(acc_tuned$metrics)) maps <- c(maps, list(accuracy_tuned_metrics = acc_tuned$metrics))
  }
  
  if (!length(maps)) return(NA_real_)
  
  # --- flatten maps into a single name -> value registry (depth <= 2–3) ---
  key_norm   <- character()
  key_raw    <- character()
  val_store  <- list()
  
  .collect <- function(x, prefix = "") {
    if (is.list(x)) {
      for (nm in names(x)) {
        child <- x[[nm]]
        pfx <- if (nzchar(prefix)) paste0(prefix, ".", nm) else nm
        if (is.list(child)) {
          # one more level
          .collect(child, pfx)
        } else {
          # atomic leaf
          key_norm <<- c(key_norm, .norm(nm))
          key_raw  <<- c(key_raw,  pfx)
          val_store[[length(val_store) + 1L]] <<- child
        }
      }
    } else {
      # unnamed atomic (rare)
      key_norm <<- c(key_norm, .norm(prefix))
      key_raw  <<- c(key_raw, prefix)
      val_store[[length(val_store) + 1L]] <<- x
    }
  }
  
  for (m in maps) .collect(m)
  
  # coerce to numeric where possible
  vals_num <- suppressWarnings(as.numeric(unlist(val_store, use.names = FALSE)))
  # keep only those that are actually numeric scalars
  ok_num <- is.finite(vals_num)
  key_norm <- key_norm[ok_num]
  key_raw  <- key_raw[ok_num]
  vals_num <- vals_num[ok_num]
  if (!length(vals_num)) return(NA_real_)
  
  # --- aliasing for common names / variants ---
  req <- .norm(metric_name)
  alias <- list(
    "accuracy"  = c("accuracy", "accuracypercent", "acc"),
    "precision" = c("precision", "prec"),
    "recall"    = c("recall", "tpr", "sensitivity"),
    "f1"        = c("f1", "f1score", "f1_macro", "macrof1", "f1macro"),
    "macro_f1"  = c("macrof1", "f1macro", "f1_macro"),
    "micro_f1"  = c("microf1", "f1micro", "f1_micro"),
    "ndcg"      = c("ndcg", "ndcg@5", "ndcg@10", "ndcg@k"),
    "mse"       = c("mse"),
    "mae"       = c("mae"),
    "rmse"      = c("rmse"),
    "top1"      = c("top1", "top_1", "top-1")
  )
  cand <- unique(c(alias[[req]] %||% character(0), req))
  
  # 1) exact (normalized) match
  hit <- which(key_norm %in% cand)
  # 2) fallback: contains match
  if (!length(hit)) {
    hit <- unlist(lapply(cand, function(k) which(grepl(k, key_norm, fixed = TRUE))))
    hit <- unique(hit)
  }
  if (!length(hit)) return(NA_real_)
  
  # choose first finite numeric
  vals <- vals_num[hit]
  idx  <- which(is.finite(vals))[1L]
  if (is.na(idx)) return(NA_real_)
  vals[idx]
}


# utils.R
# -------------------------------------------------------
# Best-model finder by TARGET_METRIC (general, any kind)
# Depends on: bm_list_all(), bm_select_exact(), .metric_minimize(), .get_metric_from_meta()
# -------------------------------------------------------
find_best_model <- function(target_metric_name_best,
                            kind_filter  = c("Main","Temp"),
                            ens_filter   = NULL,
                            model_filter = NULL,
                            dir = .BM_DIR) {
  minimize <- .metric_minimize(target_metric_name_best)
  
  df <- bm_list_all(dir)
  if (!nrow(df)) {
    cat("\n==== FIND_BEST_MODEL ====\nNo candidates in env/RDS.\n")
    return(list(best_row=NULL, meta=NULL, tbl=data.frame(), minimize=minimize))
  }
  
  if (length(kind_filter))    df <- df[df$kind  %in% kind_filter, , drop = FALSE]
  if (!is.null(ens_filter))   df <- df[df$ens   %in% ens_filter,  , drop = FALSE]
  if (!is.null(model_filter)) df <- df[df$model %in% model_filter,, drop = FALSE]
  if (!nrow(df)) {
    cat("\n==== FIND_BEST_MODEL ====\nNo candidates after filters.\n")
    return(list(best_row=NULL, meta=NULL, tbl=df, minimize=minimize))
  }
  
  df$metric_value <- vapply(seq_len(nrow(df)), function(i) {
    meta_i <- tryCatch(bm_select_exact(df$kind[i], df$ens[i], df$model[i], dir = dir), error = function(e) NULL)
    if (is.null(meta_i)) return(NA_real_)
    .get_metric_from_meta(meta_i, target_metric_name_best)
  }, numeric(1))
  
  cat("\n==== FIND_BEST_MODEL (ANY KIND) ====\n")
  ok <- is.finite(df$metric_value)
  if (!any(ok)) {
    print(df[, c("name","kind","ens","model","source")], row.names = FALSE)
    cat("All metric values are NA/Inf for", target_metric_name_best, "\n")
    return(list(best_row=NULL, meta=NULL, tbl=df, minimize=minimize))
  }
  
  df_ok <- df[ok, , drop = FALSE]
  ord   <- if (minimize) order(df_ok$metric_value) else order(df_ok$metric_value, decreasing = TRUE)
  df_ok <- df_ok[ord, , drop = FALSE]
  
  top  <- df_ok[1, , drop = FALSE]
  meta <- bm_select_exact(top$kind, top$ens, top$model, dir = dir)
  
  print(head(df_ok[, c("name","kind","ens","model","metric_value")], 10), row.names = FALSE)
  cat(sprintf("→ Selected: %s | %s=%.6f (%s better)\n",
              top$name, target_metric_name_best, top$metric_value,
              if (minimize) "lower" else "higher"))
  
  list(best_row = top, meta = meta, tbl = df_ok, minimize = minimize)
}

#used to calc error in predict() in legacy, but now use helper and used in train()

# if (!is.null(predicted_outputAndTime$predicted_output_l2)) {
#   
#   all_predicted_outputs[[i]]       <- predicted_outputAndTime$predicted_output_l2$predicted_output
#   all_prediction_times[[i]]        <- predicted_outputAndTime$train_reg_prediction_time
#   all_errors[[i]]                  <- compute_error(predicted_outputAndTime$predicted_output_l2$predicted_output, y, CLASSIFICATION_MODE)


compute_error <- function(
    predicted_output,
    labels,
    CLASSIFICATION_MODE = c("binary","multiclass","regression")
) {
  CLASSIFICATION_MODE <- match.arg(CLASSIFICATION_MODE)
  if (is.null(labels)) return(NULL)
  
  # --- normalize predictions to a matrix ---
  P <- as.matrix(predicted_output)
  n <- nrow(P); k <- ncol(P)
  
  # --- helpers ---
  as_mat <- function(x) {
    if (is.data.frame(x)) return(as.matrix(x))
    if (is.vector(x))     return(matrix(x, ncol = 1L))
    as.matrix(x)
  }
  align_rows <- function(M, rows) {
    if (nrow(M) == rows) return(M)
    if (nrow(M) > rows)  return(M[seq_len(rows), , drop = FALSE])
    pad <- matrix(rep(M[nrow(M), , drop = FALSE], length.out = (rows - nrow(M)) * ncol(M)),
                  nrow = rows - nrow(M), byrow = TRUE)
    rbind(M, pad)
  }
  pad_or_trim_cols <- function(M, cols, pad = 0) {
    if (ncol(M) == cols) return(M)
    if (ncol(M) > cols)  return(M[, seq_len(cols), drop = FALSE])
    cbind(M, matrix(pad, nrow = nrow(M), ncol = cols - ncol(M)))
  }
  
  # =========================
  # BINARY CLASSIFICATION
  # =========================
  if (CLASSIFICATION_MODE == "binary") {
    pred <- if (k == 1L) P else matrix(P[, k], ncol = 1L)  # take last col if logits/probs provided
    L <- labels
    
    if (is.factor(L)) {
      L <- as.integer(L) - 1L
    } else if (is.character(L)) {
      lu <- sort(unique(L))
      L  <- as.integer(factor(L, levels = lu)) - 1L
    } else {
      L <- suppressWarnings(as.numeric(L))
      if (all(is.na(L))) {
        lu <- sort(unique(as.character(labels)))
        L  <- as.integer(factor(as.character(labels), levels = lu)) - 1L
      }
    }
    
    L <- as_mat(L)
    if (ncol(L) > 1L) L <- matrix(L[, ncol(L)], ncol = 1L)
    L <- align_rows(L, n)
    L <- pad_or_trim_cols(L, 1L, pad = 0)
    
    return(abs(L - pred))
  }
  
  # =========================
  # MULTICLASS CLASSIFICATION
  # =========================
  if (CLASSIFICATION_MODE == "multiclass") {
    L <- labels
    if (is.data.frame(L)) L <- as.matrix(L)
    
    # If labels are already one-hot / probability matrix → align & diff
    if (!is.null(dim(L)) && ncol(L) > 1L) {
      Y <- align_rows(as.matrix(L), n)
      Y <- pad_or_trim_cols(Y, k, pad = 0)
      return(abs(Y - P))
    }
    
    # Otherwise, treat labels as class IDs / names
    class_order <- if (!is.null(colnames(P))) colnames(P) else sort(unique(as.character(L)))
    
    if (is.factor(L)) {
      L_idx <- as.integer(factor(L, levels = class_order))
    } else if (is.character(L)) {
      L_idx <- as.integer(factor(L, levels = class_order))
    } else {
      L_num <- suppressWarnings(as.numeric(L))
      if (!all(is.na(L_num))) {
        if (min(L_num, na.rm = TRUE) == 0 && max(L_num, na.rm = TRUE) <= (k - 1)) {
          L_idx <- as.integer(L_num + 1L)
        } else {
          L_idx <- as.integer(L_num)
        }
      } else {
        L_idx <- as.integer(factor(as.character(L), levels = class_order))
      }
    }
    
    if (anyNA(L_idx)) L_idx[is.na(L_idx)] <- 1L
    L_idx <- rep(L_idx, length.out = n)
    
    Y <- matrix(0, nrow = n, ncol = k)
    Y[cbind(seq_len(n), pmax(1L, pmin(k, L_idx)))] <- 1L
    
    return(abs(Y - P))
  }
  
  # =========================
  # REGRESSION
  # =========================
  # Goal: return absolute error |Y - P| with shape n x k.
  # - Accepts labels as vector/matrix/data.frame.
  # - Coerces to numeric, aligns rows, and matches columns:
  #   * If labels have 1 col and k > 1 → replicate that column across k.
  #   * If labels have >1 col → trim or pad with zeros to match k.
  L <- labels
  
  # Coerce to numeric matrix safely
  to_numeric_matrix <- function(x) {
    M <- as_mat(x)
    # Try numeric coercion while preserving shape
    suppressWarnings(storage.mode(M) <- "double")
    # If still all NA (e.g., character), try a character→numeric pass
    if (all(is.na(M))) {
      M_chr <- as_mat(as.character(x))
      suppressWarnings(storage.mode(M_chr) <- "double")
      M <- M_chr
    }
    M
  }
  
  Y <- to_numeric_matrix(L)
  if (is.null(dim(Y))) Y <- matrix(Y, ncol = 1L)
  
  # Align rows
  Y <- align_rows(Y, n)
  
  # Align columns to predictions
  if (ncol(Y) == 1L && k > 1L) {
    Y <- matrix(rep(Y[, 1L], times = k), nrow = n, ncol = k)
  } else {
    Y <- pad_or_trim_cols(Y, k, pad = 0)
  }
  
  return(abs(Y - P))
}

optimizers_log_update <- function(
    optimizer, epoch, layer, target,
    grads_matrix,            # gradient(s) passed in
    P_before = NULL,         # param before (weights or biases)
    P_after  = NULL,         # param after  (weights or biases)
    update_applied = NULL,   # the actual update step applied (same shape as P)
    verbose = verbose
) {
  
  # ---- helpers ----
  .as_num <- function(x) {
    if (is.list(x)) x <- x[[1]]
    if (is.null(x)) return(numeric())
    as.numeric(x)
  }
  .stats <- function(x) {
    v <- suppressWarnings(.as_num(x))
    if (!length(v) || all(!is.finite(v))) return("min=NA mean=NA max=NA")
    sprintf("min=%.3g mean=%.3g max=%.3g", min(v, na.rm = TRUE),
            mean(v, na.rm = TRUE), max(v, na.rm = TRUE))
  }
  .shape <- function(x) {
    if (is.list(x)) x <- x[[1]]
    if (is.null(x)) return("∅")
    d <- dim(x)
    if (is.null(d)) sprintf("len=%d", length(x)) else sprintf("%dx%d", d[1], d[2])
  }
  
  # ---- build message ----
  grads_msg  <- sprintf("grads(%s):[%s]",  .shape(grads_matrix),  .stats(grads_matrix))
  pre_msg    <- if (!is.null(P_before)) sprintf(" | %s_pre(%s):[%s]",  target, .shape(P_before), .stats(P_before)) else ""
  post_msg   <- if (!is.null(P_after))  sprintf(" | %s_post(%s):[%s]", target, .shape(P_after),  .stats(P_after))  else ""
  update_msg <- if (!is.null(update_applied)) sprintf(" | update(%s):[%s]", .shape(update_applied), .stats(update_applied)) else ""
  
  cat(sprintf("[OPT=%s][E%d][L%d][%s] %s%s%s%s\n",
              toupper(optimizer), epoch, layer, target,
              grads_msg, pre_msg, post_msg, update_msg))
  
}


#################################################################################################
# MAIN TEST PREDICT-ONLY FUNCTION
#################################################################################################

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
    AGG_METRICS_FILE     = NULL,                      # if provided, append all seeds into one metrics RDS table here
    MODEL_SLOT           = 1L                         # <<< NEW: which MAIN slot we're evaluating
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
    if (mode == "regression" && ncol(P) > 1L) { P <- P[,1,drop=FALSE] }
    P
  }
  
  # ---------------- config + env bits ----------------
  CLASSIFICATION_MODE <- tolower(CLASSIFICATION_MODE)
  stopifnot(CLASSIFICATION_MODE %in% c("binary","multiclass","regression"))
  CLASS_THRESHOLD <- as.numeric(get0("CLASS_THRESHOLD", inherits=TRUE, ifnotfound=0.5))
  SONN           <- get0("SONN",     inherits=TRUE, ifnotfound=NULL)
  verbose_flag   <- isTRUE(get0("verbose", inherits=TRUE, ifnotfound=FALSE))
  
  cat("[CFG] SPLIT=", INPUT_SPLIT, " | CLASS_MODE=", CLASSIFICATION_MODE,
      " | RUN_INDEX=", RUN_INDEX, " | SEED=", SEED, " | SLOT=", MODEL_SLOT,
      " | OUT=", OUTPUT_DIR, "\n", sep="")
  
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
  # tag the slot for aggregate metrics
  row_df$model_slot <- as.integer(MODEL_SLOT)
  
  # ---------------- compact summary (results_df) ----------------
  pred_hash <- tryCatch(digest_safe(round(P[seq_len(min(nrow(P), 2000)), , drop=FALSE], 6)), error=function(e) NA_character_)
  results_df <- data.frame(
    kind = if (LOAD_FROM_RDS) "RDS" else "ENV",
    ens  = 0L,
    model = as.integer(MODEL_SLOT),                 # <<< reflect the slot
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
        meta_source    = "mixed",                     # <<< multiple slots likely
        entries        = list(),                      # per-seed/slot entries
        seeds          = integer(0)
      )
    } else {
      # ensure meta_source doesn't mislead when mixing slots
      if (is.null(pack$meta_source) || identical(pack$meta_source, "")) pack$meta_source <- "mixed"
    }
    key <- sprintf("seed_%s_run_%03d_model_%02d",
                   as.character(SEED), as.integer(RUN_INDEX), as.integer(MODEL_SLOT))
    entry <- list(
      run_index   = as.integer(RUN_INDEX),
      seed        = as.integer(SEED),
      model_slot  = as.integer(MODEL_SLOT),
      meta_var    = as.character(ENV_META_NAME),     # <<< which ENV var was used
      results_df  = results_df,
      prediction_sig = pred_hash
    )
    if (isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) {
      # name by ENV var for debuggability
      entry$predictions <- setNames(list(P), ENV_META_NAME)
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
      prediction_sigs = results_df$pred_sig,
      model_slot      = as.integer(MODEL_SLOT)
    )
    if (isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) {
      predict_pack$predictions <- setNames(list(P), ENV_META_NAME)
    }
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
      cat("[SAVE] metrics     → ", metrics_path, " | rows=", nrow(row_df), " cols=", ncol(row_df), "\n")
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
