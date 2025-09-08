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

.apply_scaling_if_any <- function(Xi, meta) {
  center <- try(meta$preprocess$center, silent = TRUE)
  scale_ <- try(meta$preprocess$scale,  silent = TRUE)
  if (!inherits(center, "try-error") && !is.null(center) &&
      !inherits(scale_,  "try-error") && !is.null(scale_)) {
    if (!is.null(names(center)) && !is.null(colnames(Xi))) {
      common <- intersect(names(center), colnames(Xi))
      if (length(common)) {
        Xi[, common] <- sweep(Xi[, common, drop = FALSE], 2, center[common], "-")
        Xi[, common] <- sweep(Xi[, common, drop = FALSE], 2, scale_[common],  "/")
      }
    } else {
      Xi <- scale(Xi, center = center, scale = scale_)
    }
  }
  Xi
}

.as_pred_matrix <- function(pred) {
  if (is.list(pred) && !is.null(pred$predicted_output)) pred <- pred$predicted_output
  if (is.list(pred) && length(pred) == 1L && !is.matrix(pred)) pred <- pred[[1]]
  if (is.null(pred) || length(pred) == 0) return(matrix(numeric(0), nrow = 0, ncol = 1))
  if (is.data.frame(pred)) pred <- as.matrix(pred)
  if (is.vector(pred)) pred <- matrix(as.numeric(pred), ncol = 1)
  pred <- as.matrix(pred); storage.mode(pred) <- "double"; pred
}

.safe_run_predict <- function(X, meta, model_index = 1L, ML_NN = TRUE, ...) {
  tryCatch(
    .run_predict(X = X, meta = meta, model_index = model_index, ML_NN = ML_NN, ...),
    error = function(e) { 
      message(sprintf("  ! predict failed for %s: %s",
                      as.character(meta$model_serial_num %||% "unknown"), conditionMessage(e)))
      NULL
    }
  )
}

## PREDICT SHIM: .run_predict
## Only defined if your pipeline hasn’t provided it.
if (!exists(".run_predict", inherits = TRUE)) {
  .run_predict <- function(X, meta, model_index = 1L, ML_NN = TRUE, ...) {
    if (is.null(meta)) stop(".run_predict: 'meta' is NULL")
    X <- as.matrix(X); storage.mode(X) <- "double"
    if (nrow(X) == 0) return(list(predicted_output = matrix(numeric(0), nrow = 0, ncol = 1)))
    
    # Pull best records (weights/biases)
    rec <- extract_best_records(meta, ML_NN = ML_NN, model_index = model_index)
    
    # Infer sizes
    input_size   <- ncol(X)
    hidden_sizes <- meta$hidden_sizes %||% meta$model$hidden_sizes %||% c(32L, 16L)
    output_size  <- as.integer(meta$output_size %||% 1L)
    num_networks <- max(1L,
                        as.integer(meta$num_networks %||% length(meta$best_weights_record) %||% 1L),
                        model_index)
    
    N            <- as.integer(meta$N        %||% nrow(X))
    lambda       <- as.numeric(meta$lambda   %||% 0)
    init_method  <- meta$init_method   %||% "xavier"
    custom_scale <- meta$custom_scale  %||% NULL
    activation_functions <- get0("activation_functions",
                                 ifnotfound = meta$activation_functions %||% NULL,
                                 inherits = TRUE)
    ML_NN <- isTRUE(meta$ML_NN) || isTRUE(ML_NN)
    
    # Build a minimal DESONN and select SONN
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
    
    # Stateless prediction (pass weights/biases directly)
    out <- model_obj$predict(
      Rdata   = X,
      weights = rec$weights,
      biases  = rec$biases,
      activation_functions = activation_functions
    )
    
    # Normalize return
    pred <- out$predicted_output %||% out
    if (is.list(pred) && length(pred) == 1L && !is.matrix(pred)) pred <- pred[[1]]
    if (is.data.frame(pred)) pred <- as.matrix(pred)
    if (is.vector(pred))     pred <- matrix(as.numeric(pred), ncol = 1)
    pred <- as.matrix(pred); storage.mode(pred) <- "double"
    
    # If values look like logits, squash
    if (any(pred < 0, na.rm = TRUE) || any(pred > 1, na.rm = TRUE)) {
      pred <- plogis(pred)
    }
    
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

aggregate_predictions <- function(predicted_output_list, method = c("mean","median","vote"), weights = NULL) {
  stopifnot(length(predicted_output_list) >= 1)
  method <- match.arg(method)
  P <- do.call(cbind, lapply(predicted_output_list, as.numeric))
  if (!is.null(weights)) {
    stopifnot(length(weights) == ncol(P))
    w <- weights / sum(weights)
  } else {
    w <- rep(1/ncol(P), ncol(P))
  }
  if (method == "mean")   as.matrix(drop(P %*% w))
  else if (method == "median") as.matrix(apply(P, 1, stats::median))
  else { # vote → probability = mean; you can threshold later if you want hard labels
    as.matrix(rowMeans(P))
  }
}

pick_representative_sonn <- function(SONN_list, predicted_output_list, labels) {
  # Pick the model with best F1 (at 0.5) on the provided labels
  f1_at_05 <- function(p, y) {
    yhat <- as.integer(p >= 0.5)
    tp <- sum(yhat==1 & y==1); fp <- sum(yhat==1 & y==0); fn <- sum(yhat==0 & y==1)
    prec <- if ((tp+fp)==0) 0 else tp/(tp+fp)
    rec  <- if ((tp+fn)==0) 0 else tp/(tp+fn)
    if ((prec+rec)==0) 0 else 2*prec*rec/(prec+rec)
  }
  scores <- vapply(seq_along(predicted_output_list),
                   function(i) f1_at_05(predicted_output_list[[i]], labels),
                   numeric(1))
  SONN_list[[ which.max(scores) ]]
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
one_hot_from_ids <- function(ids, K, N = NULL) {
  if (is.null(N)) N <- length(ids)
  M <- matrix(0, nrow = N, ncol = K)
  ok <- ids >= 1 & ids <= K & is.finite(ids)
  if (any(ok)) M[cbind(seq_len(N)[ok], ids[ok])] <- 1L
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