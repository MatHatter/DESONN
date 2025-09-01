# utils/bootstrap_metadata.R
# ---------------------------------------------------------------------
# Purpose:
# - Snapshot/load/auto-find DESONN metadata objects named like:
#     Ensemble_<Kind>_<ens>_model_<idx>_(metadata|metada)
# - Provide helpers to list/select models (env + RDS)
# - Provide a hard-stop helper for one-shot “prepare disk only” runs
# - Provide metric-driven selection + predict helpers (NO find_best_model here)
# ---------------------------------------------------------------------

# ===== small utils ====================================================
`%||%` <- get0("%||%", ifnotfound = function(a,b) if (is.null(a) || !length(a)) b else a)

# ===== hard stop helper (always defined here) =========================
.hard_stop <- function(msg = "[prepare_disk_only] Done; stopping script.") {
  cat(msg, "\n")
  if (interactive()) {
    stop(msg, call. = FALSE)
  } else {
    quit(save = "no", status = 0, runLast = FALSE)
  }
}

# ===== config =========================================================
.BM_DIR <- "artifacts"
if (!dir.exists(.BM_DIR)) dir.create(.BM_DIR, recursive = TRUE, showWarnings = FALSE)

# Regex to match metadata symbols in .GlobalEnv / filenames
.BM_PAT <- "^Ensemble_([^_]+)_(\\d+)_model_(\\d+)_(?:metadata|metada)$"

# ===== core helpers: list env / rds ==================================
# Never overwrite: if the RDS already exists, just return its path.
bm_save_copy <- function(obj_name, dir = .BM_DIR, compress = TRUE) {
  if (!exists(obj_name, envir = .GlobalEnv, inherits = FALSE)) {
    stop(sprintf("bm_save_copy(): '%s' not found in .GlobalEnv.", obj_name))
  }
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  
  path <- file.path(dir, paste0(obj_name, ".rds"))
  if (file.exists(path)) {
    return(invisible(path))  # do NOT overwrite
  }
  saveRDS(get(obj_name, envir = .GlobalEnv), path, compress = compress)
  invisible(path)
}

bm_list_env <- function() {
  nm <- ls(envir = .GlobalEnv, all.names = FALSE)
  nm <- nm[grepl(.BM_PAT, nm)]
  if (!length(nm)) {
    return(data.frame(name = character(), kind = character(), ens = integer(), model = integer()))
  }
  m <- regexec(.BM_PAT, nm); parts <- regmatches(nm, m)
  do.call(rbind, lapply(parts, function(p) data.frame(
    name  = p[1], kind = p[2],
    ens   = as.integer(p[3]), model = as.integer(p[4]),
    stringsAsFactors = FALSE
  )))
}

bm_list_rds <- function(dir = .BM_DIR) {
  f <- list.files(dir, pattern = "^Ensemble_.*_model_\\d+_(?:metadata|metada)\\.rds$", full.names = TRUE)
  if (!length(f)) {
    return(data.frame(path = character(), name = character(), kind = character(),
                      ens = integer(), model = integer(), mtime = as.POSIXct(character())))
  }
  nm <- sub("\\.rds$", "", basename(f))
  m  <- regexec(.BM_PAT, nm); parts <- regmatches(nm, m)
  do.call(rbind, lapply(seq_along(parts), function(i) data.frame(
    path  = f[i], name  = nm[i],
    kind  = parts[[i]][2],
    ens   = as.integer(parts[[i]][3]),
    model = as.integer(parts[[i]][4]),
    mtime = file.info(f[i])$mtime,
    stringsAsFactors = FALSE
  )))
}

# One-line union: show everything you have (env + rds)
bm_list_all <- function(dir = .BM_DIR) {
  env <- bm_list_env(); if (nrow(env)) env$source <- "env"
  rds <- bm_list_rds(dir); if (nrow(rds)) rds$source <- "rds"
  if (nrow(rds)) rds <- rds[, c("name","kind","ens","model","source")]
  out <- rbind(env, rds)
  unique(out[order(out$kind, out$ens, out$model), ], incomparables = FALSE)
}

bm_print_all <- function(dir = .BM_DIR) {
  cat("Discovered models (kind / ens / model):\n")
  df <- bm_list_all(dir)
  if (!nrow(df)) { cat("  [none]\n"); return(invisible(df)) }
  apply(df, 1, function(r) {
    cat(sprintf("  %-40s  kind=%-5s  ens=%d  model=%d  (%s)\n",
                r["name"], r["kind"], as.integer(r["ens"]), as.integer(r["model"]), r["source"]))
  })
  invisible(df)
}

# ===== preference scoring (kept for compatibility paths) =============
bm_pick <- function(candidates, prefer_kind = c("Main","Temp"), prefer_ens = NULL, prefer_model = NULL) {
  if (!nrow(candidates)) return(NULL)
  candidates$score <- 0
  if (length(prefer_kind)) {
    kscore <- match(tolower(candidates$kind), tolower(prefer_kind))
    kscore[is.na(kscore)] <- length(prefer_kind) + 1L
    candidates$score <- candidates$score - kscore
  }
  if (length(prefer_ens)) {
    escore <- match(candidates$ens, prefer_ens)
    escore[is.na(escore)] <- length(prefer_ens) + 1L
    candidates$score <- candidates$score - escore
  }
  if (length(prefer_model)) {
    mscore <- match(candidates$model, prefer_model)
    mscore[is.na(mscore)] <- length(prefer_model) + 1L
    candidates$score <- candidates$score - mscore
  }
  candidates[order(candidates$score, decreasing = TRUE), ][1, , drop = FALSE]
}

# ===== ensure one metadata object in env (prefer env; else load RDS) ==
bm_ensure_loaded <- function(
    name_hint    = NULL,
    prefer_kind  = c("Main","Temp"),
    prefer_ens   = c(0L, 1L),
    prefer_model = NULL,     # NULL removes bias toward model_1
    dir = .BM_DIR
) {
  if (!is.null(name_hint) && nzchar(name_hint)) {
    if (exists(name_hint, envir = .GlobalEnv, inherits = FALSE)) return(get(name_hint, envir = .GlobalEnv))
    rpath <- file.path(dir, paste0(name_hint, ".rds"))
    if (file.exists(rpath)) {
      obj <- readRDS(rpath)
      assign(name_hint, obj, envir = .GlobalEnv)
      return(obj)
    }
    stop(sprintf("bm_ensure_loaded(): '%s' not in env and '%s' not found.", name_hint, rpath))
  }
  env_df <- bm_list_env()
  pick   <- bm_pick(env_df, prefer_kind, prefer_ens, prefer_model)
  if (!is.null(pick)) return(get(pick$name, envir = .GlobalEnv))
  
  rds_df <- bm_list_rds(dir)
  if (!nrow(rds_df)) stop("bm_ensure_loaded(): no metadata object in env or RDS.")
  rds_df <- rds_df[order(rds_df$mtime, decreasing = TRUE), ]
  pick   <- bm_pick(rds_df, prefer_kind, prefer_ens, prefer_model)
  obj    <- readRDS(pick$path)
  assign(pick$name, obj, envir = .GlobalEnv)
  obj
}

bm_clear_env_except <- function(keep_names) {
  all_objs <- ls(envir = .GlobalEnv, all.names = FALSE)
  to_rm    <- setdiff(all_objs, keep_names)
  if (length(to_rm)) rm(list = to_rm, envir = .GlobalEnv)
  invisible(TRUE)
}

# ===== PHASE 1: snapshot a single pick to RDS and wipe env ============
bm_prepare_disk_only <- function(
    name_hint    = NULL,
    prefer_kind  = c("Main","Temp"),
    prefer_ens   = c(0L, 1L),
    prefer_model = NULL,     # NULL removes model_1 bias
    dir = .BM_DIR
) {
  bm_ensure_loaded(name_hint, prefer_kind, prefer_ens, prefer_model, dir = dir)
  env_df <- bm_list_env()
  chosen <- bm_pick(env_df, prefer_kind, prefer_ens, prefer_model)
  if (is.null(chosen)) stop("bm_prepare_disk_only(): no metadata object found in env.")
  rds_path <- bm_save_copy(chosen$name, dir = dir)
  bm_clear_env_except(keep_names = character(0))
  invisible(rds_path)
}

# --- helper: choose one row deterministically when duplicates exist ---
.bm_choose_one <- function(df) {
  if (!nrow(df)) return(NULL)
  pref_suffix <- as.integer(grepl("metadata$", df$name))  # prefer *_metadata
  has_mtime   <- "mtime" %in% names(df)
  ord <- if (has_mtime) order(-pref_suffix, -as.numeric(df$mtime), df$name)
  else            order(-pref_suffix, df$name)
  df[ord[1], , drop = FALSE]
}

# ===== explicit selector for exact (kind, ens, model) ================
bm_select_exact <- function(kind = "Main", ens = 0L, model = 1L, dir = .BM_DIR) {
  # --- DE-DUPLICATE same (kind, ens, model), prefer RDS unless flag set ---
  
  if (!isTRUE(prepare_disk_only_FROM_RDS)) {
    env <- bm_list_env()
    hit <- subset(env, kind == kind & ens == ens & model == model)
    if (nrow(hit)) {
      pick <- .bm_choose_one(hit)
      return(get(pick$name, envir = .GlobalEnv))
    }
  }
  
  rds <- bm_list_rds(dir)
  hit <- subset(rds, kind == kind & ens == ens & model == model)
  if (nrow(hit)) {
    pick <- .bm_choose_one(hit)
    return(readRDS(pick$path))
  }
  
  stop(sprintf("Model not found: kind=%s ens=%d model=%d", kind, ens, model))
}

# ===== convenience auto-find (env) ===================================
.auto_find_meta <- function() {
  env_df <- bm_list_env()
  if (!nrow(env_df)) return(NULL)
  chosen <- bm_pick(env_df, prefer_kind = c("Main","Temp"),
                    prefer_ens = c(0L,1L), prefer_model = NULL)
  get(chosen$name, envir = .GlobalEnv)
}

# ===== record extractor (adapt to your metadata schema) ===============
extract_best_records <- function(meta, ML_NN, model_index = 1L) {
  bw <- meta$best_weights_record
  bb <- meta$best_biases_record
  
  pick_i <- suppressWarnings(as.integer(model_index))
  if (!is.finite(pick_i) || pick_i < 1L) pick_i <- 1L
  
  if (!is.null(bw) && !is.null(bb)) {
    if (is.list(bw) && length(bw) >= 1 && is.list(bw[[1]])) {
      idx <- if (pick_i <= length(bw)) pick_i else 1L
      w <- bw[[idx]]
      b <- bb[[min(idx, length(bb))]]
      return(list(weights = w, biases = b))
    } else {
      return(list(weights = bw, biases = bb))
    }
  }
  
  rec <- tryCatch(meta$records[[pick_i]], error = function(e) NULL)
  if (!is.null(rec) && !is.null(rec$weights) && !is.null(rec$biases)) {
    return(list(weights = rec$weights, biases = rec$biases))
  }
  
  stop(sprintf("No weights/biases found in metadata for model_index=%d", pick_i))
}

# ===== architecture inference (for rebuild) ==========================
.infer_hidden_sizes_from_weights <- function(w_list) {
  if (!is.list(w_list) || !length(w_list)) return(NULL)
  dims <- lapply(w_list, function(m) if (is.matrix(m)) dim(m) else NULL)
  dims <- Filter(Negate(is.null), dims)
  if (!length(dims)) return(NULL)
  if (length(dims) == 1L) return(integer(0))
  as.integer(vapply(dims[-length(dims)], function(d) d[2], integer(1)))
}

# ===== rebuild a DESONN container from metadata ======================
DESONN_from_metadata <- function(meta) {
  if (!exists("DESONN", inherits = TRUE))
    stop("DESONN class is not loaded; source the file that defines DESONN before predicting.")
  
  input_size   <- as.integer(meta$input_size %||% NA_integer_)
  output_size  <- as.integer(meta$output_size %||% NA_integer_)
  N            <- as.integer(meta$N %||% NA_integer_)
  lambda       <- as.numeric(meta$lambda %||% 0)
  ML_NN        <- isTRUE(meta$ML_NN) %||% TRUE
  method       <- meta$method %||% "xavier"
  custom_scale <- meta$custom_scale %||% NULL
  num_networks <- as.integer(meta$num_networks %||% 1L)
  
  hidden_sizes <- meta$hidden_sizes %||% (meta$architecture$hidden_sizes %||% NULL)
  if (is.null(hidden_sizes)) {
    bw <- meta$best_weights_record
    w0 <- if (is.list(bw) && length(bw) && is.list(bw[[1]])) bw[[1]] else bw
    hidden_sizes <- .infer_hidden_sizes_from_weights(w0) %||% integer(0)
  }
  
  DESONN$new(
    num_networks    = max(1L, num_networks),
    input_size      = input_size,
    hidden_sizes    = hidden_sizes,
    output_size     = output_size,
    N               = N,
    lambda          = lambda,
    ensemble_number = meta$ensemble_number %||% 1L,
    ensembles       = NULL,
    ML_NN           = ML_NN,
    method          = method,
    custom_scale    = custom_scale
  )
}

# =========================
# Metric helpers (general)
# =========================
.metric_minimize <- get0(".metric_minimize", ifnotfound = function(metric_name) {
  m <- tolower(metric_name)
  any(grepl("(loss|mse|mae|rmse|logloss|cross.?entropy|error|nll)", m))
})

.get_metric_from_meta <- function(meta, metric_name) {
  v <- suppressWarnings(tryCatch(meta$performance_metric[[metric_name]], error = function(e) NA_real_))
  if (is.finite(v)) return(as.numeric(v))
  if (exists(".resolve_metric_from_pm", inherits = TRUE)) {
    v <- suppressWarnings(tryCatch(.resolve_metric_from_pm(meta$performance_metric, metric_name), error = function(e) NA_real_))
    if (is.finite(v)) return(as.numeric(v))
  }
  NA_real_
}

# =========================
# Metric-ranked batch builder (dedup env+rds)
# =========================
.make_batch_by_metric <- function(scope = c("one","group-best","all"),
                                  kind_filter  = c("Main","Temp"),
                                  ens_filter   = NULL,
                                  model_filter = NULL,
                                  dir = .BM_DIR) {
  scope <- match.arg(scope)
  
  # 1) get all, then dedupe so we keep only one row per (kind,ens,model),
  #    preferring env over rds and *_metadata over *_metada, newest first.
  df <- bm_list_all(dir)
  if (!nrow(df)) return(list())
  df <- .bm_dedupe_rows(df)
  
  # 2) apply filters
  if (length(kind_filter))    df <- df[df$kind  %in% kind_filter, , drop = FALSE]
  if (!is.null(ens_filter))   df <- df[df$ens   %in% ens_filter,  , drop = FALSE]
  if (!is.null(model_filter)) df <- df[df$model %in% model_filter,, drop = FALSE]
  if (!nrow(df)) return(list())
  
  # 3) compute the ranking metric
  metric_name <- get0("TARGET_METRIC", ifnotfound = "macro_f1")
  df$metric_value <- vapply(seq_len(nrow(df)), function(i) {
    meta_i <- tryCatch(
      bm_select_exact(df$kind[i], df$ens[i], df$model[i], dir = dir),
      error = function(e) NULL
    )
    if (is.null(meta_i)) return(NA_real_)
    .get_metric_from_meta(meta_i, metric_name)
  }, numeric(1))
  
  ok <- is.finite(df$metric_value)
  if (!any(ok)) return(list())
  df <- df[ok, , drop = FALSE]
  
  # 4) rank
  minimize <- .metric_minimize(metric_name)
  ord <- if (minimize) order(df$metric_value) else order(df$metric_value, decreasing = TRUE)
  df  <- df[ord, , drop = FALSE]
  
  # 5) build batch entries
  build_entry <- function(rowi) {
    meta <- bm_select_exact(rowi$kind, rowi$ens, rowi$model, dir = dir)
    list(row = rowi, meta = meta)
  }
  
  if (scope == "one") {
    top <- df[1, , drop = FALSE]
    return(list(build_entry(top)))
  }
  
  if (scope == "group-best") {
    out <- lapply(split(df, df$ens), function(dfe) build_entry(dfe[1, , drop = FALSE]))
    return(Filter(Negate(is.null), out))
  }
  
  # scope == "all"
  lapply(seq_len(nrow(df)), function(i) build_entry(df[i, , drop = FALSE]))
}


# =========================
# "first | all | pick" selectors & snapshotters
# =========================
.bm_dedupe_rows <- function(df) {
  if (!nrow(df)) return(df)
  pref_env    <- if ("source" %in% names(df)) as.integer(df$source == "env") else integer(nrow(df))
  pref_suffix <- as.integer(grepl("metadata$", df$name))
  has_mtime   <- "mtime" %in% names(df)
  ord <- if (has_mtime) order(-pref_env, -pref_suffix, -as.numeric(df$mtime), df$name)
  else            order(-pref_env, -pref_suffix, df$name)
  df <- df[ord, , drop = FALSE]
  key <- paste(df$kind, df$ens, df$model, sep = "|")
  df[!duplicated(key), , drop = FALSE]
}

bm_select_by_choice <- function(
    choice       = c("first","all","pick"),
    kind_filter  = "Main",
    ens_filter   = NULL,
    model_filter = NULL,
    dir          = .BM_DIR
) {
  choice <- match.arg(choice)
  
  df <- bm_list_all(dir)
  if (!nrow(df)) stop("bm_select_by_choice(): no models found in env or RDS.")
  
  if (!is.null(kind_filter))  df <- df[df$kind  %in% kind_filter, , drop = FALSE]
  if (!is.null(ens_filter))   df <- df[df$ens   %in% ens_filter,  , drop = FALSE]
  if (!is.null(model_filter)) df <- df[df$model %in% model_filter, , drop = FALSE]
  if (!nrow(df)) stop("bm_select_by_choice(): no models after applying filters.")
  
  df <- .bm_dedupe_rows(df)
  df <- df[order(df$kind, df$ens, df$model, df$name), , drop = FALSE]
  
  if (choice == "all")   return(df)
  if (choice == "first") return(df[1, , drop = FALSE])
  
  if (nrow(df) == 1L) return(df)
  
  if (interactive()) {
    cat("\nSelect a model to export:\n")
    opts <- sprintf("[%d] %s  (kind=%s, ens=%d, model=%d, src=%s)",
                    seq_len(nrow(df)), df$name, df$kind, df$ens, df$model,
                    if ("source" %in% names(df)) df$source else "unknown")
    writeLines(opts)
    sel <- suppressWarnings(as.integer(readline("Enter number: ")))
    if (!is.finite(sel) || sel < 1L || sel > nrow(df)) {
      cat("Invalid selection; defaulting to first.\n")
      return(df[1, , drop = FALSE])
    }
    return(df[sel, , drop = FALSE])
  } else {
    return(df[1, , drop = FALSE])
  }
}

bm_prepare_disk_by_choice <- function(
    choice       = c("first","all","pick"),
    kind_filter  = "Main",
    ens_filter   = NULL,
    model_filter = NULL,
    dir          = .BM_DIR,
    clear_env    = TRUE
) {
  rows <- bm_select_by_choice(choice = choice,
                              kind_filter  = kind_filter,
                              ens_filter   = ens_filter,
                              model_filter = model_filter,
                              dir          = dir)
  
  ensure_one <- function(row) {
    nm <- row[["name"]]
    if (!exists(nm, envir = .GlobalEnv, inherits = FALSE)) {
      meta <- bm_select_exact(kind = row[["kind"]],
                              ens  = as.integer(row[["ens"]]),
                              model= as.integer(row[["model"]]),
                              dir  = dir)
      assign(nm, meta, envir = .GlobalEnv)
    }
    invisible(nm)
  }
  
  if (nrow(rows) == 1L) {
    nm       <- ensure_one(rows[1, ])
    rds_path <- bm_save_copy(nm, dir = dir)
    if (isTRUE(clear_env)) bm_clear_env_except(character(0))
    return(invisible(rds_path))
  }
  
  paths <- character(nrow(rows))
  for (i in seq_len(nrow(rows))) {
    nm        <- ensure_one(rows[i, ])
    paths[i]  <- bm_save_copy(nm, dir = dir)
  }
  if (isTRUE(clear_env)) bm_clear_env_except(character(0))
  invisible(paths)
}

# =========================
# Predict runners (stateful/stateless)
# =========================

.do_predict_stateless <- function(X, meta, rec, ML_NN = TRUE, ...) {
  des <- DESONN_from_metadata(meta)
  model <- if (length(des$ensemble)) des$ensemble[[1]] else des
  
  # Best-effort: set params if supported
  try(if (is.function(model$set_params)) model$set_params(weights = rec$weights, biases = rec$biases), silent = TRUE)
  
  activation_functions <- meta$activation_functions %||% NULL
  
  if (!is.function(model$predict))
    stop("Constructed model has no $predict() method; ensure DESONN/SONN classes are loaded.")
  
  # NOTE: do NOT pass dropout; your predict(...) doesn't take it.
  model$predict(
    Rdata = X,
    weights = rec$weights,
    biases  = rec$biases,
    activation_functions = activation_functions,
    ...
  )
}

.do_predict_stateful <- function(X, meta, rec, ML_NN = TRUE, ...) {
  des <- DESONN_from_metadata(meta)
  model <- if (length(des$ensemble)) des$ensemble[[1]] else des
  try(if (is.function(model$set_params)) model$set_params(weights = rec$weights, biases = rec$biases), silent = TRUE)
  if (!is.function(model$predict))
    stop("Constructed model has no $predict() method; ensure DESONN/SONN classes are loaded.")
  model$predict(Rdata = X, ...)
}
