# utils/bootstrap_metadata.R
# ---------------------------------------------------------------------
# Purpose: Snapshot/load/auto-find DESONN metadata objects named like
#   Ensemble_<Kind>_<ens>_model_<idx>_(metadata|metada)
# + Provide a hard-stop helper for one-shot “prepare disk only” runs.
# + Provide robust extract_best_records() that prefers best_*_record.
# ---------------------------------------------------------------------

# ----- hard stop helper ----------------------------------------------
.hard_stop <- function(msg = "[prepare_disk_only] Done; stopping script.") {
  cat(msg, "\n")
  if (interactive()) {
    stop(msg, call. = FALSE)
  } else {
    quit(save = "no")
  }
}

# ----- config ---------------------------------------------------------
.BM_DIR <- "artifacts"  # where metadata RDS goes
if (!dir.exists(.BM_DIR)) dir.create(.BM_DIR, recursive = TRUE, showWarnings = FALSE)

# Regex to match metadata symbols in .GlobalEnv
.BM_PAT <- "^Ensemble_([^_]+)_(\\d+)_model_(\\d+)_(?:metadata|metada)$"

# ----- tiny utils -----------------------------------------------------
bm_save_copy <- function(obj_name, dir = .BM_DIR, compress = TRUE) {
  if (!exists(obj_name, envir = .GlobalEnv, inherits = FALSE)) {
    stop(sprintf("bm_save_copy(): object '%s' not found in .GlobalEnv.", obj_name))
  }
  path <- file.path(dir, paste0(obj_name, ".rds"))
  saveRDS(get(obj_name, envir = .GlobalEnv), path, compress = compress)
  invisible(path)
}

bm_list_env <- function() {
  nm <- ls(envir = .GlobalEnv, all.names = FALSE)
  nm <- nm[grepl(.BM_PAT, nm)]
  if (!length(nm)) {
    return(data.frame(name = character(), kind = character(), ens = integer(), model = integer()))
  }
  m <- regexec(.BM_PAT, nm)
  parts <- regmatches(nm, m)
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
  m  <- regexec(.BM_PAT, nm)
  parts <- regmatches(nm, m)
  do.call(rbind, lapply(seq_along(parts), function(i) data.frame(
    path  = f[i], name  = nm[i],
    kind  = parts[[i]][2],
    ens   = as.integer(parts[[i]][3]),
    model = as.integer(parts[[i]][4]),
    mtime = file.info(f[i])$mtime,
    stringsAsFactors = FALSE
  )))
}

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

bm_ensure_loaded <- function(
    name_hint = NULL,
    prefer_kind = c("Main","Temp"),
    prefer_ens = c(0L, 1L),
    prefer_model = 1L,
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

# ---- PHASE 1: prepare disk-only artifact (wipe env completely) ----
bm_prepare_disk_only <- function(
    name_hint = NULL,
    prefer_kind = c("Main","Temp"),
    prefer_ens  = c(0L, 1L),
    prefer_model= 1L,
    dir = .BM_DIR
) {
  bm_ensure_loaded(name_hint, prefer_kind, prefer_ens, prefer_model, dir = dir)
  env_df <- bm_list_env()
  chosen <- bm_pick(env_df, prefer_kind, prefer_ens, prefer_model)
  if (is.null(chosen)) stop("[bm_prepare_disk_only] No metadata object found in env.")
  
  rds_path <- bm_save_copy(chosen$name, dir = dir)
  bm_clear_env_except(keep_names = character(0))
  invisible(rds_path)
}

# ----- metadata shape & selection -----------------------------------
.is_metadata_like <- function(x) {
  is.list(x) && any(c(
    "best_weights_record","best_biases_record",
    "best_weights","best_biases",
    "weights","biases",
    "records"  # fallback structure
  ) %in% names(x))
}

# Prefer best_*_record; otherwise sensible fallbacks
extract_best_records <- function(meta, ML_NN = TRUE, model_index = 1L) {
  # 1) Explicit best_*_record (most reliable)
  if (!is.null(meta$best_weights_record) && !is.null(meta$best_biases_record)) {
    w <- meta$best_weights_record
    b <- meta$best_biases_record
    # Handle possible nested list structure for ensembles/models
    if (is.list(w) && length(w) && is.list(w[[1]])) {
      w <- w[[model_index]]
    }
    if (is.list(b) && length(b) && is.list(b[[1]])) {
      b <- b[[model_index]]
    }
    return(list(weights = w, biases = b))
  }
  
  # 2) Legacy best_* (non-record)
  if (!is.null(meta$best_weights) && !is.null(meta$best_biases)) {
    return(list(weights = meta$best_weights, biases = meta$best_biases))
  }
  
  # 3) Flat weights/biases directly on meta
  if (!is.null(meta$weights) && !is.null(meta$biases)) {
    return(list(weights = meta$weights, biases = meta$biases))
  }
  
  # 4) records[[model_index]] fallback
  if (!is.null(meta$records) && length(meta$records) >= model_index) {
    rec <- meta$records[[model_index]]
    if (!is.null(rec$weights) && !is.null(rec$biases)) {
      return(list(weights = rec$weights, biases = rec$biases))
    }
  }
  
  stop("[extract_best_records] Could not find weights/biases in metadata.")
}

# Auto-find the *best* metadata currently in RAM
.auto_find_meta <- function(prefer_kind = c("Main","Temp"), prefer_ens = c(0L,1L), prefer_model = 1L) {
  env_df <- bm_list_env()
  if (!nrow(env_df)) return(NULL)
  pick <- bm_pick(env_df, prefer_kind = prefer_kind, prefer_ens = prefer_ens, prefer_model = prefer_model)
  obj  <- get(pick$name, envir = .GlobalEnv)
  if (.is_metadata_like(obj)) obj else NULL
}
