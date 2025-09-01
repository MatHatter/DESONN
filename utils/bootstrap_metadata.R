# ---------------------------------------------------------------------
# Purpose:
# - Snapshot/load/auto-find DESONN metadata objects named like:
#     Ensemble_<Kind>_<ens>_model_<idx>_(metadata|metada)[_YYYYMMDD_HHMMSS]
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
# Allow optional timestamp suffix: _YYYYMMDD_HHMMSS
.BM_PAT <- "^Ensemble_([^_]+)_(\\d+)_model_(\\d+)_(?:metadata|metada)(?:_\\d{8}_\\d{6})?$"

# ===== core helpers: list env / rds ==================================
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
  f <- list.files(dir, pattern = "^Ensemble_.*_model_\\d+_(?:metadata|metada).*\\.rds$", full.names = TRUE)
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

# ===== preference scoring =============================================
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

# (rest of file unchanged … your existing bm_ensure_loaded, bm_select_exact,
# bm_prepare_disk_only, .bm_choose_one, extract_best_records, etc.)
