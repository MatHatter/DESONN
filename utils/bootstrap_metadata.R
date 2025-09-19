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
.BM_DIR <- get0(".BM_DIR", ifnotfound = "artifacts")
if (!dir.exists(.BM_DIR)) dir.create(.BM_DIR, recursive = TRUE, showWarnings = FALSE)

# Regex to match metadata symbols in .GlobalEnv / filenames
# Allow optional timestamp suffix: _YYYYMMDD_HHMMSS
.BM_PAT <- "^Ensemble_([^_]+)_(\\d+)_model_(\\d+)_(?:metadata|metada)(?:_\\d{8}_\\d{6})?$"

# -------- internal: strip trailing timestamp from base name -----------
.bm_strip_ts <- function(x) sub("_(\\d{8}_\\d{6})$", "", x)

# ===== core helpers: list env / rds ==================================
bm_save_copy <- function(obj_name, dir = .BM_DIR, compress = TRUE) {
  if (!exists(obj_name, envir = .GlobalEnv, inherits = FALSE)) {
    stop(sprintf("bm_save_copy(): '%s' not found in .GlobalEnv.", obj_name))
  }
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(dir, paste0(obj_name, ".rds"))
  if (file.exists(path)) return(invisible(path))  # do NOT overwrite
  saveRDS(get(obj_name, envir = .GlobalEnv), path, compress = compress)
  invisible(path)
}

bm_list_env <- function() {
  nm <- ls(envir = .GlobalEnv, all.names = FALSE)
  nm <- nm[grepl(.BM_PAT, nm)]
  if (!length(nm)) {
    return(data.frame(name = character(), kind = character(), ens = integer(), model = integer(),
                      source = character(), stringsAsFactors = FALSE))
  }
  m <- regexec(.BM_PAT, nm); parts <- regmatches(nm, m)
  out <- do.call(rbind, lapply(parts, function(p) {
    if (length(p) >= 4) {
      data.frame(
        name  = p[1], kind = p[2],
        ens   = as.integer(p[3]), model = as.integer(p[4]),
        source = "env", stringsAsFactors = FALSE
      )
    } else NULL
  }))
  rownames(out) <- NULL
  out
}

bm_list_rds <- function(dir = .BM_DIR) {
  if (!dir.exists(dir)) {
    return(data.frame(path = character(), name = character(), kind = character(),
                      ens = integer(), model = integer(), mtime = as.POSIXct(character()),
                      source = character(), stringsAsFactors = FALSE))
  }
  f <- list.files(dir, pattern = "^Ensemble_.*_model_\\d+_(?:metadata|metada).*\\.rds$", full.names = TRUE)
  if (!length(f)) {
    return(data.frame(path = character(), name = character(), kind = character(),
                      ens = integer(), model = integer(), mtime = as.POSIXct(character()),
                      source = character(), stringsAsFactors = FALSE))
  }
  nm <- sub("\\.rds$", "", basename(f))
  m  <- regexec(.BM_PAT, nm); parts <- regmatches(nm, m)
  out <- do.call(rbind, lapply(seq_along(parts), function(i) {
    p <- parts[[i]]
    if (length(p) >= 4) {
      data.frame(
        path  = f[i], name  = nm[i],
        kind  = p[2], ens   = as.integer(p[3]), model = as.integer(p[4]),
        mtime = file.info(f[i])$mtime,
        source = "rds", stringsAsFactors = FALSE
      )
    } else NULL
  }))
  rownames(out) <- NULL
  out
}

bm_list_all <- function(dir = .BM_DIR) {
  env <- bm_list_env()
  rds <- bm_list_rds(dir)
  cols <- c("name","kind","ens","model","source")
  if (nrow(env)) env <- env[, cols, drop = FALSE]
  if (nrow(rds)) rds <- rds[, cols, drop = FALSE]
  out <- rbind(env, rds)
  if (!nrow(out)) {
    return(data.frame(name=character(), kind=character(), ens=integer(), model=integer(),
                      source=character(), stringsAsFactors = FALSE))
  }
  # ensure stable sort
  out <- out[order(out$kind, out$ens, out$model, match(tolower(out$source), c("env","rds"))), , drop = FALSE]
  rownames(out) <- NULL
  out
}

bm_print_all <- function(dir = .BM_DIR) {
  cat("Discovered models (kind / ens / model):\n")
  df <- bm_list_all(dir)
  if (!nrow(df)) { cat("  [none]\n"); return(invisible(df)) }
  apply(df, 1, function(r) {
    cat(sprintf("  %-40s  kind=%-6s ens=%-3d model=%-3d (%s)\n",
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

# ===== choose-one policy (env preferred unless overridden) ============
.bm_choose_one <- function(df, prefer_kind = c("Main","Temp"),
                           prefer_ens = NULL, prefer_model = NULL,
                           prefer_source = c("env","rds")) {
  if (!nrow(df)) stop(".bm_choose_one: empty candidate set.")
  df <- df[order(match(tolower(df$source), tolower(prefer_source))), , drop = FALSE]
  pick <- bm_pick(df, prefer_kind = prefer_kind, prefer_ens = prefer_ens, prefer_model = prefer_model)
  pick
}

# ===== exact selector (returns the metadata object) ===================
bm_select_exact <- function(kind = "Main", ens = 0L, model = 1L, dir = .BM_DIR,
                            prefer_source = c("env","rds")) {
  # First look in env by exact symbol; then on disk
  # Accept either with or without timestamp on disk.
  # Env path:
  env_name <- sprintf("Ensemble_%s_%d_model_%d_metadata", kind, as.integer(ens), as.integer(model))
  if (exists(env_name, envir = .GlobalEnv, inherits = FALSE) &&
      "list" %in% class(get(env_name, envir = .GlobalEnv))) {
    return(get(env_name, envir = .GlobalEnv))
  }
  # Disk path(s):
  patt <- sprintf("^Ensemble_%s_%d_model_%d_(?:metadata|metada)(?:_\\d{8}_\\d{6})?\\.rds$",
                  kind, as.integer(ens), as.integer(model))
  files <- list.files(dir, pattern = patt, full.names = TRUE)
  if (length(files)) {
    # If multiple, prefer the newest mtime
    info <- file.info(files)
    files <- files[order(info$mtime, decreasing = TRUE)]
    return(readRDS(files[1]))
  }
  stop(sprintf("Model not found: kind=%s ens=%d model=%d", kind, as.integer(ens), as.integer(model)))
}

# ===== clear env except a set of names =================================
bm_clear_env_except <- function(keep_names = character()) {
  keep_base <- unique(.bm_strip_ts(sub("\\.rds$", "", keep_names)))
  all_objs  <- ls(envir = .GlobalEnv, all.names = FALSE)
  # We keep any object whose base name (without timestamp) is in keep_base
  base_of <- function(x) .bm_strip_ts(x)
  to_rm <- setdiff(all_objs, all_objs[base_of(all_objs) %in% keep_base])
  if (length(to_rm)) rm(list = to_rm, envir = .GlobalEnv)
  invisible(TRUE)
}

# ===== prepare disk only (legacy exact-name path) ======================
bm_prepare_disk_only <- function(name_hint,
                                 prefer_kind  = c("Main","Temp"),
                                 prefer_ens   = c(0L, 1L),
                                 prefer_model = 1L,
                                 dir = .BM_DIR) {
  df <- bm_list_all(dir)
  if (!nrow(df)) stop("bm_prepare_disk_only: no candidates in env or RDS.")
  # Narrow to those that match base (without timestamp)
  base <- .bm_strip_ts(name_hint)
  df$base <- .bm_strip_ts(df$name)
  subset_df <- df[df$base == base, , drop = FALSE]
  if (!nrow(subset_df)) stop(sprintf("bm_prepare_disk_only: '%s' not found among candidates.", name_hint))
  pick <- .bm_choose_one(subset_df, prefer_kind = prefer_kind,
                         prefer_ens = prefer_ens, prefer_model = prefer_model,
                         prefer_source = c("env","rds"))
  if (is.null(pick) || !nrow(pick)) stop("bm_prepare_disk_only: no pick returned.")
  # Save to RDS if the pick came from env; else return existing path
  if (identical(tolower(pick$source[1]), "env")) {
    path <- bm_save_copy(pick$name[1], dir = dir)
    return(path)
  } else {
    # Already on disk; return its most recent file
    patt <- paste0("^", pick$name[1], "(?:_\\d{8}_\\d{6})?\\.rds$")
    files <- list.files(dir, pattern = patt, full.names = TRUE)
    if (!length(files)) stop("bm_prepare_disk_only: RDS disappeared between list and save.")
    info <- file.info(files)
    files <- files[order(info$mtime, decreasing = TRUE)]
    return(files[1])
  }
}

# ===== CHOICE-BASED export: "first" | "all" | "pick" ===================
bm_prepare_disk_by_choice <- function(choice = c("first","all","pick"),
                                      kind_filter  = get0("KIND_FILTER",  ifnotfound = c("Main","Temp")),
                                      ens_filter   = get0("ENS_FILTER",   ifnotfound = NULL),
                                      model_filter = get0("MODEL_FILTER", ifnotfound = NULL),
                                      dir = .BM_DIR,
                                      clear_env = FALSE) {
  choice <- match.arg(choice)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  
  df <- bm_list_all(dir)
  if (!nrow(df)) stop("bm_prepare_disk_by_choice: no candidates found in env or RDS.")
  
  # apply filters
  if (length(kind_filter))   df <- df[df$kind  %in% kind_filter, , drop = FALSE]
  if (!is.null(ens_filter))  df <- df[df$ens   %in% ens_filter,  , drop = FALSE]
  if (!is.null(model_filter)) df <- df[df$model %in% model_filter, , drop = FALSE]
  if (!nrow(df)) stop("bm_prepare_disk_by_choice: no candidates after applying filters.")
  
  # stable ordering: by kind, ens, model, then prefer ENV over RDS for saving
  src_rank <- match(tolower(df$source), c("env","memory","workspace","rds","file","disk"))
  src_rank[is.na(src_rank)] <- 99L
  df <- df[order(df$kind, df$ens, df$model, src_rank), , drop = FALSE]
  
  pick_rows <- switch(choice,
                      "first" = df[1, , drop = FALSE],
                      "all"   = df,
                      "pick"  = {
                        idx <- as.integer(get0("PICK_INDEX", ifnotfound = 1L))
                        if (idx < 1L || idx > nrow(df)) stop(sprintf("PICK_INDEX=%d out of range [1..%d]", idx, nrow(df)))
                        df[idx, , drop = FALSE]
                      }
  )
  
  save_one <- function(row) {
    nm <- row[["name"]]; src <- tolower(row[["source"]])
    if (identical(src, "env")) {
      path <- bm_save_copy(nm, dir = dir)
      return(path)
    } else if (identical(src, "rds")) {
      patt  <- paste0("^", nm, "(?:_\\d{8}_\\d{6})?\\.rds$")
      files <- list.files(dir, pattern = patt, full.names = TRUE)
      if (!length(files)) stop(sprintf("RDS file for '%s' not found (race?).", nm))
      info <- file.info(files)
      files <- files[order(info$mtime, decreasing = TRUE)]
      return(files[1])
    } else {
      # Unexpected source; try env-first fallback
      if (exists(nm, envir = .GlobalEnv, inherits = FALSE)) {
        path <- bm_save_copy(nm, dir = dir)
        return(path)
      }
      stop(sprintf("Unsupported source '%s' for '%s'.", src, nm))
    }
  }
  
  paths <- vapply(seq_len(nrow(pick_rows)), function(i) save_one(pick_rows[i, , drop = FALSE]),
                  FUN.VALUE = character(1))
  if (isTRUE(clear_env)) {
    rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)
    gc()
  }
  unname(paths)
}

# ===== metric helpers used by predict-only ranking =====================
.metric_minimize <- function(metric_name) {
  # Return TRUE if lower is better for this metric
  m <- tolower(metric_name)
  # Assume classification metrics are maximize; losses (mse, rmse, mae, crossentropy) minimize
  if (m %in% c("mse","rmse","mae","crossentropy","logloss","loss")) return(TRUE)
  FALSE
}

.get_metric_from_meta <- function(meta, metric_name = "accuracy") {
  m <- tolower(metric_name)
  # Try multiple slots in order of likelihood
  # 1) validation_metrics list
  val <- tryCatch(meta$validation_metrics, error = function(e) NULL)
  if (is.list(val) && length(val)) {
    cand <- val[[m]] %||% val[[toupper(m)]] %||% val[[tools::toTitleCase(m)]]
    if (is.numeric(cand) && length(cand) == 1) return(as.numeric(cand))
  }
  # 2) performance_metric top-level
  pm <- tryCatch(meta$performance_metric, error = function(e) NULL)
  if (is.numeric(pm) && length(pm) == 1 && m %in% c("accuracy","f1","precision","recall")) {
    return(as.numeric(pm))
  }
  # 3) summary_stats or similar
  ss <- tryCatch(meta$summary_stats, error = function(e) NULL)
  if (is.list(ss) && length(ss)) {
    cand <- ss[[m]] %||% ss[[toupper(m)]] %||% ss[[tools::toTitleCase(m)]]
    if (is.numeric(cand) && length(cand) == 1) return(as.numeric(cand))
  }
  # Fallback: NA
  NA_real_
}
