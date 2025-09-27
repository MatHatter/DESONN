# =====================================================================
# File: artifacts/PredictOnly/LoadandPredict.R
#
# Purpose:
#   Unified "predict-only" entrypoint that hides the stateful/stateless
#   distinction in comments, not code.
#
#   SOURCE OPTIONS (via 'source' arg):
#     • "env"           → use in-memory metadata objects
#       (conceptually “stateful”)
#     • "SingleRuns"    → read metadata .rds under artifacts/SingleRuns/<folder>/models/
#     • "EnsembleRuns"  → read metadata .rds under:
#           artifacts/EnsembleRuns/<folder>/models/main
#           artifacts/EnsembleRuns/<folder>/models/temp_eXX
#       (both conceptually “stateless” from disk)
#
#   FOLDER RESOLUTION:
#     • If 'source' is SingleRuns/EnsembleRuns and 'folder' is NULL,
#       we auto-pick the most recent subfolder.
#     • If 'folder' is provided, we use it exactly.
#
#   MODEL SELECTION POLICY:
#     • selection="first" (default) → evaluate the first model file found.
#     • selection="all"             → evaluate every model file discovered.
#
# Outputs go to: artifacts/PredictOnly/<run_dir_name>/
#
# Prereqs already defined in your session / utils:
#   - DDESONN_predict_eval()
#   - .filter_flat_metric_names()
#   - .strip_metric_prefixes()
# =====================================================================

# Sanity: evaluator must exist
if (!exists("DDESONN_predict_eval") || !is.function(DDESONN_predict_eval)) {
  stop("DDESONN_predict_eval() is not available in scope.")
}

LoadandPredict <- function(
    source = c("SingleRuns","EnsembleRuns","env"),
    folder = NULL,                      # exact subfolder under artifacts/<source>/..., or NULL → most recent
    env_meta_name = NULL,               # only when source="env"; NULL → auto-pick first canonical name
    predict_split = "test",             # "test" | "validation" | "train"
    CLASSIFICATION_MODE = "binary",     # ALL CAPS
    run_index = 1L,
    seed_val  = 1L,
    run_dir_name = format(Sys.time(), "%Y%m%d_%H%M%S_predict"),
    selection = c("first","all"),       # model selection policy inside the folder
    save_predictions_column = FALSE
) {
  source    <- match.arg(source)
  selection <- match.arg(selection)
  
  # -------------------------------------------------------------------
  # Prepare prediction output directory + aggregate files
  # -------------------------------------------------------------------
  base_dir <- file.path("artifacts", "PredictOnly", run_dir_name)
  if (!dir.exists(base_dir)) dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
  
  agg_predictions_file <- file.path(base_dir, paste0("agg_predictions_", predict_split, ".rds"))
  agg_metrics_file     <- file.path(base_dir, paste0("agg_metrics_",     predict_split, ".rds"))
  
  # ================================================================
  # PATH A: ENVIRONMENT (conceptually “stateful”)
  # ================================================================
  if (identical(source, "env")) {
    if (is.null(env_meta_name) || !nzchar(env_meta_name)) {
      # Minimal inline scan (no helpers here)
      candidates <- character(0)
      for (k in 1:64) candidates <- c(candidates, sprintf("Ensemble_Main_0_model_%d_metadata", k))
      for (e in 1:8) for (k in 1:64) candidates <- c(candidates, sprintf("Ensemble_Temp_%d_model_%d_metadata", e, k))
      hits <- candidates[sapply(candidates, function(nm) exists(nm, envir = .GlobalEnv))]
      if (!length(hits)) stop("No canonical metadata object found in the environment.")
      env_meta_name <- hits[1]
    } else if (!exists(env_meta_name, envir = .GlobalEnv)) {
      stop(sprintf("Specified env_meta_name not found in .GlobalEnv: %s", env_meta_name))
    }
    
    slot <- suppressWarnings(as.integer(sub("^.*_model_([0-9]+)_metadata$", "\\1", env_meta_name)))
    if (!is.finite(slot)) slot <- 1L
    
    ok <- tryCatch({
      DDESONN_predict_eval(
        LOAD_FROM_RDS = FALSE,
        ENV_META_NAME = env_meta_name,
        INPUT_SPLIT   = predict_split,
        CLASSIFICATION_MODE = CLASSIFICATION_MODE,
        RUN_INDEX = run_index,
        SEED      = seed_val,
        OUTPUT_DIR = base_dir,
        OUT_DIR_ASSERT = base_dir,
        SAVE_METRICS_RDS = TRUE,
        METRICS_PREFIX   = sprintf("metrics_%s", predict_split),
        SAVE_PREDICTIONS_COLUMN_IN_RDS = isTRUE(save_predictions_column),
        AGG_PREDICTIONS_FILE = agg_predictions_file,
        AGG_METRICS_FILE     = agg_metrics_file,
        MODEL_SLOT           = slot
      )
      TRUE
    }, error = function(e) {
      message(sprintf("[LoadandPredict:env] ERROR: %s", conditionMessage(e))); FALSE
    })
    
    cat(sprintf("[LoadandPredict:env] seed=%s slot=%d wrote? %s\n",
                as.character(seed_val), slot, ok))
    
    # Post-filter aggregate metrics names
    if (file.exists(agg_metrics_file)) {
      df <- try(readRDS(agg_metrics_file), silent = TRUE)
      if (!inherits(df, "try-error") && is.data.frame(df) && NROW(df)) {
        nms <- names(df)
        keep_map  <- setNames(as.list(rep(TRUE, length(nms))), nms)
        kept_list <- .filter_flat_metric_names(keep_map)
        keep_nms  <- names(kept_list)
        df <- df[, intersect(keep_nms, names(df)), drop = FALSE]
        df <- `colnames<-`(df, .strip_metric_prefixes(names(df)))
        saveRDS(df, agg_metrics_file)
      }
    }
    
    return(invisible(list(
      source          = "env",
      run_dir         = base_dir,
      agg_predictions = agg_predictions_file,
      agg_metrics     = agg_metrics_file,
      n_models_scored = 1L
    )))
  }
  
  # ================================================================
  # PATH B: FOLDERS (SingleRuns / EnsembleRuns) — conceptually “stateless”
  # ================================================================
  
  # 1) Resolve root folder under artifacts/<source>
  family_root <- file.path("artifacts", source)
  if (!dir.exists(family_root)) stop(sprintf("Family root does not exist: %s", family_root))
  
  # If user did not provide a folder, pick most recent subfolder
  run_root <- NULL
  if (!is.null(folder) && nzchar(folder)) {
    run_root <- file.path(family_root, folder)
    if (!dir.exists(run_root)) stop(sprintf("Requested folder not found: %s", run_root))
  } else {
    kids <- list.dirs(family_root, full.names = TRUE, recursive = FALSE)
    kids <- kids[dir.exists(kids)]
    if (!length(kids)) stop(sprintf("No subfolders found in: %s", family_root))
    info <- file.info(kids)
    run_root <- kids[order(info$mtime, decreasing = TRUE)][1L]
  }
  cat(sprintf("[LoadandPredict:%s] Using folder: %s\n", source, run_root))
  
  # 2) Build models roots STRICTLY as requested
  models_root <- file.path(run_root, "models")
  if (!dir.exists(models_root)) {
    stop(sprintf("[LoadandPredict:%s] Expected models/ not found under: %s", source, run_root))
  }
  
  searched_dirs <- character(0)
  candidate_files <- character(0)
  
  pattern_main <- "^Ensemble_Main_[0-9]+_model_[0-9]+_metadata.*\\.rds$"
  pattern_temp <- "^Ensemble_Temp_[0-9]+_model_[0-9]+_metadata.*\\.rds$"
  
  if (identical(source, "SingleRuns")) {
    # SingleRuns → ONLY models/ (no subfolders)
    searched_dirs <- c(searched_dirs, models_root)
    cf <- list.files(models_root, pattern = pattern_main, full.names = TRUE, recursive = FALSE)
    candidate_files <- c(candidate_files, cf)
  } else {
    # EnsembleRuns → models/main + models/temp_eXX (no deeper recursion)
    main_dir <- file.path(models_root, "main")
    if (dir.exists(main_dir)) {
      searched_dirs <- c(searched_dirs, main_dir)
      cf_main <- list.files(main_dir, pattern = pattern_main, full.names = TRUE, recursive = FALSE)
      candidate_files <- c(candidate_files, cf_main)
    }
    
    # temp_eXX peers under models/
    temp_dirs <- list.dirs(models_root, full.names = TRUE, recursive = FALSE)
    temp_dirs <- temp_dirs[grepl("(^|/)temp_e\\d{2}$", temp_dirs)]
    if (length(temp_dirs)) {
      searched_dirs <- c(searched_dirs, temp_dirs)
      for (td in temp_dirs) {
        cf_tmp <- list.files(td, pattern = pattern_temp, full.names = TRUE, recursive = FALSE)
        candidate_files <- c(candidate_files, cf_tmp)
      }
    }
  }
  
  cat("[LoadandPredict:", source, "] searched:\n  - ",
      paste(unique(normalizePath(searched_dirs, winslash = "/", mustWork = FALSE)), collapse = "\n  - "),
      "\n", sep = "")
  
  if (!length(candidate_files)) {
    stop(sprintf("[LoadandPredict:%s] No metadata .rds found under models/ in: %s", source, run_root))
  }
  
  # 3) Selection policy
  candidate_files <- sort(candidate_files, decreasing = FALSE)
  chosen <- if (identical(selection, "first")) candidate_files[1L] else candidate_files
  
  cat("[LoadandPredict:", source, "] selected file(s):\n  - ",
      paste(chosen, collapse = "\n  - "), "\n", sep = "")
  
  # 4) Point evaluator at models root via .BM_DIR so base-name lookups work
  old_bm <- if (exists(".BM_DIR", inherits = TRUE)) get(".BM_DIR", inherits = TRUE) else NULL
  assign(".BM_DIR", models_root, envir = .GlobalEnv)
  on.exit({
    if (is.null(old_bm)) {
      if (exists(".BM_DIR", envir = .GlobalEnv, inherits = FALSE)) rm(".BM_DIR", envir = .GlobalEnv)
    } else {
      assign(".BM_DIR", old_bm, envir = .GlobalEnv)
    }
  }, add = TRUE)
  
  # 5) Evaluate selected files
  n_ok <- 0L
  for (p in chosen) {
    b <- basename(p)
    # Strip optional trailing timestamp or any tail after "..._metadata"
    base_name <- sub("\\.rds$", "", b)
    base_name <- sub("_(\\d{8}_\\d{6})$", "", base_name)     # if timestamp exists
    # ensure we end exactly at "..._metadata"
    base_name <- sub("(_.*)?$", "", sub("(.*_metadata).*", "\\1", base_name))
    
    slot <- suppressWarnings(as.integer(sub("^Ensemble_(?:Main|Temp)_[0-9]+_model_([0-9]+)_metadata.*$", "\\1", b)))
    if (!is.finite(slot)) slot <- 1L
    
    ok <- tryCatch({
      DDESONN_predict_eval(
        LOAD_FROM_RDS = TRUE,                  # from disk by BASE name
        ENV_META_NAME = base_name,
        INPUT_SPLIT   = predict_split,
        CLASSIFICATION_MODE = CLASSIFICATION_MODE,
        RUN_INDEX = run_index,
        SEED      = seed_val,
        OUTPUT_DIR = base_dir,
        OUT_DIR_ASSERT = base_dir,
        SAVE_METRICS_RDS = TRUE,
        METRICS_PREFIX   = sprintf("metrics_%s", predict_split),
        SAVE_PREDICTIONS_COLUMN_IN_RDS = isTRUE(save_predictions_column),
        AGG_PREDICTIONS_FILE = agg_predictions_file,
        AGG_METRICS_FILE     = agg_metrics_file,
        MODEL_SLOT           = slot
      )
      TRUE
    }, error = function(e) {
      message(sprintf("[LoadandPredict:%s] ERROR for %s: %s", source, b, conditionMessage(e)))
      FALSE
    })
    
    n_ok <- n_ok + as.integer(ok)
    cat(sprintf("[LoadandPredict:%s] seed=%s slot=%d wrote? %s (%s)\n",
                source, as.character(seed_val), slot, ok, b))
  }
  
  # 6) Post-filter aggregate metrics names (uses utils helpers)
  if (file.exists(agg_metrics_file)) {
    df <- try(readRDS(agg_metrics_file), silent = TRUE)
    if (!inherits(df, "try-error") && is.data.frame(df) && NROW(df)) {
      nms <- names(df)
      keep_map  <- setNames(as.list(rep(TRUE, length(nms))), nms)
      kept_list <- .filter_flat_metric_names(keep_map)
      keep_nms  <- names(kept_list)
      df <- df[, intersect(keep_nms, names(df)), drop = FALSE]
      df <- `colnames<-`(df, .strip_metric_prefixes(names(df)))
      saveRDS(df, agg_metrics_file)
    }
  }
  
  invisible(list(
    source          = source,
    folder_used     = run_root,
    models_root     = models_root,
    run_dir         = base_dir,
    agg_predictions = agg_predictions_file,
    agg_metrics     = agg_metrics_file,
    n_models_scored = n_ok
  ))
}

# -----------------------
# Examples (commented)
# -----------------------

# 1) From most-recent EnsembleRuns folder; evaluate FIRST model found
LoadandPredict(
  source="EnsembleRuns",
  folder=NULL,
  predict_split="test",
  CLASSIFICATION_MODE="binary",
  run_index=1L, seed_val=1L,
  run_dir_name="predict_from_latest_ensemble",
  selection="first"
)

# 2) From a specific SingleRuns folder; evaluate ALL models found
# LoadandPredict(
#   source="SingleRuns",
#   folder="20250922_175159__m1_wSeed",
#   predict_split="test",
#   CLASSIFICATION_MODE="binary",
#   run_index=1L, seed_val=1L,
#   run_dir_name="predict_all_from_single_run",
#   selection="all"
# )

# 3) From ENV (most-recent model in memory), auto-pick first canonical object
# LoadandPredict(
#   source="env",
#   env_meta_name=NULL,
#   predict_split="test",
#   CLASSIFICATION_MODE="binary",
#   run_index=1L, seed_val=1L,
#   run_dir_name="predict_from_env"
# )
