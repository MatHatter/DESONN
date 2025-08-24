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


# ---- helper for
# =======================
# PREDICT-ONLY SHORT-CIRCUIT
# =======================
# if (!train) in TestDESONN.R
# Auto-detect a saved metadata object in the workspace and return the OBJECT itself.
# Pref order: Main_1, Main_0, any Main_*, then Temp_1, then any Temp_*.
if (!exists(".auto_find_meta", inherits = TRUE)) {
  # Auto-detect a saved metadata object in the workspace and return the OBJECT itself.
  # Preference order: Main_1, Main_0, any Main_*, then Temp_1, then any Temp_*.
  .auto_find_meta <- function() {
    all_vars <- ls(.GlobalEnv)
    hits <- grep("^Ensemble_(Main|Temp)_[0-9]+_model_[0-9]+_metadat?a$",
                 all_vars, ignore.case = TRUE, value = TRUE)
    if (!length(hits)) return(NULL)
    
    # Build integer scores with a simple loop (no vapply)
    scores <- integer(length(hits))
    for (i in seq_along(hits)) {
      nm <- hits[i]
      is_main <- grepl("^Ensemble_Main_", nm, ignore.case = TRUE)
      ens_num <- suppressWarnings(as.integer(
        sub("^Ensemble_(?:Main|Temp)_([0-9]+).*", "\\1", nm, perl = TRUE)
      ))
      
      s <-
        if (is_main && !is.na(ens_num) && ens_num == 1L) 1L
      else if (is_main && !is.na(ens_num) && ens_num == 0L) 2L
      else if (is_main)                                   3L
      else if (!is_main && !is.na(ens_num) && ens_num == 1L) 4L
      else 5L
      
      scores[i] <- as.integer(s)
    }
    
    # Order by score and return the first object
    hits <- hits[order(scores)]
    get(hits[1], envir = .GlobalEnv)
  }
}


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

if (!exists("extract_best_records", inherits = TRUE)) {
  # Works with either top-level best_*_record or nested best_model_metadata
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

# Optional: only needed if you want to target a specific metadata explicitly
if (!exists(".find_meta", inherits = TRUE)) {
  .find_meta <- function(kind = c("Main","Temp"), ens, model) {
    kind <- match.arg(kind)
    base <- sprintf("Ensemble_%s_%d_model_%d_", kind, as.integer(ens), as.integer(model))
    cands <- c(paste0(base,"metadata"), paste0(base,"metada"))  # tolerate typo
    for (nm in cands) if (exists(nm, envir = .GlobalEnv)) return(get(nm, envir = .GlobalEnv))
    NULL
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










tune_threshold <- function(predicted_output, labels) {
  thresholds <- seq(0.05, 0.95, by = 0.01)
  
  f1_scores <- sapply(thresholds, function(t) {
    preds <- ifelse(predicted_output >= t, 1, 0)
    TP <- sum(preds == 1 & labels == 1)
    FP <- sum(preds == 1 & labels == 0)
    FN <- sum(preds == 0 & labels == 1)
    precision <- TP / (TP + FP + 1e-8)
    recall <- TP / (TP + FN + 1e-8)
    f1 <- 2 * precision * recall / (precision + recall + 1e-8)
    return(f1)
  })
  
  best_threshold <- thresholds[which.max(f1_scores)]
  binary_preds <- ifelse(predicted_output >= best_threshold, 1, 0)
  
  return(list(
    best_threshold = best_threshold,
    binary_preds = binary_preds,
    f1_scores = f1_scores
  ))
}

tune_threshold_accuracy <- function(predicted_output, labels) {
  thresholds <- seq(0.05, 0.95, by = 0.01)
  
  accuracies <- sapply(thresholds, function(t) {
    preds <- ifelse(predicted_output >= t, 1, 0)
    sum(preds == labels) / length(labels)
  })
  
  best_threshold <- thresholds[which.max(accuracies)]
  binary_preds <- ifelse(predicted_output >= best_threshold, 1, 0)
  
  return(list(
    best_threshold = best_threshold,
    binary_preds = binary_preds,
    accuracy_scores = accuracies
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


