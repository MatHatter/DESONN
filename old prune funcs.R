find_and_print_best_performing_models <- function(performance_names, relevance_names, performance_metrics, relevance_metrics, target_metric_name_best) {
  
  # Initialize an empty list to hold all metric names
  all_metric_names <- union(unlist(performance_names), unlist(relevance_names))
  
  # Check if the target metric exists in the list of all metrics
  if (!(target_metric_name_best %in% all_metric_names)) {
    cat("Target metric", target_metric_name_best, "not found.\n")
    return(NULL)
  }
  
  # Determine if the target metric should be minimized or maximized
  minimize_metric <- target_metric_name_best %in% c("speed_learn", "speed", "memory_usage", "quantization_error", "topographic_error", "clustering_quality_db", "MSE", "MAE", "RMSE", "serendipity", "diversity", "hit_rate")
  maximize_metric <- target_metric_name_best %in% c("precision", "recall", "f1_score", "ndcg", "mean_precision", "robustness", "generalization_ability")
  
  # Initialize the best performance value appropriately
  if (minimize_metric) {
    best_performance_value <- Inf
  } else if (maximize_metric) {
    best_performance_value <- -Inf
  } else {
    best_performance_value <- -Inf  # Default to maximizing if not listed
  }
  
  best_model_index <- NULL
  
  # Find the best-performing model for the target metric in performance_metrics
  for (i in seq_along(performance_metrics)) {
    if (target_metric_name_best %in% performance_names[[i]]) {
      metric_index <- match(target_metric_name_best, performance_names[[i]])
      cat("Checking performance model", i, "for metric", target_metric_name_best, "with metric index", metric_index, "\n")  # Debug print
      
      if (!is.na(metric_index) && metric_index <= length(performance_metrics[[i]])) {
        metric_value <- performance_metrics[[i]][[metric_index]]
        cat("Metric value for performance model", i, "is", toString(metric_value), "\n")  # Debug print
        
        if (is.numeric(metric_value) && !is.na(metric_value)) {
          if ((minimize_metric && metric_value < best_performance_value) ||
              (maximize_metric && metric_value > best_performance_value)) {
            best_performance_value <- metric_value
            best_model_index <- i
            cat("New best model index", best_model_index, "with value", best_performance_value, "\n")  # Debug print
          }
        }
      } else {
        cat("Invalid metric index or metric not found for performance model", i, "\n")  # Debug print
      }
    }
  }
  
  
  # Find the best-performing model for the target metric in relevance_metrics
  for (i in seq_along(relevance_metrics)) {
    if (target_metric_name_best %in% relevance_names[[i]]) {
      metric_index <- match(target_metric_name_best, relevance_names[[i]])
      cat("Checking relevance model", i, "for metric", target_metric_name_best, "with metric index", metric_index, "\n")  # Debug print
      if (!is.na(metric_index) && metric_index <= length(relevance_metrics[[i]])) {
        metric_value <- relevance_metrics[[i]][[metric_index]]
        cat("Metric value for relevance model", i, "is", metric_value, "\n")  # Debug print
        if (is.numeric(metric_value) && !is.na(metric_value)) {
          if ((minimize_metric && metric_value < best_performance_value) ||
              (maximize_metric && metric_value > best_performance_value)) {
            best_performance_value <- metric_value
            best_model_index <- i + length(performance_metrics)  # Adjust index for relevance metrics
            cat("New best model index", best_model_index, "with value", best_performance_value, "\n")  # Debug print
          }
        }
      } else {
        cat("Invalid metric index or metric not found for relevance model", i, "\n")  # Debug print
      }
    }
  }
  
  # Print the result
  if (!is.null(best_model_index)) {
    cat("Best performing model for metric:", target_metric_name_best, "is SONN",
        best_model_index, "with value:", best_performance_value, "\n")
  } else {
    cat("No model found for metric:", target_metric_name_best, "\n")
  }
  
  return(list(best_model_index = best_model_index, target_metric_name_best = target_metric_name_best))
}

find_and_print_worst_performing_models <- function(performance_names, relevance_names, performance_metrics, relevance_metrics, target_metric_name_worst, ensemble_number) {
  # Initialize an empty list to hold all metric names
  all_metric_names <- union(unlist(performance_names), unlist(relevance_names))
  
  # Check if the target metric exists in the list of all metrics
  if (!(target_metric_name_worst %in% all_metric_names)) {
    cat("Target metric", target_metric_name_worst, "not found.\n")
    return(NULL)
  }
  
  # Determine if the target metric should be minimized or maximized
  minimize_metric <- target_metric_name_worst %in% c("speed_learn", "speed", "memory_usage", "quantization_error", "topographic_error", "clustering_quality_db", "MSE", "MAE", "RMSE", "serendipity", "diversity", "hit_rate")
  maximize_metric <- target_metric_name_worst %in% c("precision", "recall", "f1_score", "ndcg", "mean_precision", "robustness", "generalization_ability")
  
  # Initialize the worst performance value appropriately
  if (minimize_metric) {
    worst_performance_value <- -Inf
  } else if (maximize_metric) {
    worst_performance_value <- Inf
  } else {
    worst_performance_value <- Inf  # Default to minimizing if not listed
  }
  
  worst_model_index <- NULL
  
  # Find the worst-performing model for the target metric in performance_metrics
  for (i in seq_along(performance_metrics)) {
    if (target_metric_name_worst %in% performance_names[[i]]) {
      metric_index <- match(target_metric_name_worst, performance_names[[i]])
      cat("Checking performance model", i, "for metric", target_metric_name_worst, "with metric index", metric_index, "\n")  # Debug print
      
      if (!is.na(metric_index) && metric_index <= length(performance_metrics[[i]])) {
        metric_value <- performance_metrics[[i]][[metric_index]]
        cat("Metric value for performance model", i, "is", toString(metric_value), "\n")  # Debug print
        
        if (is.numeric(metric_value) && !is.na(metric_value)) {
          if ((minimize_metric && metric_value > worst_performance_value) ||
              (maximize_metric && metric_value < worst_performance_value)) {
            worst_performance_value <- metric_value
            worst_model_index <- i
            cat("New worst model index", worst_model_index, "with value", worst_performance_value, "\n")  # Debug print
          }
        }
      } else {
        cat("Invalid metric index or metric not found for performance model", i, "\n")  # Debug print
      }
    }
  }
  
  
  # Find the worst-performing model for the target metric in relevance_metrics
  for (i in seq_along(relevance_metrics)) {
    if (target_metric_name_worst %in% relevance_names[[i]]) {
      metric_index <- match(target_metric_name_worst, relevance_names[[i]])
      cat("Checking relevance model", i, "for metric", target_metric_name_worst, "with metric index", metric_index, "\n")  # Debug print
      if (!is.na(metric_index) && metric_index <= length(relevance_metrics[[i]])) {
        metric_value <- relevance_metrics[[i]][[metric_index]]
        cat("Metric value for relevance model", i, "is", metric_value, "\n")  # Debug print
        if (is.numeric(metric_value) && !is.na(metric_value)) {
          if ((minimize_metric && metric_value > worst_performance_value) ||
              (maximize_metric && metric_value < worst_performance_value)) {
            worst_performance_value <- metric_value
            worst_model_index <- i + length(performance_metrics)  # Adjust index for relevance metrics
            cat("New worst model index", worst_model_index, "with value", worst_performance_value, "\n")  # Debug print
          }
        }
      } else {
        cat("Invalid metric index or metric not found for relevance model", i, "\n")  # Debug print
      }
    }
  }
  
  # Print the result
  if (!is.null(worst_model_index)) {
    cat("Worst performing model for metric:", target_metric_name_worst, "is DESONN", ensemble_number, "SONN",
        worst_model_index, "with value:", worst_performance_value, "\n")
  } else {
    cat("No model found for metric:", target_metric_name_worst, "\n")
  }
  
  return(worst_model_index)
}

add_network_to_ensemble <- function(ensembles,
                                    target_metric_name_best,
                                    removed_network,
                                    ensemble_number) {
  
  target_metric_name_best <- if (is.character(target_metric_name_best) && length(target_metric_name_best) == 1) {
    target_metric_name_best
  } else {
    deparse(substitute(target_metric_name_best))
  }
  
  `%||%` <- function(a,b) if (is.null(a) || length(a)==0 || (length(a)==1 && is.na(a))) b else a
  
  # --- figure out which slot to replace (as it used to be implicit) ---
  worst_model_index <- attr(removed_network, "slot_index") %||% NA_integer_
  
  # Extract performance and relevance metrics
  performance_metrics <- lapply(ensembles$temp_ensemble, function(x) x$performance_metric)
  relevance_metrics   <- lapply(ensembles$temp_ensemble, function(x) x$relevance_metric)
  
  performance_names <- lapply(performance_metrics, names)
  relevance_names   <- lapply(relevance_metrics,   names)
  
  # helper from your code
  extract_and_combine_metrics <- function(metrics){
    combined_data <- lapply(seq_along(metrics), function(i) {
      data <- metrics[[i]]
      flattened_data <- purrr::map2(data, names(data), function(metric, name) {
        if (length(metric) > 1) {
          rlang::set_names(as.list(metric), paste0(name, "_", seq_along(metric)))
        } else {
          rlang::set_names(list(metric), name)
        }
      }) %>% purrr::flatten()
      return(flattened_data)
    })
    return(combined_data)
  }
  
  # fixed_relevance
  relevance_metrics <- extract_and_combine_metrics(relevance_metrics)
  
  # Extract optimal epochs etc.
  extract_optimal_epoch <- lapply(ensembles$temp_ensemble, function(x) x$optimal_epoch)
  runid             <- unlist(lapply(ensembles$temp_ensemble, function(x) x$run_id))
  model_iter_num_id <- unlist(lapply(ensembles$temp_ensemble, function(x) x$model_iter_num))
  loss_increase_flag <- unlist(lapply(ensembles$temp_ensemble, function(x) x$loss_increase_flag))
  
  # Data frames
  df_performance_metrics <- do.call(rbind, lapply(performance_metrics, data.frame))
  df_performance_metrics$runid <- runid
  df_performance_metrics$model_index <- model_iter_num_id
  df_performance_metrics$loss_increase_flag <- loss_increase_flag
  df_performance_metrics$Optimal_Epochs <- extract_optimal_epoch
  df_performance_metrics <- tidyr::gather(df_performance_metrics, key = "Metric", value = "Value",
                                          -runid, -model_index, -Optimal_Epochs, -loss_increase_flag)
  
  df_relevance_metrics <- do.call(rbind, lapply(relevance_metrics, data.frame))
  df_relevance_metrics$runid <- runid
  df_relevance_metrics$model_index <- model_iter_num_id
  df_relevance_metrics$loss_increase_flag <- loss_increase_flag
  df_relevance_metrics$Optimal_Epochs <- extract_optimal_epoch
  df_relevance_metrics <- tidyr::gather(df_relevance_metrics, key = "Metric", value = "Value",
                                        -runid, -model_index, -Optimal_Epochs, -loss_increase_flag)
  
  df_metrics <<- rbind(df_performance_metrics, df_relevance_metrics)
  
  # --- your original model selection ---
  best_performing_models <<- find_and_print_best_performing_models(
    performance_names, relevance_names, performance_metrics, relevance_metrics, target_metric_name_best
  )
  
  metric_to_vlookup <- best_performing_models$target_metric_name_best
  best_model_index  <<- best_performing_models$best_model_index
  
  if (is.numeric(best_model_index) && (!is.infinite(best_model_index) || !is.na(best_model_index) || !is.null(best_model_index))) {
    result <<- df_metrics %>%
      dplyr::filter(Metric == metric_to_vlookup,
                    model_index == best_model_index,
                    Optimal_Epochs > 1,
                    loss_increase_flag == FALSE) %>%
      dplyr::summarise(best_model_index = dplyr::first(model_index))
    
    target_metric_name_best_value <<- df_metrics %>%
      dplyr::filter(Metric == metric_to_vlookup, model_index == best_model_index) %>%
      dplyr::summarise(Value = dplyr::first(Value)) %>%
      dplyr::pull(Value)
    
    optimal_epoch <<- df_metrics %>%
      dplyr::filter(Metric == "Optimal_Epochs", model_index == best_model_index) %>%
      dplyr::summarise(Value = dplyr::first(Value)) %>%
      dplyr::pull(Value)
    
    filtered_df <- df_metrics %>%
      dplyr::filter(Metric == metric_to_vlookup, model_index == best_model_index)
    
    loss_increase_flag_value <- filtered_df$loss_increase_flag
    
    if ((nrow(result) > 0 && !is.na(result)) ||
        (nrow(result) > 0 && !is.null(result)) &&
        (!is.null(optimal_epoch) || optimal_epoch > 1 || !is.na(optimal_epoch)) &&
        loss_increase_flag_value == FALSE) {
      best_model_index_new <<-  result$best_model_index
    } else {
      best_model_index_new <- NULL
      print("Optimal Epoch is <= 1 or does not exist and/or losses exceed initial loss.")
    }
  } else {
    best_model_index_new <- NULL
    result <<- NA
    target_metric_name_best_value <<- NA
  }
  
  best_model_metadata <- list()
  
  # if a best-performing model is found (old path)
  if (!is.null(best_model_index_new)) {
    best_model <- ensembles$temp_ensemble[[best_model_index]]
    best_model$ensemble_number <- best_model$ensemble_number + 1
    
    best_model_metadata <- list(
      ensemble_index = ensemble_number + 1,
      model_index = best_model_index,
      input_size = best_model$input_size,
      output_size = best_model$output_size,
      N = best_model$N,
      never_ran_flag = best_model$never_ran_flag,
      num_samples = best_model$num_samples,
      num_test_samples = best_model$num_test_samples,
      num_training_samples = best_model$num_training_samples,
      num_validation_samples = best_model$num_validation_samples,
      num_networks = best_model$num_networks,
      update_weights = best_model$update_weights,
      update_biases = best_model$update_biases,
      lr = best_model$lr,
      lambda = best_model$lambda,
      num_epochs = best_model$num_epochs,
      optimal_epoch = best_model$optimal_epoch,
      run_id = best_model$run_id,
      model_iter_num = best_model$model_iter_num,
      ensemble_number = ensemble_number,
      threshold = best_model$threshold,
      predicted_output = best_model$predicted_output,
      predicted_output_tail = best_model$predicted_output_tail,
      actual_values_tail = best_model$actual_values_tail,
      differences = best_model$differences,
      summary_stats = best_model$summary_stats,
      boxplot_stats = best_model$boxplot_stats,
      X = best_model$X,
      y = best_model$y,
      best_weights_record = best_model$best_weights_record,
      best_biases_record = best_model$best_biases_record,
      weights_record2 = best_model$weights_record2,
      biases_record2 = best_model$biases_record2,
      lossesatoptimalepoch = best_model$lossesatoptimalepoch,
      loss_increase_flag = best_model$loss_increase_flag,
      performance_metric = best_model$performance_metric,
      relevance_metric = best_model$relevance_metric
    )
    
    cat("Adding model from temp ensemble to main ensemble based on metric:", target_metric_name_best, "\n")
    
    # prepare removed network metrics for comparison
    removed_network_performance_metric <- list(removed_network$performance_metric)
    removed_network_relevance_metric   <- extract_and_combine_metrics(list(removed_network$relevance_metric))
    
    removed_network_df_performance_metrics <- do.call(rbind, lapply(removed_network_performance_metric, data.frame)) %>%
      tidyr::gather(key = "Metric", value = "Value")
    removed_network_df_relevance_metrics   <- do.call(rbind, lapply(removed_network_relevance_metric, data.frame)) %>%
      tidyr::gather(key = "Metric", value = "Value")
    
    df_removed_network_performance_metrics_relevance_metrics <- rbind(
      removed_network_df_performance_metrics,
      removed_network_df_relevance_metrics
    )
    
    removed_network_metric <- df_removed_network_performance_metrics_relevance_metrics %>%
      dplyr::filter(Metric == metric_to_vlookup) %>%
      dplyr::summarise(Value = dplyr::first(Value)) %>%
      dplyr::pull(Value)
    
    # -------- STRICT METRIC OVERRIDE (matches how it used to behave) --------
    override_when_temp_better <- TRUE
    if (isTRUE(override_when_temp_better) &&
        is.finite(target_metric_name_best_value) &&
        is.finite(removed_network_metric)) {
      
      # get slot if not set
      if (is.na(worst_model_index)) {
        # try to find the slot by matching removed serial against current main
        removed_serial <- removed_network$performance_metric$model_serial_num %||%
          removed_network$model_serial_num %||% NA_character_
        if (is.character(removed_serial) && nzchar(removed_serial)) {
          main_serials <- vapply(seq_along(ensembles$main_ensemble), function(i) {
            mi <- ensembles$main_ensemble[[i]]
            mi$performance_metric$model_serial_num %||% mi$model_serial_num %||% NA_character_
          }, character(1))
          hit <- which(main_serials == removed_serial)
          if (length(hit)) worst_model_index <- hit[1] else worst_model_index <- length(ensembles$main_ensemble)
        } else {
          worst_model_index <- length(ensembles$main_ensemble)
        }
      }
      
      # Minimization by default for error metrics like MSE
      # If you're maximizing, flip the sign here or compare accordingly.
      if (target_metric_name_best_value < removed_network_metric) {
        cat(sprintf("[OVERRIDE] %s: temp %.6f < main %.6f -> replacing slot %d\n",
                    metric_to_vlookup, target_metric_name_best_value, removed_network_metric, worst_model_index))
        
        # perform replacement
        ensembles$main_ensemble[[worst_model_index]] <- best_model
        
        # keep old metadata layered
        add_metadata_layer <- function(model, new_metadata) {
          if (!is.null(model$metadata)) {
            iteration <- 2
            while (!is.null(model[[paste0("metadata", iteration)]])) iteration <- iteration + 1
            model[[paste0("metadata", iteration)]] <- new_metadata
          } else {
            model$metadata <- new_metadata
          }
          model
        }
        updated_model <- add_metadata_layer(ensembles$main_ensemble[[worst_model_index]], removed_network)
        ensembles$main_ensemble[[worst_model_index]] <<- updated_model
        
        return(list(
          updated_ensemble = ensembles$main_ensemble,
          added_network    = list(
            serial = best_model$performance_metric$model_serial_num %||% best_model$model_serial_num %||% NA_character_,
            metric_value = target_metric_name_best_value
          ),
          removed_network  = list(
            serial = removed_network$performance_metric$model_serial_num %||% removed_network$model_serial_num %||% NA_character_,
            metric_value = removed_network_metric
          ),
          worst_model_index = worst_model_index
        ))
      }
    }
    
    # ----------------- ORIGINAL PATH (unchanged) -----------------
    if (!is.null(removed_network) &&
        target_metric_name_best_value < removed_network_metric &&
        nrow(result) > 0 && !is.na(result)) {
      
      print(paste("Index of the removed network:", worst_model_index))
      
      run_result_var_name <- paste0("run_results_1_", worst_model_index)
      best_model_with_removed_network_metadata <- list(
        best_model_metadata = best_model,
        metadata = removed_network
      )
      assign(run_result_var_name, best_model_with_removed_network_metadata, envir = .GlobalEnv)
      
      ensembles$main_ensemble[[worst_model_index]] <- best_model
      
      add_metadata_layer <- function(model, new_metadata) {
        if (!is.null(model$metadata)) {
          iteration <- 2
          while (!is.null(model[[paste0("metadata", iteration)]])) iteration <- iteration + 1
          model[[paste0("metadata", iteration)]] <- new_metadata
        } else {
          model$metadata <- new_metadata
        }
        model
      }
      updated_model <- add_metadata_layer(ensembles$main_ensemble[[worst_model_index]], removed_network)
      ensembles$main_ensemble[[worst_model_index]] <<- updated_model
      
    } else if (!is.null(removed_network) &&
               target_metric_name_best_value > removed_network_metric &&
               (nrow(result) <= 0 || is.na(result))) {
      ensembles$main_ensemble <- append(ensembles$main_ensemble, list(removed_network),
                                        after = max(1, (worst_model_index %||% length(ensembles$main_ensemble)) - 1))
      cat("Temp Ensemble based on metric:", target_metric_name_best, "with a value of:",
          target_metric_name_best_value, "is worse than Main Ensemble's", target_metric_name_best,
          "with a value of:", removed_network_metric, "\n")
    }
  } else {
    cat("No best model found in the temp ensemble based on metric:", target_metric_name_best, "\n")
  }
  
  # Keep temp_ensemble ready
  ensembles$temp_ensemble <<- ensembles$temp_ensemble
  
  # Return a shape compatible with your caller
  return(list(
    updated_ensemble = ensembles$main_ensemble,
    # these two are NA if no swap happened (safe for your logging code)
    added_network    = list(serial = NA_character_, metric_value = NA_real_),
    removed_network  = list(serial = NA_character_, metric_value = NA_real_),
    worst_model_index = worst_model_index %||% NA_integer_
  ))
}

prune_network_from_ensemble <- function(ensembles, target_metric_name_worst) {
  
  # normalize metric name if called with bare symbol (e.g., MSE)
  target_metric_name_worst <- if (is.character(target_metric_name_worst) && length(target_metric_name_worst) == 1) {
    target_metric_name_worst
  } else {
    deparse(substitute(target_metric_name_worst))
  }
  
  
  # Initialize lists to store performance and relevance metrics
  performance_metrics <- list()
  relevance_metrics <- list()
  performance_names <- list()
  relevance_names <- list()
  
  # Initialize lists to store performance and relevance metrics
  performance_metrics <- list()
  relevance_metrics <- list()
  
  # Loop over each ensemble
  for (i in seq_along(ensembles$main_ensemble)) {
    # Check if best_model_metadata exists in the i-th ensemble
    if (!is.null(ensembles$main_ensemble[[i]]$best_model_metadata)) {
      # Extract performance metric from best_model_metadata
      performance_metrics[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$performance_metric
      # Extract relevance metric from best_model_metadata
      relevance_metrics[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$relevance_metric
    } else {
      # Extract performance metric from the i-th ensemble
      performance_metrics[[i]] <- ensembles$main_ensemble[[i]]$performance_metric
      # Extract relevance metric from the i-th ensemble
      relevance_metrics[[i]] <- ensembles$main_ensemble[[i]]$relevance_metric
    }
  }
  
  performance_metrics_review <<- performance_metrics
  relevance_metrics_review <<- relevance_metrics
  
  # Extract metric names
  performance_names <- lapply(performance_metrics, names)
  relevance_names <- lapply(relevance_metrics, names)
  
  # Find the worst performing model
  worst_model_index <- find_and_print_worst_performing_models(performance_names, relevance_names, performance_metrics, relevance_metrics, target_metric_name_worst, ensemble_number = j)
  
  # # Initialize lists to store run_ids and MSE performance metrics
  # run_ids <- list()
  # mse_metrics <- list()
  
  # # Loop over each ensemble
  # for (i in seq_along(ensembles$main_ensemble)) {
  #     # Check if best_model_metadata exists in the i-th ensemble
  #     if (!is.null(ensembles$main_ensemble[[i]]$best_model_metadata)) {
  #         # Extract run_id and MSE performance metric from best_model_metadata
  #         run_ids[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$run_id
  #         mse_metrics[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$performance_metric$MSE
  #     } else {
  #         # Extract run_id and MSE performance metric from the i-th ensemble
  #         run_ids[[i]] <- ensembles$main_ensemble[[i]]$run_id
  #         mse_metrics[[i]] <- ensembles$main_ensemble[[i]]$performance_metric$MSE
  #     }
  # }
  #
  # # Print the run_ids and MSE performance metrics
  # print("_____________run_ids________________________________")
  # print(run_ids)
  # print("_____________mse_metrics________________________________")
  # print(mse_metrics)
  
  
  
  # If a worst-performing model is found
  if (!is.null(worst_model_index)) {
    # Remove the worst-performing model from the main ensemble
    removed_network <- ensembles$main_ensemble[[worst_model_index]]
    attr(removed_network, "slot_index") <- worst_model_index
    
    ensembles$main_ensemble <- ensembles$main_ensemble[-worst_model_index]
    
    # Print information about the removed network
    cat("Removed network", worst_model_index, "from the main ensemble based on worst", target_metric_name_worst, "metric\n")
    
    return(list(removed_network = removed_network, updated_ensemble = ensembles, worst_model_index = worst_model_index))
  } else {
    cat("No worst-performing model found for metric:", target_metric_name_worst, "\n")
    return(NULL)
  }
}
