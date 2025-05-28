apply_optimizer_update <- function(optimizer, optimizer_params, grads_matrix, lr, beta1, beta2, epsilon, epoch, self, layer, target) {  
  if (layer == 1 && epoch %% 5 == 0) cat("[OPT] Epoch", epoch, "Layer", layer, "\n")
  if (optimizer == "adam") {
    # === Step 1: Get param & grads ===
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    grad_matrix  <- grads_matrix
    
    # === Step 2: Validate ===
    if (is.null(param_matrix) || is.null(grad_matrix)) {
      cat("[Adam] Skipping layer", layer, "- param or grad is NULL.\n")
      return(list(
        updated_param = param_matrix,
        updated_optimizer_params = optimizer_params
      ))
    }
    
    param_matrix <- as.matrix(param_matrix)
    grad_matrix  <- as.matrix(grad_matrix)
    
    # === ⚠️ Compress large gradients for bias updates ===
    if (target == "bias" && nrow(grad_matrix) > 1) {
      cat("[Adam] Compressing bias gradient using colMeans for layer", layer, "\n")
      grad_matrix <- matrix(colMeans(grad_matrix), nrow = 1)
    }
    
    # === ⏱ Start timing ===
    start <- Sys.time()
    
    # === Step 3: Adam update ===
    adam_result <- adam_update(
      optimizer_params[[layer]],
      grad_matrix,
      lr = lr,
      beta1 = beta1,
      beta2 = beta2,
      epsilon = epsilon,
      t = epoch
    )
    
    updated_param <- adam_result$weights_update
    if (is.null(updated_param)) {
      updated_param <- matrix(0, nrow = nrow(param_matrix), ncol = ncol(param_matrix))
    } else if (is.list(updated_param)) {
      updated_param <- matrix(unlist(updated_param), nrow = nrow(param_matrix), ncol = ncol(param_matrix))
    } else {
      updated_param <- matrix(updated_param, nrow = nrow(param_matrix), ncol = ncol(param_matrix))
    }
    
    # === Step 4: Subtract ===
    param_matrix <- param_matrix - updated_param
    
    # === ⏱ End timing ===
    cat("[TIME] Epoch", epoch, "Layer", layer, "-", target, "update took", Sys.time() - start, "\n")
    
    # === Step 5: Save ===
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
    
    optimizer_params[[layer]] <- adam_result
  }
  
  
  
  else if (optimizer == "rmsprop") {
    # Select target matrix
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Update optimizer parameters
    optimizer_params[[layer]] <- rmsprop_update(optimizer_params[[layer]], grads_matrix, lr, beta2, epsilon)
    
    # Extract updates
    updated_param <- optimizer_params[[layer]]$updates
    if (is.list(updated_param)) updated_param <- unlist(updated_param)
    
    # Perform update
    if (length(updated_param) == prod(dim(param_matrix))) {
      cat("Dimensions match exactly. Performing subtraction.\n")
      param_matrix <- param_matrix - matrix(updated_param, nrow = nrow(param_matrix), byrow = TRUE)
    } else if (prod(dim(param_matrix)) == 1) {
      cat("Handling scalar param update.\n")
      update_value <- sum(updated_param)
      param_matrix <- param_matrix - update_value
    } else {
      cat("Dimensions do not match exactly. Adjusting dimensions for subtraction.\n")
      repeated_updated_param <- rep(updated_param, length.out = prod(dim(param_matrix)))
      param_matrix <- param_matrix - matrix(repeated_updated_param, nrow = nrow(param_matrix), byrow = TRUE)
    }
    
    # Assign result
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  else if (optimizer == "sgd") {
    # Select target matrix
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Perform SGD update
    optimizer_params[[layer]] <- sgd_update(optimizer_params[[layer]], grads_matrix, lr)
    updated_param <- optimizer_params[[layer]]$weights_update
    
    if (is.null(updated_param)) stop("Updated parameter update is NULL.")
    if (is.list(updated_param)) updated_param <- unlist(updated_param)
    
    # Matrix case
    if (is.matrix(updated_param)) {
      if (identical(dim(param_matrix), dim(updated_param))) {
        param_matrix <- param_matrix - updated_param
      } else {
        # Adjust dimensions row-wise
        if (nrow(param_matrix) != nrow(updated_param)) {
          if (nrow(param_matrix) > nrow(updated_param)) {
            updated_param <- rbind(updated_param, matrix(0, nrow = nrow(param_matrix) - nrow(updated_param), ncol = ncol(updated_param)))
          } else {
            updated_param <- updated_param[1:nrow(param_matrix), , drop = FALSE]
          }
        }
        # Adjust dimensions column-wise
        if (ncol(param_matrix) != ncol(updated_param)) {
          if (ncol(param_matrix) > ncol(updated_param)) {
            updated_param <- cbind(updated_param, matrix(0, nrow = nrow(updated_param), ncol = ncol(param_matrix) - ncol(updated_param)))
          } else {
            updated_param <- updated_param[, 1:ncol(param_matrix), drop = FALSE]
          }
        }
        param_matrix <- param_matrix - updated_param
      }
      
    } else if (is.vector(updated_param)) {
      # Vector case
      if (length(updated_param) == length(param_matrix)) {
        param_matrix <- param_matrix - updated_param
      } else {
        updated_param <- rep(updated_param, length.out = length(param_matrix))
        param_matrix <- param_matrix - updated_param
      }
      
    } else if (prod(dim(param_matrix)) == 1) {
      # Scalar case
      cat("Handling scalar update.\n")
      update_value <- sum(updated_param)
      param_matrix <- param_matrix - update_value
      
    } else {
      stop("Unable to adjust dimensions for updated_param.")
    }
    
    # Write updated param back
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  
  else if (optimizer == "sgd_momentum") {
    # Select target matrix
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Perform SGD momentum update
    optimizer_params[[layer]] <- sgd_momentum_update(optimizer_params[[layer]], grads_matrix, lr)
    updated_param <- optimizer_params[[layer]]$weights_update
    
    if (is.null(updated_param)) stop("Updated parameter update is NULL.")
    if (is.list(updated_param)) updated_param <- unlist(updated_param)
    
    if (is.matrix(updated_param)) {
      if (identical(dim(param_matrix), dim(updated_param))) {
        param_matrix <- param_matrix - updated_param
      } else {
        # Adjust row dimensions
        if (nrow(param_matrix) != nrow(updated_param)) {
          if (nrow(param_matrix) > nrow(updated_param)) {
            updated_param <- rbind(updated_param, matrix(0, nrow = nrow(param_matrix) - nrow(updated_param), ncol = ncol(updated_param)))
          } else {
            updated_param <- updated_param[1:nrow(param_matrix), , drop = FALSE]
          }
        }
        # Adjust column dimensions
        if (ncol(param_matrix) != ncol(updated_param)) {
          if (ncol(param_matrix) > ncol(updated_param)) {
            updated_param <- cbind(updated_param, matrix(0, nrow = nrow(updated_param), ncol = ncol(param_matrix) - ncol(updated_param)))
          } else {
            updated_param <- updated_param[, 1:ncol(param_matrix), drop = FALSE]
          }
        }
        param_matrix <- param_matrix - updated_param
      }
      
    } else if (is.vector(updated_param)) {
      if (length(updated_param) == length(param_matrix)) {
        param_matrix <- param_matrix - updated_param
      } else {
        updated_param <- rep(updated_param, length.out = length(param_matrix))
        param_matrix <- param_matrix - updated_param
      }
      
    } else if (prod(dim(param_matrix)) == 1) {
      cat("Handling scalar update.\n")
      update_value <- sum(updated_param)
      param_matrix <- param_matrix - update_value
      
    } else {
      stop("Unable to adjust dimensions for updated_param.")
    }
    
    # Commit update
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  else if (optimizer == "nag") {
    # Select parameter based on target
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Perform NAG update
    optimizer_params[[layer]] <- nag_update(optimizer_params[[layer]], grads_matrix, lr, beta = 0.9)
    updated_param <- optimizer_params[[layer]]$weights_update
    
    if (is.null(updated_param)) stop("Updated weights update is NULL.")
    if (is.list(updated_param)) updated_param <- unlist(updated_param)
    
    if (is.matrix(updated_param)) {
      if (identical(dim(param_matrix), dim(updated_param))) {
        param_matrix <- param_matrix - updated_param
      } else {
        if (nrow(param_matrix) != nrow(updated_param)) {
          if (nrow(param_matrix) > nrow(updated_param)) {
            updated_param <- rbind(updated_param, matrix(0, nrow = nrow(param_matrix) - nrow(updated_param), ncol = ncol(updated_param)))
          } else {
            updated_param <- updated_param[1:nrow(param_matrix), , drop = FALSE]
          }
        }
        if (ncol(param_matrix) != ncol(updated_param)) {
          if (ncol(param_matrix) > ncol(updated_param)) {
            updated_param <- cbind(updated_param, matrix(0, nrow = nrow(updated_param), ncol = ncol(param_matrix) - ncol(updated_param)))
          } else {
            updated_param <- updated_param[, 1:ncol(param_matrix), drop = FALSE]
          }
        }
        param_matrix <- param_matrix - updated_param
      }
      
    } else if (is.vector(updated_param)) {
      if (length(updated_param) == length(param_matrix)) {
        param_matrix <- param_matrix - updated_param
      } else {
        updated_param <- rep(updated_param, length.out = length(param_matrix))
        param_matrix <- param_matrix - updated_param
      }
      
    } else if (prod(dim(param_matrix)) == 1) {
      cat("Handling scalar weight update.\n")
      update_value <- sum(updated_param)
      param_matrix <- param_matrix - update_value
      
    } else {
      stop("Unable to adjust dimensions for updated_param.")
    }
    
    # Commit the updated parameter
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  else if (optimizer == "ftrl") {
    # Select parameter matrix based on target
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Generate dummy gradients from current parameter (as per your original design)
    grads_list <- lapply(param_matrix, function(weight) {
      if (is.null(dim(weight))) {
        matrix(weight, nrow = length(weight), ncol = 1)
      } else {
        weight
      }
    })
    
    # Perform FTRL update
    ftrl_results <- ftrl_update(optimizer_params[[layer]], grads_list, lr, alpha = 0.1, beta = 1.0, lambda1 = 0.01, lambda2 = 0.01)
    optimizer_params[[layer]] <- ftrl_results$params
    updated_param <- ftrl_results$weights_update
    
    if (is.null(updated_param)) stop("Updated weights update is NULL.")
    if (is.list(updated_param)) updated_param <- unlist(updated_param)
    
    if (is.matrix(updated_param)) {
      if (identical(dim(param_matrix), dim(updated_param))) {
        param_matrix <- param_matrix - updated_param
      } else {
        if (nrow(param_matrix) != nrow(updated_param)) {
          if (nrow(param_matrix) > nrow(updated_param)) {
            updated_param <- rbind(updated_param, matrix(0, nrow = nrow(param_matrix) - nrow(updated_param), ncol = ncol(updated_param)))
          } else {
            updated_param <- updated_param[1:nrow(param_matrix), , drop = FALSE]
          }
        }
        if (ncol(param_matrix) != ncol(updated_param)) {
          if (ncol(param_matrix) > ncol(updated_param)) {
            updated_param <- cbind(updated_param, matrix(0, nrow = nrow(updated_param), ncol = ncol(param_matrix) - ncol(updated_param)))
          } else {
            updated_param <- updated_param[, 1:ncol(param_matrix), drop = FALSE]
          }
        }
        param_matrix <- param_matrix - updated_param
      }
    } else if (is.vector(updated_param)) {
      if (length(updated_param) == length(param_matrix)) {
        param_matrix <- param_matrix - updated_param
      } else {
        updated_param <- rep(updated_param, length.out = length(param_matrix))
        param_matrix <- param_matrix - updated_param
      }
    } else if (prod(dim(param_matrix)) == 1) {
      cat("Handling scalar weight update.\n")
      param_matrix <- param_matrix - sum(updated_param)
    } else {
      cat("Dimensions or type of updated_param are not suitable for subtraction.\n")
      if (is.vector(updated_param)) {
        repeated_updated_param <- rep(updated_param, length.out = length(param_matrix))
        param_matrix <- param_matrix - repeated_updated_param
      } else {
        cat("Unable to adjust dimensions for updated_param.\n")
      }
    }
    
    # Save updated matrix back to the model
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  else if (optimizer == "lamb") {
    # Choose param and grads source
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Generate dummy gradients (as per your original logic)
    grads_vector <- if (is.null(dim(param_matrix))) {
      matrix(runif(n = length(param_matrix)), nrow = length(param_matrix), ncol = 1)
    } else {
      matrix(runif(n = prod(dim(param_matrix))), nrow = nrow(param_matrix), ncol = ncol(param_matrix))
    }
    
    # Ensure optimizer params are numeric
    optimizer_params[[layer]]$param <- as.numeric(optimizer_params[[layer]]$param)
    optimizer_params[[layer]]$m     <- as.numeric(optimizer_params[[layer]]$m)
    optimizer_params[[layer]]$v     <- as.numeric(optimizer_params[[layer]]$v)
    
    # Perform LAMB update
    lamb_results <- lamb_update(
      optimizer_params[[layer]],
      as.numeric(grads_vector),
      lr,
      beta1 = 0.9,
      beta2 = 0.999,
      eps   = 1e-8,
      lambda = 0.01
    )
    
    # Save new optimizer state
    optimizer_params[[layer]] <- lamb_results$params
    updated_param <- lamb_results$update
    
    if (is.null(updated_param)) stop("Updated parameter from LAMB is NULL.")
    if (is.list(updated_param)) updated_param <- unlist(updated_param)
    
    # Subtract update
    if (is.matrix(updated_param)) {
      if (identical(dim(param_matrix), dim(updated_param))) {
        param_matrix <- param_matrix - updated_param
      } else {
        if (nrow(param_matrix) != nrow(updated_param)) {
          if (nrow(param_matrix) > nrow(updated_param)) {
            updated_param <- rbind(updated_param, matrix(0, nrow = nrow(param_matrix) - nrow(updated_param), ncol = ncol(updated_param)))
          } else {
            updated_param <- updated_param[1:nrow(param_matrix), , drop = FALSE]
          }
        }
        if (ncol(param_matrix) != ncol(updated_param)) {
          if (ncol(param_matrix) > ncol(updated_param)) {
            updated_param <- cbind(updated_param, matrix(0, nrow = nrow(updated_param), ncol = ncol(param_matrix) - ncol(updated_param)))
          } else {
            updated_param <- updated_param[, 1:ncol(param_matrix), drop = FALSE]
          }
        }
        param_matrix <- param_matrix - updated_param
      }
    } else if (is.vector(updated_param) && length(updated_param) == length(param_matrix)) {
      param_matrix <- param_matrix - updated_param
    } else {
      if (is.vector(updated_param)) {
        repeated_update <- rep(updated_param, length.out = length(param_matrix))
        param_matrix <- param_matrix - repeated_update
      } else {
        cat("Unable to adjust dimensions for LAMB update.\n")
      }
    }
    
    # Write back to model
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  else if (optimizer == "lookahead") {
    # Select matrix based on target
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Create gradient matrix matching the param's shape
    grad_matrix <- matrix(runif(n = length(param_matrix)), nrow = nrow(param_matrix), ncol = ncol(param_matrix))
    grads_list <- list(param = grad_matrix)
    
    # Extract relevant optimizer params
    current_params <- optimizer_params[[layer]]
    params <- list(
      param = param_matrix,
      m = current_params$m,
      v = current_params$v,
      r = current_params$r,
      slow_weights = current_params$slow_weights,
      lookahead_counter = current_params$lookahead_counter,
      lookahead_step = current_params$lookahead_step
    )
    
    # Perform Lookahead update
    lookahead_results <- lookahead_update(
      list(params),
      list(grads_list),
      lr,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 1e-8,
      lookahead_step = params$lookahead_step,
      base_optimizer = "adam_update",
      t = epoch,
      lambda = lambda
    )
    
    # Extract updated values
    updated_param <- lookahead_results$param
    new_m <- lookahead_results$m
    new_v <- lookahead_results$v
    new_r <- lookahead_results$r
    new_slow_weights <- lookahead_results$slow_weights
    new_lookahead_counter <- lookahead_results$lookahead_counter
    
    if (is.null(updated_param)) {
      cat("Updated param from Lookahead is NULL. Skipping update.\n")
    } else {
      # Handle updates
      if (is.matrix(updated_param) && all(dim(updated_param) == dim(param_matrix))) {
        param_matrix <- param_matrix - updated_param
      } else if (is.vector(updated_param) && length(updated_param) == length(param_matrix)) {
        param_matrix <- param_matrix - updated_param
      } else if (is.vector(updated_param)) {
        repeated_update <- matrix(rep(updated_param, length.out = length(param_matrix)), nrow = nrow(param_matrix), ncol = ncol(param_matrix))
        param_matrix <- param_matrix - repeated_update
      } else {
        cat("Unable to adjust dimensions for Lookahead update.\n")
      }
      
      # Write back updated parameters
      if (target == "weights") {
        self$weights[[layer]] <- param_matrix
      } else {
        self$biases[[layer]] <- param_matrix
      }
      
      # Save updated optimizer state
      optimizer_params[[layer]]$m <- new_m
      optimizer_params[[layer]]$v <- new_v
      optimizer_params[[layer]]$r <- new_r
      optimizer_params[[layer]]$slow_weights <- new_slow_weights
      optimizer_params[[layer]]$lookahead_counter <- new_lookahead_counter
    }
  }
  
  else if (optimizer == "adagrad") {
    # Select parameter matrix (weights or biases)
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Compute the gradients
    grads_matrix <- matrix(runif(n = length(param_matrix)), nrow = nrow(param_matrix), ncol = ncol(param_matrix))
    grads_vector <- as.numeric(grads_matrix)
    
    # Perform Adagrad update
    adagrad_results <- adagrad_update(optimizer_params[[layer]], grads_vector, lr)
    optimizer_params[[layer]] <- adagrad_results$params
    r_values <- adagrad_results$r
    
    # Ensure r_values is numeric
    if (is.list(r_values)) {
      r_values <- unlist(r_values)
    }
    
    # Calculate update
    updated_param <- grads_vector / (sqrt(r_values) + epsilon)
    if (is.list(updated_param)) {
      updated_param <- unlist(updated_param)
    }
    
    # Perform subtraction
    if (length(updated_param) == prod(dim(param_matrix))) {
      cat("Adagrad: Dimensions match exactly. Performing subtraction.\n")
      param_matrix <- param_matrix - matrix(updated_param, nrow = nrow(param_matrix), ncol = ncol(param_matrix), byrow = TRUE)
    } else {
      cat("Adagrad: Dimensions do not match. Adjusting.\n")
      repeated_update <- rep(updated_param, length.out = nrow(param_matrix) * ncol(param_matrix))
      param_matrix <- param_matrix - matrix(repeated_update, nrow = nrow(param_matrix), ncol = ncol(param_matrix), byrow = TRUE)
    }
    
    # Write back to self
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  else if (optimizer == "adadelta") {
    # Select the parameter matrix (weights or biases)
    param_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Compute the gradients (you can replace with actual gradient logic)
    grads_matrix <- matrix(runif(n = length(param_matrix)), nrow = nrow(param_matrix), ncol = ncol(param_matrix))
    grads_vector <- as.numeric(grads_matrix)
    
    # Perform Adadelta update
    adadelta_results <- adadelta_update(optimizer_params[[layer]], grads_vector, lr, epsilon)
    optimizer_params[[layer]] <- adadelta_results$params
    delta_w <- adadelta_results$delta_w
    
    # Ensure delta_w is numeric
    if (is.list(delta_w)) delta_w <- unlist(delta_w)
    if (!is.numeric(delta_w)) stop("delta_w contains non-numeric values. Please check the adadelta_update function.")
    
    # Calculate the update
    updated_param <- delta_w / (sqrt(delta_w) + epsilon)
    if (is.list(updated_param)) updated_param <- unlist(updated_param)
    
    # Apply the update to the correct parameter matrix
    if (length(updated_param) == prod(dim(param_matrix))) {
      cat("Adadelta: Dimensions match. Performing subtraction.\n")
      param_matrix <- param_matrix - matrix(updated_param, nrow = nrow(param_matrix), ncol = ncol(param_matrix), byrow = TRUE)
    } else {
      cat("Adadelta: Adjusting dimensions for subtraction.\n")
      repeated_update <- rep(updated_param, length.out = nrow(param_matrix) * ncol(param_matrix))
      param_matrix <- param_matrix - matrix(repeated_update, nrow = nrow(param_matrix), ncol = ncol(param_matrix), byrow = TRUE)
    }
    
    # Save result back to model
    if (target == "weights") {
      self$weights[[layer]] <- param_matrix
      # ✅ Keep this print line exactly as requested
      cat("After Adadelta update - Dimensions of self$weights[[layer]]:", dim(self$weights[[layer]]), "\n")
    } else {
      self$biases[[layer]] <- param_matrix
    }
  }
  
  
  return(list(
    updated_param = updated_param,
    updated_optimizer_params = optimizer_params
  ))
  
  
}