apply_optimizer_update <- function(optimizer, optimizer_params, grads_matrix, lr, beta1, beta2, epsilon, epoch, self, layer, target) {
  
  if (optimizer == "adam") {
    cat(">> Optimizer = adam\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    # Boost learning rate for output layer
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    # Update optimizer params using Adam
    optimizer_params[[layer]] <- adam_update(
      optimizer_params[[layer]],
      grads = list(grads_matrix),
      lr = lr * layer_boost,
      beta1 = beta1,
      beta2 = beta2,
      epsilon = epsilon,
      t = epoch
    )
    
    # Select correct update matrix based on target
    if (target == "weights") {
      update <- optimizer_params[[layer]]$weights_update
    } else if (target == "biases") {
      update <- optimizer_params[[layer]]$biases_update
    } else {
      stop("Unknown target: must be 'weights' or 'biases'")
    }
    
    # Fix: allow both SL NN (update is list of 1 matrix) and ML NN (direct matrix)
    update_matrix <- if (is.list(update) && length(update) == 1) update[[1]] else update
    
    if (is.null(update_matrix)) {
      stop(paste0("Update matrix for layer ", layer, " is NULL — check gradients or optimizer output."))
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    update_len <- length(update_matrix)
    
    if (is.null(target_dim)) {
      stop(paste0("Target matrix dimensions for layer ", layer, " are NULL."))
    }
    
    # ------------------------------------------
    #     SHAPE FIXES FOR WEIGHTS VS. BIASES
    # ------------------------------------------
    if (target == "biases") {
      if (length(update_matrix) == prod(target_dim)) {
        updated <- matrix(update_matrix, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update_matrix) == 1) {
        updated <- matrix(rep(update_matrix, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update_matrix, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (update_len == prod(target_dim)) {
        updated <- matrix(update_matrix, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update_matrix, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update_matrix, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # ✅ Clip weights after update if target is "weights"
      clip_threshold <- 5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    # ✅ Diagnostic: log stats after update
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated), 
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    # Assign updated weights or biases back to self using subtraction
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }



    
  else if (optimizer == "rmsprop") {
    # Update weights using RMSprop optimizer
    optimizer_params_weights[[layer]] <- rmsprop_update(optimizer_params_weights[[layer]], grads_matrix, lr, beta2, epsilon)
    
    # Accessing the returned values
    updated_weights_update <- optimizer_params_weights[[layer]]$updates
    
    # Convert updated_weights_update to a numeric vector if it's a list
    if (is.list(updated_weights_update)) {
      updated_weights_update <- unlist(updated_weights_update)
    }
    
    # Perform subtraction for weights if dimensions match exactly
    if (length(updated_weights_update) == prod(dim(self$weights[[layer]]))) {
      cat("Dimensions match exactly. Performing subtraction.\n")
      self$weights[[layer]] <- self$weights[[layer]] - matrix(updated_weights_update, nrow = nrow(self$weights[[layer]]), byrow = TRUE)
    } else if (prod(dim(self$weights[[layer]])) == 1) {
      # Handle scalar weight updates
      cat("Handling scalar weight update.\n")
      update_value <- sum(updated_weights_update)  # Assuming summing the updates is appropriate
      self$weights[[layer]] <- self$weights[[layer]] - update_value
    } else {
      cat("Dimensions do not match exactly. Adjusting dimensions for subtraction.\n")
      repeat_times <- ceiling(nrow(self$weights[[layer]]) * ncol(self$weights[[layer]]) / length(updated_weights_update))
      repeated_updated_weights_update <- rep(updated_weights_update, length.out = nrow(self$weights[[layer]]) * ncol(self$weights[[layer]]))
      self$weights[[layer]] <- self$weights[[layer]] - matrix(repeated_updated_weights_update, nrow = nrow(self$weights[[layer]]), byrow = TRUE)
    }
  }
  else if (optimizer == "sgd") {
    # Perform SGD update for the weights
    optimizer_params_weights[[layer]] <- sgd_update(optimizer_params_weights[[layer]], grads_matrix, lr)
    updated_weights_update <- optimizer_params_weights[[layer]]$weights_update
    
    # Ensure updated_weights_update is not NULL
    if (is.null(updated_weights_update)) {
      stop("Updated weights update is NULL.")
    }
    
    # Ensure updated_weights_update is a matrix or vector
    if (is.list(updated_weights_update)) {
      updated_weights_update <- unlist(updated_weights_update)
    }
    
    # Check the type of updated_weights_update
    if (is.matrix(updated_weights_update)) {
      # If updated_weights_update is a matrix
      if (identical(dim(self$weights[[layer]]), dim(updated_weights_update))) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Adjust dimensions to match if necessary
        if (nrow(self$weights[[layer]]) != nrow(updated_weights_update)) {
          if (nrow(self$weights[[layer]]) > nrow(updated_weights_update)) {
            updated_weights_update <- rbind(updated_weights_update, matrix(0, nrow = nrow(self$weights[[layer]]) - nrow(updated_weights_update), ncol = ncol(updated_weights_update)))
          } else if (nrow(self$weights[[layer]]) < nrow(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[1:nrow(self$weights[[layer]]), , drop = FALSE]
          }
        }
        if (ncol(self$weights[[layer]]) != ncol(updated_weights_update)) {
          if (ncol(self$weights[[layer]]) > ncol(updated_weights_update)) {
            updated_weights_update <- cbind(updated_weights_update, matrix(0, nrow = nrow(updated_weights_update), ncol = ncol(self$weights[[layer]]) - ncol(updated_weights_update)))
          } else if (ncol(self$weights[[layer]]) < ncol(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[, 1:ncol(self$weights[[layer]]), drop = FALSE]
          }
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (is.vector(updated_weights_update)) {
      # If updated_weights_update is a vector
      if (length(updated_weights_update) == length(self$weights[[layer]])) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Repeat or truncate updated_weights_update to match self$weights[[layer]]
        if (length(updated_weights_update) > length(self$weights[[layer]])) {
          updated_weights_update <- updated_weights_update[1:length(self$weights[[layer]])]
        } else if (length(updated_weights_update) < length(self$weights[[layer]])) {
          updated_weights_update <- rep(updated_weights_update, length.out = length(self$weights[[layer]]))
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (prod(dim(self$weights[[layer]])) == 1) {
      # Handle scalar weight updates
      cat("Handling scalar weight update.\n")
      update_value <- sum(updated_weights_update)  # Assuming summing the updates is appropriate
      self$weights[[layer]] <- self$weights[[layer]] - update_value
    } else {
      stop("Unable to adjust dimensions for updated_weights_update.")
    }
  }
  
  else if (optimizer == "sgd_momentum") {
    # Perform SGD update for the weights
    optimizer_params_weights[[layer]] <- sgd_momentum_update(optimizer_params_weights[[layer]], grads_matrix, lr)
    updated_weights_update <- optimizer_params_weights[[layer]]$weights_update
    
    # Ensure updated_weights_update is not NULL
    if (is.null(updated_weights_update)) {
      stop("Updated weights update is NULL.")
    }
    
    # Ensure updated_weights_update is a matrix or vector
    if (is.list(updated_weights_update)) {
      updated_weights_update <- unlist(updated_weights_update)
    }
    
    # Check the type of updated_weights_update
    if (is.matrix(updated_weights_update)) {
      # If updated_weights_update is a matrix
      if (identical(dim(self$weights[[layer]]), dim(updated_weights_update))) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Adjust dimensions to match if necessary
        if (nrow(self$weights[[layer]]) != nrow(updated_weights_update)) {
          if (nrow(self$weights[[layer]]) > nrow(updated_weights_update)) {
            updated_weights_update <- rbind(updated_weights_update, matrix(0, nrow = nrow(self$weights[[layer]]) - nrow(updated_weights_update), ncol = ncol(updated_weights_update)))
          } else if (nrow(self$weights[[layer]]) < nrow(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[1:nrow(self$weights[[layer]]), , drop = FALSE]
          }
        }
        if (ncol(self$weights[[layer]]) != ncol(updated_weights_update)) {
          if (ncol(self$weights[[layer]]) > ncol(updated_weights_update)) {
            updated_weights_update <- cbind(updated_weights_update, matrix(0, nrow = nrow(updated_weights_update), ncol = ncol(self$weights[[layer]]) - ncol(updated_weights_update)))
          } else if (ncol(self$weights[[layer]]) < ncol(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[, 1:ncol(self$weights[[layer]]), drop = FALSE]
          }
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (is.vector(updated_weights_update)) {
      # If updated_weights_update is a vector
      if (length(updated_weights_update) == length(self$weights[[layer]])) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Repeat or truncate updated_weights_update to match self$weights[[layer]]
        if (length(updated_weights_update) > length(self$weights[[layer]])) {
          updated_weights_update <- updated_weights_update[1:length(self$weights[[layer]])]
        } else if (length(updated_weights_update) < length(self$weights[[layer]])) {
          updated_weights_update <- rep(updated_weights_update, length.out = length(self$weights[[layer]]))
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (prod(dim(self$weights[[layer]])) == 1) {
      # Handle scalar weight updates
      cat("Handling scalar weight update.\n")
      update_value <- sum(updated_weights_update)  # Assuming summing the updates is appropriate
      self$weights[[layer]] <- self$weights[[layer]] - update_value
    } else {
      stop("Unable to adjust dimensions for updated_weights_update.")
    }
  }
  
  
  else if (optimizer == "nag") {
    # Perform NAG update for the weights
    optimizer_params_weights[[layer]] <- nag_update(optimizer_params_weights[[layer]], grads_matrix, lr, beta = 0.9)
    updated_weights_update <- optimizer_params_weights[[layer]]$weights_update
    
    # Ensure updated_weights_update is not NULL
    if (is.null(updated_weights_update)) {
      stop("Updated weights update is NULL.")
    }
    
    # Ensure updated_weights_update is a matrix or vector
    if (is.list(updated_weights_update)) {
      updated_weights_update <- unlist(updated_weights_update)
    }
    
    # Check the type of updated_weights_update
    if (is.matrix(updated_weights_update)) {
      # If updated_weights_update is a matrix
      if (identical(dim(self$weights[[layer]]), dim(updated_weights_update))) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Adjust dimensions to match if necessary
        if (nrow(self$weights[[layer]]) != nrow(updated_weights_update)) {
          if (nrow(self$weights[[layer]]) > nrow(updated_weights_update)) {
            updated_weights_update <- rbind(updated_weights_update, matrix(0, nrow = nrow(self$weights[[layer]]) - nrow(updated_weights_update), ncol = ncol(updated_weights_update)))
          } else if (nrow(self$weights[[layer]]) < nrow(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[1:nrow(self$weights[[layer]]), , drop = FALSE]
          }
        }
        if (ncol(self$weights[[layer]]) != ncol(updated_weights_update)) {
          if (ncol(self$weights[[layer]]) > ncol(updated_weights_update)) {
            updated_weights_update <- cbind(updated_weights_update, matrix(0, nrow = nrow(updated_weights_update), ncol = ncol(self$weights[[layer]]) - ncol(updated_weights_update)))
          } else if (ncol(self$weights[[layer]]) < ncol(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[, 1:ncol(self$weights[[layer]]), drop = FALSE]
          }
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (is.vector(updated_weights_update)) {
      # If updated_weights_update is a vector
      if (length(updated_weights_update) == length(self$weights[[layer]])) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Repeat or truncate updated_weights_update to match self$weights[[layer]]
        if (length(updated_weights_update) > length(self$weights[[layer]])) {
          updated_weights_update <- updated_weights_update[1:length(self$weights[[layer]])]
        } else if (length(updated_weights_update) < length(self$weights[[layer]])) {
          updated_weights_update <- rep(updated_weights_update, length.out = length(self$weights[[layer]]))
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (prod(dim(self$weights[[layer]])) == 1) {
      # Handle scalar weight updates
      cat("Handling scalar weight update.\n")
      update_value <- sum(updated_weights_update)  # Assuming summing the updates is appropriate
      self$weights[[layer]] <- self$weights[[layer]] - update_value
    } else {
      stop("Unable to adjust dimensions for updated_weights_update.")
    }
  }
  else if (optimizer == "ftrl") {
    # Compute the gradients for weights if not already available
    grads_weights <- lapply(self$weights, function(weight) {
      if (is.null(dim(weight))) {
        # Convert vectors to matrices with a single column
        matrix(weight, nrow = length(weight), ncol = 1)
      } else {
        weight  # Already a matrix
      }
    })
    
    # Perform FTRL update for the weights
    ftrl_results <- ftrl_update(optimizer_params_weights[[layer]], grads_weights, lr, alpha = 0.1, beta = 1.0, lambda1 = 0.01, lambda2 = 0.01)
    optimizer_params_weights[[layer]] <- ftrl_results$params
    updated_weights_update <- ftrl_results$weights_update
    
    # Ensure updated_weights_update is not NULL
    if (is.null(updated_weights_update)) {
      stop("Updated weights update is NULL.")
    }
    
    # Ensure updated_weights_update is a matrix or vector
    if (is.list(updated_weights_update)) {
      updated_weights_update <- unlist(updated_weights_update)
    }
    
    # Check the type of updated_weights_update
    if (is.matrix(updated_weights_update)) {
      # If updated_weights_update is a matrix
      if (identical(dim(self$weights[[layer]]), dim(updated_weights_update))) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Adjust dimensions to match if necessary
        if (nrow(self$weights[[layer]]) != nrow(updated_weights_update)) {
          if (nrow(self$weights[[layer]]) > nrow(updated_weights_update)) {
            updated_weights_update <- rbind(updated_weights_update, matrix(0, nrow = nrow(self$weights[[layer]]) - nrow(updated_weights_update), ncol = ncol(updated_weights_update)))
          } else if (nrow(self$weights[[layer]]) < nrow(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[1:nrow(self$weights[[layer]]), , drop = FALSE]
          }
        }
        if (ncol(self$weights[[layer]]) != ncol(updated_weights_update)) {
          if (ncol(self$weights[[layer]]) > ncol(updated_weights_update)) {
            updated_weights_update <- cbind(updated_weights_update, matrix(0, nrow = nrow(updated_weights_update), ncol = ncol(self$weights[[layer]]) - ncol(updated_weights_update)))
          } else if (ncol(self$weights[[layer]]) < ncol(updated_weights_update)) {
            updated_weights_update <- updated_weights_update[, 1:ncol(self$weights[[layer]]), drop = FALSE]
          }
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (is.vector(updated_weights_update)) {
      # If updated_weights_update is a vector
      if (length(updated_weights_update) == length(self$weights[[layer]])) {
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        # Repeat or truncate updated_weights_update to match self$weights[[layer]]
        if (length(updated_weights_update) > length(self$weights[[layer]])) {
          updated_weights_update <- updated_weights_update[1:length(self$weights[[layer]])]
        } else if (length(updated_weights_update) < length(self$weights[[layer]]) && !prod(dim(self$weights[[layer]])) == 1) {
          updated_weights_update <- rep(updated_weights_update, length.out = length(self$weights[[layer]]))
        }
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      }
    } else if (prod(dim(self$weights[[layer]])) == 1) {
      # Handle scalar weight updates
      cat("Handling scalar weight update.\n")
      update_value <- sum(updated_weights_update)  # Assuming summing the updates is appropriate
      self$weights[[layer]] <- self$weights[[layer]] - update_value
    } else {
      cat("Dimensions or type of updated_weights_update are not suitable for subtraction.\n")
      cat("Attempting to adjust dimensions if possible.\n")
      
      # Attempting to adjust dimensions if updated_weights_update is not directly suitable
      if (is.vector(updated_weights_update)) {
        repeated_updated_weights_update <- rep(updated_weights_update, length.out = length(self$weights[[layer]]))
        self$weights[[layer]] <- self$weights[[layer]] - repeated_updated_weights_update
      } else {
        cat("Unable to adjust dimensions for updated_weights_update.\n")
      }
    }
  }
  else if (optimizer == "lamb") {
    # Compute the gradients for weights if not already available
    grads_weights <- lapply(self$weights, function(weight) {
      if (is.null(dim(weight))) {
        # Convert vectors to matrices with a single column
        grad <- matrix(runif(n = length(weight)), nrow = length(weight), ncol = 1)
      } else {
        # If weight is already a matrix
        grad <- matrix(runif(n = prod(dim(weight))), nrow = nrow(weight), ncol = ncol(weight))
      }
      as.numeric(grad)  # Ensure the gradient is numeric
    })
    
    # Perform LAMB update for the weights
    for (layer in seq_along(self$weights)) {
      # Ensure grads_weights[[layer]] and self$weights[[layer]] are matrices or vectors of the same dimension
      grads_weights[[layer]] <- as.numeric(grads_weights[[layer]])
      
      # Ensure params have valid numeric matrices or vectors
      params <- optimizer_params_weights[[layer]]
      params$param <- as.numeric(params$param)
      params$m <- as.numeric(params$m)
      params$v <- as.numeric(params$v)
      
      # Perform LAMB update
      lamb_results <- lamb_update(params, grads_weights[[layer]], lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, lambda = 0.01)
      optimizer_params_weights[[layer]] <- lamb_results$params
      updated_weights_update <- lamb_results$update
      
      # Ensure updated_weights_update is not NULL
      if (is.null(updated_weights_update)) {
        stop("Updated weights update is NULL.")
      }
      
      # Check the type of updated_weights_update
      if (is.matrix(updated_weights_update)) {
        # If updated_weights_update is a matrix
        if (identical(dim(self$weights[[layer]]), dim(updated_weights_update))) {
          self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
        } else {
          # Adjust dimensions to match if necessary
          if (nrow(self$weights[[layer]]) != nrow(updated_weights_update)) {
            if (nrow(self$weights[[layer]]) > nrow(updated_weights_update)) {
              updated_weights_update <- rbind(updated_weights_update, matrix(0, nrow = nrow(self$weights[[layer]]) - nrow(updated_weights_update), ncol = ncol(updated_weights_update)))
            } else if (nrow(self$weights[[layer]]) < nrow(updated_weights_update)) {
              updated_weights_update <- updated_weights_update[1:nrow(self$weights[[layer]]), , drop = FALSE]
            }
          }
          if (ncol(self$weights[[layer]]) != ncol(updated_weights_update)) {
            if (ncol(self$weights[[layer]]) > ncol(updated_weights_update)) {
              updated_weights_update <- cbind(updated_weights_update, matrix(0, nrow = nrow(updated_weights_update), ncol = ncol(self$weights[[layer]]) - ncol(updated_weights_update)))
            } else if (ncol(self$weights[[layer]]) < ncol(updated_weights_update)) {
              updated_weights_update <- updated_weights_update[, 1:ncol(self$weights[[layer]]), drop = FALSE]
            }
          }
          self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
        }
      } else if (is.vector(updated_weights_update) && length(updated_weights_update) == length(self$weights[[layer]])) {
        # If updated_weights_update is a vector
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        cat("Dimensions or type of updated_weights_update are not suitable for subtraction.\n")
        cat("Attempting to adjust dimensions if possible.\n")
        
        # Attempting to adjust dimensions if updated_weights_update is not directly suitable
        if (is.vector(updated_weights_update)) {
          repeated_updated_weights_update <- rep(updated_weights_update, length.out = length(self$weights[[layer]]))
          self$weights[[layer]] <- self$weights[[layer]] - repeated_updated_weights_update
        } else {
          cat("Unable to adjust dimensions for updated_weights_update.\n")
        }
      }
    }
  }
  else if (optimizer == "lookahead") {
    # Ensure gradients for weights are properly initialized
    grads_weights <- lapply(self$weights, function(weights) {
      # Create a gradient matrix with the same dimensions as weights
      grad_matrix <- matrix(runif(n = length(weights)), nrow = nrow(weights), ncol = ncol(weights))
      list(param = grad_matrix)  # Ensure 'param' element is included
    })
    
    # Perform Lookahead update for the weights
    for (layer in seq_along(self$weights)) {
      # Ensure 'param' element is present in grads_weights
      grads_list <- list(param = grads_weights[[layer]]$param)
      
      # Extract optimizer parameters for the current layer
      params <- list(
        param = self$weights[[layer]],
        m = optimizer_params_weights[[layer]]$m,
        v = optimizer_params_weights[[layer]]$v,
        r = optimizer_params_weights[[layer]]$r,
        slow_weights = optimizer_params_weights[[layer]]$slow_weights,
        lookahead_counter = optimizer_params_weights[[layer]]$lookahead_counter,
        lookahead_step = optimizer_params_weights[[layer]]$lookahead_step
      )
      
      # Perform Lookahead update
      lookahead_results <- lookahead_update(
        list(params),  # Wrap params in a list
        list(grads_list),  # Wrap grads_list in a list
        lr,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
        lookahead_step,
        base_optimizer = "adam_update",  # Use "adam" for weights
        t = epoch,
        lambda
      )
      
      # Extract updated weights and other parameters
      updated_weights_update <- lookahead_results$param
      new_m <- lookahead_results$m
      new_v <- lookahead_results$v
      new_r <- lookahead_results$r
      new_slow_weights <- lookahead_results$slow_weights
      new_lookahead_counter <- lookahead_results$lookahead_counter
      
      # Ensure updated_weights_update is not NULL
      if (is.null(updated_weights_update)) {
        cat("Updated weights update is NULL.\n")
        next  # Skip to the next layer
      }
      
      # Check the type of updated_weights_update
      if (is.matrix(updated_weights_update) && all(dim(updated_weights_update) == dim(self$weights[[layer]]))) {
        # If updated_weights_update is a matrix with matching dimensions
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else if (is.vector(updated_weights_update) && length(updated_weights_update) == length(self$weights[[layer]])) {
        # If updated_weights_update is a vector with matching length
        self$weights[[layer]] <- self$weights[[layer]] - updated_weights_update
      } else {
        cat("Dimensions or type of updated_weights_update are not suitable for subtraction.\n")
        cat("Attempting to adjust dimensions if possible.\n")
        
        # Attempting to adjust dimensions if updated_weights_update is not directly suitable
        if (is.vector(updated_weights_update)) {
          repeated_updated_weights_update <- matrix(rep(updated_weights_update, length.out = length(self$weights[[layer]])), nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]))
          self$weights[[layer]] <- self$weights[[layer]] - repeated_updated_weights_update
        } else {
          cat("Unable to adjust dimensions for updated_weights_update.\n")
        }
      }
      
      # Update optimizer parameters
      optimizer_params_weights[[layer]]$m <- new_m
      optimizer_params_weights[[layer]]$v <- new_v
      optimizer_params_weights[[layer]]$r <- new_r
      optimizer_params_weights[[layer]]$slow_weights <- new_slow_weights
      optimizer_params_weights[[layer]]$lookahead_counter <- new_lookahead_counter
    }
  }
  
  
  
  
  else if (optimizer == "adagrad") {
    # Compute the gradients for weights if not already available
    grads_weights <- lapply(self$weights, function(weight) {
      # Example gradient computation (replace with actual gradients)
      grad_matrix <- matrix(runif(n = length(weight)), nrow = nrow(weight), ncol = ncol(weight))
      as.numeric(grad_matrix)  # Ensure the gradient is numeric
    })
    
    # Perform Adagrad update for the weights
    for (layer in seq_along(self$weights)) {
      # Ensure grads_weights[[layer]] is a numeric vector
      grads_matrix <- as.numeric(grads_weights[[layer]])
      
      # Perform Adagrad update
      adagrad_results <- adagrad_update(optimizer_params_weights[[layer]], grads_matrix, lr)
      optimizer_params_weights[[layer]] <- adagrad_results$params
      r_values <- adagrad_results$r
      
      # Ensure r_values is numeric
      if (is.list(r_values)) {
        r_values <- unlist(r_values)
      }
      
      # Calculate updated_weights_update
      updated_weights_update <- grads_matrix / (sqrt(r_values) + epsilon)
      
      # Convert updated_weights_update to a numeric vector if it's a list
      if (is.list(updated_weights_update)) {
        updated_weights_update <- unlist(updated_weights_update)
      }
      
      # Perform subtraction for weights if dimensions match
      if (length(updated_weights_update) == prod(dim(self$weights[[layer]]))) {
        cat("Dimensions match exactly. Performing subtraction.\n")
        self$weights[[layer]] <- self$weights[[layer]] - matrix(updated_weights_update, nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]), byrow = TRUE)
      } else {
        cat("Dimensions do not match exactly. Adjusting dimensions for subtraction.\n")
        repeat_times <- ceiling(nrow(self$weights[[layer]]) * ncol(self$weights[[layer]]) / length(updated_weights_update))
        repeated_updated_weights_update <- rep(updated_weights_update, length.out = nrow(self$weights[[layer]]) * ncol(self$weights[[layer]]))
        self$weights[[layer]] <- self$weights[[layer]] - matrix(repeated_updated_weights_update, nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]), byrow = TRUE)
      }
    }
  }
  else if (optimizer == "adadelta") {
    # Compute the gradients for weights if not already available
    grads_weights <- lapply(self$weights, function(weight) {
      # Example gradient computation (replace with actual gradients)
      grad_matrix <- matrix(runif(n = length(weight)), nrow = nrow(weight), ncol = ncol(weight))
      as.numeric(grad_matrix)  # Ensure the gradient is numeric
    })
    
    # Perform Adadelta update for the weights
    for (layer in seq_along(self$weights)) {
      # Ensure grads_weights[[layer]] is a numeric vector
      grads_matrix <- as.numeric(grads_weights[[layer]])
      
      # Perform Adadelta update
      adadelta_results <- adadelta_update(optimizer_params_weights[[layer]], grads_matrix, lr, epsilon)
      optimizer_params_weights[[layer]] <- adadelta_results$params
      delta_w <- adadelta_results$delta_w
      
      # Ensure delta_w is numeric
      if (is.list(delta_w)) {
        delta_w <- unlist(delta_w)
      }
      
      # Check if delta_w is numeric
      if (!is.numeric(delta_w)) {
        stop("delta_w contains non-numeric values. Please check the adadelta_update function.")
      }
      
      # Calculate updated_weights_update
      updated_weights_update <- delta_w / (sqrt(delta_w) + epsilon)
      
      # Convert updated_weights_update to a numeric vector if it's a list
      if (is.list(updated_weights_update)) {
        updated_weights_update <- unlist(updated_weights_update)
      }
      
      # Perform subtraction for weights if dimensions match
      if (length(updated_weights_update) == prod(dim(self$weights[[layer]]))) {
        cat("Dimensions match exactly. Performing subtraction.\n")
        self$weights[[layer]] <- self$weights[[layer]] - matrix(updated_weights_update, nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]), byrow = TRUE)
      } else {
        cat("Dimensions do not match exactly. Adjusting dimensions for subtraction.\n")
        repeat_times <- ceiling(nrow(self$weights[[layer]]) * ncol(self$weights[[layer]]) / length(updated_weights_update))
        repeated_updated_weights_update <- rep(updated_weights_update, length.out = nrow(self$weights[[layer]]) * ncol(self$weights[[layer]]))
        self$weights[[layer]] <- self$weights[[layer]] - matrix(repeated_updated_weights_update, nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]), byrow = TRUE)
      }
      
      # Print dimensions after Adadelta update
      cat("After Adadelta update - Dimensions of self$weights[[layer]]:", dim(self$weights[[layer]]), "\n")
    }
  }
  
  
  return(list(
    updated_weights_or_biases = self[[target]][[layer]],
    updated_optimizer_params = optimizer_params[[layer]])
  )
  
  
}