apply_optimizer_update <- function(optimizer, optimizer_params, grads_matrix, lr, beta1, beta2, epsilon, epoch, self, layer, target,
                                   alpha = NULL, lambda1 = NULL, lambda2 = NULL) {
  
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
      clip_threshold <- .5
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
    cat(">> Optimizer = rmsprop\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    if (!exists("target_dim")) {
      target_dim <- if (target == "biases") {
        dim(as.matrix(self$biases[[layer]]))
      } else {
        dim(self$weights[[layer]])
      }
    }
    
    
    # Boost LR for output layer if needed
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    # --- FIX: Make sure grads is a list of 2D matrix ---
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # Update call
    optimizer_params[[layer]] <- rmsprop_update(
      optimizer_params[[layer]],
      grads = grads_input,
      lr = lr * layer_boost,
      beta2 = beta2,
      epsilon = epsilon
    )
    
    update <- optimizer_params[[layer]]$updates
    update_matrix <- if (is.list(update) && length(update) == 1) update[[1]] else update
    
    # --- YOUR DIMENSIONAL HANDLING BLOCK (UNCHANGED) ---
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
      if (length(update_matrix) == prod(target_dim)) {
        updated <- matrix(update_matrix, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update_matrix, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update_matrix, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      clip_threshold <- 5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
  }
  
  else if (optimizer == "sgd") {
    cat(">> Optimizer = sgd\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    #$$$$$$$$$$$$$ Ensure optimizer_params is long enough and has the layer slot
    while (length(optimizer_params) < layer) {
      optimizer_params[[length(optimizer_params) + 1]] <- NULL
    }
    if (is.null(optimizer_params[[layer]])) {
      stop(paste("Missing optimizer_params for layer", layer))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1  # Placeholder
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # SGD update
    sgd_result <- sgd_update(
      params = optimizer_params[[layer]],
      grads = grads_input,
      lr = lr * layer_boost
    )
    
    optimizer_params[[layer]] <- sgd_result$params
    
    update <- if (target == "weights") {
      sgd_result$weights_update[[1]]
    } else {
      sgd_result$biases_update[[1]]
    }
    
    if (is.null(update)) {
      stop(paste("SGD update is NULL for layer", layer, "and target", target))
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (length(update) == prod(target_dim)) {
      updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
    } else if (length(update) == 1) {
      updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    } else {
      repeated <- rep(update, length.out = prod(target_dim))
      updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
    }
    
    if (target == "weights") {
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated), 
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    # $$$$$$$$$$$$ Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
    
    # $$$$$$$$$$$$ Build return object
    updated_optimizer <- list(
      updated_optimizer_params = optimizer_params[[layer]],
      updated_weights_or_biases = updated
    )
    
  }
  
  
  
  else if (optimizer == "sgd_momentum") {
    cat(">> Optimizer = sgd_momentum\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    # $$$$$$$$$$$$ Ensure optimizer_params has this layer
    while (length(optimizer_params) < layer) {
      optimizer_params[[length(optimizer_params) + 1]] <- NULL
    }
    if (is.null(optimizer_params[[layer]])) {
      stop(paste("Missing optimizer_params for layer", layer))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # $$$$$$$$$$$$ Momentum update
    sgd_result <- sgd_momentum_update(
      params   = optimizer_params[[layer]],
      grads    = grads_input,
      lr       = lr * layer_boost,
      momentum = beta1
    )
    optimizer_params[[layer]] <- sgd_result$params
    
    update <- if (target == "weights") {
      sgd_result$weights_update[[1]]
    } else {
      sgd_result$biases_update[[1]]
    }
    if (is.null(update)) {
      stop(paste("SGD momentum update is NULL for layer", layer, "and target", target))
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim    <- dim(as.matrix(target_matrix))
    
    if (length(update) == prod(target_dim)) {
      updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
    } else if (length(update) == 1) {
      updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    } else {
      repeated <- rep(update, length.out = prod(target_dim))
      updated  <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
    }
    
    if (target == "weights") {
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (SGD momentum, layer", layer, "):",
        "min =", min(updated),
        ", mean =", mean(updated),
        ", max =", max(updated), "\n")
    
    # $$$$$$$$$$$$ Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else {
      self$biases[[layer]]  <- self$biases[[layer]] - updated
    }
    
    # $$$$$$$$$$$$ Return object to caller
    updated_optimizer <- list(
      updated_optimizer_params   = optimizer_params[[layer]],
      updated_weights_or_biases = updated
    )
  }
  
  
  
  else if (optimizer == "nag") {
    cat(">> Optimizer = nag\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # ✅ NAG Update
    nag_result <- nag_update(
      params = optimizer_params[[layer]],
      grads = grads_input,
      lr = lr * layer_boost,
      beta = beta1
    )
    
    optimizer_params[[layer]] <- nag_result$params
    
    # ✅ Use correct update from result
    update <- if (target == "weights") nag_result$weights_update[[1]] else nag_result$biases_update[[1]]
    
    # Align shapes
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clipping
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated), 
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  
  else if (optimizer == "ftrl") {
    cat(">> Optimizer = ftrl\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # ✅ FTRL Update
    ftrl_result <- ftrl_update(
      params   = optimizer_params[[layer]],
      grads    = grads_input,
      lr       = lr * layer_boost,
      alpha    = 0.1,
      beta     = 1.0,
      lambda1  = 0.01,
      lambda2  = 0.01
    )
    
    optimizer_params[[layer]] <- ftrl_result$params
    
    # ✅ Choose correct update
    update <- if (target == "weights") ftrl_result$weights_update[[1]] else ftrl_result$biases_update[[1]]
    
    # ✅ Align shape
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clipping
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated), 
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    # ✅ Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  else if (optimizer == "lamb") {
    cat(">> Optimizer = lamb\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1))
    } else {
      list(grads_matrix)
    }
    
    # ✅ LAMB Update
    lamb_result <- lamb_update(
      params = optimizer_params[[layer]],
      grads = grads_input[[1]],
      lr = lr * layer_boost,
      beta1 = beta1,
      beta2 = beta2,
      eps = epsilon,
      lambda = 0.01
    )
    
    optimizer_params[[layer]] <- lamb_result$params
    
    # ✅ Select correct update
    update <- if (target == "weights") lamb_result$weights_update[[1]] else lamb_result$biases_update[[1]]
    
    # Align shapes
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clipping
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated), 
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  else if (optimizer == "lookahead") {
    cat(">> Optimizer = lookahead\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    # ✅ Call the lookahead optimizer
    lookahead_result <- lookahead_update(
      params = optimizer_params[[layer]],
      grads_list = list(grads_matrix),
      lr = lr * layer_boost,
      beta1 = beta1,
      beta2 = beta2,
      epsilon = epsilon,
      lookahead_step = lookahead_step,
      base_optimizer = "adam_update",
      epoch = epoch,
      lambda = lambda
    )
    
    # ✅ Update the state
    optimizer_params[[layer]] <- lookahead_result
    
    # ✅ Extract update for weight or bias
    update <- if (target == "weights") lookahead_result$weights_update else {
      if (!is.null(lookahead_result$biases_update)) lookahead_result$biases_update else matrix(0, nrow = 1, ncol = 1)
    }
    
    # ✅ Align update shape to target
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (length(update) == prod(target_dim)) {
      updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
    } else if (length(update) == 1) {
      updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    } else {
      repeated <- rep(update, length.out = prod(target_dim))
      updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
    }
    
    # ✅ Optionally clip weights
    if (target == "weights") {
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated),
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    # ✅ Apply the update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  
  
  else if (optimizer == "adagrad") {
    cat(">> Optimizer = adagrad\n")
    cat("Layer:", layer, "\n")
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # ✅ Call Adagrad update
    adagrad_result <- adagrad_update(
      params = optimizer_params[[layer]],
      grads = grads_input,
      lr = lr * layer_boost,
      epsilon = epsilon
    )
    
    optimizer_params[[layer]] <- adagrad_result$params
    
    # ✅ Use correct update depending on target
    update <- if (target == "weights") adagrad_result$weights_update[[1]] else adagrad_result$biases_update[[1]]
    
    # Align shapes
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clipping (if you want it)
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated),
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  else if (optimizer == "adadelta") {
    cat(">> Optimizer = adadelta (", target, ")\n")
    cat("Layer:", layer, "\n")
    
    err <- errors[[layer]]
    err_dims <- dim(err)
    cat("errors dim:\n"); print(err_dims)
    
    # Normalize error to matrix shape
    if (is.null(err_dims)) {
      err <- matrix(err, nrow = 1)
    } else if (length(err_dims) == 1) {
      err <- matrix(err, nrow = 1)
    }
    
    # Safe to compute gradient matrix now
    grads_matrix <- colSums(err)
    
    cat("grads_matrix dim:\n")
    print(dim(grads_matrix))
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- list(matrix(grads_matrix, nrow = 1))
    
    adadelta_result <- adadelta_update(
      params = optimizer_params[[layer]],
      grads = grads_input,
      lr = lr * layer_boost,
      epsilon = epsilon
    )
    
    optimizer_params[[layer]] <- adadelta_result$params
    
    update <- if (target == "weights") adadelta_result$weights_update[[1]] else adadelta_result$biases_update[[1]]
    
    if (is.null(update) || !is.numeric(update)) {
      warning("WARNING: update (delta_w) is NULL or non-numeric. Skipping update for layer", layer, " target:", target)
      return(NULL)
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (is.null(target_dim)) {
      warning("WARNING: target matrix has NULL dimension for layer", layer)
      return(NULL)
    }
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clip for weight updates
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    cat("Updated", target, "summary (layer", layer, "): min =", min(updated),
        ", mean =", mean(updated), ", max =", max(updated), "\n")
    
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  return(list(
    updated_weights_or_biases = self[[target]][[layer]],
    updated_optimizer_params = optimizer_params[[layer]])
  )
  
  
}