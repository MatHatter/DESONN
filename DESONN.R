# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Ensemble Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
# 
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

source("utils/utils.R")
source("optimizers.R")
source("activation_functions.R")
source("reports/evaluate_predictions_report.R")
function(showlibraries){
  # Fake function for collapse feature.
  ## ====== REQUIRED PACKAGES (only if not already loaded) ======
  # install.packages("R6")
  # install.packages("cluster")
  # install.packages("fpc", type = "source")  # Still needed as source for some systems
  # install.packages("tibble")
  # install.packages("dplyr")
  # install.packages("tidyverse")  # Includes ggplot2, dplyr, purrr, readr, etc.
  # install.packages("ggplot2")
  # install.packages("plotly")
  # install.packages("gridExtra")
  # install.packages("rlist")
  # install.packages("writexl")
  # install.packages("readxl")
  # install.packages("tidyr")
  # install.packages("purrr")
  # install.packages("pracma")
  # install.packages("randomForest")
  # install.packages("openxlsx")
  # install.packages("ggplotify")
  library(R6)
  library(cluster)
  library(fpc)
  library(tibble)
  library(dplyr)
  library(tidyverse)
  library(ggplot2)
  library(plotly)
  library(gridExtra)
  library(rlist)
  library(writexl)
  library(readxl)
  library(tidyr)
  library(purrr)
  library(pracma)
  library(randomForest)
  library(openxlsx)
  library(pROC)
  library(ggplotify)
}

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#_____/\\\\\\\\\\\__________/\\\\\________/\\\\\_____/\\\___/\\\\\_____/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#___/\\\/////////\\\______/\\\///\\\_____\/\\\\\\___\/\\\__\/\\\\\\___\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#__\//\\\______\///_____/\\\/__\///\\\___\/\\\/\\\__\/\\\__\/\\\/\\\__\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#___\////\\\___________/\\\______\//\\\__\/\\\//\\\_\/\\\__\/\\\//\\\_\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#______\////\\\_______\/\\\_______\/\\\__\/\\\\//\\\\/\\\__\/\\\\//\\\\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#_________\////\\\____\//\\\______/\\\___\/\\\_\//\\\/\\\__\/\\\_\//\\\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#__/\\\______\//\\\____\///\\\__/\\\_____\/\\\__\//\\\\\\__\/\\\__\//\\\\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#_\///\\\\\\\\\\\/_______\///\\\\\/______\/\\\___\//\\\\\__\/\\\___\//\\\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#___\///////////___________\/////________\///_____\/////___\///_____\/////_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Step 1: Define the Self-Organizing Neural Network (SONN) class


SONN <- R6Class(
  "SONN",
  lock_objects = FALSE,
  public = list(
    input_size = NULL,  # Define input_size as a public property
    hidden_sizes = NULL,
    output_size = NULL,
    num_layers = NULL,
    lambda = NULL,
    weights = NULL,
    biases = NULL,
    ML_NN = NULL,
    N = NULL,
    map = NULL,
    threshold = NULL,
    model_iter_num = NULL,
    optimal_epoch = NULL,
    activation_functions = NULL,
    activation_functions_learn = NULL,
    dropout_rates = NULL,
    dropout_rates_learn = NULL,
    
    initialize = function(input_size, hidden_sizes = NULL, output_size, Rdata = NULL, N,  lambda, ML_NN, dropout_rates = NULL, activation_functions_learn = NULL, activation_functions = NULL, method = init_method, custom_scale = custom_scale) {
      
      
      
      # Initialize SONN parameters and architecture
      # Define functions for self-organization, learning, and prediction
      self$input_size <- input_size
      if (ML_NN){
        self$hidden_sizes <- hidden_sizes
      }
      self$output_size <- output_size
      self$lambda <- lambda  # Regularization parameter
      self$ML_NN <- ML_NN
      self$num_layers <- length(hidden_sizes) + 1  # including the output layer
      
      # Initialize activation functions
      # self$activation_functions <- vector("list", self$num_layers)
      # self$activation_functions_learn <- vector("list", self$num_layers)
      
      self$dropout_rates <- dropout_rates
      self$dropout_rates_learn <- self$dropout_rates
      # Initialize self as an environment or list
      # self <- new.env()  # You could also use self <- list()
      
      self$output_size <- output_size
      self$lambda <- lambda  # Regularization parameter
      self$ML_NN <- ML_NN
      self$num_layers <- length(hidden_sizes) + 1  # including the output layer
      
      self$dropout_rates <- dropout_rates
      self$dropout_rates_learn <- self$dropout_rates
      
      # self$weights <- vector("list", self$num_layers)
      # self$biases <- vector("list", self$num_layers)
      
      
      
      
      
      
      # Initialize weights and biases for subsequent layers if ML_NN is TRUE
      if (ML_NN) {
        # Initialize weights and biases using specified initialization method
        init <- self$initialize_weights(input_size, hidden_sizes, output_size, method = init_method, custom_scale)
        self$weights <- init$weights
        self$biases <- init$biases
      }else{
        self$weights <- matrix(runif(input_size *  output_size), ncol = output_size, nrow = input_size) #should highly consider removing column if not relevant
        self$biases <- rnorm(output_size, mean = 0, sd = 0.01)
      }
      
      
      
      weights_stored <<- as.matrix(self$weights[[1]])
      biases_stored <<- as.matrix(self$biases)
      
      
      
      # for (m in 1:self$num_layers) {
      #     weight_name <- ifelse(m == 1, "weights", paste0("weights", m))
      #     cat("Weight matrix", weight_name, ":\n")
      #
      #     # Check if the weight matrix exists and is not NULL
      #     if (length(self$weights) >= m && !is.null(self$weights[[m]])) {
      #         print(self$weights[[m]])
      #     } else {
      #         cat("Weight matrix", weight_name, "is NULL or not initialized.\n")
      #     }}
      
      
      # Function to find factors of N that are as close as possible to each other
      find_grid_dimensions <- function(N) {
        factors <- unlist(sapply(1:floor(sqrt(N)), function(x) {
          if (N %% x == 0) return(c(x, N / x))
        }))
        factors <- sort(unique(factors))  # Sort and remove duplicates
        
        # Find the index of the factor that is closest to the square root of N
        sqrt_N <- sqrt(N)
        idx <- which.min(abs(factors - sqrt_N))
        
        # Return the pair of factors closest to each other
        c(factors[idx], N / factors[idx])
      }
      
      # Use the function to dynamically calculate grid_rows and grid_cols
      grid_dimensions <- find_grid_dimensions(N)
      grid_rows <- grid_dimensions[1]
      grid_cols <- grid_dimensions[2]
      
      self$map <- matrix(1:N, nrow = grid_rows, ncol = grid_cols)
      
      # Configuration flags for enabling/disabling per-SONN model training plots
      self$PerEpochViewPlotsConfig <- list(
        accuracy_plot = accuracy_plot,  # training accuracy/loss
        saturation_plot = saturation_plot,  # output saturation
        max_weight_plot = max_weight_plot,  # max weight magnitude
        viewAllPlots = viewAllPlots,
        verbose    = verbose
      )
      
    },
    initialize_weights = function(input_size, hidden_sizes, output_size, method = init_method, custom_scale = NULL) {
      weights <- list()
      biases <- list()
      
      # clip_weights <- function(W, limit = .08) {
      #   return(pmin(pmax(W, -limit), limit))
      # }
      
      init_weight <- function(fan_in, fan_out, method, custom_scale) {
        if (method == "xavier") {
          scale <- ifelse(is.null(custom_scale), 0.5, custom_scale)
          sd <- sqrt(2 / (fan_in + fan_out)) * scale
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
        } else if (method == "he") {
          scale <- ifelse(is.null(custom_scale), 1.0, custom_scale)
          sd <- sqrt(2 / fan_in) * scale
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
        } else if (method == "lecun") {
          scale <- ifelse(is.null(custom_scale), 1.0, custom_scale)
          sd <- sqrt(1 / fan_in) * scale
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
        } else if (method == "orthogonal") {
          A <- matrix(rnorm(fan_in * fan_out), nrow = fan_in, ncol = fan_out)
          Q <- qr.Q(qr(A))
          if (ncol(Q) < fan_out) {
            Q <- cbind(Q, matrix(0, nrow = fan_in, ncol = fan_out - ncol(Q)))
          }
          W <- Q
        } else if (method == "variance_scaling") {
          scale <- ifelse(is.null(custom_scale), 0.5, custom_scale)
          sd <- sqrt(1 / (fan_in + fan_out)) * scale
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = min(sd, 0.2)), nrow = fan_in, ncol = fan_out)
        } else if (method == "glorot_uniform") {
          limit <- sqrt(6 / (fan_in + fan_out))
          W <- matrix(runif(fan_in * fan_out, min = -limit, max = limit), nrow = fan_in, ncol = fan_out)
        } else {
          sd <- 0.01
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
        }
        return(W)
        # return(clip_weights(W, limit = .08))
      }
      
      # Initialize first hidden layer
      weights[[1]] <- init_weight(input_size, hidden_sizes[1], method, custom_scale)
      biases[[1]] <- matrix(rnorm(hidden_sizes[1], mean = 0, sd = 0.01), ncol = 1)  # slightly higher bias init
      
      # Intermediate hidden layers
      for (layer in 2:length(hidden_sizes)) {
        weights[[layer]] <- init_weight(hidden_sizes[layer - 1], hidden_sizes[layer], method, custom_scale)
        biases[[layer]] <- matrix(rnorm(hidden_sizes[layer], mean = 0, sd = 0.01), ncol = 1)
      }
      
      # Output layer
      last_hidden_size <- hidden_sizes[[length(hidden_sizes)]]
      weights[[length(hidden_sizes) + 1]] <- init_weight(last_hidden_size, output_size, method, custom_scale)
      biases[[length(hidden_sizes) + 1]] <- matrix(rnorm(output_size, mean = 0, sd = 0.01), ncol = 1)
      
      self$weights <- weights
      self$biases <- biases
      
      return(list(weights = weights, biases = biases))
    },
    # Dropout functions with no default rates (training only)
    dropout_forward = function(x, rate) {
      # Keep shapes stable & support NULL / edge rates
      if (is.null(rate) || rate <= 0 || rate >= 1) {
        return(list(out = x, mask = NULL, scale = 1))
      }
      x <- as.matrix(x)
      mask <- matrix(rbinom(length(x), 1, 1 - rate), nrow = nrow(x), ncol = ncol(x))
      scale <- 1 / (1 - rate)              # inverted dropout scaling
      list(out = x * mask * scale, mask = mask, scale = scale)
    },
    dropout_backward = function(grad, mask, rate) {
      # Reuse the SAME mask as forward; apply the same scaling
      if (is.null(mask) || is.null(rate) || rate <= 0 || rate >= 1) {
        return(grad)
      }
      grad <- as.matrix(grad)
      grad * mask * (1 / (1 - rate))
    }
    ,# Method to perform self-organization
    viewPerEpochPlots = function(name) {
      cfg <- self$PerEpochViewPlotsConfig
      on_all <- isTRUE(cfg$viewAllPlots) || isTRUE(cfg$verbose)
      isTRUE(cfg[[name]]) || on_all
    },
    self_organize = function(Rdata, labels, lr) {
      print("------------------------self-organize-begin----------------------------------------")
      
      
      
      
      
      if (self$ML_NN) {
        # Multi-layer mode: First layer
        input_rows <- nrow(Rdata)
        output_cols <- ncol(self$weights[[1]])
        
        if (length(self$biases[[1]]) == 1) {
          bias_matrix <- matrix(self$biases[[1]], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases[[1]]) == output_cols) {
          bias_matrix <- matrix(self$biases[[1]], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases[[1]]) < output_cols) {
          bias_matrix <- matrix(rep(self$biases[[1]], length.out = output_cols), 
                                nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else {
          bias_matrix <- matrix(self$biases[[1]][1:output_cols], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        }
        
        outputs <- Rdata %*% self$weights[[1]] + bias_matrix
        
      } else {
        # Single-layer mode
        input_rows <- nrow(Rdata)
        output_cols <- ncol(self$weights)
        
        if (length(self$biases) == 1) {
          bias_matrix <- matrix(self$biases, nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases) == output_cols) {
          bias_matrix <- matrix(self$biases, nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases) < output_cols) {
          bias_matrix <- matrix(rep(self$biases, length.out = output_cols), 
                                nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else {
          bias_matrix <- matrix(self$biases[1:output_cols], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        }
        
        outputs <- Rdata %*% self$weights + bias_matrix 
      }
      
      
      
      
      if (self$ML_NN) {
        hidden_outputs <- list()
        hidden_outputs[[1]] <- outputs
        
        outputs <- vector("list", self$num_layers)
        outputs[[1]] <- hidden_outputs[[1]]
        
        broadcast_bias <- function(bias, nrow_out, ncol_out) {
          if (length(bias) == 1) {
            matrix(bias, nrow_out, ncol_out, byrow = TRUE)
          } else if (length(bias) == ncol_out) {
            matrix(bias, nrow_out, ncol_out, byrow = TRUE)
          } else if (length(bias) < ncol_out) {
            matrix(rep(bias, length.out = ncol_out), nrow_out, ncol_out, byrow = TRUE)
          } else if (length(bias) > ncol_out) {
            matrix(bias[1:ncol_out], nrow_out, ncol_out, byrow = TRUE)
          } else {
            stop("Bias shape mismatch")
          }
        }
        
        for (layer in 2:self$num_layers) {
          input <- hidden_outputs[[layer - 1]]
          weights <- self$weights[[layer]]
          biases <- broadcast_bias(self$biases[[layer]], nrow(input), ncol(weights))
          
          if (ncol(input) != nrow(weights)) {
            if (ncol(input) == ncol(weights)) {
              weights <- t(weights)
            } else {
              stop("Dimensions of hidden_outputs and weights are not conformable")
            }
          }
          
          hidden_outputs[[layer]] <- input %*% weights + biases
          outputs[[layer]] <- hidden_outputs[[layer]]
        }
      }
      
      print("str(outputs)")
      str(outputs)
      
      
      
      
      if (self$ML_NN) {
        print(paste("LAYER", self$num_layers))
        
        expected_shape <- dim(outputs[[self$num_layers]])
        input_shape <- dim(Rdata)
        
        if (!all(expected_shape == input_shape)) {
          cat("Mismatch between Rdata and outputs[[num_layers]]: correcting...\n")
          # Try to reshape outputs to match Rdata
          output_matrix <- matrix(
            rep(outputs[[self$num_layers]], length.out = nrow(Rdata) * ncol(Rdata)),
            nrow = nrow(Rdata),
            ncol = ncol(Rdata),
            byrow = FALSE
          )
        } else {
          output_matrix <- outputs[[self$num_layers]]
        }
        
        error_1000x10 <- Rdata - output_matrix
        
      } else {
        if (!all(dim(outputs) == dim(Rdata))) {
          cat("Mismatch between Rdata and outputs (single-layer): correcting...\n")
          output_matrix <- matrix(
            rep(outputs, length.out = nrow(Rdata) * ncol(Rdata)),
            nrow = nrow(Rdata),
            ncol = ncol(Rdata),
            byrow = FALSE
          )
        } else {
          output_matrix <- outputs
        }
        
        error_1000x10 <- Rdata - output_matrix
      }
      
      # Store output error
      errors <- vector("list", self$num_layers)
      errors[[self$num_layers]] <- as.matrix(error_1000x10)
      str(errors[[self$num_layers]])
      
      
      # Store output error
      errors <- vector("list", self$num_layers)
      errors[[self$num_layers]] <- as.matrix(error_1000x10)
      str(errors[[self$num_layers]])
      
      
      
      
      # Propagate the error backwards
      if (self$ML_NN) {
        for (layer in (self$num_layers - 1):1) {
          cat("Layer:", layer, "\n")
          
          # Load weights and error from next layer
          weights_next <- self$weights[[layer + 1]]
          errors_next  <- errors[[layer + 1]]
          
          # Check for NULLs
          if (is.null(weights_next) || is.null(errors_next)) {
            cat(paste("Skipping layer", layer, "- weights or errors are NULL\n"))
            next
          }
          
          # Print actual dimensions
          weight_dims <- dim(weights_next)
          error_dims  <- dim(errors_next)
          cat("Weights dimensions:\n"); print(weight_dims)
          cat("Errors dimensions:\n"); print(error_dims)
          
          # Sanity checks
          if (is.null(weight_dims) || is.null(error_dims)) {
            cat(paste("Skipping layer", layer, "- dimensions are NULL\n"))
            next
          }
          
          # Adjust shape dynamically
          expected_error_cols <- ncol(weights_next)
          actual_error_cols   <- ncol(errors_next)
          actual_error_rows   <- nrow(errors_next)
          
          # Match error columns to weights' output dim
          if (actual_error_cols != expected_error_cols) {
            if (actual_error_cols > expected_error_cols) {
              errors_next <- errors_next[, 1:expected_error_cols, drop = FALSE]
            } else {
              errors_next <- matrix(
                rep(errors_next, length.out = actual_error_rows * expected_error_cols),
                nrow = actual_error_rows,
                ncol = expected_error_cols
              )
            }
          }
          
          
          # Propagate error
          cat("Backpropagating errors for layer", layer, "\n")
          errors[[layer]] <- errors_next %*% t(weights_next)
        }
      }
      
      else {
        cat("Single Layer Backpropagation\n")
        
        # Check existence
        weights_sl <- self$weights[[1]]
        errors_sl  <- errors[[1]]
        
        if (is.null(weights_sl) || is.null(errors_sl)) {
          stop("Error: Weights or errors for single layer do not exist.")
        }
        
        # Ensure matrix form
        weights_sl <- as.matrix(weights_sl)
        errors_sl  <- as.matrix(errors_sl)
        
        # Print current dimensions
        weight_dims <- dim(weights_sl)
        error_dims  <- dim(errors_sl)
        cat("Weights dimensions:\n"); print(weight_dims)
        cat("Errors dimensions:\n"); print(error_dims)
        
        if (is.null(weight_dims) || is.null(error_dims)) {
          stop("Error: Dimensions for weights or errors are NULL.")
        }
        
        # Target: errors[[1]] = weights %*% errors
        # Align shapes: [n_input, n_output] %*% [batch_size, n_output]^T
        
        # Ensure error has matching columns
        expected_cols <- ncol(weights_sl)
        actual_cols   <- ncol(errors_sl)
        actual_rows   <- nrow(errors_sl)
        
        if (actual_cols != expected_cols) {
          if (actual_cols > expected_cols) {
            errors_sl <- errors_sl[, 1:expected_cols, drop = FALSE]
          } else {
            errors_sl <- matrix(
              rep(errors_sl, length.out = actual_rows * expected_cols),
              nrow = actual_rows,
              ncol = expected_cols
            )
          }
        }
        
        # If error is still misaligned for matrix multiplication, transpose
        if (ncol(errors_sl) != ncol(weights_sl)) {
          if (ncol(errors_sl) == nrow(weights_sl)) {
            weights_sl <- t(weights_sl)
          } else {
            cat("Warning: shape mismatch persists in single-layer case\n")
          }
        }
        
        # Perform backpropagation step
        cat("Performing matrix multiplication for single layer\n")
        errors[[1]] <- errors_sl %*% t(weights_sl)
      }
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      print("str(errors)")
      str(errors)
      
      
      if (self$ML_NN) {
        
        # Ensure errors[[1]] has same number of rows as Rdata
        if (nrow(errors[[1]]) != nrow(Rdata)) {
          if (nrow(errors[[1]]) > nrow(Rdata)) {
            errors[[1]] <- errors[[1]][1:nrow(Rdata), , drop = FALSE]
          } else {
            errors[[1]] <- errors[[1]][rep(1:nrow(errors[[1]]), length.out = nrow(Rdata)), , drop = FALSE]
          }
        }
        
        # Update weights for the first layer
        if (ncol(errors[[1]]) == nrow(self$weights[[1]])) {
          self$weights[[1]] <- self$weights[[1]] - (lr * t(Rdata) %*% errors[[1]])
        } else if (nrow(t(errors[[1]])) == nrow(self$weights[[1]]) && ncol(t(errors[[1]])) < ncol(Rdata)) {
          self$weights[[1]] <- self$weights[[1]] - ((lr * t(Rdata) %*% errors[[1]]))[, 1:ncol(self$weights[[1]])]
        } else if (prod(dim(self$weights[[1]])) == 1) {
          update_value <- lr * sum(t(Rdata) %*% errors[[1]])
          self$weights[[1]] <- self$weights[[1]] - update_value
        } else {
          self$weights[[1]] <- self$weights[[1]] - (lr * apply(t(Rdata) %*% errors[[1]], 2, mean))
        }
        
        # Update biases for the first layer
        if (length(self$biases[[1]]) < length(colMeans(errors[[1]]))) {
          colMeans_shortened <- colMeans(errors[[1]])[1:length(self$biases[[1]])]
          self$biases[[1]] <- self$biases[[1]] - (lr * colMeans_shortened)
        } else if (length(self$biases[[1]]) > length(colMeans(errors[[1]]))) {
          colMeans_extended <- rep(colMeans(errors[[1]]), length.out = length(self$biases[[1]]))
          self$biases[[1]] <- self$biases[[1]] - (lr * colMeans_extended)
        } else {
          self$biases[[1]] <- self$biases[[1]] - (lr * colMeans(errors[[1]]))
        }
        
      }
      else {
        # Robust single-layer weight update
        gradient <- tryCatch({
          grad <- t(Rdata) %*% error_1000x10
          if (all(dim(self$weights) == dim(grad))) {
            grad
          } else if (prod(dim(self$weights)) == 1) {
            sum(grad)
          } else if (ncol(self$weights) < ncol(grad)) {
            grad[, 1:ncol(self$weights), drop = FALSE]
          } else if (ncol(self$weights) > ncol(grad)) {
            matrix(
              rep(grad, length.out = nrow(self$weights) * ncol(self$weights)),
              nrow = nrow(self$weights),
              ncol = ncol(self$weights)
            )
          } else {
            apply(grad, 2, mean)
          }
        }, error = function(e) {
          apply(t(Rdata) %*% error_1000x10, 2, mean)
        })
        
        # Update weights
        if (is.matrix(gradient)) {
          self$weights <- self$weights - (lr * gradient)
        } else {
          self$weights <- self$weights - (lr * matrix(gradient, nrow = nrow(self$weights), ncol = ncol(self$weights)))
        }
        
        # Robust single-layer bias update
        if (length(self$biases) == ncol(error_1000x10)) {
          self$biases <- self$biases - (lr * colMeans(error_1000x10))
        } else if (length(self$biases) < ncol(error_1000x10)) {
          self$biases <- self$biases - (lr * colMeans(error_1000x10)[1:length(self$biases)])
        } else {
          extended <- rep(colMeans(error_1000x10), length.out = length(self$biases))
          self$biases <- self$biases - (lr * extended)
        }
      }
      
      
      if (self$ML_NN) {
        for (layer in 2:self$num_layers) {
          
          # Ensure the number of columns in errors[[layer]] matches the number of rows in hidden_outputs[[layer - 1]]
          if (ncol(errors[[layer]]) != nrow(hidden_outputs[[layer - 1]])) {
            if (ncol(errors[[layer]]) > nrow(hidden_outputs[[layer - 1]])) {
              # Truncate columns of errors[[layer]] to match the number of rows in hidden_outputs[[layer - 1]]
              errors[[layer]] <- errors[[layer]][, 1:nrow(hidden_outputs[[layer - 1]]), drop = FALSE]
            } else {
              # Replicate columns of errors[[layer]] to match the number of rows in hidden_outputs[[layer - 1]]
              errors[[layer]] <- errors[[layer]][, rep(1:ncol(errors[[layer]]), length.out = nrow(hidden_outputs[[layer - 1]])), drop = FALSE]
            }
          }
          
          
          
          
          # Calculate dimensions
          result_dim <- dim(errors[[layer]] %*% (hidden_outputs[[layer - 1]]))
          weight_dim <- dim(self$weights[[layer]])
          
          print("results_dim")
          print(dim(errors[[layer]]))
          print("hidden_outputs[[layer - 1]])")
          print(dim(hidden_outputs[[layer - 1]]))
          
          # Update weights for the layer
          if (ncol(self$weights[[layer]]) == ncol(hidden_outputs[[layer - 1]])) {
            if (all(weight_dim == result_dim)) {
              grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
              self$weights[[layer]] <- self$weights[[layer]] - lr * grad
            } else {
              cat("Dimensions mismatch, handling default case for weights.\n")
              grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
              grad <- grad[1:nrow(self$weights[[layer]]), 1:ncol(self$weights[[layer]])]
              self$weights[[layer]] <- self$weights[[layer]] - lr * grad
            }
            
          } else if (nrow(self$weights[[layer]]) == ncol(hidden_outputs[[layer - 1]]) &&
                     ncol(self$weights[[layer]]) < ncol(hidden_outputs[[layer - 1]])) {
            grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
            grad <- grad[1:nrow(self$weights[[layer]]), 1:ncol(self$weights[[layer]])]
            self$weights[[layer]] <- self$weights[[layer]] - lr * grad
            
          } else if (prod(weight_dim) == 1) {
            grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
            update_value <- lr * sum(grad)
            self$weights[[layer]] <- self$weights[[layer]] - update_value
            
          } else {
            grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
            grad <- grad[1:nrow(self$weights[[layer]]), 1:ncol(self$weights[[layer]])]
            self$weights[[layer]] <- self$weights[[layer]] - lr * grad
          }
          
          
          
          # Update biases for the layer
          if (length(self$biases[[layer]]) < length(colMeans(errors[[layer]]))) {
            colMeans_shortened <- colMeans(errors[[layer]])[1:length(self$biases[[layer]])]
            self$biases[[layer]] <- self$biases[[layer]] - (lr * colMeans_shortened)
          } else if (length(self$biases[[layer]]) > length(colMeans(errors[[layer]]))) {
            colMeans_extended <- rep(colMeans(errors[[layer]]), length.out = length(self$biases[[layer]]))
            self$biases[[layer]] <- self$biases[[layer]] - (lr * colMeans_extended)
          } else {
            self$biases[[layer]] <- self$biases[[layer]] - (lr * colMeans(errors[[layer]]))
          }
          
        }
      }
      
      
      
      if (is.null(self$map)) {
        cat("[Debug] SOM not yet trained. Training now...\n")
        self$train_map(Rdata)
        
        # Determine how many SOM neurons to keep based on max allowed
        max_neurons_allowed <- 9  # e.g., for 3x3 SOM map
        map_size <- prod(dim(self$map$codes[[1]]))  # total SOM neurons
        actual_neurons <- min(map_size, max_neurons_allowed)
        
        input_dim <- ncol(Rdata)
        
        # Truncate weights to desired number of SOM neurons and match input dimension
        truncated_weights <- self$map$codes[[1]][1:actual_neurons, 1:input_dim, drop = FALSE]
        self$weights[[1]] <- matrix(truncated_weights, nrow = actual_neurons, ncol = input_dim)
        
        # Set bias to match output dimension of layer 1
        output_dim_layer1 <- if (self$ML_NN && self$num_layers >= 1) {
          ncol(self$weights[[1]])
        } else {
          1
        }
        self$biases[[1]] <- rep(0, output_dim_layer1)
        
        # Debug info
        cat("[Debug] SOM-trained weights dim after truncation:\n")
        print(dim(self$weights[[1]]))
      }
      
      
      
      
      
      
      print("------------------------self-organize-end------------------------------------------")
      
    },

    learn = function(Rdata, labels, lr, CLASSIFICATION_MODE,
                     activation_functions_learn, dropout_rates_learn, sample_weights) {
      print("------------------------learn-begin-------------------------------------------------")
      start_time <- Sys.time()

      `%||%` <- function(x, y) if (is.null(x)) y else x
      .safe_get <- function(lst, idx) {
        if (is.list(lst) && length(lst) >= idx && idx >= 1) lst[[idx]] else NULL
      }
      
      ## ---------------------------
      ## Labels & sample_weights prep
      ## ---------------------------
      if (identical(CLASSIFICATION_MODE, "binary")) {
        if (!is.numeric(labels) || !is.matrix(labels) || ncol(labels) != 1) {
          labels <- matrix(as.numeric(labels), ncol = 1)
        }
        pos_weight <- 2
        neg_weight <- 1
        if (is.null(sample_weights)) {
          sample_weights <- ifelse(labels == 1, pos_weight, neg_weight)
        }
        sample_weights <- matrix(sample_weights, nrow = nrow(labels), ncol = 1)
        
        if (!is.matrix(labels)) labels <- as.matrix(labels)
        if (length(dim(labels)) == 2 && nrow(labels) == ncol(labels)) {
          labels <- matrix(diag(labels), ncol = 1)
        }
        labels <- matrix(as.numeric(labels), ncol = 1)
        
      } else if (identical(CLASSIFICATION_MODE, "multiclass")) {
        # one-hot if needed
        if (is.matrix(labels) && ncol(labels) >= 2) {
          labels_mat <- labels
        } else {
          lbl_vec <- as.vector(labels)
          if (!is.null(self$class_levels) && length(self$class_levels) > 0) {
            lbl_fac <- factor(lbl_vec, levels = self$class_levels)
          } else {
            lbl_fac <- factor(lbl_vec)
            self$class_levels <- levels(lbl_fac)
          }
          labels_mat <- model.matrix(~ lbl_fac - 1)
          colnames(labels_mat) <- as.character(self$class_levels)
        }
        labels <- as.matrix(labels_mat)
        
        if (is.null(sample_weights)) {
          sample_weights <- rep(1, nrow(labels))
        }
        sample_weights <- matrix(sample_weights, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
        
      } else if (identical(CLASSIFICATION_MODE, "regression")) {
        if (!is.numeric(labels)) labels <- as.numeric(labels)
        if (!is.matrix(labels) || ncol(labels) != 1L) {
          labels <- matrix(labels, ncol = 1L)
        }
        storage.mode(labels) <- "double"
        
        if (is.null(sample_weights)) {
          sample_weights <- rep(1, nrow(labels))
        }
        sample_weights <- matrix(as.numeric(sample_weights), nrow = nrow(labels), ncol = 1L)
        
      } else {
        stop(sprintf("Unknown CLASSIFICATION_MODE: %s", CLASSIFICATION_MODE))
      }
      
      ## ---------------------------
      ## Normalize dropout list to num_layers
      ## ---------------------------
      self$dropout_rates_learn <- if (is.list(dropout_rates_learn)) dropout_rates_learn else list(dropout_rates_learn)
      if (length(self$dropout_rates_learn) < self$num_layers) {
        self$dropout_rates_learn <- c(self$dropout_rates_learn,
                                      rep(list(NULL), self$num_layers - length(self$dropout_rates_learn)))
      } else if (length(self$dropout_rates_learn) > self$num_layers) {
        self$dropout_rates_learn <- self$dropout_rates_learn[1:self$num_layers]
      }
      
      ## ---------------------------
      ## Initialize outputs
      ## ---------------------------
      predicted_output_learn <- NULL
      error_learn <- NULL
      dim_hidden_layers_learn <- list()
      predicted_output_learn_hidden <- NULL
      bias_gradients <- list()
      grads_matrix <- list()
      errors <- list()
      
      ## ================================================================
      ## MULTI-LAYER MODE
      ## ================================================================
      if (self$ML_NN) {
        hidden_outputs <- vector("list", self$num_layers)
        activation_derivatives <- vector("list", self$num_layers)
        dropout_masks <- rep(list(NULL), self$num_layers)   # store masks for backprop
        dim_hidden_layers_learn <- vector("list", self$num_layers)
        
        input_matrix <- as.matrix(Rdata)
        
        # Forward pass
        for (layer in 1:self$num_layers) {
          weights_matrix <- as.matrix(self$weights[[layer]])
          bias_vec <- as.numeric(unlist(self$biases[[layer]]))
          input_data <- if (layer == 1) input_matrix else hidden_outputs[[layer - 1]]
          input_data <- as.matrix(input_data)
          
          input_rows <- nrow(input_data)
          weights_rows <- nrow(weights_matrix)
          weights_cols <- ncol(weights_matrix)
          
          cat(sprintf("[Debug] Layer %d : input dim = %d x %d | weights dim = %d x %d\n",
                      layer, input_rows, ncol(input_data), weights_rows, weights_cols))
          
          if (ncol(input_data) != weights_rows) {
            stop(sprintf("Layer %d: input cols (%d) do not match weights rows (%d)",
                         layer, ncol(input_data), weights_rows))
          }
          
          if (length(bias_vec) == 1) {
            bias_matrix <- matrix(bias_vec, nrow = input_rows, ncol = weights_cols)
          } else if (length(bias_vec) == weights_cols) {
            bias_matrix <- matrix(rep(bias_vec, each = input_rows), nrow = input_rows)
          } else if (length(bias_vec) == input_rows * weights_cols) {
            bias_matrix <- matrix(bias_vec, nrow = input_rows)
          } else {
            stop(sprintf("Layer %d: invalid bias shape: length = %d", layer, length(bias_vec)))
          }
          
          Z <- input_data %*% weights_matrix + bias_matrix
          
          activation_function <- if (length(activation_functions_learn) >= layer) activation_functions_learn[[layer]] else NULL
          activation_name <- if (is.function(activation_function)) attr(activation_function, "name") else "none"
          cat(sprintf("[Debug] Layer %d : Activation Function = %s\n", layer, activation_name))
          
          A <- if (is.function(activation_function)) activation_function(Z) else Z
          
          # Dropout on hidden layers only
          rate <- .safe_get(self$dropout_rates_learn, layer)
          if (layer == self$num_layers) rate <- NULL
          do_out <- self$dropout_forward(A, rate)
          A <- do_out$out
          dropout_masks[[layer]] <- do_out$mask
          
          hidden_outputs[[layer]] <- A
          
          # Derivatives for hidden layers; for output, we’ll handle with CE shortcut
          if (activation_name != "none") {
            derivative_name <- paste0(activation_name, "_derivative")
            if (!exists(derivative_name, mode = "function")) {
              stop(paste("Layer", layer, ": Activation derivative function", derivative_name, "does not exist."))
            }
            activation_derivatives[[layer]] <- get(derivative_name, mode = "function")(Z)
          } else {
            activation_derivatives[[layer]] <- matrix(1, nrow = nrow(Z), ncol = ncol(Z))
          }
          
          dim_hidden_layers_learn[[layer]] <- dim(A)
        }
        
        predicted_output_learn <- hidden_outputs[[self$num_layers]]
        predicted_output_learn_hidden <- hidden_outputs
        
        # Error (kept as (pred - y) * w so we can reuse as CE delta at output)
        error_learn <- (predicted_output_learn - labels) * sample_weights
        
        # Backward pass with CE shortcut at the output
        error_backprop <- error_learn
        for (layer in self$num_layers:1) {
          
          # Use CE shortcut on the output layer IF classification + (sigmoid|softmax)
          use_ce_shortcut <- FALSE
          if (identical(CLASSIFICATION_MODE, "binary") || identical(CLASSIFICATION_MODE, "multiclass")) {
            act_fun <- if (length(activation_functions_learn) >= layer) activation_functions_learn[[layer]] else NULL
            act_name <- if (is.function(act_fun)) attr(act_fun, "name") else "none"
            if (layer == self$num_layers && (act_name %in% c("sigmoid", "softmax"))) {
              use_ce_shortcut <- TRUE
            }
          }
          
          if (use_ce_shortcut) {
            delta <- error_learn  # (A_L - Y) * w, no multiply by derivative
          } else {
            delta <- error_backprop * activation_derivatives[[layer]]
          }
          
          # Apply SAME mask/rate as forward for this layer (output layer had rate=NULL)
          rate <- .safe_get(self$dropout_rates_learn, layer)
          mask <- .safe_get(dropout_masks, layer)
          delta <- self$dropout_backward(delta, mask, rate)
          
          # Gradients
          bias_gradients[[layer]] <- matrix(colMeans(delta), nrow = 1)        # average over batch
          input_for_grad <- if (layer == 1) input_matrix else hidden_outputs[[layer - 1]]
          grads_matrix[[layer]] <- t(input_for_grad) %*% delta                # weight grads
          
          errors[[layer]] <- delta
          
          # Propagate to previous layer
          if (layer > 1) {
            weights_t <- t(as.matrix(self$weights[[layer]]))
            error_backprop <- delta %*% weights_t
          }
        }
        
        ## ================================================================
        ## SINGLE-LAYER MODE
        ## ================================================================
      } else {
        cat("Single Layer Learning Phase\n")
        
        X <- as.matrix(Rdata)
        weights_matrix <- as.matrix(self$weights)
        bias_vec <- as.numeric(unlist(self$biases))
        
        if (ncol(X) != nrow(weights_matrix)) {
          stop(sprintf("SL NN: input cols (%d) do not match weights rows (%d)", ncol(X), nrow(weights_matrix)))
        }
        
        if (length(bias_vec) == 1) {
          bias_matrix <- matrix(bias_vec, nrow = nrow(X), ncol = ncol(weights_matrix))
        } else if (length(bias_vec) == ncol(weights_matrix)) {
          bias_matrix <- matrix(rep(bias_vec, each = nrow(X)), nrow = nrow(X))
        } else if (length(bias_vec) == nrow(X) * ncol(weights_matrix)) {
          bias_matrix <- matrix(bias_vec, nrow = nrow(X))
        } else {
          stop(sprintf("SL NN: invalid bias shape: length = %d", length(bias_vec)))
        }
        
        # Optional input dropout (SL)
        if (!is.list(self$dropout_rates_learn)) self$dropout_rates_learn <- list(self$dropout_rates_learn)
        rate <- .safe_get(self$dropout_rates_learn, 1)
        do_x <- self$dropout_forward(X, rate)
        X_dropped <- do_x$out
        mask <- do_x$mask
        
        Z <- X_dropped %*% weights_matrix + bias_matrix
        
        # Activation
        if (is.function(activation_functions_learn)) {
          activation_function <- activation_functions_learn
        } else {
          if (!is.list(activation_functions_learn)) activation_functions_learn <- list(activation_functions_learn)
          activation_function <- activation_functions_learn[[1]]
        }
        activation_name <- if (is.function(activation_function)) attr(activation_function, "name") else "none"
        A <- if (is.function(activation_function)) activation_function(Z) else Z
        predicted_output_learn <- A
        
        if (identical(CLASSIFICATION_MODE, "multiclass") && ncol(predicted_output_learn) != ncol(labels)) {
          stop(sprintf("SL NN (multiclass): output cols (%d) != label cols (%d).",
                       ncol(predicted_output_learn), ncol(labels)))
        }
        
        # Error
        error_learn <- (predicted_output_learn - labels) * sample_weights
        dim_hidden_layers_learn[[1]] <- dim(predicted_output_learn)
        
        # CE shortcut at output if (binary/multiclass) & (sigmoid/softmax)
        use_ce_shortcut <- FALSE
        if (identical(CLASSIFICATION_MODE, "binary") || identical(CLASSIFICATION_MODE, "multiclass")) {
          if (activation_name %in% c("sigmoid", "softmax")) use_ce_shortcut <- TRUE
        }
        
        if (use_ce_shortcut) {
          delta <- error_learn
        } else {
          deriv_fn_name <- if (is.function(activation_function)) paste0(attr(activation_function, "name"), "_derivative") else NULL
          activation_deriv <- if (!is.null(deriv_fn_name) && exists(deriv_fn_name)) {
            get(deriv_fn_name)(Z)
          } else {
            matrix(1, nrow = nrow(Z), ncol = ncol(Z))
          }
          delta <- error_learn * activation_deriv
        }
        
        # Backprop through input dropout (SL path used mask on X)
        delta <- self$dropout_backward(delta, mask, rate)
        
        bias_gradients[[1]] <- matrix(colMeans(delta), nrow = 1)
        grads_matrix[[1]] <- t(X_dropped) %*% delta
        errors[[1]] <- delta
      }
      
      learn_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      print("------------------------learn-end-------------------------------------------------")
      
      return(list(learn_output = predicted_output_learn, learn_time = learn_time, error = error_learn, dim_hidden_layers = dim_hidden_layers_learn, hidden_outputs = predicted_output_learn_hidden, grads_matrix = grads_matrix, bias_gradients = bias_gradients, errors = errors))
    },
  
    # Method to perform prediction
    predict = function(Rdata, weights, biases, activation_functions, verbose=FALSE, debug=FALSE) {
      # ---- Debug/Verbose toggles ----
      if (is.null(debug)) {
        debug <- isTRUE(get0("DEBUG_PREDICT_FORWARD", inherits = TRUE, ifnotfound = FALSE))
      }
      .dbg <- function(...) if (isTRUE(debug))   cat("[PRED-DBG] ", sprintf(...), "\n", sep = "")
      .vbs <- function(...) if (isTRUE(verbose) && !isTRUE(debug)) cat(sprintf(...), "\n")
      
      # ---------- last-layer variance probe (local helper) ----------
      probe_last_layer <- function(Z_last, A_last, last_af_name = NA_character_, tag = "[PROBE]") {
        vZ  <- as.vector(Z_last); vA <- as.vector(A_last)
        sdZ <- stats::sd(vZ, na.rm = TRUE); sdA <- stats::sd(vA, na.rm = TRUE)
        rngZ <- range(vZ, na.rm = TRUE); rngA <- range(vA, na.rm = TRUE)
        
        # Print to verbose (nice, concise) OR debug (more hints). Debug contains diagnostics too.
        if (isTRUE(debug)) {
          cat(sprintf(
            "%s last_af=%s | sd(Z)=%.6g | sd(A)=%.6g | range(Z)=[%.6g, %.6g] | range(A)=[%.6g, %.6g]\n",
            tag, as.character(last_af_name), sdZ, sdA, rngZ[1], rngZ[2], rngA[1], rngA[2]
          ))
          eps_flat <- 1e-6
          if (sdZ < eps_flat) {
            cat(sprintf("%s DIAG: Z_last is ~flat -> training collapse.\n", tag))
          } else if (sdA < sdZ * 1e-3) {
            cat(sprintf("%s DIAG: Z_last has spread but A_last is squashed -> activation/head mismatch.\n", tag))
          } else {
            cat(sprintf("%s DIAG: Variance preserved across head.\n", tag))
          }
        } else if (isTRUE(verbose)) {
          cat(sprintf(
            "%s head=%s | sd(Z)=%.6g → sd(A)=%.6g\n",
            tag, as.character(last_af_name), sdZ, sdA
          ))
        }
        invisible(list(sdZ = sdZ, sdA = sdA, rngZ = rngZ, rngA = rngA))
      }
      # --------------------------------------------------------------
      
      # If weights/biases are missing → fall back to internal state (stateful mode)
      if (is.null(weights)) {
        if (!is.null(self$weights)) weights <- self$weights else stop("predict(): weights not provided and self$weights is NULL.")
      }
      if (is.null(biases)) {
        if (!is.null(self$biases))  biases  <- self$biases  else stop("predict(): biases not provided and self$biases is NULL.")
      }
      
      # Ensure lists
      if (!is.list(weights)) weights <- list(weights)
      if (!is.list(biases))  biases  <- list(biases)
      if (!is.null(activation_functions) && !is.list(activation_functions)) activation_functions <- list(activation_functions)
      
      start_time  <- Sys.time()
      output      <- as.matrix(Rdata)
      num_layers  <- length(weights)
      
      # Input diagnostics
      .dbg("INPUT dims=%d x %d | mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
           nrow(output), ncol(output),
           mean(output), stats::sd(as.vector(output)), min(output),
           stats::median(as.vector(output)), max(output))
      .vbs("Predict: X dims=%d x %d", nrow(output), ncol(output))
      
      for (layer in seq_len(num_layers)) {
        w <- as.matrix(weights[[layer]])
        b <- as.numeric(unlist(biases[[layer]]))
        
        # Broadcast bias to match samples × units
        n_samples <- nrow(output)
        n_units   <- ncol(w)
        if (length(b) == 1) {
          bias_mat <- matrix(b, nrow = n_samples, ncol = n_units, byrow = TRUE)
        } else if (length(b) == n_units) {
          bias_mat <- matrix(b, nrow = n_samples, ncol = n_units, byrow = TRUE)
        } else {
          bias_mat <- matrix(rep(b, length.out = n_units), nrow = n_samples, ncol = n_units, byrow = TRUE)
        }
        
        # Weights/bias debug info
        .dbg("L%02d: W dims=%d x %d | W mean=%.6g sd=%.6g min=%.6g max=%.6g",
             layer, nrow(w), ncol(w), mean(w), stats::sd(as.vector(w)), min(w), max(w))
        
        # Concise bias summary for verbose mode (and richer in debug)
        if (isTRUE(debug) || isTRUE(verbose)) {
          b_sd   <- if (length(b) > 1) stats::sd(b) else 0
          b_min  <- if (length(b) > 0) min(b) else NA_real_
          b_max  <- if (length(b) > 0) max(b) else NA_real_
          if (isTRUE(debug)) {
            b_head <- paste(utils::head(round(b, 6), 6), collapse = ", ")
            cat(sprintf("[BIAS] L%02d: len=%d | mean=%.6g sd=%.6g | range=[%s, %s] | head=[%s]\n",
                        layer, length(b), mean(b), b_sd,
                        format(b_min, digits = 6), format(b_max, digits = 6), b_head))
          } else {
            cat(sprintf("[BIAS] L%02d: len=%d | mean=%.6g sd=%.6g | range=[%s, %s]\n",
                        layer, length(b), mean(b), b_sd,
                        format(b_min, digits = 6), format(b_max, digits = 6)))
          }
        }
        
        # Linear transformation
        output <- output %*% w + bias_mat
        
        # Pre-activation stats
        .dbg("L%02d: Z dims=%d x %d | Z mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
             layer, nrow(output), ncol(output),
             mean(output), stats::sd(as.vector(output)), min(output),
             stats::median(as.vector(output)), max(output))
        
        # Per-layer probe (before activation)
        if (isTRUE(debug) || isTRUE(verbose)) {
          sdZ  <- stats::sd(as.vector(output))
          rngZ <- range(as.vector(output))
          if (isTRUE(debug)) {
            cat(sprintf("[L%02d-PROBE] sd(Z)=%.6g | range(Z)=[%.6g, %.6g]\n", layer, sdZ, rngZ[1], rngZ[2]))
          } else {
            cat(sprintf("[L%02d] sd(Z)=%.6g\n", layer, sdZ))
          }
        }
        
        Z_curr <- output  # keep for last-layer probe
        
        # Apply activation if provided
        if (!is.null(activation_functions) &&
            length(activation_functions) >= layer &&
            is.function(activation_functions[[layer]])) {
          
          act_name <- tryCatch({
            nm <- attr(activation_functions[[layer]], "name")
            if (is.null(nm)) "function" else nm
          }, error = function(e) "function")
          
          .dbg("L%02d: ACT[%s] applying...", layer, act_name)
          output <- activation_functions[[layer]](output)
          
          # After-activation probe
          if (isTRUE(debug) || isTRUE(verbose)) {
            sdA  <- stats::sd(as.vector(output))
            rngA <- range(as.vector(output))
            if (isTRUE(debug)) {
              cat(sprintf("[L%02d-PROBE] sd(A)=%.6g | range(A)=[%.6g, %.6g]\n", layer, sdA, rngA[1], rngA[2]))
            } else {
              cat(sprintf("[L%02d] sd(A)=%.6g\n", layer, sdA))
            }
          }
          
          # Last-layer variance probe
          if (layer == num_layers) {
            last_af_name <- tryCatch(tolower(attr(activation_functions[[layer]], "name")),
                                     error = function(e) NA_character_)
            probe_last_layer(Z_last = Z_curr, A_last = output,
                             last_af_name = last_af_name, tag = "[PROBE]")
          }
        } else {
          .dbg("L%02d: ACT[identity] (no activation function provided for this layer)", layer)
          
          if (isTRUE(debug) || isTRUE(verbose)) {
            sdA  <- stats::sd(as.vector(output))
            rngA <- range(as.vector(output))
            if (isTRUE(debug)) {
              cat(sprintf("[L%02d-PROBE] sd(A)=%.6g | range(A)=[%.6g, %.6g]\n", layer, sdA, rngA[1], rngA[2]))
            } else {
              cat(sprintf("[L%02d] sd(A)=%.6g\n", layer, sdA))
            }
          }
          
          if (layer == num_layers) {
            probe_last_layer(Z_last = Z_curr, A_last = output,
                             last_af_name = "identity", tag = "[PROBE]")
          }
        }
      }
      
      # Always do HEAD-DBG if last layer
      if (layer == num_layers) {
        cat("\n[HEAD-DBG] ---- Last layer diagnostic ----\n")
        cat(sprintf("[HEAD-DBG] W_last dims=%d x %d | mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                    nrow(w), ncol(w), mean(w), sd(as.vector(w)), min(w), max(w)))
        cat(sprintf("[HEAD-DBG] b_last len=%d | mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                    length(b), mean(b), if (length(b)>1) sd(b) else 0, min(b), max(b)))
        cat(sprintf("[HEAD-DBG] Z_last: mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                    mean(Z_curr), sd(as.vector(Z_curr)), min(Z_curr), max(Z_curr)))
        cat(sprintf("[HEAD-DBG] A_last: mean=%.6f sd=%.6f min=%.6f max=%.6f\n\n",
                    mean(output), sd(as.vector(output)), min(output), max(output)))
        
        probe_last_layer(Z_last = Z_curr, A_last = output,
                         last_af_name = act_name, tag = "[PROBE]")
      }
      
      end_time <- Sys.time()
      prediction_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      .dbg("DONE | total_time=%.6fs | FINAL dims=%d x %d | mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
           prediction_time, nrow(output), ncol(output),
           mean(output), stats::sd(as.vector(output)), min(output),
           stats::median(as.vector(output)), max(output))
      
      if (isTRUE(verbose) && !isTRUE(debug)) {
        cat(sprintf("Predict complete in %.4fs | Output dims=%d x %d\n",
                    prediction_time, nrow(output), ncol(output)))
      }
      
      return(list(predicted_output = output, prediction_time = prediction_time))
    }
    ,# Method for training the SONN with L2 regularization
    train_with_l2_regularization = function(Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, model_iter_num, update_weights, update_biases, use_biases, ensemble_number, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type, sample_weights, X_validation, y_validation, threshold_function, ML_NN, train, verbose) {
      
      start_time <- Sys.time()
      
      # -----------------------
      # epoch list bookkeeping
      # -----------------------
      if (never_ran_flag == TRUE || !hyperparameter_grid_setup) {
        losses <- numeric(num_epochs)
        epoch_in_list <- num_epochs
      } else if (never_ran_flag == FALSE && hyperparameter_grid_setup && !use_loaded_weights) {
        epoch_in_list <- num_epochs[[model_iter_num]] # if error here, you need to run all over again.
        losses <- vector("list", length = epoch_in_list)
      } else {
        epoch_in_list <- optimal_epochs[[model_iter_num]]
        losses <- vector("list", length = epoch_in_list)
      }
      
      # ----------------------------
      # (unchanged) state/optimizer
      # ----------------------------
      prev_weights <- NULL
      prev_biases  <- NULL
      optimizer_params_weights <- vector("list", self$num_layers)
      optimizer_params_biases  <- vector("list", self$num_layers)
      
      best_val_acc <- -Inf
      best_val_epoch <- NA
      best_train_acc <- -Inf
      best_epoch <- NA
      val_accuracy_log <- c()
      train_accuracy_log <- c()
      loss_log <- c()
      learning_rate_log <- c()
      val_loss_log <- c()
      train_loss_log <- c()
      mean_output_log <- c()
      sd_output_log <- c()
      max_weight_log <- c()
      
      # =========================
      # Local helpers (debug friendly)
      # =========================
      
      .extract_vec <- function(x) {
        if (is.data.frame(x)) return(x[[1]])
        if (is.matrix(x) || is.array(x)) {
          if (ncol(x) == 1L) return(as.vector(x[,1]))
          return(as.vector(x[,1]))
        }
        x
      }
      
      .align_len <- function(v, n) {
        if (length(v) > n) v[seq_len(n)]
        else if (length(v) < n) c(v, rep(NA, n - length(v)))
        else v
      }
      
      .build_targets <- function(labels, n, K, CLASSIFICATION_MODE) {
        lv <- .align_len(.extract_vec(labels), n)
        cat("[dbg] targets: CLASSIFICATION_MODE =", CLASSIFICATION_MODE, "\n")
        cat("[dbg] targets: labels class =", paste(class(labels), collapse=","), " | lv length =", length(lv), "\n")
        
        if (identical(CLASSIFICATION_MODE, "multiclass")) {
          stopifnot(K >= 2)
          if (is.matrix(labels) && nrow(labels) >= n && ncol(labels) == K &&
              all(labels[seq_len(n), , drop=FALSE] %in% c(0,1))) {
            Y <- matrix(as.numeric(labels[seq_len(n), , drop=FALSE]), n, K)
            y_idx <- max.col(Y, ties.method = "first")
            cat("[dbg] targets: using provided one-hot (n×K)\n")
            return(list(Y=Y, y_idx=y_idx))
          } else {
            f <- if (is.factor(lv)) lv else factor(lv)
            idx <- as.integer(f)
            L <- nlevels(f)
            if (L > K) {
              cat(sprintf("[dbg] targets: L=%d > K=%d, truncating indices > K to K\n", L, K))
              idx[idx > K] <- K
            }
            cat("[dbg] targets: factor levels L =", L, " | head idx =", paste(utils::head(idx,6), collapse=", "), "\n")
            # use your global one_hot_from_ids
            return(list(Y=one_hot_from_ids(idx, K, N=n), y_idx=idx))
          }
          
        } else if (identical(CLASSIFICATION_MODE, "binary")) {
          stopifnot(K == 1)
          if (is.factor(lv)) {
            y <- as.integer(lv) - 1L
          } else {
            y <- suppressWarnings(as.numeric(lv))
            if (all(is.na(y))) { f <- factor(lv); y <- as.integer(f) - 1L }
          }
          y[is.na(y)] <- 0L
          y <- pmin(pmax(as.numeric(y), 0), 1)
          cat("[dbg] targets: binary y summary -> mean=", mean(y), " | sum=", sum(y), "\n")
          return(list(y=y))
          
        } else if (identical(CLASSIFICATION_MODE, "regression")) {
          cat("[dbg] targets: regression path | n=", n, " K=", K, "\n")
          
          if (is.matrix(labels) || is.data.frame(labels)) {
            Ytmp <- suppressWarnings(matrix(as.numeric(as.matrix(labels)),
                                            nrow = nrow(as.matrix(labels)),
                                            ncol = ncol(as.matrix(labels))))
          } else {
            Ytmp <- suppressWarnings(matrix(as.numeric(lv), nrow = length(lv), ncol = 1L))
          }
          
          if (all(is.na(Ytmp))) {
            cat("[dbg] targets: all NA after coercion; filling zeros\n")
            Ytmp <- matrix(0, nrow = max(1L, nrow(Ytmp)), ncol = max(1L, ncol(Ytmp)))
          }
          
          # conform rows
          if (nrow(Ytmp) > n) {
            Ytmp <- Ytmp[seq_len(n), , drop = FALSE]
          } else if (nrow(Ytmp) < n) {
            add <- n - nrow(Ytmp)
            last_row <- if (nrow(Ytmp) >= 1) Ytmp[nrow(Ytmp), , drop = FALSE] else matrix(0, 1, max(1L, ncol(Ytmp)))
            Ytmp <- rbind(Ytmp, do.call(rbind, replicate(add, last_row, simplify = FALSE)))
          }
          # conform cols
          if (ncol(Ytmp) > K) {
            Ytmp <- Ytmp[, seq_len(K), drop = FALSE]
          } else if (ncol(Ytmp) < K) {
            if (ncol(Ytmp) == 0L) Ytmp <- matrix(0, nrow = nrow(Ytmp), ncol = 1L)
            col_map <- rep(seq_len(ncol(Ytmp)), length.out = K)
            Ytmp <- Ytmp[, col_map, drop = FALSE]
          }
          
          storage.mode(Ytmp) <- "double"
          cat("[dbg] targets: regression Y dim ->", paste(dim(Ytmp), collapse="x"),
              " | summary mean=", mean(Ytmp), "\n")
          
          # provide both Y and y for downstream compatibility
          return(list(Y = Ytmp, y = as.numeric(Ytmp[,1])))
          
        } else {
          stop("Unknown CLASSIFICATION_MODE. Use 'multiclass', 'binary', or 'regression'.")
        }
      }
      
      .bce_loss <- function(p, y, eps=1e-12) {
        p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
        y <- pmin(pmax(as.numeric(y), 0), 1)
        val <- mean(-(y*log(p) + (1-y)*log(1-p)))
        if (!is.finite(val)) val <- NA_real_
        val
      }
      
      .ce_loss_multiclass <- function(P, Y, eps=1e-12) {
        P <- pmin(pmax(as.matrix(P), eps), 1 - eps)  # n×K
        if (is.vector(Y)) Y <- one_hot_from_ids(as.integer(Y), K=ncol(P), N=nrow(P))
        val <- mean(-rowSums(Y * log(P)))
        if (!is.finite(val)) val <- NA_real_
        val
      }
      
      
      # ======== TRAIN LOOP ========
      if (train) {

        # --- LOGS (empty at start, will fill each epoch) ---
        train_accuracy_log <- numeric(0)
        train_loss_log     <- numeric(0)
        mean_output_log    <- numeric(0)
        sd_output_log      <- numeric(0)
        max_weight_log     <- numeric(0)
        val_accuracy_log   <- numeric(0)
        val_loss_log       <- numeric(0)
        
        
        for (epoch in 1:epoch_in_list) {
          
          # lr <- lr_scheduler(epoch)
          cat("Epoch:", epoch, "| Learning Rate:", lr, "\n")
          num_epochs_check <<- num_epochs
          
          # 1) Train step
          learn_result <- self$learn(
            Rdata = Rdata,
            labels = labels,
            lr = lr,
            CLASSIFICATION_MODE = CLASSIFICATION_MODE,
            activation_functions_learn = activation_functions,
            dropout_rates_learn = dropout_rates,
            sample_weights = sample_weights
          )
          # --- Error debug ---
          if (!is.null(learn_result$error)) {
            cat("[TRAIN-DBG] last-layer error summary ->",
                " min=", min(learn_result$error, na.rm=TRUE),
                " mean=", mean(learn_result$error, na.rm=TRUE),
                " max=", max(learn_result$error, na.rm=TRUE),
                " sd=",  sd(learn_result$error, na.rm=TRUE), "\n")
          }
          
          # =========================
          # TRAINING METRICS (mode-aware; replaces the "ORIGINAL BLOCK" preview)
          # =========================
          
          probs_train <- learn_result$learn_output
          probs_train <- as.matrix(probs_train)
          storage.mode(probs_train) <- "double"
          
          n <- nrow(probs_train)
          K <- max(1L, ncol(probs_train))
          
          # Align labels to n rows (trim only; no recycling)
          labels_epoch <- if (is.matrix(labels)) {
            if (nrow(labels) == n) labels else labels[seq_len(min(nrow(labels), n)), , drop = FALSE]
          } else if (is.data.frame(labels)) {
            v <- labels[[1]]
            v[seq_len(min(length(v), n))]
          } else {
            labels[seq_len(min(length(labels), n))]
          }
          
          # Build targets like in validation
          targs_tr <- .build_targets(labels_epoch, n, K, CLASSIFICATION_MODE)
          
          # Compute metrics by mode
          if (identical(CLASSIFICATION_MODE, "multiclass")) {
            stopifnot(K >= 2)
            pred_idx_tr   <- max.col(probs_train, ties.method = "first")
            train_accuracy <- mean(pred_idx_tr == targs_tr$y_idx, na.rm = TRUE)
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss <- .ce_loss_multiclass(probs_train, targs_tr$Y)
            } else {
              train_loss <- mean((probs_train - targs_tr$Y)^2, na.rm = TRUE)
            }
            
          } else if (identical(CLASSIFICATION_MODE, "binary")) {
            stopifnot(K == 1)
            preds_bin_tr   <- as.integer(probs_train >= 0.5)
            train_accuracy <- mean(preds_bin_tr == targs_tr$y, na.rm = TRUE)
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss <- .bce_loss(probs_train, targs_tr$y)
            } else {
              train_loss <- mean((probs_train - matrix(targs_tr$y, ncol = 1))^2, na.rm = TRUE)
            }
            
          } else if (identical(CLASSIFICATION_MODE, "regression")) {
            # No accuracy for regression; report NA and proper loss instead
            y_reg     <- if (is.matrix(labels_epoch)) as.numeric(labels_epoch[,1]) else as.numeric(labels_epoch)
            preds_reg <- as.numeric(probs_train[,1])
            train_loss <- mean((preds_reg - y_reg)^2, na.rm = TRUE)
            train_accuracy <- NA_real_
            
          } else {
            stop("Unknown CLASSIFICATION_MODE.")
          }
          
          # Log metrics (single source of truth)
          train_accuracy_log <- c(train_accuracy_log, train_accuracy)
          train_loss_log     <- c(train_loss_log,     train_loss)
          
          # Track best training accuracy only when defined (classification)
          if (!is.na(train_accuracy) && (is.na(best_train_acc) || train_accuracy > best_train_acc)) {
            best_train_acc   <- train_accuracy
            best_epoch_train <- epoch
          }
          
          cat(sprintf(
            "Epoch %d | Train %s: %s | Loss: %.6f\n",
            epoch,
            if (identical(CLASSIFICATION_MODE, "regression")) "R²/Acc" else "Accuracy",
            if (is.na(train_accuracy)) "NA" else sprintf("%.2f%%", 100 * train_accuracy),
            train_loss
          ))
          
          
          predicted_output_train_reg <- learn_result
          predicted_output_train_reg_prediction_time <- learn_result$learn_time
          
          # Predicted output (use correct output layer)
          if (self$ML_NN) {
            predicted_output <- predicted_output_train_reg$hidden_outputs[[self$num_layers]]
          } else {
            predicted_output <- predicted_output_train_reg$learn_output
          }
          # # === PROBE: check preds vs labels ===
          # probe_preds_vs_labels(predicted_output, labels, tag = paste0("TRAIN-EPOCH", epoch))
          # if (epoch == num_epochs) {
          #   probe_preds_vs_labels(predicted_output, labels,
          #                         tag = paste0("TRAIN-EPOCH", epoch),
          #                         save_global = TRUE)
          # }
          # 
          
          # Output saturation diagnostics
          mean_output <- mean(predicted_output)
          sd_output   <- sd(predicted_output)
          cat("Mean Output:", round(mean_output, 4), "| StdDev:", round(sd_output, 4), "\n")
          mean_output_log <- c(mean_output_log, mean_output)
          sd_output_log   <- c(sd_output_log, sd_output)
          
          # Optional: compute and store loss
          train_loss <- mean((predicted_output - labels)^2, na.rm = TRUE)
          train_loss_log <- c(train_loss_log, train_loss)
          
          # Weight explosion diagnostics
          if (exists("best_weights_record")) {
            max_weight <- max(sapply(best_weights_record, function(w) max(abs(w))))
            cat("Max Weight Abs:", round(max_weight, 4), "\n")
          } else {
            max_weight <- NA
          }
          max_weight_log <- c(max_weight_log, max_weight)
          
          ## =========================
          ## SONN — Top 3 Per-epoch Plots (your original scaffolding)
          ## =========================
          if (is.null(self$PerEpochlViewPlotsConfig)) self$PerEpochlViewPlotsConfig <- list()
          .fix_flag <- function(v, default) { if (isTRUE(v)) TRUE else if (isFALSE(v)) FALSE else default }
          defaults <- list(accuracy_plot = TRUE, saturation_plot = TRUE, max_weight_plot = TRUE, viewAllPlots = FALSE, verbose = FALSE)
          for (nm in names(defaults)) {
            self$PerEpochlViewPlotsConfig[[nm]] <- .fix_flag(self$PerEpochlViewPlotsConfig[[nm]], defaults[[nm]])
          }
          pe <- self$PerEpochlViewPlotsConfig
          message(sprintf("SONN per-epoch flags → acc=%s, sat=%s, max=%s, all=%s, verbose=%s",
                          pe$accuracy_plot, pe$saturation_plot, pe$max_weight_plot, pe$viewAllPlots, pe$verbose))
          message(sprintf("SONN gate eval → acc=%s, sat=%s, max=%s",
                          self$viewPerEpochPlots("accuracy_plot"),
                          self$viewPerEpochPlots("saturation_plot"),
                          self$viewPerEpochPlots("max_weight_plot")))
          if (!dir.exists("plots")) dir.create("plots", recursive = TRUE, showWarnings = FALSE)
          ens <- as.integer(if (!is.null(self$ensemble_number)) self$ensemble_number else get0("ensemble_number", 1L))
          mod <- as.integer(if (exists("model_iter_num", inherits = TRUE)) model_iter_num else get0("model_iter_num", 1L))
          
          # =========================
          # MERGED-IN (SECOND BLOCK) — gradients, normalization, BLOCK A metrics
          # =========================
          
          # ---- Gradients & misc outputs from learn() (normalize shapes early) ----
          weight_gradients_raw <- learn_result$grads_matrix
          bias_gradients_raw   <- learn_result$bias_gradients
          errors               <- learn_result$errors
          error                <- learn_result$error
          dim_hidden_layers    <- learn_result$dim_hidden_layers
          
          .as_list <- function(x, L) {
            if (is.null(x)) {
              vector("list", L)
            } else if (is.list(x)) {
              if (length(x) < L) c(x, vector("list", L - length(x))) else x[seq_len(L)]
            } else {
              lst <- vector("list", L); lst[[L]] <- x; lst
            }
          }
          
          weight_gradients <- .as_list(weight_gradients_raw, if (isTRUE(self$ML_NN)) self$num_layers else 1L)
          bias_gradients   <- .as_list(bias_gradients_raw,   if (isTRUE(self$ML_NN)) self$num_layers else 1L)
          
          # Conform gradient shapes to weight shapes
          for (lyr in seq_along(weight_gradients)) {
            if (isTRUE(self$ML_NN)) {
              W <- as.matrix(self$weights[[lyr]])
            } else {
              W <- if (is.list(self$weights)) as.matrix(self$weights[[1]]) else as.matrix(self$weights)
            }
            if (is.null(weight_gradients[[lyr]])) {
              weight_gradients[[lyr]] <- matrix(0, nrow = nrow(W), ncol = ncol(W))
            } else {
              G <- as.matrix(weight_gradients[[lyr]])
              if (!all(dim(G) == dim(W))) {
                G2 <- matrix(0, nrow = nrow(W), ncol = ncol(W))
                r <- min(nrow(G), nrow(W)); c <- min(ncol(G), ncol(W))
                G2[seq_len(r), seq_len(c)] <- G[seq_len(r), seq_len(c)]
                G <- G2
              }
              weight_gradients[[lyr]] <- G
            }
            
            if (is.null(bias_gradients[[lyr]])) {
              bias_gradients[[lyr]] <- matrix(0, nrow = 1, ncol = ncol(W))
            } else {
              b <- as.numeric(bias_gradients[[lyr]])
              if (length(b) != ncol(W)) {
                b2 <- numeric(ncol(W)); len <- min(length(b), ncol(W)); b2[seq_len(len)] <- b[seq_len(len)]
                b <- b2
              }
              bias_gradients[[lyr]] <- matrix(b, nrow = 1L)
            }
          }
          
          cat(sprintf("Grad norm L1 by layer: %s\n",
                      paste(vapply(weight_gradients, function(G) sum(abs(G), na.rm=TRUE), numeric(1)),
                            collapse=" | ")))
          
          # 2) Final head / predictions (already set as probs_train above)
          storage.mode(probs_train) <- "double"
          n <- nrow(probs_train); K <- ncol(probs_train)
          
          # 3) Align labels for this epoch (trim to n; no padding)
          if (is.matrix(labels)) {
            labels_epoch <- if (nrow(labels) == n) labels else labels[seq_len(n), , drop = FALSE]
          } else {
            labels_epoch <- if (length(labels) == n) labels else labels[seq_len(min(length(labels), n))]
          }
          if (CLASSIFICATION_MODE == "binary") {
            predictions <- probs_train                # n×1
          } else if (CLASSIFICATION_MODE == "multiclass") {
            predictions <- probs_train                # n×K
          } else if (CLASSIFICATION_MODE == "regression") {
            predictions <- probs_train                # n×1 continuous
          } else stop("Unknown CLASSIFICATION_MODE")
          # =========================
          # BLOCK A — Accuracy & Saturation (multiclass/binary/regression aware)
          # =========================
          cat(sprintf("[dbg] BLOCK A: n=%d, K=%d | probs_train range=[%.6f, %.6f]\n",
                      n, K, min(probs_train), max(probs_train)))
          cat("[dbg] BLOCK A: CLASSIFICATION_MODE =", CLASSIFICATION_MODE, "\n")
          
          if (identical(CLASSIFICATION_MODE, "multiclass")) {
            targs <- .build_targets(labels_epoch, n, K, CLASSIFICATION_MODE)
            stopifnot(K >= 2)
            pred_idx <- max.col(probs_train, ties.method = "first")
            cat("[dbg] BLOCK A: pred_idx head =", paste(utils::head(pred_idx, 6), collapse=", "), "\n")
            cat("[dbg] BLOCK A: lbl_idx head  =", paste(utils::head(targs$y_idx, 6), collapse=", "), "\n")
            train_accuracy_blockA <- mean(pred_idx == targs$y_idx, na.rm = TRUE)
            train_accuracy_log    <- c(train_accuracy_log, train_accuracy_blockA)
            cat(sprintf("[dbg] BLOCK A: train_accuracy=%.6f\n", train_accuracy_blockA))
            
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss_blockA <- .ce_loss_multiclass(probs_train, targs$Y)
              cat(sprintf("[dbg] BLOCK A: CE loss=%.6f\n", train_loss_blockA))
            } else {
              train_loss_blockA <- mean((probs_train - targs$Y)^2, na.rm = TRUE)
              cat(sprintf("[dbg] BLOCK A: MSE loss=%.6f\n", train_loss_blockA))
            }
            train_loss_log <- c(train_loss_log, train_loss_blockA)
            
          } else if (identical(CLASSIFICATION_MODE, "binary")) {
            targs <- .build_targets(labels_epoch, n, K, CLASSIFICATION_MODE)
            stopifnot(K == 1)
            preds_bin_blockA <- as.integer(probs_train >= 0.5)
            cat("[dbg] BLOCK A: preds_bin head =", paste(utils::head(preds_bin_blockA, 6), collapse=", "), "\n")
            cat("[dbg] BLOCK A: y head        =", paste(utils::head(targs$y, 6), collapse=", "), "\n")
            train_accuracy_blockA <- mean(preds_bin_blockA == targs$y, na.rm = TRUE)
            train_accuracy_log    <- c(train_accuracy_log, train_accuracy_blockA)
            cat(sprintf("[dbg] BLOCK A: train_accuracy=%.6f\n", train_accuracy_blockA))
            
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss_blockA <- .bce_loss(probs_train, targs$y)
              cat(sprintf("[dbg] BLOCK A: BCE loss=%.6f\n", train_loss_blockA))
            } else {
              train_loss_blockA <- mean((probs_train - matrix(targs$y, ncol=1))^2, na.rm = TRUE)
              cat(sprintf("[dbg] BLOCK A: MSE loss=%.6f\n", train_loss_blockA))
            }
            train_loss_log <- c(train_loss_log, train_loss_blockA)
            
          } else if (identical(CLASSIFICATION_MODE, "regression")) {
            y_reg    <- if (is.matrix(labels_epoch)) as.numeric(labels_epoch[,1]) else as.numeric(labels_epoch)
            preds_reg <- as.numeric(probs_train[,1])
            cat("[dbg] BLOCK A: y_reg head     =", paste(utils::head(y_reg, 6), collapse=", "), "\n")
            cat("[dbg] BLOCK A: preds_reg head =", paste(utils::head(preds_reg, 6), collapse=", "), "\n")
            
            train_loss_blockA <- mean((preds_reg - y_reg)^2, na.rm = TRUE)
            cat(sprintf("[dbg] BLOCK A: MSE loss=%.6f\n", train_loss_blockA))
            train_loss_log <- c(train_loss_log, train_loss_blockA)
            
            train_accuracy_blockA <- NA_real_
            train_accuracy_log    <- c(train_accuracy_log, NA_real_)
            cat("[dbg] BLOCK A: train_accuracy=NA (regression)\n")
            
            mae  <- mean(abs(preds_reg - y_reg), na.rm = TRUE)
            vary <- stats::var(y_reg, na.rm = TRUE)
            r2   <- if (is.finite(vary) && vary > 0) 1 - train_loss_blockA / vary else NA_real_
            cat(sprintf("[dbg] BLOCK A: MAE=%.6f | R^2=%s\n", mae, ifelse(is.na(r2), "NA", sprintf("%.6f", r2))))
          } else {
            stop("Unknown CLASSIFICATION_MODE. Use 'multiclass', 'binary', or 'regression'.")
          }
          
          # keep your best tracker (already updated earlier by original block); optional reinforce:
          if (is.na(best_train_acc) || (!is.na(train_accuracy) && train_accuracy > best_train_acc)) {
            best_train_acc   <- train_accuracy
            best_epoch_train <- epoch
            cat(sprintf("[dbg] BLOCK A: new best_train_acc=%.6f at epoch=%d\n", best_train_acc, best_epoch_train))
          }
          
          # Saturation stats already computed above from predicted_output
          cat(sprintf("[dbg] BLOCK A: saturation mean=%.6f | sd=%.6f\n", mean_output, sd_output))
          
          
          
          # -----------------------------------------
          # (unchanged) plotting & filename handling
          # -----------------------------------------
          fname <- make_fname_prefix(
            do_ensemble     = isTRUE(get0("do_ensemble", ifnotfound = FALSE)),
            num_networks    = get0("num_networks", ifnotfound = NULL),
            total_models    = if (!is.null(self$ensemble)) length(self$ensemble) else get0("num_networks", ifnotfound = NULL),
            ensemble_number = if (exists("self", inherits = TRUE)) self$ensemble_number else get0("ensemble_number", ifnotfound = NULL),
            model_index     = get0("model_iter_num", ifnotfound = NULL),
            who             = "SONN"
          )
          cat("[fname probe] -> ", fname("probe.png"), "\n")
          
          if (!dir.exists("plots")) dir.create("plots", recursive = TRUE, showWarnings = FALSE)
          
          plot_title_prefix <- if (isTRUE(get0("do_ensemble", ifnotfound = FALSE))) {
            sprintf("DESONN%s SONN%s | lr: %s | lambda: %s",
                    if (!is.null(self$ensemble_number)) paste0(" ", self$ensemble_number) else "",
                    if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                    lr, lambda)
          } else {
            sprintf("SONN%s | lr: %s | lambda: %s",
                    if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                    lr, lambda)
          }
          
          pad <- function(x, n){ length(x) <- n; x }
          nA  <- max(length(train_accuracy_log), length(train_loss_log),
                     length(mean_output_log), length(sd_output_log))
          df_accsat <- data.frame(
            Epoch      = seq_len(nA),
            Accuracy   = pad(train_accuracy_log, nA),
            Loss       = pad(train_loss_log, nA),
            MeanOutput = pad(mean_output_log, nA),
            StdOutput  = pad(sd_output_log, nA)
          )
          
          if (self$viewPerEpochPlots("accuracy_plot")) {
            tryCatch({
              p <- ggplot(df_accsat, aes(x = Epoch)) +
                geom_line(aes(y = Accuracy), size = 1) +
                geom_line(aes(y = Loss),     size = 1) +
                labs(title = paste(plot_title_prefix, "— Training Accuracy (blue) & Loss (red)"),
                     y = "Value") +
                theme_minimal() +
                theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))
              out <- file.path("plots", fname("training_accuracy_loss_plot.png"))
              message("📸 save: ", out)
              ggsave(filename = out, plot = p, width = 6, height = 4, dpi = 300, device = "png")
            }, error = function(e) message("❌ accuracy_loss_plot: ", e$message))
          }
          
          if (self$viewPerEpochPlots("saturation_plot")) {
            tryCatch({
              p <- ggplot(df_accsat, aes(x = Epoch)) +
                geom_line(aes(y = MeanOutput), size = 1) +
                geom_line(aes(y = StdOutput),  size = 1) +
                labs(title = paste(plot_title_prefix, "— Output Mean & Std Dev"),
                     y = "Output Value") +
                theme_minimal() +
                theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))
              out <- file.path("plots", fname("output_saturation_plot.png"))
              message("📸 save: ", out)
              ggsave(filename = out, plot = p, width = 6, height = 4, dpi = 300, device = "png")
            }, error = function(e) message("❌ output_saturation_plot: ", e$message))
          }
          
          # 5) Regularization (ensure reg_loss_total exists)
          reg_loss_total <- 0
          if (!is.null(reg_type)) {
            if (isTRUE(self$ML_NN)) {
              for (layer in 1:self$num_layers) {
                weights_layer <- self$weights[[layer]]
                if (reg_type == "L1") {
                  reg_loss_total <- reg_loss_total + self$lambda * sum(abs(weights_layer), na.rm = TRUE)
                } else if (reg_type == "L2") {
                  reg_loss_total <- reg_loss_total + self$lambda * sum(weights_layer^2, na.rm = TRUE)
                } else if (reg_type == "L1_L2") {
                  l1_ratio <- 0.5
                  reg_loss_total <- reg_loss_total + self$lambda * (
                    l1_ratio * sum(abs(weights_layer), na.rm = TRUE) +
                      (1 - l1_ratio) * sum(weights_layer^2, na.rm = TRUE)
                  )
                } else if (reg_type == "Group_Lasso") {
                  if (is.null(self$groups) || is.null(self$groups[[layer]])) {
                    self$groups <- if (is.null(self$groups)) vector("list", self$num_layers) else self$groups
                    self$groups[[layer]] <- list(1:ncol(weights_layer))
                  }
                  reg_loss_total <- reg_loss_total + self$lambda * sum(sapply(self$groups[[layer]], function(group) {
                    sqrt(sum(weights_layer[, group]^2, na.rm = TRUE))
                  }))
                } else if (reg_type == "Max_Norm") {
                  max_norm <- 1.0
                  norm_weight <- sqrt(sum(weights_layer^2, na.rm = TRUE))
                  reg_loss_total <- reg_loss_total + self$lambda * ifelse(norm_weight > max_norm, 1, 0)
                } else if (reg_type == "Orthogonality") {
                  WtW <- t(weights_layer) %*% weights_layer
                  I <- diag(ncol(WtW))
                  reg_loss_total <- reg_loss_total + self$lambda * sum((WtW - I)^2)
                } else if (reg_type == "Sparse_Bayesian") {
                  stop("Sparse Bayesian Learning is not implemented in this code.")
                } else {
                  message("Invalid regularization type. Using 0.")
                }
              } # end for layer
            } else {
              # -------- single-layer reg --------
              weights_list  <- if (is.list(self$weights)) self$weights else list(self$weights)
              weights_layer <- weights_list[[1]]
              if (reg_type == "L1") {
                reg_loss_total <- self$lambda * sum(abs(weights_layer), na.rm = TRUE)
              } else if (reg_type == "L2") {
                reg_loss_total <- self$lambda * sum(weights_layer^2, na.rm = TRUE)
              } else if (reg_type == "L1_L2") {
                l1_ratio <- 0.5
                reg_loss_total <- self$lambda * (
                  l1_ratio * sum(abs(weights_layer), na.rm = TRUE) +
                    (1 - l1_ratio) * sum(weights_layer^2, na.rm = TRUE)
                )
              } else if (reg_type == "Group_Lasso") {
                if (is.null(self$groups)) self$groups <- list(1:ncol(weights_layer))
                reg_loss_total <- self$lambda * sum(sapply(self$groups, function(group) {
                  sqrt(sum(weights_layer[, group]^2, na.rm = TRUE))
                }))
              } else if (reg_type == "Max_Norm") {
                max_norm <- 1.0
                norm_weight <- sqrt(sum(weights_layer^2, na.rm = TRUE))
                reg_loss_total <- self$lambda * ifelse(norm_weight > max_norm, 1, 0)
              } else if (reg_type == "Orthogonality") {
                WtW <- t(weights_layer) %*% weights_layer
                I <- diag(ncol(WtW))
                reg_loss_total <- self$lambda * sum((WtW - I)^2)
              } else if (reg_type == "Sparse_Bayesian") {
                stop("Sparse Bayesian Learning is not implemented in this code.")
              } else {
                message("Invalid regularization type. Using 0.")
              }
            } # end ML_NN else
          } # end if !is.null(reg_type)
          
          
          
          # ===== Loss (train) =====
          losses[[epoch]] <- loss_function(
            predictions         = predictions,     # defined right after learn_result
            labels              = labels_epoch,    # aligned to n above
            CLASSIFICATION_MODE = CLASSIFICATION_MODE,
            reg_loss_total      = reg_loss_total,  # always defined
            loss_type           = loss_type
          )

          # ===== Initialize records and optimizer params (unchanged) =====
          if (self$ML_NN) {
            weights_record <- vector("list", self$num_layers)
            biases_record  <- vector("list", self$num_layers)
          }
          

          
          if (update_weights) {
            if (self$ML_NN) {
              for (layer in 1:self$num_layers) {
                if (!is.null(self$weights[[layer]]) && !is.null(optimizer)) {
                  
                  if (is.null(optimizer_params_weights[[layer]])) {
                    optimizer_params_weights[[layer]] <- initialize_optimizer_params(
                      optimizer,
                      list(dim(self$weights[[layer]])),
                      lookahead_step,
                      layer
                    )
                    
                    # $$$$$$$$$$$$$$$ DEBUGGING OUTPUT $$$$$$$$$$$$$$$
                    cat(">>> After initialize_optimizer_params() for layer", layer, "\n")
                    str(optimizer_params_weights[[layer]])  # structure of this specific layer’s params
                    print(names(optimizer_params_weights[[layer]]))  # list element names
                  }
                  
                  
                  
                  # Get weight gradients from learn()
                  grads_matrix <- weight_gradients[[layer]]
                  
                  # Clip weight gradient
                  grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5.0)
                  
                  
                  # --------- Apply Regularization to Weight Gradient ---------
                  if (!is.null(reg_type)) {
                    if (reg_type == "L2") {
                      weight_update <- lr * grads_matrix + self$lambda * self$weights[[layer]]
                      
                    } else if (reg_type == "L1") {
                      l1_grad <- self$lambda * sign(self$weights[[layer]])
                      weight_update <- lr * grads_matrix + l1_grad
                      
                    } else if (reg_type == "L1_L2") {
                      l1_ratio <- 0.5
                      l1_grad <- l1_ratio * sign(self$weights[[layer]])
                      l2_grad <- (1 - l1_ratio) * self$weights[[layer]]
                      weight_update <- lr * grads_matrix + self$lambda * (l1_grad + l2_grad)
                      
                    } else if (reg_type == "Group_Lasso") {
                      if (is.null(self$groups) || is.null(self$groups[[layer]])) {
                        self$groups[[layer]] <- list(1:ncol(self$weights[[layer]]))  # Default: entire layer as one group
                      }
                      
                      group_lasso_grad <- matrix(0, nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]))
                      
                      for (group in self$groups[[layer]]) {
                        group_weights <- self$weights[[layer]][, group, drop = FALSE]
                        norm_group <- sqrt(sum(group_weights^2, na.rm = TRUE)) + 1e-8
                        group_lasso_grad[, group] <- group_weights / norm_group
                      }
                      
                      weight_update <- lr * grads_matrix + self$lambda * group_lasso_grad
                    }
                    else if (reg_type == "Max_Norm") {
                      max_norm <- 1.0
                      weight_norms <- sqrt(colSums(self$weights[[layer]]^2, na.rm = TRUE))
                      clipped_weights <- self$weights[[layer]]
                      for (j in seq_along(weight_norms)) {
                        if (weight_norms[j] > max_norm) {
                          clipped_weights[, j] <- (clipped_weights[, j] / weight_norms[j]) * max_norm
                        }
                      }
                      weight_update <- lr * grads_matrix + self$lambda * (self$weights[[layer]] - clipped_weights)
                      
                    } else {
                      # Unknown reg_type → fall back
                      cat("Warning: Unknown reg_type provided. No regularization applied.\n")
                      weight_update <- lr * grads_matrix
                    }
                  } else {
                    # No regularization type specified → default to pure gradient descent
                    weight_update <- lr * grads_matrix
                  }
                  
                  
                  
                  # Apply weight update safely
                  if (all(dim(grads_matrix) == dim(self$weights[[layer]]))) {
                    self$weights[[layer]] <- self$weights[[layer]] - weight_update
                  } else if (prod(dim(self$weights[[layer]])) == 1) {
                    self$weights[[layer]] <- self$weights[[layer]] - sum(weight_update)
                  } else {
                    self$weights[[layer]] <- self$weights[[layer]] - apply(weight_update, 2, mean)
                  }
                  
                  
                  
                  
                  # --- Gradient Clipping ---
                  # max_norm <- 5
                  # grad_norm <- sqrt(sum(grads_matrix^2))
                  # 
                  # if (grad_norm > max_norm) {
                  #   grads_matrix <- grads_matrix * (max_norm / grad_norm)
                  #   cat("Clipped grads_matrix norm from", grad_norm, "to", max_norm, "at layer", layer, "\n")
                  # } else {
                  #   cat("No clipping needed. Grad norm at layer", layer, ":", grad_norm, "\n")
                  # }
                  
                  
                  
                  
                  
                  
                  
                  if (!is.null(optimizer_params_weights[[layer]]) && !is.null(optimizer)) {
                    if (optimizer == "adam") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "weights",
                        verbose = verbose
                      )
                      
                      
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    
                    else if (optimizer == "rmsprop") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "weights",
                        verbose = verbose
                      )
                      
                      
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    else if (optimizer == "sgd") {
                      
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = NA,
                        beta2 = NA,
                        epsilon = NA,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "weights",
                        verbose = verbose
                      )
                      
                      
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    else if (optimizer == "sgd_momentum") {
                      
                      
                      # Apply SGD momentum update (weights)
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,      # Used as momentum
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "weights",
                        verbose = verbose
                      )
                      
                      
                      # $$$$$$$$$$$$ Final update with optional clipping
                      # updated_weights_matrix <- updated_optimizer$updated_weights_or_biases
                      # clip_threshold <- 0.5
                      # updated_weights_matrix <- pmin(pmax(updated_weights_matrix, -clip_threshold), clip_threshold)
                      
                      # self$weights[[layer]] <- self$weights[[layer]] - updated_weights_matrix
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    else if (optimizer == "nag") {
                      # ---- Weights ----
                      updated_optimizer <- apply_optimizer_update(
                        optimizer        = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix     = grads_matrix,  # dL/dW_l
                        lr               = lr,
                        beta1            = beta1,         # momentum coefficient (NAG uses this)
                        beta2            = beta2,         # unused by NAG, kept for signature symmetry
                        epsilon          = epsilon,       # unused by NAG
                        epoch            = epoch,
                        self             = self,
                        layer            = layer,
                        target           = "weights",
                        verbose          = verbose
                      )
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                      
                    }
                    
                    else if (optimizer == "ftrl") {
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer        = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix     = grads_matrix,   # dL/dW_l
                        lr               = lr,
                        beta1            = beta1,          # optional (not core to FTRL)
                        beta2            = beta2,          # optional
                        alpha            = 0.1,            # FTRL specific
                        lambda1          = 0.01,
                        lambda2          = 0.01,
                        epsilon          = epsilon,
                        epoch            = epoch,
                        self             = self,
                        layer            = layer,
                        target           = "weights",
                        verbose          = verbose
                      )
                      
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    
                    else if (optimizer == "lamb") {
                      
                      grads_input <- if (is.list(grads_matrix)) { grads_matrix } 
                      else if (is.null(dim(grads_matrix))) { list(matrix(grads_matrix, nrow = 1, ncol = 1)) } 
                      else if (length(dim(grads_matrix)) == 1) { list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1)) } 
                      else { list(grads_matrix) }
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer        = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix     = grads_matrix,   # helper will normalize
                        lr               = lr,
                        beta1            = beta1,
                        beta2            = beta2,
                        epsilon          = epsilon,
                        epoch            = epoch,
                        self             = self,
                        layer            = layer,
                        target           = "weights",
                        verbose          = verbose
                      )
                      
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    else if (optimizer == "lookahead") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer        = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix     = weight_gradients[[layer]],
                        lr               = lr,
                        beta1            = beta1,
                        beta2            = beta2,
                        epsilon          = epsilon,
                        epoch            = epoch,
                        self             = self,
                        layer            = layer,
                        target           = "weights",
                        verbose          = verbose
                      )
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "adagrad") {
                      
                      grads_input <- if (is.list(grads_matrix)) { grads_matrix } 
                      else if (is.null(dim(grads_matrix))) { list(matrix(grads_matrix, nrow = 1, ncol = 1)) } 
                      else if (length(dim(grads_matrix)) == 1) { list(matrix(grads_matrix, nrow = 1)) } 
                      else { list(grads_matrix) }
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer        = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix     = grads_matrix,
                        lr               = lr,
                        beta1            = beta1,   # not used, kept for signature consistency
                        beta2            = beta2,   # not used, kept for signature consistency
                        epsilon          = epsilon,
                        epoch            = epoch,
                        self             = self,
                        layer            = layer,
                        target           = "weights",
                        verbose          = verbose
                      )
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "adadelta") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,   # not used by Adadelta, but passed for consistency
                        beta2 = beta2,   # not used by Adadelta
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "weights",
                        verbose = verbose
                      )
                      
                      
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    
                  }}
              }}
            else {
              # ---------------- SINGLE-LAYER NN ----------------
              if (!is.null(self$weights) && !is.null(optimizer)) {
                
                # Init optimizer params if needed
                if (is.null(optimizer_params_weights[[1]])) {
                  optimizer_params_weights[[1]] <- initialize_optimizer_params(
                    optimizer,
                    list(dim(self$weights)),
                    lookahead_step,
                    1L
                  )
                  cat(">>> SL initialize_optimizer_params done for layer 1\n")
                  str(optimizer_params_weights[[1]])
                  print(names(optimizer_params_weights[[1]]))
                }
                
                # Get weight gradients from learn()
                grads_matrix <- weight_gradients[[1]]
                
                # Clip gradient
                grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5.0)
                
                # --------- Regularization ---------
                if (!is.null(reg_type)) {
                  if (reg_type == "L2") {
                    reg_term <- self$lambda * self$weights
                    weight_update <- lr * grads_matrix + reg_term
                    
                  } else if (reg_type == "L1") {
                    reg_term <- self$lambda * sign(self$weights)
                    weight_update <- lr * grads_matrix + reg_term
                    
                  } else if (reg_type == "L1_L2") {
                    l1_ratio <- 0.5
                    reg_term <- self$lambda * (l1_ratio * sign(self$weights) + (1 - l1_ratio) * self$weights)
                    weight_update <- lr * grads_matrix + reg_term
                    
                  } else if (reg_type == "Group_Lasso") {
                    norm_weights <- sqrt(sum(self$weights^2, na.rm = TRUE)) + 1e-8
                    reg_term <- self$lambda * (self$weights / norm_weights)
                    weight_update <- lr * grads_matrix + reg_term
                    
                  } else if (reg_type == "Max_Norm") {
                    max_norm <- 1.0
                    norm_weights <- sqrt(sum(self$weights^2, na.rm = TRUE))
                    clipped_weights <- if (norm_weights > max_norm) {
                      (self$weights / norm_weights) * max_norm
                    } else {
                      self$weights
                    }
                    reg_term <- self$lambda * (self$weights - clipped_weights)
                    weight_update <- lr * grads_matrix + reg_term
                    
                  } else {
                    cat("Warning: Unknown reg_type in SL. No regularization applied.\n")
                    weight_update <- lr * grads_matrix
                  }
                } else {
                  weight_update <- lr * grads_matrix
                }
                
                # ------------------- DEBUG -------------------
                cat(">> SL grads_matrix dim:\n"); print(dim(grads_matrix))
                cat("SL grads_matrix summary:\n"); print(summary(as.vector(grads_matrix)))
                
                # ------------------- OPTIMIZER DISPATCH (SL) -------------------
                if (!is.null(optimizer_params_weights[[1]])) {
                  
                  if (optimizer %in% c("adam","rmsprop","sgd","sgd_momentum","nag","ftrl","lamb","lookahead","adagrad","adadelta")) {
                    updated_optimizer <- apply_optimizer_update(
                      optimizer        = optimizer,
                      optimizer_params = optimizer_params_weights,
                      grads_matrix     = grads_matrix,
                      lr               = lr,
                      beta1            = beta1,
                      beta2            = beta2,
                      epsilon          = epsilon,
                      epoch            = epoch,
                      self             = self,
                      layer            = 1L,          # <-- IMPORTANT: integer scalar, avoids closure clash
                      target           = "weights",
                      verbose = verbose
                    )
                    
                    # For consistency with your ML path:
                    # - Most branches set absolute weights: assign directly
                    # - Branches that return a delta (e.g., your LAMB/lookahead code) should be handled inside apply_optimizer_update
                    self$weights <- updated_optimizer$updated_weights_or_biases
                    optimizer_params_weights[[1]] <- updated_optimizer$updated_optimizer_params
                    
                    cat(">> SL updated weights summary: min =", min(self$weights),
                        ", mean =", mean(self$weights),
                        ", max =", max(self$weights), "\n")
                    
                  } else {
                    # Unknown optimizer → fall back to manual update
                    if (all(dim(grads_matrix) == dim(self$weights))) {
                      self$weights <- self$weights - weight_update
                    } else if (prod(dim(self$weights)) == 1) {
                      self$weights <- self$weights - sum(weight_update)
                    } else {
                      self$weights <- self$weights - apply(weight_update, 2, mean)
                    }
                  }
                  
                } else {
                  # Params not initialized (shouldn’t happen) → safe fallback
                  if (all(dim(grads_matrix) == dim(self$weights))) {
                    self$weights <- self$weights - weight_update
                  } else if (prod(dim(self$weights)) == 1) {
                    self$weights <- self$weights - sum(weight_update)
                  } else {
                    self$weights <- self$weights - apply(weight_update, 2, mean)
                  }
                }
              }
              
            }
          }
          # Record the updated weight matrix
          if (self$ML_NN) {
            for (layer in 1:self$num_layers) {
              weights_record[[layer]] <- as.matrix(self$weights[[layer]])
            }
          } else {
            weights_record <- as.matrix(self$weights)
          }
          
          # =========================
          # BLOCK B — Weights (Max Weight + Plot)
          #   Place THIS BLOCK UNDER your final weight-update in the epoch.
          #   (assumes `fname` was already built at top of epoch loop)
          # =========================
          
          # post-update max|W| and log
          max_weight <- tryCatch({
            stopifnot(is.list(self$weights), length(self$weights) >= 1)
            max(unlist(lapply(self$weights, function(W) max(abs(as.numeric(W)), na.rm = TRUE))), na.rm = TRUE)
          }, error = function(e) NA_real_)
          max_weight_log <- c(max_weight_log, max_weight)
          
          # DF for MaxWeight only
          df_maxw <- data.frame(
            Epoch     = seq_len(length(max_weight_log)),
            MaxWeight = max_weight_log
          )
          
          # ensure output dir + title
          if (!dir.exists("plots")) dir.create("plots", recursive = TRUE, showWarnings = FALSE)
          if (!exists("plot_title_prefix", inherits = TRUE)) {
            plot_title_prefix <- if (isTRUE(get0("do_ensemble", ifnotfound = FALSE))) {
              sprintf("DESONN%s SONN%s | lr: %s | lambda: %s",
                      if (!is.null(self$ensemble_number)) paste0(" ", self$ensemble_number) else "",
                      if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                      lr, lambda)
            } else {
              sprintf("SONN%s | lr: %s | lambda: %s",
                      if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                      lr, lambda)
            }
          }
          
          # 3) Max Weight Magnitude
          if (self$viewPerEpochPlots("max_weight_plot")) {
            tryCatch({
              p <- ggplot(df_maxw, aes(x = Epoch, y = MaxWeight)) +
                geom_line(size = 1) +
                labs(title = paste(plot_title_prefix, "— Max Weight Magnitude Over Time"),
                     y = "Max |Weight|") +
                theme_minimal() +
                theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))
              out <- file.path("plots", fname("max_weight_plot.png"))
              message("📸 save: ", out)
              ggsave(filename = out, plot = p, width = 6, height = 4, dpi = 300, device = "png")
            }, error = function(e) message("❌ max_weight_plot: ", e$message))
          }
          
          
          
          # Update biases
          if (update_biases) {
            if (self$ML_NN) {
              for (layer in 1:self$num_layers) {
                if (!is.null(self$biases[[layer]]) && !is.null(optimizer)) {
                  
                  # Initialize optimizer parameters only if not already done
                  if (is.null(optimizer_params_biases[[layer]])) {
                    optimizer_params_biases[[layer]] <- initialize_optimizer_params(
                      optimizer,
                      list(dim(as.matrix(self$biases[[layer]]))),
                      lookahead_step,
                      layer
                    )
                  }
                  
                  # Get bias gradients from learn()
                  grads_matrix <- bias_gradients[[layer]]
                  
                  # Clip bias gradient
                  grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5)
                  
                  
                  
                  # --- Align dimensions if needed ---
                  bias_shape <- dim(as.matrix(self$biases[[layer]]))
                  grad_shape <- dim(grads_matrix)
                  
                  if (!all(bias_shape == grad_shape)) {
                    if (prod(grad_shape) == 1) {
                      grads_matrix <- matrix(grads_matrix, nrow = bias_shape[1], ncol = bias_shape[2])
                    } else if (prod(bias_shape) == 1) {
                      self$biases[[layer]] <- matrix(self$biases[[layer]], nrow = grad_shape[1], ncol = grad_shape[2])
                    } else {
                      grads_matrix <- matrix(rep(grads_matrix, length.out = prod(bias_shape)), 
                                             nrow = bias_shape[1], ncol = bias_shape[2])
                    }
                  }
                  
                  # --------- Apply Regularization to Bias Gradient ---------
                  if (!is.null(reg_type)) {
                    if (reg_type == "L2") {
                      reg_term <- self$lambda * self$biases[[layer]]
                      bias_update <- lr * grads_matrix + reg_term
                      
                    } else if (reg_type == "L1") {
                      reg_term <- self$lambda * sign(self$biases[[layer]])
                      bias_update <- lr * grads_matrix + reg_term
                      
                    } else if (reg_type == "L1_L2") {
                      l1_ratio <- 0.5
                      l1_grad <- l1_ratio * sign(self$biases[[layer]])
                      l2_grad <- (1 - l1_ratio) * self$biases[[layer]]
                      reg_term <- self$lambda * (l1_grad + l2_grad)
                      bias_update <- lr * grads_matrix + reg_term
                      
                    } else if (reg_type == "Group_Lasso") {
                      norm_bias <- sqrt(sum(self$biases[[layer]]^2, na.rm = TRUE)) + 1e-8
                      reg_term <- self$lambda * (self$biases[[layer]] / norm_bias)
                      bias_update <- lr * grads_matrix + reg_term
                    }
                    else if (reg_type == "Max_Norm") {
                      max_norm <- 1.0
                      norm_bias <- sqrt(sum(self$biases[[layer]]^2, na.rm = TRUE))
                      clipped_bias <- if (norm_bias > max_norm) {
                        (self$biases[[layer]] / norm_bias) * max_norm
                      } else {
                        self$biases[[layer]]
                      }
                      reg_term <- self$lambda * (self$biases[[layer]] - clipped_bias)
                      bias_update <- lr * grads_matrix + reg_term
                      
                    } else {
                      cat("Warning: Unknown reg_type in ML bias update. No regularization applied.\n")
                      bias_update <- lr * grads_matrix
                    }
                    
                  } else {
                    # Default: No regularization
                    bias_update <- lr * grads_matrix
                  }
                  
                  
                  
                  # Apply bias update safely
                  if (all(dim(grads_matrix) == dim(self$biases[[layer]]))) {
                    self$biases[[layer]] <- self$biases[[layer]] - bias_update
                  } else if (prod(dim(self$biases[[layer]])) == 1) {
                    self$biases[[layer]] <- self$biases[[layer]] - sum(bias_update)
                  } else {
                    self$biases[[layer]] <- self$biases[[layer]] - apply(bias_update, 2, mean)
                  }
                  
                  
                  
                  # Apply optimizer update if optimizer is specified
                  if (!is.null(optimizer_params_biases[[layer]]) && !is.null(optimizer)) {
                    if (optimizer == "adam") {
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                      
                    }
                    else if (optimizer == "rmsprop") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = bias_gradients[[layer]],
                        lr = lr,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    else if (optimizer == "sgd") {
                      
                      
                      # Apply SGD update (extra args passed for compatibility)
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,         # Not used by SGD, but kept for unified signature
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    
                    else if (optimizer == "sgd_momentum") {
                      
                      # Apply SGD momentum update (structured like SGD block)
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,         # Used as momentum in this case
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    
                    else if (optimizer == "nag") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,  # Only beta1 needed for NAG
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    else if (optimizer == "ftrl") {
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer         = optimizer,
                        optimizer_params  = optimizer_params_biases,
                        grads_matrix      = grads_matrix,
                        lr                = lr,
                        alpha             = 0.1,
                        beta1             = 1.0,
                        lambda1           = 0.01,
                        lambda2           = 0.01,
                        epsilon           = epsilon,
                        epoch             = epoch,
                        self              = self,
                        layer             = layer,
                        target            = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "lamb") {
                      
                      # Standardize grads_matrix to list of matrices
                      grads_input <- if (is.list(grads_matrix)) {
                        grads_matrix
                      } else if (is.null(dim(grads_matrix))) {
                        list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1))
                      } else if (length(dim(grads_matrix)) == 1) {
                        list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1))
                      } else {
                        list(grads_matrix)
                      }
                      
                      # Apply LAMB update through unified optimizer interface
                      updated_optimizer <- apply_optimizer_update(
                        optimizer         = optimizer,
                        optimizer_params  = optimizer_params_biases,
                        grads_matrix      = grads_matrix,
                        lr                = lr,
                        beta1             = beta1,
                        beta2             = beta2,
                        epsilon           = epsilon,
                        epoch             = epoch,
                        self              = self,
                        layer             = layer,
                        target            = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    
                    else if (optimizer == "lookahead") {
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        layer = layer,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    
                    else if (optimizer == "adagrad") {
                      
                      grads_matrix <- if (is.null(dim(errors[[layer]]))) {
                        matrix(errors[[layer]], nrow = 1)  # Convert vector to 1-row matrix
                      } else {
                        errors[[layer]]  # Already matrix, use directly
                      }
                      grads_matrix <- colSums(grads_matrix)  # Now always works
                      # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                      
                      # Unified call to apply_optimizer_update
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,  # not used by Adagrad but passed for uniformity
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      # Update optimizer state
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    
                    else if (optimizer == "adadelta") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = grads_matrix, #colSums(errors[[layer]])
                        lr = lr,
                        beta1 = beta1,  # unused by Adadelta but passed for uniformity
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases",
                        verbose = verbose
                      )
                      
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                      
                    }
                    
                    
                  } }}}else {# ---------------- SINGLE-LAYER BIAS UPDATE ----------------
                    cat("Single Layer Bias Update\n")
                    
                    # 1) Ensure biases matrix [1 x n_units]
                    if (is.null(self$biases)) stop("Biases are NULL in single-layer mode.")
                    if (!is.matrix(self$biases)) self$biases <- matrix(as.numeric(self$biases), nrow = 1)
                    
                    # 2) Ensure optimizer params list + init slot 1
                    if (is.null(optimizer_params_biases)) optimizer_params_biases <- list()
                    if (!is.null(optimizer) && is.null(optimizer_params_biases[[1]])) {
                      optimizer_params_biases[[1]] <- initialize_optimizer_params(
                        optimizer,
                        list(dim(self$biases)),
                        lookahead_step,
                        1L
                      )
                      cat(">>> SL initialize_optimizer_params (bias) done for layer 1\n")
                      str(optimizer_params_biases[[1]])
                      print(names(optimizer_params_biases[[1]]))
                    }
                    
                    # 3) Gradient from errors (per-unit mean)
                    bias_grad <- colMeans(errors[[1]], na.rm = TRUE)
                    
                    # shape to [1 x n_units]
                    bias_grad <- matrix(rep(bias_grad, length.out = ncol(self$biases)), nrow = 1)
                    
                    # optional clip (kept consistent with weights)
                    bias_grad <- clip_gradient_norm(bias_grad, max_norm = 5.0)
                    
                    # Debug
                    cat("SL bias_grad dim:", paste(dim(bias_grad), collapse = "x"), "\n")
                    cat("SL bias_grad summary:\n"); print(summary(as.vector(bias_grad)))
                    
                    # 4) Optimizer dispatch (preferred path)
                    if (!is.null(optimizer) && !is.null(optimizer_params_biases[[1]])) {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer        = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix     = bias_grad,   # pass pure gradient (not lr*grad + reg)
                        lr               = lr,
                        beta1            = beta1,
                        beta2            = beta2,
                        epsilon          = epsilon,
                        epoch            = epoch,
                        self             = self,
                        layer            = 1L,          # <= important: integer index, avoids 'closure' clash
                        target           = "biases",
                        verbose = verbose
                      )
                      
                      # Most optimizers return absolute updated params in your codebase:
                      self$biases <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_biases[[1]] <- updated_optimizer$updated_optimizer_params
                      
                      cat(">> SL updated biases summary: min =", min(self$biases),
                          ", mean =", mean(self$biases),
                          ", max =", max(self$biases), "\n")
                      
                    } else {
                      # 5) Manual / fallback update with regularization (mirrors weights SL fallback)
                      
                      bias_update <- lr * bias_grad
                      
                      if (!is.null(reg_type)) {
                        if (reg_type == "L2") {
                          reg_term <- self$lambda * self$biases
                          bias_update <- bias_update + reg_term
                          
                        } else if (reg_type == "L1") {
                          reg_term <- self$lambda * sign(self$biases)
                          bias_update <- bias_update + reg_term
                          
                        } else if (reg_type == "L1_L2") {
                          l1_ratio <- 0.5
                          l1_grad  <- l1_ratio * sign(self$biases)
                          l2_grad  <- (1 - l1_ratio) * self$biases
                          bias_update <- bias_update + self$lambda * (l1_grad + l2_grad)
                          
                        } else if (reg_type == "Group_Lasso") {
                          norm_bias <- sqrt(sum(self$biases^2, na.rm = TRUE)) + 1e-8
                          reg_term  <- self$lambda * (self$biases / norm_bias)
                          bias_update <- bias_update + reg_term
                          
                        } else if (reg_type == "Max_Norm") {
                          max_norm  <- 1.0
                          norm_bias <- sqrt(sum(self$biases^2, na.rm = TRUE))
                          clipped_bias <- if (norm_bias > max_norm) {
                            (self$biases / norm_bias) * max_norm
                          } else {
                            self$biases
                          }
                          reg_term <- self$lambda * (self$biases - clipped_bias)
                          bias_update <- bias_update + reg_term
                          
                        } else {
                          cat("Warning: Unknown reg_type in SL bias update. No regularization applied.\n")
                          # keep bias_update as lr * grad
                        }
                      }
                      
                      # Final manual apply
                      self$biases <- self$biases - bias_update
                    }
                    
                    
                  }
          }
          
          if(use_biases) {
            if (self$ML_NN) {
              for (layer in 1:self$num_layers) {
                biases_record[[layer]] <- as.matrix(self$biases[[layer]])
              }
            } else {
              
              biases_record <- as.matrix(self$biases)
            }
            
          } else {
            
            if (self$ML_NN) {
              for (layer in 1:self$num_layers) {
                
                self$biases[[layer]] <- 0
                biases_record[[layer]] <- as.matrix(self$biases[[layer]])
              }
            } else {
              self$biases <- 0
              biases_record <- as.matrix(self$biases)
            }
          }
          
          
          # ===== Validation-or-Training Metrics Block (predict + safe shape handling) =====
          # Runs validation metrics when explicitly enabled and both X_validation/y_validation exist.
          # Otherwise, if validation_metrics == FALSE, falls back to training metrics using X_train/y_train.
          
          ## ================== BEGIN PATCHED BLOCK (no predicted_output_val / predicted_output_train_reg inits) ==================
          
          # --- init holders (keep last vs best separate) ---
          last_val_probs    <- NULL
          last_val_labels   <- NULL
          last_train_probs  <- NULL
          last_train_labels <- NULL
          
          # keep full predict()-style objects from the LAST epoch
          last_val_predict   <- NULL
          last_train_predict <- NULL
          
          # --- init "best" holders so they ALWAYS exist in-scope ---
          best_val_probs            <- NULL
          best_val_labels           <- NULL
          best_val_prediction_time  <- NA_real_
          best_val_acc              <- NA_real_
          best_val_epoch            <- NA_integer_
          best_val_n_eff            <- NA_integer_         # lock the slice length used for best
          
          best_train_acc            <- if (exists("best_train_acc")) best_train_acc else NA_real_
          best_train_epoch          <- if (exists("best_train_epoch")) best_train_epoch else NA_integer_
          
          # Best weights/biases (snapshot)
          best_weights <- NULL
          best_biases  <- NULL
          
          if (!is.null(X_validation) && !is.null(y_validation) && isTRUE(validation_metrics)) {
            
            # -------- Validation path --------
            predicted_output_val <- tryCatch(
              self$predict(
                Rdata                = X_validation,
                weights              = if (isTRUE(self$ML_NN)) weights_record else self$weights,
                biases               = if (isTRUE(self$ML_NN)) biases_record  else self$biases,
                activation_functions = activation_functions,
                verbose = verbose,
                debug = debug
              ),
              error = function(e) {
                message("Validation predict failed: ", e$message)
                NULL
              }
            )
            
            if (!is.null(predicted_output_val)) {
              # Pull probabilities / logits
              probs_val <- if (!is.null(predicted_output_val$predicted_output)) {
                predicted_output_val$predicted_output
              } else {
                predicted_output_val
              }
              probs_val <- as.matrix(probs_val)  # ensure matrix
              
              # Use nrow/ncol—not length()
              n_val <- nrow(probs_val)
              K_val <- max(1L, ncol(probs_val))
              
              cat("Debug (val): nrow(X_val)=", nrow(X_validation),
                  " nrow(probs_val)=", n_val,
                  " ncol(probs_val)=", K_val, "\n")
              
              # --- Normalize y_validation to vector or one-hot matrix ---
              if (is.data.frame(y_validation)) {
                y_val_vec <- y_validation[[1]]
                len_y <- length(y_val_vec)
              } else if (is.matrix(y_validation)) {
                if (ncol(y_validation) == 1L) {
                  y_val_vec <- y_validation[, 1]; len_y <- length(y_val_vec)
                } else {
                  y_val_vec <- y_validation;      len_y <- nrow(y_validation)
                }
              } else {
                y_val_vec <- y_validation;        len_y <- length(y_val_vec)
              }
              
              # Align by trimming only (no recycling)
              n_eff <- min(nrow(X_validation), n_val, len_y)
              if (n_eff <= 0) stop("Validation sizes yield zero effective rows.")
              
              probs_val <- probs_val[seq_len(n_eff), , drop = FALSE]
              if (is.matrix(y_val_vec) && ncol(y_val_vec) > 1L) {
                y_val_epoch <- y_val_vec[seq_len(n_eff), , drop = FALSE]
              } else {
                y_val_epoch <- y_val_vec[seq_len(n_eff)]
              }
              
              # --- capture LAST-epoch VAL outputs BEFORE computing/overwriting with best snapshot ---
              last_val_probs   <- probs_val
              last_val_labels  <- y_val_epoch
              last_val_predict <- predicted_output_val   # keep full predict()-style list
              cat("[DBG] Captured LAST-epoch validation probs/labels and predict() list\n")
              
              # Build targets (for loss only; acc handled by accuracy())
              targs_val <- .build_targets(y_val_epoch, n_eff, K_val, CLASSIFICATION_MODE)
              
              # --- ACCURACY via helper (ALWAYS 0.5 threshold for binary) ---
              val_acc <- accuracy(
                SONN                = self,
                Rdata               = X_validation[seq_len(n_eff), , drop = FALSE],
                labels              = y_val_epoch,
                CLASSIFICATION_MODE = CLASSIFICATION_MODE,
                predicted_output    = probs_val,
                verbose             = isTRUE(debug)
              )
              
              # --- LOSS (unchanged)
              if (identical(CLASSIFICATION_MODE, "multiclass")) {
                stopifnot(K_val >= 2)
                val_loss <- if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  .ce_loss_multiclass(probs_val, targs_val$Y)
                } else {
                  mean((probs_val - targs_val$Y)^2, na.rm = TRUE)
                }
              } else { # binary
                val_loss <- if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  .bce_loss(probs_val, targs_val$y)
                } else {
                  mean((probs_val - matrix(targs_val$y, ncol = 1))^2, na.rm = TRUE)
                }
              }
              
              # Log metrics
              val_accuracy_log <- c(val_accuracy_log, val_acc)
              val_loss_log     <- c(val_loss_log,     val_loss)
              
              self$best_weights <- NULL  # keep original behavior
              self$best_biases  <- NULL
              
              # Track best (by validation accuracy at fixed 0.5 threshold)
              if (is.na(best_val_acc) || (!is.na(val_acc) && val_acc > best_val_acc)) {
                best_val_acc       <- val_acc
                best_val_epoch     <- epoch
                best_val_n_eff     <- n_eff          # lock the exact slice length used
                
                # Snapshot best params (deep copy)
                if (isTRUE(self$ML_NN)) {
                  best_weights <- lapply(self$weights, as.matrix)
                  best_biases  <- lapply(self$biases,  as.matrix)
                } else {
                  best_weights <- as.matrix(self$weights)
                  best_biases  <- as.matrix(self$biases)
                }
                
                # capture this epoch's validation prediction_time if present
                if (!is.null(predicted_output_val$prediction_time)) {
                  best_val_prediction_time <- predicted_output_val$prediction_time
                }
                
                # definitive best probs/labels at this epoch (already sliced to n_eff)
                best_val_probs  <- as.matrix(probs_val)
                best_val_labels <- if (is.matrix(y_val_epoch)) {
                  y_val_epoch
                } else {
                  matrix(y_val_epoch, ncol = if (identical(CLASSIFICATION_MODE, "multiclass")) K_val else 1L)
                }
                
                cat("New best model saved at epoch", epoch,
                    "| Val Acc (0.5 thr):", round(100 * val_acc, 2), "%\n")
              }
            }
            
          } else if (!is.null(X_train) && !is.null(y_train) && isFALSE(validation_metrics)) {
            
            # -------- Training path (when validation metrics are disabled) --------
            predicted_output_train <- tryCatch(
              self$predict(
                Rdata                = X_train,  # use X_train (not X) to avoid ambiguity
                weights              = if (isTRUE(self$ML_NN)) weights_record else self$weights,
                biases               = if (isTRUE(self$ML_NN)) biases_record  else self$biases,
                activation_functions = activation_functions,
                verbose = verbose,
                debug = debug
              ),
              error = function(e) {
                message("Training predict failed: ", e$message)
                NULL
              }
            )
            
            if (!is.null(predicted_output_train)) {
              
              # Pull probabilities / logits
              probs_tr <- if (!is.null(predicted_output_train$predicted_output)) {
                predicted_output_train$predicted_output
              } else {
                predicted_output_train
              }
              probs_tr <- as.matrix(probs_tr)  # ensure matrix
              
              n_tr <- nrow(probs_tr)
              K_tr <- max(1L, ncol(probs_tr))
              
              cat("Debug (train): nrow(X_train)=", nrow(X_train),
                  " nrow(probs_tr)=", n_tr,
                  " ncol(probs_tr)=", K_tr, "\n")
              
              # --- Normalize y_train to vector or one-hot matrix ---
              if (is.data.frame(y_train)) {
                y_tr_vec <- y_train[[1]]; len_y_tr <- length(y_tr_vec)
              } else if (is.matrix(y_train)) {
                if (ncol(y_train) == 1L) {
                  y_tr_vec <- y_train[, 1]; len_y_tr <- length(y_tr_vec)
                } else {
                  y_tr_vec <- y_train;      len_y_tr <- nrow(y_train)
                }
              } else {
                y_tr_vec <- y_train;        len_y_tr <- length(y_tr_vec)
              }
              
              # Align by trimming only (no recycling)
              n_eff_tr <- min(nrow(X_train), n_tr, len_y_tr)
              if (n_eff_tr <= 0) stop("Training sizes yield zero effective rows.")
              
              probs_tr <- probs_tr[seq_len(n_eff_tr), , drop = FALSE]
              if (is.matrix(y_tr_vec) && ncol(y_tr_vec) > 1L) {
                y_tr_epoch <- y_tr_vec[seq_len(n_eff_tr), , drop = FALSE]
              } else {
                y_tr_epoch <- y_tr_vec[seq_len(n_eff_tr)]
              }
              
              # --- capture LAST-epoch TRAIN outputs BEFORE best-snapshot logic ---
              last_train_probs    <- probs_tr
              last_train_labels   <- y_tr_epoch
              last_train_predict  <- predicted_output_train   # keep full predict()-style list
              cat("[DBG] Captured LAST-epoch training probs/labels and predict() list\n")
              
              # Build targets (for loss only; acc handled by accuracy())
              targs_tr <- .build_targets(y_tr_epoch, n_eff_tr, K_tr, CLASSIFICATION_MODE)
              
              # --- ACCURACY via helper (always 0.5 for binary) ---
              tr_acc <- accuracy(
                SONN                = self,
                Rdata               = X_train[seq_len(n_eff_tr), , drop = FALSE],
                labels              = y_tr_epoch,
                CLASSIFICATION_MODE = CLASSIFICATION_MODE,
                predicted_output    = probs_tr,
                verbose             = isTRUE(debug)
              )
              
              # --- LOSS (unchanged)
              if (identical(CLASSIFICATION_MODE, "multiclass")) {
                stopifnot(K_tr >= 2)
                tr_loss <- if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  .ce_loss_multiclass(probs_tr, targs_tr$Y)
                } else {
                  mean((probs_tr - targs_tr$Y)^2, na.rm = TRUE)
                }
              } else { # binary
                tr_loss <- if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  .bce_loss(probs_tr, targs_tr$y)
                } else {
                  mean((probs_tr - matrix(targs_tr$y, ncol = 1))^2, na.rm = TRUE)
                }
              }
              
              # Log metrics (training)
              train_accuracy_log <- c(train_accuracy_log, tr_acc)
              train_loss_log     <- c(train_loss_log,     tr_loss)
              
              self$best_weights <- NULL  # keep original behavior
              self$best_biases  <- NULL
              
              # Track "best" by training accuracy when validation is disabled
              if (is.na(best_train_acc) || (!is.na(tr_acc) && tr_acc > best_train_acc)) {
                best_train_acc   <- tr_acc
                best_train_epoch <- epoch
                
                # Snapshot best params
                if (isTRUE(self$ML_NN)) {
                  best_weights <- lapply(self$weights, as.matrix)
                  best_biases  <- lapply(self$biases,  as.matrix)
                } else {
                  best_weights <- as.matrix(self$weights)
                  best_biases  <- as.matrix(self$biases)
                }
                
                # (Optional) expose best train artifacts
                assign("best_train_probs",  probs_tr,   envir = .GlobalEnv)
                assign("best_train_labels", y_tr_epoch, envir = .GlobalEnv)
                
                cat("New best (train) model saved at epoch", epoch,
                    "| Train Acc (0.5 thr):", round(100 * tr_acc, 2), "%\n")
              }
            }
          }
          
          # -------- FINALIZE WITH BEST SNAPSHOT (critical fix) --------
          # After the epoch loop completes, ensure the outputs you return are from the BEST params.
          use_best <- !is.null(best_weights) && !is.null(best_biases)
          
          if (isTRUE(use_best)) {
            # recompute TRAIN preds with best snapshot (train stays separate)
            if (!is.null(X_train)) {
              pred_train_best <- tryCatch(
                self$predict(
                  Rdata                = X_train,
                  weights              = best_weights,
                  biases               = best_biases,
                  activation_functions = activation_functions,
                  verbose = FALSE, debug = FALSE
                ),
                error = function(e) NULL
              )
              if (!is.null(pred_train_best)) {
                predicted_output_train_reg <- pred_train_best
              }
            }
            
            # recompute VAL preds with best snapshot when applicable (val stays separate)
            if (!is.null(X_validation) && !is.null(y_validation) && isTRUE(validation_metrics)) {
              pred_val_best <- tryCatch(
                self$predict(
                  Rdata                = X_validation,
                  weights              = best_weights,
                  biases               = best_biases,
                  activation_functions = activation_functions,
                  verbose = FALSE, debug = FALSE
                ),
                error = function(e) NULL
              )
              if (!is.null(pred_val_best)) {
                predicted_output_val <- pred_val_best
                if (!is.null(pred_val_best$prediction_time)) {
                  best_val_prediction_time <- pred_val_best$prediction_time
                }
                
                probs_val_best <- if (!is.null(pred_val_best$predicted_output)) {
                  pred_val_best$predicted_output
                } else {
                  pred_val_best
                }
                probs_val_best <- as.matrix(probs_val_best)
                
                # Recreate label vector/matrix
                if (is.data.frame(y_validation)) {
                  y_val_vec2 <- y_validation[[1]]
                } else if (is.matrix(y_validation)) {
                  y_val_vec2 <- if (ncol(y_validation) == 1L) y_validation[, 1] else y_validation
                } else {
                  y_val_vec2 <- y_validation
                }
                
                # Use the SAME slice length captured at best epoch when possible
                n_eff2 <- min(nrow(X_validation), nrow(probs_val_best), if (is.matrix(y_val_vec2)) nrow(y_val_vec2) else length(y_val_vec2))
                if (is.finite(best_val_n_eff) && !is.na(best_val_n_eff)) {
                  n_eff2 <- min(n_eff2, best_val_n_eff)
                }
                if (n_eff2 <= 0) stop("Validation sizes yield zero effective rows (best snapshot).")
                
                probs_val_best <- probs_val_best[seq_len(n_eff2), , drop = FALSE]
                if (is.matrix(y_val_vec2) && !is.null(dim(y_val_vec2)) && ncol(y_val_vec2) > 1L) {
                  y_val_epoch2 <- y_val_vec2[seq_len(n_eff2), , drop = FALSE]
                } else {
                  y_val_epoch2 <- y_val_vec2[seq_len(n_eff2)]
                }
                
                best_val_probs  <- probs_val_best
                best_val_labels <- if (is.matrix(y_val_epoch2)) {
                  y_val_epoch2
                } else {
                  matrix(y_val_epoch2, ncol = if (identical(CLASSIFICATION_MODE, "multiclass")) ncol(best_val_probs) else 1L)
                }
                
              }
            }
            
            cat(sprintf("[BEST-SNAPSHOT] using %s epoch=%s | best_val_acc=%.7f | thr=0.5 | n_eff=%s\n",
                        if (isTRUE(validation_metrics)) "validation-best" else "train-best",
                        if (isTRUE(validation_metrics)) as.character(best_val_epoch) else as.character(best_train_epoch),
                        if (is.na(best_val_acc)) NA_real_ else best_val_acc,
                        as.character(best_val_n_eff)))
          } else {
            cat("[BEST-SNAPSHOT] no best snapshot captured; returning last evaluated predictions.\n")
          }
          
          # --- SINGLE CRITICAL DEBUG+STOP (right before performance assembly) ---
          if (isTRUE(validation_metrics) && isTRUE(use_best) && (is.null(best_val_probs) || is.null(best_val_labels) || is.na(best_val_acc))) {
            stop("[STOP] BEST snapshot selected for validation but best_val_probs/labels/acc are incomplete. Investigate best-val capture.")
          }
          
          # === CRITICAL: Keep predict()-style list in predicted_output_train_reg,
          #     but choose it based on LAST-epoch context using your preferred vars ===
          if (isTRUE(train) && isTRUE(validation_metrics)) {
            # Training that evaluated on validation → use LAST-epoch VAL predict list
            predicted_output_train_reg <- last_val_predict
          } else if (!isTRUE(train) && !isTRUE(validation_metrics)) {
            # Non-training run without validation metrics → use LAST-epoch TRAIN predict list
            predicted_output_train_reg <- last_train_predict
          } else {
            # leave as-is from earlier logic (or NULL if none)
            predicted_output_train_reg <- predicted_output_train_reg
          }
          
          ## =================== END PATCHED BLOCK ===================
          
          
    


          
          
        }            
        cat(sprintf("\nBest Training Accuracy: %.2f%% at Epoch %d\n", 100 * best_train_acc, best_epoch_train))
        
        
        cat("Best Epoch (validation accuracy):", best_val_epoch, "\n")
        cat("Best Validation Accuracy:", round(100 * best_val_acc, 2), "%\n")
        
      }else {predicted_output_train_reg_prediction_time <- NULL
      weights_record <- NULL
      biases_record <- NULL
      dim_hidden_layers <- NULL}
      
      
      
      

      
      # probe_last_layer(self$weights, self$biases, y, tag="TRAIN-END")
      
      
      if (train) {
        if (self$ML_NN) {
          for (layer in 1:self$num_layers) {
            cat(sprintf("Layer %d weights summary:\n", layer))
            print(summary(as.vector(weights_record[[layer]])))
            
            cat(sprintf("Layer %d biases summary:\n", layer))
            print(summary(as.vector(biases_record[[layer]])))
          }
        } else {
          cat("Single-layer weights summary:\n")
          print(summary(as.vector(weights_record)))
          
          cat("Single-layer biases summary:\n")
          print(summary(as.vector(biases_record)))
        }
      }
      
      # Note: The below loop was originally meant to save per-ensemble weights/biases,
      # but it incorrectly grabs only the final model's weights every time.
      # We now pass the final trained weights/biases directly through the return object
      # of `train_with_l2_regularization()` instead.
      
      # Dynamic assignment of weights and biases records
      # for (i in 1:length(self$ensemble)) {
      #   weight_record_name <- paste0("weights_record_", i)
      #   bias_record_name <- paste0("biases_record_", i)
      #   assign(weight_record_name, as.matrix(self$weights), envir = .GlobalEnv)
      #   assign(bias_record_name, as.matrix(self$biases), envir = .GlobalEnv)
      # }
      
      
      # if (ML_NN) {
      #     for (m in 1:length(self$weights)) {
      #         weight_name <- ifelse(m == 1, "weights", paste0("weights", m))
      #         cat("Weight matrix", weight_name, ":\n")
      #
      #         if (length(self$weights) >= m && !is.null(self$weights[[m]])) {
      #             print(self$weights[[m]])
      #         } else {
      #             cat("Weight matrix", weight_name, "is NULL or not initialized.\n")
      #         }
      #         cat("\n")
      #     }
      # } else {
      #     print(self$weights)
      # }
      
      
      # Print the loss for the current epoch
      # print(paste("Loss for epoch", epoch, ":", round(losses[epoch], 6)))
      
      

      if(never_ran_flag == TRUE) {
        # Find the index where the validation loss starts to increase
        optimal_epoch <- which(diff(losses) > 0)[1]
        
        # Check if loss at any epoch is greater than loss at epoch 0
        loss_increase_flag <- any(losses > losses[1])
        
      }else if(never_ran_flag == FALSE){ #unlist due to initialization of these losses under train_with_l2_regularization
        losses <- unlist(losses)
        optimal_epoch <- which(diff(losses) > 0)[1]
        
        # Check if loss at any epoch is greater than loss at epoch 0
        loss_increase_flag <- any(losses > losses[1])
        
        # Print the last element in list because we ran before and it is the optimal so it won't show on the graph (unless you explicitly define it like this)
        # if(is.na(optimal_epoch)){
        #     optimal_epoch <- losses[[length(losses)]]
        # }
      }
      
      
      if(!predict_models){
        tryCatch({
          if (any(is.na(losses))) {
            message("NA detected in losses. Cannot return loss_status.")
          } else if (any(losses > 10000) || isTRUE(loss_increase_flag)) {
            assign("loss_status", 'exceeds_10000', envir = .GlobalEnv)
          } else {
            assign("loss_status", 'ok', envir = .GlobalEnv)
          }
        }, error = function(e) {
          message("Error detected: ", e$message)
          message("Cannot return loss_status.")
        })
      }
      
      
      
      # Record the loss at the optimal epoch, or fall back to the last epoch's loss if no optimal epoch was found
      lossesatoptimalepoch <- if (is.na(optimal_epoch)) tail(losses, 1) else losses[optimal_epoch]
      
      
      
      # Check if all values are finite
      tryCatch({
        if (all(is.finite(losses))) {
          # Plot the loss over epochs
          plot_epochs <- plot(losses, type = 'l', main = paste('Loss Over Epochs for DESONN', ensemble_number, 'SONN', model_iter_num, 'lr:', lr, 'lambda:', lambda), xlab = 'Epoch', ylab = 'Loss', col = 'turquoise', lwd = 2.0)
          
          # Add a point or line indicating the optimal epoch
          points(optimal_epoch, losses[optimal_epoch], col = 'limegreen', pch = 16)
          
          # Calculate the percentage for the offset
          offset_percent <- 6  # change this to the percentage you want
          offset <- max(losses) * offset_percent / 100
          
          # Add text for the equation
          eq <- paste("Optimal Epoch:", optimal_epoch, "\nLoss:", round(losses[optimal_epoch], 2))
          text(optimal_epoch + 1.65, losses[optimal_epoch] + offset, eq, pos = 4, col = "limegreen", adj = 0)
          
          # Save the plot # not good wil update later
          # saveRDS(plot_epochs, paste("plot_epochs_DESONN", ensemble_number, "SONN", model_iter_num, ".rds", sep = ""))
        } else {
          # Print a warning message and skip the plotting
          print(paste("Warning: Non-finite values detected in losses for DESONN", ensemble_number, "SONN", model_iter_num, ". Skipping plot."))
        }
      }, error = function(e) {
        # Handle the specific error you want to bypass
        if (grepl("figure margins too large", e$message)) {
          print("Warning: Plotting error (figure margins too large). Skipping plot.")
        } else {
          # Print any other unexpected errors
          print(paste("Unexpected error:", e$message))
        }
      })
      
      
      end_time <- Sys.time()
      
      # Calculate the training time
      training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

      # Return the predicted output
      return(list(predicted_output_l2 = predicted_output_train_reg, training_time = training_time, best_train_acc = best_train_acc, best_epoch_train = best_epoch_train, best_val_acc = best_val_acc, best_val_epoch = best_val_epoch, best_val_prediction_time = best_val_prediction_time, learn_output = learn_result$learn_output, learn_time = learn_result$learn_time, learn_dim_hidden_layers = learn_result$dim_hidden_layers, learn_hidden_outputs = learn_result$hidden_outputs, learn_grads_matrix = learn_result$grads_matrix, learn_bias_gradients = learn_result$bias_gradients, learn_errors = learn_result$errors, optimal_epoch = optimal_epoch, weights_record = weights_record, biases_record = biases_record, best_weights_record = best_weights, best_biases_record = best_biases, lossesatoptimalepoch = lossesatoptimalepoch, loss_increase_flag = loss_increase_flag, loss_status = loss_status, dim_hidden_layers = dim_hidden_layers, predicted_output_val = predicted_output_val, best_val_probs = best_val_probs, best_val_labels = best_val_labels))
    },
    # # Method to calculate performance and relevance
    # calculate_metrics = function(Rdata) {
    #
    #     # Calculate performance
    #     self$performance <- calculate_performance(self, Rdata)
    #     # Calculate relevance
    #     self$relevance <- calculate_relevance(self, Rdata)
    # },
    # Method to get the neuron's position in the map
    get_neuron_position = function(neuron_index) {
      if (is.null(self$map)) {
        stop("Map has not been initialized.")
      }
      return(which(self$map == neuron_index, arr.ind = TRUE))
      
    },# Custom print method
    print = function() {
      # cat("SONN Object\n")
      # cat("Weights:\n")
      # print(self$weights)
      # cat("Biases:\n")
      # print(self$biases)
      # cat("Map:\n")
      # print(self$map)
      # Print other properties and methods as needed
    }
  )
)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#________  ___________ _________________    _______    _______   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#\______ \ \_   _____//   _____/\_____  \   \      \   \      \  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# |    |  \ |    __)_ \_____  \  /   |   \  /   |   \  /   |   \ $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# |    `   \|        \/        \/    |    \/    |    \/    |    \$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#/_______  /_______  /_______  /\_______  /\____|__  /\____|__ / $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#    \/        \/        \/         \/         \/         \/     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Step 2: Define the Dynamic Ensemble of Self-Organizing Neural Networks (DESONN) class
DESONN <- R6Class(
  "DESONN",
  lock_objects = FALSE,
  public = list(
    ensemble = NULL,  # Define ensemble as a public property
    performance = NULL,   # Initialize performance as NULL
    relevance = NULL,     # Initialize relevance as NULL
    performance_augmentation = NULL,  # Add this line
    relevance_augmentation = NULL,  # Add this line if you plan to use it
    results_list_learnOnly = NULL,
    results_list = NULL,
    predicted_outputAndTime = NULL,
    numeric_columns = NULL,
    #ensemble_number = NULL,
    # Constructor
    initialize = function(num_networks, input_size, hidden_sizes, output_size, N, lambda, ensemble_number, ensembles, ML_NN, method = init_method, custom_scale = custom_scale) {
      
      
      # Initialize an ensemble of SONN networks
      self$ensemble <- lapply(1:num_networks, function(i) {
        # Determine ensemble and model names
        # if (hyperparameter_grid_setup) {
        #   ensemble_number <- j
        # } else {
        #   ensemble_number <- j
        # }
        
        if (firstRun) {
          ensemble_number <- j
        } else {
          ensemble_number <- j + 1
        }
        
        ensemble_name <- ensemble_number
        model_name <- i
        
        # Initialize variables
        run_result <- NULL
        weights_record_extract <- NULL
        biases_record_extract <- NULL
        
        # Construct result variable name dynamically
        if (use_loaded_weights || predict_models) {
          if (startsWith(paste0("run_results_1_", i), "run_results_1_")) {
            result_variable_name <- paste0("run_results_1_", i)
          } else if (startsWith(paste0("run_results_1_", i), "official_run_results_1_")) {
            result_variable_name <- paste0("official_run_results_1_", i)
          } else {
            result_variable_name <- paste0("official_", i)
            cat("Trying to get variable:", result_variable_name, "\n")
          }
          
          # Ensure the variable exists before accessing it
          if (exists(result_variable_name, envir = .GlobalEnv)) {
            # Access the variable
            run_result <- get(result_variable_name, envir = .GlobalEnv)
            
            if (!is.null(run_result$weights_record) &&
                !is.null(run_result$biases_record) &&
                is.null(run_result$metadata)) {
              # Initialize lists within lists for weights and biases
              weights_record_extract <- vector("list", length = length(run_result$best_weights_record))
              biases_record_extract <- vector("list", length = length(run_result$best_biases_record))
              
              # Extract and unlist the first element of weights_record and biases_record
              weights_record_extract[[1]] <- unlist(official_run_results_1_1$best_weights_record[[1]][[1]])
              biases_record_extract[[1]] <- unlist(official_run_results_1_1$best_biases_record[[1]][[1]])
              
              # Check if ML_NN is TRUE before accessing weights and biases for additional layers
              if (ML_NN == TRUE) {
                for (k in 2:(length(hidden_sizes)+1)) {
                  weights_record_extract[[k]] <- unlist(official_run_results_1_1$best_weights_record[[1]][[k]])
                  biases_record_extract[[k]] <- unlist(official_run_results_1_1$best_biases_record[[1]][[k]])
                }
              }
            } else if (!is.null(run_result$best_model_metadata$best_weights_record) &&
                       !is.null(run_result$best_model_metadata$best_biases_record) &&
                       !is.null(run_result$metadata)) {
              # Initialize lists within lists for weights and biases
              weights_record_extract <- vector("list", length = length(run_result$best_model_metadata$best_weights_record))
              biases_record_extract <- vector("list", length = length(run_result$best_model_metadata$best_biases_record))
              
              # Extract and unlist the first element of weights_record and biases_record from best_model_metadata
              weights_record_extract[[1]] <- unlist(run_result$best_model_metadata$best_weights_record[[1]][[1]])
              biases_record_extract[[1]] <- unlist(run_result$best_model_metadata$best_biases_record[[1]][[1]])
              
              # Check if ML_NN is TRUE before accessing weights and biases for additional layers
              if (ML_NN == TRUE) {
                for (k in 2:(length(hidden_sizes)+1)) {
                  weights_record_extract[[k]] <- unlist(run_result$best_model_metadata$best_weights_record[[1]][[k]])
                  biases_record_extract[[k]] <- unlist(run_result$best_model_metadata$best_biases_record[[1]][[k]])
                }
              }
            } else {
              stop(paste("Weights or biases not found in", result_variable_name))
            }
          } else {
            stop(paste("Variable", result_variable_name, "does not exist."))
          }
        }
        # Instantiate SONN network based on conditions
        if (ML_NN) {
          new_network <- SONN$new(input_size = input_size, hidden_sizes = hidden_sizes, output_size = output_size, N = N, lambda = lambda, ML_NN = ML_NN, activation_functions_learn = activation_functions_learn, activation_functions = activation_functions, method = init_method, custom_scale = custom_scale)
        } else {
          new_network <- SONN$new(input_size = input_size, output_size = output_size, N = N, lambda = lambda, ML_NN = ML_NN, activation_functions_learn = activation_functions_learn, activation_functions = activation_functions, method = init_method, custom_scale = custom_scale)
        }
        
        # Set names for the model
        attr(new_network, "ensemble_name") <- ensemble_name
        attr(new_network, "model_name") <- model_name
        
        
        
        # Check if ML_NN is TRUE before loading weights and biases for multiple layers
        if (ML_NN == TRUE) {
          if (!is.null(weights_record_extract[[1]]) && !is.null(biases_record_extract[[1]])) {
            # print((as.matrix(new_network$weights)))
            # print(weights_record_extract)
            # print(biases_record_extract)
            # for (k in 1:(length(hidden_sizes)+1)) {
            new_network$load_all_weights(weights_list = weights_record_extract)
            new_network$load_all_biases(biases_list = biases_record_extract[[k]])
            # }
          }
        }else{
          # Load weights and biases for the first layer
          if (!is.null(weights_record_extract[[1]]) && !is.null(biases_record_extract[[1]])) {
            # print((as.matrix(new_network$weights)))
            new_network$load_weights(new_weights = weights_record_extract)
            new_network$load_biases(new_biases = biases_record_extract)
          }
        }
        
        return(new_network)
      })
      
      
      self$predicted_outputAndTime <- list() #vector("list", length(self$ensemble) * 2) #* nrow(hyperparameter_grid))
      results_list_learnOnly <- list() #vector("list", length(self$ensemble) * 2) #* nrow(hyperparameter_grid))
      results_list <- list() #vector("list", length(self$ensemble) * 2) #* nrow(hyperparameter_grid))
      self$numeric_columns <- NULL
      
      self$ensembles <- ensembles
      
      # Configuration flags for enabling/disabling per-DESONN model performance/relevance plots
      self$FinalUpdatePerformanceandRelevanceViewPlotsConfig  <- list(
        performance_high_mean_plots = performance_high_mean_plots,  # high mean performance plots
        performance_low_mean_plots  = performance_low_mean_plots,   # low mean performance plots
        relevance_high_mean_plots   = relevance_high_mean_plots,    # high mean relevance plots
        relevance_low_mean_plots    = relevance_low_mean_plots,     # low mean relevance plots
        viewAllPlots = viewAllPlots,
        verbose      = verbose  
      )
      
    },
    
    # Function to normalize specific columns in the data
    normalize_data = function(Rdata, numeric_columns) {
      # Calculate mean and standard deviation for each numeric feature
      means <- colMeans(Rdata[, numeric_columns])
      std_devs <- apply(Rdata[, numeric_columns], 2, sd)
      
      # Print mean and standard deviation before normalization
      print(paste("Before normalization - Mean: ", means))
      print(paste("Before normalization - Standard Deviation: ", std_devs))
      
      # Normalize the numeric data
      normalized_Rdata <- Rdata
      normalized_Rdata[, numeric_columns] <- scale(Rdata[, numeric_columns], center = means, scale = std_devs)
      
      # Calculate mean and standard deviation after normalization
      normalized_means <- colMeans(normalized_Rdata[, numeric_columns])
      normalized_std_devs <- apply(normalized_Rdata[, numeric_columns], 2, sd)
      
      # Print mean and standard deviation after normalization
      print(paste("After normalization - Mean: ", normalized_means))
      print(paste("After normalization - Standard Deviation: ", normalized_std_devs))
      
      return(normalized_Rdata)
    }
    
    ,
    # Function to perform batch normalization on specific columns in the data
    batch_normalize_data = function(Rdata, numeric_columns, gamma_bn, beta_bn, epsilon_bn = epsilon_bn, momentum_bn = momentum_bn, is_training_bn = is_training_bn) {
      
      if (is_training_bn) {
        # Training mode: Compute mean and variance from the current batch
        batch_mean_bn <- colMeans(Rdata[, numeric_columns])
        batch_var_bn <- apply(Rdata[, numeric_columns], 2, var)
        
        # Update running statistics
        if (is.null(running_mean_bn)) {
          running_mean_bn <- batch_mean_bn
          running_var_bn <- batch_var_bn
        } else {
          running_mean_bn <- momentum_bn * running_mean_bn + (1 - momentum_bn) * batch_mean_bn
          running_var_bn <- momentum_bn * running_var_bn + (1 - momentum_bn) * batch_var_bn
        }
        
        # Normalize using batch statistics
        normalized_data_bn <- Rdata
        normalized_data_bn[, numeric_columns] <- (Rdata[, numeric_columns] - batch_mean_bn) / sqrt(batch_var_bn + epsilon_bn)
        
        # Apply gamma and beta
        normalized_data_bn[, numeric_columns] <- (normalized_data_bn[, numeric_columns] * gamma_bn) + beta_bn
        
        # Print diagnostic information
        print(paste("Batch Mean: ", batch_mean_bn))
        print(paste("Batch Variance: ", batch_var_bn))
        
      } else {
        # Inference mode: Use running statistics computed during training
        if (is.null(running_mean_bn) || is.null(running_var_bn)) {
          stop("Running mean and variance must be provided for inference.")
        }
        
        # Normalize using running statistics
        normalized_data_bn <- Rdata
        normalized_data_bn[, numeric_columns] <- (Rdata[, numeric_columns] - running_mean_bn) / sqrt(running_var_bn + epsilon_bn)
        
        # Apply gamma and beta
        normalized_data_bn[, numeric_columns] <- (normalized_data_bn[, numeric_columns] * gamma_bn) + beta_bn
      }
      
      # Print diagnostic information
      post_norm_mean_bn <- colMeans(normalized_data_bn[, numeric_columns])
      post_norm_var_bn <- apply(normalized_data_bn[, numeric_columns], 2, var)
      
      print(paste("After normalization - Mean: ", post_norm_mean_bn))
      print(paste("After normalization - Variance: ", post_norm_var_bn))
      
      # Return the normalized data and the updated running mean/variance
      return(list(normalized_data = normalized_data_bn, running_mean = running_mean_bn, running_var = running_var_bn))
    },
    # Function to calculate a reasonable batch size
    calculate_batch_size = function(data_size, max_batch_size = 512, min_batch_size = 16) {
      # Get the number of rows from the dataset
      n <- nrow(data_size)
      
      # Default to the minimum batch size
      batch_size <- min_batch_size
      
      # Adjust batch size if the dataset is sufficiently large
      if (n >= 1000) {
        # Set batch size to a fraction of dataset size, constrained by max_batch_size
        batch_size <- min(max_batch_size, n / 10)
      }
      
      # Return the computed batch size
      return(batch_size)
    }, 
    viewFinalUpdatePerformanceandRelevancePlots = function(name) {
      cfg <- self$FinalUpdatePerformanceandRelevanceViewPlotsConfig
      on_all <- isTRUE(cfg$viewAllPlots) || isTRUE(cfg$verbose)
      val <- cfg[[name]]
      flag <- isTRUE(val) || (is.logical(val) && length(val) == 1 && !is.na(val) && val)
      on_all || flag
    }
    ,
    
    train = function(Rdata, labels, lr, lr_decay_rate, lr_decay_epoch, lr_min, ensemble_number, num_epochs, use_biases, threshold, reg_type, numeric_columns, CLASSIFICATION_MODE, activation_functions_learn, activation_functions, dropout_rates_learn, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, batch_normalize_data, gamma_bn = NULL, beta_bn = NULL, epsilon_bn = 1e-5, momentum_bn = 0.9, is_training_bn = TRUE, shuffle_bn = FALSE, loss_type, sample_weights, preprocessScaledData, X_validation, y_validation, validation_metrics, threshold_function, ML_NN, train, viewTables, verbose) {
      
      
      if (!is.null(numeric_columns) && !batch_normalize_data) {
        # Normalize the input data
        Rdata <- self$normalize_data(Rdata, numeric_columns)
        
        # Optionally normalize labels if they are continuous, otherwise skip
        # If labels are binary or categorical, normalization should not be applied
        # if (is.numeric(labels) && CLASSIFICATION_MODE == "regression") {
        #   labels <- self$normalize_data(labels, numeric_columns)
        # }
      }
      
      # Initialize batch normalization parameters if not set
      if (batch_normalize_data) {
        if (is.null(gamma_bn)) gamma_bn <- rep(1, length(numeric_columns))
        if (is.null(beta_bn)) beta_bn <- rep(0, length(numeric_columns))
        if (is.null(self$mean_bn)) self$mean_bn <- rep(0, length(numeric_columns))
        if (is.null(self$var_bn)) self$var_bn <- rep(1, length(numeric_columns))
      }
      
      for (epoch in 1:num_epochs) {
        # Create mini-batches
        n <- nrow(Rdata)
        indices <- 1:n
        if (shuffle_bn) {
          indices <- sample(indices)  # Shuffle data if required
        }
        
        batch_size <- self$calculate_batch_size(Rdata)
        mini_batches <- split(indices, ceiling(seq_along(indices) / batch_size))
        
        for (batch in mini_batches) {
          # Extract the current mini-batch
          batch_data <- Rdata[batch, ]
          batch_labels <- labels[batch]
          
          # Perform batch normalization if specified
          if (batch_normalize_data) {
            if (is_training_bn) {
              # Training mode: Compute mean and variance from the current batch
              batch_mean_bn <- colMeans(batch_data[, numeric_columns])
              batch_var_bn <- apply(batch_data[, numeric_columns], 2, var)
              
              # Update running statistics
              self$mean_bn <- momentum_bn * self$mean_bn + (1 - momentum_bn) * batch_mean_bn
              self$var_bn <- momentum_bn * self$var_bn + (1 - momentum_bn) * batch_var_bn
              
              # Normalize using batch statistics
              batch_data[, numeric_columns] <- (batch_data[, numeric_columns] - batch_mean_bn) / sqrt(batch_var_bn + epsilon_bn)
              
              # Apply gamma and beta
              batch_data[, numeric_columns] <- (batch_data[, numeric_columns] * gamma_bn) + beta_bn
              
              if(verbose){
              # Print diagnostic information for the current mini-batch
              print(paste("Batch Mean: ", batch_mean_bn))
              print(paste("Batch Variance: ", batch_var_bn))
              }
            } else {
              # Inference mode: Use running statistics computed during training
              if (is.null(self$mean_bn) || is.null(self$var_bn)) {
                stop("Running mean and variance must be provided for inference.")
              }
              
              # Normalize using running statistics
              batch_data[, numeric_columns] <- (batch_data[, numeric_columns] - self$mean_bn) / sqrt(self$var_bn + epsilon_bn)
              
              # Apply gamma and beta
              batch_data[, numeric_columns] <- (batch_data[, numeric_columns] * gamma_bn) + beta_bn
              
              # Print diagnostic information for the current mini-batch
              post_norm_mean_bn <- colMeans(batch_data[, numeric_columns])
              post_norm_var_bn <- apply(batch_data[, numeric_columns], 2, var)
              if(verbose){
              print(paste("After normalization - Mean: ", post_norm_mean_bn))
              print(paste("After normalization - Variance: ", post_norm_var_bn))
              }
            }
          }
          
          # Training logic (forward pass, loss computation, backpropagation, parameter updates) goes here
          
        }
      }
      
      
      # Initialize lists to store results
      all_predicted_outputAndTime    <- vector("list", length(self$ensemble))
      all_predicted_outputs_learn    <- vector("list", length(self$ensemble))
      all_predicted_outputs          <- vector("list", length(self$ensemble))
      all_prediction_times           <- vector("list", length(self$ensemble))
      all_training_times             <- vector("list", length(self$ensemble))
      all_best_val_prediction_time   <- vector("list", length(self$ensemble))
      all_learn_times                <- vector("list", length(self$ensemble))
      all_ensemble_name_model_name   <- vector("list", length(self$ensemble))
      all_model_iter_num             <- vector("list", length(self$ensemble))
      all_best_train_acc             <- vector("list", length(self$ensemble))
      all_best_epoch_train           <- vector("list", length(self$ensemble))
      all_best_val_acc               <- vector("list", length(self$ensemble))
      all_best_val_epoch             <- vector("list", length(self$ensemble))
      
      # NEW: Extended debug/tracking
      all_errors                     <- vector("list", length(self$ensemble))
      all_hidden_outputs             <- vector("list", length(self$ensemble))
      all_layer_dims                 <- vector("list", length(self$ensemble))
      all_best_val_probs <- vector("list", length(self$ensemble))
      all_best_val_labels <- vector("list", length(self$ensemble))
      all_weights <- vector("list", length(self$ensemble))
      all_biases <- vector("list", length(self$ensemble))
      all_activation_functions <- vector("list", length(self$ensemble))
      
      # my_optimal_epoch_out_vector    <- vector("list", length(self$ensemble))
      
      if (never_ran_flag == TRUE) {
        for (i in 1:length(self$ensemble)) {
          # Add Ensemble and Model names to performance_list
          ensemble_name <- attr(self$ensemble[[i]], "ensemble_name")
          model_name <- attr(self$ensemble[[i]], "model_name")
          
          ensemble_name_model_name <- paste("Ensemble:", ensemble_name, "Model:", model_name)
          
          model_iter_num <- i
          
          
          
          
          # self$ensemble[[i]]$self_organize(Rdata, labels, lr)
          if (learnOnlyTrainingRun == FALSE) {
            # learn_results <- self$ensemble[[i]]$learn(Rdata, labels, lr, activation_functions_learn, dropout_rates_learn)
            predicted_outputAndTime <- suppressMessages(
              self$ensemble[[i]]$train_with_l2_regularization(
                Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, model_iter_num, update_weights, update_biases, use_biases, ensemble_number, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type, sample_weights, X_validation, y_validation, threshold_function, ML_NN, train, verbose
              ))
            

            
            # -- Start: Store core model info --
            all_ensemble_name_model_name[[i]] <- ensemble_name_model_name
            
            all_model_iter_num[[i]] <- model_iter_num

            
            all_predicted_outputAndTime[[i]] <- list(
              predicted_output         = predicted_outputAndTime$predicted_output_l2$predicted_output, #this is last_val_predict or last_train_predict based on what is toggled upstream (isTrue(validation_metrics))
              prediction_time          = predicted_outputAndTime$predicted_output_l2$prediction_time,
              training_time            = predicted_outputAndTime$training_time,
              best_val_prediction_time = predicted_outputAndTime$best_val_prediction_time,
              optimal_epoch            = predicted_outputAndTime$optimal_epoch,
              weights_record           = predicted_outputAndTime$best_weights_record,
              biases_record            = predicted_outputAndTime$best_biases_record,
              losses_at_optimal_epoch  = predicted_outputAndTime$lossesatoptimalepoch,
              best_train_acc           = predicted_outputAndTime$best_train_acc,
              best_epoch_train         = predicted_outputAndTime$best_epoch_train,
              best_val_acc             = predicted_outputAndTime$best_val_acc,
              best_val_epoch           = predicted_outputAndTime$best_val_epoch
            )

            # Optional storage
            # my_optimal_epoch_out_vector[[i]] <<- predicted_outputAndTime$optimal_epoch
            # ----------------------------------
            
            # Continue if predictions are available
            if (!is.null(predicted_outputAndTime$predicted_output_l2)) {
              
              all_predicted_outputs[[i]]        <- predicted_outputAndTime$predicted_output_l2$predicted_output
              all_training_times                <- predicted_outputAndTime$training_time
              # all_prediction_times[[i]]         <- predicted_outputAndTime$train_reg_prediction_time
              all_best_val_prediction_time[[i]] <- predicted_outputAndTime$best_val_prediction_time
              all_errors[[i]]                   <- compute_error(predicted_outputAndTime$predicted_output_l2$predicted_output, y, CLASSIFICATION_MODE)
              all_hidden_outputs[[i]]           <- predicted_outputAndTime$learn_hidden_outputs
              all_layer_dims[[i]]               <- predicted_outputAndTime$learn_dim_hidden_layers
              all_best_val_probs[[i]]           <- predicted_outputAndTime$best_val_probs
              all_best_val_labels[[i]]          <- predicted_outputAndTime$best_val_labels
              all_weights[[i]]                  <- predicted_outputAndTime$best_weights_record
              all_biases[[i]]                   <- predicted_outputAndTime$best_biases_record
              all_activation_functions[[i]]     <- activation_functions_learn
              all_best_train_acc[[i]]           <- predicted_outputAndTime$best_train_acc
              all_best_epoch_train[[i]]         <- predicted_outputAndTime$best_epoch_train
              all_best_val_acc[[i]]             <- predicted_outputAndTime$best_val_acc
              all_best_val_epoch[[i]]           <- predicted_outputAndTime$best_val_epoch

              # --- Debug prints ---
              cat(">> Ensemble Index:", i, "\n")
              cat("Predicted Output (first 5):\n"); print(head(all_predicted_outputs[[i]], 5))
              cat("Prediction Time:\n"); print(all_prediction_times[[i]])
              cat("Shape of Predicted Output:\n"); print(dim(all_predicted_outputs[[i]]))
              
              cat("Error Preview (first 5):\n"); print(head(all_errors[[i]], 5))
              if(ML_NN){
                cat("Hidden Output Layer Count:\n"); print(length(all_hidden_outputs[[i]]))
                cat("Hidden Layer Dims:\n"); print(all_layer_dims[[i]])
              }
              cat("Best Validation Probabilities (first 5):\n"); print(head(all_best_val_probs[[i]], 5))
              cat("Best Validation Labels (first 5):\n"); print(head(all_best_val_labels[[i]], 5))
              
              # Debug weights and biases
              cat("Weights Record (layer 1 preview):\n"); str(all_weights[[i]][[1]])
              cat("Biases Record (layer 1 preview):\n"); str(all_biases[[i]][[1]])
              
              # Debug activation functions
              cat("Activation Functions Used:\n"); print(all_activation_functions[[i]])
              cat("--------------------------------------------------------\n")
              
              
            } else {
              cat("WARNING: predicted_output_l2$learn_output is NULL at ensemble index", i, "\n")
              str(predicted_outputAndTime)
            }
            
            
            
            
          }
        }
        
        ###########code from old code###########
        print(all_ensemble_name_model_name)
        
        
        for (i in seq_along(all_predicted_outputAndTime)) {
          cat("\n── Model", i, "──\n")
          model_result <- all_predicted_outputAndTime[[i]]
          
          if (is.null(model_result)) {
            cat("Empty slot.\n")
            next
          }
          
          cat("Prediction length:", length(model_result$predicted_output), "\n")
          cat("Prediction time:", model_result$prediction_time, "\n")
          cat("Training time:", model_result$training_time, "\n")
          cat("Optimal epoch:", model_result$optimal_epoch, "\n")
          cat("Loss at optimal:", model_result$losses_at_optimal_epoch, "\n")
          
          # ---- Weights ----
          if (is.list(model_result$weights_record)) {
            cat("Weights record dims by layer:\n")
            for (L in seq_along(model_result$weights_record)) {
              W <- model_result$weights_record[[L]]
              if (!is.null(W)) {
                cat(sprintf("  Layer %d: ", L)); print(dim(W))
              } else {
                cat(sprintf("  Layer %d: NULL\n", L))
              }
            }
          } else {
            cat("Weights record dims (SL): "); print(dim(model_result$weights_record))
          }
          
          # ---- Biases ----
          if (is.list(model_result$biases_record)) {
            cat("Biases record length by layer:\n")
            for (L in seq_along(model_result$biases_record)) {
              b <- model_result$biases_record[[L]]
              if (!is.null(b)) {
                # bias could be vector or 1-col matrix
                blen <- if (is.matrix(b)) nrow(b) * ncol(b) else length(b)
                cat(sprintf("  Layer %d: %d\n", L, blen))
              } else {
                cat(sprintf("  Layer %d: NULL\n", L))
              }
            }
          } else {
            cat("Biases record length (SL): ")
            blen <- if (is.matrix(model_result$biases_record)) length(model_result$biases_record) else length(model_result$biases_record)
            print(blen)
          }
          
          
          # Sys.sleep(0.25)  # pause slightly for readability
        }
        
        # all_ensemble_name_model_name <<- do.call(c, all_ensemble_name_model_name)
        
        performance_relevance_data <- self$update_performance_and_relevance(
          Rdata                        = Rdata,
          labels                       = labels,
          preprocessScaledData         = preprocessScaledData,
          X_validation                 = X_validation,
          y_validation                 = y_validation,
          validation_metrics           = validation_metrics,
          lr                           = lr,
          CLASSIFICATION_MODE          = CLASSIFICATION_MODE,
          ensemble_number              = ensemble_number,
          model_iter_num               = model_iter_num,
          num_epochs                   = num_epochs,
          threshold                    = threshold,
          threshold_function           = threshold_function,
          learn_results                = learn_results,
          predicted_output_list        = all_predicted_outputs,
          all_best_val_probs           = all_best_val_probs,
          all_best_val_labels          = all_best_val_labels,
          all_best_val_prediction_time = all_best_val_prediction_time,
          learn_time                   = NULL,
          prediction_time_list         = all_prediction_times,
          run_id                       = all_ensemble_name_model_name,
          all_predicted_outputAndTime  = all_predicted_outputAndTime,
          all_weights                  = all_weights,
          all_biases                   = all_biases,
          all_activation_functions     = all_activation_functions,
          all_best_train_acc           = all_best_train_acc,
          all_best_epoch_train         = all_best_epoch_train,
          all_best_val_acc             = all_best_val_acc,
          all_best_val_epoch           = all_best_val_epoch,
          ML_NN = ML_NN,
          viewTables = viewTables,
          verbose = verbose
        )

        `%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a
        
        # Prints to RStudio Plots pane ONLY. Never saves. Handles ggplot, list, and nested list.
        print_plotlist_verbose <- function(x, label = NULL, print_plots = TRUE) {
          lab <- label %||% "Plot"
          if (inherits(x, c("gg","ggplot"))) {
            if (print_plots) print(x)
            return(invisible(NULL))
          }
          if (is.list(x)) {
            for (i in seq_along(x)) {
              item <- x[[i]]
              if (inherits(item, c("gg","ggplot")) && print_plots) print(item)
              if (is.list(item)) {
                for (j in seq_along(item)) {
                  p <- item[[j]]
                  if (inherits(p, c("gg","ggplot")) && print_plots) print(p)
                }
              }
            }
          }
          invisible(NULL)
        }
        
        # =========================
        # DESONN — Final perf/relevance lists (uses self$viewFinalUpdatePerformanceandRelevancePlots)
        # =========================
        
        `%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a
        
        # Helper: ask your R6 toggle
        .allow <- function(name) {
          # name should match a field in self$FinalUpdatePerformanceandRelevanceViewPlotsConfig
          # on_all = viewAllPlots || verbose
          self$viewFinalUpdatePerformanceandRelevancePlots(name)
        }
        
        # Optional: allow disabling saving too (defaults TRUE if unset)
        .save_enabled <- isTRUE(self$FinalUpdatePerformanceandRelevanceViewPlotsConfig$saveAlso %||% TRUE)
        
        # Prepare output dir only if we might save
        if (.save_enabled && !dir.exists("plots")) dir.create("plots", recursive = TRUE, showWarnings = FALSE)
        
        ens <- as.integer(ensemble_number)
        tot <- if (!is.null(self$ensemble)) length(self$ensemble) else as.integer(get0("num_networks", ifnotfound = 1L))
        mod <- if (exists("model_iter_num", inherits = TRUE) && length(model_iter_num)) as.integer(model_iter_num) else 1L
        
        # Ensure we have a filename namer
        if (!exists("fname", inherits = TRUE) || !is.function(fname)) {
          fname <- make_fname_prefix(
            isTRUE(get0("do_ensemble", ifnotfound = FALSE)),
            num_networks    = tot,
            total_models    = tot,
            ensemble_number = ens,
            model_index     = mod,
            who             = "DESONN"
          )
        }
        
        .slug <- function(s) {
          s <- trimws(as.character(s))
          s <- gsub("\\s+", "_", s)
          s <- gsub("[^A-Za-z0-9_]+", "_", s)
          tolower(gsub("_+", "_", s))
        }
        
        .plot_label_slug <- function(p) {
          .first_label <- function(obj) {
            if (is.null(obj)) return(NULL)
            s <- tryCatch(as.character(obj), error = function(e) NULL)
            if (is.null(s) || !length(s)) return(NULL)
            s1 <- s[[1]]; if (is.null(s1) || !nzchar(s1)) return(NULL)
            s1
          }
          t1 <- .first_label(tryCatch(p$labels$title, error = function(e) NULL))
          if (!is.null(t1)) return(.slug(t1))
          y1 <- .first_label(tryCatch(p$labels$y,     error = function(e) NULL))
          if (!is.null(y1)) return(.slug(y1))
          NULL
        }
        
        # Walk any mixture of ggplot / list / nested list; save/print only if allowed(name)
        .walk_save_view <- function(x, base, idx_env, name_flag) {
          if (is.null(x) || !length(x)) return(invisible(NULL))
          
          # Respect your config per group
          do_action <- .allow(name_flag)
          
          save_one <- function(p, nm_fallback) {
            if (!do_action) return(invisible(NULL))   # neither save nor print if flag is off
            
            nm <- .plot_label_slug(p) %||% .slug(nm_fallback)
            idx_env[[nm]] <- (idx_env[[nm]] %||% 0L) + 1L
            file_base <- sprintf("%s_%03d", nm, idx_env[[nm]])
            out <- file.path("plots", fname(sprintf("%s.png", file_base)))
            
            # Save only if global save is enabled
            if (.save_enabled) {
              try(suppressWarnings(suppressMessages(
                ggsave(out, p, width = 6, height = 4, dpi = 300)
              )), silent = TRUE)
            }
            
            # Print (view) — gated by same per-group flag
            try(print(p), silent = TRUE)
          }
          
          if (inherits(x, c("gg","ggplot"))) {
            save_one(x, base)
          } else if (is.list(x)) {
            for (k in seq_along(x)) {
              elem <- x[[k]]
              if (inherits(elem, c("gg","ggplot"))) {
                save_one(elem, sprintf("%s_%02d", base, k))
              } else if (is.list(elem)) {
                .walk_save_view(elem, sprintf("%s_%02d", base, k), idx_env, name_flag)
              }
            }
          }
          invisible(NULL)
        }
        
        # Independent counters per group to keep filenames stable
        .idx <- new.env(parent = emptyenv())
        
        # Map each holder to its config flag name (keys must match your config fields)
        .walk_save_view(performance_relevance_data$performance_high_mean_plots, "performance_high_mean",
                        .idx, "performance_high_mean_plots")
        .walk_save_view(performance_relevance_data$performance_low_mean_plots,  "performance_low_mean",
                        .idx, "performance_low_mean_plots")
        .walk_save_view(performance_relevance_data$relevance_high_mean_plots,   "relevance_high_mean",
                        .idx, "relevance_high_mean_plots")
        .walk_save_view(performance_relevance_data$relevance_low_mean_plots,    "relevance_low_mean",
                        .idx, "relevance_low_mean_plots")
        
        invisible(NULL)
        
        
        
        
        
        
        
        
        # At the end of the training process, call the predict function
        # trained_predictions <<- self$predict(Rdata, labels, activation_functions)
        # print(dim(labels))
        predicted_outputAndTime$loss_status <- 'exceeds_10000'
        
        
        

        
        
        
      }
      
      return(list(predicted_outputAndTime = predicted_outputAndTime, performance_relevance_data = performance_relevance_data))
    }
    , # Method for updating performance and relevance metrics
    
    update_performance_and_relevance = function(Rdata, labels, preprocessScaledData, X_validation, y_validation, validation_metrics, lr, CLASSIFICATION_MODE, ensemble_number, model_iter_num, num_epochs, threshold, threshold_function, learn_results, predicted_output_list, all_best_val_probs, all_best_val_labels, all_best_val_prediction_time, learn_time, prediction_time_list, run_id, all_predicted_outputAndTime, all_weights, all_biases, all_activation_functions, all_best_train_acc, all_best_epoch_train, all_best_val_acc, all_best_val_epoch, ML_NN, viewTables, verbose) {
      
      
      # Initialize lists to store performance and relevance metrics for each SONN
      performance_list <- list()
      relevance_list <- list()
      model_name_list <-  list()
      #████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
      
      # Calculate performance and relevance for each SONN in the ensemble
      if (never_ran_flag == TRUE) {
        
        for (i in 1:length(self$ensemble)) {
          
          
          best_val_probs <- all_best_val_probs[[i]]
          best_val_labels <- all_best_val_labels[[i]]
          best_val_prediction_time <- all_best_val_prediction_time[[i]]
          
          best_train_acc <- all_best_train_acc
          best_epoch_train <- all_best_epoch_train
          best_val_acc <- all_best_val_acc
          best_val_epoch <- all_best_val_epoch
            
          single_predicted_outputAndTime <- all_predicted_outputAndTime[[i]]  # metadata
          single_predicted_output <- predicted_output_list[[i]]
          single_ensemble_name_model_name <- run_id[[i]]
          
          if (learnOnlyTrainingRun == FALSE) {
            
            if (hyperparameter_grid_setup) {
              cat("___________________________________________________________________________\n")
              cat("______________________________DESONN_", ensemble_number , "_SONN_", i, "______________________________\n", sep = "")
            } else {
              cat("___________________________________________________________________________\n")
              cat("______________________________DESONN_", ensemble_number, "_SONN_", i, "______________________________\n", sep = "")
            }
            
            single_prediction_time <- prediction_time_list[[i]]
            
            #brought X_validation and y_validation as close as possible to metrics without "doubling-up" vars per se
            if (validation_metrics){
              Rdata = X_validation
              labels = y_validation
            }
            
            # === Evaluate Prediction Diagnostics ===
            if (!is.null(X_validation) && !is.null(y_validation) && isTRUE(validation_metrics)) {
              eval_result <- EvaluatePredictionsReport(
                X_validation = X_validation,
                y_validation = y_validation,
                CLASSIFICATION_MODE = CLASSIFICATION_MODE,
                probs = single_predicted_output,
                predicted_outputAndTime = single_predicted_outputAndTime,
                threshold_function = threshold_function,    # still accepted, no-op
                all_best_val_probs = best_val_probs,
                all_best_val_labels = best_val_labels,
                verbose = TRUE,
                # --- NEW ---
                accuracy_plot = "both",                    # or "default" or "both"
                tuned_threshold_override = NULL,
                SONN
              )
              
            }
            
            # Safely get number of columns from many shapes
            safe_ncol <- function(x) {
              if (is.null(x)) return(0L)
              # If it's a list from self$predict, try common fields first
              if (is.list(x)) {
                if (!is.null(x$predicted_output)) return(safe_ncol(x$predicted_output))
                if (!is.null(x$preds))            return(safe_ncol(x$preds))
                if (!is.null(x$output))           return(safe_ncol(x$output))
                # last resort: if it still has a dim attribute
                if (!is.null(dim(x))) return(ncol(x))
                return(0L)
              }
              if (is.matrix(x))      return(ncol(x))
              if (is.data.frame(x))  return(ncol(x))
              if (!is.null(dim(x)))  return(dim(x)[2])
              # vectors/scalars count as 1 col (binary probs)
              if (is.atomic(x))      return(1L)
              0L
            }
            
            k_labels <- safe_ncol(y_validation)
            k_probs  <- safe_ncol(single_predicted_output)
            
            # Prefer label-driven K when available; otherwise use predictions
            K <- if (k_labels > 0L) max(1L, k_labels) else max(1L, k_probs)
            
            
            # Pull out both fields (back-compat + multiclass)
            best_threshold_scalar <- eval_result$best_threshold          # numeric (binary) or NA (multiclass)
            best_thresholds_vec   <- eval_result$best_thresholds         # vector: length 1 (binary) or K (multiclass)
            
            # Decide what to store/use
            if (K == 1L) {
              # Binary: prefer tuned scalar; fallback to 0.5 if NA
              threshold_used   <- if (is.finite(best_threshold_scalar)) best_threshold_scalar else 0.5
              thresholds_used  <- best_thresholds_vec  # length-1 vector (kept for consistency)
            } else {
              # Multiclass: no single scalar; keep the whole vector
              threshold_used   <- NA_real_
              # if somehow missing, fallback to 0.5 per class
              thresholds_used  <- if (!is.null(best_thresholds_vec) && length(best_thresholds_vec) == K) {
                best_thresholds_vec
              } else {
                rep(0.5, K)
              }
            }
            
            # Optional: logs
            if (isTRUE(verbose)) {
              if (K == 1L) {
                message(sprintf("[train] Using tuned binary threshold: %.3f", threshold_used))
              } else {
                message(sprintf("[train] Using tuned per-class thresholds: %s",
                                paste0(sprintf("%.3f", thresholds_used), collapse = ", ")))
              }
            }
            
            use_best_val <- TRUE
            
 
            if (use_best_val && !is.null(best_val_probs) && !is.null(best_val_labels)) {
              probs   <- best_val_probs
              targets <- best_val_labels
              prediction_time <- best_val_prediction_time
              cat("[calculate_performance] Using best validation snapshot (@ best epoch)\n")
            } else {
              probs   <- predicted_output
              targets <- labels
              prediction_time <- single_prediction_time
              cat("[calculate_performance] Using last-epoch predictions\n")
            }
            

            
            # # --- SINGLE CRITICAL DEBUG+STOP ---
            # acc_tmp <- mean((probs >= threshold) == as.integer(targets))
            # cat(sprintf("[DBG][model %d] acc_tmp=%.7f vs best_val_acc=%.7f | branch=%s\n",
            #             i,
            #             acc_tmp,
            #             ifelse(exists("best_val_acc"), best_val_acc, NA),
            #             if (exists("best_val_probs") && !is.null(best_val_probs)) "BEST" else "LAST"))
            # if (use_best_val && !is.null(best_val_probs) && abs(acc_tmp - best_val_acc) > 1e-9) {
            #   stop(sprintf("[STOP][model %d] accuracy mismatch between recompute (%.7f) and stored best_val_acc (%.7f)",
            #                i, acc_tmp, best_val_acc))
            # }
            
            
            performance_list[[i]] <- calculate_performance(
              SONN = self$ensemble[[i]],
              Rdata = Rdata,
              labels = targets,
              lr = lr,
              CLASSIFICATION_MODE = CLASSIFICATION_MODE,
              model_iter_num = i,
              num_epochs = num_epochs,
              threshold = threshold,
              learn_time = learn_time,
              predicted_output = probs,
              prediction_time = prediction_time,
              ensemble_number = ensemble_number,
              run_id = run_id,
              weights = all_weights[[i]],
              biases = all_biases[[i]],
              activation_functions = all_activation_functions[[i]],
              ML_NN = ML_NN,
              verbose = verbose
            )
            
            relevance_list[[i]] <- calculate_relevance(
              self$ensemble[[i]],
              Rdata = Rdata, 
              labels = targets,
              CLASSIFICATION_MODE = CLASSIFICATION_MODE,
              model_iter_num = i,
              predicted_output = probs, 
              ensemble_number = ensemble_number,
              weights = self$ensemble[[i]]$weights,
              biases = self$ensemble[[i]]$biases,
              activation_functions = self$ensemble[[i]]$activation_functions,
              ML_NN = ML_NN,
              verbose = verbose
            )
            
            performance_metric <- performance_list[[i]]$metrics
            
            
            
            relevance_metric <- relevance_list[[i]]$metrics
            
            
            
            
            if (ensemble_number < 1 && length(self$ensemble) >= 1 || (verbose && (ensemble_number < 1 && length(self$ensemble) >= 1))){
              cat(">> METRICS FOR ENSEMBLE:", ensemble_number, "MODEL:", i, "\n")
              print(performance_metric)
              print(relevance_metric)
              
            }
            
          }
          
          
          cat("\n====================================\n")
          cat("🔍 DEBUG: Preparing to store metadata\n")
          cat("Ensemble number: ", ensemble_number, "\n")
          cat("Model iteration: ", i, "\n")
          cat("Run ID: ", single_ensemble_name_model_name, "\n")
          cat("Predicted output shape:\n"); print(dim(single_predicted_output))
          cat("Checking self$ensemble[[", i, "]]\n")
          str(self$ensemble[[i]])
          cat("====================================\n\n")
          
          
          self$store_metadata(run_id = single_ensemble_name_model_name, CLASSIFICATION_MODE, ensemble_number, preprocessScaledData, validation_metrics, model_iter_num = i, num_epochs, threshold = NULL, predicted_output = single_predicted_output, actual_values = y, all_weights = all_weights, all_biases = all_biases, performance_metric = performance_metric, relevance_metric = relevance_metric, predicted_outputAndTime = single_predicted_outputAndTime, best_val_prediction_time = best_val_prediction_time, best_train_acc = best_train_acc, best_epoch_train = best_epoch_train, best_val_acc = best_val_acc, best_val_epoch = best_val_epoch)
          
        }
      }
      
      #████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
      
      # Extract names and metrics for performance and relevance
      performance_metrics <- lapply(seq_along(performance_list), function(i) performance_list[[i]]$metrics) #<<-
      performance_names <- lapply(seq_along(performance_list), function(i) performance_list[[i]]$names) #<<-
      
      relevance_metrics <- lapply(seq_along(relevance_list), function(i) relevance_list[[i]]$metrics) #<<-
      relevance_names <- lapply(seq_along(relevance_list), function(i) relevance_list[[i]]$names) #<<-
      
      # Check for NULL values in performance_metrics and relevance_metrics
      check_null <- function(metrics_list) {
        unlist(lapply(seq_along(metrics_list), function(i) {
          if (is.null(metrics_list[[i]])) {
            return(paste0("NULL at index: ", i))
          } else {
            return(paste0("Not NULL at index: ", i))
          }
        }))
      }
      check_null(performance_metrics)
      null_check_performance <- check_null(performance_metrics) #<<-
      check_null(relevance_metrics)
      null_check_relevance <- check_null(relevance_metrics) #<<-
      
      
      
      
      # Convert the matrix to a data frame without modifying row names
      # Initialize empty vectors to store values and row names
      
      # Function to process performance metrics
      process_performance <- function(metrics_data, model_names, high_threshold = 10, verbose = FALSE) {
        EXCLUDE_METRICS_REGEX <- paste(
          c(
            "^accuracy_precision_recall_f1_tuned_accuracy_percent$",
            "^accuracy_percent$",
            "^accuracy_precision_recall_f1_tuned_y_pred_class\\d+$",
            "^y_pred_class\\d+$",
            "^accuracy_precision_recall_f1_tuned_best_thresholds?$",
            "^best_thresholds?$",
            "^accuracy_precision_recall_f1_tuned_grid_used",
            "^grid_used"
          ),
          collapse = "|"
        )
        
        # ---- model names handling ----
        if (length(model_names) == 1L && length(metrics_data) > 1L) {
          model_names <- rep(model_names, length(metrics_data))
        }
        if (is.null(model_names) || length(model_names) != length(metrics_data)) {
          model_names <- paste0("Model_", seq_along(metrics_data))
        }
        
        # ---- helpers ----
        to_numeric_safely <- function(v) {
          v <- as.character(v)
          cleaned <- gsub("[^0-9eE+\\-\\.]", "", v)
          suppressWarnings(as.numeric(cleaned))
        }
        norm_atom <- function(x) {
          if (inherits(x, "Duration"))  return(as.numeric(x))                 # seconds
          if (inherits(x, "difftime"))  return(as.numeric(x, units = "secs")) # seconds
          if (inherits(x, "POSIXct") || inherits(x, "POSIXt")) return(as.numeric(x))
          if (inherits(x, "Date"))      return(as.numeric(x))
          if (is.logical(x))            return(as.numeric(x))
          if (is.factor(x))             return(as.character(x))
          x
        }
        # Flatten into named atomic elements (as list of scalars/vectors)
        flatten_metrics <- function(x, prefix = NULL) {
          out <- list()
          nm_prefix <- function(base, name) if (is.null(base) || base == "") name else paste0(base, "_", name)
          
          if (is.null(x)) return(out)
          
          if (is.atomic(x) && length(x) >= 1L) {
            # keep vectors; caller will split to rows
            nm <- if (is.null(prefix)) "value" else prefix
            out[[nm]] <- x
            return(out)
          }
          
          if (is.data.frame(x)) {
            for (nm in names(x)) {
              out <- c(out, flatten_metrics(x[[nm]], nm_prefix(prefix, nm)))
            }
            return(out)
          }
          
          if (is.list(x)) {
            nms <- names(x)
            for (i in seq_along(x)) {
              nm <- if (!is.null(nms) && nzchar(nms[i])) nms[i] else as.character(i)
              out <- c(out, flatten_metrics(x[[i]], nm_prefix(prefix, nm)))
            }
            return(out)
          }
          
          out  # unknown type -> ignore
        }
        
        build_long_df <- function(lst, model_name) {
          if (is.null(lst) || length(lst) == 0L) {
            if (isTRUE(verbose)) cat("[process_performance] empty metrics for", model_name, "\n")
            return(data.frame(Model_Name = character(0), Metric = character(0), Value = numeric(0)))
          }
          rows <- list()
          idx <- 1L
          for (nm in names(lst)) {
            val <- lst[[nm]]
            if (length(val) == 0L) next
            val <- norm_atom(val)
            if (length(val) == 0L) next
            
            # split vectors to multiple rows with indexed metric names (stable & unique)
            if (length(val) > 1L) {
              for (k in seq_along(val)) {
                rows[[idx]] <- data.frame(
                  Model_Name = model_name,
                  Metric     = paste0(nm, "_", k),
                  Value      = as.character(val[[k]]),
                  stringsAsFactors = FALSE, check.names = FALSE
                )
                idx <- idx + 1L
              }
            } else {
              rows[[idx]] <- data.frame(
                Model_Name = model_name,
                Metric     = nm,
                Value      = as.character(val),
                stringsAsFactors = FALSE, check.names = FALSE
              )
              idx <- idx + 1L
            }
          }
          if (length(rows) == 0L) {
            data.frame(Model_Name = character(0), Metric = character(0), Value = numeric(0))
          } else {
            do.call(rbind, rows)
          }
        }
        
        high_mean_df <- NULL
        low_mean_df  <- NULL
        
        for (i in seq_along(metrics_data)) {
          mdl_name <- model_names[[i]]
          met_raw  <- metrics_data[[i]]
          
          flat <- flatten_metrics(met_raw, NULL)
          long <- build_long_df(flat, mdl_name)
          
          if (!nrow(long)) next
          
          # drop unwanted metrics
          long <- long[!grepl(EXCLUDE_METRICS_REGEX, long$Metric), , drop = FALSE]
          if (!nrow(long)) next
          
          # numeric coercion
          long$Value <- to_numeric_safely(long$Value)
          long <- long[is.finite(long$Value), , drop = FALSE]
          if (!nrow(long)) next
          
          mean_metrics <- long |>
            dplyr::group_by(Metric) |>
            dplyr::summarise(mean_value = mean(Value, na.rm = TRUE), .groups = "drop")
          
          high_metrics <- mean_metrics |>
            dplyr::filter(mean_value > high_threshold) |>
            dplyr::pull(Metric)
          
          high_mean_df <- dplyr::bind_rows(high_mean_df, long[long$Metric %in% high_metrics, , drop = FALSE])
          low_mean_df  <- dplyr::bind_rows(low_mean_df,  long[!long$Metric %in% high_metrics, , drop = FALSE])
        }
        
        list(high_mean_df = high_mean_df, low_mean_df = low_mean_df)
      }
      
      # Assuming performance_metrics and relevance_metrics are already defined
      performance_results <- process_performance(performance_metrics, run_id) #<<-
      relevance_results <- process_performance(relevance_metrics, run_id) #<<-
      
      performance_high_mean_df <- performance_results$high_mean_df #<<-
      performance_low_mean_df <- performance_results$low_mean_df #<<-
      
      relevance_high_mean_df <- relevance_results$high_mean_df #<<-
      relevance_low_mean_df <- relevance_results$low_mean_df #<<-
      
      # Function to check and print if a dataframe is NULL
      check_and_print_null <- function(df, df_name) {
        if (is.null(df)) {
          print(paste(df_name, "is NULL"))
          return(TRUE)
        } else {
          print(paste(df_name, "is not NULL"))
          return(FALSE)
        }
      }
      
      # Check and print if any of the dataframes are NULL
      performance_high_mean_is_null <- check_and_print_null(performance_high_mean_df, "performance_high_mean_df")
      performance_low_mean_is_null <- check_and_print_null(performance_low_mean_df, "performance_low_mean_df")
      relevance_high_mean_is_null <- check_and_print_null(relevance_high_mean_df, "relevance_high_mean_df")
      relevance_low_mean_is_null <- check_and_print_null(relevance_low_mean_df, "relevance_low_mean_df")
      
      # Call the functions and get the plots only if the dataframes are not NULL
      # print("Calling Performance update_performance_and_relevance_high")
      performance_high_mean_plots <- if (!performance_high_mean_is_null) {
        self$update_performance_and_relevance_high(performance_high_mean_df)
      } else {
        NULL
      }
      performance_low_mean_plots <- if (!performance_low_mean_is_null) {
        self$update_performance_and_relevance_low(performance_low_mean_df)
      } else {
        NULL
      }
      # print("Finished Performance update_performance_and_relevance_low")
      # print("Calling Relevance update_performance_and_relevance_high")
      relevance_high_mean_plots <- if (!relevance_high_mean_is_null) {
        self$update_performance_and_relevance_high(relevance_high_mean_df)
      } else {
        NULL
      }
      # print("Finished Relevance update_performance_and_relevance_high")
      # print("Calling Relevance update_performance_and_relevance_low")
      relevance_low_mean_plots <- if (!relevance_low_mean_is_null) {
        self$update_performance_and_relevance_low(relevance_low_mean_df)
      } else {
        NULL
      }
      # 
      # # -----------------------------------------------
      # # Grouped metrics + printing policy (final)
      # # -----------------------------------------------
      # # Predeclare so they're always in scope
      perf_df <- relev_df <- NULL
      perf_group_summary <- relev_group_summary <- NULL
      group_perf <- group_relev <- NULL
      # 
      # # Build per-model long DFs (works for 1+ models)
      # perf_df  <- flatten_metrics_to_df(performance_list, run_id)
      # relev_df <- flatten_metrics_to_df(relevance_list,     run_id)
      # 
      # # --- Vanilla group summaries (across models) ---
      # perf_group_summary  <- summarize_grouped(perf_df)
      # relev_group_summary <- summarize_grouped(relev_df)
      # 
      # # --- Optional notify user ---
      # if (!isTRUE(verbose) && !isTRUE(viewTables)) {
      #   cat("\n[ℹ] Group summaries computed silently. 
      # Set `verbose = TRUE` to print data frames, 
      # or `viewTables = TRUE` to see tables.\n")
      # }
      # 
      # # Grouped metrics (run whenever you have ≥1 model)
      # if (ensemble_number >= 1 && length(self$ensemble) > 1) {
      #   group_perf <- calculate_performance_grouped(
      #     SONN_list             = self$ensemble,
      #     Rdata                 = Rdata,
      #     labels                = labels,
      #     lr                    = lr,
      #     CLASSIFICATION_MODE   = CLASSIFICATION_MODE,
      #     num_epochs            = num_epochs,
      #     threshold             = threshold,
      #     predicted_output_list = predicted_output_list,
      #     prediction_time_list  = prediction_time_list,
      #     ensemble_number       = ensemble_number,
      #     run_id                = run_id,
      #     ML_NN                 = ML_NN,
      #     verbose               = verbose,
      #     agg_method            = "mean",
      #     metric_mode           = "aggregate_predictions+rep_sonn",
      #     weights_list          = NULL,
      #     biases_list           = NULL,
      #     act_list              = NULL
      #   )
      #   
      #   group_relev <- calculate_relevance_grouped(
      #     SONN_list             = self$ensemble,
      #     Rdata                 = Rdata,
      #     labels                = labels,
      #     CLASSIFICATION_MODE   = CLASSIFICATION_MODE,
      #     predicted_output_list = predicted_output_list,
      #     ensemble_number       = ensemble_number,
      #     run_id                = run_id,
      #     ML_NN                 = ML_NN,
      #     verbose               = verbose,
      #     agg_method            = "mean",
      #     metric_mode           = "aggregate_predictions+rep_sonn"
      #   )
      #   
      #   # ---------- Printing policy ----------
      #   # Tables (DF heads) print if EITHER verbose OR viewTables
      #   if (isTRUE(verbose) || isTRUE(viewTables)) {
      #     if (!is.null(perf_df))  { cat("\n--- performance_long_df (head) ---\n"); print(utils::head(perf_df, 12)) }
      #     if (!is.null(relev_df)) { cat("\n--- relevance_long_df (head) ---\n"); print(utils::head(relev_df, 12)) }
      #   }
      #   
      #   # Summaries + grouped metrics print ONLY when verbose = TRUE
      #   if (isTRUE(verbose)) {
      #     if (!is.null(perf_group_summary))  { cat("\n=== PERFORMANCE group summary ===\n"); print(perf_group_summary) }
      #     if (!is.null(relev_group_summary)) { cat("\n=== RELEVANCE group summary ===\n"); print(relev_group_summary) }
      #     if (!is.null(group_perf))  { cat("\n=== GROUPED PERFORMANCE metrics ===\n"); print(group_perf$metrics) }
      #     if (!is.null(group_relev)) { cat("\n=== GROUPED RELEVANCE metrics ===\n"); print(group_relev$metrics) }
      #   }
        
      # }
      
      
      
      
      # Return the lists of plots
      return(list(performance_metric = performance_metric, relevance_metric = relevance_metric, performance_high_mean_plots = performance_high_mean_plots, performance_low_mean_plots = performance_low_mean_plots, relevance_high_mean_plots = relevance_high_mean_plots, relevance_low_mean_plots = relevance_low_mean_plots, performance_group_summary = perf_group_summary, relevance_group_summary = relev_group_summary, performance_long_df = perf_df, relevance_long_df = relev_df, performance_grouped = if (exists("group_perf")  && !is.null(group_perf))  group_perf$metrics  else NULL, relevance_grouped   = if (exists("group_relev") && !is.null(group_relev)) group_relev$metrics else NULL, threshold = threshold_used, thresholds = thresholds_used, accuracy = eval_result$accuracy, accuracy_percent = eval_result$accuracy_percent, metrics = if (!is.null(eval_result$metrics)) eval_result$metrics else NULL, misclassified = if (!is.null(eval_result$misclassified)) eval_result$misclassified else NULL))
      
      
    },
    # Function to identify outliers
    identify_outliers = function(y) {
      o <- boxplot.stats(y)$out
      return(if(length(o) == 0) NA else o)
    },
    
    # Function to create bin labels
    create_bin_labels = function(x) {
      breaks <- c(0, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
      labels <- cut(x, breaks = breaks, include.lowest = TRUE, right = FALSE, labels = FALSE)
      return(sapply(labels, function(l) {
        switch(as.character(l),
               "1" = "0%-0.05%",
               "2" = "0.05%-0.1%",
               "3" = "0.1%-0.5%",
               "4" = "0.5%-1%",
               "5" = "1%-2%",
               "6" = "2%-5%",
               "7" = "5%-10%",
               "8" = "10%-20%",
               "9" = "20%-30%",
               "10" = "30%-40%",
               "11" = "40%-50%",
               "12" = "50%-60%",
               "13" = "60%-70%",
               "14" = "70%-80%",
               "15" = "80%-90%",
               "16" = "90%-100%",
               "100%+")  # Add a catch-all label for unexpected values
      }))
    },
    
    update_performance_and_relevance_high = function(high_mean_df) {
      # Initialize empty lists to store the plots
      high_mean_plots <- list()
      
      # Loop over each unique metric
      for (metric in unique(high_mean_df$Metric)) {
        # Filter out rows where the Value is 0 for metrics containing "precision" or "mean_precision"
        filtered_high_mean_df <- high_mean_df[!(grepl("precision", high_mean_df$Metric, ignore.case = TRUE) & high_mean_df$Value == 0), ]
        
        # Filter out rows where the Value is NA or infinite
        filtered_high_mean_df <- filtered_high_mean_df[!is.na(filtered_high_mean_df$Value) & !is.infinite(filtered_high_mean_df$Value), ]
        
        # Subset the data for the current metric
        plot_data_high <- filtered_high_mean_df[filtered_high_mean_df$Metric == metric, ]
        
        # Check if plot_data is not empty
        if (nrow(plot_data_high) > 0) {
          # Add a column to identify outliers
          plot_data <- plot_data_high %>%
            mutate(Outlier = ifelse(Value %in% self$identify_outliers(Value), Value, NA))
          
          # Add columns for outliers
          plot_data$Model_Name_Outlier <- plot_data$Model_Name
          
          # Set the RowName to NA where there are no outliers
          plot_data$Model_Name_Outlier[is.na(plot_data$Outlier)] <- NA
          
          # Create bin labels for "precisions" or "mean_precisions"
          if (grepl("precision", metric, ignore.case = TRUE)) {
            plot_data$Title <- paste0("Boxplot for ", metric, " (", self$create_bin_labels(plot_data$Value), ")")
          } else {
            plot_data$Title <- paste("Boxplot for", metric)
          }
          
          # Create box plot
          high_mean_plot <- ggplot(plot_data, aes(x = Metric, y = Value)) +
            geom_boxplot() +
            labs(title = unique(plot_data$Title),
                 x = "Metric",
                 y = "Value") +
            theme_minimal()
          
          # Add text labels for the outliers
          high_mean_plot <- high_mean_plot +
            geom_text(aes(label = Model_Name_Outlier), na.rm = TRUE, hjust = -0.3)
          
          # Store the plot in the list
          high_mean_plots[[metric]] <- high_mean_plot
          
          # Save the plot in the "plot" folder (cross-platform)
          # ggsave(
          #   file.path("plots", paste0("high_mean_plot_", gsub("[^A-Za-z0-9_]", "_", metric), ".png")),
          #   high_mean_plot,
          #   width = 6,
          #   height = 4,
          #   dpi = 300
          # )
        }
      }
      return(high_mean_plots)
    }
    ,
    
    update_performance_and_relevance_low = function(low_mean_df) {
      low_mean_plots <- list()
      
      # Loop over each unique metric
      for (metric in unique(low_mean_df$Metric)) {
        # Filter out rows where the Value is 0 for metrics containing "precision" or "mean_precision"
        filtered_low_mean_df <- low_mean_df[!(grepl("precision", low_mean_df$Metric, ignore.case = TRUE) & low_mean_df$Value == 0), ]
        
        # Filter out rows where the Value is NA or infinite
        filtered_low_mean_df <- filtered_low_mean_df[!is.na(filtered_low_mean_df$Value) & !is.infinite(filtered_low_mean_df$Value), ]
        
        # Subset the data for the current metric
        plot_data_low <- filtered_low_mean_df[filtered_low_mean_df$Metric == metric, ]
        
        # Check if plot_data is not empty
        if (nrow(plot_data_low) > 0) {
          # Add a column to identify outliers
          plot_data <- plot_data_low %>%
            mutate(Outlier = ifelse(Value %in% self$identify_outliers(Value), Value, NA))
          
          # Add columns for outliers
          plot_data$Model_Name_Outlier <- plot_data$Model_Name
          
          # Set the RowName to NA where there are no outliers
          plot_data$Model_Name_Outlier[is.na(plot_data$Outlier)] <- NA
          
          # Create bin labels for "precisions" or "mean_precisions"
          if (grepl("precision", metric, ignore.case = TRUE)) {
            plot_data$Title <- paste0("Boxplot for ", metric, " (", self$create_bin_labels(plot_data$Value), ")")
          } else {
            plot_data$Title <- paste("Boxplot for", metric)
          }
          
          # Create box plot
          low_mean_plot <- ggplot(plot_data, aes(x = Metric, y = Value)) +
            geom_boxplot() +
            labs(title = unique(plot_data$Title),
                 x = "Metric",
                 y = "Value") +
            theme_minimal()
          
          # Add text labels for the outliers
          low_mean_plot <- low_mean_plot +
            geom_text(aes(label = Model_Name_Outlier), na.rm = TRUE, hjust = -0.3)
          
          # Store the plot in the list
          low_mean_plots[[metric]] <- low_mean_plot
          
          # Save the plot in the "plot" folder (cross-platform)
          # ggsave(
          #   file.path("plots", paste0("low_mean_plot_", gsub("[^A-Za-z0-9_]", "_", metric), ".png")),
          #   low_mean_plot,
          #   width = 6,
          #   height = 4,
          #   dpi = 300
          # )
        }
      }
      
      return(low_mean_plots)
    }
    ,
    store_metadata = function(run_id, CLASSIFICATION_MODE, ensemble_number, preprocessScaledData, validation_metrics, model_iter_num, num_epochs, threshold, all_weights, all_biases, predicted_output, actual_values, performance_metric, relevance_metric, predicted_outputAndTime, best_val_prediction_time, best_train_acc, best_epoch_train, best_val_acc, best_val_epoch) {
      
      # ---------------- helpers (lightweight; keep most original structure) ----------------
      to_num_mat <- function(x) {
        # Coerce common inputs to a numeric matrix (no factors/characters left)
        if (is.data.frame(x)) {
          if (ncol(x) == 1L && (is.factor(x[[1]]) || is.character(x[[1]]))) {
            x <- matrix(as.numeric(as.factor(x[[1]])), ncol = 1L)
          } else {
            x[] <- lapply(x, function(col) if (is.numeric(col)) col else as.numeric(as.factor(col)))
            x <- as.matrix(x)
          }
        } else if (is.matrix(x)) {
          if (!is.numeric(x)) x <- apply(x, 2, function(col) if (is.numeric(col)) col else as.numeric(as.factor(col)))
          x <- as.matrix(x)
        } else if (is.factor(x) || is.character(x)) {
          x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
        } else if (is.atomic(x) && !is.matrix(x)) {
          x <- matrix(x, ncol = 1L)
        } else {
          x <- as.matrix(x)
          if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
        }
        storage.mode(x) <- "double"
        x
      }
      clamp <- function(v, lo, hi) pmax(lo, pmin(hi, v))
      
      # ---------------- pick target set as before ----------------
      target_raw <- if (isTRUE(validation_metrics)) {
        get0("y_validation", ifnotfound = actual_values, inherits = TRUE)
      } else {
        actual_values
      }
      
      # ---------------- Conform/compute by CLASSIFICATION_MODE (prevents non-numeric - operator) ----------------
      # Coerce raw inputs to matrices we can work with
      Lm0 <- to_num_mat(target_raw)
      Pm0 <- to_num_mat(predicted_output)
      
      # Ensure same number of rows (trim to min; preserves original spirit without rowname alignment)
      n_common <- min(nrow(Lm0), nrow(Pm0))
      if (n_common == 0L) stop("[store_metadata] Empty labels/predictions after trim.")
      if (nrow(Lm0) != nrow(Pm0)) {
        Lm0 <- Lm0[seq_len(n_common), , drop = FALSE]
        Pm0 <- Pm0[seq_len(n_common), , drop = FALSE]
      }
      
      # ---- STRICT: require explicit CLASSIFICATION_MODE; no auto-infer ----
      mode <- tolower(as.character(CLASSIFICATION_MODE))
      if (!mode %in% c("binary","multiclass","regression")) {
        stop("Invalid CLASSIFICATION_MODE. Must be one of: 'binary', 'multiclass', or 'regression'.")
      }
      
      # We'll produce: target_matrix, predicted_output_matrix, error_prediction, differences
      if (identical(mode, "binary")) {
        # ----- Binary: target = {0,1}, predicted = p(pos) -----
        if (ncol(Lm0) == 2L) {
          y_true <- as.integer(Lm0[,2] >= Lm0[,1])
        } else {
          v <- as.numeric(Lm0[,1])
          u <- sort(unique(as.integer(round(v))))
          if (length(u) == 2L) {
            y_true <- as.integer(v == max(u))  # map larger value to 1
          } else if (all(v %in% c(0,1))) {
            y_true <- as.integer(v)
          } else {
            y_true <- as.integer(v >= 0.5)
          }
        }
        # predicted positive-class probability
        p_pos <- if (ncol(Pm0) >= 2L) as.numeric(Pm0[,2]) else as.numeric(Pm0[,1])
        p_pos[!is.finite(p_pos)] <- NA_real_
        
        target_matrix            <- matrix(as.numeric(y_true), ncol = 1L)
        predicted_output_matrix  <- matrix(p_pos, ncol = 1L)
        
        error_prediction <- target_matrix - predicted_output_matrix
        differences      <- error_prediction
        
      } else if (identical(mode, "multiclass")) {
        # ----- Multiclass: target = one-hot (K), predicted = prob (K) -----
        Kp <- ncol(Pm0); Kl <- ncol(Lm0)
        K  <- if (Kp > 1L) Kp else if (Kl > 1L) Kl else max(2L, Kp, Kl)
        
        # Build true class IDs
        if (Kl > 1L) {
          true_ids <- max.col(Lm0, ties.method = "first")
        } else {
          # Single-column labels could be factor IDs; try to read from original target_raw if factor/character
          if (is.data.frame(target_raw) && ncol(target_raw) == 1L && (is.factor(target_raw[[1]]) || is.character(target_raw[[1]]))) {
            true_ids <- as.integer(factor(target_raw[[1]]))
          } else {
            vv <- as.integer(round(Lm0[,1]))
            if (length(vv) && min(vv, na.rm = TRUE) == 0L) vv <- vv + 1L
            true_ids <- vv
          }
          true_ids <- clamp(true_ids, 1L, K)
        }
        target_matrix <- one_hot_from_ids(true_ids, K)
        
        # Predicted probabilities: ensure N x K (replicate/truncate if needed)
        if (ncol(Pm0) < K) {
          total_needed <- n_common * K
          rep_vec <- rep(as.vector(Pm0), length.out = total_needed)
          predicted_output_matrix <- matrix(rep_vec, nrow = n_common, ncol = K, byrow = FALSE)
        } else {
          predicted_output_matrix <- Pm0[, seq_len(K), drop = FALSE]
        }
        
        error_prediction <- target_matrix - predicted_output_matrix
        differences      <- error_prediction
        
      } else {
        # ----- Regression: preserve original shape logic (replicate/truncate columns) -----
        if (ncol(Lm0) != ncol(Pm0)) {
          total_elements_needed <- nrow(Lm0) * ncol(Lm0)
          if (ncol(Pm0) < ncol(Lm0)) {
            rep_factor <- ceiling(total_elements_needed / length(Pm0))
            replicated_predicted_output <- rep(Pm0, rep_factor)[1:total_elements_needed]
            predicted_output_matrix <- matrix(replicated_predicted_output,
                                              nrow = nrow(Lm0), ncol = ncol(Lm0), byrow = FALSE)
          } else {
            truncated_predicted_output <- Pm0[, 1:ncol(Lm0), drop = FALSE]
            predicted_output_matrix <- matrix(truncated_predicted_output,
                                              nrow = nrow(Lm0), ncol = ncol(Lm0), byrow = FALSE)
          }
        } else {
          predicted_output_matrix <- Pm0
        }
        target_matrix    <- Lm0
        error_prediction <- target_matrix - predicted_output_matrix
        differences      <- error_prediction
      }
      
      # --- Calculate summary statistics (as in original) ---
      summary_stats <- summary(differences)
      boxplot_stats <- boxplot.stats(as.numeric(differences))
      
      # --- Load plot_epochs from file (placeholder kept) ---
      # plot_epochs <- readRDS(paste0("plot_epochs_DESONN", ensemble_number, "SONN", model_iter_num, ".rds"))
      plot_epochs <- NULL
      
      
      # --- Generate model_serial_num (preserved) ---
      model_serial_num <- sprintf("%d.0.%d", as.integer(ensemble_number), as.integer(model_iter_num))
      
      # --- Build filename prefix / artifact names (preserved) ---
      fname <- make_fname_prefix(
        do_ensemble     = isTRUE(get0("do_ensemble", ifnotfound = FALSE, inherits = TRUE)),
        num_networks    = get0("num_networks", ifnotfound = NULL, inherits = TRUE),
        total_models    = get0("num_networks", ifnotfound = NULL, inherits = TRUE),
        ensemble_number = ensemble_number,
        model_index     = model_iter_num,
        who             = "SONN"
      )
      artifact_names <- list(
        training_accuracy_loss_plot = fname("training_accuracy_loss_plot.png"),
        output_saturation_plot      = fname("output_saturation_plot.png"),
        max_weight_plot             = fname("max_weight_plot.png")
      )
      artifact_paths <- lapply(artifact_names, function(nm) file.path("plots", nm))
      
      # --- Create metadata list (preserved, using explicit CLASSIFICATION_MODE) ---
      metadata <- list(
        input_size = input_size,
        output_size = output_size,
        N = N,
        never_ran_flag = never_ran_flag,
        num_samples = total_num_samples,
        num_test_samples = num_test_samples,
        num_training_samples = num_training_samples,
        num_validation_samples = num_validation_samples,
        num_networks = num_networks,
        update_weights = update_weights,
        update_biases = update_biases,
        lr = lr,
        lambda = lambda,
        num_epochs = num_epochs,
        optimal_epoch = predicted_outputAndTime$optimal_epoch,
        run_id = run_id,
        ensemble_number = ensemble_number,
        model_iter_num = model_iter_num,
        model_serial_num = model_serial_num,
        threshold = threshold,
        CLASSIFICATION_MODE = mode,
        
        # --- Predictions / errors
        predicted_output = predicted_output,
        predicted_output_tail = tail(predicted_output),
        actual_values_tail = tail(actual_values),
        differences = tail(differences),
        summary_stats = summary_stats,
        boxplot_stats = boxplot_stats,
        
        # --- Preprocessing
        preprocessScaledData = preprocessScaledData,
        target_transform = preprocessScaledData$target_transform,
        
        # --- Data
        X = X,
        y = y,
        X_test = X_test_scaled,
        y_test = y_test,
        
        # --- Training state
        lossesatoptimalepoch = predicted_outputAndTime$lossesatoptimalepoch,
        loss_increase_flag = predicted_outputAndTime$loss_increase_flag,
        performance_metric = performance_metric,
        relevance_metric = relevance_metric,
        plot_epochs = plot_epochs,
        best_weights_record = all_weights,
        best_biases_record = all_biases,
        
        # --- Artifacts
        fname_artifact_names = artifact_names,
        fname_artifact_paths = artifact_paths,
        validation_metrics = validation_metrics,
        
        # ✅ NEW: model-critical configs
        activation_functions = activation_functions,
        dropout_rates        = dropout_rates,
        hidden_sizes         = self$hidden_sizes %||% hidden_sizes,
        output_size          = self$output_size %||% output_size,
        ML_NN                = self$ML_NN %||% ML_NN,
        
        best_val_prediction_time = best_val_prediction_time,
        best_train_acc = best_train_acc,
        best_epoch_train = best_epoch_train,
        best_val_acc = best_val_acc,
        best_val_epoch = best_val_epoch
      )
      
      
      metadata_main_ensemble <- list()
      metadata_temp_ensemble <- list()
      
      # --- Store metadata by ensemble type (preserved) ---
      if (ensemble_number <= 1) {
        print(paste("Storing metadata for main ensemble model", model_iter_num, "as", model_serial_num))
        assign(paste0("Ensemble_Main_", ensemble_number, "_model_", model_iter_num, "_metadata"),
               metadata, envir = .GlobalEnv)
      } else {
        print(paste("Storing metadata for temp ensemble model", model_iter_num, "as", model_serial_num))
        assign(paste0("Ensemble_Temp_", ensemble_number, "_model_", model_iter_num, "_metadata"),
               metadata, envir = .GlobalEnv)
      }
    }
    
    
    
  )
)

is_binary <- function(column) {
  unique_values <- unique(column)
  return(length(unique_values) == 2)
}





clip_gradient_norm <- function(gradient, min_norm = 1e-3, max_norm = 5) {
  if (any(is.na(gradient)) || all(gradient == 0)) return(gradient)
  
  grad_norm <- sqrt(sum(gradient^2, na.rm = TRUE))
  
  if (is.na(grad_norm)) return(gradient)  # added line
  
  if (grad_norm > max_norm) {
    gradient <- gradient * (max_norm / grad_norm)
  } else if (grad_norm < min_norm && grad_norm > 0) {
    gradient <- gradient * (min_norm / grad_norm)
  }
  
  return(gradient)
}

lr_scheduler <- function(epoch, initial_lr = lr, decay_rate = 0.5, decay_epoch = 20, min_lr = 1e-6) {
  decayed_lr <- initial_lr * decay_rate ^ floor(epoch / decay_epoch)
  return(max(min_lr, decayed_lr))
}

calculate_performance <- function(SONN, Rdata, labels, lr, CLASSIFICATION_MODE, model_iter_num, num_epochs, threshold, learn_time, predicted_output, prediction_time, ensemble_number, run_id, weights, biases, activation_functions, ML_NN, verbose) {
  
  # --- Elbow method for clustering (robust) ---
  calculate_wss <- function(X, max_k = 15L) {
    max_k <- min(max_k, max(2L, nrow(X) - 1L))
    wss <- numeric(max_k)
    for (k in 1:max_k) {
      wss[k] <- kmeans(X, centers = k, iter.max = 20)$tot.withinss
    }
    wss
  }
  wss <- calculate_wss(Rdata)
  dd  <- diff(diff(wss))
  if (length(dd)) {
    optimal_k <- max(2L, which.max(dd) + 1L)
  } else {
    optimal_k <- min(3L, max(2L, nrow(Rdata) - 1L))
  }
  cluster_assignments <- kmeans(Rdata, centers = optimal_k, iter.max = 50)$cluster
  
  
  cat("Length of SONN$weights: ", length(SONN$weights), "\n")
  cat("Length of SONN$map: ", if (is.null(SONN$map)) "NULL" else length(SONN$map), "\n")
  
  
  # --- Metrics (all take SONN) ---
  perf_metrics <- list(
    quantization_error            = quantization_error(SONN, Rdata, run_id, verbose),
    topographic_error             = topographic_error(SONN, Rdata, threshold, verbose),
    clustering_quality_db         = clustering_quality_db(SONN, Rdata, cluster_assignments, verbose),
    MSE                           = MSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    MAE                           = MAE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    RMSE                          = RMSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    R2                            = R2(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    MAPE                          = MAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    SMAPE                         = SMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    WMAPE                         = WMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    MASE                          = MASE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    accuracy                      = accuracy(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    precision                     = precision(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    recall                        = recall(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    f1_score                      = f1_score(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    confusion_matrix              = confusion_matrix(SONN, labels, CLASSIFICATION_MODE, predicted_output, threshold, verbose),
    accuracy_precision_recall_f1_tuned                = accuracy_precision_recall_f1_tuned(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, metric_for_tuning = "accuracy", grid, verbose),
    speed                         = speed(SONN, prediction_time, verbose),
    speed_learn                   = speed_learn(SONN, learn_time, verbose),
    memory_usage                  = memory_usage(SONN, Rdata, verbose),
    robustness                    = robustness(SONN, Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, dropout_rates, verbose),
    custom_relative_error_binned  = custom_relative_error_binned(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
  )
  

  for (name in names(perf_metrics)) {
    val <- perf_metrics[[name]]
    if (is.null(val) || any(is.na(val)) || isTRUE(val)) {
      perf_metrics[[name]] <- NA_real_
    }
  }
  
  
  
  return(list(metrics = perf_metrics, names = names(perf_metrics)))
}


calculate_relevance <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, ML_NN, verbose) {
  
  # --- Standardize single-layer to list format ---
  if (!is.list(SONN$weights)) {
    SONN$weights <- list(SONN$weights)
  }
  
  # --- Active relevance metrics ---
  rel_metrics <- list(
    hit_rate     = tryCatch(hit_rate(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose), error = function(e) NULL),
    ndcg         = tryCatch(ndcg(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose), error = function(e) NULL),
    diversity    = tryCatch(diversity(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose), error = function(e) NULL),
    serendipity  = tryCatch(serendipity(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose), error = function(e) NULL)
  )
  
  
  # --- Inactive for future implementation ---
  # precision_boolean = precision_boolean(...)
  # recall            = recall(...)
  # f1_score          = f1_score(...)
  # mean_precision    = mean_precision(...)
  # novelty           = novelty(...)
  
  # --- Validate and clean ---
  all_possible_metrics <- c("hit_rate", "ndcg", "diversity", "serendipity",
                            "precision_boolean", "recall", "f1_score", "mean_precision", "novelty")
  
  for (metric_name in all_possible_metrics) {
    if (!metric_name %in% names(rel_metrics)) {
      rel_metrics[[metric_name]] <- NA_real_
    } else {
      val <- rel_metrics[[metric_name]]
      if (is.null(val) || any(is.na(val)) || isTRUE(val)) {
        rel_metrics[[metric_name]] <- NA_real_
      }
    }
  }
  
  
  return(list(metrics = rel_metrics, names = names(rel_metrics)))
}




calculate_performance_grouped <- function(SONN_list, Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, threshold, predicted_output_list, prediction_time_list, ensemble_number, run_id, ML_NN, verbose,
                                          agg_method = c("mean","median","vote"),
                                          metric_mode = c("aggregate_predictions+rep_sonn", "average_per_model"),
                                          weights_list = NULL, biases_list = NULL, act_list = NULL
) {
  agg_method <- match.arg(agg_method)
  metric_mode <- match.arg(metric_mode)
  
  # 1) Aggregate predictions once
  p_agg <- aggregate_predictions(predicted_output_list, method = agg_method)
  pred_time_agg <- mean(unlist(prediction_time_list), na.rm = TRUE)
  
  if (metric_mode == "aggregate_predictions+rep_sonn") {
    # 2a) Use ONE representative SONN (best F1) so we can reuse your metric code as-is
    rep_sonn <- pick_representative_sonn(SONN_list, predicted_output_list, labels)
    rep_w <- rep_sonn$weights; rep_b <- rep_sonn$biases; rep_af <- rep_sonn$activation_functions
    
    calculate_performance(
      SONN             = rep_sonn,
      Rdata            = Rdata,
      labels           = labels,
      lr               = lr,
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      model_iter_num   = NA_integer_,
      num_epochs       = num_epochs,
      threshold        = threshold,
      learn_time       = NA_real_,
      predicted_output = p_agg,
      prediction_time  = pred_time_agg,
      ensemble_number  = ensemble_number,
      run_id           = paste0(run_id[[1]], "::GROUP"),
      weights          = rep_w,
      biases           = rep_b,
      activation_functions = rep_af,
      ML_NN            = ML_NN,
      verbose          = verbose
    )
  } else {
    # 2b) Average per-model metrics (no re-implementation): compute each, then average numerics
    perfs <- lapply(seq_along(SONN_list), function(i) {
      calculate_performance(
        SONN             = SONN_list[[i]],
        Rdata            = Rdata,
        labels           = labels,
        lr               = lr,
        CLASSIFICATION_MODE = CLASSIFICATION_MODE,
        model_iter_num   = i,
        num_epochs       = num_epochs,
        threshold        = threshold,
        learn_time       = NA_real_,
        predicted_output = predicted_output_list[[i]],
        prediction_time  = prediction_time_list[[i]],
        ensemble_number  = ensemble_number,
        run_id           = run_id[[i]],
        weights          = if (is.null(weights_list)) SONN_list[[i]]$weights else weights_list[[i]],
        biases           = if (is.null(biases_list))  SONN_list[[i]]$biases  else biases_list[[i]],
        activation_functions = if (is.null(act_list)) SONN_list[[i]]$activation_functions else act_list[[i]],
        ML_NN            = ML_NN,
        verbose          = FALSE
      )
    })
    # fold to a single metrics list by averaging numeric leafs
    keys <- Reduce(union, lapply(perfs, `[[`, "names"))
    avg_metrics <- lapply(keys, function(k) {
      vals <- lapply(perfs, function(p) p$metrics[[k]])
      nums <- suppressWarnings(as.numeric(unlist(vals)))
      if (all(is.na(nums))) NULL else mean(nums, na.rm = TRUE)
    })
    names(avg_metrics) <- keys
    list(metrics = avg_metrics, names = names(avg_metrics))
  }
}

calculate_relevance_grouped <- function(SONN_list, Rdata, labels, CLASSIFICATION_MODE, predicted_output_list, ensemble_number, run_id, ML_NN, verbose,
                                        agg_method = c("mean","median","vote"),
                                        metric_mode = c("aggregate_predictions+rep_sonn", "average_per_model")
) {
  agg_method <- match.arg(agg_method)
  metric_mode <- match.arg(metric_mode)
  p_agg <- aggregate_predictions(predicted_output_list, method = agg_method)
  
  if (metric_mode == "aggregate_predictions+rep_sonn") {
    rep_sonn <- pick_representative_sonn(SONN_list, predicted_output_list, labels)
    calculate_relevance(
      SONN                 = rep_sonn,
      Rdata                = Rdata,
      labels               = labels,
      CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
      model_iter_num       = NA_integer_,
      predicted_output     = p_agg,
      ensemble_number      = ensemble_number,
      weights              = rep_sonn$weights,
      biases               = rep_sonn$biases,
      activation_functions = rep_sonn$activation_functions,
      ML_NN                = ML_NN,
      verbose              = verbose
    )
  } else {
    rels <- lapply(seq_along(SONN_list), function(i) {
      calculate_relevance(
        SONN                 = SONN_list[[i]],
        Rdata                = Rdata,
        labels               = labels,
        CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
        model_iter_num       = i,
        predicted_output     = predicted_output_list[[i]],
        ensemble_number      = ensemble_number,
        weights              = SONN_list[[i]]$weights,
        biases               = SONN_list[[i]]$biases,
        activation_functions = SONN_list[[i]]$activation_functions,
        ML_NN                = ML_NN,
        verbose              = FALSE
      )
    })
    keys <- Reduce(union, lapply(rels, `[[`, "names"))
    avg_metrics <- lapply(keys, function(k) {
      vals <- lapply(rels, function(r) r$metrics[[k]])
      nums <- suppressWarnings(as.numeric(unlist(vals)))
      if (all(is.na(nums))) NULL else mean(nums, na.rm = TRUE)
    })
    names(avg_metrics) <- keys
    list(metrics = avg_metrics, names = names(avg_metrics))
  }
}



# Unified Loss Function (minimal edits, original style preserved)
# Supports binary, multiclass, and regression.
# Strictly enforces valid (mode, loss_type) combos with clear error messages.
loss_function <- function(predictions, labels, CLASSIFICATION_MODE, reg_loss_total, loss_type) {
  # Default reg_loss_total to 0 if NULL
  if (is.null(reg_loss_total)) reg_loss_total <- 0
  
  print(dim(predictions))
  print(dim(labels))
  
  # Handle missing or NULL loss_type gracefully
  if (is.null(loss_type)) {
    print("Loss type is NULL. Please specify 'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'.")
    return(NA)
  }
  
  # Normalize inputs
  mode <- tolower(as.character(CLASSIFICATION_MODE))
  lt   <- tolower(as.character(loss_type))
  
  # ---- Compatibility checks (clear + strict) ----
  if (mode == "regression") {
    if (lt %in% c("crossentropy", "categoricalcrossentropy")) {
      stop("Invalid loss_type for regression: '", loss_type,
           "'. Use 'MSE' or 'MAE' for CLASSIFICATION_MODE = 'regression'.\n",
           "# Tip: Cross-entropy losses require class probabilities and one-hot/0-1 labels; ",
           "regression targets are continuous.")
    }
  } else if (mode == "binary") {
    if (lt == "categoricalcrossentropy") {
      stop("Invalid loss_type for binary: 'CategoricalCrossEntropy'. ",
           "Use 'CrossEntropy' (binary) or 'MSE'/'MAE'.")
    }
  } else if (mode == "multiclass") {
    # all four allowed; CE names both valid here
    # no-op
    ;
  } else {
    stop("Unknown CLASSIFICATION_MODE: must be 'binary', 'multiclass', or 'regression'")
  }
  
  P <- as.matrix(predictions)
  n <- nrow(P); K <- ncol(P)
  
  # small helpers (kept)
  one_hot <- function(idx, n, K) {
    Y <- matrix(0, n, K)
    ok <- !is.na(idx) & idx >= 1 & idx <= K
    if (any(ok)) Y[cbind(which(ok), idx[ok])] <- 1
    Y
  }
  row_softmax <- function(X) {
    X <- as.matrix(X)
    m <- apply(X, 1L, max)
    ex <- exp(sweep(X, 1L, m, "-"))
    ex / rowSums(ex)
  }
  
  if (lt == "mse") {
    loss <- mean((P - as.matrix(labels))^2, na.rm = TRUE)
    
  } else if (lt == "mae") {
    loss <- mean(abs(P - as.matrix(labels)), na.rm = TRUE)
    
  } else if (lt %in% c("crossentropy", "categoricalcrossentropy")) {
    eps <- 1e-12
    
    if (mode == "binary") {
      # Binary Cross-Entropy
      y <- if (is.matrix(labels)) labels[,1] else labels
      y <- suppressWarnings(as.numeric(y))
      if (all(is.na(y))) y <- as.integer(factor(labels)) - 1L
      y[is.na(y)] <- 0
      y <- pmin(pmax(y, 0), 1)
      
      # assume P is sigmoid probs; clamp
      P <- pmin(pmax(P, eps), 1 - eps)
      loss <- -mean(y * log(P) + (1 - y) * log(1 - P))
      
    } else if (mode == "multiclass") {
      stopifnot(K >= 2)
      # If labels already one-hot n×K, use them; else factor -> indices -> one-hot
      if (is.matrix(labels) && nrow(labels) >= n && ncol(labels) == K &&
          all(labels[seq_len(n), , drop = FALSE] %in% c(0,1))) {
        Y <- matrix(as.numeric(labels[seq_len(n), , drop = FALSE]), n, K)
      } else {
        f  <- if (is.factor(labels)) labels else factor(labels)
        ix <- as.integer(f)
        L  <- nlevels(f)
        if (L > K) ix[ix > K] <- K
        Y <- one_hot(ix, n, K)
      }
      # ensure probabilities
      if (any(P < 0) || any(P > 1) || any(abs(rowSums(P) - 1) > 1e-6)) {
        P <- row_softmax(P)
      }
      P <- pmin(pmax(P, eps), 1 - eps)
      loss <- -mean(rowSums(Y * log(P)))
      
    } else if (mode == "regression") {
      # Should never get here due to compatibility check above
      stop("Cross-entropy not valid for regression.")
    }
    
  } else {
    print("Invalid loss type. Choose from 'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'.")
    return(NA)
  }
  
  total_loss <- loss + reg_loss_total
  return(total_loss)
}



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#      _     _      _     _      _     _      _     _      _     _      _     _      _     _    $$$$$$$$$$$$$$$$$$$$$$$
#     (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)   $$$$$$$$$$$$$$$$$$$$$$$
#     / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \     $$$$$$$$$$$$$$$$$$$$$$$
#   __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__   $$$$$$$$$$$$$$$$$$$$$$$
#  (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)  $$$$$$$$$$$$$$$$$$$$$$$
#    || M ||      || E ||      || T ||      || R ||      || I ||      || C ||      || S ||      $$$$$$$$$$$$$$$$$$$$$$$
# _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._     $$$$$$$$$$$$$$$$$$$$$$$
#(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)    $$$$$$$$$$$$$$$$$$$$$$$
# `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'     $$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

quantization_error <- function(SONN, Rdata, run_id, verbose) {
  
  # keep your structure; just coerce data once
  if (!is.matrix(Rdata)) Rdata <- as.matrix(Rdata)
  storage.mode(Rdata) <- "double"
  
  if (ML_NN) {
    # --- ML path: get a matrix W from SONN$weights (first input-facing layer) ---
    if (is.list(SONN$weights)) {
      # pick the first layer that matches NCOL(Rdata); fallback to first layer
      idx <- which(vapply(SONN$weights, function(w)
        is.matrix(w) && (ncol(w) == NCOL(Rdata) || nrow(w) == NCOL(Rdata)), logical(1L)))[1]
      if (is.na(idx)) idx <- 1L
      W <- as.matrix(SONN$weights[[idx]])
    } else if (is.matrix(SONN$weights)) {
      W <- as.matrix(SONN$weights)
    } else if (is.matrix(SONN)) {
      # legacy: codebook passed directly as a matrix
      W <- as.matrix(SONN)
    } else {
      if (isTRUE(verbose)) cat("[quantization error]: NA\n")
      return(NA_real_)
    }
    
  } else {
    
    # --- SL path: expect SONN$weights to be a matrix ---
    if (!is.matrix(SONN$weights)) SONN$weights <- as.matrix(SONN$weights)
    W <- SONN$weights
  }
  
  # orient W so columns = features in Rdata
  storage.mode(W) <- "double"
  if (ncol(W) != NCOL(Rdata) && nrow(W) == NCOL(Rdata)) W <- t(W)
  if (ncol(W) != NCOL(Rdata)) {
    if (isTRUE(verbose)) cat("[quantization error]: NA\n")
    return(NA_real_)
  }
  
  # === your original distance logic (but use W, not SONN) ===
  distances <- apply(Rdata, 1L, function(x) {
    neuron_distances <- apply(W, 1L, function(w) {
      sqrt(sum((x - w)^2))
    })
    min(neuron_distances)
  })
  
  if (!length(distances) || all(!is.finite(distances))) {
    if (isTRUE(verbose)) cat("[quantization error]: NA\n")
    return(NA_real_)
  }
  
  mean_distance <- mean(distances, na.rm = TRUE)
  
  # if (isTRUE(verbose)) {
  #   cat("[quantization error]: ", format(mean_distance, digits = 7), "\n", sep = "")
  # }
  return(mean_distance)
}

# Model-only topo error: uses layer-1 weights + map inside SONN
topographic_error <- function(SONN, Rdata, threshold, verbose) {
  # --- normalize to list-of-matrix ---
  if (is.matrix(SONN)) SONN <- list(weights = list(as.matrix(SONN)))
  if (is.matrix(SONN$weights)) SONN$weights <- list(as.matrix(SONN$weights))
  if (is.null(SONN$map)) {
    m <- nrow(SONN$weights[[1]])
    r <- max(1L, floor(sqrt(m))); while (m %% r && r > 1L) r <- r - 1L
    c <- max(1L, m %/% r)
    SONN$map <- list(matrix(seq_len(m), nrow = r, ncol = c, byrow = TRUE))
  } else if (!is.list(SONN$map)) {
    SONN$map <- list(as.matrix(SONN$map))
  } else if (!is.matrix(SONN$map[[1]])) {
    SONN$map[[1]] <- as.matrix(SONN$map[[1]])
  }
  
  W <- as.matrix(SONN$weights[[1]])   # units x features
  M <- as.matrix(SONN$map[[1]])       # grid with labels 1..m
  X <- as.matrix(Rdata)
  storage.mode(W) <- "double"; storage.mode(X) <- "double"
  
  # ---------- ALIGN W to X (prevents sweep() length warnings) ----------
  align_to_X <- function(W, X) {
    # Name-aware alignment if both have colnames
    if (!is.null(colnames(W)) && !is.null(colnames(X))) {
      wanted <- colnames(X)
      miss <- setdiff(wanted, colnames(W))
      if (length(miss)) {
        W <- cbind(W, matrix(0, nrow = nrow(W), ncol = length(miss),
                             dimnames = list(NULL, miss)))
      }
      # drop extras and reorder to X's order
      W <- W[, wanted, drop = FALSE]
      return(W)
    }
    # Fallback: align by width only
    kx <- ncol(X); kw <- ncol(W)
    if (kw > kx) {
      W[, seq_len(kx), drop = FALSE]
    } else if (kw < kx) {
      cbind(W, matrix(0, nrow = nrow(W), ncol = kx - kw))
    } else {
      W
    }
  }
  W <- align_to_X(W, X)
  
  m <- nrow(W); n <- nrow(X)
  if (m < 2L || n < 1L) return(NA_real_)
  
  # If map size doesn't match units, rebuild a compact grid
  if (length(M) != m) {
    r <- max(1L, floor(sqrt(m))); while (m %% r && r > 1L) r <- r - 1L
    c <- max(1L, m %/% r)
    M <- matrix(seq_len(m), nrow = r, ncol = c, byrow = TRUE)
  }
  
  if (isTRUE(verbose)) {
    cat(sprintf("[topo] dim(X)=%dx%d | dim(W)=%dx%d | units=%d\n",
                nrow(X), ncol(X), nrow(W), ncol(W), m))
    if (!is.null(colnames(X)) && !is.null(colnames(W))) {
      same_names <- identical(colnames(X), colnames(W))
      cat("[topo] colnames(X)==colnames(W): ", same_names, "\n", sep = "")
    }
  }
  
  # ---------- Fast pairwise squared distances: no sweep(), no warnings ----------
  # D_ij = ||X_i||^2 + ||W_j||^2 - 2 * X_i · W_j
  X2 <- rowSums(X * X)                  # n x 1
  W2 <- rowSums(W * W)                  # m x 1
  G  <- X %*% t(W)                      # n x m
  D  <- matrix(X2, n, m) + matrix(W2, n, m, byrow = TRUE) - 2 * G
  
  # BMU / second-BMU
  bmu <- max.col(-D, ties.method = "first")
  D[cbind(seq_len(n), bmu)] <- Inf
  sbmu <- max.col(-D, ties.method = "first")
  
  # grid coords for units 1..m
  coords <- matrix(NA_integer_, m, 2)
  for (k in 1:m) {
    pos <- which(M == k, arr.ind = TRUE)
    coords[k, ] <- if (length(pos)) c(pos[1, 1], pos[1, 2]) else c(1L, k)
  }
  
  dgrid <- sqrt(rowSums((coords[bmu, , drop = FALSE] - coords[sbmu, , drop = FALSE])^2))
  err <- mean(dgrid > 1)
  
  if (isTRUE(verbose)) cat("[topo] error =", err, "\n")
  err
}

is.adjacent <- function(map, neuron1, neuron2) {
  # 💡 Ensure map rownames exist and match neuron indices
  if (is.null(rownames(map))) {
    rownames(map) <- as.character(1:nrow(map))
  }
  
  coord1 <- map[as.character(neuron1), , drop = FALSE]
  coord2 <- map[as.character(neuron2), , drop = FALSE]
  
  if (nrow(coord1) == 0 || nrow(coord2) == 0) {
    stop(paste("Neurons", neuron1, "or", neuron2, "not found in the map"))
  }
  
  grid_dist <- sum(abs(coord1 - coord2))
  
  return(grid_dist == 1)
}

# Clustering quality (Davies-Bouldin index)
clustering_quality_db <- function(SONN, Rdata, cluster_assignments, verbose) {
  if (is.null(cluster_assignments)) {
    stop("Cluster assignments not available. Perform kmeans clustering first.")
  }
  
  # Ensure Rdata is a numeric matrix
  if (!is.matrix(Rdata)) Rdata <- as.matrix(Rdata)
  Rdata <- apply(Rdata, 2, as.numeric)  # force numeric values
  
  # Compute centroids and ensure it's numeric matrix
  centroids <- aggregate(Rdata, by = list(cluster_assignments), FUN = mean)[, -1]
  centroids <- as.matrix(centroids)
  centroids <- apply(centroids, 2, as.numeric)
  
  n_clusters <- nrow(centroids)
  
  # Split indices by cluster
  cluster_indices <- split(seq_len(nrow(Rdata)), cluster_assignments)
  
  # Precompute intra-cluster dispersion for each cluster (Si)
  S <- numeric(n_clusters)
  for (i in seq_len(n_clusters)) {
    data_i <- Rdata[cluster_indices[[i]], , drop = FALSE]
    centroid_i <- centroids[i, ]
    centroid_matrix <- matrix(centroid_i, nrow(data_i), ncol(data_i), byrow = TRUE)
    S[i] <- mean(rowSums((data_i - centroid_matrix)^2))
  }
  
  # Precompute inter-cluster distances (squared Euclidean)
  D <- as.matrix(dist(centroids))^2
  
  # Compute Davies-Bouldin index
  db_index <- 0
  for (i in seq_len(n_clusters)) {
    max_ratio <- -Inf
    for (j in seq_len(n_clusters)) {
      if (i != j) {
        ratio <- (S[i] + S[j]) / sqrt(D[i, j])
        if (ratio > max_ratio) max_ratio <- ratio
      }
    }
    db_index <- db_index + max_ratio
  }
  
  db_index <- db_index / n_clusters
  
  # if (verbose) {
  #   cat("clustering_quality_db:", db_index, "\n")
  # }
  
  return(db_index)
}


# Debug MSE that USES CLASSIFICATION_MODE for shape handling (silent version)
# Signature: MSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MSE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  # support %||% like base R (define before use)
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  # --- helpers ---
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      mm <- model.matrix(~ . - 1, data = x)
      x <- as.matrix(mm)
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1)
    } else {
      x <- as.matrix(x)
    }
    storage.mode(x) <- "double"
    x
  }
  one_hot <- function(idx, K) {
    idx <- as.integer(idx)
    idx[is.na(idx)] <- 1L
    idx[idx < 1L] <- 1L
    idx[idx > K]  <- K
    M <- matrix(0, nrow = length(idx), ncol = K)
    if (length(idx)) M[cbind(seq_along(idx), idx)] <- 1
    storage.mode(M) <- "double"
    M
  }
  
  # --- coerce to numeric matrices ---
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  
  # --- ROW ALIGNMENT (no recycling) ---
  n_common <- min(nrow(lbl), nrow(pred))
  if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # --- MODE-BASED SHAPE HARMONIZATION ---
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  
  if (identical(mode, "multiclass")) {
    # Expect pred: N x K
    K <- ncol(pred)
    if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      # class index 1..K or 0..K-1 -> one-hot
      u <- sort(unique(as.integer(lbl[, 1])))
      if (length(u)) {
        if (min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) {
          lbl <- lbl + 1
        }
      }
      lbl <- one_hot(lbl[, 1], K)
    } else if (ncol(lbl) != K) {
      return(NA_real_)
    }
    
  } else if (identical(mode, "binary")) {
    # Conventions:
    #   pred: N x 1 (p_pos) OR N x 2 ([p_neg, p_pos])
    #   lbl : N x 1 (0/1 or two unique values) OR N x 2 one-hot
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      # Reduce labels to 0/1 using column 2 as positive
      lbl <- matrix(lbl[, 2], ncol = 1L)
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      # Expand labels to one-hot [neg, pos]
      v <- as.integer(round(lbl[, 1]))
      v[is.na(v)] <- 0L
      v[v != 0L]  <- 1L
      lbl <- cbind(1 - v, v)
      storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      # Ensure {0,1}
      uniq <- sort(unique(as.integer(lbl[, 1])))
      if (!all(uniq %in% c(0L, 1L))) {
        if (length(uniq) == 2L) {
          lbl[, 1] <- ifelse(lbl[, 1] == uniq[2], 1, 0)
        } else {
          return(NA_real_)
        }
      }
    } else if (ncol(pred) == 2L && ncol(lbl) == 2L) {
      # already aligned
      # nothing to do
    } else {
      return(NA_real_)
    }
    
  } else if (!identical(mode, "regression")) {
    # Unknown mode
    return(NA_real_)
  }
  
  # --- FINAL COLUMN CHECK (avoid replicate/truncate for class probs) ---
  if (ncol(lbl) != ncol(pred)) {
    if (identical(mode, "regression")) {
      # Conform via replicate/truncate
      total_needed <- nrow(lbl) * ncol(lbl)
      if (ncol(pred) < ncol(lbl)) {
        vec <- rep(pred, length.out = total_needed)
        pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
      } else {
        pred <- pred[, 1:ncol(lbl), drop = FALSE]
        pred <- matrix(pred, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
      }
    } else {
      return(NA_real_)
    }
  }
  
  # --- NA handling ---
  cc <- stats::complete.cases(cbind(lbl, pred))
  if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]
  pred <- pred[cc, , drop = FALSE]
  if (!nrow(lbl)) return(NA_real_)
  
  # --- Compute error + MSE ---
  err <- pred - lbl
  mse <- mean(err^2)
  
  mse
}

# Debug R^2 that USES CLASSIFICATION_MODE for shape handling (silent version)
# Signature: R2(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
R2 <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  # --- helpers ---
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      mm <- model.matrix(~ . - 1, data = x); x <- as.matrix(mm)
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1)
    } else x <- as.matrix(x)
    storage.mode(x) <- "double"; x
  }
  one_hot <- function(idx, K) {
    idx <- as.integer(idx); idx[is.na(idx)] <- 1L; idx[idx < 1L] <- 1L; idx[idx > K] <- K
    M <- matrix(0, nrow = length(idx), ncol = K)
    if (length(idx)) M[cbind(seq_along(idx), idx)] <- 1
    storage.mode(M) <- "double"; M
  }
  
  # --- coerce ---
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  
  # --- row align ---
  n_common <- min(nrow(lbl), nrow(pred)); if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # --- mode-based shape handling ---
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  
  if (identical(mode, "multiclass")) {
    K <- ncol(pred); if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(lbl[, 1])))
      if (length(u) && min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) lbl <- lbl + 1
      lbl <- one_hot(lbl[, 1], K)
    } else if (ncol(lbl) != K) return(NA_real_)
    return(NA_real_)  # R² not meaningful for classification
    
  } else if (identical(mode, "binary")) {
    # Align like in MSE, then bail (R² not meaningful for classification)
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      lbl <- matrix(lbl[, 2], ncol = 1L)
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      v <- as.integer(round(lbl[, 1])); v[is.na(v)] <- 0L; v[v != 0L] <- 1L
      lbl <- cbind(1 - v, v); storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      uniq <- sort(unique(as.integer(lbl[, 1])))
      if (!all(uniq %in% c(0L, 1L))) {
        if (length(uniq) == 2L) lbl[, 1] <- ifelse(lbl[, 1] == uniq[2], 1, 0) else return(NA_real_)
      }
    } else if (!(ncol(pred) == 2L && ncol(lbl) == 2L)) return(NA_real_)
    return(NA_real_)
  } else if (!identical(mode, "regression")) return(NA_real_)
  
  # --- final column check (regression may replicate/truncate) ---
  if (ncol(lbl) != ncol(pred)) {
    total_needed <- nrow(lbl) * ncol(lbl)
    if (ncol(pred) < ncol(lbl)) {
      vec <- rep(pred, length.out = total_needed)
      pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    } else pred <- pred[, 1:ncol(lbl), drop = FALSE]
  }
  
  # --- NA handling ---
  cc <- stats::complete.cases(cbind(lbl, pred)); if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]; pred <- pred[cc, , drop = FALSE]; if (!nrow(lbl)) return(NA_real_)
  
  # --- R² across all columns (if multi-target) ---
  y  <- as.numeric(lbl)
  yhat <- as.numeric(pred)
  ss_res <- sum((y - yhat)^2)
  ss_tot <- sum((y - mean(y))^2)
  if (ss_tot == 0) return(NA_real_)
  1 - ss_res / ss_tot
}

# Signature: MAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MAPE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                 eps_abs = 1e-6, eps_pct = 1e-3) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  to_num_mat <- function(x) {
    if (is.data.frame(x)) x <- model.matrix(~ . - 1, x)
    else if (is.factor(x) || is.character(x)) x <- matrix(as.numeric(as.factor(x)), ncol = 1)
    else if (is.atomic(x) && !is.matrix(x)) x <- matrix(x, ncol = 1)
    else x <- as.matrix(x)
    storage.mode(x) <- "double"; x
  }
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)
  
  y   <- to_num_mat(labels)
  yhat<- to_num_mat(predicted_output)
  
  n <- min(nrow(y), nrow(yhat)); if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  # Conform columns for regression
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) {
      yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    } else yhat <- yhat[, 1:ncol(y), drop = FALSE]
  }
  
  cc <- stats::complete.cases(cbind(y, yhat)); if (!any(cc)) return(NA_real_)
  y <- y[cc, , drop = FALSE]; yhat <- yhat[cc, , drop = FALSE]; if (!nrow(y)) return(NA_real_)
  
  # Denominator floor: max(|y|, eps_abs, eps_pct*mean|y|)
  mean_abs_y <- max(mean(abs(y)), eps_abs)
  denom_floor <- max(eps_abs, eps_pct * mean_abs_y)
  
  rel <- abs(yhat - y) / pmax(abs(y), denom_floor)
  mean(rel, na.rm = TRUE) * 100
}

# Signature: SMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
SMAPE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                  eps = 1e-6) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  to_num_mat <- function(x) {
    if (is.data.frame(x)) x <- model.matrix(~ . - 1, x)
    else if (is.factor(x) || is.character(x)) x <- matrix(as.numeric(as.factor(x)), ncol = 1)
    else if (is.atomic(x) && !is.matrix(x)) x <- matrix(x, ncol = 1)
    else x <- as.matrix(x)
    storage.mode(x) <- "double"; x
  }
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)
  
  y   <- to_num_mat(labels)
  yhat<- to_num_mat(predicted_output)
  
  n <- min(nrow(y), nrow(yhat)); if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    else yhat <- yhat[, 1:ncol(y), drop = FALSE]
  }
  
  cc <- stats::complete.cases(cbind(y, yhat)); if (!any(cc)) return(NA_real_)
  y <- y[cc, , drop = FALSE]; yhat <- yhat[cc, , drop = FALSE]; if (!nrow(y)) return(NA_real_)
  
  denom <- (abs(y) + abs(yhat)) / 2
  ok <- is.finite(denom) & (denom > eps)
  if (!any(ok)) return(NA_real_)
  mean(abs(yhat[ok] - y[ok]) / denom[ok]) * 100
}

# Signature: WMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
WMAPE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                  denom_floor = 1e-6) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  to_num_mat <- function(x) {
    if (is.data.frame(x)) x <- model.matrix(~ . - 1, x)
    else if (is.factor(x) || is.character(x)) x <- matrix(as.numeric(as.factor(x)), ncol = 1)
    else if (is.atomic(x) && !is.matrix(x)) x <- matrix(x, ncol = 1)
    else x <- as.matrix(x)
    storage.mode(x) <- "double"; x
  }
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)
  
  y   <- to_num_mat(labels)
  yhat<- to_num_mat(predicted_output)
  
  n <- min(nrow(y), nrow(yhat)); if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    else yhat <- yhat[, 1:ncol(y), drop = FALSE]
  }
  
  cc <- stats::complete.cases(cbind(y, yhat)); if (!any(cc)) return(NA_real_)
  y <- y[cc, , drop = FALSE]; yhat <- yhat[cc, , drop = FALSE]; if (!nrow(y)) return(NA_real_)
  
  num <- sum(abs(yhat - y))
  den <- sum(abs(y))
  if (!is.finite(den) || den < denom_floor) return(NA_real_)
  (num / den) * 100
}

# Debug MASE that USES CLASSIFICATION_MODE for shape handling (silent version)
# Signature: MASE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MASE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                 lag = 1L) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  # --- helpers ---
  to_num_mat <- function(x) {
    if (is.data.frame(x)) {
      mm <- model.matrix(~ . - 1, data = x); x <- as.matrix(mm)
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1)
    } else x <- as.matrix(x)
    storage.mode(x) <- "double"; x
  }
  
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)
  
  y    <- to_num_mat(labels)
  yhat <- to_num_mat(predicted_output)
  
  # --- align rows ---
  n <- min(nrow(y), nrow(yhat))
  if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  # --- conform columns ---
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) {
      yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    } else {
      yhat <- yhat[, 1:ncol(y), drop = FALSE]
    }
  }
  
  # --- NA handling ---
  cc <- stats::complete.cases(cbind(y, yhat))
  if (!any(cc)) return(NA_real_)
  y    <- y[cc, , drop = FALSE]
  yhat <- yhat[cc, , drop = FALSE]
  if (!nrow(y)) return(NA_real_)
  
  # --- Model error (MAE) ---
  mae_model <- mean(abs(yhat - y))
  
  # --- Baseline error (lag-1 naïve forecast) ---
  if (nrow(y) <= lag) return(NA_real_)
  diffs <- abs(y[(lag + 1):nrow(y), , drop = FALSE] - y[1:(nrow(y) - lag), , drop = FALSE])
  mae_naive <- mean(diffs, na.rm = TRUE)
  
  if (!is.finite(mae_naive) || mae_naive == 0) return(NA_real_)
  
  # --- Ratio: model error / naive error ---
  mase <- mae_model / mae_naive
  mase
}


# Generalization ability
generalization_ability <- function(SONN, Rdata, verbose) {
  # Split the Rdata into training and testing sets
  set.seed(123)
  train_idx <- sample(1:nrow(Rdata), 0.8 * nrow(Rdata))
  train_Rdata <- Rdata[train_idx, ]
  test_Rdata <- Rdata[-train_idx, ]
  
  # if (verbose) {
  #   print("generalization_ability")
  # }
  # if (verbose) {
  #   print("generalization_ability complete")
  # }
}

# Speed
speed_learn <- function(SONN, learn_time, verbose) {
  
  # if (verbose) {
  #   print("speed")
  #   print(learn_time)
  # }
  return(learn_time)
  # if (verbose) {
  #   print("speed complete")
  # }
}

# Speed
speed <- function(SONN, prediction_time, verbose) {
  
  # if (verbose) {
  #   print("speed")
  #   print(prediction_time)
  # }
  return(prediction_time)
  # if (verbose) {
  #   print("speed complete")
  # }
}

# Memory usage
memory_usage <- function(SONN, Rdata, verbose) {
  
  # Calculate the memory usage of the SONN object
  object_size <- object.size(SONN)
  
  # Calculate the memory usage of the Rdata
  Rdata_size <- object.size(Rdata)
  # if (verbose) {
  #   print("memory")
  #   print(object_size + Rdata_size)
  # }
  # Return the total memory usage without the word "bytes"
  return(as.numeric(gsub("bytes", "", object_size + Rdata_size)))
  # if (verbose) {
  #   print("memory complete")
  # }
}

# Robustness
robustness <- function(SONN, Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, dropout_rates, verbose) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      mm <- model.matrix(~ . - 1, data = x); x <- as.matrix(mm)
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1L)
    } else {
      x <- as.matrix(x)
    }
    storage.mode(x) <- "double"; x
  }
  one_hot <- function(idx, K) {
    idx <- as.integer(idx)
    idx[idx < 1L] <- 1L; idx[idx > K] <- K
    M <- matrix(0, nrow = length(idx), ncol = K)
    M[cbind(seq_along(idx), idx)] <- 1
    storage.mode(M) <- "double"; M
  }
  
  # build noisy data
  Rdata <- as.matrix(Rdata)
  storage.mode(Rdata) <- "double"
  set.seed(123) # deterministic noise for reproducibility
  noisy_Rdata <- Rdata + rnorm(n = nrow(Rdata) * ncol(Rdata), mean = 0, sd = 0.2)
  noisy_Rdata <- matrix(noisy_Rdata, nrow = nrow(Rdata), ncol = ncol(Rdata))
  storage.mode(noisy_Rdata) <- "double"
  
  # Add outliers
  n_out_rows <- max(1L, min(nrow(noisy_Rdata), as.integer(round(0.02 * nrow(noisy_Rdata)))))
  outliers <- matrix(rnorm(n_out_rows * ncol(noisy_Rdata), mean = 5, sd = 1),
                     nrow = n_out_rows, ncol = ncol(noisy_Rdata))
  idx_rows <- sample.int(nrow(noisy_Rdata), n_out_rows)
  noisy_Rdata[idx_rows, ] <- outliers
  
  learnOnlyTrainingRun <- get0("learnOnlyTrainingRun", ifnotfound = FALSE, inherits = TRUE)
  plot_robustness      <- get0("plot_robustness",      ifnotfound = FALSE, inherits = TRUE)
  
  accuracy <- NA_real_
  
  if (!isTRUE(learnOnlyTrainingRun)) {
    # predict on noisy data
    pred_obj <- SONN$predict(noisy_Rdata, weights, biases, activation_functions, verbose, debug)
    pred_raw <- pred_obj$predicted_output
    
    # coerce to numeric matrices
    lbl  <- to_num_mat(labels, "labels")
    pred <- to_num_mat(pred_raw, "predicted_output")
    
    # row alignment
    n_common <- min(nrow(lbl), nrow(pred))
    if (n_common == 0L) return(NA_real_)
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
    
    # mode-based harmonization
    mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
    
    if (identical(mode, "multiclass")) {
      K <- ncol(pred)
      if (K < 2L) return(NA_real_)
      if (ncol(lbl) == 1L) {
        u <- sort(unique(as.integer(lbl[,1])))
        if (length(u) && min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) {
          lbl <- lbl + 1
        }
        lbl <- one_hot(lbl[,1], K)
      } else if (ncol(lbl) != K) {
        return(NA_real_)
      }
    } else if (identical(mode, "binary")) {
      if (ncol(pred) == 1L && ncol(lbl) == 2L) {
        lbl <- matrix(lbl[,2], ncol = 1L)
      } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
        v <- as.integer(round(lbl[,1])); v[is.na(v)] <- 0L
        lbl <- cbind(1 - v, v); storage.mode(lbl) <- "double"
      } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
        uniq <- sort(unique(as.integer(lbl[,1])))
        if (!all(uniq %in% c(0L,1L))) {
          if (length(uniq) == 2L) {
            lbl[,1] <- ifelse(lbl[,1] == uniq[2], 1, 0)
          } else {
            return(NA_real_)
          }
        }
      } else if (!(ncol(pred) == 2L && ncol(lbl) == 2L)) {
        return(NA_real_)
      }
    } else if (!identical(mode, "regression")) {
      return(NA_real_)
    }
    
    # final column check
    if (!identical(mode, "regression") && ncol(lbl) != ncol(pred)) return(NA_real_)
    if (identical(mode, "regression") && ncol(lbl) != ncol(pred)) {
      total_needed <- nrow(lbl) * ncol(lbl)
      if (ncol(pred) < ncol(lbl)) {
        vec <- rep(pred, length.out = total_needed)
        pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
      } else {
        pred <- pred[, 1:ncol(lbl), drop = FALSE]
        pred <- matrix(pred, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
      }
    }
    
    # NA handling
    cc <- stats::complete.cases(cbind(lbl, pred))
    if (!any(cc)) return(NA_real_)
    lbl  <- lbl[cc, , drop = FALSE]
    pred <- pred[cc, , drop = FALSE]
    if (!nrow(lbl)) return(NA_real_)
    
    # compute MSE on noisy predictions
    err <- pred - lbl
    losses <- mean(err^2)
    accuracy <- losses
    
    # optional plotting (only meaningful with multiple noise Sds)
    if (isTRUE(plot_robustness) && length(losses) > 1L) {
      if (!any(is.nan(losses)) && !any(is.infinite(losses))) {
        plot(losses, type = 'l',
             main = paste('Loss over noise SD for SONN', model_iter_num),
             xlab = 'Noise step', ylab = 'MSE', lwd = 2)
      }
    }
    
  } else {
    # training branch
    invisible(SONN$learn(noisy_Rdata, labels, lr))
  }
  
  return(accuracy)
}

# Hit Rate
hit_rate <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose) {
  Rdata <- data.frame(Rdata)
  
  # Identify the relevant Rdata points
  relevant_Rdata <- Rdata[Rdata$class == "relevant", ]
  
  # Calculate the hit rate
  if (nrow(relevant_Rdata) == 0L) {
    return(NA_real_)
  }
  
  hit_rate <- sum(predicted_output %in% relevant_Rdata$id) / nrow(relevant_Rdata)
  
  hit_rate
}


# -------------------------
# Accuracy (mode-aware: "binary" | "multiclass" | "regression")
# Signature: accuracy(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass: predicted_output must be N×K probs (K>1). labels = class indices (1..K or 0..K-1) or one-hot N×K.
# - Binary    : predicted_output = N×1 (p_pos) or N×2 ([p_neg, p_pos]); labels = 0/1, two unique values, or one-hot N×2.
# - Regression: accuracy undefined -> returns NA_real_.
accuracy <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  dbg <- function(...) if (isTRUE(verbose)) cat(..., "\n")
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      x <- as.matrix(model.matrix(~ . - 1, data = x))
    } else if (is.matrix(x)) {
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1L)
    } else {
      x <- as.matrix(x)
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    }
    storage.mode(x) <- "double"; x
  }
  
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      if (isTRUE(verbose)) dbg("[accuracy] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      if (isTRUE(verbose)) dbg("[accuracy] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  
  
  # ---------- coerce ----------
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[accuracy] Empty labels or predictions.")
  if (isTRUE(verbose)) dbg(sprintf("[accuracy] initial dims: labels=%d x %d | pred=%d x %d",
                                   nrow(lbl), ncol(lbl), nrow(pred), ncol(pred)))
  
  # ---------- row alignment (NO aggressive recycling) ----------
  
  
  if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < 0.9) {
      stop(sprintf("[accuracy] Row mismatch too large (labels=%d, pred=%d). Pass aligned inputs.",
                   nrow(lbl), nrow(pred)))
    }
    if (isTRUE(verbose)) dbg(sprintf("[accuracy] Minor row mismatch -> trimming to %d rows", n_common))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (isTRUE(verbose)) dbg(paste("[accuracy] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[accuracy] Multiclass mode requires predicted_output with K>1 columns.")
    
    # true classes
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    } else {
      stop(sprintf("[accuracy] Multiclass labels have %d columns; predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    
    # predicted classes
    if (ncol(pred) != K) stop("[accuracy] Multiclass predictions must be N×K probabilities.")
    pred_class <- max.col(pred, ties.method = "first")
    
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[accuracy] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    acc <- mean(pred_class == true_class)
    return(round(as.numeric(acc), 8))
    
  } else if (identical(mode, "binary")) {
    # probabilities for positive class
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[accuracy] Unexpected binary prediction shape: %d columns.", ncol(pred)))
    }
    
    # true labels -> 0/1
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # [neg,pos] -> 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[accuracy] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[accuracy] Unexpected binary label shape: %d columns.", ncol(lbl)))
    }
    
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[accuracy] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    acc <- mean(y_pred == y_true)
    return(round(as.numeric(acc), 8))
    
  } else if (identical(mode, "regression")) {
    if (isTRUE(verbose)) dbg("[accuracy] Regression mode: accuracy undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[accuracy] Unhandled mode.")
}

# -------------------------
# Precision (mode-aware: "binary" | "multiclass" | "regression")
# Signature: precision(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass expects predicted_output as N×K probabilities (K>1). labels can be class indices (1..K or 0..K-1) or one-hot N×K.
# - Binary accepts predicted_output as N×1 (p_pos) or N×2 ([p_neg, p_pos]). labels as 0/1, two unique values, or one-hot N×2.
# - Regression: precision is undefined -> returns NA_real_ (with optional debug note).
precision <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  dbg <- function(...) if (isTRUE(verbose)) cat(..., "\n")
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      x <- as.matrix(model.matrix(~ . - 1, data = x))
    } else if (is.matrix(x)) {
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1L)
    } else {
      x <- as.matrix(x)
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    }
    storage.mode(x) <- "double"; x
  }
  
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      # called like precision(..., preds, TRUE)
      if (isTRUE(verbose)) dbg("[precision] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      # called like precision(..., preds)
      if (isTRUE(verbose)) dbg("[precision] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # ---------- coerce ----------
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[precision] Empty labels or predictions.")
  
  # ---------- row alignment (NO aggressive recycling) ----------
  if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < 0.9) {
      stop(sprintf("[precision] Row mismatch too large (labels=%d, pred=%d). Pass aligned inputs.", nrow(lbl), nrow(pred)))
    }
    if (isTRUE(verbose)) dbg(sprintf("[precision] Minor row mismatch -> trimming to %d rows", n_common))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (isTRUE(verbose)) dbg(paste("[precision] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[precision] Multiclass mode requires predicted_output with K>1 columns.")
    
    # Build true_class from labels
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    } else {
      stop(sprintf("[precision] Multiclass labels have %d columns, predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    
    # Predicted class (argmax over K)
    pred_class <- if (ncol(pred) == K) max.col(pred, ties.method = "first") else
      stop("[precision] Multiclass predictions must be N×K probabilities.")
    
    # drop any NA rows (defensive)
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[precision] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    # Macro-averaged precision
    prec_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fp <- sum(pred_class == k & true_class != k)
      prec_per_class[k] <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    }
    return(as.numeric(mean(prec_per_class)))
    
  } else if (identical(mode, "binary")) {
    # Determine positive probabilities and true labels
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]  # assume column 2 is positive class prob
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[precision] Unexpected binary prediction shape: %d cols.", ncol(pred)))
    }
    
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # map one-hot [neg,pos] to 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          # map smaller -> 0, larger -> 1
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[precision] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[precision] Unexpected binary label shape: %d cols.", ncol(lbl)))
    }
    
    # Threshold (0.5 by default)
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[precision] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    tp <- sum(y_pred == 1L & y_true == 1L)
    fp <- sum(y_pred == 1L & y_true == 0L)
    prec <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    return(round(as.numeric(prec), 8))
    
  } else if (identical(mode, "regression")) {
    if (isTRUE(verbose)) dbg("[precision] Regression mode: precision is undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[precision] Unhandled mode.")
}

# -------------------------
# Recall (mode-aware: "binary" | "multiclass" | "regression")
# Signature: recall(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass: macro-averaged recall across K classes.
# - Binary    : standard recall = TP / (TP + FN), using threshold 0.5 (or column 2 for N×2 preds).
# - Regression: undefined -> returns NA_real_.
# Set global option RECALL_MIN_OVERLAP <- 0.80 to control row-trim tolerance (default 0.80).
recall <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = TRUE) {
  dbg <- function(...) if (isTRUE(verbose)) cat(..., "\n")
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      x <- as.matrix(model.matrix(~ . - 1, data = x))
    } else if (is.matrix(x)) {
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1L)
    } else {
      x <- as.matrix(x)
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    }
    storage.mode(x) <- "double"; x
  }
  
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      if (isTRUE(verbose)) dbg("[recall] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      if (isTRUE(verbose)) dbg("[recall] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # ---------- coerce ----------
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[recall] Empty labels or predictions.")
  if (isTRUE(verbose)) dbg(sprintf("[recall] initial dims: labels=%d x %d | pred=%d x %d",
                                   nrow(lbl), ncol(lbl), nrow(pred), ncol(pred)))
  
  # ---------- row alignment (prefer by rownames, else tolerant trim) ----------
  align_by_rownames <- !is.null(rownames(lbl)) && !is.null(rownames(pred))
  if (align_by_rownames) {
    common_ids <- intersect(rownames(lbl), rownames(pred))
    if (length(common_ids) == 0L) stop("[recall] No overlapping rownames between labels and predictions.")
    lbl  <- lbl[common_ids, , drop = FALSE]
    pred <- pred[common_ids, , drop = FALSE]
    if (isTRUE(verbose)) dbg(sprintf("[recall] Aligned by rownames: %d rows", length(common_ids)))
  } else if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    min_overlap <- get0("RECALL_MIN_OVERLAP", ifnotfound = 0.80, inherits = TRUE)
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < min_overlap) {
      stop(sprintf("[recall] Row mismatch too large (labels=%d, pred=%d). Overlap=%.3f < %.2f.",
                   nrow(lbl), nrow(pred), ratio, min_overlap))
    }
    if (isTRUE(verbose)) dbg(sprintf("[recall] Trimmed to %d rows (overlap=%.3f >= %.2f)", n_common, ratio, min_overlap))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (isTRUE(verbose)) dbg(paste("[recall] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[recall] Multiclass mode requires predicted_output with K>1 columns.")
    
    # true classes
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
    } else {
      stop(sprintf("[recall] Multiclass labels have %d columns; predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    true_class[true_class < 1L] <- 1L
    true_class[true_class > K]  <- K
    
    # predicted classes
    if (ncol(pred) != K) stop("[recall] Multiclass predictions must be N×K probabilities.")
    pred_class <- max.col(pred, ties.method = "first")
    
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[recall] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    rec_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fn <- sum(pred_class != k & true_class == k)
      rec_per_class[k] <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    }
    return(as.numeric(mean(rec_per_class)))
    
  } else if (identical(mode, "binary")) {
    # probabilities for positive class
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[recall] Unexpected binary prediction shape: %d columns.", ncol(pred)))
    }
    
    # true labels -> 0/1
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # [neg,pos] -> 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[recall] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[recall] Unexpected binary label shape: %d columns.", ncol(lbl)))
    }
    
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[recall] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    tp <- sum(y_pred == 1L & y_true == 1L)
    fn <- sum(y_pred == 0L & y_true == 1L)
    rec <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    return(round(as.numeric(rec), 8))
    
  } else if (identical(mode, "regression")) {
    if (isTRUE(verbose)) dbg("[recall] Regression mode: recall undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[recall] Unhandled mode.")
}

# -------------------------
# F1 Score (mode-aware: "binary" | "multiclass" | "regression")
# Signature: f1_score(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass: macro-averaged F1 across K classes.
# - Binary    : standard F1 from thresholded predictions (0.5 on p_pos).
# - Regression: undefined -> returns NA_real_.
# Set global option F1_MIN_OVERLAP <- 0.80 to control row-trim tolerance (default 0.80).
f1_score <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = TRUE) {
  dbg <- function(...) if (isTRUE(verbose)) cat(..., "\n")
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      x <- as.matrix(model.matrix(~ . - 1, data = x))
    } else if (is.matrix(x)) {
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1L)
    } else {
      x <- as.matrix(x)
      if (!is.numeric(x)) x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
    }
    storage.mode(x) <- "double"; x
  }
  
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      if (isTRUE(verbose)) dbg("[f1] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      if (isTRUE(verbose)) dbg("[f1] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # ---------- coerce ----------
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[f1] Empty labels or predictions.")
  if (isTRUE(verbose)) dbg(sprintf("[f1] initial dims: labels=%d x %d | pred=%d x %d",
                                   nrow(lbl), ncol(lbl), nrow(pred), ncol(pred)))
  
  # ---------- row alignment (prefer by rownames, else tolerant trim) ----------
  align_by_rownames <- !is.null(rownames(lbl)) && !is.null(rownames(pred))
  if (align_by_rownames) {
    common_ids <- intersect(rownames(lbl), rownames(pred))
    if (length(common_ids) == 0L) stop("[f1] No overlapping rownames between labels and predictions.")
    lbl  <- lbl[common_ids, , drop = FALSE]
    pred <- pred[common_ids, , drop = FALSE]
    if (isTRUE(verbose)) dbg(sprintf("[f1] Aligned by rownames: %d rows", length(common_ids)))
  } else if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    min_overlap <- get0("F1_MIN_OVERLAP", ifnotfound = 0.80, inherits = TRUE)
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < min_overlap) {
      stop(sprintf("[f1] Row mismatch too large (labels=%d, pred=%d). Overlap=%.3f < %.2f.",
                   nrow(lbl), nrow(pred), ratio, min_overlap))
    }
    if (isTRUE(verbose)) dbg(sprintf("[f1] Trimmed to %d rows (overlap=%.3f >= %.2f)", n_common, ratio, min_overlap))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (isTRUE(verbose)) dbg(paste("[f1] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[f1] Multiclass mode requires predicted_output with K>1 columns.")
    
    # true classes
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
    } else {
      stop(sprintf("[f1] Multiclass labels have %d columns; predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    true_class[true_class < 1L] <- 1L
    true_class[true_class > K]  <- K
    
    # predicted classes
    if (ncol(pred) != K) stop("[f1] Multiclass predictions must be N×K probabilities.")
    pred_class <- max.col(pred, ties.method = "first")
    
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[f1] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    # macro F1
    f1_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fp <- sum(pred_class == k & true_class != k)
      fn <- sum(pred_class != k & true_class == k)
      p  <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
      r  <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
      f1_per_class[k] <- if ((p + r) == 0) 0 else 2 * p * r / (p + r)
    }
    return(as.numeric(mean(f1_per_class)))
    
  } else if (identical(mode, "binary")) {
    # probabilities for positive class
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[f1] Unexpected binary prediction shape: %d columns.", ncol(pred)))
    }
    
    # true labels -> 0/1
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # [neg,pos] -> 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[f1] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[f1] Unexpected binary label shape: %d columns.", ncol(lbl)))
    }
    
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (isTRUE(verbose)) dbg(sprintf("[f1] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    tp <- sum(y_pred == 1L & y_true == 1L)
    fp <- sum(y_pred == 1L & y_true == 0L)
    fn <- sum(y_pred == 0L & y_true == 1L)
    
    p  <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    r  <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    f1 <- if ((p + r) == 0) 0 else 2 * p * r / (p + r)
    return(round(as.numeric(f1), 8))
    
  } else if (identical(mode, "regression")) {
    if (isTRUE(verbose)) dbg("[f1] Regression mode: F1 undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[f1] Unhandled mode.")
}

confusion_matrix <- function(SONN, labels, CLASSIFICATION_MODE, predicted_output,
                             threshold = NULL, verbose = FALSE) {
  # only for binary
  if (!identical(CLASSIFICATION_MODE, "binary")) {
    if (verbose) message("[CONFUSION_MATRIX] Only valid for binary classification.")
    return(NULL)
  }
  
  # resolve threshold robustly: arg > SONN$threshold > 0.5
  thr <- if (!is.null(threshold)) {
    as.numeric(threshold)[1]
  } else if (!is.null(SONN$threshold)) {
    as.numeric(SONN$threshold)[1]
  } else {
    0.5
  }
  
  # normalize shapes/types
  if (!is.matrix(labels)) labels <- matrix(as.numeric(labels), ncol = 1)
  if (!is.matrix(predicted_output)) predicted_output <- matrix(as.numeric(predicted_output), ncol = 1)
  
  preds   <- as.integer(predicted_output >= thr)
  actuals <- as.integer(labels)
  
  TP <- sum(preds == 1 & actuals == 1, na.rm = TRUE)
  FP <- sum(preds == 1 & actuals == 0, na.rm = TRUE)
  TN <- sum(preds == 0 & actuals == 0, na.rm = TRUE)
  FN <- sum(preds == 0 & actuals == 1, na.rm = TRUE)
  
  if (verbose) message(sprintf("[CONFUSION_MATRIX] thr=%.3f  TP=%d FP=%d TN=%d FN=%d", thr, TP, FP, TN, FN))
  
  list(TP = TP, FP = FP, TN = TN, FN = FN)
}


# =========================
# Global threshold helpers
# =========================

desonn_set_threshold <- function(value) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value <= 0 || value >= 1) {
    stop("[desonn_set_threshold] value must be a single numeric in (0,1).")
  }
  .GlobalEnv$DESONN_THRESHOLDS <- list(binary = as.numeric(value))
  invisible(TRUE)
}

desonn_get_threshold <- function(default = NA_real_) {
  x <- get0("DESONN_THRESHOLDS", envir = .GlobalEnv, inherits = FALSE)
  if (is.null(x) || is.null(x$binary)) return(default)
  as.numeric(x$binary)
}

desonn_clear_threshold <- function() {
  if (exists("DESONN_THRESHOLDS", envir = .GlobalEnv, inherits = FALSE)) {
    rm("DESONN_THRESHOLDS", envir = .GlobalEnv)
  }
  invisible(TRUE)
}

# =========================
# accuracy_precision_recall_f1_tuned (revised)
# =========================
accuracy_precision_recall_f1_tuned <- function(
    SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output,
    metric_for_tuning = c("accuracy","f1","precision","recall",
                          "macro_f1","macro_precision","macro_recall"),
    threshold_grid = seq(0.05, 0.95, by = 0.01),
    verbose = FALSE
) {
  dbg <- function(...) if (isTRUE(verbose)) cat(..., "\n")
  metric_for_tuning <- match.arg(metric_for_tuning)
  
  # --- helpers ---
  to_num_mat <- function(x) {
    if (is.data.frame(x)) x <- as.matrix(x)
    if (is.list(x) && length(x) == 1L) x <- x[[1L]]
    if (!is.matrix(x)) x <- as.matrix(x)
    storage.mode(x) <- "double"
    x
  }
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  infer_mode <- function(L, P) if (max(ncol(L), ncol(P)) > 1L) "multiclass" else "binary"
  
  sanitize_grid_simple <- function(g) {
    if (is.function(g)) {
      attempt <- try(g(), silent = TRUE)
      g <- if (!inherits(attempt, "try-error")) attempt else seq(0.05, 0.95, by = 0.01)
    }
    if (is.language(g) || is.symbol(g)) {
      attempt <- try(eval(g, parent.frame()), silent = TRUE)
      g <- if (!inherits(attempt, "try-error")) attempt else seq(0.05, 0.95, by = 0.01)
    }
    if (is.list(g)) g <- unlist(g, use.names = FALSE)
    g <- suppressWarnings(as.numeric(g))
    g <- g[is.finite(g)]
    g <- sort(unique(g))
    g <- g[g > 0 & g < 1]
    if (!length(g)) g <- seq(0.05, 0.95, by = 0.01)
    g
  }
  
  # --- coerce & trim ---
  L <- to_num_mat(labels)
  P <- to_num_mat(predicted_output)
  n <- min(nrow(L), nrow(P))
  if (n == 0L) stop("[accuracy_precision_recall_f1_tuned] empty inputs after trim.")
  L <- L[seq_len(n), , drop = FALSE]
  P <- P[seq_len(n), , drop = FALSE]
  
  # --- resolve mode ---
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(L, P)
  thr_grid <- sanitize_grid_simple(threshold_grid)
  
  # =========================
  # ===== REGRESSION ========
  # =========================
  if (identical(mode, "regression")) {
    return(list(
      accuracy  = NA_real_,
      precision = NA_real_,
      recall    = NA_real_,
      f1        = NA_real_,
      confusion_matrix = NULL,
      details = list(
        best_threshold = NA_real_,
        y_pred_class   = NA,
        grid_used      = thr_grid,
        tuned_by       = "n/a"
      )
    ))
  }
  
  # ============================
  # ======== BINARY PATH ========
  # ============================
  if (identical(mode, "binary")) {
    # --- Labels (expecting 0/1) ---
    y_true <- if (ncol(L) == 1L) {
      v <- as.numeric(L[,1])
      if (all(v %in% c(0,1))) as.integer(v) else as.integer(v >= 0.5)
    } else {
      as.integer(max.col(L, ties.method = "first") - 1L)
    }
    
    # --- Probabilities (1-col sigmoid output) ---
    if (ncol(P) != 1L) {
      stop("[accuracy_precision_recall_f1_tuned] Binary mode expects 1-column probabilities (sigmoid). Got ", ncol(P), " columns.")
    }
    p_pos <- as.numeric(P[,1])
    
    # --- Global threshold logic ---
    global_th <- desonn_get_threshold(default = NA_real_)
    tuned_now <- FALSE
    
    if (is.na(global_th)) {
      dbg("[accuracy_precision_recall_f1_tuned] No global threshold set. Tuning on provided data (use validation here).")
      metrics <- c("accuracy","f1","precision","recall")
      if (!(metric_for_tuning %in% metrics)) {
        metric_for_tuning <- switch(metric_for_tuning,
                                    macro_f1="f1",
                                    macro_precision="precision",
                                    macro_recall="recall",
                                    metric_for_tuning)
      }
      
      best <- list(th = 0.5, score = -Inf)
      for (th in thr_grid) {
        preds <- as.integer(p_pos >= th)
        TP <- sum(preds == 1L & y_true == 1L)
        FP <- sum(preds == 1L & y_true == 0L)
        TN <- sum(preds == 0L & y_true == 0L)
        FN <- sum(preds == 0L & y_true == 1L)
        
        acc <- (TP + TN) / length(y_true)
        pre <- if ((TP + FP) > 0) TP / (TP + FP) else 0
        rec <- if ((TP + FN) > 0) TP / (TP + FN) else 0
        f1  <- if ((pre + rec) > 0) 2 * pre * rec / (pre + rec) else 0
        
        score <- switch(metric_for_tuning,
                        accuracy = acc, f1 = f1, precision = pre, recall = rec)
        if (score > best$score || (abs(score - best$score) < .Machine$double.eps^0.5 &&
                                   abs(th - 0.5) < abs(best$th - 0.5))) {
          best <- list(th=th, score=score)
        }
      }
      desonn_set_threshold(best$th)
      global_th <- best$th
      tuned_now <- TRUE
      dbg(sprintf("[accuracy_precision_recall_f1_tuned] Tuned global threshold = %.4f (metric=%s)",
                  global_th, metric_for_tuning))
    } else {
      dbg(sprintf("[accuracy_precision_recall_f1_tuned] Using existing global threshold = %.4f.", global_th))
    }
    
    # --- Apply chosen threshold (tuned or stored) ---
    cm <- confusion_matrix(
      SONN = SONN,
      labels = matrix(y_true, ncol = 1),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1),
      threshold = global_th,
      verbose = FALSE
    )
    
    TP <- cm$TP; FP <- cm$FP; TN <- cm$TN; FN <- cm$FN
    
    total <- length(y_true)
    acc <- (TP + TN) / total
    pre <- if ((TP + FP) > 0) TP / (TP + FP) else 0
    rec <- if ((TP + FN) > 0) TP / (TP + FN) else 0
    f1  <- if ((pre + rec) > 0) 2 * pre * rec / (pre + rec) else 0
    
    y_pred_class <- as.integer(p_pos >= global_th)
    
    return(list(
      accuracy  = as.numeric(acc),
      precision = as.numeric(pre),
      recall    = as.numeric(rec),
      f1        = as.numeric(f1),
      confusion_matrix = cm,
      details   = list(
        best_threshold = as.numeric(global_th),
        y_pred_class   = y_pred_class,
        grid_used      = thr_grid,
        tuned_by       = if (tuned_now) metric_for_tuning else "applied-global"
      )
    ))
  }
  
  # ===============================
  # ===== MULTICLASS (argmax) =====
  # ===============================
  Kp <- ncol(P); Kl <- ncol(L)
  if (Kp > 1L && Kl > 1L && Kp != Kl) {
    K <- min(Kp, Kl)
    Pk <- P[, seq_len(K), drop = FALSE]
    Lk <- L[, seq_len(K), drop = FALSE]
  } else {
    Pk <- if (Kp > 1L) P else NULL
    Lk <- if (Kl > 1L) L else NULL
  }
  
  true_ids <- if (!is.null(Lk)) {
    max.col(Lk, ties.method = "first")
  } else {
    K <- max(2L, Kp)
    v <- as.integer(round(L[,1]))
    if (length(v) && min(v, na.rm = TRUE) == 0L) v <- v + 1L
    v[v < 1L] <- 1L; v[v > K] <- K
    v
  }
  pred_ids <- if (!is.null(Pk)) max.col(Pk, ties.method = "first") else rep(1L, length(true_ids))
  
  acc <- mean(pred_ids == true_ids, na.rm = TRUE)
  
  # macro precision/recall/F1
  K <- max(true_ids, pred_ids, na.rm = TRUE)
  macro_prec <- macro_rec <- macro_f1 <- numeric(K)
  for (k in seq_len(K)) {
    TPk <- sum(pred_ids == k & true_ids == k)
    FPk <- sum(pred_ids == k & true_ids != k)
    FNk <- sum(pred_ids != k & true_ids == k)
    prec <- if ((TPk + FPk) > 0) TPk / (TPk + FPk) else 0
    rec  <- if ((TPk + FNk) > 0) TPk / (TPk + FNk) else 0
    f1   <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0
    macro_prec[k] <- prec; macro_rec[k] <- rec; macro_f1[k] <- f1
  }
  
  list(
    accuracy  = round(as.numeric(acc), 8),
    precision = mean(macro_prec),
    recall    = mean(macro_rec),
    f1        = mean(macro_f1),
    confusion_matrix = NULL,   # only binary returns list(TP,FP,TN,FN)
    details   = list(
      best_threshold = NA_real_,
      y_pred_class   = pred_ids,
      grid_used      = thr_grid,
      tuned_by       = "argmax-macro"
    )
  )
}





# Mean Absolute Error (silent) — uses CLASSIFICATION_MODE for safe shape handling
# Signature: MAE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MAE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      mm <- model.matrix(~ . - 1, data = x); x <- as.matrix(mm)
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1L)
    } else {
      x <- as.matrix(x)
    }
    storage.mode(x) <- "double"
    x
  }
  one_hot <- function(idx, K) {
    idx <- as.integer(idx)
    idx[is.na(idx)] <- 1L
    idx[idx < 1L] <- 1L
    idx[idx > K]  <- K
    M <- matrix(0, nrow = length(idx), ncol = K)
    if (length(idx)) M[cbind(seq_along(idx), idx)] <- 1
    storage.mode(M) <- "double"
    M
  }
  
  # Coerce to numeric matrices
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  
  # Row alignment (no recycling)
  n_common <- min(nrow(lbl), nrow(pred))
  if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # Mode-based shape harmonization
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(lbl[, 1])))
      if (length(u) && min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) {
        lbl <- lbl + 1
      }
      lbl <- one_hot(lbl[, 1], K)
    } else if (ncol(lbl) != K) {
      return(NA_real_)
    }
    
  } else if (identical(mode, "binary")) {
    # pred: N×1 (p_pos) OR N×2 ([p_neg,p_pos]); lbl: N×1 (0/1 or two values) OR N×2 one-hot
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      lbl <- matrix(lbl[, 2], ncol = 1L)
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      v <- as.integer(round(lbl[, 1])); v[is.na(v)] <- 0L; v[v != 0L] <- 1L
      lbl <- cbind(1 - v, v); storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      uniq <- sort(unique(as.integer(lbl[, 1])))
      if (!all(uniq %in% c(0L, 1L))) {
        if (length(uniq) == 2L) {
          lbl[, 1] <- ifelse(lbl[, 1] == uniq[2], 1, 0)
        } else {
          return(NA_real_)
        }
      }
    } else if (!(ncol(pred) == 2L && ncol(lbl) == 2L)) {
      return(NA_real_)
    }
    
  } else if (!identical(mode, "regression")) {
    return(NA_real_)
  }
  
  # Final column check
  if (!identical(mode, "regression") && ncol(lbl) != ncol(pred)) return(NA_real_)
  
  # Only in regression allow replicate/truncate to conform columns
  if (identical(mode, "regression") && ncol(lbl) != ncol(pred)) {
    total_needed <- nrow(lbl) * ncol(lbl)
    if (ncol(pred) < ncol(lbl)) {
      vec <- rep(pred, length.out = total_needed)
      pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    } else {
      pred <- pred[, 1:ncol(lbl), drop = FALSE]
      pred <- matrix(pred, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    }
  }
  
  # NA handling
  cc <- stats::complete.cases(cbind(lbl, pred))
  if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]
  pred <- pred[cc, , drop = FALSE]
  if (!nrow(lbl)) return(NA_real_)
  
  # Compute MAE
  mae <- mean(abs(pred - lbl))
  mae
}

# NDCG (Normalized Discounted Cumulative Gain)
ndcg <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose = FALSE) {
  # Convert to data frame if needed
  Rdata <- data.frame(Rdata)
  
  # Define relevance scores: assume binary relevance (1 for "relevant", 0 otherwise)
  relevance_scores <- ifelse(labels == "relevant", 1, 0)
  
  # Create a data frame to pair predicted outputs with relevance
  df <- data.frame(
    prediction = predicted_output,
    relevance  = relevance_scores
  )
  
  # Sort by prediction score descending (as if ranked)
  df_sorted <- df[order(-df$prediction), ]
  
  # Compute DCG
  gains <- (2^df_sorted$relevance - 1) / log2(1 + seq_along(df_sorted$relevance))
  dcg <- sum(gains)
  
  # Compute ideal DCG (perfect ranking)
  ideal_relevance <- sort(df$relevance, decreasing = TRUE)
  ideal_gains <- (2^ideal_relevance - 1) / log2(1 + seq_along(ideal_relevance))
  idcg <- sum(ideal_gains)
  
  # Avoid division by zero
  ndcg_value <- if (idcg == 0) 0 else dcg / idcg
  
  return(ndcg_value)
}


# -------------------------
# Custom Relative Error (binned), mode-aware (silent)
# Signature:
#   custom_relative_error_binned(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose=FALSE)
# Behavior:
#   - regression (default): elementwise |pred - label| / |label| with epsilon guard, then binned.
#   - binary: per-row error = |1 - p_true|, where p_true = p(pos) if y=1; else 1 - p(pos) if y=0.
#   - multiclass: per-row error = |1 - p_true| where p_true is probability of the true class.
# Returns: named LIST of bin means.
custom_relative_error_binned <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE) {
  # --- helpers ---
  to_num_mat <- function(x) {
    if (is.data.frame(x)) x <- as.matrix(x)
    else if (is.atomic(x) && !is.matrix(x)) x <- matrix(x, ncol = 1L)
    else x <- as.matrix(x)
    storage.mode(x) <- "double"
    x
  }
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  infer_mode <- function(L, P) {
    if (max(ncol(L), ncol(P)) > 1L) "multiclass" else "regression"
  }
  
  # bin edges: 0%..100% -> 0..1
  bins <- c(0, 0.05, 0.10, 0.50, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  brks <- bins / 100
  bin_names <- paste0("rel_", bins[-length(bins)], "_", bins[-1], "pct")
  
  # empty-output helper
  empty_out <- function(val = NA_real_) {
    x <- rep(val, length(bin_names))
    names(x) <- bin_names
    as.list(x)
  }
  
  # --- coerce & trim to common rows ---
  L <- to_num_mat(labels)
  P <- to_num_mat(predicted_output)
  n <- min(nrow(L), nrow(P))
  if (n == 0L) return(empty_out())
  
  L <- L[seq_len(n), , drop = FALSE]
  P <- P[seq_len(n), , drop = FALSE]
  
  # --- resolve mode ---
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(L, P)
  
  # --- compute vector of per-sample errors ---
  if (identical(mode, "binary")) {
    # labels -> y_true in {0,1}
    if (ncol(L) == 2L) {
      y_true <- as.integer(L[, 2] >= L[, 1])
    } else {
      v <- as.numeric(L[, 1])
      if (all(v %in% c(0, 1))) y_true <- as.integer(v) else y_true <- as.integer(v >= 0.5)
    }
    # preds -> p(pos)
    p_pos <- if (ncol(P) >= 2L) as.numeric(P[, 2]) else as.numeric(P[, 1])
    p_pos[!is.finite(p_pos)] <- NA_real_
    
    # p_true = p(pos) if y=1, else 1 - p(pos)
    p_true <- ifelse(y_true == 1L, p_pos, 1 - p_pos)
    err <- abs(1 - p_true)          # in [0,1]
    vals <- err[is.finite(err)]
    
  } else if (identical(mode, "multiclass")) {
    # Determine K (prefer predictions)
    Kp <- ncol(P); Kl <- ncol(L)
    if (Kp > 1L && Kl > 1L && Kp != Kl) {
      K  <- min(Kp, Kl)
      Pk <- P[, seq_len(K), drop = FALSE]
      Lk <- L[, seq_len(K), drop = FALSE]
    } else {
      Pk <- if (Kp > 1L) P else NULL
      Lk <- if (Kl > 1L) L else NULL
    }
    
    # true class id (1..K)
    if (!is.null(Lk)) {
      true_ids <- max.col(Lk, ties.method = "first")
      K <- ncol(Lk)
    } else {
      K <- max(2L, Kp)  # at least 2
      v <- as.integer(round(L[, 1]))
      if (length(v) && min(v, na.rm = TRUE) == 0L) v <- v + 1L
      v[v < 1L] <- 1L; v[v > K] <- K
      true_ids <- v
    }
    
    # probability assigned to true class
    if (!is.null(Pk)) {
      idx <- cbind(seq_len(nrow(Pk)), true_ids)
      p_true <- as.numeric(Pk[idx])
    } else {
      # degenerate: no multi-prob predictions; pretend all prob on class 1
      p_true <- as.numeric(true_ids == 1L)
    }
    p_true[!is.finite(p_true)] <- NA_real_
    
    err  <- abs(1 - p_true)         # in [0,1]
    vals <- err[is.finite(err)]
    
  } else {  # regression (default)
    # Align shapes (replicate/truncate columns)
    if (ncol(L) != ncol(P)) {
      if (ncol(P) < ncol(L)) {
        total <- nrow(L) * ncol(L)
        Pm <- matrix(rep(P, length.out = total), nrow = nrow(L), ncol = ncol(L), byrow = FALSE)
      } else {
        Pm <- matrix(P[, 1:ncol(L)], nrow = nrow(L), ncol = ncol(L), byrow = FALSE)
      }
    } else {
      Pm <- P
    }
    error_prediction <- Pm - L
    denom <- abs(L)
    denom[denom == 0 | !is.finite(denom)] <- .Machine$double.eps
    percentage_difference <- as.numeric(abs(error_prediction) / denom)
    vals <- percentage_difference[is.finite(percentage_difference)]
  }
  
  # --- binning ---
  mean_precisions <- numeric(length(brks) - 1)
  if (length(vals) > 0L) {
    idx <- findInterval(vals, brks, rightmost.closed = TRUE, all.inside = TRUE)
    for (j in seq_along(mean_precisions)) {
      v <- vals[idx == j]
      mean_precisions[j] <- if (length(v)) mean(v) else 0
    }
  } else {
    mean_precisions[] <- 0
  }
  
  names(mean_precisions) <- bin_names
  list(custom_relative_error_binned = mean_precisions)
}


# Diversity (Shannon entropy) — robust
# Signature kept: diversity(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose = FALSE)
diversity <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose = FALSE) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "multiclass"))
  eps  <- .Machine$double.eps
  
  # Not meaningful for regression
  if (identical(mode, "regression")) return(NA_real_)
  
  # Helper: safe entropy of a probability vector
  H_vec <- function(p) {
    p <- as.numeric(p)
    if (!length(p)) return(NA_real_)
    p[!is.finite(p) | p < 0] <- 0
    s <- sum(p)
    if (!is.finite(s) || s <= 0) return(NA_real_)
    p <- p / s
    p <- pmax(p, eps)
    -sum(p * log2(p))
  }
  
  # If matrix: treat rows as class distributions (preferred for softmax outputs)
  if (is.matrix(predicted_output)) {
    P <- predicted_output
    storage.mode(P) <- "double"
    P[!is.finite(P)] <- 0
    
    if (ncol(P) == 1L) {
      # Binary prob column → expand to [p_neg, p_pos]
      p1 <- pmin(pmax(P[, 1], 0), 1)       # clamp to [0,1]
      P  <- cbind(1 - p1, p1)
    }
    
    # Row-wise normalization to probabilities
    row_sums <- rowSums(P)
    bad <- !is.finite(row_sums) | row_sums <= 0
    if (any(bad)) {
      # fallback to uniform for bad rows
      P[bad, ] <- 1 / ncol(P)
      row_sums <- rowSums(P)
    }
    P <- P / row_sums
    
    # Per-row Shannon entropy, then aggregate (mean)
    P <- pmax(P, eps)
    H_row <- -rowSums(P * log2(P))
    H <- mean(H_row, na.rm = TRUE)
    if (!is.finite(H)) return(NA_real_)
    return(H)
  }
  
  # If vector:
  v <- as.vector(predicted_output)
  
  # (a) If it looks like class labels (character/factor or integer-ish):
  if (is.factor(v) || is.character(v)) {
    p <- as.numeric(table(v))
    return(H_vec(p))
  }
  
  # (b) Numeric vector → treat as scores/prob mass; normalize safely
  v <- as.numeric(v)
  v[!is.finite(v) | v < 0] <- 0
  if (!length(v) || sum(v) <= 0) return(NA_real_)
  return(H_vec(v))
}



# Root Mean Squared Error (silent + mode-aware)
# Signature: RMSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
RMSE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  # helpers
  `%||%` <- function(x, y) if (is.null(x)) y else x
  to_num_mat <- function(x, name = "obj") {
    if (is.data.frame(x)) {
      x <- as.matrix(model.matrix(~ . - 1, data = x))
    } else if (is.factor(x) || is.character(x)) {
      x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
    } else if (is.atomic(x) && !is.matrix(x)) {
      x <- matrix(x, ncol = 1L)
    } else x <- as.matrix(x)
    storage.mode(x) <- "double"; x
  }
  one_hot <- function(idx, K) {
    idx <- as.integer(idx)
    idx[is.na(idx)] <- 1L
    idx[idx < 1L] <- 1L; idx[idx > K] <- K
    M <- matrix(0, nrow = length(idx), ncol = K)
    if (length(idx)) M[cbind(seq_along(idx), idx)] <- 1
    storage.mode(M) <- "double"; M
  }
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # argument recovery (handle swapped positional args) — silent
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # coerce to numeric matrices
  lbl  <- to_num_mat(labels, "labels")
  pred <- to_num_mat(predicted_output, "predicted_output")
  
  # row alignment (NO recycling)
  n_common <- min(nrow(lbl), nrow(pred))
  if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # resolve/infer mode
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  
  # mode-based shape harmonization
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(lbl[,1])))
      if (length(u) && min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) {
        lbl <- lbl + 1
      }
      lbl <- one_hot(lbl[,1], K)
    } else if (ncol(lbl) != K) {
      return(NA_real_)
    }
  } else if (identical(mode, "binary")) {
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      lbl <- matrix(lbl[,2], ncol = 1L)  # collapse one-hot to 0/1
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      v <- as.integer(round(lbl[,1])); v[is.na(v)] <- 0L; v[v != 0L] <- 1L
      lbl <- cbind(1 - v, v); storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          lbl[,1] <- ifelse(lbl[,1] == max(u), 1, 0)
        } else {
          return(NA_real_)
        }
      }
    } else if (!(ncol(pred) == 2L && ncol(lbl) == 2L)) {
      return(NA_real_)
    }
  } else if (!identical(mode, "regression")) {
    return(NA_real_)
  }
  
  # final column check
  if (!identical(mode, "regression") && ncol(lbl) != ncol(pred)) return(NA_real_)
  if (identical(mode, "regression") && ncol(lbl) != ncol(pred)) {
    total_needed <- nrow(lbl) * ncol(lbl)
    if (ncol(pred) < ncol(lbl)) {
      vec <- rep(pred, length.out = total_needed)
      pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    } else {
      pred <- pred[, 1:ncol(lbl), drop = FALSE]
      pred <- matrix(pred, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    }
  }
  
  # NA handling
  cc <- stats::complete.cases(cbind(lbl, pred))
  if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]
  pred <- pred[cc, , drop = FALSE]
  if (!nrow(lbl)) return(NA_real_)
  
  # compute RMSE
  rmse <- sqrt(mean((pred - lbl)^2))
  rmse
}


# Serendipity
serendipity <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose = FALSE) {
  # Calculate the average number of times each prediction is made
  prediction_counts <- table(predicted_output)
  
  # Calculate the inverse of the prediction counts
  inverse_prediction_counts <- 1 / prediction_counts
  
  # Return the average inverse prediction count
  mean(inverse_prediction_counts, na.rm = TRUE)
}
