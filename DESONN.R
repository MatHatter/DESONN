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
      self$input_size <- self$process_input_size(input_size)
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
      
      clip_weights <- function(W, limit = .08) {
        return(pmin(pmax(W, -limit), limit))
      }
      
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
        
        return(clip_weights(W, limit = .08))
      }
      
      # Initialize first hidden layer
      weights[[1]] <- init_weight(input_size, hidden_sizes[1], method, custom_scale)
      biases[[1]] <- matrix(rnorm(hidden_sizes[1], mean = 0, sd = 0.05), ncol = 1)  # slightly higher bias init
      
      # Intermediate hidden layers
      for (layer in 2:length(hidden_sizes)) {
        weights[[layer]] <- init_weight(hidden_sizes[layer - 1], hidden_sizes[layer], method, custom_scale)
        biases[[layer]] <- matrix(rnorm(hidden_sizes[layer], mean = 0, sd = 0.05), ncol = 1)
      }
      
      # Output layer
      last_hidden_size <- hidden_sizes[[length(hidden_sizes)]]
      weights[[length(hidden_sizes) + 1]] <- init_weight(last_hidden_size, output_size, method, custom_scale)
      biases[[length(hidden_sizes) + 1]] <- matrix(rnorm(output_size, mean = 0, sd = 0.05), ncol = 1)
      
      self$weights <- weights
      self$biases <- biases
      
      return(list(weights = weights, biases = biases))
    },
    
    
    process_input_size = function(input_size) {
      # If the input is a numeric vector
      if (is.numeric(input_size)) {
        # If the vector has two elements, multiply them
        if (length(input_size) == 2) {
          return(input_size[1] * input_size[2])
          # If the vector has one element, return the element
        } else if (length(input_size) == 1) {
          return(input_size[1])
        }
        # If the input is a single number, return the number
      } else if (is.numeric(input_size) && length(input_size) == 1) {
        # Check if the number is an integer (no decimals)
        if (input_size == floor(input_size)) {
          return(input_size)
        } else {
          stop("Invalid input. Please enter an integer.")
        }
      } else {
        stop("Invalid input. Please enter a numeric vector or a single number.")
      }
    },
    # Method to store weights for a specific layer beyond the first one #don't really need this, this is just to view the weights for my own due diligence #sonn$store_weights_specific(2)
    store_weights_specific = function(layer) {
      weight_name <- paste0("weights", layer)
      bias_name <- paste0("biases", layer)
      weights_stored_specific <<- as.matrix(self[[weight_name]])
      biases_stored_specific <<- as.matrix(self[[bias_name]])
    },# Method to load weights for the first layer
    load_weights = function(new_weights) {
      if (predict_models) {
        self$weights <- unlist(new_weights)
      } else {
        self$weights <- new_weights
      }
    },# Method to load biases for the first layer
    load_biases = function(new_biases) {
      if (predict_models) {
        self$biases <- unlist(new_biases)
      } else {
        self$biases <- new_biases
      }
    },# Method to load weights for all layers
    load_all_weights = function(weights_list) {
      for (i in 1:length(weights_list)) {
        
        self$weights[[i]] <- unlist(weights_list[[i]])  # Assuming weights_list[[i]] is a list of matrices
      }
    },# Method to load biases for all layers
    load_all_biases = function(biases_list) {
      for (i in 1:length(biases_list)) {
        self$biases[[i]] <- unlist(biases_list[[i]])  # Assuming biases_list[[i]] is a list of vectors
      }
    },
    # Dropout function with no default rate
    dropout = function(x, rate) {
      # If no rate is provided, return x as is
      if (is.null(rate)) {
        return(x)
      }
      
      # Create a dropout mask
      mask <- runif(length(x)) > rate
      
      # Apply the mask and scale the activations
      x <- x * mask / (1 - rate)
      
      return(x)
    },# Method to perform self-organization
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
    # Method to perform learning
    learn = function(Rdata, labels, lr, activation_functions_learn, dropout_rates_learn, sample_weights) {
      print("------------------------learn-begin-------------------------------------------------")
      start_time <- Sys.time()
      
      if (!is.numeric(labels) || !is.matrix(labels) || ncol(labels) != 1) {
        labels <- matrix(as.numeric(labels), ncol = 1)
      }
      
      pos_weight <- 2
      neg_weight <- 1
      
      if (is.null(sample_weights)) {
        sample_weights <- ifelse(labels == 1, pos_weight, neg_weight)
      }
      sample_weights <- matrix(sample_weights, nrow = nrow(labels), ncol = 1)
      
      self$dropout_rates_learn <- dropout_rates_learn
      
      if (!is.matrix(labels)) labels <- as.matrix(labels)
      if (length(dim(labels)) == 2 && nrow(labels) == ncol(labels)) {
        labels <- matrix(diag(labels), ncol = 1)
      }
      labels <- matrix(as.numeric(labels), ncol = 1)
      
      predicted_output_learn <- NULL
      error_learn <- NULL
      dim_hidden_layers_learn <- list()
      predicted_output_learn_hidden <- NULL
      bias_gradients <- list()
      grads_matrix <- list()
      weight_gradients <- list()
      
      errors <- list()  # $$$$$$$$$$$$$$$$ Added for per-layer error tracking
      
      if (self$ML_NN) {
        hidden_outputs <- vector("list", self$num_layers)
        activation_derivatives <- vector("list", self$num_layers)
        dim_hidden_layers_learn <- vector("list", self$num_layers)
        input_matrix <- as.matrix(Rdata)
        
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
          cat(sprintf("[Debug] Layer %d : class = %s | dim = %s\n", layer, class(input_data), paste(dim(input_data), collapse = " x ")))
          
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
          
          cat(sprintf("[Debug] Layer %d : Z summary BEFORE clipping:\n", layer))
          print(summary(as.vector(Z)))
          
          activation_function <- activation_functions_learn[[layer]]
          activation_name <- if (!is.null(activation_function)) attr(activation_function, "name") else "none"
          cat(sprintf("[Debug] Layer %d : Activation Function = %s\n", layer, activation_name))
          
          clip_limit <- switch(activation_name,
                               "sigmoid" = 80,
                               "tanh" = 10,
                               "softmax" = 15,
                               "relu" = 100,
                               "leaky_relu" = 200,
                               90)
          
          Z_max <- max(Z)
          Z_min <- min(Z)
          if (Z_max > clip_limit || Z_min < -clip_limit) {
            cat(sprintf("[Debug] Layer %d : Z clipped. Pre-clip range: [%.2f, %.2f] | Clip limit: ±%g\n", layer, Z_min, Z_max, clip_limit))
          }
          Z <- pmin(pmax(Z, -clip_limit), clip_limit)
          
          cat(sprintf("[Debug] Layer %d : Z summary AFTER clipping:\n", layer))
          print(summary(as.vector(Z)))
          
          hidden_output <- if (!is.null(activation_function)) activation_function(Z) else Z
          hidden_outputs[[layer]] <- hidden_output
          
          if (is.list(self$dropout_rates_learn) &&
              length(self$dropout_rates_learn) >= layer &&
              !is.null(self$dropout_rates_learn[[layer]]) &&
              self$dropout_rates_learn[[layer]] > 0 &&
              self$dropout_rates_learn[[layer]] < 1) {
            
            cat(sprintf("\n[Debug] Layer %d : Hidden output BEFORE dropout (sample):\n", layer))
            print(head(hidden_output, 3))
            
            hidden_output <- self$dropout(hidden_output, self$dropout_rates_learn[[layer]])
            hidden_outputs[[layer]] <- hidden_output
            
            hidden_outputs[[layer]] <- self$dropout(hidden_output, self$dropout_rates_learn[[layer]])
            
            cat(sprintf("[Debug] Layer %d : Hidden output AFTER  dropout (sample):\n", layer))
            print(head(hidden_output, 3))
          }
          
          cat(sprintf("[Debug] Layer %d : predicted_output_learn dim = %d x %d\n", layer, nrow(hidden_output), ncol(hidden_output)))
          
          if (activation_name != "none") {
            derivative_name <- paste0(activation_name, "_derivative")
            cat(sprintf("[Debug] Layer %d : Derivative function = %s\n", layer, derivative_name))
            if (!exists(derivative_name, mode = "function")) {
              stop(paste("Layer", layer, ": Activation derivative function", derivative_name, "does not exist."))
            }
            deriv <- get(derivative_name, mode = "function")(Z)
          } else {
            deriv <- matrix(1, nrow = nrow(Z), ncol = ncol(Z))
          }
          
          activation_derivatives[[layer]] <- deriv
          hidden_outputs[[layer]] <- hidden_output
          dim_hidden_layers_learn[[layer]] <- dim(hidden_output)
        }
        
        predicted_output_learn <- hidden_outputs[[self$num_layers]]
        predicted_output_learn_hidden <- hidden_outputs
        error_learn <- (predicted_output_learn - labels) * sample_weights
        
        error_backprop <- error_learn
        for (layer in self$num_layers:1) {
          delta <- error_backprop * activation_derivatives[[layer]]
          bias_gradients[[layer]] <- matrix(colMeans(delta), nrow = 1)
          errors[[layer]] <- delta  # $$$$$$$$$$$$$$$$ Added
          
          input_for_grad <- if (layer == 1) input_matrix else hidden_outputs[[layer - 1]]
          grads_matrix[[layer]] <- t(input_for_grad) %*% delta
          
          if (layer > 1) {
            weights_t <- t(as.matrix(self$weights[[layer]]))
            error_backprop <- delta %*% weights_t
          }
        }
      } else {
        cat("Single Layer Learning Phase\n")
        weights_matrix <- as.matrix(self$weights)
        bias_vec <- as.numeric(unlist(self$biases))
        input_rows <- nrow(Rdata)
        weights_rows <- nrow(weights_matrix)
        weights_cols <- ncol(weights_matrix)
        
        if (ncol(Rdata) != weights_rows) {
          stop(sprintf("SL NN: input cols (%d) do not match weights rows (%d)", ncol(Rdata), weights_rows))
        }
        
        if (length(bias_vec) == 1) {
          bias_matrix <- matrix(bias_vec, nrow = input_rows, ncol = weights_cols)
        } else if (length(bias_vec) == weights_cols) {
          bias_matrix <- matrix(rep(bias_vec, each = input_rows), nrow = input_rows)
        } else if (length(bias_vec) == input_rows * weights_cols) {
          bias_matrix <- matrix(bias_vec, nrow = input_rows)
        } else {
          stop(sprintf("SL NN: invalid bias shape: length = %d", length(bias_vec)))
        }
        
        if (!is.list(self$dropout_rates_learn)) {
          self$dropout_rates_learn <- list(self$dropout_rates_learn)
        }
        rate <- self$dropout_rates_learn[[1]]
        if (!is.null(rate) && rate > 0 && rate < 1) {
          cat(sprintf("[Debug] SL NN : Applying dropout with rate = %.2f\n", rate))
          Rdata <- self$dropout(Rdata, rate)
        }
        
        Z <- Rdata %*% weights_matrix + bias_matrix
        
        cat(sprintf("[Debug]: Z summary BEFORE clipping.\n"))
        print(summary(as.vector(Z)))
        
        if (is.function(activation_functions_learn)) {
          activation_function <- activation_functions_learn
        } else {
          if (!is.list(activation_functions_learn)) {
            activation_functions_learn <- list(activation_functions_learn)
          }
          activation_function <- activation_functions_learn[[1]]
        }
        
        activation_name <- attr(activation_function, "name")
        clip_limit <- switch(activation_name,
                             "sigmoid" = 50,
                             "tanh" = 10,
                             "softmax" = 15,
                             "relu" = 100,
                             "leaky_relu" = 200,
                             90)
        
        Z_max <- max(Z)
        Z_min <- min(Z)
        if (Z_max > clip_limit || Z_min < -clip_limit) {
          cat(sprintf("[Debug] SL NN : Z clipped. Pre-clip range: [%.2f, %.2f] | Clip limit: ±%g\n", Z_min, Z_max, clip_limit))
        }
        Z <- pmin(pmax(Z, -clip_limit), clip_limit)
        
        cat("[Debug] SL NN : Z summary AFTER clipping:\n")
        print(summary(as.vector(Z)))
        
        predicted_output_learn <- if (!is.null(activation_function)) activation_function(Z) else Z
        error_learn <- (predicted_output_learn - labels) * sample_weights
        
        dim_hidden_layers_learn[[1]] <- dim(predicted_output_learn)
        
        activation_deriv <- if (!is.null(activation_function)) {
          deriv_fn <- paste0(attr(activation_function, "name"), "_derivative")
          if (!exists(deriv_fn)) matrix(1, nrow = nrow(Z), ncol = ncol(Z)) else get(deriv_fn)(Z)
        } else matrix(1, nrow = nrow(Z), ncol = ncol(Z))
        
        delta <- error_learn * activation_deriv
        bias_gradients[[1]] <- matrix(colMeans(delta), nrow = 1)
        grads_matrix[[1]] <- t(Rdata) %*% delta
        errors[[1]] <- delta  # $$$$$$$$$$$$$$$$ Added for SL NN
      }
      
      learn_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      print("------------------------learn-end-------------------------------------------------")
      
      return(list(
        learn_output = predicted_output_learn,
        learn_time = learn_time,
        error = error_learn,
        dim_hidden_layers = dim_hidden_layers_learn,
        hidden_outputs = predicted_output_learn_hidden,
        grads_matrix = grads_matrix,
        bias_gradients = bias_gradients,
        errors = errors  # $$$$$$$$$$$$$$$$ Added
      ))
    }
    
    
    
    
    ,# Method to perform prediction
    predict = function(Rdata, weights = NULL, biases = NULL, activation_functions = NULL) {
      # If weights/biases are missing → fall back to internal state (stateful mode)
      if (is.null(weights)) {
        if (!is.null(self$weights)) {
          weights <- self$weights
        } else {
          stop("predict(): weights not provided and self$weights is NULL.")
        }
      }
      if (is.null(biases)) {
        if (!is.null(self$biases)) {
          biases <- self$biases
        } else {
          stop("predict(): biases not provided and self$biases is NULL.")
        }
      }
      
      # Ensure lists
      if (!is.list(weights)) weights <- list(weights)
      if (!is.list(biases)) biases <- list(biases)
      if (!is.null(activation_functions) && !is.list(activation_functions)) {
        activation_functions <- list(activation_functions)
      }
      
      start_time <- Sys.time()
      output <- as.matrix(Rdata)
      num_layers <- length(weights)
      
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
          bias_mat <- matrix(rep(b, length.out = n_units), nrow = n_samples,
                             ncol = n_units, byrow = TRUE)
        }
        
        # Linear transformation
        output <- output %*% w + bias_mat
        
        # Apply activation if provided
        if (!is.null(activation_functions) &&
            length(activation_functions) >= layer &&
            is.function(activation_functions[[layer]])) {
          output <- activation_functions[[layer]](output)
        }
      }
      
      end_time <- Sys.time()
      prediction_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      return(list(predicted_output = output, prediction_time = prediction_time))
    },# Method for training the SONN with L2 regularization
    train_with_l2_regularization = function(Rdata, labels, lr, num_epochs, model_iter_num, update_weights, update_biases, ensemble_number, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type, sample_weights, X_validation, y_validation, threshold_function, ML_NN, train, verbose) {
      
      # Initialize learning rate scheduler
      # lr_scheduler <- function(epoch, initial_lr = lr) {
      #     return(initial_lr * (1 + 0.01 * (epoch - 1)))
      # }
      
      # lr_scheduler <- function(epoch, initial_lr = lr) {
      #     return(initial_lr / (1 + 0.1 * sqrt(epoch)))
      # }
      
      start_time <- Sys.time()
      # 
      
      
      # Convert labels to a column matrix if it is a vector
      # if (is.vector(labels)) {
      #     labels <- matrix(labels, ncol = 1)
      # }
      
      if(never_ran_flag == TRUE || !hyperparameter_grid_setup) {
        losses <- numeric(num_epochs)
        epoch_in_list <- num_epochs
      } else if (never_ran_flag == FALSE && hyperparameter_grid_setup  && !use_loaded_weights) {
        epoch_in_list <- num_epochs[[model_iter_num]] #if error here, you need to run all over again.
        losses <- vector("list", length = epoch_in_list)
      } else {
        epoch_in_list <- optimal_epochs[[model_iter_num]]
        losses <- vector("list", length = epoch_in_list)
      }
      
      # Initialize variables to store the previous weights and biases
      prev_weights <- NULL
      prev_biases <- NULL
      
      
      # Initialize optimizer parameters for weights and biases
      optimizer_params_weights <- vector("list", self$num_layers)
      optimizer_params_biases <- vector("list", self$num_layers)
      
      best_val_acc <- -Inf
      best_val_epoch <- NA
      best_train_acc <- -Inf
      best_epoch <- NA
      val_accuracy_log <- c()
      train_accuracy_log <- c()
      loss_log <- c()
      learning_rate_log <- c()
      # Add tracking logs
      val_loss_log <- c()
      train_loss_log <- c()
      mean_output_log <- c()
      sd_output_log <- c()
      max_weight_log <- c()
      
      if (train) {
        for (epoch in 1:epoch_in_list) {
          
          
          
          
          lr <- lr_scheduler(epoch)
          
          cat("Epoch:", epoch, "| Learning Rate:", lr, "\n")
          

          
          num_epochs_check <<- num_epochs
          
          # start_time <- Sys.time()
          
          
          
          # Train model and get predictions
          learn_result <- self$learn(
            Rdata = Rdata,
            labels = labels,
            lr = lr,
            activation_functions_learn = activation_functions,
            dropout_rates_learn = dropout_rates,
            sample_weights = sample_weights
          )
          
          # =========================
          # BLOCK A — Accuracy & Saturation
          #   Place this RIGHT AFTER: learn_result <- self$learn(...)
          #   (assumes `fname` was already built at top of epoch loop)
          # =========================
          
          # --- per-epoch logs ---
          probs_train        <- learn_result$learn_output
          binary_preds_train <- ifelse(probs_train >= 0.5, 1, 0)
          train_accuracy     <- mean(binary_preds_train == labels, na.rm = TRUE)
          train_accuracy_log <- c(train_accuracy_log, train_accuracy)
          
          if (!exists("best_train_acc") || (!is.na(train_accuracy) && train_accuracy > best_train_acc)) {
            best_train_acc   <- train_accuracy
            best_epoch_train <- epoch
          }
          
          # correct layer for saturation
          predicted_output <- if (isTRUE(self$ML_NN)) learn_result$hidden_outputs[[self$num_layers]] else learn_result$learn_output
          mean_output      <- mean(predicted_output)
          sd_output        <- sd(predicted_output)
          mean_output_log  <- c(mean_output_log, mean_output)
          sd_output_log    <- c(sd_output_log, sd_output)
          
          # optional loss
          train_loss     <- mean((predicted_output - labels)^2, na.rm = TRUE)
          train_loss_log <- c(train_loss_log, train_loss)
          
          # Build the naming closure via utils::make_fname_prefix (NULL-safe)
          fname <- make_fname_prefix(
            do_ensemble     = isTRUE(get0("do_ensemble", ifnotfound = FALSE)),
            num_networks    = get0("num_networks", ifnotfound = NULL),
            total_models    = if (!is.null(self$ensemble)) length(self$ensemble) else get0("num_networks", ifnotfound = NULL),
            ensemble_number = if (exists("self", inherits = TRUE)) self$ensemble_number else get0("ensemble_number", ifnotfound = NULL),
            model_index     = get0("model_iter_num", ifnotfound = NULL),
            who             = "SONN"
          )
          
          # sanity probe (should print one non-empty filename)
          cat("[fname probe] -> ", fname("probe.png"), "\n")
          
          
          # ensure output dir
          if (!dir.exists("plots")) dir.create("plots", recursive = TRUE, showWarnings = FALSE)
          
          # title
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
          
          # DF (no MaxWeight here) with padding to align lengths
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
          
          # 1) Accuracy & Loss
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
    
          # 2) Output Saturation
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
          
          
          
          
          predicted_output_train_reg <- learn_result
          predicted_output_train_reg_prediction_time <- learn_result$learn_time
          
          # predicted_output_train_reg_prediction_time <- Sys.time() - start_time
          
          # Extract predicted output and error
          if (self$ML_NN) {
            predicted_output <- predicted_output_train_reg$hidden_outputs[[self$num_layers]]
          } else {
            predicted_output <- predicted_output_train_reg$learn_output
          }
          
          error <- predicted_output_train_reg$error
          errors <- predicted_output_train_reg$errors
          bias_gradients <- predicted_output_train_reg$bias_gradients  # <---- EXTRACT BIAS GRADIENTS
          weight_gradients <- predicted_output_train_reg$grads_matrix 
          dim_hidden_layers <- predicted_output_train_reg$dim_hidden_layers
          
          # Extract hidden outputs only for multi-layer networks
          if (self$ML_NN) {
            hidden_outputs <- predicted_output_train_reg$hidden_outputs
            
            # Diagnostic print
            cat("DEBUG: Type of hidden_outputs =", typeof(hidden_outputs), "\n")
            cat("DEBUG: Class of hidden_outputs =", class(hidden_outputs), "\n")
            
            if (is.list(hidden_outputs)) {
              # Check if each element is a matrix or can be coerced
              dim_hidden_layers <- lapply(hidden_outputs, function(layer_output) {
                if (is.matrix(layer_output)) {
                  return(dim(layer_output))
                } else if (is.vector(layer_output)) {
                  return(length(layer_output))
                } else if (is.null(layer_output)) {
                  return(NULL)
                } else {
                  return(dim(as.matrix(layer_output)))
                }
              })
            } else if (is.matrix(hidden_outputs)) {
              dim_hidden_layers <- list(dim(hidden_outputs))
            } else if (is.vector(hidden_outputs)) {
              dim_hidden_layers <- list(length(hidden_outputs))
            } else {
              dim_hidden_layers <- list(NULL)
            }
          } else {
            dim_hidden_layers <- predicted_output_train_reg$hidden_outputs #This = NULL, I could've set to NULL, but I instead passed the NULL from predict() through this variable.
          }
          
          # --- Regularization Loss Computation Only ---
          if (self$ML_NN) {
            reg_loss_total <- 0
            
            if (!is.null(reg_type)) {
              for (layer in 1:self$num_layers) {
                weights_layer <- self$weights[[layer]]
                
                if (reg_type == "L1") {
                  reg_loss <- self$lambda * sum(abs(weights_layer), na.rm = TRUE)
                  
                } else if (reg_type == "L2") {
                  reg_loss <- self$lambda * sum(weights_layer^2, na.rm = TRUE)
                  
                } else if (reg_type == "L1_L2") {
                  l1_ratio <- 0.5
                  reg_loss <- self$lambda * (
                    l1_ratio * sum(abs(weights_layer), na.rm = TRUE) +
                      (1 - l1_ratio) * sum(weights_layer^2, na.rm = TRUE)
                  )
                  
                } else if (reg_type == "Group_Lasso") {
                  if (is.null(self$groups) || is.null(self$groups[[layer]])) {
                    self$groups <- if (is.null(self$groups)) vector("list", self$num_layers) else self$groups
                    self$groups[[layer]] <- list(1:ncol(weights_layer))
                  }
                  reg_loss <- self$lambda * sum(sapply(self$groups[[layer]], function(group) {
                    sqrt(sum(weights_layer[, group]^2, na.rm = TRUE))
                  }))
                }
                else if (reg_type == "Max_Norm") {
                  max_norm <- 1.0
                  norm_weight <- sqrt(sum(weights_layer^2, na.rm = TRUE))
                  reg_loss <- self$lambda * ifelse(norm_weight > max_norm, 1, 0)
                  
                }
                else if (reg_type == "Orthogonality") {
                  WtW <- t(weights_layer) %*% weights_layer
                  I <- diag(ncol(WtW))
                  reg_loss <- self$lambda * sum((WtW - I)^2)
                }
                
                else if (reg_type == "Sparse_Bayesian") {
                  stop("Sparse Bayesian Learning is not implemented in this code.")
                  
                } else {
                  print("Invalid regularization type. Choose 'L1', 'L2', 'L1_L2', 'Group_Lasso', 'Max_Norm', or 'Sparse_Bayesian'. However, NULL is acceptable.")
                  reg_loss <- 0
                }
                
                reg_loss_total <- reg_loss_total + reg_loss
              }
            } else {
              reg_loss_total <- 0
            }
          }
          
          else {
            reg_loss_total <- 0
            
            if (!is.null(reg_type)) {
              # Ensure self$weights is treated as a list
              weights_list <- if (is.list(self$weights)) self$weights else list(self$weights)
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
                weights_layer <- self$weights
                
                if (is.null(self$groups)) {
                  self$groups <- list(1:ncol(weights_layer))  # Treat entire layer as one group
                }
                
                reg_loss_total <- self$lambda * sum(sapply(self$groups, function(group) {
                  sqrt(sum(weights_layer[, group]^2, na.rm = TRUE))
                }))
              }
              else if (reg_type == "Max_Norm") {
                max_norm <- 1.0
                norm_weight <- sqrt(sum(weights_layer^2, na.rm = TRUE))
                reg_loss_total <- self$lambda * ifelse(norm_weight > max_norm, 1, 0)
                
              } 
              else if (reg_type == "Orthogonality") {
                WtW <- t(weights_layer) %*% weights_layer
                I <- diag(ncol(WtW))
                reg_loss_total <- self$lambda * sum((WtW - I)^2)
                
              } else if (reg_type == "Sparse_Bayesian") {
                stop("Sparse Bayesian Learning is not implemented in this code.")
                
              } else {
                print("Invalid regularization type. Choose 'L1', 'L2', 'L1_L2', 'Group_Lasso', 'Max_Norm', or 'Sparse_Bayesian'. However, NULL is acceptable.")
                reg_loss_total <- 0
              }
            } else {
              reg_loss_total <- 0
            }
          }
          
          
          # Record the loss for this epoch
          # losses[[epoch]] <- mean(error_last_layer^2) + reg_loss_total
          predictions <- if (self$ML_NN) hidden_outputs[[self$num_layers]] else predicted_output_train_reg$learn_output
          
          # Ensure predictions match label dimensions before loss calculation
          if (!all(dim(predictions) == dim(labels))) {
            predictions <- matrix(rep(predictions, length.out = length(labels)),
                                  nrow = nrow(labels), ncol = ncol(labels))
          }
          
          losses[[epoch]] <- loss_function(
            predictions = predictions,
            labels = labels,
            reg_loss_total = reg_loss_total,
            loss_type = loss_type
          )
          
          
          # Initialize records and optimizer parameters if ML_NN is TRUE
          if (self$ML_NN) {
            weights_record <- vector("list", self$num_layers)
            biases_record <- vector("list", self$num_layers)
          }
          
          
          
          
          # Update weights and biases code removed for chatgpt to process
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
                  
                  
                  cat(">> Gradients for layer", layer, "\n")
                  cat("grads_matrix dim:\n")
                  print(dim(grads_matrix))
                  
                  cat("grads_matrix summary:\n")
                  print(summary(as.vector(grads_matrix)))
                  
                  
                  
                  
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
                        target = "weights"
                      )
                      
                      cat(">> Updated weights summary (post-Adam): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      
                      self$weights[[layer]] <- updated_optimizer$updated_weights_or_biases
                      
                      
                      
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
                        target = "weights"
                      )
                      
                      cat(">> Updated weights summary (post-RMSprop): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$weights[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    else if (optimizer == "sgd") {
                      
                      cat("DEBUG: Is optimizer_params_weights available? ", exists("optimizer_params_weights"), "\n")
                      cat("DEBUG: Type: ", typeof(optimizer_params_weights), "\n")
                      cat("DEBUG: Length: ", length(optimizer_params_weights), "\n")
                      
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
                        target = "weights"
                      )
                      
                      
                      
                      
                      # $$$$$$$$$$$$ Diagnostic print
                      cat(">> Updated weights summary (post-SGD): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      # $$$$$$$$$$$$ Final update
                      self$weights[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    else if (optimizer == "sgd_momentum") {
                      cat("DEBUG: Is optimizer_params_weights available? ", exists("optimizer_params_weights"), "\n")
                      cat("DEBUG: Type: ", typeof(optimizer_params_weights), "\n")
                      cat("DEBUG: Length: ", length(optimizer_params_weights), "\n")
                      
                      # $$$$$$$$$$$$ Apply SGD momentum update (weights)
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
                        target = "weights"
                      )
                      
                      # $$$$$$$$$$$$ Diagnostic print
                      cat(">> Updated weights summary (SGD momentum): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      # $$$$$$$$$$$$ Final update with optional clipping
                      updated_weights_matrix <- updated_optimizer$updated_weights_or_biases
                      clip_threshold <- 0.5
                      updated_weights_matrix <- pmin(pmax(updated_weights_matrix, -clip_threshold), clip_threshold)
                      
                      self$weights[[layer]] <- self$weights[[layer]] - updated_weights_matrix
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    else if (optimizer == "nag") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,  # Nag only uses momentum (beta1)
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "weights"
                      )
                      
                      cat(">> Updated weights summary (post-NAG): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$weights[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "ftrl") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer         = optimizer,
                        optimizer_params  = optimizer_params_weights,
                        grads_matrix      = grads_matrix,
                        lr                = lr,
                        beta1             = beta1,     # Optional, can be used for adaptive variants
                        beta2             = beta2,     # Optional, can be used for smoothing
                        alpha             = 0.1,       # FTRL specific
                        lambda1           = 0.01,
                        lambda2           = 0.01,
                        epsilon           = epsilon,
                        epoch             = epoch,
                        self              = self,
                        layer             = layer,
                        target            = "weights"
                      )
                      
                      cat(">> Updated weights summary (post-FTRL): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$weights[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "lamb") {
                      cat(">> Optimizer = lamb\n")
                      cat("Layer:", layer, "\n")
                      cat("grads_matrix dim:\n")
                      print(dim(grads_matrix))
                      
                      # Standardize grads_matrix to list of matrices
                      grads_input <- if (is.list(grads_matrix)) {
                        grads_matrix
                      } else if (is.null(dim(grads_matrix))) {
                        list(matrix(grads_matrix, nrow = 1, ncol = 1))
                      } else if (length(dim(grads_matrix)) == 1) {
                        list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1))
                      } else {
                        list(grads_matrix)
                      }
                      
                      # Call LAMB via apply_optimizer_update
                      updated_optimizer <- apply_optimizer_update(
                        optimizer         = optimizer,
                        optimizer_params  = optimizer_params_weights,
                        grads_matrix      = grads_matrix,
                        lr                = lr,
                        beta1             = beta1,
                        beta2             = beta2,
                        epsilon           = epsilon,
                        epoch             = epoch,
                        self              = self,
                        layer             = layer,
                        target            = "weights"
                      )
                      
                      # Logging
                      cat(">> Updated weights summary (post-LAMB): min =", 
                          min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      # Apply update
                      self$weights[[layer]] <- self$weights[[layer]] - updated_optimizer$updated_weights_or_biases
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "lookahead") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        layer = layer,
                        grads_matrix = weight_gradients[[layer]],
                        lr = lr,
                        beta1 = beta1,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        target = "weights"
                      )
                      
                      
                      
                      cat(">> Updated weights summary (post-lookahead): min =", 
                          min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$weights[[layer]] <- self$weights[[layer]] - updated_optimizer$updated_weights_or_biases
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "adagrad") {
                      cat(">> Optimizer = adagrad\n")
                      cat("Layer:", layer, "\n")
                      cat("grads_matrix dim:\n")
                      print(dim(grads_matrix))
                      
                      # Use apply_optimizer_update to match unified format
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix,
                        lr = lr,
                        beta1 = beta1,  # Not used in Adagrad, but kept for consistency
                        beta2 = beta2,  # Not used either
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "weights"
                      )
                      
                      # Extract updated weights
                      updated_weights <- updated_optimizer$updated_weights_or_biases
                      
                      # Print summary
                      cat(">> Updated weights summary (post-Adagrad): min =", min(updated_weights), 
                          ", mean =", mean(updated_weights), 
                          ", max =", max(updated_weights), "\n")
                      
                      # Apply weight update
                      self$weights[[layer]] <- updated_weights
                      
                      # Update internal optimizer state
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
                        target = "weights"
                      )
                      
                      cat(">> Updated weights summary (post-Adadelta): min =", 
                          min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$weights[[layer]] <- updated_optimizer$updated_weights_or_biases
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
                      target           = "weights"
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
                  
                  cat(">> Bias gradients for layer", layer, "\n")
                  cat("grads_matrix dim:\n")
                  print(dim(grads_matrix))
                  cat("grads_matrix summary:\n")
                  print(summary(as.vector(grads_matrix)))
                  
                  
                  # Apply optimizer update if optimizer is specified
                  if (!is.null(optimizer_params_biases[[layer]]) && !is.null(optimizer)) {
                    if (optimizer == "adam") {
                      
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = bias_gradients[[layer]],
                        lr = lr,
                        beta1 = beta1,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases"
                      )
                      
                      self$biases[[layer]] <- updated_optimizer$updated_weights_or_biases
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
                        target = "biases"
                      )
                      
                      cat(">> Updated biases summary (post-RMSprop): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$biases[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    else if (optimizer == "sgd") {
                      
                      
                      # $$$$$$$$$$$$ Fix: Apply SGD update (extra args passed for compatibility)
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
                        target = "biases"
                      )
                      
                      # $$$$$$$$$$$$ Diagnostic print
                      cat(">> Updated biases summary (post-SGD): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      # $$$$$$$$$$$$ Final updates
                      self$biases[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "sgd_momentum") {
                      
                      # $$$$$$$$$$$$ Fix: Apply SGD momentum update (structured like SGD block)
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
                        target = "biases"
                      )
                      
                      # $$$$$$$$$$$$ Diagnostic print
                      cat(">> Updated biases summary (SGD momentum): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      # $$$$$$$$$$$$ Final updates
                      self$biases[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    
                    
                    else if (optimizer == "nag") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = bias_gradients[[layer]],
                        lr = lr,
                        beta1 = beta1,  # Only beta1 needed for NAG
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases"
                      )
                      
                      cat(">> Updated biases summary (post-NAG): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$biases[[layer]] <- updated_optimizer$updated_weights_or_biases
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
                        target            = "biases"
                      )
                      
                      cat(">> Updated biases summary (post-FTRL): min =", min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$biases[[layer]] <- updated_optimizer$updated_weights_or_biases
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "lamb") {
                      cat(">> Optimizer = lamb (biases)\n")
                      cat("Layer:", layer, "\n")
                      cat("grads_matrix dim:\n")
                      print(dim(grads_matrix))
                      
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
                        target            = "biases"
                      )
                      
                      # Logging
                      cat(">> Updated biases summary (post-LAMB): min =", 
                          min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      # Update biases
                      self$biases[[layer]] <- self$biases[[layer]] - updated_optimizer$updated_weights_or_biases
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    else if (optimizer == "lookahead") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_weights,
                        layer = layer,
                        grads_matrix = weight_gradients[[layer]],
                        lr = lr,
                        beta1 = beta1,
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        target = "biases"
                      )
                      
                      
                      cat(">> Updated biases summary (post-lookahead): min =", 
                          min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$biases[[layer]] <- self$biases[[layer]] - updated_optimizer$updated_weights_or_biases
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    
                    
                    else if (optimizer == "adagrad") {
                      cat(">> Optimizer = adagrad (biases)\n")
                      cat("Layer:", layer, "\n")
                      cat("errors dim:\n")
                      print(dim(errors[[layer]]))
                      
                      # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                      # FIXED: Make sure grads_matrix is always a matrix, even if 1D input
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
                        target = "biases"
                      )
                      
                      # Extract updated bias vector
                      updated_biases <- updated_optimizer$updated_weights_or_biases
                      
                      cat(">> Updated biases summary (post-Adagrad): min =", min(updated_biases), 
                          ", mean =", mean(updated_biases), 
                          ", max =", max(updated_biases), "\n")
                      
                      # Apply to model
                      self$biases[[layer]] <- self$biases[[layer]] - updated_biases
                      
                      # Update optimizer state
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                    
                    
                    else if (optimizer == "adadelta") {
                      updated_optimizer <- apply_optimizer_update(
                        optimizer = optimizer,
                        optimizer_params = optimizer_params_biases,
                        grads_matrix = colSums(errors[[layer]]),
                        lr = lr,
                        beta1 = beta1,  # unused by Adadelta but passed for uniformity
                        beta2 = beta2,
                        epsilon = epsilon,
                        epoch = epoch,
                        self = self,
                        layer = layer,
                        target = "biases"
                      )
                      
                      cat(">> Updated biases summary (post-Adadelta): min =", 
                          min(updated_optimizer$updated_weights_or_biases), 
                          ", mean =", mean(updated_optimizer$updated_weights_or_biases), 
                          ", max =", max(updated_optimizer$updated_weights_or_biases), "\n")
                      
                      self$biases[[layer]] <- self$biases[[layer]] - updated_optimizer$updated_weights_or_biases
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
                        target           = "biases"
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
          
          if (self$ML_NN) {
            for (layer in 1:self$num_layers) {
              biases_record[[layer]] <- as.matrix(self$biases[[layer]])
            }
          } else {
            
            biases_record <- as.matrix(self$biases)
          }
          
          
          
          
          predicted_output_val <- self$predict(Rdata = X_validation,
                                               weights = weights_record,
                                               biases = biases_record,
                                               activation_functions = activation_functions)
          
          probs_val <- predicted_output_val$predicted_output
          
          # Debug: Check predicted_output
          cat("Debug: length(probs_val) =", length(probs_val), "\n")
          if (length(probs_val) == 0) {
            stop("predicted_output is empty — check your predict() output")
          }
          
          # Debug: Check labels alignment
          cat("Debug: length(y_validation) =", length(y_validation), "\n")
          if (length(probs_val) != length(y_validation)) {
            stop(paste("Length mismatch: probs =", length(probs_val), ", y_validation =", length(y_validation)))
          }
          
          # Binary thresholding
          binary_preds_val <- ifelse(probs_val >= 0.5, 1, 0)
          
          # Calculate validation accuracy
          val_accuracy <- mean(binary_preds_val == y_validation, na.rm = TRUE)
          
          # Initialize validation accuracy log if not already
          if (!exists("val_accuracy_log")) val_accuracy_log <- c()
          val_accuracy_log <- c(val_accuracy_log, val_accuracy)
          
          if (!exists("best_val_acc") || (!is.na(val_accuracy) && val_accuracy > best_val_acc)) {
            best_val_acc <- val_accuracy
            best_val_epoch <- epoch
            
            # Save best model
            if (self$ML_NN) {
              best_weights <- lapply(self$weights, function(w) as.matrix(w))
              best_biases  <- lapply(self$biases,  function(b) as.matrix(b))
            } else {
              best_weights <- as.matrix(self$weights)
              best_biases  <- as.matrix(self$biases)
            }
            
            
            assign("final_weights_record", best_weights, envir = .GlobalEnv) #not really necessary as we are returning best_weights_records now
            assign("final_biases_record", best_biases, envir = .GlobalEnv) #not really necessary as we are returning best_biases_records now
            
            assign("best_val_probs", probs_val, envir = .GlobalEnv)
            assign("best_val_labels", y_validation, envir = .GlobalEnv)
            
            cat("New best model saved at epoch", epoch, "| Val Acc:", round(100 * val_accuracy, 2), "%\n")
          }
          
          
          
          
          
        }            
        cat(sprintf("\nBest Training Accuracy: %.2f%% at Epoch %d\n", 100 * best_train_acc, best_epoch_train))
        
        
        cat("Best Epoch (validation accuracy):", best_val_epoch, "\n")
        cat("Best Validation Accuracy:", round(100 * best_val_acc, 2), "%\n")
        
      }else {predicted_output_train_reg_prediction_time <- NULL
      weights_record <- NULL
      biases_record <- NULL
      dim_hidden_layers <- NULL}
      
      # WIP: may need to load best_weights/best_biases from saved model files or checkpoints here before predicting
      if (!train) {
        predicted_output_train_reg <- self$predict(Rdata, weights = best_weights, biases = best_biases, activation_functions)
        
      }
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
      
      
      end_time <- Sys.time()
      
      # Calculate the training time
      training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
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
      
      
      
      # Return the predicted output
      return(list(predicted_output_l2 = predicted_output_train_reg, train_reg_prediction_time = predicted_output_train_reg_prediction_time, training_time = training_time, optimal_epoch = optimal_epoch, weights_record = weights_record, biases_record = biases_record, best_weights_record = best_weights, best_biases_record = best_biases, lossesatoptimalepoch = lossesatoptimalepoch, loss_increase_flag = loss_increase_flag, loss_status = loss_status, dim_hidden_layers = dim_hidden_layers, predicted_output_val = predicted_output_val, best_val_probs = best_val_probs, best_val_labels = best_val_labels))
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
    calculate_batch_size = function(data_size, max_batch_size = 512, min_batch_size = 32) {
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
    
    train = function(Rdata, labels, lr, lr_decay_rate, lr_decay_epoch, lr_min, ensemble_number, num_epochs, threshold, reg_type, numeric_columns, activation_functions_learn, activation_functions, dropout_rates_learn, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, batch_normalize_data, gamma_bn = NULL, beta_bn = NULL, epsilon_bn = 1e-5, momentum_bn = 0.9, is_training_bn = TRUE, shuffle_bn = FALSE, loss_type, sample_weights, X_validation, y_validation, validation_metrics, threshold_function, ML_NN, train, viewTables, verbose) {
      
      
      if (!is.null(numeric_columns) && !batch_normalize_data) {
        # Normalize the input data
        Rdata <- self$normalize_data(Rdata, numeric_columns)
        
        # Optionally normalize labels if they are continuous, otherwise skip
        # If labels are binary or categorical, normalization should not be applied
        if (is.numeric(labels) && !binary_flag) {
          labels <- self$normalize_data(labels, numeric_columns)
        }
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
              
              # Print diagnostic information for the current mini-batch
              print(paste("Batch Mean: ", batch_mean_bn))
              print(paste("Batch Variance: ", batch_var_bn))
              
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
              print(paste("After normalization - Mean: ", post_norm_mean_bn))
              print(paste("After normalization - Variance: ", post_norm_var_bn))
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
      all_learn_times                <- vector("list", length(self$ensemble))
      all_ensemble_name_model_name   <- vector("list", length(self$ensemble))
      all_model_iter_num             <- vector("list", length(self$ensemble))
      
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
          
          
          
          
          self$ensemble[[i]]$self_organize(Rdata, labels, lr)
          if (learnOnlyTrainingRun == FALSE) {
            # learn_results <- self$ensemble[[i]]$learn(Rdata, labels, lr, activation_functions_learn, dropout_rates_learn)
            predicted_outputAndTime <- suppressMessages(
              self$ensemble[[i]]$train_with_l2_regularization(
                Rdata, labels, lr, num_epochs, model_iter_num, update_weights, update_biases, ensemble_number, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type, sample_weights, X_validation, y_validation, threshold_function, ML_NN, train, verbose
              ))
            
            
            
            
            
            # -- Start: Store core model info --
            all_ensemble_name_model_name[[i]] <- ensemble_name_model_name
            
            all_model_iter_num[[i]] <- model_iter_num
            
            
            
            all_predicted_outputAndTime[[i]] <- list(
              predicted_output = if (validation_metrics) predicted_outputAndTime$predicted_output_val$predicted_output else predicted_outputAndTime$predicted_output_l2$learn_output,
              prediction_time = predicted_outputAndTime$predicted_output_l2$prediction_time,
              training_time = predicted_outputAndTime$training_time,
              optimal_epoch = predicted_outputAndTime$optimal_epoch,
              weights_record = predicted_outputAndTime$best_weights_record,
              biases_record = predicted_outputAndTime$best_biases_record,
              losses_at_optimal_epoch = predicted_outputAndTime$lossesatoptimalepoch
            )
            
            
            # Optional storage
            # my_optimal_epoch_out_vector[[i]] <<- predicted_outputAndTime$optimal_epoch
            # ----------------------------------
            
            # Continue if predictions are available
            if (!is.null(predicted_outputAndTime$predicted_output_l2$learn_output)) {
              
              all_predicted_outputs[[i]]       <-  if (validation_metrics) predicted_outputAndTime$predicted_output_val$predicted_output else predicted_outputAndTime$predicted_output_l2$learn_output
              all_prediction_times[[i]]        <- predicted_outputAndTime$train_reg_prediction_time
              all_errors[[i]]                  <- predicted_outputAndTime$predicted_output_l2$error
              all_hidden_outputs[[i]]          <- predicted_outputAndTime$predicted_output_l2$hidden_outputs
              all_layer_dims[[i]]              <- predicted_outputAndTime$predicted_output_l2$dim_hidden_layers
              all_best_val_probs[[i]]          <- predicted_outputAndTime$best_val_probs
              all_best_val_labels[[i]]         <- predicted_outputAndTime$best_val_labels
              all_weights[[i]]                 <- predicted_outputAndTime$best_weights_record
              all_biases[[i]]                  <- predicted_outputAndTime$best_biases_record
              all_activation_functions[[i]]    <- activation_functions_learn
              
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
            
            
            #look into later. must take in Rdata and labels too because we can compare metrics later
            
            # === Evaluate Prediction Diagnostics ===
            if (!is.null(X_validation) && !is.null(y_validation)) {
              eval_result <- EvaluatePredictionsReport(
                X_validation = X_validation,
                y_validation = y_validation,
                probs = all_predicted_outputs[[i]],
                predicted_outputAndTime = predicted_outputAndTime,
                threshold_function = threshold_function,
                best_val_probs = all_best_val_probs[[i]],
                best_val_labels = all_best_val_labels[[i]],
                verbose = verbose
                
              )
            }
            
            # -------------------------------
            # After EvaluatePredictionsReport
            # -------------------------------
            # Determine K from either labels or predicted outputs
            K <- max(
              ncol(as.matrix(y_validation)),
              ncol(as.matrix(all_predicted_outputs[[i]]))
            )
            
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
          X_validation                 = X_validation,
          y_validation                 = y_validation,
          validation_metrics           = validation_metrics,
          lr                           = lr,
          ensemble_number              = ensemble_number,
          model_iter_num               = model_iter_num,
          num_epochs                   = num_epochs,
          threshold                    = threshold,
          learn_results                = learn_results,
          predicted_output_list        = all_predicted_outputs,
          learn_time                   = NULL,
          prediction_time_list         = all_prediction_times,
          run_id                       = all_ensemble_name_model_name,
          all_predicted_outputAndTime  = all_predicted_outputAndTime,
          all_weights                  = all_weights,
          all_biases                   = all_biases,
          all_activation_functions     = all_activation_functions,
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
      
      return(list(predicted_output = predicted_outputAndTime$predicted_output_l2$predicted_output, threshold = threshold_used, thresholds = thresholds_used, accuracy = eval_result$accuracy, accuracy_percent = eval_result$accuracy_percent, metrics = if (!is.null(eval_result$metrics)) eval_result$metrics else NULL, misclassified = if (!is.null(eval_result$misclassified)) eval_result$misclassified else NULL, performance_relevance_data  = performance_relevance_data))
    }
    , # Method for updating performance and relevance metrics
    
    update_performance_and_relevance = function(Rdata, labels, X_validation, y_validation, validation_metrics, lr, ensemble_number, model_iter_num, num_epochs, threshold, learn_results, predicted_output_list, learn_time, prediction_time_list, run_id, all_predicted_outputAndTime, all_weights, all_biases, all_activation_functions, ML_NN, viewTables, verbose) {
      
      
      # Initialize lists to store performance and relevance metrics for each SONN
      performance_list <- list()
      relevance_list <- list()
      model_name_list <-  list()
      #████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
      
      # Calculate performance and relevance for each SONN in the ensemble
      if (never_ran_flag == TRUE) {
        
        for (i in 1:length(self$ensemble)) {
          
          
          
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

            
            performance_list[[i]] <- calculate_performance(
              SONN = self$ensemble[[i]],
              Rdata = Rdata,
              labels = labels,
              lr = lr,
              model_iter_num = i,
              num_epochs = num_epochs,
              threshold = threshold,
              learn_time = learn_time,
              predicted_output = single_predicted_output,
              prediction_time = single_prediction_time,
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
              labels = labels, 
              model_iter_num = i,
              predicted_output = single_predicted_output, 
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
          
          
          self$store_metadata(run_id = single_ensemble_name_model_name, ensemble_number, validation_metrics, model_iter_num = i, num_epochs, threshold = NULL, predicted_output = single_predicted_output, actual_values = y, all_weights = all_weights, all_biases = all_biases, performance_metric = performance_metric, relevance_metric = relevance_metric, predicted_outputAndTime = single_predicted_outputAndTime)
          
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
      process_performance <- function(metrics_data, model_names) {
        # Exclude noisy/derived metrics from plotting (keep interface the same)
        EXCLUDE_METRICS_REGEX <- paste(
          c(
            "^accuracy_tuned_accuracy_percent$",   # drop percent plots
            "^accuracy_percent$",                  # generic percent (if present)
            "^accuracy_tuned_y_pred_class\\d+$",   # per-class 0/1 series
            "^y_pred_class\\d+$",                  # safety w/out prefix
            "^accuracy_tuned_best_thresholds?$",   # thresholds (optional)
            "^best_thresholds?$",                  # thresholds (generic)
            
            # >>> NEW: drop grid-used meta fields <<<
            "^accuracy_tuned_grid_used",           # e.g., accuracy_tuned_grid_used1
            "^grid_used"                           # any plain grid_used*
          ),
          collapse = "|"
        )
        
        # Initialize variables to store results
        high_mean_df <- NULL
        low_mean_df  <- NULL
        
        # Loop through each model's metrics data
        for (i in seq_along(metrics_data)) {
          data <- metrics_data[[i]]
          model_name <- model_names[[i]]
          
          # Flatten nested lists and structure into named lists or atomic values
          flattened_data <- lapply(seq_along(data), function(j) {
            metric <- data[[j]]
            metric_name <- names(data)[j]
            
            if (is.list(metric)) {
              # Flatten the list into named individual metrics
              unlisted_metric <- unlist(metric)
              if (length(unlisted_metric) == length(names(unlisted_metric))) {
                set_names(as.list(unlisted_metric), paste0(metric_name, "_", names(unlisted_metric)))
              } else {
                set_names(as.list(unlisted_metric), paste0(metric_name, "_", seq_along(unlisted_metric)))
              }
            } else {
              # If not a list, keep it as is
              set_names(list(metric), metric_name)
            }
          }) %>%
            purrr::list_flatten()
          
          # Add the Model_Name
          flattened_data$Model_Name <- model_name
          
          # Convert to data frame
          df <- as.data.frame(flattened_data, stringsAsFactors = FALSE)
          
          # Ensure all columns are of atomic types
          df[] <- lapply(df, function(x) if (is.list(x)) unlist(x) else x)
          
          # Convert numeric columns to character to avoid type issues during pivoting
          df[] <- lapply(df, function(x) if (is.numeric(x)) as.character(x) else x)
          
          # Reshape the data into long format
          df <- df %>%
            pivot_longer(cols = -c(Model_Name), names_to = "Metric", values_to = "Value")
          
          # --- NEW: drop unwanted metrics by regex (minimal change) ---
          df <- df[!grepl(EXCLUDE_METRICS_REGEX, df$Metric), , drop = FALSE]
          
          # Attempt to convert "Value" column to numeric, handling warnings
          df$Value <- suppressWarnings(as.numeric(df$Value))
          
          # Filter out NA values
          df <- df[complete.cases(df), ]
          
          # Calculate mean for each metric
          mean_metrics <- df %>%
            group_by(Metric) %>%
            summarise(mean_value = mean(Value, na.rm = TRUE), .groups = "drop")
          
          # Identify metrics with means greater than 10
          high_mean_metrics <- mean_metrics %>%
            filter(mean_value > 10) %>%
            pull(Metric)
          
          # Filter dataframe for metrics with means greater than 10
          high_mean_df <- bind_rows(
            high_mean_df,
            df %>% filter(Metric %in% high_mean_metrics, !is.na(Value))
          )
          
          # Filter dataframe for metrics with means less than or equal to 10
          low_mean_df <- bind_rows(
            low_mean_df,
            df %>% filter(!Metric %in% high_mean_metrics, !is.na(Value))
          )
        }
        
        # Return the filtered data frames
        return(list(high_mean_df = high_mean_df, low_mean_df = low_mean_df))
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
      
      # -----------------------------------------------
      # Grouped metrics + printing policy (final)
      # -----------------------------------------------
      # Predeclare so they're always in scope
      perf_df <- relev_df <- NULL
      perf_group_summary <- relev_group_summary <- NULL
      group_perf <- group_relev <- NULL
      
      # Build per-model long DFs (works for 1+ models)
      perf_df  <- flatten_metrics_to_df(performance_list, run_id)
      relev_df <- flatten_metrics_to_df(relevance_list,     run_id)
      
      # --- Vanilla group summaries (across models) ---
      perf_group_summary  <- summarize_grouped(perf_df)
      relev_group_summary <- summarize_grouped(relev_df)
      
      # --- Optional notify user ---
      if (!isTRUE(verbose) && !isTRUE(viewTables)) {
        cat("\n[ℹ] Group summaries computed silently. 
      Set `verbose = TRUE` to print data frames, 
      or `viewTables = TRUE` to see tables.\n")
      }
      
      # Grouped metrics (run whenever you have ≥1 model)
      if (ensemble_number >= 1 && length(self$ensemble) > 1) {
        group_perf <- calculate_performance_grouped(
          SONN_list             = self$ensemble,
          Rdata                 = Rdata,
          labels                = labels,
          lr                    = lr,
          num_epochs            = num_epochs,
          threshold             = threshold,
          predicted_output_list = predicted_output_list,
          prediction_time_list  = prediction_time_list,
          ensemble_number       = ensemble_number,
          run_id                = run_id,
          ML_NN                 = ML_NN,
          verbose               = verbose,
          agg_method            = "mean",
          metric_mode           = "aggregate_predictions+rep_sonn"
        )
        
        group_relev <- calculate_relevance_grouped(
          SONN_list             = self$ensemble,
          Rdata                 = Rdata,
          labels                = labels,
          predicted_output_list = predicted_output_list,
          ensemble_number       = ensemble_number,
          run_id                = run_id,
          ML_NN                 = ML_NN,
          verbose               = verbose,
          agg_method            = "mean",
          metric_mode           = "aggregate_predictions+rep_sonn"
        )
      
        perf_df <<- perf_df
        relev_df <<- relev_df
      # ---------- Printing policy ----------
      # Tables (DF heads) print if EITHER verbose OR viewTables
      if (isTRUE(verbose) || isTRUE(viewTables)) {
        if (!is.null(perf_df))  { cat("\n--- performance_long_df (head) ---\n"); print(utils::head(perf_df, 12)) }
        if (!is.null(relev_df)) { cat("\n--- relevance_long_df (head) ---\n"); print(utils::head(relev_df, 12)) }
      }
      
      # Summaries + grouped metrics print ONLY when verbose = TRUE
      if (isTRUE(verbose)) {
        if (!is.null(perf_group_summary))  { cat("\n=== PERFORMANCE group summary ===\n"); print(perf_group_summary) }
        if (!is.null(relev_group_summary)) { cat("\n=== RELEVANCE group summary ===\n"); print(relev_group_summary) }
        if (!is.null(group_perf))  { cat("\n=== GROUPED PERFORMANCE metrics ===\n"); print(group_perf$metrics) }
        if (!is.null(group_relev)) { cat("\n=== GROUPED RELEVANCE metrics ===\n"); print(group_relev$metrics) }
      }
      
      }
      
      
      
      
      # Return the lists of plots
      return(list(performance_high_mean_plots = performance_high_mean_plots, performance_low_mean_plots = performance_low_mean_plots, relevance_high_mean_plots = relevance_high_mean_plots, relevance_low_mean_plots = relevance_low_mean_plots, performance_group_summary = perf_group_summary, relevance_group_summary = relev_group_summary, performance_long_df = perf_df, relevance_long_df = relev_df, performance_grouped = if (exists("group_perf")  && !is.null(group_perf))  group_perf$metrics  else NULL, relevance_grouped   = if (exists("group_relev") && !is.null(group_relev)) group_relev$metrics else NULL))
      
      
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
store_metadata = function(run_id, ensemble_number, validation_metrics, model_iter_num, num_epochs, threshold, all_weights, all_biases, predicted_output, actual_values, performance_metric, relevance_metric, predicted_outputAndTime) {
      
      
      # --- Conform predicted_output shape to match actual_values ---
      if (ncol(actual_values) != ncol(predicted_output)) {
        total_elements_needed <- nrow(actual_values) * ncol(actual_values)
        
        if (ncol(predicted_output) < ncol(actual_values)) {
          rep_factor <- ceiling(total_elements_needed / length(predicted_output))
          replicated_predicted_output <- rep(predicted_output, rep_factor)[1:total_elements_needed]
          predicted_output_matrix <- matrix(replicated_predicted_output,
                                            nrow = nrow(actual_values), ncol = ncol(actual_values), byrow = FALSE)
        } else {
          truncated_predicted_output <- predicted_output[, 1:ncol(actual_values)]
          predicted_output_matrix <- matrix(truncated_predicted_output,
                                            nrow = nrow(actual_values), ncol = ncol(actual_values), byrow = FALSE)
        }
      } else {
        predicted_output_matrix <- predicted_output
      }
  
      # --- Calculate error and summary statistics ---
      error_prediction <- (if (validation_metrics) y_validation else actual_values) - predicted_output_matrix
      differences     <- error_prediction
      summary_stats   <- summary(differences)
      boxplot_stats   <- boxplot.stats(differences)
      
      # --- Load plot_epochs from file (placeholder) ---
      # plot_epochs <- readRDS(paste0("plot_epochs_DESONN", ensemble_number, "SONN", model_iter_num, ".rds"))
      plot_epochs <- NULL
      
      # --- Prepare weight and bias records ---
      best_weights_records <- list()
      best_biases_records  <- list()
      for (i in 1:num_networks) {
        best_weights_records[[i]] <- all_weights[[i]]
        best_biases_records[[i]]  <- all_biases[[i]]
      }
      
      # --- Generate model_serial_num ---
      model_serial_num <- sprintf("%d.0.%d", as.integer(ensemble_number), as.integer(model_iter_num))
      
      # --- Build filename prefix / artifact names (NULL-safe via updated make_fname_prefix) ---
      fname <- make_fname_prefix(
        do_ensemble     = isTRUE(get0("do_ensemble", ifnotfound = FALSE)),
        num_networks    = get0("num_networks", ifnotfound = NULL),
        total_models    = get0("num_networks", ifnotfound = NULL),  # falls back to num_networks
        ensemble_number = ensemble_number,   # may be NULL; builder handles it
        model_index     = model_iter_num,    # may be NULL; builder handles it
        who             = "SONN"
      )
      
      artifact_names <- list(
        training_accuracy_loss_plot = fname("training_accuracy_loss_plot.png"),
        output_saturation_plot      = fname("output_saturation_plot.png"),
        max_weight_plot             = fname("max_weight_plot.png")
      )
      artifact_paths <- lapply(artifact_names, function(nm) file.path("plots", nm))
      
      # --- Create metadata list (unchanged + adds fname artifacts) ---
      metadata <- list(
        input_size = input_size,
        output_size = output_size,
        N = N,
        never_ran_flag = never_ran_flag,
        num_samples = num_samples,
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
        predicted_output = predicted_output,
        predicted_output_tail = tail(predicted_output),
        actual_values_tail = tail(actual_values),
        differences = tail(differences),
        summary_stats = summary_stats,
        boxplot_stats = boxplot_stats,
        X = X,
        y = y,
        lossesatoptimalepoch = predicted_outputAndTime$lossesatoptimalepoch,
        loss_increase_flag = predicted_outputAndTime$loss_increase_flag,
        performance_metric = performance_metric,
        relevance_metric = relevance_metric,
        plot_epochs = plot_epochs,
        best_weights_record = best_weights_records,
        best_biases_record = best_biases_records,
        
        # NEW: filename artifacts
        fname_artifact_names = artifact_names,  # e.g., "SONN_2_training_accuracy_loss_plot.png"
        fname_artifact_paths = artifact_paths,   # e.g., "plots/SONN_2_training_accuracy_loss_plot.png"
        validation_metrics = validation_metrics
      )
      
      metadata_main_ensemble <- list()
      metadata_temp_ensemble <- list()
      
      # --- Store metadata by ensemble type (unchanged) ---
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


initialize_optimizer_params <- function(optimizer, dim, lookahead_step, layer, verbose = FALSE) {
  if (length(dim) == 2 && is.null(layer)) {
    dim <- list(dim)
  }
  
  layer_dim <- dim[[1]]  # always using first dim block
  if (length(layer_dim) != 2 || any(is.na(layer_dim)) || any(layer_dim <= 0)) {
    cat("Invalid dimensions detected. Setting default dimension [1, 1].\n")
    layer_dim <- c(1, 1)
  }
  
  nrow_dim <- layer_dim[1]
  ncol_dim <- layer_dim[2]
  
  current_layer <- if (!is.null(layer)) layer else 1
  cat("Layer", current_layer, "dimensions: nrow =", nrow_dim, ", ncol =", ncol_dim, "\n")
  
  param_init <- matrix(rnorm(nrow_dim * ncol_dim), nrow = nrow_dim, ncol = ncol_dim)
  
  entry <- switch(optimizer,
                  adam = list(param = param_init, m = matrix(0, nrow = nrow_dim, ncol = ncol_dim), v = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  rmsprop = list(param = param_init, m = matrix(0, nrow = nrow_dim, ncol = ncol_dim), v = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  adadelta = list(param = param_init, m = matrix(0, nrow = nrow_dim, ncol = ncol_dim), v = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  adagrad = list(param = param_init, r = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  sgd = list(param = param_init, momentum = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  sgd_momentum = list(param = param_init, momentum = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  nag = list(param = param_init,
                             momentum = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                             fast_weights = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                             fast_biases = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  ftrl = list(param = param_init,
                              z = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                              n = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  lamb = list(param = param_init,
                              m = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                              v = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                              r = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  lookahead = list(param = param_init,
                                   m = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   v = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   r = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   slow_weights = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   lookahead_counter = 0,
                                   lookahead_step = lookahead_step),
                  stop(paste("Optimizer", optimizer, "not supported."))
  )
  
  if (verbose) {
    cat("Layer", current_layer, "optimizer tracking params initialized:\n")
    str(entry)
  }
  
  return(entry)
}




adam_update <- function(params, grads, lr, beta1, beta2, epsilon, t) {
  # Force grads into a list if it's not already
  if (!is.list(grads)) {
    grads <- list(as.matrix(grads))
  }
  
  # Initialize m and v as lists
  if (!is.list(params$m)) {
    params$m <- vector("list", length(grads))
  }
  if (!is.list(params$v)) {
    params$v <- vector("list", length(grads))
  }
  
  # # Learning rate scheduler
  # lr_schedule <- function(t, initial_lr) {
  #   decay_rate <- 0.01
  #   initial_lr * exp(-decay_rate * t)
  # }
  # lr <- lr_schedule(t, lr)
  
  # Update moment estimates
  for (i in seq_along(grads)) {
    grad_matrix <- grads[[i]]
    grad_dims <- dim(grad_matrix)
    if (is.null(grad_dims)) grad_matrix <- matrix(grad_matrix, nrow = 1)
    
    # Initialize if missing or shape mismatch
    if (is.null(params$m[[i]]) || !all(dim(params$m[[i]]) == dim(grad_matrix))) {
      params$m[[i]] <- matrix(0, nrow = nrow(grad_matrix), ncol = ncol(grad_matrix))
    }
    if (is.null(params$v[[i]]) || !all(dim(params$v[[i]]) == dim(grad_matrix))) {
      params$v[[i]] <- matrix(0, nrow = nrow(grad_matrix), ncol = ncol(grad_matrix))
    }
    
    # Update m and v
    params$m[[i]] <- beta1 * params$m[[i]] + (1 - beta1) * grad_matrix
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grad_matrix ^ 2)
  }
  
  # Bias correction
  m_hat <- lapply(params$m, function(m) m / (1 - beta1 ^ t))
  v_hat <- lapply(params$v, function(v) v / (1 - beta2 ^ t))
  
  # Compute updates
  weights_update <- Map(function(m, v) lr * m / (sqrt(v) + epsilon), m_hat, v_hat)
  
  return(list(
    m = params$m,
    v = params$v,
    weights_update = weights_update,
    biases_update = weights_update  # identical in single-layer mode
  ))
}

rmsprop_update <- function(params, grads, lr, beta2 = 0.999, epsilon = 1e-8) {
  # Force grads into a list of matrices
  if (!is.list(grads)) {
    grads <- list(as.matrix(grads))
  }
  
  # Initialize v as list if not already
  if (!is.list(params$v)) {
    params$v <- vector("list", length(grads))
  }
  
  updates <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_matrix <- grads[[i]]
    
    # Ensure it's a matrix (handle scalar/1D vectors)
    grad_matrix <- if (is.null(dim(grad_matrix))) matrix(grad_matrix, nrow = 1) else as.matrix(grad_matrix)
    
    # Initialize v if missing or shape mismatch
    if (is.null(params$v[[i]]) || !all(dim(params$v[[i]]) == dim(grad_matrix))) {
      params$v[[i]] <- matrix(0, nrow = nrow(grad_matrix), ncol = ncol(grad_matrix))
    }
    
    # Update v
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grad_matrix ^ 2)
    
    # Compute update
    updates[[i]] <- lr * grad_matrix / (sqrt(params$v[[i]]) + epsilon)
  }
  
  return(list(
    v = params$v,
    updates = updates
  ))
}


adagrad_update <- function(params, grads, lr, epsilon = 1e-8) {
  # Initialize r as a list if not already
  if (!is.list(params$r)) {
    params$r <- vector("list", length(grads))
  }
  
  # Initialize updates
  updates <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Init r if missing or mismatched
    if (is.null(params$r[[i]]) || !all(dim(params$r[[i]]) == grad_dims)) {
      params$r[[i]] <- array(0, dim = grad_dims)
    }
    
    # Update r
    params$r[[i]] <- params$r[[i]] + grads[[i]]^2
    
    # Compute update
    updates[[i]] <- lr * grads[[i]] / (sqrt(params$r[[i]]) + epsilon)
  }
  
  # Return as full structured output
  return(list(
    params = params,
    weights_update = updates,
    biases_update = updates  # If shared logic for both — differentiate if needed
  ))
}

adadelta_update <- function(params, grads, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, t = 1) {
  if (!is.list(params$m)) params$m <- vector("list", length(grads))
  if (!is.list(params$v)) params$v <- vector("list", length(grads))
  
  updates <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    # Ensure gradient is in array form
    grad_dims <- dim(grads[[i]])
    if (is.null(grad_dims)) {
      grads[[i]] <- array(grads[[i]], dim = c(length(grads[[i]]), 1))
      grad_dims <- dim(grads[[i]])
    }
    
    # Initialize accumulators with same shape
    if (is.null(params$m[[i]]) || !identical(dim(params$m[[i]]), grad_dims)) {
      params$m[[i]] <- array(0, dim = grad_dims)
    }
    if (is.null(params$v[[i]]) || !identical(dim(params$v[[i]]), grad_dims)) {
      params$v[[i]] <- array(0, dim = grad_dims)
    }
    
    # Clean NaNs from gradients
    grads[[i]][is.na(grads[[i]])] <- 0
    params$v[[i]][is.na(params$v[[i]])] <- 0
    
    # Adadelta updates
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grads[[i]] ^ 2)
    v_hat <- params$v[[i]] / (1 - beta2 ^ t)
    
    delta <- (sqrt(params$m[[i]] + epsilon) / sqrt(v_hat + epsilon)) * grads[[i]]
    delta[is.nan(delta) | is.infinite(delta)] <- 0
    delta <- pmin(pmax(delta, -5), 5)  # optional clip
    
    params$m[[i]] <- beta1 * params$m[[i]] + (1 - beta1) * (delta ^ 2)
    updates[[i]] <- delta
  }
  
  return(list(
    params = list(m = params$m, v = params$v),
    weights_update = updates,
    biases_update = updates
  ))
}

# Stochastic Gradient Descent with Momentum
# Define the sgd_update function with improvements
# Improved SGD Update Function
# Define the sgd_update function
sgd_momentum_update <- function(params, grads, lr, momentum) {
  # Initialize momentum as a list if it is not already
  if (!is.list(params$momentum)) {
    params$momentum <- vector("list", length(grads))
  }
  
  # Initialize weights_update and biases_update as lists
  weights_update <- vector("list", length(grads))
  biases_update  <- vector("list", length(grads))
  
  # Update each element of momentum and calculate weights_update and biases_update
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    if (is.null(params$momentum[[i]]) || !all(dim(params$momentum[[i]]) == grad_dims)) {
      params$momentum[[i]] <- array(0, dim = grad_dims)
    }
    
    # Momentum update
    params$momentum[[i]] <- momentum * params$momentum[[i]] - lr * grads[[i]]
    
    weights_update[[i]] <- params$momentum[[i]]
    biases_update[[i]]  <- lr * grads[[i]]  # This is just a placeholder; bias logic might differ
  }
  
  # Standardize dimensions to match grads
  for (i in seq_along(weights_update)) {
    if (is.null(dim(grads[[i]]))) {
      weights_update[[i]] <- array(weights_update[[i]], dim = c(1))
      biases_update[[i]]  <- array(biases_update[[i]],  dim = c(1))
    } else {
      weights_update[[i]] <- matrix(weights_update[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
      biases_update[[i]]  <- matrix(biases_update[[i]],  nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
    }
  }
  
  return(list(params = params, weights_update = weights_update, biases_update = biases_update))
}

sgd_update <- function(params, grads, lr) {
  updated_params <- list()
  
  weights_update <- list()
  biases_update  <- list()
  
  for (i in seq_along(grads)) {
    grad_matrix <- as.matrix(grads[[i]])
    param_matrix <- as.matrix(params$param)
    
    # Safe reshape if needed
    if (!all(dim(grad_matrix) == dim(param_matrix))) {
      grad_matrix <- matrix(rep(grad_matrix, length.out = length(param_matrix)), nrow = nrow(param_matrix))
    }
    
    update_matrix <- lr * grad_matrix
    updated_param_matrix <- param_matrix - update_matrix
    
    # Store updates
    updated_params$param <- updated_param_matrix
    updated_params$momentum <- params$momentum  # even if unused
    
    weights_update[[i]] <- update_matrix
    biases_update[[i]]  <- matrix(0, nrow = 1, ncol = 1)  # dummy for compatibility
  }
  
  return(list(
    params = updated_params,
    weights_update = weights_update,
    biases_update = biases_update
  ))
}



nag_update <- function(params, grads, lr, beta1 = 0.9) {
  weights_update <- vector("list", length(grads))
  biases_update <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    if (is.null(grad_dims)) {
      grad_dims <- c(length(grads[[i]]), 1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    if (length(params$momentum) < i) {
      params$momentum[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    if (length(params$fast_weights) < i) {
      params$fast_weights[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    if (length(params$fast_biases) < i) {
      params$fast_biases[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    
    # ✅ Use beta1
    params$momentum[[i]] <- beta1 * params$momentum[[i]] + grads[[i]]
    weights_update[[i]] <- lr * (beta1 * params$momentum[[i]] + grads[[i]])
    biases_update[[i]] <- lr * grads[[i]]
    
    params$fast_weights[[i]] <- params$fast_weights[[i]] - weights_update[[i]]
    params$fast_biases[[i]] <- params$fast_biases[[i]] - biases_update[[i]]
  }
  
  return(list(params = params, weights_update = weights_update, biases_update = biases_update))
}

ftrl_update <- function(params, grads, lr,
                        alpha   = 0.1,
                        beta    = beta1,
                        lambda1 = 0.01,
                        lambda2 = 0.01) {
  # 1) Wrap single matrix/vector into a list
  if (!is.list(grads)) {
    grads <- list(grads)
  }
  n_grads <- length(grads)
  
  # 2) Ensure params$z and params$n exist and are lists of correct length
  if (!is.list(params$z) || length(params$z) != n_grads) {
    params$z <- vector("list", n_grads)
  }
  if (!is.list(params$n) || length(params$n) != n_grads) {
    params$n <- vector("list", n_grads)
  }
  
  # Prepare outputs
  weights_update <- vector("list", n_grads)
  biases_update  <- vector("list", n_grads)
  
  for (i in seq_len(n_grads)) {
    # 3) Force grad to matrix
    grad_i <- grads[[i]]
    if (!is.matrix(grad_i)) {
      grad_i <- matrix(grad_i, nrow = length(grad_i), ncol = 1)
    }
    
    # 4) Initialize z[i], n[i] if missing or wrong shape
    if (is.null(params$z[[i]]) ||
        !identical(dim(params$z[[i]]), dim(grad_i))) {
      params$z[[i]] <- matrix(0, nrow = nrow(grad_i), ncol = ncol(grad_i))
    }
    if (is.null(params$n[[i]]) ||
        !identical(dim(params$n[[i]]), dim(grad_i))) {
      params$n[[i]] <- matrix(0, nrow = nrow(grad_i), ncol = ncol(grad_i))
    }
    
    # Shortcut references
    z_i <- params$z[[i]]
    n_i <- params$n[[i]]
    
    # 5) Compute new accumulators
    n_new <- n_i + grad_i^2
    sigma <- (sqrt(n_new) - sqrt(n_i)) / alpha
    z_new <- z_i + grad_i - sigma * 0  # if you had slow weights, you'd subtract them here
    
    # 6) FTRL update step
    denom <- (beta + sqrt(n_new)) / alpha + lambda2
    w_update <- -1/denom * (z_new - lambda1 * sign(z_new))
    
    # 7) Store updates
    weights_update[[i]] <- w_update
    biases_update[[i]]  <- matrix(0,
                                  nrow = nrow(grad_i),
                                  ncol = ncol(grad_i))  # still zero for biases
    
    # 8) Save back updated accumulators
    params$z[[i]] <- z_new
    params$n[[i]] <- n_new
  }
  
  return(list(
    params         = params,
    weights_update = weights_update,
    biases_update  = biases_update
  ))
}


# LAMB Update Function #$$$$$$$$$$$$$
lamb_update <- function(params, grads, lr, beta1, beta2, eps, lambda) {
  # Ensure param and grads are numeric vectors
  params$param <- as.numeric(params$param)
  grads <- as.numeric(grads)
  
  # Ensure m and v are initialized and numeric
  if (is.null(params$m)) params$m <- rep(0, length(grads))
  if (is.null(params$v)) params$v <- rep(0, length(grads))
  
  m <- beta1 * params$m + (1 - beta1) * grads
  v <- beta2 * params$v + (1 - beta2) * (grads^2)
  
  m_hat <- m / (1 - beta1)
  v_hat <- v / (1 - beta2)
  
  update <- m_hat / (sqrt(v_hat) + eps)
  
  # Trust ratio scaling (LAMB-specific)
  w_norm <- sqrt(sum(params$param^2))
  u_norm <- sqrt(sum(update^2))
  trust_ratio <- ifelse(w_norm > 0 && u_norm > 0, w_norm / u_norm, 1.0)
  
  update <- trust_ratio * update
  
  # Apply weight decay
  update <- update + lambda * params$param
  
  updated_param <- params$param - lr * update
  
  return(list(
    params = list(
      param = updated_param,
      m = m,
      v = v
    ),
    weights_update = list(update),  # Required for downstream [[1]]
    biases_update = list(rep(0, length(update)))  # Placeholder for biases
  ))
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

calculate_performance <- function(SONN, Rdata, labels, lr, model_iter_num, num_epochs, threshold, learn_time, predicted_output, prediction_time, ensemble_number, run_id, weights, biases, activation_functions, ML_NN, verbose) {
  
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
    MSE                           = MSE(SONN, Rdata, labels, predicted_output, verbose),
    MAE                           = MAE(SONN, Rdata, labels, predicted_output, verbose),
    RMSE                          = RMSE(SONN, Rdata, labels, predicted_output, verbose),
    accuracy                      = accuracy(SONN, Rdata, labels, predicted_output, verbose),
    precision                     = precision(SONN, Rdata, labels, predicted_output, verbose),
    recall                        = recall(SONN, Rdata, labels, predicted_output, verbose),
    f1_score                      = f1_score(SONN, Rdata, labels, predicted_output, verbose),
    accuracy_tuned                = accuracy_tuned(SONN, Rdata, labels, predicted_output, metric_for_tuning = "accuracy", grid, verbose),
    speed                         = speed(SONN, prediction_time, verbose),
    speed_learn                   = speed_learn(SONN, learn_time, verbose),
    memory_usage                  = memory_usage(SONN, Rdata, verbose),
    robustness                    = robustness(SONN, Rdata, labels, lr, num_epochs, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, dropout_rates, verbose),
    custom_relative_error_binned  = custom_relative_error_binned(SONN, Rdata, labels, predicted_output, verbose)
  )
  
  
  # --- Clean invalid metrics (NULL, NA, or TRUE by accident) ---
  for (name in names(perf_metrics)) {
    val <- perf_metrics[[name]]
    if (is.null(val) || any(is.na(val)) || isTRUE(val)) {
      perf_metrics[[name]] <- NULL
    }
  }
  
  return(list(metrics = perf_metrics, names = names(perf_metrics)))
}


calculate_relevance <- function(SONN, Rdata, labels, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, ML_NN, verbose) {
  
  # --- Standardize single-layer to list format ---
  if (!is.list(SONN$weights)) {
    SONN$weights <- list(SONN$weights)
  }
  
  # --- Active relevance metrics ---
  rel_metrics <- list(
    hit_rate     = tryCatch(hit_rate(SONN, Rdata, predicted_output, labels, verbose), error = function(e) NULL),
    ndcg         = tryCatch(ndcg(SONN, Rdata, predicted_output, labels, verbose), error = function(e) NULL),
    diversity    = tryCatch(diversityfunction(SONN, Rdata, predicted_output, verbose), error = function(e) NULL),
    serendipity  = tryCatch(serendipityfunction(SONN, Rdata, predicted_output, verbose), error = function(e) NULL)
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
      rel_metrics[[metric_name]] <- NULL
    } else {
      val <- rel_metrics[[metric_name]]
      if (is.null(val) || any(is.na(val)) || isTRUE(val)) {
        rel_metrics[[metric_name]] <- NULL
      }
    }
  }
  
  return(list(metrics = rel_metrics, names = names(rel_metrics)))
}




calculate_performance_grouped <- function(SONN_list, Rdata, labels, lr, num_epochs, threshold, predicted_output_list, prediction_time_list, ensemble_number, run_id, ML_NN, verbose,
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

calculate_relevance_grouped <- function(SONN_list, Rdata, labels, predicted_output_list, ensemble_number, run_id, ML_NN, verbose,
    agg_method = c("mean","median","vote"),
    metric_mode = c("aggregate_predictions+rep_sonn", "average_per_model")
) {
  agg_method <- match.arg(agg_method)
  metric_mode <- match.arg(metric_mode)
  p_agg <- aggregate_predictions(predicted_output_list, method = agg_method)
  
  if (metric_mode == "aggregate_predictions+rep_sonn") {
    rep_sonn <- pick_representative_sonn(SONN_list, predicted_output_list, labels)
    calculate_relevance(
      SONN             = rep_sonn,
      Rdata            = Rdata,
      labels           = labels,
      model_iter_num   = NA_integer_,
      predicted_output = p_agg,
      ensemble_number  = ensemble_number,
      weights          = rep_sonn$weights,
      biases           = rep_sonn$biases,
      activation_functions = rep_sonn$activation_functions,
      ML_NN            = ML_NN,
      verbose          = verbose
    )
  } else {
    rels <- lapply(seq_along(SONN_list), function(i) {
      calculate_relevance(
        SONN             = SONN_list[[i]],
        Rdata            = Rdata,
        labels           = labels,
        model_iter_num   = i,
        predicted_output = predicted_output_list[[i]],
        ensemble_number  = ensemble_number,
        weights          = SONN_list[[i]]$weights,
        biases           = SONN_list[[i]]$biases,
        activation_functions = SONN_list[[i]]$activation_functions,
        ML_NN            = ML_NN,
        verbose          = FALSE
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






###prune

# Loss Function: Computes the loss based on the type specified and includes regularization term
loss_function <- function(predictions, labels, reg_loss_total, loss_type) {
  # Default reg_loss_total to 0 if NULL
  if (is.null(reg_loss_total)) reg_loss_total <- 0
  
  print(dim(predictions))
  print(dim(labels))
  
  # Handle missing or NULL loss_type gracefully
  if (is.null(loss_type)) {
    print("Loss type is NULL. Please specify 'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'.")
    return(NA)
  }
  
  # Compute loss based on type
  if (loss_type == "MSE") {
    loss <- mean((predictions - labels)^2)
    
  } else if (loss_type == "MAE") {
    loss <- mean(abs(predictions - labels))
    
  } else if (loss_type == "CrossEntropy") {
    epsilon <- 1e-15
    predictions <- pmax(pmin(predictions, 1 - epsilon), epsilon)
    loss <- -mean(labels * log(predictions) + (1 - labels) * log(1 - predictions))
    
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
#      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \    $$$$$$$$$$$$$$$$$$$$$$$
#   __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__   $$$$$$$$$$$$$$$$$$$$$$$
#  (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)  $$$$$$$$$$$$$$$$$$$$$$$
#    || M ||      || E ||      || T ||      || R ||      || I ||      || C ||      || S ||      $$$$$$$$$$$$$$$$$$$$$$$
# _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._     $$$$$$$$$$$$$$$$$$$$$$$
#(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)    $$$$$$$$$$$$$$$$$$$$$$$
#`-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'      $$$$$$$$$$$$$$$$$$$$$$$
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
topographic_error <- function(SONN, Rdata, threshold, verbose = FALSE) {
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
clustering_quality_db <- function(SONN, Rdata, cluster_assignments, verbose = FALSE) {
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



# Classification accuracy placeholder
MSE <- function(SONN, Rdata, labels, predicted_output, verbose) {
  
  # for(i in num_layers){
  
  if (ncol(labels) != ncol(predicted_output)) {
    if (ncol(predicted_output) < ncol(labels)) {
      # Calculate the required replication factor
      rep_factor <- ceiling((nrow(labels) * ncol(labels)) / length(predicted_output))
      # Create the replicated vector and check its length
      replicated_predicted_output <- rep(predicted_output, rep_factor)
      # Truncate the replicated vector to match the required length
      replicated_predicted_output <- replicated_predicted_output[1:(nrow(labels) * ncol(labels))]
      # Create the matrix and check its dimensions
      predicted_output_matrix <- matrix(replicated_predicted_output, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
    } else {
      # Truncate predicted_output to match the number of columns in labels
      truncated_predicted_output <- predicted_output[, 1:ncol(labels)]
      # Create the matrix and check its dimensions
      predicted_output_matrix <- matrix(truncated_predicted_output, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
    }
  } else {
    predicted_output_matrix <- predicted_output
  }
  
  # Calculate the error
  error_prediction <- predicted_output_matrix - labels
  
  # Calculate the classification accuracy on the  Rdata
  accuracy <- mean((error_prediction)^2)
  # if (verbose) {
  #   print("MSE")
  #   print(accuracy)
  # }
  # Return the accuracy
  return(accuracy)
  # if (verbose) {
  #   print("MSE complete")
  # }
  
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
robustness <- function(SONN, Rdata, labels, lr, num_epochs, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, dropout_rates, verbose) {
  # --- DEBUG CHECK FOR CORRUPTED BIASES ---
  cat(">>> DEBUG: Checking SONN$biases types...\n")
  
  if (is.null(SONN$biases)) {
    cat("SONN$biases is NULL\n")
  } else if (!is.list(SONN$biases)) {
    cat("SONN$biases is not a list, type is:", class(SONN$biases), "\n")
  } else {
    for (i in seq_along(SONN$biases)) {
      bias_class <- class(SONN$biases[[i]])
      cat(sprintf("Layer %d bias type: %s\n", i, bias_class))
      if (is.function(SONN$biases[[i]])) {
        stop(sprintf("ERROR: SONN$biases[[%d]] is a function (closure). It should be numeric!", i))
      }
    }
  }
  
  # my_robustness_vector <<- seq(40, 0.1, by = -0.2)
  # losses <- rep(0, length(my_robustness_vector))
  
  # Add noise to the Rdata
  noisy_Rdata <<- as.matrix(Rdata + rnorm(n = nrow(Rdata) * ncol(Rdata), mean = 0, sd = 0.2)) #my_robustness_vector[200]))
  
  # Add outliers to the Rdata
  outliers <<- as.matrix(rnorm(n = 1.5 * ncol(noisy_Rdata), mean = 5, sd = 1), ncol = ncol(noisy_Rdata))
  
  noisy_Rdata[sample(1:nrow(noisy_Rdata), nrow(outliers)), ] <- outliers
  
  if(learnOnlyTrainingRun == FALSE){
    
    # Predict the class of the noisy Rdata
    noisy_Rdata_predictions <- SONN$predict(noisy_Rdata, weights, biases, activation_functions)#(Rdata, weights, biases, activation_functions)
    
    # Add debug statements to print dimensions and lengths
    # print(paste("Dimensions of labels:", paste(dim(labels), collapse = " x ")))
    # print(paste("Dimensions of noisy_Rdata_predictions$predicted_output:", paste(dim(noisy_Rdata_predictions$predicted_output), collapse = " x ")))
    
    if (ncol(labels) != ncol(noisy_Rdata_predictions$predicted_output)) {
      if (ncol(noisy_Rdata_predictions$predicted_output) < ncol(labels)) {
        # Calculate the required replication factor
        rep_factor <- ceiling((nrow(labels) * ncol(labels)) / length(noisy_Rdata_predictions$predicted_output))
        # Create the replicated vector and check its length
        replicated_predicted_output <- rep(noisy_Rdata_predictions$predicted_output, rep_factor)
        # Truncate the replicated vector to match the required length
        replicated_predicted_output <- replicated_predicted_output[1:(nrow(labels) * ncol(labels))]
        # Create the matrix and check its dimensions
        predicted_output_matrix <- matrix(replicated_predicted_output, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
      } else {
        # Truncate predicted_output to match the number of columns in labels
        truncated_predicted_output <- noisy_Rdata_predictions$predicted_output[, 1:ncol(labels)]
        # Create the matrix and check its dimensions
        predicted_output_matrix <- matrix(truncated_predicted_output, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
      }
    } else {
      predicted_output_matrix <- noisy_Rdata_predictions$predicted_output
    }
    
    # Calculate the error
    error_1000x1_r <- predicted_output_matrix - labels
    
    
    # print("Calculation complete")
    
    
    losses <- mean(error_1000x1_r^2)
    
    # Find the index where the validation loss starts to increase
    optimal_epoch_robust <- which(diff(losses) > 0)[1]
    
    if(plot_robustness){
      if (any(is.nan(losses)) || any(is.infinite(losses))) {
        # Handle NaN or Inf values in 'losses'
        print("NaN or Inf values detected in 'losses'. Cannot plot.")
      } else {
        # Plot the loss over epochs
        plot(losses, type = 'l', main = paste('Loss Over 0.2 SD increase in Noisy Data for SONN', model_iter_num), xlab = 'Robustness', ylab = 'Loss per 0.2 SD Increase', col = 'skyblue', lwd = 2.0)
        
        # Add a point or line indicating the optimal epoch
        points(optimal_epoch_robust, losses[optimal_epoch_robust], col = 'blue', pch = 16)
        
        # Add text for the equation
        eq <- paste("Optimal SD:", optimal_epoch_robust, "\nLoss:", round(losses[optimal_epoch_robust], 2))
        text(optimal_epoch_robust - 0.5, losses[optimal_epoch_robust] + 0.5, eq, pos = 4, col = "blue", adj = 0)
      }
    }
    
    # Calculate the classification accuracy on the noisy Rdata
    accuracy <- mean((noisy_Rdata_predictions$predicted_output - labels)^2)
    
    
  }else if (learnOnlyTrainingRun == TRUE){
    
    learn_r <- SONN$learn(noisy_Rdata, labels, lr)
  }
  
  # for(i in 1:length(my_robustness_vector)){
  #
  #
  #
  #
  # }
  # if (verbose) {
  #   print("robustness")
  #   print(accuracy)
  # }
  # Return the accuracy on the noisy Rdata
  return(accuracy)
  # if (verbose) {
  #   print("robustness complete")
  # }
  
}

# Hit Rate
hit_rate <- function(SONN, Rdata, predicted_output, labels, verbose) {
  # Predict the output for each Rdata point
  # print("hit rate before predict")
  # predictions <- SONN$predict(Rdata, weights, biases, activation_functions)
  # print("hit rate after predict")
  Rdata <- data.frame(Rdata)
  # Identify the relevant Rdata points
  relevant_Rdata <<- Rdata[Rdata$class == "relevant", ]
  
  # Calculate the hit rate
  hit_rate <- sum(predicted_output %in% relevant_Rdata$id) / nrow(relevant_Rdata)
  # if (verbose) {
  #   print("hit_rate")
  #   print(hit_rate)
  # }
  # Return the hit rate
  return(hit_rate)
  # if (verbose) {
  #   print("hit_rate complete")
  # }
}

# -------------------------
# Accuracy (auto binary/multiclass)
# -------------------------
accuracy <- function(SONN, Rdata, labels, predicted_output, verbose) {
  labels <- as.matrix(labels); predicted_output <- as.matrix(predicted_output)
  storage.mode(labels) <- "double"; storage.mode(predicted_output) <- "double"
  
  nL <- ncol(labels); nP <- ncol(predicted_output)
  is_multiclass <- (max(nL, nP) > 1L)
  
  if (is_multiclass) {
    # --- prepare prediction matrix with K columns ---
    K <- max(nL, nP)
    if (nP < K) {
      total_needed <- nrow(labels) * K
      replicated <- rep(as.vector(predicted_output), length.out = total_needed)
      predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = K, byrow = FALSE)
    } else if (nP > K) {
      predicted_output_matrix <- predicted_output[, 1:K, drop = FALSE]
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    # --- true class ids ---
    if (nL > 1L) {
      true_class <- max.col(labels, ties.method = "first")
    } else {
      true_class <- as.integer(labels[, 1])
      if (min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    }
    
    pred_class <- max.col(predicted_output_matrix, ties.method = "first")
    acc <- mean(pred_class == true_class)
  } else {
    # --- binary path (single column) ---
    # Align to 1 column (replicate/truncate style)
    if (nL != nP) {
      if (nP < nL) {
        total_needed <- nrow(labels) * nL
        replicated <- rep(as.vector(predicted_output), length.out = total_needed)
        predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = nL, byrow = FALSE)
      } else {
        predicted_output_matrix <- predicted_output[, 1:nL, drop = FALSE]
      }
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    y_true <- as.numeric(labels[, 1])
    u <- sort(unique(y_true))
    thresh <- if (length(u) == 2L && all(is.finite(u))) (u[1] + u[2]) / 2 else 0.5
    y_pred <- as.integer(predicted_output_matrix[, 1] >= thresh)
    y_true_class <- if (all(y_true %in% c(0, 1))) as.integer(y_true) else as.integer(y_true >= thresh)
    acc <- mean(y_pred == y_true_class)
  }
  
  # if (verbose) { print("Accuracy"); print(acc) }
  return(as.numeric(acc))
  # if (verbose) { print("Accuracy complete") }
}

# -------------------------
# Precision (macro for multiclass; standard for binary)
# -------------------------
precision <- function(SONN, Rdata, labels, predicted_output, verbose) {
  labels <- as.matrix(labels); predicted_output <- as.matrix(predicted_output)
  storage.mode(labels) <- "double"; storage.mode(predicted_output) <- "double"
  
  nL <- ncol(labels); nP <- ncol(predicted_output)
  is_multiclass <- (max(nL, nP) > 1L)
  
  if (is_multiclass) {
    K <- max(nL, nP)
    if (nP < K) {
      total_needed <- nrow(labels) * K
      replicated <- rep(as.vector(predicted_output), length.out = total_needed)
      predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = K, byrow = FALSE)
    } else if (nP > K) {
      predicted_output_matrix <- predicted_output[, 1:K, drop = FALSE]
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    if (nL > 1L) {
      true_class <- max.col(labels, ties.method = "first")
    } else {
      true_class <- as.integer(labels[, 1])
      if (min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    }
    
    pred_class <- max.col(predicted_output_matrix, ties.method = "first")
    
    prec_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fp <- sum(pred_class == k & true_class != k)
      prec_per_class[k] <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    }
    prec <- mean(prec_per_class)
  } else {
    if (nL != nP) {
      if (nP < nL) {
        total_needed <- nrow(labels) * nL
        replicated <- rep(as.vector(predicted_output), length.out = total_needed)
        predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = nL, byrow = FALSE)
      } else {
        predicted_output_matrix <- predicted_output[, 1:nL, drop = FALSE]
      }
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    y_true <- as.numeric(labels[, 1])
    u <- sort(unique(y_true))
    thresh <- if (length(u) == 2L && all(is.finite(u))) (u[1] + u[2]) / 2 else 0.5
    y_pred <- as.integer(predicted_output_matrix[, 1] >= thresh)
    y_true_class <- if (all(y_true %in% c(0, 1))) as.integer(y_true) else as.integer(y_true >= thresh)
    
    tp <- sum(y_pred == 1L & y_true_class == 1L)
    fp <- sum(y_pred == 1L & y_true_class == 0L)
    prec <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
  }
  
  # if (verbose) { print("Precision"); print(prec) }
  return(as.numeric(prec))
  # if (verbose) { print("Precision complete") }
}

# -------------------------
# Recall (macro for multiclass; standard for binary)
# -------------------------
recall <- function(SONN, Rdata, labels, predicted_output, verbose) {
  labels <- as.matrix(labels); predicted_output <- as.matrix(predicted_output)
  storage.mode(labels) <- "double"; storage.mode(predicted_output) <- "double"
  
  nL <- ncol(labels); nP <- ncol(predicted_output)
  is_multiclass <- (max(nL, nP) > 1L)
  
  if (is_multiclass) {
    K <- max(nL, nP)
    if (nP < K) {
      total_needed <- nrow(labels) * K
      replicated <- rep(as.vector(predicted_output), length.out = total_needed)
      predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = K, byrow = FALSE)
    } else if (nP > K) {
      predicted_output_matrix <- predicted_output[, 1:K, drop = FALSE]
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    if (nL > 1L) {
      true_class <- max.col(labels, ties.method = "first")
    } else {
      true_class <- as.integer(labels[, 1])
      if (min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    }
    
    pred_class <- max.col(predicted_output_matrix, ties.method = "first")
    
    rec_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fn <- sum(pred_class != k & true_class == k)
      rec_per_class[k] <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    }
    rec <- mean(rec_per_class)
  } else {
    if (nL != nP) {
      if (nP < nL) {
        total_needed <- nrow(labels) * nL
        replicated <- rep(as.vector(predicted_output), length.out = total_needed)
        predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = nL, byrow = FALSE)
      } else {
        predicted_output_matrix <- predicted_output[, 1:nL, drop = FALSE]
      }
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    y_true <- as.numeric(labels[, 1])
    u <- sort(unique(y_true))
    thresh <- if (length(u) == 2L && all(is.finite(u))) (u[1] + u[2]) / 2 else 0.5
    y_pred <- as.integer(predicted_output_matrix[, 1] >= thresh)
    y_true_class <- if (all(y_true %in% c(0, 1))) as.integer(y_true) else as.integer(y_true >= thresh)
    
    tp <- sum(y_pred == 1L & y_true_class == 1L)
    fn <- sum(y_pred == 0L & y_true_class == 1L)
    rec <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
  }
  
  # if (verbose) { print("Recall"); print(rec) }
  return(as.numeric(rec))
  # if (verbose) { print("Recall complete") }
}

# -------------------------
# F1 Score (macro for multiclass; standard for binary)
# -------------------------
f1_score <- function(SONN, Rdata, labels, predicted_output, verbose) {
  labels <- as.matrix(labels); predicted_output <- as.matrix(predicted_output)
  storage.mode(labels) <- "double"; storage.mode(predicted_output) <- "double"
  
  nL <- ncol(labels); nP <- ncol(predicted_output)
  is_multiclass <- (max(nL, nP) > 1L)
  
  if (is_multiclass) {
    K <- max(nL, nP)
    if (nP < K) {
      total_needed <- nrow(labels) * K
      replicated <- rep(as.vector(predicted_output), length.out = total_needed)
      predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = K, byrow = FALSE)
    } else if (nP > K) {
      predicted_output_matrix <- predicted_output[, 1:K, drop = FALSE]
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    if (nL > 1L) {
      true_class <- max.col(labels, ties.method = "first")
    } else {
      true_class <- as.integer(labels[, 1])
      if (min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    }
    
    pred_class <- max.col(predicted_output_matrix, ties.method = "first")
    
    f1_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fp <- sum(pred_class == k & true_class != k)
      fn <- sum(pred_class != k & true_class == k)
      p <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
      r <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
      f1_per_class[k] <- if ((p + r) == 0) 0 else 2 * p * r / (p + r)
    }
    f1 <- mean(f1_per_class)
  } else {
    if (nL != nP) {
      if (nP < nL) {
        total_needed <- nrow(labels) * nL
        replicated <- rep(as.vector(predicted_output), length.out = total_needed)
        predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = nL, byrow = FALSE)
      } else {
        predicted_output_matrix <- predicted_output[, 1:nL, drop = FALSE]
      }
    } else {
      predicted_output_matrix <- predicted_output
    }
    
    y_true <- as.numeric(labels[, 1])
    u <- sort(unique(y_true))
    thresh <- if (length(u) == 2L && all(is.finite(u))) (u[1] + u[2]) / 2 else 0.5
    y_pred <- as.integer(predicted_output_matrix[, 1] >= thresh)
    y_true_class <- if (all(y_true %in% c(0, 1))) as.integer(y_true) else as.integer(y_true >= thresh)
    
    tp <- sum(y_pred == 1L & y_true_class == 1L)
    fp <- sum(y_pred == 1L & y_true_class == 0L)
    fn <- sum(y_pred == 0L & y_true_class == 1L)
    
    p <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    r <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    f1 <- if ((p + r) == 0) 0 else 2 * p * r / (p + r)
  }
  
  # if (verbose) { print("F1 Score"); print(f1) }
  return(as.numeric(f1))
  # if (verbose) { print("F1 Score complete") }
}

accuracy_tuned <- function(
    SONN, Rdata, labels, predicted_output,
    metric_for_tuning = c("accuracy", "f1", "precision", "recall",
                          "macro_f1", "macro_precision", "macro_recall"),
    threshold_grid = seq(0.05, 0.95, by = 0.01),
    verbose = FALSE
) {
  metric_for_tuning <- match.arg(metric_for_tuning)
  grid <- sanitize_grid(threshold_grid)
  
  # --- helpers (local, no deps) ---
  .to_matrix <- function(x) { x <- as.matrix(x); storage.mode(x) <- "double"; x }
  .labels_to_ids <- function(L) {
    # If already a vector of class ids (1..K), return as-is (numeric/integer).
    if (is.null(dim(L))) return(as.integer(L))
    # Otherwise assume one-hot / probability matrix → argmax
    max.col(L, ties.method = "first")
  }
  .binary_metrics <- function(y_true01, y_pred01) {
    y_true01 <- as.integer(y_true01); y_pred01 <- as.integer(y_pred01)
    TP <- sum(y_true01 == 1L & y_pred01 == 1L, na.rm = TRUE)
    FP <- sum(y_true01 == 0L & y_pred01 == 1L, na.rm = TRUE)
    FN <- sum(y_true01 == 1L & y_pred01 == 0L, na.rm = TRUE)
    denom_p <- TP + FP; denom_r <- TP + FN
    precision <- if (denom_p > 0) TP / denom_p else 0
    recall    <- if (denom_r > 0) TP / denom_r else 0
    F1        <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
    list(precision = as.numeric(precision),
         recall = as.numeric(recall),
         F1 = as.numeric(F1))
  }
  .macro_metrics <- function(y_true_ids, y_pred_ids, K = NULL) {
    y_true_ids <- as.integer(y_true_ids)
    y_pred_ids <- as.integer(y_pred_ids)
    if (is.null(K)) K <- max(c(y_true_ids, y_pred_ids), na.rm = TRUE)
    prec <- rec <- f1 <- numeric(K)
    for (k in seq_len(K)) {
      TP <- sum(y_true_ids == k & y_pred_ids == k, na.rm = TRUE)
      FP <- sum(y_true_ids != k & y_pred_ids == k, na.rm = TRUE)
      FN <- sum(y_true_ids == k & y_pred_ids != k, na.rm = TRUE)
      p  <- if (TP + FP > 0) TP / (TP + FP) else 0
      r  <- if (TP + FN > 0) TP / (TP + FN) else 0
      f  <- if ((p + r) > 0) 2 * p * r / (p + r) else 0
      prec[k] <- p; rec[k] <- r; f1[k] <- f
    }
    list(precision = mean(prec), recall = mean(rec), F1 = mean(f1))
  }
  
  # --- coerce to matrices for existing utilities you have ---
  L <- .to_matrix(labels)
  P <- .to_matrix(predicted_output)
  
  # --- binary vs multiclass inference (use your existing inferer if available) ---
  is_binary <- FALSE
  if (exists("infer_is_binary", mode = "function")) {
    bin_info <- infer_is_binary(L, P); is_binary <- isTRUE(bin_info$is_binary)
  } else {
    is_binary <- (max(ncol(L), ncol(P)) <= 1L)
  }
  
  if (is_binary) {
    # Labels → 0/1 vector
    if (exists("labels_to_binary_vec", mode = "function")) {
      Lb <- labels_to_binary_vec(L)
    } else {
      # fallback: if matrix with one col, assume 0/1 already
      Lb <- as.integer(L[, 1] > 0.5)
    }
    if (is.null(Lb)) stop("Binary inferred, but labels cannot be coerced to {0,1}.")
    
    # Predictions → positive-class probability
    if (exists("preds_to_pos_prob", mode = "function")) {
      Ppos <- preds_to_pos_prob(P)
    } else {
      Ppos <- as.numeric(P)  # assume already a prob col
    }
    if (is.null(Ppos)) stop("Binary inferred, but predictions cannot be coerced to a single probability column.")
    
    # Tune threshold
    thr_res <- tune_threshold_accuracy(
      predicted_output = matrix(Ppos, ncol = 1L),
      labels           = matrix(Lb,   ncol = 1L),
      metric           = metric_for_tuning,
      threshold_grid   = grid,
      verbose          = FALSE
    )
    best_t <- as.numeric(thr_res$thresholds[1])
    
    # Classify
    y_pred_class <- as.integer(Ppos >= best_t)
    
    # Accuracy (use your accuracy() to keep semantics)
    acc_prop <- accuracy(
      SONN = SONN, Rdata = Rdata,
      labels = matrix(Lb, ncol = 1L),
      predicted_output = matrix(y_pred_class, ncol = 1L),
      verbose = FALSE
    )
    acc_prop <- as.numeric(acc_prop)
    
    # Binary metrics
    m <- .binary_metrics(Lb, y_pred_class)
    
    return(list(
      accuracy  = acc_prop,
      precision = m$precision,
      recall    = m$recall,
      F1        = m$F1,
      details   = list(
        best_threshold  = best_t,
        best_thresholds = thr_res$thresholds,  # keep vector (best first)
        y_pred_class    = y_pred_class,
        grid_used       = grid
      )
    ))
  }
  
  # ---------------- Multiclass ----------------
  K <- max(ncol(L), ncol(P)); if (K < 1L) K <- 1L
  
  thr_res <- tune_threshold_accuracy(
    predicted_output = P,
    labels           = L,
    metric           = metric_for_tuning,
    threshold_grid   = grid,
    verbose          = FALSE
  )
  best_thresholds <- thr_res$thresholds
  pred_ids        <- thr_res$y_pred_class  # 1..K
  
  # One-hot for your existing accuracy()
  N <- if (!is.null(nrow(P))) nrow(P) else length(pred_ids)
  P_for_acc <- one_hot_from_ids(pred_ids, K = K, N = N)
  
  acc_prop <- accuracy(
    SONN = SONN, Rdata = Rdata,
    labels = L,
    predicted_output = P_for_acc,
    verbose = FALSE
  )
  acc_prop <- as.numeric(acc_prop)
  
  # Macro metrics at the tuned thresholds
  true_ids <- .labels_to_ids(L)
  mm <- .macro_metrics(true_ids, pred_ids, K = K)
  
  if (verbose) {
    message(sprintf("[Multiclass] acc = %.6f | macro P=%.6f R=%.6f F1=%.6f",
                    acc_prop, mm$precision, mm$recall, mm$F1))
  }
  
  list(
    accuracy  = acc_prop,
    precision = mm$precision,
    recall    = mm$recall,
    F1        = mm$F1,
    details   = list(
      best_threshold  = NA_real_,         # per-class thresholds provided below
      best_thresholds = best_thresholds,  # length K
      y_pred_class    = pred_ids,         # integers 1..K
      grid_used       = grid
    )
  )
}


MAE <- function(SONN, Rdata, labels, predicted_output, verbose) {
  
  if (ncol(labels) != ncol(predicted_output)) {
    if (ncol(predicted_output) < ncol(labels)) {
      # Calculate the required replication factor
      rep_factor <- ceiling((nrow(labels) * ncol(labels)) / length(predicted_output))
      # Create the replicated vector and check its length
      replicated_predicted_output <- rep(predicted_output, rep_factor)
      # Truncate the replicated vector to match the required length
      replicated_predicted_output <- replicated_predicted_output[1:(nrow(labels) * ncol(labels))]
      # Create the matrix and check its dimensions
      predicted_output_matrix <- matrix(replicated_predicted_output, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
    } else {
      # Truncate predicted_output to match the number of columns in labels
      truncated_predicted_output <- predicted_output[, 1:ncol(labels)]
      # Create the matrix and check its dimensions
      predicted_output_matrix <- matrix(truncated_predicted_output, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
    }
  } else {
    predicted_output_matrix <- predicted_output
  }
  
  # Calculate the error
  error_prediction <- predicted_output_matrix - labels
  
  
  # Calculate the classification accuracy on the noisy Rdata
  accuracy <- mean(abs(error_prediction))
  
  # if (verbose) {
  #   print("MAE")
  #   print(accuracy)
  # }
  # Return the accuracy
  return(accuracy)
  # if (verbose) {
  #   print("MAE complete")
  # }
}

# NDCG (Normalized Discounted Cumulative Gain)
ndcg <- function(SONN, Rdata, predicted_output, labels, verbose) {
  # Convert to data frame if needed
  Rdata <- data.frame(Rdata)
  
  # Define relevance scores: assume binary relevance (1 for "relevant", 0 otherwise)
  relevance_scores <- ifelse(labels == "relevant", 1, 0)
  
  # Create a data frame to pair predicted outputs with relevance
  df <- data.frame(
    prediction = predicted_output,
    relevance = relevance_scores
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
  
  # if (exists("verbose") && verbose) {
  #   print("ndcg")
  #   print(ndcg_value)
  # }
  
  return(ndcg_value)
}


# MAP (Mean Average Precision)
custom_relative_error_binned <- function(SONN, Rdata, labels, predicted_output, verbose) {
  # coerce
  labels <- as.matrix(labels); predicted_output <- as.matrix(predicted_output)
  storage.mode(labels) <- "double"; storage.mode(predicted_output) <- "double"
  
  # align shapes (your logic)
  if (ncol(labels) != ncol(predicted_output)) {
    if (ncol(predicted_output) < ncol(labels)) {
      rep_factor <- ceiling((nrow(labels) * ncol(labels)) / length(predicted_output))
      replicated <- rep(predicted_output, rep_factor)[1:(nrow(labels) * ncol(labels))]
      predicted_output_matrix <- matrix(replicated, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
    } else {
      predicted_output_matrix <- matrix(predicted_output[, 1:ncol(labels)],
                                        nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
    }
  } else {
    predicted_output_matrix <- predicted_output
  }
  
  # error & safe percentage diff
  error_prediction <- predicted_output_matrix - labels
  denom <- abs(labels)
  denom[denom == 0 | !is.finite(denom)] <- .Machine$double.eps
  percentage_difference <- as.numeric(abs(error_prediction) / denom)
  
  # bins (percent thresholds) and labels
  bins <- c(0, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  brks <- bins / 100
  bin_names <- paste0("rel_", bins[-length(bins)], "_", bins[-1], "pct")
  
  # assign to bins (skip non-finite)
  finite_idx <- is.finite(percentage_difference)
  vals <- percentage_difference[finite_idx]
  
  mean_precisions <- numeric(length(brks) - 1)
  if (length(vals) > 0L) {
    idx <- findInterval(vals, brks, rightmost.closed = TRUE, all.inside = TRUE)
    for (j in seq_along(mean_precisions)) {
      v <- vals[idx == j]
      mean_precisions[j] <- if (length(v) == 0L) 0 else mean(v)
    }
  } else {
    mean_precisions[] <- 0
  }
  
  # if (isTRUE(verbose)) {
  #   print("mean_precisions")
  #   print(mean_precisions)
  # }
  
  # IMPORTANT: return a named LIST of scalars (not a vector) so your flattener can unlist into columns
  names(mean_precisions) <- bin_names
  return(as.list(mean_precisions))
}



# Diversity
diversity <- function(SONN, Rdata, predicted_output, verbose) {
  # Replace zero values to avoid log2(0)
  predicted_output[predicted_output == 0] <- .Machine$double.eps
  
  # Normalize predicted_output to a valid probability distribution if needed
  if (sum(predicted_output) != 1) {
    predicted_output <- predicted_output / sum(predicted_output)
  }
  
  # Calculate entropy (Shannon diversity)
  entropy <- -sum(predicted_output * log2(predicted_output), na.rm = TRUE)
  
  # Handle unexpected Inf/NaN
  if (!is.finite(entropy)) {
    entropy <- NA
    warning("Entropy calculation returned Inf or NaN. Setting to NA.")
  }
  
  # if (verbose) {
  #   cat("Diversity (entropy):\n")
  #   print(entropy)
  # }
  
  return(entropy)
}

# Novelty placeholder
RMSE <- function(SONN, Rdata, labels, predicted_output, verbose) {
  # Calculate the squared error between each prediction and its corresponding label
  squared_errors <- mapply(function(x, y) {
    (x - y)^2  # Squared error calculation
  }, predicted_output, labels)
  
  # Calculate the mean squared error
  mean_squared_error <- mean(squared_errors, na.rm = TRUE)
  
  # Calculate the RMSE
  rmse <- sqrt(mean_squared_error)
  
  # if (verbose) {
  #   print("RMSE")
  #   print(rmse)
  # }
  return(rmse)
  # if (verbose) {
  #   print("RMSE complete")
  # }
}

# Serendipity
serendipity <- function(SONN, Rdata, predicted_output, verbose) {
  # # Predict the output for each Rdata point
  # predictions <- SONN$predict(Rdata, labels)
  
  # Calculate the average number of times each prediction is made
  prediction_counts <- table(predicted_output)
  
  # Calculate the inverse of the prediction counts
  inverse_prediction_counts <- 1 / prediction_counts
  # if (verbose) {
  #   print("serendipity")
  #   print(mean(inverse_prediction_counts, na.rm = TRUE))
  # }
  # Return the average inverse prediction count
  return(mean(inverse_prediction_counts, na.rm = TRUE))
  # if (verbose) {
  #   print("serendipity complete")
  # }
}