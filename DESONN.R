source("optimizers.R")
function(showlibraries){
# install.packages("R6")
# install.packages("tensorflow")
# install.packages("cluster")
# install.packages("fpc", type = "source")
# install.packages("rlist")
# install.packages("pracma")
# install.packages("randomForest", type = "source")
library(tensorflow)
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

        },
initialize_weights = function(input_size, hidden_sizes, output_size, method = init_method, custom_scale = NULL) {
  weights <- list()
  biases <- list()
  
  clip_weights <- function(W, limit = 5) {
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
    
    return(clip_weights(W, limit = 5))
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
}


,


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
},# Dropout function with no default rate
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
  
  print("print(str(outputs))")
  print(str(outputs))
  



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
    print(str(errors[[self$num_layers]]))
    

# Store output error
errors <- vector("list", self$num_layers)
errors[[self$num_layers]] <- as.matrix(error_1000x10)
print(str(errors[[self$num_layers]]))




    # Propagate the error backwards
    if (self$ML_NN) {
      for (layer in (self$num_layers - 1):1) {
        cat("Layer:", layer, "\n")
        
        # Check for existence of weights and errors
        if (is.null(self$weights[[layer + 1]]) || is.null(errors[[layer + 1]])) {
          cat(paste("Skipping layer", layer, "- weights or errors do not exist\n"))
          next
        }
        
        # Print dimensions
        weight_dims <- dim(self$weights[[layer + 1]])
        error_dims <- dim(errors[[layer + 1]])
        cat("Weights dimensions:\n"); print(weight_dims)
        cat("Errors dimensions:\n"); print(error_dims)
        
        if (is.null(weight_dims) || is.null(error_dims)) {
          cat(paste("Skipping layer", layer, "- dimensions are NULL\n"))
          next
        }
        
        weight_in  <- weight_dims[1]  # rows = input to layer
        weight_out <- weight_dims[2]  # cols = output from layer
        
        error_rows <- error_dims[1]   # rows = batch size
        error_cols <- error_dims[2]   # cols = error signals
        
        # Adjust errors[[layer + 1]] if shape doesn't match expected [batch_size x weight_out]
        if (error_cols != weight_out) {
          if (error_cols > weight_out) {
            errors[[layer + 1]] <- errors[[layer + 1]][, 1:weight_out, drop = FALSE]
          } else {
            errors[[layer + 1]] <- matrix(
              rep(errors[[layer + 1]], length.out = error_rows * weight_out),
              nrow = error_rows,
              ncol = weight_out
            )
          }
        }
        
        # Propagate error: errors[[l]] = errors[[l+1]] %*% t(weights[[l+1]])
        cat("Backpropagating errors for layer", layer, "\n")
        errors[[layer]] <- errors[[layer + 1]] %*% t(self$weights[[layer + 1]])
      }
    }
    else {
        # Single-layer neural network
        cat("Single Layer Backpropagation\n")

        # Check if weights and errors exist
        if (is.null(self$weights[[1]]) || is.null(errors[[1]])) {
            stop("Error: Weights or errors for single layer do not exist.")
        }

        # Ensure weights are explicitly matrices
        weight_matrix <- as.matrix(self$weights[[1]])

        # Debugging: Print dimensions
        weight_dims <- dim(weight_matrix)
        error_dims <- dim(errors[[1]])
        cat("Weights dimensions:\n")
        print(weight_dims)
        cat("Errors dimensions:\n")
        print(error_dims)

        # Ensure dimensions are not NULL
        if (is.null(weight_dims) || is.null(error_dims)) {
            stop("Error: Dimensions for weights or errors are NULL.")
        }

        weight_rows <- weight_dims[1]
        weight_cols <- weight_dims[2]
        error_rows <- error_dims[1]
        error_cols <- error_dims[2]

        # Check and adjust dimensions
        if (error_cols != weight_rows) {
            if (error_cols > weight_rows) {
                errors[[1]] <- errors[[1]][, 1:weight_rows, drop = FALSE]
            } else {
                errors[[1]] <- matrix(
                    rep(errors[[1]], length.out = error_rows * weight_rows),
                    nrow = error_rows,
                    ncol = weight_rows
                )
            }
        }

        if (error_rows != weight_cols) {
            if (error_rows > weight_cols) {
                errors[[1]] <- errors[[1]][1:weight_cols, , drop = FALSE]
            } else {
                errors[[1]] <- matrix(
                    rep(errors[[1]], length.out = weight_cols * ncol(errors[[1]])),
                    nrow = weight_cols,
                    ncol = ncol(errors[[1]])
                )
            }
        }

        # Perform matrix multiplication
        cat("Performing matrix multiplication for single layer\n")
        errors[[1]] <- weight_matrix %*% errors[[1]]
    }













    print("end")
    print(str(errors))


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

    print("------------------------self-organize-end------------------------------------------")

},
# Method to perform learning
learn = function(Rdata, labels, lr, activation_functions_learn, dropout_rates_learn) {
  print("------------------------learn-begin-------------------------------------------------")
  start_time <- Sys.time()
  
  if (!is.numeric(labels) || !is.matrix(labels) || ncol(labels) != 1) {
    labels <- matrix(as.numeric(labels), ncol = 1)
  }
  
  # ----- Class Weights Based on Label Frequency -----
  num_pos <- sum(labels == 1)
  num_neg <- sum(labels == 0)
  total_samples <- num_pos + num_neg
  
  # Inverse frequency: weight more for minority class
  # pos_weight <- total_samples / (2 * num_pos)
  # neg_weight <- total_samples / (2 * num_neg)
  
  pos_weight <- 2
  neg_weight <- 1
  
  
  # Build sample weight vector
  sample_weights <- ifelse(labels == 1, pos_weight, neg_weight)
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
  
  if (self$ML_NN) {
    hidden_outputs <- vector("list", self$num_layers)
    activation_derivatives <- vector("list", self$num_layers)
    dim_hidden_layers_learn <- vector("list", self$num_layers)
    input_matrix <- as.matrix(Rdata)
    
    for (layer in 1:self$num_layers) {
      weights_matrix <- as.matrix(self$weights[[layer]])
      bias_vec <- as.numeric(unlist(self$biases[[layer]]))
      input_data <- if (layer == 1) input_matrix else hidden_outputs[[layer - 1]]
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
      
      cat(sprintf("[Debug] Layer %d : Z summary BEFORE clipping:\n", layer))
      print(summary(as.vector(Z)))
      

      
      activation_function <- activation_functions_learn[[layer]]
      activation_name <- if (!is.null(activation_function)) attr(activation_function, "name") else "none"
      cat(sprintf("[Debug] Layer %d : Activation Function = %s\n", layer, activation_name))
      
      
      
      # ----- Clipping Extreme Z Values (Activation-Aware) -----
      clip_limit <- switch(activation_name,
                           "sigmoid" = 50,
                           "tanh" = 10,
                           "softmax" = 15,
                           "relu" = 100,
                           "leaky_relu" = 100,
                           300 # Default fallback
      )
      
      Z_max <- max(Z)
      Z_min <- min(Z)
      if (Z_max > clip_limit || Z_min < -clip_limit) {
        cat(sprintf("[Debug] Layer %d : Z clipped. Pre-clip range: [%.2f, %.2f] | Clip limit: ±%g\n", layer, Z_min, Z_max, clip_limit))
      }
      Z <- pmin(pmax(Z, -clip_limit), clip_limit)
      
      Z <- Z / 4
      
      
      cat(sprintf("[Debug] Layer %d : Z summary AFTER clipping:\n", layer))
      print(summary(as.vector(Z)))
      
      
      hidden_output <- if (!is.null(activation_function)) activation_function(Z) else Z
      
      if (is.list(self$dropout_rates_learn) &&
          length(self$dropout_rates_learn) >= layer &&
          !is.null(self$dropout_rates_learn[[layer]]) &&
          self$dropout_rates_learn[[layer]] > 0 &&
          self$dropout_rates_learn[[layer]] < 1) {
        
        cat(sprintf("\n[Debug] Layer %d : Hidden output BEFORE dropout (sample):\n", layer))
        print(head(hidden_outputs[[layer]], 3))
        
        hidden_outputs[[layer]] <- self$dropout(hidden_outputs[[layer]], self$dropout_rates_learn[[layer]])
        
        cat(sprintf("[Debug] Layer %d : Hidden output AFTER  dropout (sample):\n", layer))
        print(head(hidden_outputs[[layer]], 3))
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
    
    # ----- Apply dropout using self$dropout for SL NN -----
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
    
    
    # ----- Clipping Extreme Z Values (Activation-Aware) for SL NN -----
    activation_name <- attr(activation_function, "name")
    clip_limit <- switch(activation_name,
                         "sigmoid" = 50,
                         "tanh" = 10,
                         "softmax" = 15,
                         "relu" = 100,
                         "leaky_relu" = 100,
                         300 # Default fallback
    )
    
    Z_max <- max(Z)
    Z_min <- min(Z)
    if (Z_max > clip_limit || Z_min < -clip_limit) {
      cat(sprintf("[Debug] SL NN : Z clipped. Pre-clip range: [%.2f, %.2f] | Clip limit: ±%g\n", Z_min, Z_max, clip_limit))
    }
    Z <- pmin(pmax(Z, -clip_limit), clip_limit)
    
    Z <- Z / 4
    
    
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
  }
  
  learn_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  print("------------------------learn-end-------------------------------------------------")
  
  return(list(learn_output = predicted_output_learn, learn_time = learn_time, error = error_learn, dim_hidden_layers = dim_hidden_layers_learn, hidden_outputs = predicted_output_learn_hidden, grads_matrix = grads_matrix, bias_gradients = bias_gradients))
}




,# Method to perform prediction
predict = function(Rdata, weights, biases, activation_functions) {
  # Ensure weights, biases, activation_functions are lists
  if (!is.list(weights)) weights <- list(weights)
  if (!is.list(biases)) biases <- list(biases)
  if (!is.list(activation_functions)) activation_functions <- list(activation_functions)
  
  start_time <- Sys.time()
  
  output <- as.matrix(Rdata)
  num_layers <- length(weights)
  
  for (layer in seq_len(num_layers)) {
    w <- as.matrix(weights[[layer]])
    b <- as.numeric(unlist(biases[[layer]]))
    
    # Create bias matrix with broadcasting
    n_samples <- nrow(output)
    n_units <- ncol(w)
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
    if (length(activation_functions) >= layer &&
        is.function(activation_functions[[layer]])) {
      output <- activation_functions[[layer]](output)
    }
  }
  
  end_time <- Sys.time()
  prediction_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  return(list(predicted_output = output, prediction_time = prediction_time))
},# Method for training the SONN with L2 regularization
train_with_l2_regularization = function(Rdata, labels, lr, num_epochs, model_iter_num, update_weights, update_biases, ensemble_number, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type) {
  
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
  
  for (epoch in 1:epoch_in_list) {
    # lr <- lr_scheduler(epoch, initial_lr = 0.01, decay_rate = 0.15, decay_epoch = 10)

    # lr <- lr_scheduler(epoch)
    
    # optional: log current lr
    cat("Epoch:", epoch, "| Learning Rate:", lr, "\n")
    # lr <- lr_scheduler(i, initial_lr = lr)
    #print(paste("Epoch:", epoch))
    num_epochs_check <<- num_epochs
    
    # Run forward pass using centralized logic
    # start_time <- Sys.time()
    
    learn_result <- self$learn(
      Rdata = Rdata,
      labels = labels,
      lr = lr,
      activation_functions_learn = activation_functions_learn,
      dropout_rates_learn = dropout_rates_learn
    )
    
    
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
        } else {
          reg_loss <- 0
        }
        
        reg_loss_total <- reg_loss_total + reg_loss
      }
      
    } else {  # --- Single-layer Neural Network ---
      
      reg_loss_total <- 0
      weights_layer <- self$weights[[1]]
      
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
        self$groups <- list(1)  # Treat entire layer as one group
        reg_loss_total <- self$lambda * sum(sapply(self$groups, function(group) {
          sqrt(sum(sapply(group, function(idx) {
            if (idx <= length(self$weights)) {
              sum(self$weights[[idx]]^2, na.rm = TRUE)
            } else {
              0
            }
          })))
        }))
      } else if (reg_type == "Max_Norm") {
        max_norm <- 1.0
        reg_loss_total <- self$lambda * sum(sapply(self$weights, function(w) {
          norm_weight <- sqrt(sum(w^2, na.rm = TRUE))
          if (is.na(norm_weight)) norm_weight <- 0
          if (norm_weight > max_norm) 1 else 0
        }))
      } else if (reg_type == "Sparse_Bayesian") {
        stop("Sparse Bayesian Learning is not implemented in this code.")
      } else {
        stop("Invalid regularization type. Choose 'L1', 'L2', 'L1_L2', 'Group_Lasso', 'Max_Norm', or 'Sparse_Bayesian'.")
      }
    }
    
    

    
    
    # Record the loss for this epoch
    # losses[[epoch]] <- mean(error_last_layer^2) + reg_loss_total
    predictions <- if (self$ML_NN) hidden_outputs[[self$num_layers]] else predicted_output_train_reg$predicted_output
    
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
    

    
    # Update weights
    if (update_weights) {
      if (self$ML_NN) {
        for (layer in 1:self$num_layers) {
          if (!is.null(self$weights[[layer]]) && !is.null(optimizer)) {
            
            # Initialize optimizer parameters if not already
            if (is.null(optimizer_params_weights[[layer]])) {
              optimizer_params_weights[[layer]] <- initialize_optimizer_params(
                optimizer,
                list(dim(self$weights[[layer]])),
                lookahead_step,
                layer
              )
            }
            
            # Get weight gradients from learn()
            grads_matrix <- weight_gradients[[layer]]
            
            # Clip weight gradient
            grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5)
            
            
            # Apply regularization to gradient if L2 or L1_L2
            if (reg_type == "L2" || reg_type == "L1_L2") {
              weight_update <- lr * grads_matrix + self$lambda * self$weights[[layer]]
            } else {
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
               1
             )
           }
           
           grads_matrix <- weight_gradients[[1]]
           
           # --- Regularization ---
           if (reg_type == "L2" || reg_type == "L1_L2") {
             weight_update <- lr * grads_matrix + self$lambda * self$weights
           } else {
             weight_update <- lr * grads_matrix
           }
           
           # --- Debug print ---
           cat(">> SL grads_matrix dim:\n")
           print(dim(grads_matrix))
           cat("SL grads_matrix summary:\n")
           print(summary(as.vector(grads_matrix)))
           
           # --- Apply Optimizer ---
           if (!is.null(optimizer_params_weights[[1]]) && optimizer == "adam") {
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
               layer = 1,
               target = "weights"
             )
             
             self$weights <- updated_optimizer$updated_weights_or_biases
             optimizer_params_weights[[1]] <- updated_optimizer$updated_optimizer_params
             
           } else {
             # Fallback update
             if (all(dim(grads_matrix) == dim(self$weights))) {
               self$weights <- self$weights - weight_update
             } else if (prod(dim(self$weights)) == 1) {
               self$weights <- self$weights - sum(weight_update)
             } else {
               self$weights <- self$weights - apply(weight_update, 2, mean)
             }
           }
         }
       }}
    # Record the updated weight matrix
    if (self$ML_NN) {
      for (layer in 1:self$num_layers) {
        weights_record[[layer]] <- as.matrix(self$weights[[layer]])
      }
    } else {
      weights_record <- as.matrix(self$weights)
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
            grads_matrix <- clip_gradient_norm(grads_matrix)
            
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
            
            # Apply regularization to gradient if L2 or L1_L2
            if (reg_type == "L2" || reg_type == "L1_L2") {
              bias_update <- lr * grads_matrix + self$lambda * self$biases[[layer]]
            } else {
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
                optimizer_params_biases[[layer]] <- rmsprop_update(optimizer_params_biases[[layer]], colSums(errors[[layer]]), lr, beta2, epsilon)
                
                # Accessing the returned values
                updated_biases_update <- optimizer_params_biases[[layer]]$updates
                
                # Convert updated_biases_update to a numeric vector if it's a list
                if (is.list(updated_biases_update)) {
                  updated_biases_update <- unlist(updated_biases_update)
                }
                
                # Checking and updating biases
                if (is.matrix(updated_biases_update) && identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                  cat("Dimensions match for matrix. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                  cat("Dimensions match for vector. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                } else {
                  cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                  cat("Attempting to adjust dimensions if possible.\n")
                  
                  # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                  if (is.vector(updated_biases_update)) {
                    repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                    self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                  } else {
                    cat("Unable to adjust dimensions for updated_biases_update.\n")
                  }
                }
              }
              else if (optimizer == "sgd") {
                # Perform SGD update for the biases
                optimizer_params_biases[[layer]] <- sgd_update(optimizer_params_biases[[layer]], grads_matrix, lr)
                
                # Accessing the returned values
                updated_biases_update <- optimizer_params_biases[[layer]]$biases_update
                
                # Convert updated_biases_update to a numeric vector if it's a list
                if (is.list(updated_biases_update)) {
                  updated_biases_update <- unlist(updated_biases_update)
                }
                
                # Checking and updating biases
                if (is.matrix(updated_biases_update) && identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                  cat("Dimensions match for matrix. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                  cat("Dimensions match for vector. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                } else {
                  cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                  cat("Attempting to adjust dimensions if possible.\n")
                  
                  # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                  if (is.vector(updated_biases_update)) {
                    repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                    self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                  } else {
                    cat("Unable to adjust dimensions for updated_biases_update.\n")
                  }
                }
              }
              else if (optimizer == "sgd_momentum") {
                # Perform SGD update for the biases
                optimizer_params_biases[[layer]] <- sgd_momentum_update(optimizer_params_biases[[layer]], grads_matrix, lr)
                
                # Accessing the returned values
                updated_biases_update <- optimizer_params_biases[[layer]]$biases_update
                
                # Convert updated_biases_update to a numeric vector if it's a list
                if (is.list(updated_biases_update)) {
                  updated_biases_update <- unlist(updated_biases_update)
                }
                
                # Checking and updating biases
                if (is.matrix(updated_biases_update) && identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                  cat("Dimensions match for matrix. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                  cat("Dimensions match for vector. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                } else {
                  cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                  cat("Attempting to adjust dimensions if possible.\n")
                  
                  # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                  if (is.vector(updated_biases_update)) {
                    repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                    self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                  } else {
                    cat("Unable to adjust dimensions for updated_biases_update.\n")
                  }
                }
              }else if (optimizer == "nag") {
                # Compute the gradients for biases if not already available
                # Example: grads_bias <- lapply(self$biases, function(bias) {
                #     # Compute gradients for biases (this is an example and should be replaced with actual gradient calculations)
                #     grad <- matrix(runif(n = length(bias)), nrow = nrow(bias), ncol = ncol(bias))
                #     grad
                # })
                grads_bias <- lapply(self$biases, function(bias) {
                  if (is.null(dim(bias))) {
                    # Convert vectors to matrices with a single column
                    matrix(bias, nrow = length(bias), ncol = 1)
                  } else {
                    bias  # Already a matrix
                  }
                })
                
                # Perform NAG update for the biases
                optimizer_params_biases[[layer]] <- nag_update(optimizer_params_biases[[layer]], grads_bias, lr, beta = 0.9)
                
                # Accessing the returned values
                updated_biases_update <- optimizer_params_biases[[layer]]$biases_update
                
                # Convert updated_biases_update to a numeric vector if it's a list
                if (is.list(updated_biases_update)) {
                  updated_biases_update <- unlist(updated_biases_update)
                }
                
                # Checking and updating biases
                if (is.matrix(updated_biases_update) && identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                  cat("Dimensions match for matrix. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                  cat("Dimensions match for vector. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                } else {
                  cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                  cat("Attempting to adjust dimensions if possible.\n")
                  
                  # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                  if (is.vector(updated_biases_update)) {
                    repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                    self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                  } else {
                    cat("Unable to adjust dimensions for updated_biases_update.\n")
                  }
                }
              }else if (optimizer == "ftrl") {
                # Compute the gradients for biases if not already available
                # Example: grads_bias <- lapply(self$biases, function(bias) {
                #     # Compute gradients for biases (this is an example and should be replaced with actual gradient calculations)
                #     grad <- matrix(runif(n = length(bias)), nrow = nrow(bias), ncol = ncol(bias))
                #     grad
                # })
                grads_bias <- lapply(self$biases, function(bias) {
                  if (is.null(dim(bias))) {
                    # Convert vectors to matrices with a single column
                    matrix(bias, nrow = length(bias), ncol = 1)
                  } else {
                    bias  # Already a matrix
                  }
                })
                
                # Perform FTRL update for the biases
                ftrl_results <- ftrl_update(optimizer_params_biases[[layer]], grads_bias, lr, alpha = 0.1, beta = 1.0, lambda1 = 0.01, lambda2 = 0.01)
                optimizer_params_biases[[layer]] <- ftrl_results$params
                updated_biases_update <- ftrl_results$biases_update
                
                # Ensure updated_biases_update is not NULL
                if (is.null(updated_biases_update)) {
                  stop("Updated biases update is NULL.")
                }
                
                # Ensure updated_biases_update is a matrix or vector
                if (is.list(updated_biases_update)) {
                  updated_biases_update <- unlist(updated_biases_update)
                }
                
                # Check the type of updated_biases_update
                if (is.matrix(updated_biases_update)) {
                  # If updated_biases_update is a matrix
                  if (identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                    self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                  } else {
                    # Adjust dimensions to match if necessary
                    if (nrow(self$biases[[layer]]) != nrow(updated_biases_update)) {
                      if (nrow(self$biases[[layer]]) > nrow(updated_biases_update)) {
                        updated_biases_update <- rbind(updated_biases_update, matrix(0, nrow = nrow(self$biases[[layer]]) - nrow(updated_biases_update), ncol = ncol(updated_biases_update)))
                      } else if (nrow(self$biases[[layer]]) < nrow(updated_biases_update)) {
                        updated_biases_update <- updated_biases_update[1:nrow(self$biases[[layer]]), , drop = FALSE]
                      }
                    }
                    if (ncol(self$biases[[layer]]) != ncol(updated_biases_update)) {
                      if (ncol(self$biases[[layer]]) > ncol(updated_biases_update)) {
                        updated_biases_update <- cbind(updated_biases_update, matrix(0, nrow = nrow(updated_biases_update), ncol = ncol(self$biases[[layer]]) - ncol(updated_biases_update)))
                      } else if (ncol(self$biases[[layer]]) < ncol(updated_biases_update)) {
                        updated_biases_update <- updated_biases_update[, 1:ncol(self$biases[[layer]]), drop = FALSE]
                      }
                    }
                    self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                  }
                } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                  # If updated_biases_update is a vector
                  self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                } else {
                  cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                  cat("Attempting to adjust dimensions if possible.\n")
                  
                  # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                  if (is.vector(updated_biases_update)) {
                    repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                    self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                  } else {
                    cat("Unable to adjust dimensions for updated_biases_update.\n")
                  }
                }
              }
              else if (optimizer == "lamb"){
                # Compute the gradients for biases if not already available
                grads_bias <- lapply(self$biases, function(bias) {
                  grad <- matrix(runif(n = length(bias)), nrow = length(bias), ncol = 1)
                  as.numeric(grad)  # Ensure the gradient is numeric
                })
                
                # Perform LAMB update for the biases
                for (layer in seq_along(self$biases)) {
                  # Ensure grads_bias[[layer]] and self$biases[[layer]] are matrices or vectors of the same dimension
                  grads_bias[[layer]] <- as.numeric(grads_bias[[layer]])
                  
                  # Ensure params have valid numeric matrices or vectors
                  params <- optimizer_params_biases[[layer]]
                  params$param <- as.numeric(params$param)
                  params$m <- as.numeric(params$m)
                  params$v <- as.numeric(params$v)
                  
                  optimizer_params_biases[[layer]] <- lamb_update(params, grads_bias[[layer]], lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, lambda = 0.01)
                  updated_biases_update <- optimizer_params_biases[[layer]]$update
                  
                  # Ensure updated_biases_update is not NULL
                  if (is.null(updated_biases_update)) {
                    stop("Updated biases update is NULL.")
                  }
                  
                  # Check the type of updated_biases_update
                  if (is.matrix(updated_biases_update)) {
                    # If updated_biases_update is a matrix
                    if (identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                      self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                    } else {
                      # Adjust dimensions to match if necessary
                      if (nrow(self$biases[[layer]]) != nrow(updated_biases_update)) {
                        if (nrow(self$biases[[layer]]) > nrow(updated_biases_update)) {
                          updated_biases_update <- rbind(updated_biases_update, matrix(0, nrow = nrow(self$biases[[layer]]) - nrow(updated_biases_update), ncol = ncol(updated_biases_update)))
                        } else if (nrow(self$biases[[layer]]) < nrow(updated_biases_update)) {
                          updated_biases_update <- updated_biases_update[1:nrow(self$biases[[layer]]), , drop = FALSE]
                        }
                      }
                      if (ncol(self$biases[[layer]]) != ncol(updated_biases_update)) {
                        if (ncol(self$biases[[layer]]) > ncol(updated_biases_update)) {
                          updated_biases_update <- cbind(updated_biases_update, matrix(0, nrow = nrow(updated_biases_update), ncol = ncol(self$biases[[layer]]) - ncol(updated_biases_update)))
                        } else if (ncol(self$biases[[layer]]) < ncol(updated_biases_update)) {
                          updated_biases_update <- updated_biases_update[, 1:ncol(self$biases[[layer]]), drop = FALSE]
                        }
                      }
                      self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                    }
                  } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                    # If updated_biases_update is a vector
                    self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                  } else {
                    cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                    cat("Attempting to adjust dimensions if possible.\n")
                    
                    # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                    if (is.vector(updated_biases_update)) {
                      repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                      self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                    } else {
                      cat("Unable to adjust dimensions for updated_biases_update.\n")
                    }
                  }
                }
              }else if (optimizer == "lookahead") {
                # Ensure gradients for biases are properly initialized
                grads_biases <- lapply(self$biases, function(biases) {
                  grad_vector <- runif(n = length(biases))  # Generate random gradients
                  list(param = grad_vector)  # Ensure 'param' element is included
                })
                
                # Perform Lookahead update for the biases
                for (layer in seq_along(self$biases)) {
                  # Ensure 'param' element is present in grads_list
                  grads_list <- list(param = grads_biases[[layer]]$param)
                  
                  params <- list(
                    param = self$biases[[layer]],
                    m = optimizer_params_biases[[layer]]$m,
                    v = optimizer_params_biases[[layer]]$v,
                    r = optimizer_params_biases[[layer]]$r,
                    slow_weights = optimizer_params_biases[[layer]]$slow_biases,
                    lookahead_counter = optimizer_params_biases[[layer]]$lookahead_counter,
                    lookahead_step = optimizer_params_biases[[layer]]$lookahead_step
                  )
                  
                  optimizer_params_biases[[layer]] <- lookahead_update(
                    list(params),  # Wrap params in a list
                    list(grads_list),  # Wrap grads_list in a list
                    lr,
                    beta1 = 0.9,
                    beta2 = 0.999,
                    epsilon = 1e-8,
                    lookahead_step,
                    base_optimizer = "adam_update",  # Use "adam" for biases
                    t = epoch,
                    lambda
                  )
                  
                  
                  
                  
                  
                  
                  # Extract updated biases and other parameters
                  updated_biases_update <- optimizer_params_biases[[layer]]$param
                  new_m <- optimizer_params_biases[[layer]]$m
                  new_v <- optimizer_params_biases[[layer]]$v
                  new_r <- optimizer_params_biases[[layer]]$r
                  new_slow_biases <- optimizer_params_biases[[layer]]$slow_biases
                  new_lookahead_counter <- optimizer_params_biases[[layer]]$lookahead_counter
                  
                  # Ensure updated_biases_update is not NULL
                  if (is.null(updated_biases_update)) {
                    cat("Updated biases update is NULL.\n")
                    next  # Skip to the next layer
                  }
                  
                  # Check the type of updated_biases_update
                  if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                    # If updated_biases_update is a vector with matching length
                    self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                  } else {
                    cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                    cat("Attempting to adjust dimensions if possible.\n")
                    
                    # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                    if (is.matrix(updated_biases_update)) {
                      repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                      self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                    } else {
                      cat("Unable to adjust dimensions for updated_biases_update.\n")
                    }
                  }
                  
                  # Update optimizer parameters
                  optimizer_params_biases[[layer]]$m <- new_m
                  optimizer_params_biases[[layer]]$v <- new_v
                  optimizer_params_biases[[layer]]$r <- new_r
                  optimizer_params_biases[[layer]]$slow_biases <- new_slow_biases
                  optimizer_params_biases[[layer]]$lookahead_counter <- new_lookahead_counter
                }
              }
              
              
              
              
              
              
              
              
              else if (optimizer == "adagrad") {
                optimizer_params_biases[[layer]] <- adagrad_update(optimizer_params_biases[[layer]], colSums(errors[[layer]]), lr)
                
                # Accessing the returned values
                r_values <- optimizer_params_biases[[layer]]$r
                
                # Ensure r_values is numeric
                if (is.list(r_values)) {
                  r_values <- unlist(r_values)
                }
                
                # Calculate updated_biases_update
                updated_biases_update <- r_values / (sqrt(r_values) + epsilon)
                
                # Convert updated_biases_update to a numeric vector if it's a list
                if (is.list(updated_biases_update)) {
                  updated_biases_update <- unlist(updated_biases_update)
                }
                
                # Checking and updating biases
                if (is.matrix(updated_biases_update) && identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                  cat("Dimensions match for matrix. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                  cat("Dimensions match for vector. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                } else {
                  cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                  cat("Attempting to adjust dimensions if possible.\n")
                  
                  # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                  if (is.vector(updated_biases_update)) {
                    repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                    self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                  } else {
                    cat("Unable to adjust dimensions for updated_biases_update.\n")
                  }
                }
              }
              else if (optimizer == "adadelta") {
                optimizer_params_biases[[layer]] <- adadelta_update(optimizer_params_biases[[layer]], colSums(errors[[layer]]), lr)
                
                # Accessing the returned values
                delta_b <- optimizer_params_biases[[layer]]$delta_w
                
                # Ensure delta_b is numeric
                if (is.list(delta_b)) {
                  delta_b <- unlist(delta_b)
                }
                
                # Check if delta_b is numeric
                if (!is.numeric(delta_b)) {
                  stop("delta_b contains non-numeric values. Please check the adadelta_update function.")
                }
                
                # Calculate updated_biases_update
                updated_biases_update <- delta_b / (sqrt(delta_b) + epsilon)
                
                # Convert updated_biases_update to a numeric vector if it's a list
                if (is.list(updated_biases_update)) {
                  updated_biases_update <- unlist(updated_biases_update)
                }
                
                # Checking and updating biases
                if (is.matrix(updated_biases_update) && identical(dim(self$biases[[layer]]), dim(t(updated_biases_update)))) {
                  cat("Dimensions match for matrix. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - t(updated_biases_update)
                } else if (is.vector(updated_biases_update) && length(updated_biases_update) == length(self$biases[[layer]])) {
                  cat("Dimensions match for vector. Performing subtraction.\n")
                  self$biases[[layer]] <- self$biases[[layer]] - updated_biases_update
                } else {
                  cat("Dimensions or type of updated_biases_update are not suitable for subtraction.\n")
                  cat("Attempting to adjust dimensions if possible.\n")
                  
                  # Attempting to adjust dimensions if updated_biases_update is not directly suitable
                  if (is.vector(updated_biases_update)) {
                    repeated_updated_biases_update <- rep(updated_biases_update, length.out = length(self$biases[[layer]]))
                    self$biases[[layer]] <- self$biases[[layer]] - repeated_updated_biases_update
                  } else {
                    cat("Unable to adjust dimensions for updated_biases_update.\n")
                  }
                }
              }
              
              
            } }}}else {
              cat("Single Layer Bias Update\n")
              
              # Check that self$biases is valid
              if (is.null(self$biases)) {
                stop("Biases are NULL in single-layer mode.")
              }
              
              # Convert to matrix if not already
              if (!is.matrix(self$biases)) {
                self$biases <- matrix(self$biases, nrow = 1)
              }
              
              # Initialize optimizer parameters if needed
              if (!is.null(optimizer) && is.null(optimizer_params_biases)) {
                optimizer_params_biases <- initialize_optimizer_params(
                  optimizer,
                  list(dim(self$biases)),
                  lookahead_step,
                  layer
                )
              }
              
              # Compute mean error across rows (for each neuron)
              bias_update <- colMeans(errors[[1]], na.rm = TRUE)
              
              # Reshape bias update to match self$biases dimensions
              if (length(bias_update) != ncol(self$biases)) {
                bias_update <- matrix(rep(bias_update, length.out = ncol(self$biases)), nrow = 1)
              } else {
                bias_update <- matrix(bias_update, nrow = 1)
              }
              
              # Debug prints
              cat("Bias update dimensions:", dim(bias_update), "\n")
              cat("Biases dimensions before update:", dim(self$biases), "\n")
              
              # Use optimizer if specified
              if (!is.null(optimizer)) {
                updated_optimizer <- apply_optimizer_update(
                  optimizer = optimizer,
                  optimizer_params = optimizer_params_biases,
                  grads_matrix = bias_update,
                  lr = lr,
                  beta1 = beta1,
                  beta2 = beta2,
                  epsilon = epsilon,
                  epoch = epoch,
                  self = self,
                  layer = layer,
                  target = "biases"
                )
                
                self$biases <- updated_optimizer$updated_weights_or_biases
                optimizer_params_biases <- updated_optimizer$updated_optimizer_params
              } else {
                # Fallback to SGD update if no optimizer
                self$biases <- self$biases - lr * bias_update
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
    
    
  }                # Convert weights_record and biases_record to matrices if necessary
  # weights_record <- lapply(weights_record, as.matrix)
  # biases_record <- lapply(biases_record, as.matrix)
  
  
  predicted_output_train_reg <- self$predict(Rdata, weights = weights_record, biases = biases_record, activation_functions)
  
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
  
  
  
  # Dynamic assignment of weights and biases records
  for (i in 1:length(self$ensemble)) {
    weight_record_name <- paste0("weights_record_", i)
    bias_record_name <- paste0("biases_record_", i)
    assign(weight_record_name, as.matrix(self$weights), envir = .GlobalEnv)
    assign(bias_record_name, as.matrix(self$biases), envir = .GlobalEnv)
  }
  
  
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
  
  # print("print(weights_record)")
  # print(weights_record)
  # print("print(biases_record)")
  # print(biases_record)
  # assign("loss_status", 'ok', envir = .GlobalEnv)
  # Assign a value to lossesatoptimalepoch
  lossesatoptimalepoch <- losses[optimal_epoch]
  
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
      
      # Save the plot
      saveRDS(plot_epochs, paste("plot_epochs_DESONN", ensemble_number, "SONN", model_iter_num, ".rds", sep = ""))
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
  return(list(predicted_output_l2 = predicted_output_train_reg, train_reg_prediction_time = predicted_output_train_reg_prediction_time, training_time = training_time, optimal_epoch = optimal_epoch, weights_record = weights_record, biases_record = biases_record, lossesatoptimalepoch = lossesatoptimalepoch, loss_increase_flag = loss_increase_flag, loss_status = loss_status, dim_hidden_layers = dim_hidden_layers))
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
    initialize = function(num_networks, input_size, hidden_sizes, output_size, N, lambda, ensemble_number, ML_NN, method = init_method, custom_scale = custom_scale) {
      
      
      # Initialize an ensemble of SONN networks
      self$ensemble <- lapply(1:num_networks, function(i) {
        # Determine ensemble and model names
        if (hyperparameter_grid_setup) {
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
              weights_record_extract <- vector("list", length = length(run_result$weights_record))
              biases_record_extract <- vector("list", length = length(run_result$biases_record))
              
              # Extract and unlist the first element of weights_record and biases_record
              weights_record_extract[[1]] <- unlist(official_run_results_1_1$weights_record[[1]][[1]])
              biases_record_extract[[1]] <- unlist(official_run_results_1_1$biases_record[[1]][[1]])
              
              # Check if ML_NN is TRUE before accessing weights and biases for additional layers
              if (ML_NN == TRUE) {
                for (k in 2:(length(hidden_sizes)+1)) {
                  weights_record_extract[[k]] <- unlist(official_run_results_1_1$weights_record[[1]][[k]])
                  biases_record_extract[[k]] <- unlist(official_run_results_1_1$biases_record[[1]][[k]])
                }
              }
            } else if (!is.null(run_result$best_model_metadata$weights_record) &&
                       !is.null(run_result$best_model_metadata$biases_record) &&
                       !is.null(run_result$metadata)) {
              # Initialize lists within lists for weights and biases
              weights_record_extract <- vector("list", length = length(run_result$best_model_metadata$weights_record))
              biases_record_extract <- vector("list", length = length(run_result$best_model_metadata$biases_record))
              
              # Extract and unlist the first element of weights_record and biases_record from best_model_metadata
              weights_record_extract[[1]] <- unlist(run_result$best_model_metadata$weights_record[[1]][[1]])
              biases_record_extract[[1]] <- unlist(run_result$best_model_metadata$biases_record[[1]][[1]])
              
              # Check if ML_NN is TRUE before accessing weights and biases for additional layers
              if (ML_NN == TRUE) {
                for (k in 2:(length(hidden_sizes)+1)) {
                  weights_record_extract[[k]] <- unlist(run_result$best_model_metadata$weights_record[[1]][[k]])
                  biases_record_extract[[k]] <- unlist(run_result$best_model_metadata$biases_record[[1]][[k]])
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
      if(!hyperparameter_grid_setup){
        ensembles$main_ensemble <- list(
          run_results_1_1
          # run_results_1_2,
          # run_results_1_3,
          # run_results_1_4,
          # run_results_1_5
        )
        ensembles$temp_ensemble <- vector("list", length(self$ensemble))
      }
      
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
    
    
train = function(Rdata, labels, lr, ensemble_number, num_epochs, threshold, reg_type, numeric_columns, activation_functions_learn, activation_functions, dropout_rates_learn, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, batch_normalize_data, gamma_bn = NULL, beta_bn = NULL, epsilon_bn = 1e-5, momentum_bn = 0.9, is_training_bn = TRUE, shuffle_bn = FALSE, loss_type) {
      
      
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
      all_predicted_outputAndTime <- vector("list", length(self$ensemble))
      all_predicted_outputs_learn <- vector("list", length(self$ensemble))
      all_predicted_outputs <- vector("list", length(self$ensemble))
      all_prediction_times <- vector("list", length(self$ensemble))
      all_learn_times <- vector("list", length(self$ensemble))
      all_ensemble_name_model_name <- vector("list", length(self$ensemble))
      all_model_iter_num <- vector("list", length(self$ensemble))
      
      if (never_ran_flag == TRUE) {
        for (i in 1:length(self$ensemble)) {
          # Add Ensemble and Model names to performance_list
          ensemble_name <- attr(self$ensemble[[i]], "ensemble_name")
          model_name <- attr(self$ensemble[[i]], "model_name")
          
          ensemble_name_model_name <- paste("Ensemble:", ensemble_name, "Model:", model_name)
          model_iter_num <- i
          
          
          all_ensemble_name_model_name[[i]] <- ensemble_name_model_name
          
          all_model_iter_num[[i]] <- model_iter_num
          self$ensemble[[i]]$self_organize(Rdata, labels, lr)
          if (learnOnlyTrainingRun == FALSE) {
            # learn_results <- self$ensemble[[i]]$learn(Rdata, labels, lr, activation_functions_learn, dropout_rates_learn)
            predicted_outputAndTime <- suppressMessages(invisible(
              self$ensemble[[i]]$train_with_l2_regularization(
                Rdata, labels, lr, num_epochs, model_iter_num, update_weights, update_biases, ensemble_number, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type
              ))) #<<-

            # --- PATCH: If predicted_output_l2 is missing, construct it manually ---
            # if (is.null(predicted_outputAndTime$predicted_output_l2)) {
            #   predicted_outputAndTime$predicted_output_l2 <- list()
            #   predicted_outputAndTime$predicted_output_l2$predicted_output <- predicted_outputAndTime$hidden_outputs[1:self$num_layers]
            #   predicted_outputAndTime$dim_hidden_layers <- lapply(weights_record, dim)
            # } else {
            #   # Otherwise, clean up NULL entries in the output
            #   valid_outputs <- Filter(Negate(is.null), predicted_outputAndTime$predicted_output_l2$predicted_output)
            #   valid_dims <- lapply(seq_along(valid_outputs), function(i) dim(self$weights[[i]]))
            #   predicted_outputAndTime$predicted_output_l2$predicted_output <- valid_outputs
            #   predicted_outputAndTime$dim_hidden_layers <- valid_dims
            # }
            
            # --- Final integrity check ---
            # stopifnot(length(predicted_outputAndTime$predicted_output_l2$predicted_output) == length(predicted_outputAndTime$dim_hidden_layers))
          
            
            # At the end of the training process, call the predict function
            # trained_predictions <<- self$predict(Rdata, labels, activation_functions)
            # print(dim(labels))
            predicted_outputAndTime$loss_status <- 'exceeds_10000'
            calculate_accuracy <- function(predictions, actual_labels) {
              correct_predictions <- sum(predictions == actual_labels)
              accuracy <- (correct_predictions / length(actual_labels)) * 100
              return(accuracy)
            }
            
            cat("Str in predicted_outputAndTime:\n")
            print(str(predicted_outputAndTime))
            
            cat("str of predicted_output:\n")
            print(str(predicted_outputAndTime$predicted_output))
            
            if (!is.null(predicted_outputAndTime$predicted_output_l2)) {
              cat("str predicted_output_l2$predicted_output:\n")
              print(str(predicted_outputAndTime$predicted_output_l2$predicted_output))
            }
            
            probs <- predicted_outputAndTime$predicted_output_l2$predicted_output
            # binary_preds <- probs
            # binary_preds <- ifelse(probs >= 0.5, 1, 0)
            # binary_preds <- ifelse(probs >= 0.1, 1, 0)
            
            
            # === Auto-tune threshold based on F1 ===
            threshold_result <- tune_threshold(probs, labels)
            best_threshold <- threshold_result$best_threshold
            print(paste("Best F1 threshold:", best_threshold))
            
            # === Binary prediction using best threshold ===
            predict_with_threshold <- function(probs, threshold) {
              ifelse(probs >= threshold, 1, 0)
            }
            
            binary_preds <- predict_with_threshold(probs, best_threshold)
            
            # === Accuracy Evaluation ===
            labels_flat <- as.vector(labels)
            accuracy <- calculate_accuracy(binary_preds, labels_flat)
            print(paste("Accuracy:", accuracy))
            
            # === Precision / Recall / F1 ===
            metrics <- evaluate_classification_metrics(binary_preds, labels_flat)
            print(metrics)
            
            # === Identify Misclassified Samples ===
            wrong <- which(binary_preds != labels_flat)
            
            misclassified <- data.frame(
              predicted_prob = probs[wrong],
              predicted_label = binary_preds[wrong],
              actual_label = labels_flat[wrong],
              X_validation[wrong, ]
            )
            
            # === Combine Full Rdata, Labels, Predictions ===
            Rdata_df <- as.data.frame(Rdata)
            Rdata_with_labels <- cbind(Rdata_df, Label = labels_flat)
            Rdata_predictions <- Rdata_with_labels %>%
              mutate(Predictions = binary_preds)
            
            # === Export to Excel ===
            library(writexl)
            write_xlsx(Rdata_predictions, "Rdata_predictions.xlsx")
            write_xlsx(misclassified, "misclassified_cases.xlsx")
            
            # Specify the path
            file_path <- "C:/Users/wfky1/OneDrive/Documents/R/DESONN/Rdata_predictions.xlsx"

            # Write the data to the specified folder
            write_xlsx(Rdata_predictions, file_path)
            stop()
            if(ML_NN){
              
              # Step 1: Extract predicted output
              predicted_output_raw <- predicted_outputAndTime$predicted_output_l2$predicted_output
              cat("Type of predicted_output:", typeof(predicted_output_raw), "\n")
              cat("Is list of layers?", is.list(predicted_output_raw), "\n")
              
              # Step 2: Normalize predicted output to list format
              if (!is.list(predicted_output_raw)) {
                all_layer_outputs <- list(predicted_output_raw)
              } else {
                all_layer_outputs <- predicted_output_raw
              }
              
              # Step 3: Extract dim_hidden_sizes
              dim_hidden_sizes <- predicted_outputAndTime$dim_hidden_layers
              if (is.null(dim_hidden_sizes) || length(dim_hidden_sizes) == 0 || all(sapply(dim_hidden_sizes, is.null))) {
                dim_hidden_sizes <- lapply(predicted_outputAndTime$weights_record, function(w) {
                  if (!is.null(w)) dim(w) else NULL
                })
                cat("dim_hidden_sizes reconstructed from weights_record\n")
              }
              
              # Debug structure lengths
              cat("Post-wrap: all_layer_outputs length = ", length(all_layer_outputs), "\n")
              cat("Post-wrap: dim_hidden_sizes length = ", length(dim_hidden_sizes), "\n")
              
              # Ensure valid types
              valid_outputs_logical <- sapply(all_layer_outputs, function(x) is.matrix(x) || is.vector(x))
              valid_dims_logical <- sapply(dim_hidden_sizes, function(x) is.numeric(x) && is.atomic(x) && length(x) >= 2)
              
              # Pad shorter vector if needed to prevent filtering everything
              len_diff <- length(dim_hidden_sizes) - length(all_layer_outputs)
              if (len_diff > 0) {
                all_layer_outputs <- c(all_layer_outputs, rep(list(NULL), len_diff))
                valid_outputs_logical <- c(valid_outputs_logical, rep(FALSE, len_diff))
              }
              
              # Filter only matched valid indices
              min_len <- min(length(valid_outputs_logical), length(valid_dims_logical))
              valid_indices <- which(valid_outputs_logical[1:min_len] & valid_dims_logical[1:min_len])
              
              # If nothing matched, fallback to first valid output and full dim list
              if (length(valid_indices) == 0 && length(all_layer_outputs) == 1 && length(dim_hidden_sizes) >= 1) {
                all_layer_outputs <- all_layer_outputs[1]
                dim_hidden_sizes <- dim_hidden_sizes
                cat("⚠️ Using fallback path: single output, full dim list\n")
              } else {
                all_layer_outputs <- all_layer_outputs[valid_indices]
                dim_hidden_sizes <- dim_hidden_sizes[valid_indices]
              }
              
              # Final checks
              cat("Filtered dim_hidden_sizes structure:\n")
              str(dim_hidden_sizes)
              cat("Filtered all_layer_outputs length =", length(all_layer_outputs), "\n")
              cat("Filtered dim_hidden_sizes length =", length(dim_hidden_sizes), "\n")
              
              stopifnot(length(all_layer_outputs) == length(dim_hidden_sizes))
              
              # Compute total hidden size
              total_hidden_size <- sum(sapply(dim_hidden_sizes, function(x) x[2]))
              
              
              weighted_sum_output <- NULL
              
              for (l in seq_along(all_layer_outputs)) {
                layer_output <- all_layer_outputs[[l]]
                if (is.null(dim(layer_output))) {
                  layer_output <- matrix(layer_output, ncol = 1)
                }
                
                if (is.null(weighted_sum_output)) {
                  weighted_sum_output <- matrix(0, nrow = nrow(layer_output), ncol = ncol(layer_output))
                }
                
                layer_weight <- dim_hidden_sizes[[l]][2] / total_hidden_size
                weighted_sum_output <- weighted_sum_output + layer_output * layer_weight
              }
              
              
              
              
              
              final_layer_prediction_time <- predicted_outputAndTime$train_reg_prediction_time[[length(predicted_outputAndTime$train_reg_prediction_time)]]
              
              all_predicted_outputAndTime[[i]] <- list(
                predicted_output = weighted_sum_output,
                prediction_time = final_layer_prediction_time,
                training_time = predicted_outputAndTime$training_time,
                optimal_epoch = predicted_outputAndTime$optimal_epoch,
                weights_record = predicted_outputAndTime$weights_record,
                biases_record = predicted_outputAndTime$biases_record,
                losses_at_optimal_epoch = predicted_outputAndTime$lossesatoptimalepoch,
                loss_increase_flag = predicted_outputAndTime$loss_increase_flag
              )
              
              # Store the weighted sum outputs and prediction times for this iteration
              all_predicted_outputs[[i]] <- weighted_sum_output
              all_prediction_times[[i]] <- final_layer_prediction_time
            }else{
              all_predicted_outputAndTime[[i]] <- list(
                predicted_output = predicted_outputAndTime$predicted_output_l2$predicted_output,
                prediction_time = predicted_outputAndTime$predicted_output_l2$prediction_time,
                training_time = predicted_outputAndTime$training_time,
                optimal_epoch = predicted_outputAndTime$optimal_epoch,
                weights_record = predicted_outputAndTime$weights_record,
                biases_record = predicted_outputAndTime$biases_record,
                weights_record2 = predicted_outputAndTime$weights_record2,
                biases_record2 = predicted_outputAndTime$biases_record2,
                losses_at_optimal_epoch = predicted_outputAndTime$lossesatoptimalepoch,
                loss_increase_flag = predicted_outputAndTime$loss_increase_flag
              )
              
              # Store the predictions and prediction times for this iteration
              all_predicted_outputs[[i]] <- predicted_outputAndTime$predicted_output_l2$predicted_output
              all_prediction_times[[i]] <- predicted_outputAndTime$predicted_output_l2$prediction_time
            }
            my_optimal_epoch_out_vector[[i]] <- predicted_outputAndTime$optimal_epoch #<<-
          } else if (learnOnlyTrainingRun == TRUE) {
            learn_results <- self$ensemble[[i]]$learn(Rdata, labels, lr, activation_functions_learn, dropout_rates_learn)
            all_learn_times[[i]] <- learn_results$learn_time
            all_predicted_outputs_learn[[i]] <- learn_results$predicted_output_learn
            
            
          }
        }
        
        if (learnOnlyTrainingRun == FALSE) {
          all_ensemble_name_model_name <- do.call(c, all_ensemble_name_model_name) #<<-
          
          
          if(predicted_outputAndTime$loss_status == 'ok'){
            performance_relevance_plots <- self$update_performance_and_relevance(
              Rdata, labels, lr, ensemble_number, model_iter_num = all_model_iter_num, num_epochs, threshold,
              learn_results = learn_results,
              predicted_output_list = all_predicted_outputs,
              learn_time = NULL,
              prediction_time_list = all_prediction_times, run_id = all_ensemble_name_model_name, all_predicted_outputAndTime = all_predicted_outputAndTime
            )
            return(list(loss_status = loss_status, accuracy = accuracy))
          } else {
            return(list(loss_status = loss_status, accuracy = accuracy))
          }
        } else if (learnOnlyTrainingRun == TRUE) {
          all_learn_times <- do.call(c, all_learn_times) #<<-
          all_predicted_outputs_learn <- do.call(c, all_predicted_outputs_learn) #<<-
          #all_ensemble_name_model_name <<- do.call(c, all_ensemble_name_model_name)
          
          if(predicted_outputAndTime$loss_status == 'ok'){
            performance_relevance_plots <- self$update_performance_and_relevance(
              Rdata, labels, lr, ensemble_number, model_iter_num = all_model_iter_num, num_epochs, threshold,
              learn_results = learn_results,
              predicted_output_list = all_predicted_outputs_learn,
              learn_time = all_learn_times,
              prediction_time_list = NULL, run_id = all_ensemble_name_model_name, all_predicted_outputAndTime = all_predicted_outputAndTime
            )
            return(list(loss_status = loss_status, accuracy = accuracy))
          } else {
            return(list(loss_status = loss_status, accuracy = accuracy))
          }
        }
        
        
      } else if (never_ran_flag == FALSE) {
        for (i in 1:length(self$ensemble)) {
          # Add Ensemble and Model names to performance_list
          ensemble_name <- attr(self$ensemble[[i]], "ensemble_name")
          model_name <- attr(self$ensemble[[i]], "model_name")
          
          ensemble_name_model_name <- paste("Ensemble:", ensemble_name, "Model:", model_name)
          all_ensemble_name_model_name[[i]] <- ensemble_name_model_name
          model_iter_num <- i
          all_model_iter_num[[i]] <- model_iter_num
          
          if (learnOnlyTrainingRun == FALSE) {
            predicted_outputAndTime <- suppressMessages(invisible(
              self$ensemble[[i]]$train_with_l2_regularization(
                Rdata, labels, lr, num_epochs, model_iter_num, update_weights, update_biases, ensemble_number, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type
              ))) #<<-
            
            calculate_accuracy <- function(predictions, actual_labels) {
              correct_predictions <- sum(predictions == actual_labels)
              accuracy <- (correct_predictions / length(actual_labels)) * 100
              return(accuracy)
            }
            
            # Calculate and print the accuracy
            accuracy <- calculate_accuracy(predicted_outputAndTime$predicted_output_l2$predicted_output[[num_layers]], labels)
            
            print(paste("Accuracy:", accuracy))
            
            if(ML_NN){
              
              
              # Extract the outputs and their corresponding hidden sizes (weights)
              all_layer_outputs <- predicted_outputAndTime$predicted_output_l2$predicted_output
              dim_hidden_sizes <- predicted_outputAndTime$dim_hidden_layers
              
              # Compute the sum of all hidden sizes
              total_hidden_size <- sum(sapply(dim_hidden_sizes, function(x) x[2]))
              
              # Loop through each layer output and compute weighted sum
              for (l in seq_along(all_layer_outputs)) {
                layer_output <- all_layer_outputs[[l]]
                
                # Initialize weighted_sum_output with dimensions of the current layer output
                weighted_sum_output <- array(0, dim = dim(layer_output))
                
                # Use the second dimension of the hidden size as the weight, normalized by the total hidden size
                layer_weight <- dim_hidden_sizes[[l]][2] / total_hidden_size
                
                # Compute weighted sum
                weighted_sum_output <- weighted_sum_output + layer_output * layer_weight
                
                # Print dimensions for debugging
                # cat("Dimensions of weighted_sum_output in iteration", i, ":\n")
                # print(dim(weighted_sum_output))
              }
              
              
              
              final_layer_prediction_time <- predicted_outputAndTime$train_reg_prediction_time[[length(predicted_outputAndTime$train_reg_prediction_time)]]
              
              all_predicted_outputAndTime[[i]] <- list(
                predicted_output = weighted_sum_output,
                prediction_time = final_layer_prediction_time,
                training_time = predicted_outputAndTime$training_time,
                optimal_epoch = predicted_outputAndTime$optimal_epoch,
                weights_record = predicted_outputAndTime$weights_record,
                biases_record = predicted_outputAndTime$biases_record,
                losses_at_optimal_epoch = predicted_outputAndTime$lossesatoptimalepoch,
                loss_increase_flag = predicted_outputAndTime$loss_increase_flag
              )
              
              # Store the weighted sum outputs and prediction times for this iteration
              all_predicted_outputs[[i]] <- weighted_sum_output
              all_prediction_times[[i]] <- final_layer_prediction_time
              
            }else{
              all_predicted_outputAndTime[[i]] <- list(
                predicted_output = predicted_outputAndTime$predicted_output_l2$predicted_output,
                prediction_time = predicted_outputAndTime$predicted_output_l2$prediction_time,
                training_time = predicted_outputAndTime$training_time,
                optimal_epoch = predicted_outputAndTime$optimal_epoch,
                weights_record = predicted_outputAndTime$weights_record,
                biases_record = predicted_outputAndTime$biases_record,
                weights_record2 = predicted_outputAndTime$weights_record2,
                biases_record2 = predicted_outputAndTime$biases_record2,
                losses_at_optimal_epoch = predicted_outputAndTime$lossesatoptimalepoch,
                loss_increase_flag = predicted_outputAndTime$loss_increase_flag
              )
              
              # Store the predictions and prediction times for this iteration
              all_predicted_outputs[[i]] <- predicted_outputAndTime$predicted_output_l2$predicted_output
              all_prediction_times[[i]] <- predicted_outputAndTime$predicted_output_l2$prediction_time
            }
            
            if(hyperparameter_grid_setup){
              my_optimal_epoch_out_vector[[i]] <- predicted_outputAndTime$optimal_epoch #<<-
            }
            
            all_ensemble_name_model_name <- do.call(c, all_ensemble_name_model_name) #<<-
            print("outside predict")
            if(predict_models){
              print("inside predict")
              prediction <- self$ensemble[[i]]$predict(Rdata, labels, activation_functions, dropout_rates) #<<-
              
              # Extract the outputs and their corresponding hidden sizes (weights)
              all_layer_predicted_outputs <- prediction$predicted_output #<<-
              dim_hidden_sizes_predicted <- prediction$dim_hidden_layers
              
              # Compute the sum of all hidden sizes
              total_hidden_size_predicted <- sum(sapply(dim_hidden_sizes_predicted, function(x) x[2]))
              
              # Loop through each layer output and compute weighted sum
              for (l in seq_along(all_layer_predicted_outputs)) {
                layer_predicted_output <- all_layer_predicted_outputs[[l]]
                
                # Initialize weighted_sum_predicted_output with dimensions of the current layer output
                weighted_sum_predicted_output <- array(0, dim = dim(layer_predicted_output))
                
                # Use the second dimension of the hidden size as the weight, normalized by the total hidden size
                layer_predicted_weight <- dim_hidden_sizes_predicted[[l]][2] / total_hidden_size_predicted
                
                # Compute weighted sum
                weighted_sum_predicted_output <- weighted_sum_predicted_output + layer_predicted_output * layer_predicted_weight #<<-
                
                # Print dimensions for debugging
                # cat("Dimensions of weighted_sum_output in iteration", i, ":\n")
                # print(dim(weighted_sum_output))
              }
              
              cat("___________________________prediction________________________________\n")
              # print(head(prediction$error))
              # calculate_performance(self$ensemble[[i]], Rdata, labels, lr, model_iter_num, num_epochs, threshold, prediction$predicted_output, prediction$prediction_time)
              # calculate_relevance(self$ensemble[[i]], Rdata, labels, model_iter_num, prediction$predicted_output)
              # print("___________________________prediction_end____________________________\n")
            }
          }
        }
        
        if(predicted_outputAndTime$loss_status == 'ok'){
          #I think this line below could be in the if statement
          performance_relevance_plots <- self$update_performance_and_relevance(
            Rdata, labels, lr, ensemble_number, model_iter_num = all_model_iter_num, num_epochs, threshold,
            learn_results = learn_results,
            predicted_output_list = all_predicted_outputs,
            learn_time = NULL,
            prediction_time_list = all_prediction_times, run_id = all_ensemble_name_model_name, all_predicted_outputAndTime = all_predicted_outputAndTime
          ) #<<-
          return(list(loss_status = loss_status, accuracy = accuracy))
        } else {
          return(list(loss_status = loss_status, accuracy = accuracy))
        }
      }
      
      
      
      if (showMeanBoxPlots == TRUE) {
        print(performance_relevance_plots$performance_high_mean_plots)
        print(performance_relevance_plots$relevance_high_mean_plots)
        print(performance_relevance_plots$relevance_low_mean_plots)
      }
    }, # Method for updating performance and relevance metrics

update_performance_and_relevance = function(Rdata, labels, lr, ensemble_number, model_iter_num, num_epochs, threshold, learn_results, predicted_output_list, learn_time, prediction_time_list, run_id, all_predicted_outputAndTime) {
      
      # Initialize lists to store performance and relevance metrics for each SONN
      performance_list <- list()
      relevance_list <- list()
      model_name_list <-  list()
      #████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
      
      # Calculate performance and relevance for each SONN in the ensemble
      if (never_ran_flag == TRUE){
        
        
        for (i in 1:length(self$ensemble)) {
          
          single_predicted_outputAndTime <- all_predicted_outputAndTime[[i]] #this is for store metadata
          
          single_predicted_output <- predicted_output_list[[i]]
          
          single_ensemble_name_model_name <- all_ensemble_name_model_name[[i]]
          
          if(learnOnlyTrainingRun == FALSE){
            if(hyperparameter_grid_setup){
              cat("___________________________________________________________________________\n")
              cat("______________________________DESONN_", ensemble_number + 1, "_SONN_", i, "______________________________\n", sep = "")
            }else{
              cat("___________________________________________________________________________\n")
              cat("______________________________DESONN_", ensemble_number, "_SONN_", i, "______________________________\n", sep = "")
            }
            single_predicted_output <- predicted_output_list[[i]]
            single_prediction_time <- prediction_time_list[[i]]
            
            performance_list[[i]] <- calculate_performance(self$ensemble[[i]], Rdata, labels, lr, i, num_epochs, threshold, single_predicted_output, single_prediction_time, ensemble_number, single_ensemble_name_model_name) #single_ensemble_name_model_name for quant error
            
            relevance_list[[i]] <- calculate_relevance(self$ensemble[[i]], Rdata, labels, i, single_predicted_output, ensemble_number)
            
            # # Append data frames to lists
            # model_name_list[[i]] <- run_id
            
            performance_metric <- performance_list[[i]]$metrics #<<-
            relevance_metric <- relevance_list[[i]]$metrics #<<-
            
            
          }else if(learnOnlyTrainingRun == TRUE){
            if(hyperparameter_grid_setup){
              cat("___________________________________________________________________________\n")
              cat("______________________________DESONN_", ensemble_number + 1, "_SONN_", i, "______________________________\n", sep = "")
            }else{
              cat("___________________________________________________________________________\n")
              cat("______________________________DESONN_", ensemble_number, "_SONN_", i, "______________________________\n", sep = "")
            }
            single_predicted_output_learn <- all_predicted_outputs_learn[[i]]
            single_learn_time <- all_learn_times[[i]]
            
            performance_list[[i]] <- calculate_performance_learn(self$ensemble[[i]], Rdata, labels, lr, i, num_epochs, threshold, single_predicted_output_learn, single_learn_time, ensemble_number, single_ensemble_name_model_name)
            
            relevance_list[[i]] <- calculate_relevance_learn(self$ensemble[[i]], Rdata, labels, i, single_learn_time, ensemble_number)
            
            performance_metric <- performance_list[[i]]$metrics
            relevance_metric <- relevance_list[[i]]$metrics
            
            self$store_metadata_precursor(run_id = single_ensemble_name_model_name, model_iter_num = i, num_epochs, threshold, single_predicted_output_learn, single_learn_time, actual_values = y)
            
          }
          
          self$store_metadata(run_id = single_ensemble_name_model_name, ensemble_number, model_iter_num = i, num_epochs, threshold, predicted_output = single_predicted_output, actual_values = y, performance_metric = performance_metric, relevance_metric = relevance_metric, predicted_outputAndTime = single_predicted_outputAndTime)
          
        }
      }else if(never_ran_flag == FALSE){
        for (i in 1:length(self$ensemble)) {
          if(hyperparameter_grid_setup){
            cat("___________________________________________________________________________\n")
            cat("______________________________DESONN_", ensemble_number + 1, "_SONN_", i, "______________________________\n", sep = "")
          }else{
            cat("___________________________________________________________________________\n")
            cat("______________________________DESONN_", ensemble_number, "_SONN_", i, "______________________________\n", sep = "")
          }
          single_predicted_outputAndTime <- all_predicted_outputAndTime[[i]] #this is for store metadata
          single_predicted_output <- predicted_output_list[[i]]
          single_prediction_time <- prediction_time_list[[i]]
          single_ensemble_name_model_name <- all_ensemble_name_model_name[[i]]
          
          if(learnOnlyTrainingRun == FALSE){
            
            performance_list[[i]] <- calculate_performance(self$ensemble[[i]], Rdata, labels, lr, i, num_epochs, threshold, single_predicted_output, single_prediction_time, ensemble_number, single_ensemble_name_model_name)
            
            relevance_list[[i]] <- calculate_relevance(self$ensemble[[i]], Rdata, labels, i, single_predicted_output, ensemble_number)
            
          }
          
          performance_metric <- performance_list[[i]]$metrics
          relevance_metric <- relevance_list[[i]]$metrics
          
          
          self$store_metadata(run_id = single_ensemble_name_model_name, ensemble_number, model_iter_num = i, num_epochs, threshold, predicted_output = single_predicted_output, actual_values = y, performance_metric = performance_metric, relevance_metric = relevance_metric, predicted_outputAndTime = single_predicted_outputAndTime)
          
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
      # Function to process performance metrics
      process_performance <- function(metrics_data, model_names) {
        # Initialize variables to store results
        high_mean_df <- NULL
        low_mean_df <- NULL
        
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
            flatten()
          
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
          
          # Attempt to convert "Value" column to numeric, handling warnings
          df$Value <- suppressWarnings(as.numeric(df$Value))
          
          # Filter out NA values
          df <- df[complete.cases(df), ]
          
          # Calculate mean for each metric
          mean_metrics <- df %>%
            group_by(Metric) %>%
            summarise(mean_value = mean(Value, na.rm = TRUE))
          
          # Identify metrics with means greater than 10
          high_mean_metrics <- mean_metrics %>%
            filter(mean_value > 10) %>%
            pull(Metric)
          
          # Filter dataframe for metrics with means greater than 10
          high_mean_df <- bind_rows(high_mean_df, df %>%
                                      filter(Metric %in% high_mean_metrics, !is.na(Value)))
          
          # Filter dataframe for metrics with means less than or equal to 10
          low_mean_df <- bind_rows(low_mean_df, df %>%
                                     filter(!Metric %in% high_mean_metrics, !is.na(Value)))
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
      # print("Finished Performance update_performance_and_relevance_high")
      # print("Finished Performance update_performance_and_relevance_low")
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
      # print("Finished Relevance update_performance_and_relevance_low")
      
      if(hyperparameter_grid_setup){
        if(never_ran_flag == TRUE){
          if (learnOnlyTrainingRun == FALSE) {
            run_results_1_1 <- results_list[[1]] #<<-
            # run_results_1_2 <<- results_list[[2]]
            # run_results_1_3 <<- results_list[[3]]
            # run_results_1_4 <<- results_list[[4]]
            # run_results_1_5 <<- results_list[[5]]
          }else if(learnOnlyTrainingRun == TRUE){
            results_list_learnOnly_1_1 <- results_list_learnOnly[[1]] #<<-
            results_list_learnOnly_1_2 <- results_list_learnOnly[[2]] #<<-
            results_list_learnOnly_1_3 <- results_list_learnOnly[[3]] #<<-
            results_list_learnOnly_1_4 <- results_list_learnOnly[[4]] #<<-
            results_list_learnOnly_1_5 <- results_list_learnOnly[[5]] #<<-
          }}else if(never_ran_flag == FALSE){
            if (learnOnlyTrainingRun == FALSE) {
              run_results_2_1 <- results_list[[1]] #<<-
              run_results_2_2 <- results_list[[2]] #<<-
              run_results_2_3 <- results_list[[3]] #<<-
              run_results_2_4 <- results_list[[4]] #<<-
              run_results_2_5 <- results_list[[5]] #<<-
            }
          }
      }
      
      # Call find_and_print_best_performing_models with appropriate arguments
      find_and_print_best_performing_models(performance_names, relevance_names, performance_metrics, relevance_metrics, target_metric_name_best)
      
      # Call find_and_print_worst_performing_models with appropriate arguments
      find_and_print_worst_performing_models(performance_names, relevance_names, performance_metrics, relevance_metrics, target_metric_name_worst, ensemble_number = j)
      
      # Return the lists of plots
      return(list(performance_high_mean_plots = performance_high_mean_plots, performance_low_mean_plots = performance_low_mean_plots, relevance_high_mean_plots = relevance_high_mean_plots, relevance_low_mean_plots = relevance_low_mean_plots))
      
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
        # Subset the data for the current metric
        
        # Filter out rows where the Value is 0 for metrics containing "precision" or "mean_precision"
        filtered_high_mean_df <- high_mean_df[!(grepl("precision", high_mean_df$Metric, ignore.case = TRUE) & high_mean_df$Value == 0), ]
        
        # Filter out rows where the Value is NA
        filtered_high_mean_df <- filtered_high_mean_df[!is.na(filtered_high_mean_df$Value) & !is.infinite(filtered_high_mean_df$Value), ]
        
        # Subset the data for the current metric
        plot_data_high <- filtered_high_mean_df[filtered_high_mean_df$Metric == metric, ] #<<-
        
        # Check if plot_data is not empty
        if (nrow(plot_data_high) > 0) {
          # Add a column to identify outliers
          plot_data <- plot_data_high %>%
            mutate(Outlier = ifelse(Value %in% self$identify_outliers(Value), Value, NA))
          
          #Add columns for outliers
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
        } else {
          # Print a message if there is no data to plot
          ##print(paste("No data to plot for metric:", metric))
        }
      }
      return(high_mean_plots)
    },
    
    update_performance_and_relevance_low = function(low_mean_df) {
      low_mean_plots <- list()
      # Loop over each unique metric
      for (metric in unique(low_mean_df$Metric)) {
        # Subset the data for the current metric
        
        # Filter out rows where the Value is 0 for metrics containing "precision" or "mean_precision"
        filtered_low_mean_df <- low_mean_df[!(grepl("precision", low_mean_df$Metric, ignore.case = TRUE) & low_mean_df$Value == 0), ]
        
        # Filter out rows where the Value is NA
        filtered_low_mean_df <- filtered_low_mean_df[!is.na(filtered_low_mean_df$Value) & !is.infinite(filtered_low_mean_df$Value), ]
        
        plot_data_low <- filtered_low_mean_df[filtered_low_mean_df$Metric == metric, ] #<<-
        
        # Check if plot_data is not empty
        if (nrow(plot_data_low) > 0) {
          # Add a column to identify outliers
          plot_data <- plot_data_low %>%
            mutate(Outlier = ifelse(Value %in% self$identify_outliers(Value), Value, NA))
          
          #Add columns for outliers
          plot_data$Model_Name_Outlier <- plot_data$Model_Name
          
          
          # Set the RowName to NA where there are no outliers
          plot_data$Model_Name_Outlier[is.na(plot_data$Outlier)] <- NA
          
          
          # Create bin labels for "precisions" or "mean_precisions"
          if (grepl("precision", metric, ignore.case = TRUE)) {
            plot_data$Title <- paste0("Boxplot for ", metric, " (", self$create_bin_labels(plot_data$Value), ")")
          } else {
            plot_data$Title <- paste("Boxplot for", metric) #, " Ensemble Number: ", ensemble_name)
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
        } else {
          # Print a message if there is no data to plot
          ##print(paste("No data to plot for metric:", metric))
        }
      }
      return(low_mean_plots)
      #},
      #         # Method for predicting output values
      #         predict = function(Rdata) {
      #             # Use the ensemble of SONNs to predict output values
      #             predictions <- lapply(self$ensemble, function(SONN) {
      #                 SONN$predict(Rdata)
      #             })
      #             # Combine the predictions from each SONN
      #             # (Implementation details omitted for brevity)
      #             return(combined_predictions)
      #         },
      #         # Method for predicting output values using weighted averaging
      #         predict_weighted_average = function(Rdata) {
      #             predictions <- lapply(self$ensemble, function(SONN) {
      #                 SONN$predict(Rdata)
      #             })
      #
      #             # Calculate the performance-weighted average of the predictions
      #             weighted_average_predictions <- rowMeans(sapply(seq_along(predictions), function(i) {
      #                 predictions[[i]] * self$performance[i]
      #             }))
      #
      #             return(weighted_average_predictions)
    },
    #This is for NvrRan TRUE and LearnOnlyTraining TRUE
    store_metadata_precursor = function(run_id, model_iter_num, num_epochs, threshold, single_predicted_output_learn, single_learn_time, actual_values) {
      
      if (ncol(actual_values) != ncol(single_predicted_output_learn)) {
        if (ncol(single_predicted_output_learn) < ncol(actual_values)) {
          # Calculate the required replication factor
          total_elements_needed <- nrow(actual_values) * ncol(actual_values)
          rep_factor <- ceiling(total_elements_needed / length(single_predicted_output_learn))
          
          # Create the replicated vector and check its length
          replicated_predicted_output <- rep(single_predicted_output_learn, rep_factor)
          # Truncate the replicated vector to match the required length
          replicated_predicted_output <- replicated_predicted_output[1:total_elements_needed]
          # Create the matrix and check its dimensions
          predicted_output_matrix <- matrix(replicated_predicted_output, nrow = nrow(actual_values), ncol = ncol(actual_values), byrow = FALSE)
        } else {
          # Truncate single_predicted_output_learn to match the number of columns in actual_values
          truncated_predicted_output <- single_predicted_output_learn[, 1:ncol(actual_values)]
          # Create the matrix and check its dimensions
          predicted_output_matrix <- matrix(truncated_predicted_output, nrow = nrow(actual_values), ncol = ncol(actual_values), byrow = FALSE)
        }
      } else {
        predicted_output_matrix <- single_predicted_output_learn
      }
      
      # Calculate the error
      error_prediction <- actual_values - predicted_output_matrix
      
      # Calculate differences
      differences <- error_prediction
      
      # Calculate summary statistics
      summary_stats <- summary(differences)
      boxplot_stats <- boxplot.stats(differences)
      
      # Create a list to store weights and biases records
      weights_records_learn <- list()
      biases_records_learn <- list()
      
      # Loop through each iteration and store weights and biases records in the list
      for (i in 1:num_networks) {
        weights_var_name <- paste0("weights_record_learn_", i)
        biases_var_name <- paste0("biases_record_learn_", i)
        weights_records_learn[[i]] <- get(weights_var_name)
        biases_records_learn[[i]] <- get(biases_var_name)
      }
      
      # Store results in a list
      result_learnOnly <- list(
        predicted_output = single_predicted_output_learn,
        learn_time = single_learn_time,
        differences = tail(differences),
        summary_stats = summary_stats,
        boxplot_stats_stats = boxplot_stats$stats,
        boxplot_stats_n = boxplot_stats$n,
        boxplot_stats_conf = boxplot_stats$conf,
        boxplot_stats_out = boxplot_stats$out,
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
        run_id = run_id,
        model_iter_num = model_iter_num,
        threshold = threshold,
        predicted_output_tail = tail(single_predicted_output_learn),
        actual_values_tail = tail(actual_values),
        X = X,
        y = y,
        weights_record_learn = weights_records_learn,
        biases_record_learn = biases_records_learn
      )
      
      # Store the result in the results_list_learnOnly
      results_list_learnOnly[[model_iter_num]] <- result_learnOnly #<<-
    },
    store_metadata = function(run_id, ensemble_number, model_iter_num, num_epochs, threshold, predicted_output, actual_values, performance_metric, relevance_metric, predicted_outputAndTime) {
      
      # Print the dimensions of predicted_output
      # print(paste("Dimensions of predicted_output:", paste(dim(predicted_output), collapse = " x ")))
      # print(paste("Dimensions of actual_values:", paste(dim(actual_values), collapse = " x ")))
      
      if (ncol(actual_values) != ncol(predicted_output)) {
        if (ncol(predicted_output) < ncol(actual_values)) {
          # Calculate the required replication factor
          total_elements_needed <- nrow(actual_values) * ncol(actual_values)
          rep_factor <- ceiling(total_elements_needed / length(predicted_output))
          
          # Create the replicated vector and check its length
          replicated_predicted_output <- rep(predicted_output, rep_factor)
          # Truncate the replicated vector to match the required length
          replicated_predicted_output <- replicated_predicted_output[1:total_elements_needed]
          # Create the matrix and check its dimensions
          predicted_output_matrix <- matrix(replicated_predicted_output, nrow = nrow(actual_values), ncol = ncol(actual_values), byrow = FALSE)
        } else {
          # Truncate predicted_output to match the number of columns in actual_values
          truncated_predicted_output <- predicted_output[, 1:ncol(actual_values)]
          # Create the matrix and check its dimensions
          predicted_output_matrix <- matrix(truncated_predicted_output, nrow = nrow(actual_values), ncol = ncol(actual_values), byrow = FALSE)
        }
      } else {
        predicted_output_matrix <- predicted_output
      }
      
      # Calculate the error
      error_prediction <- actual_values - predicted_output_matrix
      
      # print("Error prediction calculation complete")
      
      # Calculate differences
      differences <- error_prediction
      
      # Calculate summary statistics
      summary_stats <- summary(differences)
      boxplot_stats <- boxplot.stats(differences)
      
      # Read the plot_epochs object from the RDS file
      plot_epochs <- readRDS(paste("plot_epochs_DESONN", ensemble_number, "SONN", model_iter_num, ".rds", sep = ""))
      
      # Create a list to store weights and biases records
      weights_records <- list()
      biases_records <- list()
      
      # Loop through each iteration and store weights and biases records in the list
      for (i in 1:num_networks) {
        weights_var_name <- paste0("weights_record_", i)
        biases_var_name <- paste0("biases_record_", i)
        weights_records[[i]] <- get(weights_var_name)
        biases_records[[i]] <- get(biases_var_name)
      }
      
      # Store results in a list
      result <- list(
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
        performance_metric  = performance_metric,
        relevance_metric = relevance_metric,
        plot_epochs = plot_epochs,
        weights_record = weights_records,
        biases_record = biases_records
      )
      
      # This is for the initial finding the best model out of all the hyperparameters.
      if ((hyperparameter_grid_setup && !learnOnlyTrainingRun && (never_ran_flag || !never_ran_flag)) || predict_models) {
        print(model_iter_num)
        # Append result to the results list
        results_list[[model_iter_num]] <- result #<<-
        
        # Create dynamic variable name
        variable_name <- paste0("Ensemble_", ensemble_number, "_model_", model_iter_num)
        
        # Assign the result to the dynamic variable
        assign(variable_name, result, envir = .GlobalEnv)
      } else { # Temp iteration
        # Append result to the temporary ensemble (second list)
        ensembles$temp_ensemble[[model_iter_num]] <- result #<<-
      }
    }
    
    
    
  )
)

is_binary <- function(column) {
  unique_values <- unique(column)
  return(length(unique_values) == 2)
}


initialize_optimizer_params <- function(optimizer, dim, lookahead_step, layer) {
  params <- list()
  
  if (length(dim) == 2 && is.null(layer)) {
    # Multi-layer is not specified; wrap single-layer into a list
    dim <- list(dim)
  }
  
  for (i in seq_along(dim)) {
    layer_dim <- dim[[i]]
    
    if (length(layer_dim) != 2 || any(is.na(layer_dim)) || any(layer_dim <= 0)) {
      cat("Invalid dimensions detected for layer", i, ". Setting default dimension [1, 1].\n")
      layer_dim <- c(1, 1)
    }
    
    nrow_dim <- layer_dim[1]
    ncol_dim <- layer_dim[2]
    
    current_layer <- if (!is.null(layer)) layer else i
    cat("Layer", current_layer, "dimensions: nrow =", nrow_dim, ", ncol =", ncol_dim, "\n")
    
    param_init <- matrix(rnorm(nrow_dim * ncol_dim), nrow = nrow_dim, ncol = ncol_dim)
    
    # Store optimizer state for this layer
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
    
    params[[current_layer]] <- entry
    
    if (verbose) {
      cat("Layer", current_layer, "optimizer tracking params initialized:\n")
      print(str(entry))
    }
  }
  
  return(params)
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
  # Initialize v as a list if it's not already
  if (!is.list(params$v)) {
    params$v <- vector("list", length(grads))
  }
  
  # Update each element of v
  for (i in seq_along(grads)) {
    # Get the dimensions of the current gradient
    grad_dims <- dim(grads[[i]])
    
    # If grads[[i]] is a scalar, set grad_dims to c(1) and convert to array
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Initialize v if it doesn't exist or has wrong dimensions
    if (is.null(params$v[[i]]) || !all(dim(params$v[[i]]) == grad_dims)) {
      params$v[[i]] <- array(0, dim = grad_dims)
    }
    
    # Update v for this element
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grads[[i]] ^ 2)
  }
  
  # Compute the updates for each element
  updates <- Map(function(v, grad) lr * grad / (sqrt(v) + epsilon), params$v, grads)
  
  # Convert updates to the same format as grads (matrix or array)
  for (i in seq_along(updates)) {
    if (is.null(dim(grads[[i]]))) {
      updates[[i]] <- array(updates[[i]], dim = c(1))
    } else {
      updates[[i]] <- matrix(updates[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
    }
  }
  
  # Return updated parameters and updates
  return(list(v = params$v, updates = updates))
}

adagrad_update <- function(params, grads, lr, epsilon = 1e-8) {
  # Initialize r as a list if it's not already
  if (!is.list(params$r)) {
    params$r <- vector("list", length(grads))
  }
  
  # Update each element of r
  for (i in seq_along(grads)) {
    # Get the dimensions of the current gradient
    grad_dims <- dim(grads[[i]])
    
    # If grads[[i]] is a scalar, set grad_dims to c(1) and convert to array
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Initialize r if it doesn't exist or has wrong dimensions
    if (is.null(params$r[[i]]) || !all(dim(params$r[[i]]) == grad_dims)) {
      params$r[[i]] <- array(0, dim = grad_dims)
    }
    
    # Update r for this element
    params$r[[i]] <- params$r[[i]] + grads[[i]] ^ 2
  }
  
  # Compute the updates for each element
  updates <- Map(function(r, grad) lr * grad / (sqrt(r) + epsilon), params$r, grads)
  
  # Convert updates to the same format as grads (matrix or array)
  for (i in seq_along(updates)) {
    if (is.null(dim(grads[[i]]))) {
      updates[[i]] <- array(updates[[i]], dim = c(1))
    } else {
      updates[[i]] <- matrix(updates[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
    }
  }
  
  # Return updated parameters and updates
  return(list(r = params$r, updates = updates))
}

adadelta_update <- function(params, grads, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, t = 1) {
  # Initialize m and v as lists if they are not already
  if (!is.list(params$m)) {
    params$m <- vector("list", length(grads))
  }
  if (!is.list(params$v)) {
    params$v <- vector("list", length(grads))
  }
  
  # Initialize delta_w
  delta_w <- vector("list", length(grads))
  
  # Update each element of m and v
  for (i in seq_along(grads)) {
    # Get the dimensions of the current gradient
    grad_dims <- dim(grads[[i]])
    
    # If grads[[i]] is a scalar, set grad_dims to c(1) and convert to array
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Initialize m and v if they don't exist or have wrong dimensions
    if (is.null(params$m[[i]]) || !all(dim(params$m[[i]]) == grad_dims)) {
      params$m[[i]] <- array(0, dim = grad_dims)
    }
    if (is.null(params$v[[i]]) || !all(dim(params$v[[i]]) == grad_dims)) {
      params$v[[i]] <- array(0, dim = grad_dims)
    }
    
    # Update m and v for this element
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grads[[i]] ^ 2)
    
    # Bias correction
    v_hat <- params$v[[i]] / (1 - beta2 ^ t)
    
    # Calculate the Adadelta update
    delta <- (sqrt(params$m[[i]] + epsilon) / sqrt(v_hat + epsilon)) * grads[[i]]
    
    # Update m for this element
    params$m[[i]] <- beta1 * params$m[[i]] + (1 - beta1) * (delta ^ 2)
    
    # Store the update
    delta_w[[i]] <- delta
  }
  
  # Convert delta_w to the same format as grads (matrix or array)
  for (i in seq_along(delta_w)) {
    if (is.null(dim(grads[[i]]))) {
      delta_w[[i]] <- array(delta_w[[i]], dim = c(1))
    } else {
      delta_w[[i]] <- matrix(delta_w[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
    }
  }
  
  # Return updated parameters, m, v, and delta_w
  return(list(m = params$m, v = params$v, delta_w = delta_w))
}

# Stochastic Gradient Descent with Momentum
# Define the sgd_update function with improvements
# Improved SGD Update Function
# Define the sgd_update function
sgd_momentum_update <- function(params, grads, lr) {
  
  momentum <- 0.9
  
  # Initialize momentum as a list if it is not already
  if (!is.list(params$momentum)) {
    params$momentum <- vector("list", length(grads))
  }
  
  # Initialize weights_update and biases_update as lists
  weights_update <- vector("list", length(grads))
  biases_update <- vector("list", length(grads))
  
  # Update each element of momentum and calculate weights_update and biases_update
  for (i in seq_along(grads)) {
    # Get the dimensions of the current gradient
    grad_dims <- dim(grads[[i]])
    
    # If grads[[i]] is a scalar, set grad_dims to c(1) and convert to array
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Initialize momentum if it doesn't exist or has wrong dimensions
    if (is.null(params$momentum[[i]]) || !all(dim(params$momentum[[i]]) == grad_dims)) {
      params$momentum[[i]] <- array(0, dim = grad_dims)
    }
    
    # Update momentum for this element
    params$momentum[[i]] <- momentum * params$momentum[[i]] - lr * grads[[i]]
    
    # Calculate weights_update for this element
    weights_update[[i]] <- params$momentum[[i]]
    
    # Calculate biases_update for this element (biases are updated directly without momentum)
    biases_update[[i]] <- lr * grads[[i]]
  }
  
  # Convert weights_update and biases_update to the same format as grads (matrix or array)
  for (i in seq_along(weights_update)) {
    if (is.null(dim(grads[[i]]))) {
      weights_update[[i]] <- array(weights_update[[i]], dim = c(1))
      biases_update[[i]] <- array(biases_update[[i]], dim = c(1))
    } else {
      weights_update[[i]] <- matrix(weights_update[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
      biases_update[[i]] <- matrix(biases_update[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
    }
  }
  
  # Return updated parameters, weights_update, and biases_update
  return(list(params = params, weights_update = weights_update, biases_update = biases_update))
}

sgd_update <- function(params, grads, lr) {
  # Initialize weights_update and biases_update as lists
  weights_update <- vector("list", length(grads))
  biases_update <- vector("list", length(grads))
  
  # Update weights and biases
  for (i in seq_along(grads)) {
    # Get the dimensions of the current gradient
    grad_dims <- dim(grads[[i]])
    
    # If grads[[i]] is a scalar, set grad_dims to c(1) and convert to array
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Calculate weights_update for this element
    weights_update[[i]] <- lr * grads[[i]]
    
    # Calculate biases_update for this element
    biases_update[[i]] <- lr * grads[[i]]  # Biases are updated with the same gradient as weights
  }
  
  # Convert weights_update and biases_update to the same format as grads (matrix or array)
  for (i in seq_along(weights_update)) {
    if (is.null(dim(grads[[i]]))) {
      weights_update[[i]] <- array(weights_update[[i]], dim = c(1))
      biases_update[[i]] <- array(biases_update[[i]], dim = c(1))
    } else {
      weights_update[[i]] <- matrix(weights_update[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
      biases_update[[i]] <- matrix(biases_update[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
    }
  }
  
  # Return updated parameters, weights_update, and biases_update
  return(list(weights_update = weights_update, biases_update = biases_update))
}

nag_update <- function(params, grads, lr, beta = 0.9) {
  weights_update <- vector("list", length(grads))
  biases_update <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    if (is.null(grad_dims)) {
      grad_dims <- c(length(grads[[i]]), 1)  # Handle gradients as vectors
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Ensure params components are initialized
    if (length(params$momentum) < i) {
      params$momentum[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    if (length(params$fast_weights) < i) {
      params$fast_weights[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    if (length(params$fast_biases) < i) {
      params$fast_biases[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    
    # Update the momentum term for weights
    params$momentum[[i]] <- beta * params$momentum[[i]] + grads[[i]]
    weights_update[[i]] <- lr * (beta * params$momentum[[i]] + grads[[i]])
    biases_update[[i]] <- lr * grads[[i]]
    
    # Update fast weights for NAG
    params$fast_weights[[i]] <- params$fast_weights[[i]] - weights_update[[i]]
    params$fast_biases[[i]] <- params$fast_biases[[i]] - biases_update[[i]]
  }
  
  return(list(params = params, weights_update = weights_update, biases_update = biases_update))
}

# Define FTRL optimizer
ftrl_update <- function(params, grads, lr, alpha = 0.1, beta = 1.0, lambda1 = 0.01, lambda2 = 0.01) {
  weights_update <- vector("list", length(grads))
  biases_update <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    if (is.null(grad_dims)) {
      grad_dims <- c(length(grads[[i]]), 1)  # Handle gradients as vectors
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Ensure params components are initialized
    if (length(params$z) < i || is.null(params$z[[i]])) {
      params$z[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    if (length(params$n) < i || is.null(params$n[[i]])) {
      params$n[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    
    # Update z and n
    params$z[[i]] <- params$z[[i]] + grads[[i]]
    params$n[[i]] <- params$n[[i]] + grads[[i]]^2
    
    # Compute the update step for weights
    weights_update[[i]] <- -((lr / (sqrt(params$n[[i]]) + beta)) * (params$z[[i]] - (lambda1 / (lr + beta)) * sign(params$z[[i]])))
    
    # Compute the update step for biases
    biases_update[[i]] <- rep(0, length(grads[[i]]))  # No bias updates for this example
    
    # Update weights
    params$z[[i]] <- params$z[[i]] - weights_update[[i]]
    params$n[[i]] <- params$n[[i]] - weights_update[[i]]
  }
  
  return(list(params = params, weights_update = weights_update, biases_update = biases_update))
}

# LAMB Update Function
lamb_update <- function(params, grads, lr, beta1, beta2, epsilon, lambda) {
  # Ensure params$param and grads are numeric
  params$param <- as.numeric(params$param)
  grads <- as.numeric(grads)
  
  # Extract the dimensions of the parameter matrix from the network structure
  nrows <- length(params$param) / length(grads)
  ncols <- length(grads)
  
  # Convert vectors to matrices for the current layer
  param_matrix <- matrix(params$param, nrow = nrows, ncol = ncols)
  grad_matrix <- matrix(grads, nrow = nrows, ncol = ncols)
  m <- matrix(params$m, nrow = nrows, ncol = ncols)
  v <- matrix(params$v, nrow = nrows, ncol = ncols)
  
  # Compute m_t and v_t
  m <- beta1 * m + (1 - beta1) * grad_matrix
  v <- beta2 * v + (1 - beta2) * (grad_matrix^2)
  
  # Compute m_hat and v_hat
  m_hat <- m / (1 - beta1)
  v_hat <- v / (1 - beta2)
  
  # Compute weight updates
  update <- m_hat / (sqrt(v_hat) + epsilon)
  
  # Apply weight decay
  update <- update / (1 + lambda * lr)
  
  # Ensure that param_matrix and update have the same dimensions
  if (all(dim(param_matrix) == dim(update))) {
    updated_param <- param_matrix - lr * update
  } else {
    stop("Dimensions of param_matrix and update do not match.")
  }
  
  # Return updated parameters and optimizers state
  list(
    param = as.numeric(updated_param),
    m = as.numeric(m),
    v = as.numeric(v),
    update = as.numeric(update)  # Ensure update is included in the return list
  )
}



lookahead_update <- function(params, grads_list, lr, beta1, beta2, epsilon, lookahead_step, base_optimizer, t, lambda) {
  updated_params_list <- list()
  
  # Check the structure of grads_list
  cat("Structure of grads_list:\n")
  print(str(grads_list))
  
  for (layer in seq_along(params)) {
    cat("Processing Layer lookahead update: ", layer, "\n")
    
    # Check the structure of each layer in grads_list
    cat("Structure of grads_list for layer", layer, ":\n")
    print(str(grads_list[[layer]]))
    
    param_list <- params[[layer]]
    
    # Check if 'param' exists in grads_list[[layer]]
    if (!"param" %in% names(grads_list[[layer]])) {
      cat("Warning: 'param' not found in grads_list for layer", layer, "\n")
      next  # Skip to the next layer if 'param' is not found
    }
    
    grad_matrix <- grads_list[[layer]]$param
    
    # Ensure the gradients are correctly structured
    if (is.null(grad_matrix)) {
      stop(paste("Missing element 'param' in gradients for layer", layer))
    }
    
    param <- param_list$param
    m <- param_list$m
    v <- param_list$v
    r <- param_list$r
    slow_param <- param_list$slow_weights  # Use slow_weights for weights or biases
    lookahead_counter <- param_list$lookahead_counter
    
    # Debug statements
    cat("Initial lookahead_counter for layer", layer, ":", lookahead_counter, "\n")
    cat("lookahead_step for layer", layer, ":", lookahead_step, "\n")
    
    # Ensure lookahead_counter is not NULL and is numeric
    if (is.null(lookahead_counter) || !is.numeric(lookahead_counter)) {
      lookahead_counter <- 0  # Initialize to 0 if missing or not numeric
      cat("lookahead_counter initialized to 0 for layer", layer, "\n")
    }
    
    if (is.null(lookahead_step) || !is.numeric(lookahead_step)) {
      stop("lookahead_step is missing or not numeric.")
    }
    
    if (base_optimizer == "adam_update") {
      # Adam Update
      m <- beta1 * m + (1 - beta1) * grad_matrix
      v <- beta2 * v + (1 - beta2) * (grad_matrix^2)
      
      m_hat <- m / (1 - beta1^t)
      v_hat <- v / (1 - beta2^t)
      
      param <- param - lr * m_hat / (sqrt(v_hat) + epsilon)
      
      updated_params_list[[layer]] <- list(
        param = param,
        m = m,
        v = v,
        r = r,
        slow_weights = slow_param,
        lookahead_counter = lookahead_counter,
        lookahead_step = lookahead_step
      )
      
    } else if (base_optimizer == "lamb_update") {
      # LAMB Update
      m <- beta1 * m + (1 - beta1) * grad_matrix
      v <- beta2 * v + (1 - beta2) * (grad_matrix^2)
      
      m_hat <- m / (1 - beta1^t)
      v_hat <- v / (1 - beta2^t)
      
      r1 <- sqrt(sum(param^2))
      r2 <- sqrt(sum((m_hat / (sqrt(v_hat) + epsilon))^2))
      
      ratio <- ifelse(r1 == 0 | r2 == 0, 1, r1 / r2)
      
      param <- param - lr * ratio * m_hat / (sqrt(v_hat) + epsilon)
      
      slow_param <- beta1 * slow_param + (1 - beta1) * param
      
      updated_params_list[[layer]] <- list(
        param = param,
        m = m,
        v = v,
        r = r,
        slow_weights = slow_param,
        lookahead_counter = lookahead_counter,
        lookahead_step = lookahead_step
      )
      
    } else {
      stop("Unsupported base optimizer.")
    }
    
    # Update the lookahead mechanism
    lookahead_counter <- lookahead_counter + 1
    
    if (lookahead_counter >= lookahead_step) {
      cat("Performing lookahead update for layer", layer, "\n")
      slow_param <- param
      lookahead_counter <- 0
    }
    
    # Store the updated parameters
    updated_params_list[[layer]] <- list(
      param = param,
      m = m,
      v = v,
      r = r,
      slow_weights = slow_param,
      lookahead_counter = lookahead_counter,
      lookahead_step = lookahead_step
    )
    
    cat("Updated lookahead_counter for layer", layer, ":", lookahead_counter, "\n")
  }
  
  return(updated_params_list)
}














clip_gradient_norm <- function(gradient, min_norm = 1e-3, max_norm = 5) {
  grad_norm <- sqrt(sum(gradient^2))
  if (grad_norm > max_norm) {
    gradient <- gradient * (max_norm / grad_norm)
  } else if (grad_norm < min_norm && grad_norm > 0) {
    gradient <- gradient * (min_norm / grad_norm)
  }
  return(gradient)
}






binary_activation_derivative <- function(x) {
  return(rep(0, length(x)))  # Non-differentiable, set to 0
}


custom_binary_activation_derivative <- function(x, threshold = -1.08) {
  return(rep(0, length(x)))  # Also a step function, gradient is zero everywhere
}


custom_activation_derivative <- function(z) {
  softplus_output <- log1p(exp(z))
  softplus_derivative <- 1 / (1 + exp(-z))  # sigmoid
  
  # If thresholding applies hard cut-off, gradient becomes 0
  return(ifelse(softplus_output > 0.00000000001, 0, softplus_derivative))
}

bent_identity_derivative <- function(x) {
  return(x / (2 * sqrt(x^2 + 1)) + 1)
}

relu_derivative <- function(x) {
  return(ifelse(x > 0, 1, 0))
}


softplus_derivative <- function(x) {
  return(1 / (1 + exp(-x)))
}

leaky_relu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

elu_derivative <- function(x, alpha = 1.0) {
  return(ifelse(x > 0, 1, alpha * exp(x)))
}

tanh_derivative <- function(x) {
  t <- tanh(x)
  return(1 - t^2)
}

sigmoid_derivative <- function(x) {
  s <- 1 / (1 + exp(-x))
  return(s * (1 - s))
}

hard_sigmoid_derivative <- function(x) {
  return(ifelse(x > -2.5 & x < 2.5, 0.2, 0))
}

swish_derivative <- function(x) {
  s <- 1 / (1 + exp(-x))  # sigmoid(x)
  return(s + x * s * (1 - s))
}

sigmoid_binary_derivative <- function(x) {
  # Approximate pseudo-gradient
  return(rep(0, length(x)))  # Step function is not differentiable
}

gelu_derivative <- function(x) {
  phi <- 0.5 * (1 + erf(x / sqrt(2)))
  dphi <- exp(-x^2 / 2) / sqrt(2 * pi)
  return(0.5 * phi + x * dphi)
}

selu_derivative <- function(x, lambda = 1.0507, alpha = 1.67326) {
  return(lambda * ifelse(x > 0, 1, alpha * exp(x)))
}

mish_derivative <- function(x) {
  sp <- log1p(exp(x))              # softplus
  tanh_sp <- tanh(sp)
  grad_sp <- 1 - exp(-sp)          # d(softplus) ≈ sigmoid(x)
  return(tanh_sp + x * grad_sp * (1 - tanh_sp^2))
}

maxout_derivative <- function(x, w1 = 0.5, b1 = 1.0, w2 = -0.5, b2 = 0.5) {
  val1 <- w1 * x + b1
  val2 <- w2 * x + b2
  return(ifelse(val1 > val2, w1, w2))  # Returns the gradient of the active unit
}

prelu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

softmax_derivative <- function(x) {
  s <- softmax(x)
  return(s * (1 - s))  # Only valid when loss is MSE, not cross-entropy
}

bent_relu_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  base_deriv <- (x / (2 * sqrt(x^2 + 1))) + 1
  return(ifelse(x > 0, base_deriv, 0))
}

bent_sigmoid_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent_part <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  sigmoid_out <- 1 / (1 + exp(-bent_part))
  dbent_dx <- (x / (2 * sqrt(x^2 + 1))) + 1
  out <- sigmoid_out * (1 - sigmoid_out) * dbent_dx
  dim(out) <- dim(x)
  return(out)
}


###################################################################################################################################################################



# -------------------------
# Activation Functions (Fixed)
# -------------------------

binary_activation <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0.5, 1, 0))
}
attr(binary_activation, "name") <- "binary_activation"

custom_binary_activation <- function(x, threshold = -1.08) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x < threshold, 0, 1))
}
attr(custom_binary_activation, "name") <- "custom_binary_activation"

custom_activation <- function(z) {
  z <- as.matrix(z); dim(z) <- dim(z)
  softplus_output <- log1p(exp(z))
  return(ifelse(softplus_output > 1e-11, 1, 0))
}
attr(custom_activation, "name") <- "custom_activation"

bent_identity <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return((sqrt(x^2 + 1) - 1) / 2 + x)
}
attr(bent_identity, "name") <- "bent_identity"

relu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, 0))
}
attr(relu, "name") <- "relu"

softplus <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(log1p(exp(x)))
}
attr(softplus, "name") <- "softplus"

leaky_relu <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * x))
}
attr(leaky_relu, "name") <- "leaky_relu"

elu <- function(x, alpha = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * (exp(x) - 1)))
}
attr(elu, "name") <- "elu"

tanh <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
}
attr(tanh, "name") <- "tanh"

sigmoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + exp(-x)))
}
attr(sigmoid, "name") <- "sigmoid"

hard_sigmoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(pmax(0, pmin(1, 0.2 * x + 0.5)))
}
attr(hard_sigmoid, "name") <- "hard_sigmoid"

swish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * sigmoid(x))
}
attr(swish, "name") <- "swish"

sigmoid_binary <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse((1 / (1 + exp(-x))) >= 0.5, 1, 0))
}
attr(sigmoid_binary, "name") <- "sigmoid_binary"

gelu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * 0.5 * (1 + erf(x / sqrt(2))))
}
attr(gelu, "name") <- "gelu"

selu <- function(x, lambda = 1.0507, alpha = 1.67326) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(lambda * ifelse(x > 0, x, alpha * exp(x) - alpha))
}
attr(selu, "name") <- "selu"

mish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * tanh(log(1 + exp(x))))
}
attr(mish, "name") <- "mish"

prelu <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * x))
}
attr(prelu, "name") <- "prelu"

softmax <- function(z) {
  z <- as.matrix(z); dim(z) <- dim(z)
  exp_z <- exp(z)
  return(exp_z / rowSums(exp_z))
}
attr(softmax, "name") <- "softmax"

# Maxout example
maxout <- function(x, w1 = 0.5, b1 = 1.0, w2 = -0.5, b2 = 0.5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(pmax(w1 * x + b1, w2 * x + b2))
}
attr(maxout, "name") <- "maxout"

bent_relu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  out <- pmax(0, ((sqrt(x^2 + 1) - 1) / 2 + x))
  dim(out) <- dim(x)
  return(out)
}
attr(bent_relu, "name") <- "bent_relu"

bent_sigmoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent_part <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  return(1 / (1 + exp(-bent_part)))
}
attr(bent_sigmoid, "name") <- "bent_sigmoid"


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




lr_scheduler <- function(epoch, initial_lr = 0.1, decay_rate = 0.5, decay_epoch = 45, min_lr = 1e-5) {
  decayed_lr <- initial_lr * decay_rate ^ floor(epoch / decay_epoch)
  return(max(min_lr, decayed_lr))
}




# Helper functions
calculate_performance <- function(SONN, Rdata, labels, lr, model_iter_num, num_epochs, threshold, predicted_output, prediction_time, ensemble_number, run_id) {
  
  max_k <- 15
  
  # Define the calculate_wss function
  calculate_wss <- function(Rdata, max_k) {
    wss <- numeric(max_k)
    for (i in 2:max_k) {
      km.out <- kmeans(Rdata, centers=i)
      wss[i] <- km.out$tot.withinss
    }
    return(wss)
  }
  
  # Use the function
  wss <- calculate_wss(Rdata, max_k)
  
  # Determine the optimal number of clusters
  optimal_k <- which(diff(diff(wss)) == max(diff(diff(wss)))) + 1
  
  # Plot the WSS to visualize the Elbow Plot
  plot(1:max_k, wss, type = "b", pch = 1, frame = TRUE,
       xlab = "Number of Clusters", ylab = "Within groups sum of squares")
  points(optimal_k, wss[optimal_k], col = "red", pch = 19) # Red circle at the optimal number of clusters
  text(optimal_k, wss[optimal_k], labels = paste("Optimal k =", optimal_k), pos = 3, offset = 1)
  
  # Initialize variables for total metrics and weights
  total_metrics <- list()
  total_weights <- list()
  
  # Metrics to calculate for each layer
  metrics_to_calculate <- c(
    "quantization_error",
    "topographic_error",
    "clustering_quality_db",
    "MSE",
    "speed",
    "memory_usage"
  )
  
  # Initialize perf_metrics to store detailed metrics
  perf_metrics <- list()
  
  # Initialize total_metrics and total_weights
  total_metrics <- list(speed = 0, memory_usage = 0)
  total_weights <- list(speed = 0, memory_usage = 0)
  
  # Check if SONN$weights is present
  if (!is.null(SONN$weights)) {
    # Iterate through each layer of SONN$weights
    for (i in 1:length(SONN$weights)) {
      layer_weights <- SONN$weights[[i]]
      map <- as.matrix(SONN$map[[i]])
      
      # Perform kmeans clustering and obtain cluster assignments
      result <- kmeans(Rdata, centers = optimal_k)
      cluster_assignments <- result$cluster  # Cluster assignments
      
      # Calculate performance metrics for this layer
      layer_perf_metrics <- list(
        quantization_error = quantization_error(layer_weights, Rdata, run_id),
        topographic_error = topographic_error(layer_weights, map, Rdata, threshold),
        clustering_quality_db = clustering_quality_db(layer_weights, Rdata, cluster_assignments),
        MSE = MSE(layer_weights, Rdata, labels, predicted_output),
        speed = speed(layer_weights, prediction_time),
        memory_usage = memory_usage(layer_weights, Rdata)
      )
      
      # Add layer performance metrics to perf_metrics only for the last layer
      if (i == length(SONN$weights)) {
        perf_metrics <- layer_perf_metrics
      }
      
      # Accumulate metrics for total (no averaging)
      total_metrics$speed <- total_metrics$speed + layer_perf_metrics$speed
      total_metrics$memory_usage <- total_metrics$memory_usage + layer_perf_metrics$memory_usage
    }
    
    # Assign total_metrics directly without averaging
    perf_metrics$speed <- total_metrics$speed
    perf_metrics$memory_usage <- total_metrics$memory_usage
    
    # Return perf_metrics with all detailed metrics and names of metrics
    return(list(metrics = perf_metrics, names = names(perf_metrics)))
  }
  else {
    # Perform kmeans clustering and obtain cluster assignments
    result <- kmeans(Rdata, centers = 3)
    cluster_assignments <- result$cluster  # Cluster assignments
    
    # Calculate performance metrics
    perf_metrics <- list(
      quantization_error = quantization_error(SONN, Rdata, run_id),
      topographic_error = topographic_error(SONN, map, Rdata, threshold),
      clustering_quality_db = clustering_quality_db(SONN, Rdata, cluster_assignments),
      MSE = MSE(SONN, Rdata, labels, predicted_output),
      speed = speed(SONN, prediction_time),
      memory_usage = memory_usage(SONN, Rdata)
    )
    
    # Get names of the metrics
    perf_names <- names(perf_metrics)
    
    # Return perf_metrics with all detailed metrics and names of metrics
    return(list(metrics = perf_metrics, names = perf_names))
  }
}

calculate_relevance <- function(SONN, Rdata, labels, model_iter_num, predicted_output, ensemble_number) {
  # Calculate relevance metrics
  rel_metrics <- list(
    #hit_rate = hit_rate(SONN, Rdata, predicted_output),
    precision = precision(SONN, Rdata, labels, predicted_output),
    #precision_boolean = precision_boolean(SONN, Rdata, labels, predicted_output),
    #recall = recall(SONN, Rdata, predicted_output),
    MAE = MAE(SONN, Rdata, labels, predicted_output),
    #f1_score = f1_score(SONN, Rdata, labels),
    #ndcg = ndcg(SONN, Rdata, predicted_output_l2),
    #mean_precision = mean_precision(SONN, Rdata, labels, predicted_output),
    diversity = diversity(SONN, Rdata, predicted_output),
    RMSE = RMSE(SONN, Rdata, labels, predicted_output),
    serendipity = serendipity(SONN, Rdata, predicted_output)
  )
  
  # Handle metrics that may not always be calculated
  for (metric_name in c("hit_rate", "recall", "f1_score", "ndcg", "precision", "MAE", "mean_precision", "diversity", "novelty", "serendipity")) {
    if (!metric_name %in% names(rel_metrics)) {
      rel_metrics[[metric_name]] <- NULL
    } else {
      metric_value <- rel_metrics[[metric_name]]
      
      # Check if the metric value is NULL or contains any NA values
      if (is.null(metric_value) || any(is.na(metric_value)) || any(isTRUE(metric_value))) {
        rel_metrics[[metric_name]] <- NULL
      }
    }
  }
  
  
  # Get names of the metrics
  rel_names <- names(rel_metrics)
  
  return(list(metrics = rel_metrics, names = rel_names))
}

calculate_performance_learn <- function(SONN, Rdata, labels, lr, model_iter_num, num_epochs, threshold, predicted_output, learn_time, ensemble_number) {
  # Initialize total_metrics and total_weights
  total_metrics <- list(speed = 0, memory_usage = 0)
  total_weights <- list(speed = 0, memory_usage = 0)
  
  # Check if SONN$weights is present
  if (!is.null(SONN$weights)) {
    # Iterate through each layer of SONN$weights
    for (i in 1:length(SONN$weights)) {
      layer_weights <- SONN$weights[[i]]
      map <- as.matrix(SONN$map[[i]])
      
      # Perform kmeans clustering and obtain cluster assignments
      result <- kmeans(Rdata, centers = optimal_k)
      cluster_assignments <- result$cluster  # Cluster assignments
      
      # Calculate performance metrics for this layer
      layer_perf_metrics <- list(
        quantization_error = quantization_error(layer_weights, Rdata, run_id),
        topographic_error = topographic_error(layer_weights, map, Rdata, threshold),
        clustering_quality_db = clustering_quality_db(layer_weights, Rdata, cluster_assignments),
        MSE = MSE(layer_weights, Rdata, labels, predicted_output),
        speed = speed_learn(layer_weights, learn_time),
        memory_usage = memory_usage(layer_weights, Rdata)
      )
      
      # Add layer performance metrics to perf_metrics only for the last layer
      if (i == length(SONN$weights)) {
        perf_metrics <- layer_perf_metrics
      }
      
      # Accumulate metrics for total (no averaging)
      total_metrics$speed <- total_metrics$speed + layer_perf_metrics$speed
      total_metrics$memory_usage <- total_metrics$memory_usage + layer_perf_metrics$memory_usage
    }
    
    # Assign total_metrics directly without averaging
    perf_metrics$speed <- total_metrics$speed
    perf_metrics$memory_usage <- total_metrics$memory_usage
    
    # Return perf_metrics with all detailed metrics and names of metrics
    return(list(metrics = perf_metrics, names = names(perf_metrics)))
  }
  else {
    # Perform kmeans clustering and obtain cluster assignments
    result <- kmeans(Rdata, centers = 3)
    cluster_assignments <- result$cluster  # Cluster assignments
    
    # Calculate performance metrics
    perf_metrics <- list(
      quantization_error = quantization_error(SONN, Rdata, run_id),
      topographic_error = topographic_error(SONN, map, Rdata, threshold),
      clustering_quality_db = clustering_quality_db(SONN, Rdata, cluster_assignments),
      MSE = MSE(SONN, Rdata, labels, predicted_output),
      speed = speed_learn(SONN, learn_time),
      memory_usage = memory_usage(SONN, Rdata)
    )
    
    # Handle metrics that may not always be calculated
    for (metric_name in metrics_to_calculate) {
      if (!metric_name %in% names(perf_metrics)) {
        perf_metrics[[metric_name]] <- NULL
      } else {
        metric_value <- perf_metrics[[metric_name]]
        
        # Check if the metric value is NULL or contains any NA values
        if (is.null(metric_value) || any(is.na(metric_value))) {
          perf_metrics[[metric_name]] <- NULL
        }
      }
    }
    
    # Get names of the metrics
    perf_names <- names(perf_metrics)
    
    return(list(metrics = perf_metrics, names = perf_names))
  }
}

calculate_relevance_learn <- function(SONN, Rdata, labels, model_iter_num, predicted_output, ensemble_number) {
  # Calculate relevance metrics
  rel_metrics <- list(
    #hit_rate = hit_rate(SONN, Rdata, predicted_output),
    precision = precision(SONN, Rdata, labels, predicted_output),
    #recall = recall(SONN, Rdata, predicted_output),
    MAE = MAE(SONN, Rdata, labels, predicted_output),
    #f1_score = f1_score(SONN, Rdata, labels),
    #ndcg = ndcg(SONN, Rdata, predicted_output_l2),
    mean_precision = mean_precision(SONN, Rdata, labels, predicted_output),
    diversity = diversity(SONN, Rdata, predicted_output),
    RMSE = RMSE(SONN, Rdata, labels, predicted_output),
    serendipity = serendipity(SONN, Rdata, predicted_output)
  )
  
  # Handle metrics that may not always be calculated
  for (metric_name in c("hit_rate", "recall", "f1_score", "ndcg", "precision", "MAE", "mean_precision", "diversity", "novelty", "serendipity")) {
    if (!metric_name %in% names(rel_metrics)) {
      rel_metrics[[metric_name]] <- NULL
    } else {
      metric_value <- rel_metrics[[metric_name]]
      
      # Check if the metric value is NULL or contains any NA values
      if (is.null(metric_value) || any(is.na(metric_value)) || any(isTRUE(metric_value))) {
        rel_metrics[[metric_name]] <- NULL
      }
    }
  }
  
  # Get names of the metrics
  rel_names <- names(rel_metrics)
  
  return(list(metrics = rel_metrics, names = rel_names))
}

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

add_network_to_ensemble <- function(ensembles, target_metric_name_best, removed_network, ensemble_number, worst_model_index) {
  
  # Extract performance and relevance metrics
  performance_metrics <- lapply(ensembles$temp_ensemble, function(x) x$performance_metric)
  relevance_metrics <- lapply(ensembles$temp_ensemble, function(x) x$relevance_metric)
  
  performance_names <- lapply(performance_metrics, function(x) names(x))
  relevance_names <- lapply(relevance_metrics, function(x) names(x))
  
  extract_and_combine_metrics <- function(metrics){
    # Combine the data into a single list of lists
    combined_data <- lapply(seq_along(metrics), function(i) {
      data <- metrics[[i]]
      # Flatten metrics with multiple values into named individual metrics
      flattened_data <- purrr::map2(data, names(data), function(metric, name) {
        if (length(metric) > 1) {
          # If metric has multiple values, name each value
          set_names(as.list(metric), paste0(name, "_", seq_along(metric)))
        } else {
          # If metric has a single value, keep it as is
          set_names(list(metric), name)
        }
      }) %>%
        flatten()
      return(flattened_data)
    })
    return(combined_data)
  }
  
  #fixed_relevance
  relevance_metrics <- extract_and_combine_metrics(relevance_metrics)
  
  # # Convert to a data frame
  # df <- bind_rows(combined_data)
  #
  # # Ensure all columns are of atomic types
  # df[] <- lapply(df, function(x) if (is.list(x)) unlist(x) else x)
  #
  # # Reshape the data into long format
  # df <- df %>%
  #     pivot_longer(cols = -c(Model_Name), names_to = "Metric", values_to = "Value")
  #
  # # Switch column order
  # df <- df[, c("Model_Name", "Metric", "Value")]
  #
  # # Convert "Value" column to numeric if possible
  # df$Value <- as.numeric(as.character(df$Value))
  #
  # # Filter out NA values
  # df <<- df[complete.cases(df), ]
  
  
  # Extract optimal epochs from ensembles$temp_ensemble
  extract_optimal_epoch <- lapply(ensembles$temp_ensemble, function(x) x$optimal_epoch)
  
  # # Step 3: Create a corresponding list for the optimal epochs
  # optimal_epochs_performance_metrics <- unlist(lapply(seq_along(extract_optimal_epoch), function(i) {
  #     rep(extract_optimal_epoch[[i]], length(performance_metrics[[i]]))
  # }))
  #
  # optimal_epochs_relevance_metrics <- unlist(lapply(seq_along(extract_optimal_epoch), function(i) {
  #     rep(extract_optimal_epoch[[i]], length(relevance_metrics[[i]]))
  # }))
  
  # Get run_id and model_iter_num_id from each element in ensembles$temp_ensemble
  runid <- unlist(lapply(ensembles$temp_ensemble, function(x) x$run_id))
  model_iter_num_id <- unlist(lapply(ensembles$temp_ensemble, function(x) x$model_iter_num))
  loss_increase_flag <- unlist(lapply(ensembles$temp_ensemble, function(x) x$loss_increase_flag))
  
  # Create a dataframe for performance metrics
  df_performance_metrics <- do.call(rbind, lapply(performance_metrics, data.frame))
  df_performance_metrics$runid <- runid
  df_performance_metrics$model_index <- model_iter_num_id
  df_performance_metrics$loss_increase_flag <- loss_increase_flag
  df_performance_metrics$Optimal_Epochs <- extract_optimal_epoch
  
  # Reshape performance metrics
  df_performance_metrics <- tidyr::gather(df_performance_metrics, key = "Metric", value = "Value", -runid, -model_index, -Optimal_Epochs, -loss_increase_flag)
  
  # Create a dataframe for relevance metrics
  df_relevance_metrics <- do.call(rbind, lapply(relevance_metrics, data.frame))
  df_relevance_metrics$runid <- runid
  df_relevance_metrics$model_index <- model_iter_num_id
  df_relevance_metrics$loss_increase_flag <- loss_increase_flag
  df_relevance_metrics$Optimal_Epochs <- extract_optimal_epoch
  
  # Reshape relevance metrics
  df_relevance_metrics <- tidyr::gather(df_relevance_metrics, key = "Metric", value = "Value", -runid, -model_index, -Optimal_Epochs, -loss_increase_flag)
  
  
  # Combine the two dataframes
  df_metrics <<- rbind(df_performance_metrics, df_relevance_metrics)
  
  
  # Find the best performing model
  best_performing_models <<- find_and_print_best_performing_models(performance_names, relevance_names, performance_metrics, relevance_metrics, target_metric_name_best)
  
  metric_to_vlookup <- best_performing_models$target_metric_name_best
  
  best_model_index <<- best_performing_models$best_model_index
  
  
  # Check if best_model_index is valid
  if (is.numeric(best_model_index) && (!is.infinite(best_model_index) || !is.na(best_model_index) || !is.null(best_model_index))) {
    # If best_model_index is valid, filter df_metrics and summarize
    result <<- df_metrics %>%
      filter(Metric == metric_to_vlookup, model_index == best_model_index, Optimal_Epochs > 1, loss_increase_flag == FALSE) %>%
      summarise(best_model_index = first(model_index))
    target_metric_name_best_value <<- df_metrics %>%
      filter(Metric == metric_to_vlookup, model_index == best_model_index) %>%
      summarise(Value = first(Value)) %>%
      pull(Value)
    optimal_epoch <<- df_metrics %>%
      filter(Metric == "Optimal_Epochs", model_index == best_model_index) %>%
      summarise(Value = first(Value)) %>%
      pull(Value)
    # Filter the data frame
    filtered_df <- df_metrics %>%
      filter(Metric == metric_to_vlookup, model_index == best_model_index)
    
    # Now you can access the loss_increase_flag value
    loss_increase_flag_value <- filtered_df$loss_increase_flag
    if((nrow(result) > 0 && !is.na(result)) || (nrow(result) > 0 && !is.null(result)) && (!is.null(optimal_epoch) || optimal_epoch > 1 || !is.na(optimal_epoch)) && loss_increase_flag_value == FALSE) {#if found best metric of interest out of the temp ... even if higher than the main ensemble ..we will account for this below
      best_model_index_new <<-  result$best_model_index
    }else{#if couldn't find best model in temp ensemble
      best_model_index_new <- NULL
      print("Optimal Epoch is <= 1 or does not exist and/or losses exceed initial loss.")
    }
    
  } else {
    best_model_index_new <- NULL
    # If best_model_index is invalid, assign NA to result
    result <<- NA
    target_metric_name_best_value <<- NA
  }
  
  
  # optimal_epoch <<- df_metrics %>%
  #     filter(Metric == "Optimal_Epochs", model_index == best_model_index) %>%
  #     summarise(Value = first(Value)) %>%
  #     pull(Value)
  #
  # loss_increase_flag_print <- df_metrics %>%
  #     filter(loss_increase_flag == TRUE, model_index == best_model_index) %>%
  #     summarise(Value = first(Value)) %>%
  #     pull(Value)
  #
  # print(paste0(metric_to_vlookup, " Metric Value: ", target_metric_name_best_value, ", Optimal Epoch: ", optimal_epoch, " Losses Exceed Intital Epoch:", loss_increase_flag_print))
  
  
  
  best_model_metadata <- list()
  
  # If a best-performing model is found
  if (!is.null(best_model_index_new)) {
    # Retrieve the best model and its associated metrics
    best_model <- ensembles$temp_ensemble[[best_model_index]]
    best_model$ensemble_number <- best_model$ensemble_number + 1
    
    
    best_model_metadata <- list(
      ensemble_index = ensemble_number + 1, #we add plus 1 because when hyperparameter_grid_setup == FALSE we did this for initalization too.
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
      weights_record = best_model$weights_record,
      biases_record = best_model$biases_record,
      weights_record2 = best_model$weights_record2,
      biases_record2 = best_model$biases_record2,
      lossesatoptimalepoch = best_model$lossesatoptimalepoch,
      loss_increase_flag = best_model$loss_increase_flag,
      performance_metric = best_model$performance_metric,
      relevance_metric = best_model$relevance_metric
    )
    
    cat("Adding model from temp ensemble to main ensemble based on metric:", target_metric_name_best, "\n")
    
    ensembles$main_ensemble[[worst_model_index]]$metadata <<- list()
    
    # Function to add a new layer of metadata dynamically
    add_metadata_layer <- function(model, new_metadata) {
      
      # Check if the first metadata slot is taken
      if (!is.null(model$metadata)) {
        # Initialize metadata counter
        iteration <- 2
        print((paste0("metadata", iteration)))
        # Loop to find the next available metadata slot
        while (!is.null(model[[paste0("metadata", iteration)]])) {
          iteration <- iteration + 1
        }
        
        # Add the new metadata to the next available slot
        model[[paste0("metadata", iteration)]] <- new_metadata
      } else {
        # If the first metadata slot is not taken, use it
        model$metadata <- new_metadata
      }
      
      return(model)
    }
    removed_network <<- removed_network
    removed_network_performance_metric <<- list(removed_network$performance_metric)
    removed_network_relevance_metric <<- extract_and_combine_metrics(list(removed_network$relevance_metric))
    
    # Create a dataframe for performance metrics
    removed_network_df_performance_metrics <<- do.call(rbind, lapply(removed_network_performance_metric, data.frame))
    
    # Reshape performance metrics
    removed_network_df_performance_metrics <<- tidyr::gather(removed_network_df_performance_metrics, key = "Metric", value = "Value")
    
    # Create a dataframe for relevance metrics
    removed_network_df_relevance_metrics <<- do.call(rbind, lapply(removed_network_relevance_metric, data.frame))
    
    # Reshape performance metrics
    removed_network_df_relevance_metrics <<- tidyr::gather(removed_network_df_relevance_metrics, key = "Metric", value = "Value")
    
    
    df_removed_network_performance_metrics_relevance_metrics <<- rbind(removed_network_df_performance_metrics, removed_network_df_relevance_metrics)
    
    removed_network_metric <- df_removed_network_performance_metrics_relevance_metrics %>%
      filter(Metric == metric_to_vlookup) %>%
      summarise(Value = first(Value)) %>%
      pull(Value)
    
    
    # Replace the removed network with the best model from the temp ensemble
    if (!is.null(removed_network) && target_metric_name_best_value < removed_network_metric && nrow(result) > 0 && !is.na(result)) { #if main is > temp, we want to remove the main and replace with temp
      # Debug: Print the index of the removed network
      print(paste("Index of the removed network:", worst_model_index))
      
      # Construct the variable name dynamically
      run_result_var_name <- paste0("run_results_1_", worst_model_index)
      
      # Create a list that includes the best model and its metadata
      best_model_with_removed_network_metadata <- list(best_model_metadata = best_model, metadata = removed_network) #incorporate the old metadata #had to add best_model_metadata = for accessing weights in intialization part of DESONN
      
      # Update the global variable run_results_1_<index> with the best model and the metadata of the model it's replacing
      assign(run_result_var_name, best_model_with_removed_network_metadata, envir = .GlobalEnv)
      
      ensembles$main_ensemble[[worst_model_index]] <- best_model
      # Add the removed network as a new layer of metadata in the replaced model / incorporate the old metadata
      updated_model <- add_metadata_layer(ensembles$main_ensemble[[worst_model_index]], removed_network)
      
      # Update the global ensemble list with the modified model
      ensembles$main_ensemble[[worst_model_index]] <<- updated_model #it was magically working without this line, but i was erroring out before maybe because of global <<- interferring with local <- so I added this line anyways, because logically this makes sense to me.
      
      # Debug: Print the metadata of the replaced model
      print(paste("Metadata of the replaced model at index", worst_model_index, ":"))
      
      
    } else if(!is.null(removed_network) && target_metric_name_best_value > removed_network_metric &&  (nrow(result) <= 0 || is.na(result))) { #if the temp ensemble's metric value is GREATER than the main ensemble's then put the removed network back in its place
      # Add the removed network back to the ensemble at its original position
      ensembles$main_ensemble <- append(ensembles$main_ensemble, list(removed_network), after = worst_model_index - 1)
      cat("Temp Ensemble based on metric:", target_metric_name_best, "with a value of:", target_metric_name_best_value , "is worse than Main Ensemble's", target_metric_name_best, "with a value of:",  removed_network_metric, "\n")
    }
  } else {
    # If no best model is found, print a message
    cat("No best model found in the temp ensemble based on metric:", target_metric_name_best, "\n")
  }
  
  # Ensure that temp_ensemble is ready for the next iteration (reuse or refresh as needed)
  ensembles$temp_ensemble <<- ensembles$temp_ensemble
  
  # Return the copy of the original main ensemble for backup or logging purposes
  return(ensembles$main_ensemble)
}

prune_network_from_ensemble <- function(ensembles, target_metric_name_worst) {
  
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
    ensembles$main_ensemble <- ensembles$main_ensemble[-worst_model_index]
    
    # Print information about the removed network
    cat("Removed network", worst_model_index, "from the main ensemble based on worst", target_metric_name_worst, "metric\n")
    
    return(list(removed_network = removed_network, updated_ensemble = ensembles, worst_model_index = worst_model_index))
  } else {
    cat("No worst-performing model found for metric:", target_metric_name_worst, "\n")
    return(NULL)
  }
}

# Loss Function: Computes the loss based on the type specified and includes regularization term
loss_function <- function(predictions, labels, reg_loss_total, loss_type) {
  print(dim(predictions))
  print(dim(labels))
  if (loss_type == "MSE") {
    # Mean Squared Error
    loss <- mean((predictions - labels)^2)
  } else if (loss_type == "MAE") {
    # Mean Absolute Error
    loss <- mean(abs(predictions - labels))
  } else if (loss_type == "CrossEntropy") {
    # Cross-Entropy Loss (Binary Classification)
    epsilon <- 1e-15  # Small value to avoid log(0)
    predictions <- pmax(pmin(predictions, 1 - epsilon), epsilon)  # Clip predictions
    loss <- -mean(labels * log(predictions) + (1 - labels) * log(1 - predictions))
  } else if (loss_type == "CategoricalCrossEntropy") {
    # Categorical Cross-Entropy Loss (Multi-Class Classification)
    epsilon <- 1e-15  # Small value to avoid log(0)
    predictions <- pmax(pmin(predictions, 1 - epsilon), epsilon)  # Clip predictions
    loss <- -mean(rowSums(labels * log(predictions)))
  } else {
    stop("Invalid loss type. Choose from 'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'.")
  }
  total_loss <- loss + reg_loss_total  # Add regularization term
  return(total_loss)
}

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#        _     _      _     _      _     _      _     _      _     _      _     _      _     _ $$$$$$$$$$$$$$$$$$$$$$$
#      (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c) $$$$$$$$$$$$$$$$$$$$$$$
#      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \   $$$$$$$$$$$$$$$$$$$$$$$
#   __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  $$$$$$$$$$$$$$$$$$$$$$$
#  (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._) $$$$$$$$$$$$$$$$$$$$$$$
#    || M ||      || E ||      || T ||      || R ||      || I ||      || C ||      || S ||     $$$$$$$$$$$$$$$$$$$$$$$
# _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._    $$$$$$$$$$$$$$$$$$$$$$$
#(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)   $$$$$$$$$$$$$$$$$$$$$$$
#`-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'     $$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

quantization_error <- function(SONN, Rdata, run_id) {
  
  if(ML_NN){
    # Ensure both SONN$weights and Rdata are numeric matrices
    if (!is.matrix(Rdata) || !is.matrix(SONN)) { #SONN matrix part might be wrong.
      if(!is.matrix(Rdata)){
        Rdata <- as.matrix(Rdata)
      }else if(!is.matrix(SONN)){
        SONN$weights <- as.matrix(SONN)
      }else{
        stop("Both SONN$weights and Rdata must be matrices.")
      }
      
      # Ensure SONN weights are numeric matrices
      if (!is.numeric(Rdata)) {
        if(!is.numeric(Rdata)){
          Rdata <- as.numeric(Rdata)
          
        }else{
          stop("Both SONN$weights and Rdata must be numeric matrices.")
        }
      }
    }
    # Calculate the distance between each Rdata point and its closest neuron
    distances <- apply(Rdata, 1, function(x) {
      # Calculate Euclidean distance between the input point and each neuron's weight vector
      neuron_distances <- apply(SONN, 1, function(w) {
        dist <- sqrt(sum((x - w)^2))  # Euclidean distance calculation
        dist  # Return the distance
      })
      
      # Return the minimum distance for the current Rdata point
      min_dist <- min(neuron_distances)
      return(min_dist)  # Return the minimum distance for the current Rdata point
    })
    
    
  }else{
    
    if (!is.matrix(Rdata) || !is.matrix(SONN$weights)) {
      if(!is.matrix(Rdata)){
        Rdata <- as.matrix(Rdata)
        
      }else if(!is.matrix(SONN$weights)){
        SONN$weights <- as.matrix(SONN$weights)
        
      }else{
        stop("Both SONN$weights and Rdata must be matrices.")
      }
      
      # Ensure SONN weights are numeric matrices
      if (!is.numeric(Rdata)) {
        if(!is.numeric(Rdata)){
          Rdata <- as.numeric(Rdata)
        }else if(!is.numeric(SONN)){
          SONN <- as.numeric(SONN)
        }else{
          stop("Both SONN$weights and Rdata must be numeric matrices.")
        }
      }
    }
    
    # Calculate the distance between each Rdata point and its closest neuron
    distances <- apply(Rdata, 1, function(x) {
      # Calculate Euclidean distance between the input point and each neuron's weight vector
      neuron_distances <- apply(SONN$weights, 1, function(w) {
        dist <- sqrt(sum((x - w)^2))  # Euclidean distance calculation
        dist  # Return the distance
      })
      
      # Return the minimum distance for the current Rdata point
      min_dist <- min(neuron_distances)
      
      return(min_dist)  # Return the minimum distance for the current Rdata point
    })
    
  }
  
  
  # Debugging step: Print the distances vector
  # print("Distances vector:")
  # print(distances)
  
  # Handle empty or all NA distances
  if (length(distances) == 0 || all(is.na(distances))) {
    warning("Distances vector is empty or contains only NA values")
    return(NA)
  }
  
  # Calculate the mean of distances, ignoring NA values
  mean_distance <- mean(distances, na.rm = TRUE)
  
  # Print the mean_distance to verify its value
  # print("Mean of distances:")
  if (verbose) {
    print("quantization error")
    print(paste("run_id:", run_id))
    print(mean_distance)
  }
  # Return the average distance, which measures how well the network represents the input data
  return(mean_distance)
  if (verbose) {
    print("quantization error complete")
  }
}

topographic_error <- function(SONN, map, Rdata, threshold) {
  
  if(ML_NN){
    # Check if SONN and Rdata are matrices
    if (!is.matrix(SONN) || !is.matrix(Rdata)) {
      if(!is.matrix(Rdata)){
        Rdata <- as.matrix(Rdata)
      }else if(!is.matrix(SONN)){
        SONN <- as.matrix(SONN)
      }else{
        stop("Both SONN$weights and Rdata must be matrices.")
      }
      
      # Ensure SONN weights are numeric matrices
      if (!is.numeric(SONN) || !is.numeric(Rdata)) {
        if(!is.numeric(Rdata)){
          Rdata <- as.numeric(Rdata)
        }else if(!is.numeric(SONN)){
          SONN <- as.numeric(SONN)
        }else{
          stop("Both SONN$weights and Rdata must be numeric matrices.")
        }
      }
    }
  }else{
    # Check if SONN and Rdata are matrices
    if (!is.matrix(SONN$weights) || !is.matrix(Rdata)) {
      if(!is.matrix(Rdata)){
        Rdata <- as.matrix(Rdata)
      }else if(!is.matrix(SONN$weights)){
        SONN$weights <- as.matrix(SONN$weights)
      }else{
        stop("Both SONN$weights and Rdata must be matrices.")
      }
      
      # Ensure SONN weights are numeric matrices
      if (!is.numeric(SONN$weights) || !is.numeric(Rdata)) {
        if(!is.numeric(Rdata)){
          Rdata <- as.numeric(Rdata)
        }else if(!is.numeric(SONN$weights)){
          SONN$weights <- as.numeric(SONN$weights)
        }else{
          stop("Both SONN$weights and Rdata must be numeric matrices.")
        }
      }}
  }
  
  
  if(ML_NN){
    # Calculate the topographic error
    errors <- apply(Rdata, 1, function(x) {
      # Calculate distances from the data point to all neurons
      distances <- apply(SONN, 1, function(w) {
        sqrt(sum((x - w)^2))
      })
      
      if (nrow(map) > 1) {
        # Check if SONN$map is a matrix
        if (!is.matrix(map)) {
          stop("SONN$map must be a matrix.")
        }
        
        # Find the indices of the first and second closest neurons
        closest_neurons <- order(distances)[1:2]
        
        # Check if the closest neurons are adjacent in the map
        error <- !is.adjacent(map, closest_neurons[1], closest_neurons[2])
        print(paste("Adjacency check for neurons", closest_neurons[1], "and", closest_neurons[2], ":", error))
      } else {
        
        # Check if the distance is greater than the threshold
        error <- min(distances) > threshold
        #print(paste("Distance check:", min(distances), ">", threshold, ":", error)) ####################do a count later
      }
      
      return(error)
    })
  } else {
    # Calculate the topographic error
    errors <- apply(Rdata, 1, function(x) {
      # Calculate distances from the data point to all neurons
      distances <- apply(SONN$weights, 1, function(w) {
        sqrt(sum((x - w)^2))
      })
      
      if (nrow(SONN$map) > 1) {
        # Check if SONN$map is a matrix
        if (!is.matrix(SONN$map)) {
          stop("SONN$map must be a matrix.")
        }
        
        # Find the indices of the first and second closest neurons
        closest_neurons <- order(distances)[1:2]
        
        # Check if the closest neurons are adjacent in the map
        error <- !is.adjacent(SONN$map, closest_neurons[1], closest_neurons[2])
        print(paste("Adjacency check for neurons", closest_neurons[1], "and", closest_neurons[2], ":", error))
      } else {
        
        # Check if the distance is greater than the threshold
        error <- min(distances) > threshold
        #print(paste("Distance check:", min(distances), ">", threshold, ":", error)) ####################do a count later
      }
      
      return(error)
    })
  }
  
  # # Debugging step: Print the errors vector
  # print("Errors vector:")
  # print(errors)
  
  # Handle empty or all NA errors
  if (length(errors) == 0 || all(is.na(errors))) {
    warning("Errors vector is empty or contains only NA values")
    return(NA)
  }
  
  # Calculate the mean of errors, ignoring NA values
  mean_error <- mean(errors, na.rm = TRUE)
  
  # Print the mean_error to verify its value
  # print("Mean of errors:")
  if (verbose) {
    print("topographic error")
    print(mean_error)
  }
  # Return the topographic error as a proportion
  return(mean_error)
  if (verbose) {
    print("topographic error complete")
  }
}

is.adjacent <- function(map, neuron1, neuron2) {
  # Find the positions of the neurons in the map
  pos1 <- which(map == neuron1, arr.ind = TRUE)
  pos2 <- which(map == neuron2, arr.ind = TRUE)
  
  # Debugging output for neuron positions
  # print(paste("Neuron positions:", neuron1, "at", pos1, "and", neuron2, "at", pos2))
  
  if (length(pos1) == 0 || length(pos2) == 0) {
    stop(paste("Neurons", neuron1, "or", neuron2, "not found in the map"))
  }
  
  # Calculate the grid distance between the two neurons
  grid_dist <- sum(abs(pos1 - pos2))
  
  # Neurons are adjacent if the grid distance is 1
  return(grid_dist == 1)
}

# Clustering quality (Davies-Bouldin index)
clustering_quality_db <- function(SONN, Rdata, cluster_assignments) {
  # Check if cluster_assignments is available
  if (is.null(cluster_assignments)) {
    stop("Cluster assignments not available. Perform kmeans clustering first.")
  }
  
  # Convert the SONN weights to a matrix (if needed)
  # mat_data <- as.matrix(SONN)
  
  # Calculate the centroids
  centroids <- aggregate(Rdata, by=list(cluster_assignments), FUN=mean)
  centroids <- centroids[,-1]  # Remove the grouping column
  
  # Initialize a matrix to store squared Euclidean distances between centroids
  n_clusters <- nrow(centroids)
  dist_mat <- matrix(0, nrow = n_clusters, ncol = n_clusters)
  
  # Calculate squared Euclidean distances between centroids
  for (i in 1:n_clusters) {
    for (j in 1:n_clusters) {
      if (i != j) {
        dist_mat[i, j] <- sum((centroids[i,] - centroids[j,])^2)
      }
    }
  }
  
  # Initialize variable to store Davies-Bouldin index
  db_index <- 0
  
  # Loop through each cluster
  for (i in 1:n_clusters) {
    # Initialize variable to store maximum inter-cluster separation
    max_inter_cluster_sep <- -Inf
    
    # Loop through other clusters
    for (j in 1:n_clusters) {
      if (i != j) {
        # Calculate intra-cluster dispersion
        s_i <- mean(sapply(which(cluster_assignments == i), function(x) sum((Rdata[x,] - centroids[i,])^2)))
        s_j <- mean(sapply(which(cluster_assignments == j), function(x) sum((Rdata[x,] - centroids[j,])^2)))
        
        # Calculate inter-cluster separation
        inter_cluster_sep <- (s_i + s_j) / sqrt(dist_mat[i, j])
        
        # Update maximum inter-cluster separation if needed
        if (inter_cluster_sep > max_inter_cluster_sep) {
          max_inter_cluster_sep <- inter_cluster_sep
        }
      }
    }
    
    # Update Davies-Bouldin index
    db_index <- db_index + max_inter_cluster_sep
  }
  
  # Calculate final Davies-Bouldin index
  db_index <- db_index / n_clusters
  
  if (verbose) {
    print("clustering_quality_db")
    print(db_index)
  }
  return(db_index)
  if (verbose) {
    print("clustering_quality_db complete")
  }
}

# Classification accuracy placeholder
MSE <- function(SONN, Rdata, labels, predicted_output) {
  
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
  if (verbose) {
    print("MSE")
    print(accuracy)
  }
  # Return the accuracy
  return(accuracy)
  if (verbose) {
    print("MSE complete")
  }
  
}

# Generalization ability
generalization_ability <- function(SONN, Rdata) {
  # Split the Rdata into training and testing sets
  set.seed(123)
  train_idx <- sample(1:nrow(Rdata), 0.8 * nrow(Rdata))
  train_Rdata <- Rdata[train_idx, ]
  test_Rdata <- Rdata[-train_idx, ]
  
  if (verbose) {
    print("generalization_ability")
  }
  if (verbose) {
    print("generalization_ability complete")
  }
}

# Speed
speed_learn <- function(SONN, learn_time) {
  
  if (verbose) {
    print("speed")
    print(learn_time)
  }
  return(learn_time)
  if (verbose) {
    print("speed complete")
  }
}

# Speed
speed <- function(SONN, prediction_time) {
  
  if (verbose) {
    print("speed")
    print(prediction_time)
  }
  return(prediction_time)
  if (verbose) {
    print("speed complete")
  }
}

# Memory usage
memory_usage <- function(SONN, Rdata) {
  
  # Calculate the memory usage of the SONN object
  object_size <- object.size(SONN)
  
  # Calculate the memory usage of the Rdata
  Rdata_size <- object.size(Rdata)
  if (verbose) {
    print("memory")
    print(object_size + Rdata_size)
  }
  # Return the total memory usage without the word "bytes"
  return(as.numeric(gsub("bytes", "", object_size + Rdata_size)))
  if (verbose) {
    print("memory complete")
  }
}

# Robustness
robustness <- function(SONN, Rdata, labels, lr, num_epochs, model_iter_num, predicted_output, ensemble_number) {
  
  # my_robustness_vector <<- seq(40, 0.1, by = -0.2)
  # losses <- rep(0, length(my_robustness_vector))
  
  # Add noise to the Rdata
  noisy_Rdata <<- as.matrix(Rdata + rnorm(n = nrow(Rdata) * ncol(Rdata), mean = 0, sd = 0.2)) #my_robustness_vector[200]))
  
  # Add outliers to the Rdata
  outliers <<- as.matrix(rnorm(n = 1.5 * ncol(noisy_Rdata), mean = 5, sd = 1), ncol = ncol(noisy_Rdata))
  
  noisy_Rdata[sample(1:nrow(noisy_Rdata), nrow(outliers)), ] <- outliers
  
  if(learnOnlyTrainingRun == FALSE){
    
    # Predict the class of the noisy Rdata
    noisy_Rdata_predictions <- SONN$predict(noisy_Rdata, labels, activation_functions, dropout_rates)
    
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
    accuracy <<- mean((noisy_Rdata_predictions$predicted_output - labels)^2)
    
    
  }else if (learnOnlyTrainingRun == TRUE){
    
    learn_r <- SONN$learn(noisy_Rdata, labels, lr)
  }
  
  # for(i in 1:length(my_robustness_vector)){
  #
  #
  #
  #
  # }
  if (verbose) {
    print("robustness")
    print(accuracy)
  }
  # Return the accuracy on the noisy Rdata
  return(accuracy)
  if (verbose) {
    print("robustness complete")
  }
  
}

# Hit Rate
hit_rate <- function(SONN, Rdata,  predicted_output) {
  # Predict the output for each Rdata point
  # print("hit rate before predict")
  # predictions <- SONN$predict(Rdata, labels)
  # print("hit rate after predict")
  Rdata <- data.frame(Rdata)
  # Identify the relevant Rdata points
  relevant_Rdata <<- Rdata[Rdata$class == "relevant", ]
  
  # Calculate the hit rate
  hit_rate <- sum(predicted_output %in% relevant_Rdata$id) / nrow(relevant_Rdata)
  if (verbose) {
    print("hit_rate")
    print(hit_rate)
  }
  # Return the hit rate
  return(hit_rate)
  if (verbose) {
    print("hit_rate complete")
  }
}

# Precision
precision <- function(SONN, Rdata, labels, predicted_output, verbose = FALSE) {
  # Predict the output for each Rdata point
  predictions <- SONN$predict(Rdata, labels, activation_functions)
  
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
  
  # Calculate percentage difference between actual and predicted values
  percentage_difference <<- abs(error_prediction) / abs(labels)
  
  # Check for NaNs in percentage_difference
  if (any(is.na(percentage_difference))) {
    print("NaNs produced; probably Inf issue error")
    return(NULL)  # Exit the function early if NaNs are found
  }
  
  # Define precision percentage bins
  bins <- c(0, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  bins_percentage <- bins / 100
  
  # Initialize counters for each bin
  bin_counts <- rep(0, length(bins) - 1)
  
  # Iterate through each prediction and assign it to the corresponding bin
  for (i in 1:length(percentage_difference)) {
    for (j in 1:(length(bins) - 1)) {
      if (percentage_difference[i] >= bins_percentage[j] && percentage_difference[i] < bins_percentage[j + 1]) {
        bin_counts[j] <- bin_counts[j] + 1
        break
      }
    }
  }
  
  bin_counts_review <<- bin_counts
  
  # Calculate precision for each bin
  precisions <<- bin_counts / sum(bin_counts)
  
  if (verbose) {
    print("precisions")
    print(precisions)
  }
  
  # Return the precisions for each bin
  return(precisions)
  
  if (verbose) {
    print("precisions complete")
  }
}

# Recall
recall <- function(SONN, Rdata, predicted_output) {
  # Predict the output for each Rdata point
  # print("recall before predict")
  # predictions <- SONN$predict(Rdata, labels)
  # print("recall after predict")
  Rdata <- data.frame(Rdata)
  # Identify the relevant Rdata points
  relevant_Rdata <- Rdata[Rdata$class == "relevant", ]
  
  # Calculate the recall
  recall <<- sum(predicted_output %in% relevant_Rdata$id) / nrow(relevant_Rdata)
  if (verbose) {
    print("recall")
  }
  # Return the recall
  return(recall)
  if (verbose) {
    print("recall complete")
  }
}

MAE <- function(SONN, Rdata, labels, predicted_output) {
  
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
  
  if (verbose) {
    print("MAE")
    print(accuracy)
  }
  # Return the accuracy
  return(accuracy)
  if (verbose) {
    print("MAE complete")
  }
}

# F1 Score
f1_score <- function(SONN, Rdata, labels) {
  # Calculate the precision and recall
  precision <- precision(SONN, Rdata, labels)
  recall <- recall(SONN, Rdata)
  
  # Calculate the F1-score
  f1_score <- 2 * (precision * recall) / (precision + recall)
  if (verbose) {
    print("f1_score")
    print(f1_score)
  }
  # Return the F1-score
  return(f1_score)
  if (verbose) {
    print("f1_score complete")
  }
}

# NDCG (Normalized Discounted Cumulative Gain)
ndcg <- function(SONN, Rdata, predicted_output) {
  # # Predict the output for each Rdata point
  # predictions <- SONN$predict(Rdata, labels)
  
  # Identify the relevant Rdata points
  relevant_Rdata <- Rdata[Rdata$class == "relevant", ]
  
  # Calculate the discounted cumulative gain (DCG)
  dcg <- sum(2^rel - 1)
  
  # Calculate the ideal discounted cumulative gain (IDCG)
  idcg <- sum(2^(sort(rel, decreasing = TRUE)) - 1)
  
  # Calculate the NDCG
  ndcg <- dcg / idcg
  if (verbose) {
    print("ndcg")
    print(ndcg)
  }
  # Return the NDCG
  return(ndcg)
  if (verbose) {
    print("ndcg complete")
  }
}

# MAP (Mean Average Precision)
mean_precision <- function(SONN, Rdata, labels, predicted_output) {
  
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
  
  
  # Calculate percentage difference between actual and predicted values
  percentage_difference <<- abs(lerror_prediction) / abs(labels)
  
  # Define precision percentage bins
  bins <- c(0, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  
  bins_percentage <<- bins / 100
  
  # Initialize a list to store precision values for each bin
  bin_counts <<- vector("list", length(bins) - 1)
  
  # Iterate through each prediction and assign it to the corresponding bin
  for (i in 1:length(percentage_difference)) {
    for (j in 1:(length(bins) - 1)) {
      if (percentage_difference[i] >= bins_percentage[j] && percentage_difference[i] < bins_percentage[j + 1]) {
        # Append the precision value to the corresponding bin
        bin_counts[[j]] <- c(bin_counts[[j]], percentage_difference[i])
        break
      }
    }
  }
  bin_counts_review2 <<- bin_counts
  
  bin_counts[sapply(bin_counts, is.null)] <- NA
  # Calculate mean precision for each bin
  mean_precisions <- sapply(bin_counts, mean, na.rm = TRUE)
  mean_precisions[is.na(mean_precisions)] <- 0
  if (verbose) {
    print("mean_precisions")
    print(mean_precisions)
  }
  # Return the mean precisions for each bin
  return(mean_precisions)
  if (verbose) {
    print("mean_precisions complete")
  }
}

# Diversity
diversity <- function(SONN, Rdata, predicted_output, verbose = FALSE) {
  # Ensure there are no zero values in predicted_output to avoid log2(0)
  predicted_output[predicted_output == 0] <- .Machine$double.eps
  
  # Calculate the entropy of the predictions
  entropy <- suppressWarnings(-sum(predicted_output * log2(predicted_output)))
  suppressWarnings({
    # Handle Inf and NaN values
    if (is.infinite(entropy) || is.nan(entropy)) {
      entropy <- NA
      warning("Calculated entropy is Inf or NaN. Returning NA.")
    }
  })
  if (verbose) {
    print("diversity")
    print(entropy)
  }
  
  # Return the entropy
  return(entropy)
  
  if (verbose) {
    print("diversity complete")
  }
}

# Novelty placeholder
RMSE <- function(SONN, Rdata, labels, predicted_output) {
  # Calculate the squared error between each prediction and its corresponding label
  squared_errors <- mapply(function(x, y) {
    (x - y)^2  # Squared error calculation
  }, predicted_output, labels)
  
  # Calculate the mean squared error
  mean_squared_error <- mean(squared_errors, na.rm = TRUE)
  
  # Calculate the RMSE
  rmse <- sqrt(mean_squared_error)
  
  if (verbose) {
    print("RMSE")
    print(rmse)
  }
  return(rmse)
  if (verbose) {
    print("RMSE complete")
  }
}

# Serendipity
serendipity <- function(SONN, Rdata, predicted_output) {
  # # Predict the output for each Rdata point
  # predictions <- SONN$predict(Rdata, labels)
  
  # Calculate the average number of times each prediction is made
  prediction_counts <- table(predicted_output)
  
  # Calculate the inverse of the prediction counts
  inverse_prediction_counts <- 1 / prediction_counts
  if (verbose) {
    print("serendipity")
    print(mean(inverse_prediction_counts, na.rm = TRUE))
  }
  # Return the average inverse prediction count
  return(mean(inverse_prediction_counts, na.rm = TRUE))
  if (verbose) {
    print("serendipity complete")
  }
}

# Helper function to adjust biases for the first layer of the predict function
# Top-level helper
adjust_biases_layer_1 <- function(biases, weights, Rdata) {
  biases <- as.numeric(unlist(biases))
  input_rows <- as.integer(nrow(Rdata))
  output_cols <- as.integer(ncol(weights))
  
  if (length(biases) == 1) {
    cat("Using single bias value:", biases, "\n")
    matrix(biases, nrow = input_rows, ncol = output_cols, byrow = TRUE)
  } else if (length(biases) < output_cols) {
    cat("Bias length and neuron count mismatch. Adjusting (replicating)...\n")
    matrix(rep(biases, length.out = output_cols), nrow = input_rows, ncol = output_cols, byrow = TRUE)
  } else if (length(biases) > output_cols) {
    cat("Bias length exceeds neuron count. Truncating...\n")
    matrix(biases[1:output_cols], nrow = input_rows, ncol = output_cols, byrow = TRUE)
  } else {
    matrix(biases, nrow = input_rows, ncol = output_cols, byrow = TRUE)
  }
}






