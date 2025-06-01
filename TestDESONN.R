source("DESONN.R")
# source("DESONN_20240629_v6.R")
# Initialize activation functions
# self$activation_functions <- vector("list", self$num_layers)
# self$activation_functions_learn <- vector("list", self$num_layers)
library(caret)
###############################################################################
function(test1){
# test1
# Define input_size, output_size, and num_networks
# Generate synthetic Rdata for training
# ML_NN <- TRUE
# ML_NN <- FALSE
# numeric_columns <- NULL
# normalize <- TRUE
# # numeric_columns <- c('age', 'creatinine_phosphokinase', 'ejection_fraction',
#                                                    # 'platelets', 'serum_creatinine', 'serum_sodium', 'time')
# total_num_samples <- 1500
# input_size <- 10
# hidden_sizes <- c(1,5)
# # dropout_rates <- c(0.1,0.2,0.3)
# dropout_rates <- NULL
# dropout_rates_learn <- dropout_rates
# # hidden_sizes <- NULL
# output_size <- 1
# num_networks <- 1
# if(!ML_NN){
#     N <- input_size + output_size  # Multiplier for data generation (not directly applicable here)
# }else{
#     N <- input_size + sum(hidden_sizes) + output_size
# }
# threshold <- 0.98
# num_samples <- if (!missing(total_num_samples)) total_num_samples else num_samples
# # Define the number of samples for each set
# num_validation_samples <- 200
# num_test_samples <- 300
# num_training_samples <- total_num_samples - numvalidation_samples - num_test_samples
#
# # Create a random permutation of row indices
# indices <- sample(1:total_num_samples)
#
# # Split the indices into training, validation, and test sets
# train_indices <- indices[1:num_training_samples]
# validation_indices <- indices[(num_training_samples + 1):(num_training_samples + num_validation_samples)]
# test_indices <- indices[(num_training_samples + num_validation_samples + 1):total_num_samples]
#
# # # Base seed
# # base_seed <- 437
# #
# # # Generate datasets
# # set.seed(base_seed)
# X <- matrix(runif(total_num_samples * input_size), ncol = input_size)
# y <- as.matrix(apply(X, 1, function(x) sum(sin(x)) + rnorm(1, sd = 0.1)))
#
# # # Create training, validation, and test sets
# X_train <- X[train_indices, ]
# y_train <- y[train_indices, ]
#
# # X_validation <- X[validation_indices, ]
# # y_validation <- y[validation_indices, ]
# #
# # X_test <- X[test_indices, ]
# # y_test <- y[test_indices, ]
#
# X <- as.matrix(X_train)
# y <- as.matrix(y_train)
}

# # Define parameters
init_method <- "xavier" #variance_scaling" #glorot_uniform" #"orthogonal" #"orthogonal" #lecun" #xavier"
optimizer <- "adam" #"lamb" #ftrl #nag #"sgd" #NULL "rmsprop" #adam
lookahead_step <- 100
batch_normalize_data <- FALSE
shuffle_bn <- TRUE
gamma_bn <- 1
beta_bn <- 1
epsilon_bn <- 1e-9  # Increase for numerical stability
momentum_bn <- 1.9  # Improved convergence
is_training_bn <- TRUE
beta1 <- 0.9  # Standard Adam value
beta2 <- 0.997  # Slightly lower for better adaptabilit


custom_scale <- .2
# epsilon <- 1e-5
# ML_NN <- TRUE
ML_NN <- TRUE

# hidden_sizes <- NULL
hidden_sizes <- c(16, 8)
input_size <- 12
#, 1, 1, 10) #,2,1,, 1)
activation_functions <- list(NULL, bent_identity, sigmoid)



activation_functions_learn <- list(NULL, bent_identity, sigmoid) #list("elu", bent_identity, "sigmoid") # list(NULL, NULL, NULL, NULL) #activation_functions #list("relu", "custom_activation", NULL, "relu")  #"custom_activation"
epsilon <- 1e-8
loss_type <- "MSE" #'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'
# activation_functions_learn <- list(NULL, "sigmoid", NULL, "sigmoid", NULL)
# dropout_rates <- c(0.1,0.2,0.3)
# Create a list of activation function names as strings
# activation_functions <- NULL # list("relu", "relu",  "relu", "sigmoid", "sigmoid_binary", "relu", "sigmoid_binary")
# activation_functions_learn <- activation_functions
dropout_rates <- list(0.3, 0.15)  # NULL for output layer
#c(0.2, 0.3, 0.3) #c(0.2, 0.3, 0.3) #c(0.5, 0.5, 0.5)#NULL #c(89.91, 90.48, 11)
dropout_rates_learn <- dropout_rates
# hidden_sizes <- NULL
num_layers <- length(hidden_sizes) + 1
output_size <- 1  # For binary classification
num_networks <- 1  # Number of trees in the Random Forest
# Create a list of activation functions
# numeric_columns <- c('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time')
# numeric_columns <- NULL
# Ensure both vectors are of the same length
# if (length(hidden_sizes) + 1 != length(activation_functions)) {
#     stop("Length of hidden_sizes and activation_functions must be equal")
# }
if(!ML_NN){
     N <- input_size + output_size  # Multiplier for data generation (not directly applicable here)
}else{
     N <- input_size + sum(hidden_sizes) + output_size
}
threshold <- 0.98  # Classification threshold (not directly used in Random Forest)

# Load the dataset
data <- read.csv("C:/Users/wfky1/Downloads/heart_failure_clinical_records.csv")

# Check for missing values
sum(is.na(data))

# Assuming there are no missing values, or handle them if they exist
# Convert categorical variables to factors if any
data <- data %>%
     mutate(across(where(is.character), as.factor))

# Split the data into features (X) and target (y)
X <- data %>% dplyr::select(-DEATH_EVENT)
y <- data %>% dplyr::select(DEATH_EVENT)

# cols_to_remove <- names(p_values)[p_values > 0.05]
#
# # Remove those columns from the dataset
# X_reduced <- X %>% select(-all_of(cols_to_remove))

# columns_to_remove <- c("anaemia", "diabetes", "high_blood_pressure", "platelets", "serum_sodium", "sex", "smoking", "time")

# Remove specified columns
# data_reduced <- data %>% select(-all_of(columns_to_remove))
numeric_columns <- c('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
'serum_creatinine', 'serum_sodium', 'time')

# Split the data into features (X) and target (y)
# X <- data_reduced %>% select(-DEATH_EVENT)
y <- data %>% dplyr::select(DEATH_EVENT)
colname_y <- colnames(y)
# Define the number of samples for each set
set.seed(16675)
total_num_samples <- nrow(data)
# Define num_samples
num_samples <- if (!missing(total_num_samples)) total_num_samples else num_samples
num_validation_samples <- 500
num_test_samples <- 500
num_training_samples <- total_num_samples - num_validation_samples - num_test_samples

# Create a random permutation of row indices
indices <- sample(1:total_num_samples)

# Split the indices into training, validation, and test sets
train_indices <- indices[1:num_training_samples]
validation_indices <- indices[(num_training_samples + 1):(num_training_samples + num_validation_samples)]
test_indices <- indices[(num_training_samples + num_validation_samples + 1):total_num_samples]

# Create training, validation, and test sets
X_train <- X[train_indices, ]
y_train <- y[train_indices, ]

X_validation <- X[validation_indices, ]
y_validation <- y[validation_indices, ]

X_test <- X[test_indices, ]
y_test <- y[test_indices, ]

#$$$$$$$$$$$$$ Feature scaling without leakage
X_train_scaled <- scale(X_train)
center <- attr(X_train_scaled, "scaled:center")
scale_ <- attr(X_train_scaled, "scaled:scale")
X_test_scaled <- scale(X_test, center = center, scale = scale_)
X_validation_scaled <- scale(X_validation, center = center, scale = scale_)

#$$$$$$$$$$$$$ Overwrite training matrix for model training
X <- as.matrix(X_train_scaled)
y <- as.matrix(y_train)


# X <- as.matrix(X_test)
# y <- as.matrix(y_test)
colnames(y) <- colname_y

binary_flag <- is_binary(y)

# # Perform Random Forest-based feature selection on training data
# library(randomForest)
# 
# rf_data <- as.data.frame(X)
# rf_data$DEATH_EVENT <- as.factor(y)
# 
# set.seed(42)
# rf_model <- randomForest(DEATH_EVENT ~ ., data = rf_data, importance = TRUE)
# 
# # Compute feature importance and select features above median
# importance_scores <- importance(rf_model, type = 2)[, 1]  # MeanDecreaseGini
# threshold <- mean(importance_scores)
# selected_features <- names(importance_scores[importance_scores > threshold])
# 
# # Filter feature matrix to selected important features
# X <- as.matrix(rf_data[, selected_features, drop = FALSE])
# 
# # Update input size for neural network initialization
# input_size <- ncol(X)
# 
# numeric_columns <- intersect(numeric_columns, selected_features)



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#    ██████  ██████  ███    ██ ████████ ██████   ██████  ██          ██████   █████  ███    ██ ███████ ██  $$$$$$$$$$$$$$
#  ██      ██    ██ ████   ██    ██    ██   ██ ██    ██ ██          ██   ██ ██   ██ ████   ██ ██      ██   $$$$$$$$$$$$$$
# ██      ██    ██ ██ ██  ██    ██    ██████  ██    ██ ██          ██████  ███████ ██ ██  ██ █████   ██    $$$$$$$$$$$$$$
#██      ██    ██ ██  ██ ██    ██    ██   ██ ██    ██ ██          ██      ██   ██ ██  ██ ██ ██      ██     $$$$$$$$$$$$$$
#██████  ██████  ██   ████    ██    ██   ██  ██████  ███████     ██      ██   ██ ██   ████ ███████ ███████ $$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
showMeanBoxPlots <- FALSE
Losses_At_Optimal_Epoch_filenumber <- 3
writeTofiles <- FALSE
#########################################################################################################################
target_metric_name_best <- 'MSE' #<<-
target_metric_name_worst <- 'MSE' #<<-
metric_name <- 'MSE'
#########################################################################################################################

nruns <- 5
verbose <<- FALSE
hyperparameter_grid_setup <- TRUE
reg_type = "L1_L2" #Max_Norm" #"Group_Lasso" #"L1_L2"
olr <- FALSE
# input_size <- 13 # This should match the actual number of features in your data
# hidden_size <- 2
loading_ensemble_1_run_ids <- FALSE
#########################################################################################################################
never_ran_flag <- TRUE

results <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)
# if(hyperparameter_grid_setup){
#     loading_ensemble_1_run_ids <- TRUE #change back to false
# }else if(!hyperparameter_grid_setup){
# loading_ensemble_1_run_ids <- TRUE
# }
# results <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)
# Iterate over each row of the hyperparameter grid
# for (j in 1:nruns) {
for(j in 1:1){
# Initialize global variable
increment_loop_flag <- FALSE
# for (j in 1:nrow(hyperparameter_grid)) {
# for (j in 1:nrow(new_grid)) {
# for (j in 1:nrow(new_grid2)) {

    # Check if the global variable is set
    if (increment_loop_flag) {
        # Increment j by 1
        j <- j + 1

        # Reset the global variable
        increment_loop_flag <- FALSE
    }


#     if(loading_ensemble_1_run_ids){
#
#         # Load the data from the RDS file
#         loaded_run_results <- readRDS("run_results_saved_63.rds") #56
#
#         # Assign each element of the loaded list to individual variables $performance_metric$MSE
#         lapply(loaded_run_results, function(result) result$performance_metric$MSE)
#
#         # Update the run_id for each sublist
#         loaded_run_results <- lapply(seq_along(loaded_run_results), function(i) {
#             sublist <- loaded_run_results[[i]]
#             sublist$run_id <- paste("Ensemble: 1 Model:", i)
#             return(sublist)
#         })
#         for(i in 1:length(loaded_run_results)) { # don't need the ensemble_number from the hyperparameter_grid_setup loop
#             loaded_run_results[[i]]$ensemble_number <- 1
#         }
#         # Assign each updated sublist to individual variables
#         official_run_results_1_1 <- loaded_run_results[[1]]
#         # official_run_results_1_2 <- loaded_run_results[[2]]
#         # official_run_results_1_3 <- loaded_run_results[[3]]
#         # official_run_results_1_4 <- loaded_run_results[[4]]
#         # official_run_results_1_5 <- loaded_run_results[[5]]
# #official is for predicting
# official_run_results <- list(
#         official_run_results_1_1
#         # official_run_results_1_2,
#         # official_run_results_1_3,
#         # official_run_results_1_4,
#         # official_run_results_1_5
#     )
#
#     # Extract optimal_epoch values
#     optimal_epochs <- sapply(official_run_results, function(x) x$optimal_epoch)
#         loading_ensemble_1_run_ids <- FALSE
#     }
#
#      run_results_1_1 <- official_run_results_1_1
    # run_results_1_2 <- official_run_results_1_2
    # run_results_1_3 <- official_run_results_1_3
    # run_results_1_4 <- official_run_results_1_4
    # run_results_1_5 <- official_run_results_1_5

    hyperparameter_grid_setup <- hyperparameter_grid_setup #FALSE means ready for temp

    if(hyperparameter_grid_setup){
        # Initialize ensembles list
        ensembles_hyperparameter_grid <- list()  # Initialize temporary ensemble as an empty list
        lr1 <- 0.0005 #c(0.001, 0.01, 0.1) #0.00001, 0.0001,
        lambda1 <- 0.0001 #c(0.01, 0.001, 0.0001, 0.00001) #1, 0.1,// Calculate the factorial of a number using a recursive function
        hyperparameter_grid <- expand.grid(lr = lr1, lambda = lambda1) %>%
            mutate_all(~ format(., scientific = FALSE))

        # Get lr and lambda values for the current row
        lr <- as.numeric(hyperparameter_grid[j, "lr"])
        lambda <- as.numeric(hyperparameter_grid[j, "lambda"])
        print(hyperparameter_grid)

    }else{
        # lr <- 0.001
        # lr <- as.numeric(lr)
        # lambda <- 0.1
        # lambda <- as.numeric(lambda)
        lr <- c(0.00001, 0.0001, 0.001, 0.01, 0.1, 1)
        lambda <- c(1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
        hyperparameter_grid <- expand.grid(lr = lr, lambda = lambda) %>%
            mutate_all(~ format(., scientific = FALSE))
        lr <- as.numeric(hyperparameter_grid[j, "lr"])
        lambda <- as.numeric(hyperparameter_grid[j, "lambda"])

        # Initialize the ensembles list with main_ensemble containing the five lists
        ensembles <<- list(
            main_ensemble = list(
                run_results_1_1),
            #     run_results_1_2,
            #     run_results_1_3,
            #     run_results_1_4,
            #     run_results_1_5
            # ),
            temp_ensemble = list() # Initialize temporary ensemble as an empty list
        )
    }

#if getting loading the best models, makes sense to

    #Im going to try running grid on weights constant and then run on looking for best metric best and worst ensemble

    # run_results_1_1 <- Ensemble_2_model_1
    # run_results_1_2 <- Ensemble_2_model_2
    # run_results_1_3 <- Ensemble_2_model_3
    # run_results_1_4 <- Ensemble_2_model_4
    # run_results_1_5 <- Ensemble_2_model_5
    # Initialize a data frame to store the results


plot_robustness <- FALSE
predict_models <- FALSE
use_loaded_weights <- FALSE
saveToDisk <- FALSE
Run1 <- TRUE
Run2 <- FALSE
Run3 <- FALSE
Run1.1 <- FALSE
Run1.2 <- FALSE

if (Run1 == TRUE && Run2 == FALSE && Run3 == FALSE && Run1.1 == FALSE && hyperparameter_grid_setup){
    learnOnlyTrainingRun <- FALSE
    update_weights <- TRUE
    update_biases <- TRUE
    never_ran_flag <- TRUE
}else if (Run1 == FALSE && Run2 == TRUE && Run3 == FALSE && Run1.1 == FALSE && hyperparameter_grid_setup){
    learnOnlyTrainingRun <- FALSE
    update_weights <- FALSE
    update_biases <- FALSE
    never_ran_flag <- FALSE

}else if (Run1 == FALSE && Run2 == FALSE && Run3 == TRUE && Run1.1 == FALSE && hyperparameter_grid_setup){
    learnOnlyTrainingRun <- TRUE
    update_weights <- FALSE
    update_biases <- FALSE
    never_ran_flag <- TRUE

#ENSEMBLE LOOPING
}else if (Run1 == FALSE && Run2 == FALSE && Run3 == FALSE && Run1.1 == TRUE && !hyperparameter_grid_setup){ #Error in match(x, table, nomatch = 0L) : 'match' requires vector arguments
    learnOnlyTrainingRun <- FALSE
    update_weights <- TRUE
    update_biases <- TRUE
    never_ran_flag <- FALSE #this is good bc it gets the weights & biases from run_results_1_1 - run_results_1_5

#PREDICTING
}else if (Run1 == FALSE && Run2 == FALSE && Run3 == FALSE && Run1.2 == TRUE && !hyperparameter_grid_setup && predict_models){
    learnOnlyTrainingRun <- FALSE
    update_weights <- FALSE
    update_biases <- FALSE
    never_ran_flag <- FALSE
    predict_models <- predict_models
}

if(never_ran_flag == FALSE) { #length(my_optimal_epoch_out_vector) > 1
    cat("________________________________________new_run_", j, "______________________________________________\n", sep = "")

    if ((!exists("my_optimal_epoch_out_vector") || is.null(my_optimal_epoch_out_vector[[1]]) || is.na(my_optimal_epoch_out_vector[[1]])) && hyperparameter_grid_setup  && !use_loaded_weights) {
        my_optimal_epoch_out_vector <- vector("list", length(1:num_networks))  # Initialize your list
        my_optimal_epoch_out_vector <- num_epochs_check
    }else if (!exists("my_optimal_epoch_out_vector") && use_loaded_weights) {
        my_optimal_epoch_out_vector <- vector("list", length(1:num_networks))  # Initialize your list
        # Extract optimal_epoch values
        my_optimal_epoch_out_vector <- optimal_epochs
        #if you have stored weights/biases/metadata you want to load from run_results_1_1 - run_results_1_5
    }


    if(hyperparameter_grid_setup){
    #num_epochs <- max(unlist(my_optimal_epoch_out_vector)) + 7
    num_epochs <- my_optimal_epoch_out_vector #<<-
    #num_epochs_max <<- max(unlist(my_optimal_epoch_out_vector))
    }else{
        num_epochs <- 100 #<<-
    }

    if(ML_NN){
        DESONN_model_2 <- DESONN$new(num_networks = num_networks, input_size = input_size, hidden_sizes = hidden_sizes, output_size = output_size, N = N, lambda = lambda, ensemble_number = ensemble_number, ML_NN = ML_NN, method = init_method, custom_scale = custom_scale)
    }else{
        DESONN_model_2 <- DESONN$new(num_networks = num_networks, input_size = input_size, output_size = output_size, N = N, lambda = lambda, ensemble_number = ensemble_number, ML_NN = ML_NN, method = init_method, custom_scale = custom_scale)
    }
    DESONN_model <- DESONN_model_2 #<<-
    SecondRunDESONN <- DESONN_model_2$train(X, y, lr, ensemble_number = j, num_epochs, threshold, reg_type, numeric_columns = numeric_columns, activation_functions_learn = activation_functions_learn, activation_functions = activation_functions,
                                            dropout_rates_learn = dropout_rates_learn, dropout_rates = dropout_rates, optimizer = optimizer, beta1 = beta1, beta2 = beta2, epsilon = epsilon, lookahead_step = lookahead_step, batch_normalize_data = batch_normalize_data, gamma_bn = gamma_bn, beta_bn = beta_bn,
                                            epsilon_bn = epsilon_bn, momentum_bn = momentum_bn, is_training_bn = is_training_bn, shuffle_bn = shuffle_bn, loss_type = loss_type)
    if (SecondRunDESONN$loss_status == 'exceeds_10000') {
        next
    }



if(!predict_models){
    if(hyperparameter_grid_setup){
        ensembles_hyperparameter_grid[[j]] <- results_list
    }else {
        print("___________________________PRUNE_ADD________________________________")
        prune_result <- prune_network_from_ensemble(ensembles, target_metric_name_worst) #<<-
        main_ensemble_copy_return <- add_network_to_ensemble(prune_result$updated_ensemble, target_metric_name_best, prune_result$removed_network, ensemble_number = j, prune_result$worst_model_index) #<<-
        print("___________________________PRUNE_ADD_end____________________________")

        # Initialize lists to store run_ids and MSE performance metrics
        run_ids <- list()
        mse_metrics <- list()

        # Loop over each ensemble
        for (i in seq_along(ensembles$main_ensemble)) {
            # Check if best_model_metadata exists in the i-th ensemble
            if (!is.null(ensembles$main_ensemble[[i]]$best_model_metadata)) {
                # Extract run_id and MSE performance metric from best_model_metadata
                run_ids[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$run_id
                mse_metrics[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$performance_metric$MSE
            } else {
                # Extract run_id and MSE performance metric from the i-th ensemble
                run_ids[[i]] <- ensembles$main_ensemble[[i]]$run_id
                mse_metrics[[i]] <- ensembles$main_ensemble[[i]]$performance_metric$MSE
            }
        }

        # Convert lists to dataframe
        results_df <- data.frame(run_id = unlist(run_ids), mse_metric = unlist(mse_metrics))

        # Print the dataframe
        print("___________________________results_df________________________________")
        print(results_df)
        print("___________________________results_df_end____________________________")

        # Save the dataframe as an RDS file
        saveRDS(results_df, file = "results_df.rds")

        # After finishing the inner loop, update the temporary ensemble (second list) in ensembles
        ensembles[[2]] <- ensembles$temp_ensemble
    }
}else{
    # Initialize lists to store run_ids and MSE performance metrics
    run_ids <- list()
    mse_metrics <- list()

    # Loop over each ensemble
    for (i in seq_along(ensembles$main_ensemble)) {
        # Check if best_model_metadata exists in the i-th ensemble
        if (!is.null(ensembles$main_ensemble[[i]]$best_model_metadata)) {
            # Extract run_id and MSE performance metric from best_model_metadata
            run_ids[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$run_id
            mse_metrics[[i]] <- ensembles$main_ensemble[[i]]$best_model_metadata$performance_metric$MSE
        } else {
            # Extract run_id and MSE performance metric from the i-th ensemble
            run_ids[[i]] <- ensembles$main_ensemble[[i]]$run_id
            mse_metrics[[i]] <- ensembles$main_ensemble[[i]]$performance_metric$MSE
        }
    }

    # Initialize lists to store run_ids and MSE performance metrics
    run_ids <- list()
    mse_metrics <- list()

    # Convert lists to dataframe
    results_df <- data.frame(run_id = unlist(run_ids), mse_metric = unlist(mse_metrics))

    # Print the dataframe
    print("___________________________results_df________________________________")
    print(results_df)
    print("___________________________results_df_end____________________________")

    # Save the dataframe as an RDS file
    saveRDS(results_df, file = "results_df_predict.rds")

}
    print("old vector used")
    #print(findout)
    # print(head(X))
}else{ #if never ran before



    if (!exists("my_optimal_epoch_out_vector")) {
        my_optimal_epoch_out_vector <- vector("list", length(1:num_networks))  # Initialize your list


    }


    # print(head(X))
    num_epochs <- 4
    never_ran_flag <- TRUE
    # Train your DESONN model

    if(ML_NN){
        DESONN_model_1 <- DESONN$new(num_networks = num_networks, input_size = input_size, hidden_sizes = hidden_sizes, output_size = output_size, N = N, lambda = lambda, ensemble_number = ensemble_number, ML_NN = ML_NN, method = init_method, custom_scale = custom_scale)

        }else{
        DESONN_model_1 <- DESONN$new(num_networks = num_networks, input_size = input_size, output_size = output_size, N = N, lambda = lambda, ensemble_number = ensemble_number, ML_NN = ML_NN, method = init_method, custom_scale = custom_scale)

        }

    DESONN_model <- DESONN_model_1
    FirstRunDESONN <- DESONN_model_1$train(X, y, lr, ensemble_number = j, num_epochs, threshold, reg_type, numeric_columns = numeric_columns, activation_functions_learn = activation_functions_learn, activation_functions = activation_functions,
                                            dropout_rates_learn = dropout_rates_learn, dropout_rates = dropout_rates, optimizer = optimizer, beta1 = beta1, beta2 = beta2, epsilon = epsilon, lookahead_step = lookahead_step, batch_normalize_data = batch_normalize_data, gamma_bn = gamma_bn, beta_bn = beta_bn,
                                            epsilon_bn = epsilon_bn, momentum_bn = momentum_bn, is_training_bn = is_training_bn, shuffle_bn = shuffle_bn, loss_type = loss_type)
    if (FirstRunDESONN$loss_status == 'exceeds_10000') {
        next
    }


    # Update the results data frame
    results <- rbind(results, data.frame(lr = lr, lambda = lambda, accuracy = FirstRunDESONN$accuracy)) #<<-

    # Save the results data frame to an RDS file
    saveRDS(results, file = "results.rds")

    if(hyperparameter_grid_setup){
        ensembles_hyperparameter_grid[[j]] <- results_list
    }else{
        print("before prune and add")
        # Prune the worst-performing model
        pruned_result <- prune_network_from_ensemble(ensembles, target_metric_name_worst) #<<-

        # Check if a model was removed and if so, add a new model
        if (!is.null(pruned_result$removed_network)) {
            # Call add_network_to_ensemble to add a new model
            main_ensemble_copy_return <- add_network_to_ensemble(pruned_result$updated_ensemble, target_metric_name_best, pruned_result$removed_network, ensemble_number = j, pruned_result$worst_model_index) #<<-

            print("___________________________________")
            pruned_result$removed_network$run_id
            pruned_result$removed_network$ensemble_number
            pruned_result$removed_network$model_iter_num
            print("___________________________________")
            ensembles$main_ensemble[[1]]$run_id
            ensembles$main_ensemble[[2]]$run_id
            ensembles$main_ensemble[[3]]$run_id
            ensembles$main_ensemble[[4]]$run_id
            ensembles$main_ensemble[[5]]$run_id
            print("___________________________________")


            } else {
            cat("No model removed. No need to add a new model.\n")
        }

        print("after prune and add")

        # After finishing the inner loop, update the temporary ensemble (second list) in ensembles
        #ensembles[[2]] <- ensembles$temp_ensemble
    }

    print("num_epochs <- 100 initialized")
    #print(findout)
    # print(head(X))
}
}


if(saveToDisk){

run_results_saved <- list(run_results_1_1)#, run_results_1_2, run_results_1_3, run_results_1_4, run_results_1_5)

# Define the base file name
base_file_name <- "run_results_saved"

# Define a function to generate a new file name
generate_new_file_name <- function(base_name) {
    i <- 1
    new_file_name <- paste0(base_name, "_", i, ".rds")
    while (file.exists(new_file_name)) {
        i <- i + 1
        new_file_name <- paste0(base_name, "_", i, ".rds")
    }
    return(new_file_name)
}

# Check if the file already exists
if (file.exists(paste0(base_file_name, ".rds"))) {
    # Generate a new file name
    file_name <- generate_new_file_name(base_file_name)
} else {
    # Use the base file name
    file_name <- paste0(base_file_name, ".rds")
}

# Save the data to an RDS file
saveRDS(run_results_saved, file_name)

cat("Data saved to file:", file_name, "\n")
}

