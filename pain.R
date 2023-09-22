library(tidyverse)
library(naniar)
library(rpart)
library(caret)
library(e1071)
library(ggplot2)

Mydata<-read.csv("D:/University/11th semester/data mining/fake_bills.csv",header = TRUE,sep = ";")
Mydata

#Bar chart of 'is_genuine' variable
ggplot(Mydata, aes(x = is_genuine)) +
  geom_bar() +
  labs(x = "Category", y = "Count") +
  ggtitle("Bar Chart of Categories")

#Histograms of numerical variables
hist(Mydata$diagonal)
hist(Mydata$height_left)
hist(Mydata$height_right)
hist(Mydata$margin_low)
hist(Mydata$margin_up)
hist(Mydata$length)

#structure of data frame
str(Mydata)

#Check for missing values
colSums(is.na(Mydata))

discard_instance = Mydata[complete.cases(Mydata$is_genuine,Mydata$diagonal,Mydata$height_left,Mydata$height_right,
                                         Mydata$margin_low,Mydata$margin_up,Mydata$length),]
discard_instance

#Detect outliers in numeric variables only
outliers <- sapply(discard_instance[, sapply(discard_instance, is.numeric)], function(x) {boxplot.stats(x)$out})

#Print the number of outliers detected for each variable
colSums(sapply(outliers, function(x) discard_instance %in% x))

#normalization
normali <- function(x) {(x-min(x))/(max(x)-min(x))}
fake_bills_norm <- as.data.frame(lapply(discard_instance[, c(2,3,4,5,6,7)], normali))
fake_bills_norm

#Add 'is_genuine' attribute to 'fake_bills_norm'
fake_bills_norm$is_genuine <- discard_instance$is_genuine

#Convert logical values to numerical values
#fake_bills_norm$is_genuine <- ifelse(fake_bills_norm$is_genuine, 1, 0)

#Convert logical values to numerical values
fake_bills_norm$is_genuine <- ifelse(fake_bills_norm$is_genuine, 1, 0)

# Define a function to calculate accuracy
calculate_accuracy <- function(predictions, actual) {
  sum(predictions == actual) / length(predictions)
}

# Define a function to predict using the custom decision tree
predict_custom_tree <- function(tree, data) {
  predictions <- vector("character", length = nrow(data))
  
  for (i in 1:nrow(data)) {
    node <- tree
    while (!node$leaf) {
      feature <- node$feature
      split_value <- node$split_value
      if (is.null(split_value)) {
        # Categorical feature
        node <- node[[as.character(data[i, feature])]]
      } else {
        # Numeric feature
        if (data[i, feature] <= split_value) {
          node <- node$left
        } else {
          node <- node$right
        }
      }
    }
    predictions[i] <- node$class
  }
  
  return(predictions)
}

# TDIDT (Top-Down Induction of Decision Trees) Algorithm
tdidt <- function(data, target_variable, features, measure_function, threshold = 0.1) {
  if (all(data[[target_variable]] == data[[target_variable]][1])) {
    # Create a leaf node with the majority class
    return(list(leaf = TRUE, class = data[[target_variable]][1]))
  } else if (length(features) == 0) {
    # Create a leaf node with the majority class
    majority_class <- names(sort(table(data[[target_variable]]), decreasing = TRUE))[1]
    return(list(leaf = TRUE, class = majority_class))
  } else {
    best_feature <- NULL
    best_measure <- -Inf  # Changed from best_gain to best_measure
    for (feature in features) {
      measure = measure_function(data, target_variable, feature)  # Use measure_function here
      if (measure > best_measure) {
        best_feature <- feature
        best_measure <- measure
      }
    }
    
    if (best_measure < threshold) {  # Changed from best_gain to best_measure
      # Create a leaf node with the majority class
      majority_class <- names(sort(table(data[[target_variable]]), decreasing = TRUE))[1]
      return(list(leaf = TRUE, class = majority_class))
    } else {
      # Create a decision node with the best feature
      subtree <- list(feature = best_feature, split_value = NULL)
      unique_values <- unique(data[[best_feature]])
      for (value in unique_values) {
        subset_data <- data[data[[best_feature]] == value, ]
        if (nrow(subset_data) == 0) {
          # Create a leaf node with the majority class
          majority_class <- names(sort(table(data[[target_variable]]), decreasing = TRUE))[1]
          subtree[[as.character(value)]] <- list(leaf = TRUE, class = majority_class)
        } else {
          new_features <- setdiff(features, best_feature)
          subtree[[as.character(value)]] <- tdidt(subset_data, target_variable, new_features, measure_function, threshold)
        }
      }
      return(subtree)
    }
  }
}



# Define your measure functions (gain, gain_ratio, gini_index) here
# Function to calculate entropy
entropy <- function(x) {
  probs <- table(x) / length(x)
  -sum(probs * log2(probs))
}

information_gain <- function(data, target_variable, features) {
  entropy_target <- entropy(data[[target_variable]])
  entropies_features <- sapply(features, function(feature) {
    entropy(data[[feature]])
  })
  target_values <- as.numeric(data[[target_variable]])  # Convert logical to numeric
  gain <- entropy_target - sum(entropies_features * (target_values / nrow(data)))
  return(gain)
}

gini <- function(x) {
  probs <- table(x) / length(x)
  sum(probs * (1 - probs))
}

gain_ratio <- function(data, target_variable, features) {
  information_gain_val = information_gain(data, target_variable, features)
  variance_target = var(data[[target_variable]])
  gain_ratio = information_gain_val / variance_target
  return(gain_ratio)
}

gini_index <- function(data, target_variable, features) {
  gini_target <- gini(data[[target_variable]])
  ginis_features <- sapply(features, function(feature) {
    gini(data[[feature]])
  })
  target_values <- as.numeric(data[[target_variable]])  # Convert logical to numeric
  gini_index <- gini_target - sum(ginis_features * (target_values / nrow(data)))
  return(gini_index)
}


# Example usage of the tdidt function
features <- c("diagonal", "height_left", "height_right", "margin_low", "margin_up", "length")
target_variable <- "is_genuine"
threshold <- 0.1

ig_tree <- tdidt(fake_bills_norm, target_variable, features, information_gain, threshold)
gain_ratio_tree <- tdidt(fake_bills_norm, target_variable, features, gain_ratio, threshold)
gini_index_tree <- tdidt(fake_bills_norm, target_variable, features, gini_index, threshold)

# Print the decision tree
print(ig_tree)
print(gain_ratio_tree)
print(gini_index_tree)

# Define a function to perform k-fold cross-validation and calculate accuracies
perform_kfold_cv <- function(data, target_variable, features, measure_function, threshold, k) {
  set.seed(123)  # For reproducibility
  folds <- createFolds(data[[target_variable]], k = k)
  
  accuracy <- numeric(k)
  
  for (i in 1:k) {
    train_indices <- unlist(folds[-i])
    test_indices <- folds[[i]]
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    # Train a decision tree
    tree <- tdidt(train_data, target_variable, features, measure_function, threshold)
    
    # Make predictions using the custom tree prediction function
    predictions <- predict_custom_tree(tree, test_data)
    
    # Calculate accuracy
    accuracy[i] <- calculate_accuracy(predictions, test_data[[target_variable]])
  }
  
  mean_accuracy <- mean(accuracy)
  return(mean_accuracy)
}

# Perform k-fold cross-validation and calculate mean accuracies
k <- 5
mean_accuracy_ig <- perform_kfold_cv(fake_bills_norm, target_variable, features, information_gain, threshold, k)
mean_accuracy_gain_ratio <- perform_kfold_cv(fake_bills_norm, target_variable, features, gain_ratio, threshold, k)
mean_accuracy_gini_index <- perform_kfold_cv(fake_bills_norm, target_variable, features, gini_index, threshold, k)

# Print mean accuracies
cat("Mean Accuracy (Information Gain):", mean_accuracy_ig, "\n")
cat("Mean Accuracy (Gain Ratio):", mean_accuracy_gain_ratio, "\n")
cat("Mean Accuracy (Gini Index):", mean_accuracy_gini_index, "\n")

# Define a function to create a confusion matrix and calculate metrics
calculate_confusion_matrix <- function(predictions, actual) {
  unique_classes <- unique(c(predictions, actual))
  num_classes <- length(unique_classes)
  
  confusion_matrix <- matrix(0, nrow = num_classes, ncol = num_classes,
                             dimnames = list(unique_classes, unique_classes))
  
  for (i in 1:length(predictions)) {
    actual_class <- actual[i]
    predicted_class <- predictions[i]
    confusion_matrix[actual_class, predicted_class] <- confusion_matrix[actual_class, predicted_class] + 1
  }
  
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  precision <- diag(confusion_matrix) / pmax(colSums(confusion_matrix), 1)  # Handle division by zero
  precision[is.na(precision)] <- 0
  
  recall <- diag(confusion_matrix) / pmax(rowSums(confusion_matrix), 1)  # Handle division by zero
  recall[is.na(recall)] <- 0
  
  f1_score <- 2 * (precision * recall) / pmax((precision + recall), 1)  # Handle division by zero
  f1_score[is.na(f1_score)] <- 0
  
  results <- list(
    ConfusionMatrix = confusion_matrix,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1Score = f1_score
  )
  
  return(results)
}



# Calculate confusion matrices and metrics for each tree type
confusion_ig <- calculate_confusion_matrix(predict_custom_tree(ig_tree, fake_bills_norm), fake_bills_norm$is_genuine)
confusion_gain_ratio <- calculate_confusion_matrix(predict_custom_tree(gain_ratio_tree, fake_bills_norm), fake_bills_norm$is_genuine)
confusion_gini_index <- calculate_confusion_matrix(predict_custom_tree(gini_index_tree, fake_bills_norm), fake_bills_norm$is_genuine)

# Print confusion matrices and metrics
print("Information Gain Tree:")
print(confusion_ig$ConfusionMatrix)
print(paste("Accuracy:", confusion_ig$Accuracy))

print("Gain Ratio Tree:")
print(confusion_gain_ratio$ConfusionMatrix)
print(paste("Accuracy:", confusion_gain_ratio$Accuracy))

print("Gini Index Tree:")
print(confusion_gini_index$ConfusionMatrix)
print(paste("Accuracy:", confusion_gini_index$Accuracy))

