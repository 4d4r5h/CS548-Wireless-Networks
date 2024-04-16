# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from FS.fpa import (
    jfs,
)  # Importing the Flower Pollination Algorithm (FPA) from the FS library

# Define the file names
filename = "NF-BOT-IOT"
optimization = "FPA"
train_filename = f"{filename}_train_preprocessed.csv"
test_filename = f"{filename}_test_preprocessed.csv"

# Load training and testing data
train_data = pd.read_csv(train_filename, sep=",", encoding="utf-8")
test_data = pd.read_csv(test_filename, sep=",", encoding="utf-8")

# Separate features and labels from the training and testing data
X_train = train_data.drop(columns=["label"], axis=1)
y_train = train_data["label"]
X_test = test_data.drop(columns=["label"], axis=1)
y_test = test_data["label"]

# Split a small sample (1%) from the training data for feature selection
X_t, _, y_t, _ = train_test_split(X_train, y_train, train_size=0.01, random_state=7)

# Define the feature file name
feature_name = f"{filename}_{optimization}_feature.csv"

# Initialize the feature file and write the header
with open(feature_name, "w") as file:
    file.write(
        "optimization,execution time of optimizer,number of features selected,selected features\n"
    )
    file.write(f"{optimization},")

# Split data into train and validation sets (70% train, 30% validation)
feat = np.asarray(X_t)
label = np.asarray(y_t)
xtrain, xtest, ytrain, ytest = train_test_split(
    feat, label, test_size=0.3, stratify=label
)

# Store training and validation sets in a dictionary
fold = {"xt": xtrain, "yt": ytrain, "xv": xtest, "yv": ytest}

# Define parameters for the feature selection algorithm
k = 5  # k-value in KNN
N = 10  # Number of chromosomes/solutions
T = 100  # Maximum number of generations/iterations
P = 0.8  # Switch probability

# Set options for the algorithm
opts = {"k": k, "fold": fold, "N": N, "T": T, "P": P}

# Perform feature selection using the chosen algorithm
start_time = time.time()  # Start timer
fmdl = jfs(feat, label, opts)  # Perform feature selection
end_time = time.time()  # End timer

# Extract selected features and execution time
sf = fmdl["sf"]  # Selected features
exe_time = end_time - start_time  # Execution time

# Append the results to the feature file
with open(feature_name, "a") as file:
    file.write(f"{exe_time},")
    file.write(f"{len(sf)},")
    file.write('"')
    column_headers = list(X_train.columns.values)
    for i in sf:
        file.write(f"{column_headers[i]},")
    file.write('"\n')

# Read the feature file and extract the selected features
feature_df = pd.read_csv(feature_name, sep=",", encoding="utf-8")
selected_feature = feature_df.iat[0, 3]  # Read the selected features
selected_feature = selected_feature[0:-1]  # Removes the last comma

# Clean the selected features string and convert to a list
selected_feature = selected_feature.strip('"')
selected_feature = list(selected_feature.split(","))

# Subset the training and testing data with the selected features
X_train = X_train[selected_feature]
X_test = X_test[selected_feature]

# Print the selected features for confirmation (optional)
print("Selected features:", selected_feature)
